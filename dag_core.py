"""Core data structures and scheduler for hierarchical DAG-based parallel execution.

Architecture:
  - Scheduler maintains a global merged DAG
  - Assigns ready nodes to free slots (displays)
  - Workers either decompose (expand node -> sub-DAG merged back) or execute
  - Sequential sub-nodes share a display via "display chains"
  - Parallel sub-nodes get separate displays
  - Recursive decomposition until CUA agents handle leaf nodes
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from display_pool import DisplayPool
from setup_executor import SetupExecutor

logger = logging.getLogger(__name__)


@dataclass
class DAGNode:
    id: str
    task_description: str
    depends_on: List[str] = field(default_factory=list)
    status: str = "pending"  # pending | running | done | failed
    result: Optional[Dict[str, Any]] = None
    agent_id: Optional[str] = None
    display_num: Optional[int] = None
    parent_node_id: Optional[str] = None
    depth: int = 0
    setup_config: List[Dict[str, Any]] = field(default_factory=list)
    max_steps: int = 30
    start_time: Optional[float] = None
    timeout_seconds: float = 600.0
    retry_count: int = 0
    max_retries: int = 1
    chain_id: Optional[str] = None
    chain_position: int = 0
    chain_successor: Optional[str] = None


@dataclass
class DisplayChain:
    """A sequential chain of nodes that share one display."""
    chain_id: str
    node_ids: List[str]
    display_num: Optional[int] = None
    status: str = "pending"  # pending | active | done | failed


@dataclass
class DAGState:
    nodes: Dict[str, DAGNode] = field(default_factory=dict)
    root_task: str = ""
    max_depth: int = 3
    created_at: float = field(default_factory=time.time)


class DAGScheduler:
    """Scheduler with display-chain awareness.

    Sequential sub-nodes detected from dependency edges share a display.
    Parallel sub-nodes get separate displays. Workers can expand nodes
    into sub-DAGs that are merged back into the global DAG.
    """

    def __init__(
        self,
        dag_state: DAGState,
        display_pool: DisplayPool,
        vm_exec: Callable[[str], Optional[dict]],
        bedrock_factory: Callable[[str, str], Any],
        model: str,
        vm_ip: str,
        server_port: int,
        output_dir: str,
        task_timeout: float = 1200.0,
        password: str = "osworld-public-evaluation",
    ):
        self.dag = dag_state
        self.display_pool = display_pool
        self.vm_exec = vm_exec
        self.bedrock_factory = bedrock_factory
        self.model = model
        self.vm_ip = vm_ip
        self.server_port = server_port
        self.output_dir = output_dir
        self.task_timeout = task_timeout
        self.password = password

        self._lock = threading.RLock()
        self._worker_threads: Dict[str, threading.Thread] = {}
        self._bedrock_clients: Dict[str, Any] = {}
        self._chains: Dict[str, DisplayChain] = {}
        self._start_time: Optional[float] = None
        self._event = threading.Event()

    def get_all_bedrock_clients(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._bedrock_clients)

    # ------------------------------------------------------------------
    # Chain detection
    # ------------------------------------------------------------------

    def _detect_chains(self, node_ids: Set[str]):
        """Detect sequential chains among a set of nodes and register them.

        A chain is a maximal path where each node has exactly one internal
        predecessor and one internal successor. Called after expansion.
        """
        successors: Dict[str, List[str]] = {nid: [] for nid in node_ids}
        predecessors: Dict[str, List[str]] = {nid: [] for nid in node_ids}

        for nid in node_ids:
            node = self.dag.nodes.get(nid)
            if not node:
                continue
            for dep in node.depends_on:
                if dep in node_ids:
                    predecessors[nid].append(dep)
                    successors[dep].append(nid)

        visited: Set[str] = set()
        for nid in node_ids:
            if nid in visited:
                continue
            if len(predecessors[nid]) != 0 and len(predecessors[nid]) != 1:
                continue
            if len(predecessors[nid]) == 1:
                pred = predecessors[nid][0]
                if len(successors[pred]) == 1:
                    continue

            chain = [nid]
            visited.add(nid)
            current = nid
            while True:
                succs = successors.get(current, [])
                if len(succs) != 1:
                    break
                next_id = succs[0]
                if len(predecessors.get(next_id, [])) != 1:
                    break
                if next_id in visited:
                    break
                chain.append(next_id)
                visited.add(next_id)
                current = next_id

            if len(chain) >= 2:
                chain_id = f"chain_{chain[0]}"
                dc = DisplayChain(chain_id=chain_id, node_ids=chain)
                self._chains[chain_id] = dc
                for i, cid in enumerate(chain):
                    cn = self.dag.nodes.get(cid)
                    if cn:
                        cn.chain_id = chain_id
                        cn.chain_position = i
                        cn.chain_successor = chain[i + 1] if i < len(chain) - 1 else None
                logger.info("Detected chain %s: %s", chain_id, chain)

    def _clear_chains_for_nodes(self, node_ids: Set[str]):
        """Remove chain membership from nodes and delete empty chains."""
        chains_to_check = set()
        for nid in node_ids:
            node = self.dag.nodes.get(nid)
            if node and node.chain_id:
                chains_to_check.add(node.chain_id)
                node.chain_id = None
                node.chain_position = 0
                node.chain_successor = None
        for cid in chains_to_check:
            chain = self._chains.get(cid)
            if chain:
                chain.node_ids = [n for n in chain.node_ids if n not in node_ids]
                if len(chain.node_ids) < 2:
                    for remaining_id in chain.node_ids:
                        rn = self.dag.nodes.get(remaining_id)
                        if rn:
                            rn.chain_id = None
                            rn.chain_position = 0
                            rn.chain_successor = None
                    del self._chains[cid]

    # ------------------------------------------------------------------
    # DAG queries
    # ------------------------------------------------------------------

    def _find_ready_nodes(self) -> List[DAGNode]:
        ready = []
        with self._lock:
            for node in self.dag.nodes.values():
                if node.status != "pending":
                    continue
                deps_met = all(
                    self.dag.nodes[dep].status == "done"
                    for dep in node.depends_on
                    if dep in self.dag.nodes
                )
                if deps_met:
                    ready.append(node)
        return ready

    def _is_complete(self) -> bool:
        with self._lock:
            return all(
                n.status in ("done", "failed")
                for n in self.dag.nodes.values()
            )

    def _gather_dependency_results(self, node: DAGNode) -> Dict[str, Any]:
        results = {}
        with self._lock:
            for dep_id in node.depends_on:
                dep = self.dag.nodes.get(dep_id)
                if dep and dep.result:
                    results[dep_id] = dep.result
        return results

    # ------------------------------------------------------------------
    # Node assignment
    # ------------------------------------------------------------------

    def _assign_node(self, node: DAGNode) -> bool:
        if node.chain_id and node.chain_position > 0:
            return self._assign_chain_continuation(node)

        display_num = self.display_pool.allocate(agent_id=f"worker_{node.id}")
        if display_num is None:
            return False

        if node.setup_config:
            executor = SetupExecutor(display_num=display_num, vm_exec=self.vm_exec)
            setup_ok = executor.execute_config(node.setup_config)
            if not setup_ok:
                logger.warning("Setup failed for node %s on display :%d, proceeding", node.id, display_num)

        if node.chain_id and node.chain_position == 0:
            chain = self._chains.get(node.chain_id)
            if chain:
                chain.display_num = display_num
                chain.status = "active"

        self._start_worker(node, display_num)
        return True

    def _assign_chain_continuation(self, node: DAGNode) -> bool:
        """Assign a chain-continuation node to the chain's existing display."""
        chain = self._chains.get(node.chain_id)
        if not chain or chain.display_num is None:
            logger.error("Chain %s has no display for node %s", node.chain_id, node.id)
            return False

        self._start_worker(node, chain.display_num)
        logger.info(
            "Chain continuation: %s on display :%d (chain=%s pos=%d)",
            node.id, chain.display_num, node.chain_id, node.chain_position,
        )
        return True

    def _start_worker(self, node: DAGNode, display_num: int):
        with self._lock:
            node.status = "running"
            node.display_num = display_num
            node.agent_id = f"worker_{node.id}"
            node.start_time = time.time()

        worker_output = os.path.join(self.output_dir, node.agent_id)
        os.makedirs(worker_output, exist_ok=True)

        bedrock = self.bedrock_factory(worker_output, node.agent_id)
        with self._lock:
            self._bedrock_clients[node.agent_id] = bedrock

        dep_results = self._gather_dependency_results(node)

        thread = threading.Thread(
            target=self._run_worker_thread,
            args=(node, bedrock, worker_output, dep_results),
            daemon=True,
            name=f"worker-{node.id}",
        )
        with self._lock:
            self._worker_threads[node.id] = thread
        thread.start()

        logger.info(
            "Assigned %s to display :%d (depth=%d, chain=%s)",
            node.id, display_num, node.depth, node.chain_id or "none",
        )

    def _run_worker_thread(self, node, bedrock, worker_output, dep_results):
        try:
            from dag_worker import run_dag_worker
            result = run_dag_worker(
                node=node,
                scheduler=self,
                vm_ip=self.vm_ip,
                server_port=self.server_port,
                bedrock=bedrock,
                model=self.model,
                output_dir=worker_output,
                password=self.password,
                dependency_results=dep_results,
            )
            if result is not None:
                if result.get("status") == "DONE":
                    self._report_completion(node.id, result)
                else:
                    self._report_failure(node.id, result.get("summary", "unknown"))
        except Exception as e:
            logger.error("Worker %s crashed: %s", node.id, e, exc_info=True)
            self._report_failure(node.id, str(e))

    # ------------------------------------------------------------------
    # Completion / failure
    # ------------------------------------------------------------------

    def _report_completion(self, node_id: str, result: Dict[str, Any]):
        with self._lock:
            node = self.dag.nodes.get(node_id)
            if not node or node.status in ("done", "failed"):
                return
            node.status = "done"
            node.result = result
            duration = time.time() - (node.start_time or time.time())
            logger.info("Node %s completed (%.1fs, depth=%d)", node_id, duration, node.depth)

            should_release = True
            if node.chain_id and node.chain_successor:
                should_release = False
                logger.info("Display :%d held for chain successor %s", node.display_num, node.chain_successor)
            elif node.chain_id and not node.chain_successor:
                chain = self._chains.get(node.chain_id)
                if chain:
                    chain.status = "done"

            if should_release and node.display_num is not None and node.display_num > 0:
                self.display_pool.release(node.display_num)

        self._event.set()

    def _report_failure(self, node_id: str, error: str):
        with self._lock:
            node = self.dag.nodes.get(node_id)
            if not node or node.status in ("done", "failed"):
                return
            self._handle_failure_locked(node, error)
            if node.display_num is not None and node.display_num > 0:
                self.display_pool.release(node.display_num)
            if node.chain_id:
                chain = self._chains.get(node.chain_id)
                if chain:
                    chain.status = "failed"
        self._event.set()

    def _handle_failure_locked(self, node: DAGNode, error: str):
        if node.retry_count < node.max_retries and error != "cascade":
            node.retry_count += 1
            node.status = "pending"
            if node.display_num is not None and node.display_num > 0:
                self.display_pool.release(node.display_num)
            node.display_num = None
            node.agent_id = None
            node.start_time = None
            logger.info("Node %s retrying (%d/%d)", node.id, node.retry_count, node.max_retries)
            return

        node.status = "failed"
        node.result = {"error": error}
        logger.error("Node %s failed: %s", node.id, error[:200])

        for other in self.dag.nodes.values():
            if node.id in other.depends_on and other.status == "pending":
                self._handle_failure_locked(other, "cascade")

    def _check_timeouts(self):
        now = time.time()
        with self._lock:
            for node in self.dag.nodes.values():
                if node.status == "running" and node.start_time:
                    if (now - node.start_time) > node.timeout_seconds:
                        logger.warning("Node %s timed out", node.id)
                        self._handle_failure_locked(node, "timeout")

    # ------------------------------------------------------------------
    # Expansion (spec §4)
    # ------------------------------------------------------------------

    def report_expansion(self, node_id: str, sub_dag_plan: List[Dict[str, Any]]):
        """Worker reports a node expansion. Merges sub-DAG, detects chains."""
        with self._lock:
            node = self.dag.nodes.get(node_id)
            if not node:
                return
            parent_display = node.display_num

            old_chain_id = node.chain_id
            if old_chain_id:
                self._clear_chains_for_nodes({node.id})

            logger.info("Expanding %s into %d sub-nodes (depth %d->%d)",
                        node_id, len(sub_dag_plan), node.depth, node.depth + 1)

            new_node_ids = self._expand_node(node, sub_dag_plan)

            self._detect_chains(new_node_ids)

            if parent_display is not None and parent_display > 0:
                inherited = False
                entry_chain_ids = set()
                for nid in new_node_ids:
                    n = self.dag.nodes.get(nid)
                    if n and n.chain_id and n.chain_position == 0:
                        entry_chain_ids.add(n.chain_id)
                    elif n and not n.chain_id:
                        has_internal_dep = any(d in new_node_ids for d in n.depends_on)
                        if not has_internal_dep:
                            entry_chain_ids.add(f"standalone_{nid}")

                if len(entry_chain_ids) == 1:
                    ecid = entry_chain_ids.pop()
                    chain = self._chains.get(ecid)
                    if chain:
                        chain.display_num = parent_display
                        chain.status = "active"
                        inherited = True
                        logger.info("Display :%d inherited by chain %s", parent_display, ecid)

                if not inherited:
                    self.display_pool.release(parent_display)

        self._event.set()

    def _expand_node(self, node: DAGNode, sub_dag_plan: List[Dict[str, Any]]) -> Set[str]:
        """Replace node with sub-DAG. Returns set of new node IDs. Lock must be held."""
        sub_nodes = []
        for step in sub_dag_plan:
            sub_id = f"{node.id}__{step['id']}"
            internal_deps = [f"{node.id}__{d}" for d in step.get("depends_on", [])]
            sub_node = DAGNode(
                id=sub_id,
                task_description=step["task"],
                depends_on=internal_deps,
                status="pending",
                parent_node_id=node.id,
                depth=node.depth + 1,
                setup_config=step.get("setup", []),
                max_steps=step.get("max_steps", node.max_steps),
                timeout_seconds=node.timeout_seconds,
                max_retries=node.max_retries,
            )
            sub_nodes.append(sub_node)

        all_sub_ids = {sn.id for sn in sub_nodes}

        for sn in sub_nodes:
            has_internal_deps = any(d in all_sub_ids for d in sn.depends_on)
            if not has_internal_deps:
                sn.depends_on = list(node.depends_on)

        depended_on_internally = set()
        for sn in sub_nodes:
            depended_on_internally.update(d for d in sn.depends_on if d in all_sub_ids)
        terminal_ids = list(all_sub_ids - depended_on_internally)

        for other in self.dag.nodes.values():
            if node.id in other.depends_on:
                other.depends_on.remove(node.id)
                other.depends_on.extend(terminal_ids)

        for sn in sub_nodes:
            self.dag.nodes[sn.id] = sn
        del self.dag.nodes[node.id]

        logger.info("Expanded %s -> %d nodes (terminals: %s)", node.id, len(sub_nodes), terminal_ids)
        return all_sub_ids

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        self._start_time = time.time()
        logger.info("DAG Scheduler starting: %d nodes, max_depth=%d",
                     len(self.dag.nodes), self.dag.max_depth)

        while not self._is_complete():
            if time.time() - self._start_time > self.task_timeout:
                logger.warning("Task timeout")
                with self._lock:
                    for n in self.dag.nodes.values():
                        if n.status in ("pending", "running"):
                            n.status = "failed"
                            n.result = {"error": "task_timeout"}
                break

            self._check_timeouts()

            ready = self._find_ready_nodes()
            assigned_any = False
            for node in ready:
                if self._assign_node(node):
                    assigned_any = True

            if not assigned_any:
                self._event.wait(timeout=0.5)
                self._event.clear()

        duration = time.time() - self._start_time
        with self._lock:
            all_done = all(n.status == "done" for n in self.dag.nodes.values())
            node_summaries = {}
            for n in self.dag.nodes.values():
                summary = n.result.get("summary", "") if n.result else ""
                node_summaries[n.id] = {
                    "status": n.status,
                    "depth": n.depth,
                    "summary": summary[:500] if summary else "",
                }

        overall_status = "DONE" if all_done else "FAIL"
        logger.info("DAG Scheduler finished: %s (%.1fs, %d nodes)",
                     overall_status, duration, len(self.dag.nodes))

        return {
            "status": overall_status,
            "duration": duration,
            "nodes": node_summaries,
            "summary": self._build_final_summary(node_summaries),
        }

    def _build_final_summary(self, node_summaries: Dict[str, Any]) -> str:
        parts = []
        for nid, info in sorted(node_summaries.items()):
            parts.append(f"[{nid}] {info['status']}: {info.get('summary', '')[:200]}")
        return "\n".join(parts)
