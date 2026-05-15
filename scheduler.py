"""Resource-aware DAG scheduler for fine-grained parallel execution.

The scheduler maintains a live DAG of actions and a resource table tracking
write conflicts.  It dispatches non-conflicting actions to parallel displays
and an active orchestrator refines the DAG during execution — splitting
bottleneck agents, narrowing footprints, and creating new nodes.

Key differences from the coarse orchestrator:
  - Actions have explicit resource footprints (not just task descriptions)
  - Conflict detection is path-based (multi-granularity resource tree)
  - Active orchestrator sees screenshots and step history, can split work
  - Resource table tracks locks, enabling safe concurrent writes
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import anthropic

from agent_utils import COMPUTER_USE_TOOL, _resize_screenshot, parse_computer_use_actions
from display_pool import DisplayPool
from resource_model import (
    AccessMode,
    ResourceFootprint,
    ResourceLock,
    ResourcePath,
    ResourceTable,
)
from setup_executor import SetupExecutor
from xvfb_display import XvfbDisplay

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# DAG node
# ------------------------------------------------------------------

class NodeStatus(Enum):
    PLANNED = "planned"       # hypothesized, not yet ready
    READY = "ready"           # all dependencies met, waiting for dispatch
    RUNNING = "running"       # dispatched to an agent
    BLOCKED = "blocked"       # agent is pre-positioned, waiting for data
    DONE = "done"             # completed successfully
    FAILED = "failed"         # failed
    REFINED = "refined"       # replaced by finer-grained children


@dataclass
class DAGNode:
    """A single node in the action DAG."""
    id: str
    task: str
    status: NodeStatus = NodeStatus.PLANNED
    footprint: ResourceFootprint = field(default_factory=ResourceFootprint)
    dependencies: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)

    # Execution state
    agent_id: Optional[str] = None
    display_num: Optional[int] = None
    result: Optional[str] = None
    step_count: int = 0
    latest_response: str = ""
    step_history: List[str] = field(default_factory=list)
    setup: List[Dict[str, Any]] = field(default_factory=list)
    latest_screenshot: Optional[bytes] = field(default=None, repr=False)

    # Data passed from completed dependencies
    context_data: Dict[str, str] = field(default_factory=dict)


# ------------------------------------------------------------------
# Event types for the scheduler loop
# ------------------------------------------------------------------

class EventType(Enum):
    AGENT_COMPLETED = "agent_completed"
    AGENT_MESSAGE = "agent_message"
    PEEK_TIMER = "peek_timer"
    TIMEOUT = "timeout"


@dataclass
class SchedulerEvent:
    type: EventType
    agent_id: Optional[str] = None
    node_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


# ------------------------------------------------------------------
# Execution tracker
# ------------------------------------------------------------------

class ExecutionTracker:
    """Accumulates per-node step history across orchestrator peeks.

    Gives the orchestrator the full picture of what each agent has done,
    not just a sliding window of recent actions.
    """

    def __init__(self) -> None:
        self._history: Dict[str, List[str]] = {}
        self._step_counts: Dict[str, int] = {}

    def update(self, node_id: str, step_count: int, step_history: List[str]) -> None:
        self._history[node_id] = list(step_history)
        self._step_counts[node_id] = step_count

    def get_history(self, node_id: str) -> List[str]:
        return self._history.get(node_id, [])


# ------------------------------------------------------------------
# Planner prompt
# ------------------------------------------------------------------

_PLANNER_PROMPT = """\
You are a task planner for a resource-aware multi-agent computer-use system.

The system has {num_displays} virtual displays. You must decompose the task into
a DAG of subtasks with explicit resource footprints. Two nodes can run in \
parallel if and only if their WRITE footprints don't overlap.

**Resource tree** (the system's conflict model):

vm/
├── fs/<path>                                    -- filesystem; prefix overlap = conflict
└── cloud/gdrive/
    ├── sheet:<doc_id>/sheet:<tab>/cells[<RANGE>] -- cells; overlapping ranges = conflict
    ├── doc:<doc_id>/section:<name>               -- doc sections; same section = conflict
    └── slides:<doc_id>/slide:<sid>/element:<eid> -- slides; same element = conflict
For Google Docs, use semantic section names (e.g., section:introduction, \
section:test2_answers) — not character offsets, since positions shift as \
content is edited.
Two WRITE paths conflict when one is a prefix of the other, or when their \
leaf ranges overlap (e.g., cells[A1:B5] and cells[B3:C10] overlap). \
READ paths never conflict with anything. Nodes that only browse/research \
have empty writes.

**Splitting rule**: To minimize end-to-end latency, split work at the \
finest non-overlapping resource boundary the tree allows. A dependency \
edge from A to B is only needed if B reads data that A produces. If B \
just happens to come after A logically but their resources don't overlap, \
they should be parallel — not chained.

**Example** — "Plan and prepare a 3-course dinner":

  Resource tree for this example:
    kitchen/
    ├── counter/
    │   ├── cutting_board
    │   ├── blender
    │   └── bowl
    ├── oven
    └── stove
    dining/table
    notepad

  Initial plan (partial observability — plan coarsely, refine later):
    plan_menu (WRITE notepad)
    cook_all (deps: [plan_menu], WRITE kitchen/) ← coarse lock, can't split yet
    set_table (WRITE dining/table)               ← no conflict with kitchen → parallel
    Key: set_table doesn't depend on cooking — different resource subtree,
    so it runs in parallel. cook_all uses a coarse lock because the menu
    isn't known yet — the orchestrator will split it later during execution.

**Data flow**: Completed nodes' result summaries are automatically passed \
as context to dependent nodes. No temp files — research nodes report \
findings in their completion summary.

**Constraints**:
- Do NOT create verification-only or read-only nodes.
- Do NOT overwrite existing template content (headers, etc.).
- Each node must be self-contained with all info the agent needs.
- Google Workspace: multiple agents CAN open the same URL concurrently.
- No Apps Script — UI only.
- Don't put display paths in footprints — displays are assigned automatically.

Output a JSON object:
{{
  "nodes": [
    {{
      "id": "short_name",
      "task": "Complete task description with all details the agent needs.",
      "dependencies": ["node_id", ...],
      "setup": [{{"type": "chrome_open_tabs", "parameters": {{"urls_to_open": ["..."]}}}}],
      "resource_footprint": {{
        "reads": [],
        "writes": ["vm/cloud/gdrive/sheet:<ID>/sheet:0/cells[A2:D2]"]
      }}
    }}
  ]
}}

Output ONLY the JSON object, no other text."""



_SCHEDULER_SYSTEM = """\
You are a resource-aware task scheduler. You maintain a DAG of actions and
dispatch non-conflicting actions to parallel displays for concurrent execution.

**Resource tree** (the system's conflict model):

vm/
├── fs/<path>                                    -- filesystem; prefix overlap = conflict
└── cloud/gdrive/
    ├── sheet:<doc_id>/sheet:<tab>/cells[<RANGE>] -- cells; overlapping ranges = conflict
    ├── doc:<doc_id>/section:<name>               -- doc sections; same section = conflict
    └── slides:<doc_id>/slide:<sid>/element:<eid> -- slides; same element = conflict

Two WRITE paths conflict when one is a prefix of the other, or when their \
leaf ranges overlap (e.g., cells[A1:B5] and cells[B3:C10] overlap). \
READ paths never conflict with anything.

cells[A1:A30] conflicts with cells[A11:A20], but WRITE cells[A1:A10] \
and WRITE cells[A21:A30] do NOT conflict with WRITE cells[A11:A20]. \
Coarse write regions can be split into finer non-overlapping sub-ranges.

Be concise and decisive. Focus on maximizing parallelism."""


# ------------------------------------------------------------------
# Worker loop (runs on a thread per agent)
# ------------------------------------------------------------------

_COMPLETION = re.compile(r'\bSUBTASK\s+COMPLETE\b', re.IGNORECASE)
_FAILURE = re.compile(r'\bSUBTASK\s+FAILED\b', re.IGNORECASE)
_MAX_STEPS = 200


def _build_worker_system_prompt(display_num: int, password: str) -> str:
    return (
        "You are a computer-use agent on Ubuntu 22.04 with a desktop environment. "
        f"Password: '{password}'. Home directory: /home/user. "
        "\n\n"
        "Your display has been prepared with the necessary applications.\n\n"
        "Your manager may send you messages during execution (shown as [MANAGER]: ...). "
        "Manager instructions OVERRIDE your original task. If the manager tells you "
        "to stop, change scope, or declares your task complete — obey immediately.\n\n"
        "PARALLEL EXECUTION: Other agents are editing the SAME document/sheet "
        "simultaneously. You will see their changes appear in real-time — this "
        "is normal. ONLY modify cells/regions in your assigned scope. IGNORE "
        "all changes outside your assigned range.\n\n"
        "IMPORTANT: Only output SUBTASK COMPLETE after you have FULLY completed "
        "your task — not when you have merely observed or opened something.\n\n"
        "When done, output SUBTASK COMPLETE followed by a brief summary. "
        "After completing your writes, verify 1-2 cells then declare complete. "
        "Do not exhaustively check every cell.\n"
        "If you cannot complete the task, output SUBTASK FAILED with explanation.\n\n"
        "Google Docs/Sheets/Slides: multiple agents can open the same URL simultaneously.\n"
        "Google Workspace: Do NOT use Apps Script — complete tasks through the UI.\n"
        "Google Sheets: Use the Name Box (top-left) to jump to cells.\n"
        "Google Workspace auth: You are already authenticated. Do NOT click 'Sign in' — "
        "the document is editable without signing in. If you see a Sign in button, ignore it.\n"
        "Existing template content should not be modified, reformatted, or deleted.\n\n"
        "Focus only on your assigned task. Do not do extra work beyond what was asked."
    )


def _action_summary(tool_input: Dict[str, Any]) -> str:
    action_type = tool_input.get("action", "")
    if action_type == "type":
        text = tool_input.get("text", "")
        return f"type: {text[:40]}" if len(text) <= 40 else f"type: {text[:37]}..."
    elif action_type == "key":
        return f"key: {tool_input.get('text', '')}"
    elif action_type in ("left_click", "right_click", "double_click", "middle_click"):
        return f"{action_type} at {tool_input.get('coordinate', [])}"
    elif action_type == "mouse_move":
        return f"move to {tool_input.get('coordinate', [])}"
    elif action_type == "scroll":
        return f"scroll {tool_input.get('scroll_direction', '')} {tool_input.get('scroll_amount', 3)}"
    elif action_type == "screenshot":
        return "screenshot"
    return f"computer: {action_type}"


def _save_screenshot(output_dir: str, step: int, shot: bytes, timestamp: float):
    with open(os.path.join(output_dir, f"step_{step:03d}.png"), "wb") as f:
        f.write(shot)
    with open(os.path.join(output_dir, f"step_{step:03d}_timestamp.txt"), "w") as f:
        f.write(f"{timestamp:.6f}\n")


def _save_action(output_dir: str, step: int, tool_input: Dict[str, Any]):
    text = _action_summary(tool_input)
    with open(os.path.join(output_dir, f"step_{step:03d}_action.txt"), "w") as f:
        f.write(text)
    with open(os.path.join(output_dir, f"step_{step:03d}_action.json"), "w") as f:
        json.dump(tool_input, f)


def _build_screenshot_observation(step: int, shot: bytes, resize: bool = True) -> List[Dict[str, Any]]:
    img_bytes = _resize_screenshot(shot) if resize else shot
    return [
        {"type": "text", "text": f"Step {step}: current desktop state."},
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.b64encode(img_bytes).decode(),
            },
        },
    ]


def _run_worker(
    node: DAGNode,
    vm_ip: str,
    server_port: int,
    bedrock: Any,
    model: str,
    output_dir: str,
    password: str,
    scheduler: Scheduler,
) -> Dict[str, Any]:
    """Run a CUA agent loop for one DAG node. Returns result dict."""
    tag = f"[{node.id}]"
    display_num = node.display_num or 0
    display = XvfbDisplay(vm_ip, server_port, display_num)
    system_prompt = _build_worker_system_prompt(display_num, password)
    tools: List[Any] = [COMPUTER_USE_TOOL]

    if display_num == 0:
        resize_factor = (1920.0 / 1280.0, 1080.0 / 720.0)
        needs_resize = True
    else:
        resize_factor = (1.0, 1.0)
        needs_resize = False

    task_output = os.path.join(output_dir, node.id)
    os.makedirs(task_output, exist_ok=True)

    initial_text = f"Your task:\n{node.task}"

    if node.context_data:
        initial_text += "\n\nContext from completed dependencies:"
        for dep_id, dep_result in node.context_data.items():
            initial_text += f"\n  [{dep_id}]: {dep_result}"

    messages: List[Dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": initial_text}]}
    ]

    pending_tool_results: List[Dict[str, Any]] = []
    final_response_text = ""
    start_time = time.time()
    step = 0

    for step in range(1, _MAX_STEPS + 1):
        if scheduler._start_time and (time.time() - scheduler._start_time) > scheduler.task_timeout:
            logger.warning("%s Task timeout reached", tag)
            break

        if node.status == NodeStatus.FAILED:
            logger.info("%s Cancelled by orchestrator, stopping", tag)
            break

        logger.info("%s Step %d", tag, step)

        shot = display.screenshot()
        step_timestamp = time.time()

        if shot:
            node.latest_screenshot = shot
            _save_screenshot(task_output, step, shot, step_timestamp)
            try:
                obs_content = _build_screenshot_observation(step, shot, resize=needs_resize)
            except Exception as e:
                logger.warning("%s Corrupt screenshot at step %d: %s", tag, step, e)
                obs_content = [
                    {"type": "text", "text": f"Step {step}: screenshot corrupted, retrying next step."}
                ]
        else:
            logger.warning("%s Screenshot failed at step %d", tag, step)
            obs_content: List[Dict[str, Any]] = [
                {"type": "text", "text": f"Step {step}: screenshot unavailable."}
            ]

        for tr in pending_tool_results:
            obs_content.insert(0, tr)
        pending_tool_results.clear()

        pending = scheduler.get_pending_messages(node.id)
        if pending:
            msg_text = "\n".join(f"[MANAGER]: {m}" for m in pending)
            obs_content.append({"type": "text", "text": msg_text})

        messages.append({"role": "user", "content": obs_content})

        try:
            content_blocks, _ = bedrock.chat(
                messages=messages,
                system=system_prompt,
                model=model,
                temperature=0.7,
                tools=tools,
            )
        except anthropic.BadRequestError as e:
            error_msg = str(e)
            if "tool_use" in error_msg and "tool_result" in error_msg:
                logger.error("%s Conversation corruption at step %d", tag, step)
                return {
                    "status": "ERROR",
                    "summary": f"Conversation corruption: {error_msg}",
                    "steps_used": step,
                    "duration": time.time() - start_time,
                }
            raise

        messages.append({"role": "assistant", "content": content_blocks})

        response_text = "".join(
            b.get("text", "") for b in content_blocks
            if isinstance(b, dict) and b.get("type") == "text"
        )
        logger.info("%s Response: %s", tag, response_text[:200])
        final_response_text = response_text

        node.step_count = step
        node.latest_response = response_text

        with open(os.path.join(task_output, f"step_{step:03d}_response.txt"), "w") as f:
            f.write(response_text)

        step_entry = f"[step {step}] {response_text.strip()}"
        step_action = ""

        for block in content_blocks:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue

            tool_name = block.get("name")
            tool_id = block.get("id")
            tool_input = block.get("input", {})

            if tool_name == "computer":
                actions = parse_computer_use_actions([block], resize_factor)
                action_code = next(
                    (a for a in actions if a not in ("DONE", "FAIL", "WAIT")), None
                )
                if action_code:
                    logger.info("%s Action: %s", tag, action_code[:120])
                    display.run_action(action_code)
                    time.sleep(1)
                    _save_action(task_output, step, tool_input)
                    step_action = _action_summary(tool_input)

                pending_tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": "Action executed.",
                })
            else:
                pending_tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": f"Unknown tool: {tool_name}. You only have the 'computer' tool.",
                })

        if step_action:
            step_entry += f" → {step_action}"
        node.step_history.append(step_entry)

        for line in final_response_text.strip().split("\n"):
            line = line.strip()
            if _COMPLETION.search(line):
                logger.info("%s Complete at step %d", tag, step)
                return {
                    "status": "DONE",
                    "summary": final_response_text,
                    "steps_used": step,
                    "duration": time.time() - start_time,
                }
            if _FAILURE.search(line):
                logger.info("%s Failed at step %d", tag, step)
                return {
                    "status": "FAIL",
                    "summary": final_response_text,
                    "steps_used": step,
                    "duration": time.time() - start_time,
                }

    logger.warning("%s Exited loop at step %d (timeout or cap)", tag, step)
    return {
        "status": "FAIL",
        "summary": f"Worker stopped at step {step}. Last: {final_response_text}",
        "steps_used": step,
        "duration": time.time() - start_time,
    }


# ------------------------------------------------------------------
# Scheduler
# ------------------------------------------------------------------

class Scheduler:
    """Resource-aware DAG scheduler.

    Maintains a live DAG with resource footprints, dispatches non-conflicting
    actions to parallel displays, and refines the DAG during execution.
    """

    def __init__(
        self,
        display_pool: DisplayPool,
        vm_exec: Callable[[str], Optional[dict]],
        bedrock_factory: Callable[[str, str], Any],
        model: str,
        vm_ip: str,
        server_port: int,
        output_dir: str,
        task_timeout: float = 1200.0,
        password: str = "osworld-public-evaluation",
        root_task: str = "",
        initial_screenshot: Optional[bytes] = None,
    ):
        self.display_pool = display_pool
        self.vm_exec = vm_exec
        self.bedrock_factory = bedrock_factory
        self.model = model
        self.vm_ip = vm_ip
        self.server_port = server_port
        self.output_dir = output_dir
        self.task_timeout = task_timeout
        self.password = password
        self.root_task = root_task

        self._lock = threading.RLock()
        self._dag: Dict[str, DAGNode] = {}
        self._resource_table = ResourceTable()
        self._threads: Dict[str, threading.Thread] = {}
        self._bedrock_clients: Dict[str, Any] = {}
        self._messages: Dict[str, List[str]] = {}
        self._events: List[SchedulerEvent] = []
        self._event_cond = threading.Condition(self._lock)

        self._start_time: Optional[float] = None
        self._work_start_time: Optional[float] = None
        self._done_time: Optional[float] = None
        self._scheduler_done = False

        self._tracker = ExecutionTracker()

        self._sched_output_dir = os.path.join(output_dir, "_scheduler")
        os.makedirs(self._sched_output_dir, exist_ok=True)
        self._execution_log: List[Dict[str, Any]] = []
        self._decision_count = 0
        self._initial_screenshot = initial_screenshot
        self._orch_action_history: List[str] = []

    # ------------------------------------------------------------------
    # DAG management
    # ------------------------------------------------------------------

    def _add_node(self, node: DAGNode) -> None:
        with self._lock:
            self._dag[node.id] = node

    def _get_ready_nodes(self) -> List[DAGNode]:
        """Return nodes whose dependencies are all done and that aren't blocked."""
        with self._lock:
            ready = []
            for node in self._dag.values():
                if node.status != NodeStatus.PLANNED:
                    continue
                deps_met = all(
                    self._dag[dep_id].status == NodeStatus.DONE
                    for dep_id in node.dependencies
                    if dep_id in self._dag
                )
                if deps_met:
                    node.status = NodeStatus.READY
                    ready.append(node)
            return ready

    def _get_dispatchable_nodes(self) -> List[DAGNode]:
        """Return ready nodes that don't conflict with running actions."""
        ready = self._get_ready_nodes()
        dispatchable = []
        for node in ready:
            if self._resource_table.can_acquire(node.footprint):
                dispatchable.append(node)
            else:
                holder = self._resource_table.conflicts_with_holder(node.footprint)
                logger.info(
                    "[scheduler] %s blocked by resource conflict with %s",
                    node.id, holder,
                )
        return dispatchable

    def _prune_unnecessary_deps(self) -> None:
        """Algorithmically remove dependency edges not justified by resource conflicts.

        For each PLANNED node, check each dependency: if the dependency's
        writes don't overlap with this node's writes, the dependency is
        purely ordering (not resource-based). Remove it unless the dependency
        is a READ node (which provides data the write node needs).
        """
        with self._lock:
            for node in self._dag.values():
                if node.status != NodeStatus.PLANNED:
                    continue
                if not node.footprint.writes():
                    continue

                to_remove = []
                for dep_id in node.dependencies:
                    dep = self._dag.get(dep_id)
                    if not dep:
                        continue
                    # Keep dependencies on READ-only nodes (data flow)
                    if not dep.footprint.writes():
                        continue
                    # Check: do the writes overlap?
                    if not node.footprint.conflicts_with(dep.footprint):
                        to_remove.append(dep_id)

                for dep_id in to_remove:
                    node.dependencies.remove(dep_id)
                    logger.info(
                        "[prune] Removed unnecessary dep %s → %s (writes don't overlap)",
                        dep_id, node.id,
                    )
                    self._log_event("prune_dep", node_id=node.id, removed_dep=dep_id)

    def _collect_dependency_results(self, node: DAGNode) -> Dict[str, str]:
        """Gather result summaries from completed dependencies."""
        results = {}
        for dep_id in node.dependencies:
            dep = self._dag.get(dep_id)
            if dep and dep.status == NodeStatus.DONE and dep.result:
                results[dep_id] = dep.result
        return results

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def _plan_initial_dag(self, bedrock: Any) -> None:
        """Use LLM to create the initial DAG from the root task."""
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": f"Task: {self.root_task}"}
        ]

        if self._initial_screenshot:
            try:
                resized = _resize_screenshot(self._initial_screenshot)
                content.append({"type": "text", "text": "Current state of the primary display (display :0):"})
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(resized).decode(),
                    },
                })
            except Exception as e:
                logger.warning("Failed to include screenshot in plan: %s", e)

        messages = [{"role": "user", "content": content}]

        try:
            content_blocks, _ = bedrock.chat(
                messages=messages,
                system=_PLANNER_PROMPT.format(
                    num_displays=len(self.display_pool.displays),
                ),
                model=self.model,
                temperature=0.3,
                max_tokens=4096,
            )
        except Exception as e:
            logger.error("[scheduler] Planning failed: %s", e)
            self._add_node(DAGNode(
                id="fallback_0",
                task=self.root_task,
            ))
            return

        response_text = "".join(
            b.get("text", "") for b in content_blocks
            if isinstance(b, dict) and b.get("type") == "text"
        )

        plan = _parse_json(response_text)
        if not plan or "nodes" not in plan:
            logger.warning("[scheduler] Invalid plan, using single-node fallback")
            self._add_node(DAGNode(
                id="fallback_0",
                task=self.root_task,
            ))
            return

        for node_data in plan["nodes"]:
            fp = _parse_footprint(node_data.get("resource_footprint", {}))
            node = DAGNode(
                id=node_data.get("id", f"node_{len(self._dag)}"),
                task=node_data.get("task", ""),
                dependencies=node_data.get("dependencies", []),
                setup=node_data.get("setup", []),
                footprint=fp,
            )
            self._add_node(node)
            logger.info(
                "[scheduler] Planned node %s: %s (footprint: %s)",
                node.id, node.task[:80], node.footprint,
            )

        with open(os.path.join(self._sched_output_dir, "initial_dag.json"), "w") as f:
            json.dump(self._serialize_dag(), f, indent=2)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _dispatch_node(self, node: DAGNode) -> bool:
        """Dispatch a ready node to an available display."""
        display_num = self.display_pool.allocate(agent_id=node.id)
        if display_num is None:
            logger.warning("[scheduler] No display available for %s", node.id)
            return False

        acquired = self._resource_table.acquire(node.id, node.footprint)
        if not acquired:
            self.display_pool.release(display_num)
            holder = self._resource_table.conflicts_with_holder(node.footprint)
            logger.info(
                "[scheduler] %s could not acquire resources (conflict with %s)",
                node.id, holder,
            )
            node.status = NodeStatus.READY
            return False

        node.display_num = display_num
        node.status = NodeStatus.RUNNING
        node.context_data = self._collect_dependency_results(node)

        thread = threading.Thread(
            target=self._run_node,
            args=(node,),
            daemon=True,
            name=f"worker-{node.id}",
        )
        self._threads[node.id] = thread
        thread.start()

        self._log_event("dispatch", node_id=node.id, display=display_num,
                       task=node.task[:200], footprint=str(node.footprint))
        logger.info(
            "[scheduler] Dispatched %s to display :%d (footprint: %s)",
            node.id, display_num, node.footprint,
        )
        return True

    def _run_node(self, node: DAGNode) -> None:
        """Run setup + worker for a DAG node. Posts completion event."""
        tag = f"[{node.id}]"
        try:
            if node.setup and node.display_num is not None:
                executor = SetupExecutor(
                    display_num=node.display_num, vm_exec=self.vm_exec
                )
                if not executor.execute_config(node.setup):
                    logger.warning("%s Setup failed, proceeding anyway", tag)
                time.sleep(5)

            agent_output = os.path.join(self.output_dir, node.id)
            os.makedirs(agent_output, exist_ok=True)

            bedrock = self.bedrock_factory(agent_output, node.id)
            with self._lock:
                self._bedrock_clients[node.id] = bedrock

            result = _run_worker(
                node=node,
                vm_ip=self.vm_ip,
                server_port=self.server_port,
                bedrock=bedrock,
                model=self.model,
                output_dir=self.output_dir,
                password=self.password,
                scheduler=self,
            )

            node.result = result.get("summary", "")
            node.status = NodeStatus.DONE if result.get("status") == "DONE" else NodeStatus.FAILED

        except Exception as e:
            logger.error("%s Failed: %s", tag, e, exc_info=True)
            node.status = NodeStatus.FAILED
            node.result = str(e)

        finally:
            self._resource_table.release(node.id)
            if node.display_num is not None:
                self.display_pool.release(node.display_num)

            with self._event_cond:
                self._events.append(SchedulerEvent(
                    type=EventType.AGENT_COMPLETED,
                    node_id=node.id,
                    result={"status": node.status.value, "summary": node.result or ""},
                ))
                self._event_cond.notify_all()

            self._log_event("complete", node_id=node.id, status=node.status.value,
                          steps=node.step_count, summary=(node.result or "")[:300])

    # ------------------------------------------------------------------
    # Event loop
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Main event-driven scheduler loop."""
        self._start_time = time.time()
        self._work_start_time = time.time()

        sched_bedrock = self.bedrock_factory(self._sched_output_dir, "scheduler")
        with self._lock:
            self._bedrock_clients["scheduler"] = sched_bedrock

        logger.info("[scheduler] Planning initial DAG...")
        self._plan_initial_dag(sched_bedrock)

        # Prune unnecessary dependencies before first dispatch
        self._prune_unnecessary_deps()

        # Dispatch all initially ready nodes
        self._dispatch_ready_nodes()

        # Event loop
        peek_interval = 30
        last_peek = time.time()

        while (time.time() - self._start_time) < self.task_timeout:
            event = self._wait_for_event(timeout=10)

            if event is None:
                # Periodic DAG status
                elapsed = time.time() - self._start_time
                logger.info("[dag] %.0fs | %s", elapsed, self._dag_status_line())

                # Periodic orchestration: review DAG + agents, take actions
                if time.time() - last_peek >= peek_interval:
                    last_peek = time.time()
                    self._orchestrate(sched_bedrock)

                # Check if all work is done
                with self._lock:
                    running = [n for n in self._dag.values() if n.status == NodeStatus.RUNNING]
                    pending = [n for n in self._dag.values()
                              if n.status in (NodeStatus.PLANNED, NodeStatus.READY)]

                if not running and not pending:
                    logger.info("[scheduler] All nodes completed")
                    self._scheduler_done = True
                    self._done_time = time.time()
                    break

                continue

            self._decision_count += 1

            if event.type == EventType.AGENT_COMPLETED:
                node = self._dag.get(event.node_id, None)
                if node:
                    logger.info(
                        "[scheduler] %s completed (%s, %d steps)",
                        node.id, node.status.value, node.step_count,
                    )

                # Prune unnecessary dependencies algorithmically
                self._prune_unnecessary_deps()

                # Dispatch newly ready nodes
                self._dispatch_ready_nodes()

                # Check if done
                with self._lock:
                    all_terminal = all(
                        n.status in (NodeStatus.DONE, NodeStatus.FAILED, NodeStatus.REFINED)
                        for n in self._dag.values()
                    )
                if all_terminal:
                    logger.info("[scheduler] All nodes terminal — done")
                    self._scheduler_done = True
                    self._done_time = time.time()
                    self._log_event("done")
                    break

        # Wait for stragglers
        for thread in list(self._threads.values()):
            thread.join(timeout=5.0)

        if self._done_time:
            duration = self._done_time - self._work_start_time
        else:
            duration = time.time() - self._work_start_time

        # Save final DAG state
        with open(os.path.join(self._sched_output_dir, "final_dag.json"), "w") as f:
            json.dump(self._serialize_dag(), f, indent=2)

        status = "DONE" if self._scheduler_done else "FAIL"

        agent_summaries = {}
        for node in self._dag.values():
            if node.status == NodeStatus.REFINED:
                continue
            agent_summaries[node.id] = {
                "status": node.status.value,
                "steps_used": node.step_count,
                "summary": node.result or "",
                "display_num": node.display_num,
            }

        logger.info(
            "[scheduler] Finished: %s (%.1fs, %d nodes)",
            status, duration, len(self._dag),
        )

        return {
            "status": status,
            "duration": duration,
            "agents": agent_summaries,
        }

    def _wait_for_event(self, timeout: float) -> Optional[SchedulerEvent]:
        """Wait for an event or timeout."""
        with self._event_cond:
            if not self._events:
                self._event_cond.wait(timeout=timeout)
            if self._events:
                return self._events.pop(0)
        return None

    def _dispatch_ready_nodes(self) -> int:
        """Dispatch all currently dispatchable nodes. Returns count dispatched."""
        dispatched = 0
        while True:
            nodes = self._get_dispatchable_nodes()
            if not nodes:
                break
            for node in nodes:
                if self._dispatch_node(node):
                    dispatched += 1
                else:
                    break
            else:
                continue
            break
        if dispatched:
            logger.info("[scheduler] Dispatched %d node(s)", dispatched)
        return dispatched


    def _orchestrate(self, bedrock: Any) -> None:
        """Active orchestrator: sees full DAG + all agents, can take actions.

        Replaces the old per-agent monitor. Runs periodically during execution.
        Can: message agents, create new nodes, split work, dispatch.
        """
        with self._lock:
            running = [n for n in self._dag.values() if n.status == NodeStatus.RUNNING]
            planned = [n for n in self._dag.values() if n.status == NodeStatus.PLANNED]
            done = [n for n in self._dag.values() if n.status == NodeStatus.DONE]

        if not running:
            return

        elapsed = time.time() - self._start_time

        # Update execution tracker
        for node in self._dag.values():
            self._tracker.update(node.id, node.step_count, node.step_history)

        # Build DAG summary
        dag_lines = []
        for node in self._dag.values():
            status_icon = {
                NodeStatus.PLANNED: "WAITING", NodeStatus.READY: "READY",
                NodeStatus.RUNNING: "RUNNING", NodeStatus.DONE: "DONE",
                NodeStatus.FAILED: "FAILED", NodeStatus.REFINED: "REFINED",
            }.get(node.status, "?")
            deps = f" deps={node.dependencies}" if node.dependencies else ""
            fp = f" writes={[str(l.path) for l in node.footprint.writes()]}" if node.footprint.writes() else ""
            dag_lines.append(f"  {node.id}: {status_icon} (step {node.step_count}){deps}{fp}")

        # Build agent progress summaries with full accumulated history
        agent_lines = []
        agent_screenshots = []
        for node in running:
            history = self._tracker.get_history(node.id) or ["(no steps yet)"]
            agent_lines.append(
                f"--- {node.id} (step {node.step_count}, display :{node.display_num}) ---\n"
                f"  Task: {node.task[:200]}\n"
                f"  Steps:\n    " + "\n    ".join(history)
            )
            if node.latest_screenshot:
                try:
                    resized = _resize_screenshot(node.latest_screenshot)
                    b64 = base64.standard_b64encode(resized).decode("utf-8")
                    agent_screenshots.append((node.id, b64))
                except Exception:
                    pass

        idle_displays = self.display_pool.get_idle_count()

        # Extract setup from existing nodes so orchestrator can copy it
        existing_setup = None
        for node in self._dag.values():
            if node.setup:
                existing_setup = node.setup
                break
        setup_hint = ""
        if existing_setup:
            setup_hint = (
                f"When creating new nodes, include this setup so they get "
                f"a browser with the right URL:\n"
                f"  \"setup\": {json.dumps(existing_setup)}\n\n"
            )

        # Build orchestrator's own action history
        history_section = ""
        if self._orch_action_history:
            history_section = (
                "Your previous actions:\n"
                + "\n".join(f"  - {a}" for a in self._orch_action_history)
                + "\n\n"
            )

        prompt = (
            f"Original task: {self.root_task}\n\n"
            f"[{elapsed:.0f}s elapsed, {idle_displays} idle displays]\n\n"
            f"{history_section}"
            f"DAG state:\n" + "\n".join(dag_lines) + "\n\n"
            f"Running agents:\n" + "\n".join(agent_lines) + "\n\n"
            "You are the orchestrator. Review agent progress and the DAG.\n\n"
            "When there are idle displays, put them to work:\n"
            "- **Split a bottleneck**: narrow a slow agent's footprint with "
            "update_footprint, create_node for the remaining region, message "
            "the original to stop after its current sub-task. Give the "
            "original the larger share — it's in flow, the new agent needs "
            "setup time. Especially impactful when it's the last agent running.\n"
            "- **Parallel exploration**: create_node for independent research "
            "(e.g., different search queries, different data sources).\n"
            "- **Pre-position**: create_node to open a document and navigate "
            "to the right place before a dependency completes.\n\n"
            "Other actions:\n"
            "- message: nudge a stuck agent or tell it to stop\n"
            "- cancel_node: kill an agent that's done but still running\n"
            "- remove_dependency: unblock a waiting node\n"
            "- wait: only if everything is progressing well\n\n"
            f"{setup_hint}"
            "Output a JSON object:\n"
            '{\n'
            '  "actions": [\n'
            '    {"type": "message", "node_id": "...", "text": "..."},\n'
            '    {"type": "create_node", "id": "...", "task": "...", '
            '"dependencies": [...], "setup": [...], '
            '"resource_footprint": {"reads": [...], "writes": [...]}},\n'
            '    {"type": "remove_dependency", "node_id": "...", "dependency_id": "..."},\n'
            '    {"type": "update_footprint", "node_id": "...", '
            '"resource_footprint": {"reads": [...], "writes": [...]}},\n'
            '    {"type": "cancel_node", "node_id": "..."},\n'
            '    {"type": "wait"}\n'
            '  ]\n'
            '}\n'
            'Use "wait" only if everything is progressing well. Actively '
            'intervene — split bottlenecks, cancel finished agents, create '
            'nodes for idle displays.'
        )

        content = []
        if self._initial_screenshot and not self._orch_action_history:
            try:
                resized = _resize_screenshot(self._initial_screenshot)
                b64_init = base64.standard_b64encode(resized).decode("utf-8")
                content.append({"type": "text", "text": "Initial environment state:"})
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": b64_init},
                })
            except Exception:
                pass
        content.append({"type": "text", "text": prompt})
        for agent_id, b64 in agent_screenshots:
            content.append({"type": "text", "text": f"\nCurrent screen for {agent_id}:"})
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": b64},
            })
        messages = [{"role": "user", "content": content}]

        try:
            content_blocks, _ = bedrock.chat(
                messages=messages,
                system=_SCHEDULER_SYSTEM,
                model=self.model,
                temperature=0.3,
                max_tokens=2048,
            )
        except Exception as e:
            logger.warning("[orchestrator] Decision failed: %s", e)
            return

        response_text = "".join(
            b.get("text", "") for b in content_blocks
            if isinstance(b, dict) and b.get("type") == "text"
        )

        result = _parse_json(response_text)
        if not result:
            return

        actions = result.get("actions", [])
        for action in actions:
            action_type = action.get("type", "wait")

            if action_type == "message":
                node_id = action.get("node_id", "")
                text = action.get("text", "")
                if node_id in self._dag and self._dag[node_id].status == NodeStatus.RUNNING and text:
                    self.send_message(node_id, text)
                    self._log_event("orchestrate_message", node_id=node_id, message=text[:200])
                    logger.info("[orchestrator] -> %s: %s", node_id, text[:120])
                    self._orch_action_history.append(f"[{elapsed:.0f}s] message {node_id}: {text[:100]}")

            elif action_type == "create_node":
                new_id = action.get("id", f"dynamic_{len(self._dag)}")
                task = action.get("task", "")
                deps = action.get("dependencies", [])
                raw_setup = action.get("setup", [])
                # Validate setup: must be a list of dicts with "type" keys
                setup = [s for s in raw_setup if isinstance(s, dict) and "type" in s] if isinstance(raw_setup, list) else []
                fp = _parse_footprint(action.get("resource_footprint", {}))

                if new_id in self._dag:
                    new_id = f"{new_id}_{len(self._dag)}"

                if task:
                    # Collect context from completed dependencies
                    context = {}
                    for dep_id in deps:
                        dep = self._dag.get(dep_id)
                        if dep and dep.status == NodeStatus.DONE and dep.result:
                            context[dep_id] = dep.result

                    new_node = DAGNode(
                        id=new_id, task=task, dependencies=deps,
                        setup=setup, footprint=fp,
                        context_data=context,
                    )
                    self._add_node(new_node)
                    self._log_event("orchestrate_create", node_id=new_id,
                                   task=task[:200], footprint=str(fp))
                    logger.info("[orchestrator] Created node %s: %s", new_id, task[:80])
                    self._orch_action_history.append(f"[{elapsed:.0f}s] create_node {new_id} ({fp})")

            elif action_type == "remove_dependency":
                node_id = action.get("node_id", "")
                dep_id = action.get("dependency_id", "")
                if node_id in self._dag and dep_id:
                    node = self._dag[node_id]
                    if dep_id in node.dependencies:
                        node.dependencies.remove(dep_id)
                        self._log_event("orchestrate_remove_dep",
                                       node_id=node_id, removed_dep=dep_id)
                        logger.info("[orchestrator] Removed dependency %s from %s",
                                   dep_id, node_id)
                        self._orch_action_history.append(f"[{elapsed:.0f}s] remove_dep {dep_id} from {node_id}")

            elif action_type == "update_footprint":
                node_id = action.get("node_id", "")
                if node_id in self._dag:
                    new_fp = _parse_footprint(action.get("resource_footprint", {}))
                    old_fp = self._dag[node_id].footprint
                    self._dag[node_id].footprint = new_fp
                    if self._dag[node_id].status == NodeStatus.RUNNING:
                        self._resource_table.release(node_id)
                        self._resource_table.acquire(node_id, new_fp)
                    self._log_event("orchestrate_update_fp",
                                   node_id=node_id,
                                   old_footprint=str(old_fp),
                                   new_footprint=str(new_fp))
                    logger.info("[orchestrator] Updated footprint of %s: %s",
                               node_id, new_fp)
                    self._orch_action_history.append(f"[{elapsed:.0f}s] update_footprint {node_id}: {old_fp} → {new_fp}")

            elif action_type == "cancel_node":
                node_id = action.get("node_id", "")
                if node_id in self._dag:
                    node = self._dag[node_id]
                    if node.status == NodeStatus.RUNNING:
                        node.status = NodeStatus.FAILED
                        node.result = "Cancelled by orchestrator"
                        self._resource_table.release(node_id)
                        if node.display_num is not None:
                            self.display_pool.release(node.display_num)
                        self._log_event("orchestrate_cancel", node_id=node_id)
                        logger.info("[orchestrator] Cancelled running node %s", node_id)
                        self._orch_action_history.append(f"[{elapsed:.0f}s] cancel_node {node_id}")
                    elif node.status == NodeStatus.PLANNED:
                        node.status = NodeStatus.FAILED
                        node.result = "Cancelled by orchestrator"
                        self._log_event("orchestrate_cancel", node_id=node_id)
                        logger.info("[orchestrator] Cancelled planned node %s", node_id)
                        self._orch_action_history.append(f"[{elapsed:.0f}s] cancel_node {node_id}")

            elif action_type == "wait":
                pass

        # After actions, try to dispatch any newly ready nodes
        self._dispatch_ready_nodes()

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------

    def send_message(self, node_id: str, message: str) -> None:
        with self._lock:
            self._messages.setdefault(node_id, []).append(message)

    def get_pending_messages(self, node_id: str) -> List[str]:
        with self._lock:
            return self._messages.pop(node_id, [])

    # ------------------------------------------------------------------
    # Logging and serialization
    # ------------------------------------------------------------------

    def _log_event(self, event_type: str, **kwargs) -> None:
        elapsed = time.time() - self._start_time if self._start_time else 0
        dag_snapshot = self._dag_status_line()
        entry = {"time": round(elapsed, 1), "event": event_type, "dag": dag_snapshot, **kwargs}
        self._execution_log.append(entry)
        log_path = os.path.join(self._sched_output_dir, "execution_log.json")
        try:
            with open(log_path, "w") as f:
                json.dump(self._execution_log, f, indent=2)
        except Exception:
            pass
        logger.info("[dag] %s | %s", event_type, dag_snapshot)

    def _dag_status_line(self) -> str:
        """One-line DAG status: each node as id:status_icon."""
        icons = {
            NodeStatus.PLANNED: "○",
            NodeStatus.READY: "◎",
            NodeStatus.RUNNING: "▶",
            NodeStatus.BLOCKED: "⏸",
            NodeStatus.DONE: "✓",
            NodeStatus.FAILED: "✗",
            NodeStatus.REFINED: "↓",
        }
        with self._lock:
            parts = []
            for node in self._dag.values():
                icon = icons.get(node.status, "?")
                step_info = f"(s{node.step_count})" if node.status == NodeStatus.RUNNING else ""
                parts.append(f"{node.id}:{icon}{step_info}")
            return "  ".join(parts)

    def _serialize_dag(self) -> Dict[str, Any]:
        nodes = []
        for node in self._dag.values():
            nodes.append({
                "id": node.id,
                "task": node.task,
                "status": node.status.value,
                "dependencies": node.dependencies,
                "children": node.children,
                "footprint": str(node.footprint),
                "display_num": node.display_num,
                "step_count": node.step_count,
                "result": (node.result or "")[:500],
            })
        return {"nodes": nodes}

    def get_all_bedrock_clients(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._bedrock_clients)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _parse_json(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        text = match.group(1)
    else:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            text = text[start:end + 1]
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError as e:
        logger.error("Failed to parse JSON: %s", e)
    return None


def _parse_footprint(fp_data: Dict[str, Any]) -> ResourceFootprint:
    """Parse a resource_footprint dict from LLM output into a ResourceFootprint."""
    locks = set()
    for path_str in fp_data.get("reads", []):
        locks.add(ResourceLock(path=ResourcePath(path_str), mode=AccessMode.READ))
    for path_str in fp_data.get("writes", []):
        locks.add(ResourceLock(path=ResourcePath(path_str), mode=AccessMode.WRITE))
    return ResourceFootprint(locks)
