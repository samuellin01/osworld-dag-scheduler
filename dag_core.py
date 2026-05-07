"""Core data structures and orchestrator for agent-level data flow.

Architecture:
  Orchestrator (plan, agent dependencies, displays, task timeout)
  ├── Manager A (watches every step, spawns helpers, scope updates)
  │   └── Worker A (CUA: computer only, task-focused)
  ├── Manager B
  │   └── Worker B
  │
  │  (Manager A spawns helper at runtime)
  ├── Manager H1 (auto-added as dependency to downstream phases)
  │   └── Helper Worker 1

Data flow:
  - Each phase declares depends_on: list of agent IDs it needs data from
  - When all dependency agents complete, the phase unblocks
  - Each agent's result is injected separately and labeled by agent ID
  - When a manager spawns a helper, the helper is automatically added
    as a dependency to downstream phases that depend on the parent agent
  - No named signals. No merging. Each piece of data has one clear source.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from display_pool import DisplayPool
from setup_executor import SetupExecutor

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class StepRecord:
    """One step of a worker's execution, visible to its manager."""
    step_num: int
    timestamp: float
    elapsed: float
    action_summary: str


@dataclass
class Phase:
    """A phase of work within an agent. Phases run sequentially on the same display."""
    id: str
    task: str
    depends_on: List[str] = field(default_factory=list)
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    current_step: int = 0
    latest_response: str = ""
    step_history: List[StepRecord] = field(default_factory=list)


@dataclass
class AgentPlan:
    """An agent's complete work plan: display setup + ordered phases."""
    id: str
    task: str
    phases: List[Phase] = field(default_factory=list)
    setup: List[Dict[str, Any]] = field(default_factory=list)
    display_num: Optional[int] = None
    status: str = "pending"
    start_time: Optional[float] = None
    end_time: Optional[float] = None


@dataclass
class DAGPlan:
    """The full execution plan: agents + dependencies."""
    agents: Dict[str, AgentPlan] = field(default_factory=dict)
    root_task: str = ""
    created_at: float = field(default_factory=time.time)


# ------------------------------------------------------------------
# Manager prompt
# ------------------------------------------------------------------

_MANAGER_PROMPT = """\
You are managing a computer-use worker agent. You watch every step it takes \
and decide when to intervene.

The worker interacts with the screen using these atomic actions only:
  click (left/right/double), type (text), key (keyboard shortcut), scroll, mouse_move
Each step = one screenshot + one action. Plan estimates in these terms.

The worker only has the computer tool. It does NOT know about other agents, \
signals, or the overall task. It just executes its assigned task. The manager \
handles all coordination.

Overall goal: {root_task}

Other agents working on this task (handled by their own managers — NOT your scope):
{sibling_agents}

Your worker's assigned task: {agent_task}
Current phase: {phase_task}
Data the worker received at phase start (from completed dependencies):
{dependency_data_summary}
Free displays available for helpers: {idle_displays}

Helpers you have already spawned (do NOT duplicate):
{helpers_already_spawned}

Your previous assessment:
{previous_assessment}

New steps since last check:
{new_steps}

Worker's latest output:
{latest_response}

Updated totals: {total_steps} steps, {total_elapsed:.0f}s elapsed

Update your assessment and decide on action:

1. Write your updated assessment:
ASSESSMENT
work_completed: <what the worker has done so far>
remaining_actions: <list concrete screen actions still needed, using the \
atomic action vocabulary (click, type, key, scroll). Group by logical chunk. \
Mark delegated work as SKIP. Example:
  A) finish reading Test 2: scroll down (2 scrolls)
  B) read Test 3: double_click file, scroll through (5 actions)
  C) compile and signal answers (1 action)
  total: ~8 actions
  independent chunks: B can run on a separate display>
parallelism_opportunity: <look at your remaining_actions — can any chunk \
run independently on a separate display? If yes, name the chunk and how many \
actions it saves from the critical path. If no, write "none". Example:
  "Chunk B (read Test 3, ~5 actions) is independent. Spawning a helper \
  reduces critical path from ~8 to ~3 actions.">

2. Decide on action — base this on your parallelism_opportunity analysis:

CONTINUE — worker is on track, no parallelism opportunity.

SPAWN_HELPER — your parallelism_opportunity identified an independent chunk \
worth offloading. Do NOT spawn for work assigned to other agents or already \
covered by helpers above.
helper_task: <the independent chunk as a self-contained task>
helper_setup: <"none" or a JSON setup action>
scope_update: <tell the worker what changed and when to declare SUBTASK \
COMPLETE. Example: "Test 3 is now handled separately. Finish Test 2 and \
declare SUBTASK COMPLETE with your answers.">

SCOPE_UPDATE — coordination-only update to the worker's scope. Use ONLY for:
  - "Your task is done. Declare SUBTASK COMPLETE."
  - "Skip X, another agent handles it."
  - "Move on to the next step."
Do NOT use to correct the worker's answers, judge format, or provide \
task-specific content. The worker has the data it needs — trust it.
scope_update: <coordination directive only>"""


MAX_HELPERS_PER_MANAGER = 3


class Manager:
    """Per-agent supervisor that watches every worker step.

    Runs in its own thread alongside the worker. Processes each new step,
    maintains a running assessment of progress, and intervenes when needed
    (spawn helpers, scope updates).
    """

    def __init__(
        self,
        agent: AgentPlan,
        orchestrator: "Orchestrator",
        bedrock: Any,
        model: str,
        sibling_agents: Optional[List[AgentPlan]] = None,
    ):
        self.agent = agent
        self.orchestrator = orchestrator
        self.bedrock = bedrock
        self.model = model
        self._last_step_seen = 0
        self._assessment = "(no assessment yet — worker just started)"
        self._helpers_spawned: List[str] = []
        self._helper_ids: List[str] = []
        self._sibling_info = self._format_siblings(sibling_agents or [])
        self._dependency_data_summary = "(none — worker has no dependency data yet)"

    def run(self):
        """Main loop: watch steps, evaluate, intervene."""
        tag = f"[mgr:{self.agent.id}]"
        logger.info("%s Manager started", tag)

        while self.agent.status == "running":
            phase = self._current_running_phase()
            if not phase:
                time.sleep(1)
                continue

            current_len = len(phase.step_history)
            if current_len <= self._last_step_seen:
                time.sleep(1)
                continue

            new_steps = phase.step_history[self._last_step_seen:current_len]
            self._last_step_seen = current_len

            try:
                self._evaluate(phase, new_steps, tag)
            except Exception as e:
                logger.warning("%s Evaluation failed: %s", tag, e)

        logger.info("%s Manager finished (agent status: %s)", tag, self.agent.status)

    def _current_running_phase(self) -> Optional[Phase]:
        for phase in self.agent.phases:
            if phase.status == "running":
                return phase
        return None

    @staticmethod
    def _format_siblings(siblings: List[AgentPlan]) -> str:
        if not siblings:
            return "(none — this is the only agent)"
        lines = []
        for s in siblings:
            lines.append(f"  - {s.id}: {s.task}")
        return "\n".join(lines)

    def _evaluate(self, phase: Phase, new_steps: List[StepRecord], tag: str):
        total_steps = phase.current_step
        total_elapsed = new_steps[-1].elapsed if new_steps else 0
        idle_displays = self.orchestrator.display_pool.get_idle_count()

        step_lines = []
        for i, rec in enumerate(new_steps):
            if i > 0:
                delta = f", {rec.elapsed - new_steps[i-1].elapsed:.1f}s since prev"
            else:
                delta = ""
            step_lines.append(f"  step {rec.step_num} (+{rec.elapsed:.0f}s{delta}): {rec.action_summary}")
        new_steps_text = "\n".join(step_lines) if step_lines else "(none)"

        if self._helpers_spawned:
            helpers_text = "\n".join(f"  - {h}" for h in self._helpers_spawned)
        else:
            helpers_text = "(none)"

        effective_idle = idle_displays if len(self._helpers_spawned) < MAX_HELPERS_PER_MANAGER else 0

        prompt = _MANAGER_PROMPT.format(
            root_task=self.orchestrator.plan.root_task[:500],
            sibling_agents=self._sibling_info,
            agent_task=self.agent.task[:300],
            phase_task=phase.task[:300],
            dependency_data_summary=self._dependency_data_summary,
            idle_displays=effective_idle,
            helpers_already_spawned=helpers_text,
            previous_assessment=self._assessment,
            new_steps=new_steps_text,
            latest_response=phase.latest_response[:2000],
            total_steps=total_steps,
            total_elapsed=total_elapsed,
        )

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        logger.info("%s Evaluating (step %d, %.0fs elapsed, %d idle displays)",
                     tag, total_steps, total_elapsed, idle_displays)

        try:
            content_blocks, _ = self.bedrock.chat(
                messages=messages,
                system="You are a task manager. Be concise and decisive.",
                model=self.model,
                temperature=0.3,
                max_tokens=600,
            )
        except Exception as e:
            logger.warning("%s LLM call failed: %s", tag, e)
            return

        response = "".join(
            b.get("text", "") for b in content_blocks
            if isinstance(b, dict) and b.get("type") == "text"
        )

        self._parse_and_act(response, tag)

    def _parse_and_act(self, response: str, tag: str):
        # Extract assessment
        if "ASSESSMENT" in response:
            assessment_lines = []
            in_assessment = False
            for line in response.split("\n"):
                stripped = line.strip()
                if stripped == "ASSESSMENT":
                    in_assessment = True
                    continue
                if in_assessment:
                    if stripped in ("CONTINUE", "SPAWN_HELPER", "SCOPE_UPDATE"):
                        break
                    assessment_lines.append(stripped)
            if assessment_lines:
                self._assessment = "\n".join(assessment_lines)

            # Log key fields
            work_completed = ""
            remaining_actions = []
            parallelism = ""
            total_est = ""
            in_remaining = False
            for line in assessment_lines:
                if line.startswith("work_completed:"):
                    work_completed = line[len("work_completed:"):].strip()
                    in_remaining = False
                elif line.startswith("remaining_actions:"):
                    in_remaining = True
                elif line.startswith("parallelism_opportunity:"):
                    parallelism = line[len("parallelism_opportunity:"):].strip()
                    in_remaining = False
                elif in_remaining:
                    if line.startswith("total:"):
                        total_est = line[len("total:"):].strip()
                        in_remaining = False
                    elif line.startswith("- ") or line.startswith("A)") or line.startswith("B)") or line.startswith("C)"):
                        remaining_actions.append(line.strip())

            actions_str = " | ".join(remaining_actions[:5]) if remaining_actions else "(none)"
            logger.info("%s Assessment: done=[%s] actions=[%s] est=%s parallel=[%s]",
                         tag, work_completed[:60], actions_str[:120],
                         total_est, parallelism[:80])

        if "SPAWN_HELPER" in response:
            if len(self._helpers_spawned) >= MAX_HELPERS_PER_MANAGER:
                logger.info("%s SPAWN_HELPER requested but at cap (%d helpers)", tag, MAX_HELPERS_PER_MANAGER)
            else:
                helper_task = ""
                helper_setup: List[Dict[str, Any]] = []
                scope_update = ""

                for line in response.split("\n"):
                    line = line.strip()
                    if line.startswith("helper_task:"):
                        helper_task = line[len("helper_task:"):].strip()
                    elif line.startswith("helper_setup:"):
                        setup_str = line[len("helper_setup:"):].strip()
                        if setup_str.lower() != "none":
                            try:
                                parsed = json.loads(setup_str)
                                helper_setup = [parsed] if isinstance(parsed, dict) else parsed
                            except json.JSONDecodeError:
                                pass
                    elif line.startswith("scope_update:"):
                        scope_update = line[len("scope_update:"):].strip()

                if helper_task:
                    logger.info("%s SPAWN_HELPER: %s", tag, helper_task[:100])

                    helper_id = self.orchestrator.spawn_helper(
                        helper_task, helper_setup, parent_agent_id=self.agent.id
                    )
                    if helper_id:
                        self._helper_ids.append(helper_id)
                        self._helpers_spawned.append(f"{helper_id}: {helper_task[:80]}")
                    if scope_update:
                        logger.info("%s Scope update -> worker: %s", tag, scope_update[:100])
                        self.orchestrator.send_message(self.agent.id, scope_update)
                else:
                    logger.info("%s SPAWN_HELPER but no task parsed", tag)

        elif "SCOPE_UPDATE" in response:
            scope_update = ""
            for line in response.split("\n"):
                line = line.strip()
                if line.startswith("scope_update:"):
                    scope_update = line[len("scope_update:"):].strip()
            if scope_update:
                logger.info("%s SCOPE_UPDATE: %s", tag, scope_update[:100])
                self.orchestrator.send_message(self.agent.id, scope_update)

        elif "CONTINUE" in response:
            logger.info("%s CONTINUE", tag)

        else:
            logger.info("%s No clear action in response", tag)


# ------------------------------------------------------------------
# Orchestrator
# ------------------------------------------------------------------

class Orchestrator:
    """Agent-level data flow orchestrator with per-agent managers.

    Coordination model:
      - Each phase declares depends_on: list of agent IDs
      - When all dependency agents complete, the phase unblocks
      - Each agent's result is injected separately, labeled by agent ID
      - When a manager spawns a helper, it's auto-added as a dependency
        to downstream phases that depend on the parent agent
    """

    def __init__(
        self,
        plan: DAGPlan,
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
        self.plan = plan
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
        # Agent completion events: agent_id -> Event (set when agent completes)
        self._agent_done_events: Dict[str, threading.Event] = {
            agent_id: threading.Event() for agent_id in plan.agents
        }
        # Agent results: agent_id -> result dict
        self._agent_results: Dict[str, Dict[str, Any]] = {}

        self._agent_threads: Dict[str, threading.Thread] = {}
        self._manager_threads: Dict[str, threading.Thread] = {}
        self._bedrock_clients: Dict[str, Any] = {}
        self._start_time: Optional[float] = None

        self._agent_messages: Dict[str, List[str]] = {}

        self._helpers: Dict[str, AgentPlan] = {}
        self._helper_count = 0

    def get_all_bedrock_clients(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._bedrock_clients)

    # ------------------------------------------------------------------
    # Agent completion tracking
    # ------------------------------------------------------------------

    def mark_agent_done(self, agent_id: str, result: Dict[str, Any]):
        """Mark an agent as complete and store its result."""
        with self._lock:
            self._agent_results[agent_id] = result
            logger.info("Agent '%s' completed with %d chars of data", agent_id, len(result.get("summary", "")))
        event = self._agent_done_events.get(agent_id)
        if event:
            event.set()

    def mark_agent_failed(self, agent_id: str, error: str):
        """Mark an agent as failed."""
        with self._lock:
            self._agent_results[agent_id] = {"status": "FAIL", "summary": f"Agent failed: {error}"}
        event = self._agent_done_events.get(agent_id)
        if event:
            event.set()

    def wait_for_agent(self, agent_id: str, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Block until an agent completes. Returns its result."""
        event = self._agent_done_events.get(agent_id)
        if not event:
            logger.error("Unknown agent: %s", agent_id)
            return None
        event.wait(timeout=timeout)
        with self._lock:
            return self._agent_results.get(agent_id)

    def is_agent_failed(self, agent_id: str) -> bool:
        with self._lock:
            result = self._agent_results.get(agent_id)
            return result is not None and result.get("status") == "FAIL"

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------

    def send_message(self, agent_id: str, message: str):
        with self._lock:
            self._agent_messages.setdefault(agent_id, []).append(message)
        logger.info("Message -> %s: %s", agent_id, message[:120])

    def get_pending_messages(self, agent_id: str) -> List[str]:
        with self._lock:
            return self._agent_messages.pop(agent_id, [])

    # ------------------------------------------------------------------
    # Helper spawning
    # ------------------------------------------------------------------

    def spawn_helper(self, task: str, setup: Optional[List[Dict[str, Any]]] = None,
                     parent_agent_id: Optional[str] = None) -> Optional[str]:
        """Spawn a helper agent and add it as a dependency to downstream phases."""
        self._helper_count += 1
        helper_id = f"helper_{self._helper_count}"

        helper = AgentPlan(
            id=helper_id,
            task=task,
            phases=[Phase(id="execute", task=task)],
            setup=setup or [],
        )

        display_num = self.display_pool.allocate(agent_id=helper_id)
        if display_num is None:
            logger.warning("No display available for helper %s", helper_id)
            return None

        helper.display_num = display_num

        with self._lock:
            self._helpers[helper_id] = helper
            # Register completion event for the helper
            self._agent_done_events[helper_id] = threading.Event()

            # Auto-add helper as dependency to downstream phases that depend on parent
            if parent_agent_id:
                all_agents = list(self.plan.agents.values()) + list(self._helpers.values())
                for agent in all_agents:
                    for phase in agent.phases:
                        if parent_agent_id in phase.depends_on and helper_id not in phase.depends_on:
                            phase.depends_on.append(helper_id)
                            logger.info("Auto-added %s as dependency to %s/%s",
                                         helper_id, agent.id, phase.id)

        thread = threading.Thread(
            target=self._run_agent_thread,
            args=(helper,),
            daemon=True,
            name=f"agent-{helper_id}",
        )
        with self._lock:
            self._agent_threads[helper_id] = thread
        thread.start()

        logger.info("Spawned helper %s on display :%d — %s", helper_id, display_num, task[:80])
        return helper_id

    # ------------------------------------------------------------------
    # Agent execution (worker + manager)
    # ------------------------------------------------------------------

    def _run_agent_thread(self, agent: AgentPlan):
        tag = f"[{agent.id}]"
        try:
            agent.status = "running"
            agent.start_time = time.time()

            if agent.setup and agent.display_num is not None:
                executor = SetupExecutor(display_num=agent.display_num, vm_exec=self.vm_exec)
                setup_ok = executor.execute_config(agent.setup)
                if not setup_ok:
                    logger.warning("%s Setup failed, proceeding anyway", tag)

            agent_output = os.path.join(self.output_dir, agent.id)
            os.makedirs(agent_output, exist_ok=True)

            worker_bedrock = self.bedrock_factory(agent_output, agent.id)
            with self._lock:
                self._bedrock_clients[agent.id] = worker_bedrock

            # Start manager
            manager_output = os.path.join(agent_output, "_manager")
            os.makedirs(manager_output, exist_ok=True)
            manager_bedrock = self.bedrock_factory(manager_output, f"mgr_{agent.id}")
            with self._lock:
                self._bedrock_clients[f"mgr_{agent.id}"] = manager_bedrock

            siblings = [a for a in self.plan.agents.values() if a.id != agent.id]
            manager = Manager(agent, self, manager_bedrock, self.model, sibling_agents=siblings)
            manager_thread = threading.Thread(
                target=manager.run, daemon=True, name=f"mgr-{agent.id}",
            )
            manager_thread.start()
            with self._lock:
                self._manager_threads[agent.id] = manager_thread

            # Run phases
            for i, phase in enumerate(agent.phases):
                if self._start_time and (time.time() - self._start_time) > self.task_timeout:
                    phase.status = "failed"
                    phase.result = {"error": "task_timeout"}
                    raise TimeoutError("Task timeout")

                logger.info("%s Phase %d/%d: %s", tag, i + 1, len(agent.phases), phase.id)

                # Wait for all dependency agents to complete
                dependency_data: Dict[str, Dict[str, Any]] = {}
                if phase.depends_on:
                    phase.status = "blocked"
                    logger.info("%s Waiting for dependencies: %s", tag, phase.depends_on)

                    for dep_agent_id in phase.depends_on:
                        remaining = None
                        if self._start_time:
                            remaining = max(1.0, self.task_timeout - (time.time() - self._start_time))

                        result = self.wait_for_agent(dep_agent_id, timeout=remaining)

                        if self.is_agent_failed(dep_agent_id):
                            phase.status = "failed"
                            phase.result = {"error": f"Dependency '{dep_agent_id}' failed"}
                            raise RuntimeError(f"Dependency '{dep_agent_id}' failed")
                        if result is None:
                            phase.status = "failed"
                            phase.result = {"error": f"Timeout waiting for '{dep_agent_id}'"}
                            raise TimeoutError(f"Dependency '{dep_agent_id}' timed out")

                        dependency_data[dep_agent_id] = result

                # Update manager with dependency data
                if dependency_data:
                    summaries = []
                    for dep_id, data in dependency_data.items():
                        summary = data.get("summary", str(data)) if isinstance(data, dict) else str(data)
                        summaries.append(f"  [{dep_id}]: {summary}")
                    manager._dependency_data_summary = "\n".join(summaries)

                phase.status = "running"
                phase.start_time = time.time()

                from dag_worker import run_phase
                result = run_phase(
                    agent=agent,
                    phase=phase,
                    phase_index=i,
                    vm_ip=self.vm_ip,
                    server_port=self.server_port,
                    bedrock=worker_bedrock,
                    model=self.model,
                    output_dir=agent_output,
                    password=self.password,
                    signal_data=dependency_data if dependency_data else None,
                    orchestrator=self,
                )

                phase.end_time = time.time()
                phase.result = result

                if result.get("status") in ("DONE",):
                    phase.status = "done"
                else:
                    phase.status = "failed"
                    raise RuntimeError(f"Phase {phase.id} failed: {result.get('summary', 'unknown')[:200]}")

            agent.status = "done"
            agent.end_time = time.time()
            duration = agent.end_time - (agent.start_time or agent.end_time)
            logger.info("%s Completed all %d phases (%.1fs)", tag, len(agent.phases), duration)

            # Mark agent as done — use last phase's result as agent result
            last_result = agent.phases[-1].result if agent.phases else {"status": "DONE", "summary": ""}
            self.mark_agent_done(agent.id, last_result)

        except Exception as e:
            logger.error("%s Failed: %s", tag, e, exc_info=True)
            agent.status = "failed"
            agent.end_time = time.time()
            self.mark_agent_failed(agent.id, str(e))

        finally:
            if agent.display_num is not None:
                self.display_pool.release(agent.display_num)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        self._start_time = time.time()
        num_agents = len(self.plan.agents)
        logger.info("Orchestrator starting: %d agents", num_agents)

        for agent in self.plan.agents.values():
            display_num = self.display_pool.allocate(agent_id=agent.id)
            if display_num is None:
                logger.error("No display available for agent %s", agent.id)
                agent.status = "failed"
                self.mark_agent_failed(agent.id, "no display available")
                continue

            agent.display_num = display_num
            thread = threading.Thread(
                target=self._run_agent_thread,
                args=(agent,),
                daemon=True,
                name=f"agent-{agent.id}",
            )
            self._agent_threads[agent.id] = thread
            thread.start()
            logger.info("Started agent %s on display :%d", agent.id, display_num)

        # Wait for all threads (including dynamically spawned helpers)
        deadline = self._start_time + self.task_timeout
        while time.time() < deadline:
            with self._lock:
                threads = dict(self._agent_threads)
            all_done = True
            for thread in threads.values():
                if thread.is_alive():
                    remaining = max(0.1, deadline - time.time())
                    thread.join(timeout=min(2.0, remaining))
                    if thread.is_alive():
                        all_done = False
            if all_done:
                break

        duration = time.time() - self._start_time
        all_agents = list(self.plan.agents.values()) + list(self._helpers.values())
        for agent in all_agents:
            t = self._agent_threads.get(agent.id)
            if t and t.is_alive():
                logger.warning("Agent %s still running after timeout", agent.id)
                agent.status = "failed"

        all_done = all(a.status == "done" for a in all_agents)
        overall_status = "DONE" if all_done else "FAIL"

        agent_summaries = {}
        for agent in all_agents:
            phase_summaries = []
            for phase in agent.phases:
                summary = ""
                if phase.result:
                    summary = phase.result.get("summary", "")[:300]
                phase_summaries.append({
                    "id": phase.id,
                    "status": phase.status,
                    "summary": summary,
                    "steps_used": phase.current_step,
                })
            agent_summaries[agent.id] = {
                "status": agent.status,
                "phases": phase_summaries,
                "display_num": agent.display_num,
            }

        total_agents = len(all_agents)
        helpers_spawned = len(self._helpers)
        logger.info("Orchestrator finished: %s (%.1fs, %d agents, %d helpers spawned)",
                     overall_status, duration, total_agents, helpers_spawned)

        return {
            "status": overall_status,
            "duration": duration,
            "agents": agent_summaries,
            "helpers_spawned": helpers_spawned,
            "summary": self._build_summary(agent_summaries),
        }

    def _build_summary(self, agent_summaries: Dict[str, Any]) -> str:
        parts = []
        for agent_id, info in sorted(agent_summaries.items()):
            parts.append(f"[{agent_id}] {info['status']}")
            for phase in info.get("phases", []):
                parts.append(f"  {phase['id']}: {phase['status']} ({phase.get('steps_used', 0)} steps) - {phase.get('summary', '')[:150]}")
        return "\n".join(parts)
