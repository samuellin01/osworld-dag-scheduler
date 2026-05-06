"""Core data structures and orchestrator for signal/await parallel execution.

Architecture:
  - Orchestrator decomposes a task into agents, each on its own display
  - Each agent runs phases sequentially on the same display
  - Cross-agent data flows through named signals (signal/await)
  - All agents start immediately; they only block at await points
  - Monitor thread peeks at agent progress and spawns helpers when needed
  - Orchestrator can message workers to narrow their focus

Example:
  Agent A: [open chrome] → [search] → [extract] → signal("results")
  Agent B: [open doc] → [navigate] → await("results") → [paste]

  Agent B's first two actions run in parallel with Agent A.
  B only blocks at the await point. Both agents start immediately.

  If A is slow, the monitor spawns a helper and messages A:
  "Helper is handling file 3, you focus on files 1-2."
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


@dataclass
class Signal:
    """A named data channel between agents."""
    name: str
    producer: str
    data: Optional[Dict[str, Any]] = None
    is_set: bool = False
    failed: bool = False
    error: Optional[str] = None


@dataclass
class Phase:
    """A phase of work within an agent. Phases run sequentially on the same display."""
    id: str
    task: str
    awaits: List[str] = field(default_factory=list)
    signals: List[str] = field(default_factory=list)
    max_steps: int = 20
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    current_step: int = 0
    latest_response: str = ""


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
    """The full execution plan: agents + signals."""
    agents: Dict[str, AgentPlan] = field(default_factory=dict)
    signals: Dict[str, Signal] = field(default_factory=dict)
    root_task: str = ""
    created_at: float = field(default_factory=time.time)


_MONITOR_PROMPT = """\
You are monitoring a computer-use agent working on a task.

Overall task: {root_task}
Agent: {agent_id} — {agent_task}
Current phase: {phase_task}
Progress: step {current_step} of {max_steps} ({pct_used:.0f}% of budget used)
Elapsed: {elapsed:.0f}s

Agent's latest output:
{latest_response}

There are {idle_displays} free displays available to spawn a helper agent.

Should we spawn a helper agent to take over part of this agent's remaining work?

Only recommend this if:
- There is clearly separable remaining work (e.g., unprocessed files, unwritten sections)
- The agent is unlikely to finish everything within its remaining step budget
- A helper can work independently on a separate display without conflicting

If YES, respond in exactly this format:
REPLAN
helper_task: <specific, self-contained task description for the helper>
helper_setup: <"none" or a JSON setup action>
message_to_agent: <tell the current agent what to skip since the helper handles it>

If NO, respond with just:
CONTINUE"""


class Orchestrator:
    """Signal/await orchestrator with runtime monitoring and replanning.

    All agents start immediately on separate displays. The monitor thread
    periodically peeks at agent progress and can:
      - Spawn helper agents on free displays for separable remaining work
      - Message workers to narrow their focus
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
        monitor_interval: float = 30.0,
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
        self.monitor_interval = monitor_interval

        self._lock = threading.RLock()
        self._signal_events: Dict[str, threading.Event] = {
            name: threading.Event() for name in plan.signals
        }
        self._agent_threads: Dict[str, threading.Thread] = {}
        self._bedrock_clients: Dict[str, Any] = {}
        self._start_time: Optional[float] = None

        # Messaging: orchestrator -> worker
        self._agent_messages: Dict[str, List[str]] = {}

        # Runtime helpers
        self._helpers: Dict[str, AgentPlan] = {}
        self._helper_count = 0
        self._replanned_agents: Set[str] = set()

    def get_all_bedrock_clients(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._bedrock_clients)

    # ------------------------------------------------------------------
    # Signal operations
    # ------------------------------------------------------------------

    def set_signal(self, name: str, data: Dict[str, Any]):
        with self._lock:
            signal = self.plan.signals.get(name)
            if not signal:
                logger.error("Unknown signal: %s", name)
                return
            signal.data = data
            signal.is_set = True
            logger.info("Signal '%s' set by %s", name, signal.producer)
        self._signal_events[name].set()

    def fail_signal(self, name: str, error: str):
        with self._lock:
            signal = self.plan.signals.get(name)
            if not signal:
                return
            signal.failed = True
            signal.error = error
            logger.warning("Signal '%s' failed: %s", name, error)
        self._signal_events[name].set()

    def wait_signal(self, name: str, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        event = self._signal_events.get(name)
        if not event:
            logger.error("Unknown signal: %s", name)
            return None

        event.wait(timeout=timeout)

        with self._lock:
            signal = self.plan.signals.get(name)
            if not signal or signal.failed:
                return None
            return signal.data

    def is_signal_failed(self, name: str) -> bool:
        with self._lock:
            signal = self.plan.signals.get(name)
            return signal.failed if signal else True

    # ------------------------------------------------------------------
    # Messaging: orchestrator -> worker
    # ------------------------------------------------------------------

    def send_message(self, agent_id: str, message: str):
        """Send a message to a running agent. Injected into its next CUA observation."""
        with self._lock:
            self._agent_messages.setdefault(agent_id, []).append(message)
        logger.info("Message -> %s: %s", agent_id, message[:120])

    def get_pending_messages(self, agent_id: str) -> List[str]:
        """Called by the worker to pick up orchestrator messages."""
        with self._lock:
            return self._agent_messages.pop(agent_id, [])

    # ------------------------------------------------------------------
    # Agent execution
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
            bedrock = self.bedrock_factory(agent_output, agent.id)
            with self._lock:
                self._bedrock_clients[agent.id] = bedrock

            for i, phase in enumerate(agent.phases):
                if self._start_time and (time.time() - self._start_time) > self.task_timeout:
                    phase.status = "failed"
                    phase.result = {"error": "task_timeout"}
                    raise TimeoutError("Task timeout")

                logger.info("%s Phase %d/%d: %s", tag, i + 1, len(agent.phases), phase.id)

                # Await signals
                signal_data: Dict[str, Any] = {}
                if phase.awaits:
                    phase.status = "blocked"
                    logger.info("%s Awaiting signals: %s", tag, phase.awaits)

                    for signal_name in phase.awaits:
                        remaining = None
                        if self._start_time:
                            remaining = max(1.0, self.task_timeout - (time.time() - self._start_time))

                        data = self.wait_signal(signal_name, timeout=remaining)

                        if self.is_signal_failed(signal_name):
                            phase.status = "failed"
                            phase.result = {"error": f"Awaited signal '{signal_name}' failed"}
                            raise RuntimeError(f"Signal '{signal_name}' failed")
                        if data is None:
                            phase.status = "failed"
                            phase.result = {"error": f"Timeout waiting for signal '{signal_name}'"}
                            raise TimeoutError(f"Signal '{signal_name}' timed out")

                        signal_data[signal_name] = data

                # Run CUA loop
                phase.status = "running"
                phase.start_time = time.time()

                from dag_worker import run_phase
                result = run_phase(
                    agent=agent,
                    phase=phase,
                    phase_index=i,
                    vm_ip=self.vm_ip,
                    server_port=self.server_port,
                    bedrock=bedrock,
                    model=self.model,
                    output_dir=agent_output,
                    password=self.password,
                    signal_data=signal_data if signal_data else None,
                    orchestrator=self,
                )

                phase.end_time = time.time()
                phase.result = result

                if result.get("status") in ("DONE", "MAX_STEPS"):
                    phase.status = "done"
                    for signal_name in phase.signals:
                        self.set_signal(signal_name, result)
                else:
                    phase.status = "failed"
                    for signal_name in phase.signals:
                        self.fail_signal(signal_name, f"Phase {phase.id} failed")
                    raise RuntimeError(f"Phase {phase.id} failed: {result.get('summary', 'unknown')[:200]}")

            agent.status = "done"
            agent.end_time = time.time()
            duration = agent.end_time - (agent.start_time or agent.end_time)
            logger.info("%s Completed all %d phases (%.1fs)", tag, len(agent.phases), duration)

        except Exception as e:
            logger.error("%s Failed: %s", tag, e, exc_info=True)
            agent.status = "failed"
            agent.end_time = time.time()
            for phase in agent.phases:
                for signal_name in phase.signals:
                    with self._lock:
                        signal = self.plan.signals.get(signal_name)
                        if signal and not signal.is_set:
                            self.fail_signal(signal_name, str(e))

        finally:
            if agent.display_num is not None:
                self.display_pool.release(agent.display_num)

    # ------------------------------------------------------------------
    # Monitor: peek at progress, spawn helpers, message workers
    # ------------------------------------------------------------------

    def _start_monitor(self):
        if self.monitor_interval <= 0:
            return
        monitor_output = os.path.join(self.output_dir, "monitor")
        os.makedirs(monitor_output, exist_ok=True)
        self._monitor_bedrock = self.bedrock_factory(monitor_output, "monitor")
        with self._lock:
            self._bedrock_clients["monitor"] = self._monitor_bedrock

        thread = threading.Thread(target=self._monitor_loop, daemon=True, name="monitor")
        thread.start()
        logger.info("Monitor started (interval=%ds)", self.monitor_interval)

    def _monitor_loop(self):
        while True:
            time.sleep(self.monitor_interval)
            if self._is_all_done():
                break
            if self._start_time and (time.time() - self._start_time) > self.task_timeout:
                break
            try:
                self._check_all_agents()
            except Exception as e:
                logger.warning("Monitor check failed: %s", e)

    def _is_all_done(self) -> bool:
        with self._lock:
            originals = all(a.status in ("done", "failed") for a in self.plan.agents.values())
            helpers = all(a.status in ("done", "failed") for a in self._helpers.values())
            return originals and helpers

    def _check_all_agents(self):
        all_agents = list(self.plan.agents.values()) + list(self._helpers.values())
        for agent in all_agents:
            if agent.status != "running":
                continue
            if agent.id in self._replanned_agents:
                continue

            phase = next((p for p in agent.phases if p.status == "running"), None)
            if not phase:
                continue

            # Only evaluate once the agent has used enough of its budget to judge
            if phase.current_step < max(5, phase.max_steps * 0.4):
                continue

            idle_count = self.display_pool.get_idle_count()
            if idle_count == 0:
                continue

            self._evaluate_and_maybe_replan(agent, phase, idle_count)

    def _evaluate_and_maybe_replan(self, agent: AgentPlan, phase: Phase, idle_displays: int):
        elapsed = time.time() - (phase.start_time or time.time())
        pct_used = (phase.current_step / phase.max_steps) * 100 if phase.max_steps > 0 else 0

        prompt = _MONITOR_PROMPT.format(
            root_task=self.plan.root_task[:500],
            agent_id=agent.id,
            agent_task=agent.task[:300],
            phase_task=phase.task[:300],
            current_step=phase.current_step,
            max_steps=phase.max_steps,
            pct_used=pct_used,
            elapsed=elapsed,
            latest_response=phase.latest_response[:2000],
            idle_displays=idle_displays,
        )

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        logger.info("Monitor evaluating %s/%s (step %d/%d, %.0fs elapsed)",
                     agent.id, phase.id, phase.current_step, phase.max_steps, elapsed)

        try:
            content_blocks, _ = self._monitor_bedrock.chat(
                messages=messages,
                system="You are a task monitoring assistant. Be decisive and brief.",
                model=self.model,
                temperature=0.3,
                max_tokens=500,
            )
        except Exception as e:
            logger.warning("Monitor LLM call failed: %s", e)
            return

        response = "".join(
            b.get("text", "") for b in content_blocks
            if isinstance(b, dict) and b.get("type") == "text"
        )

        if "REPLAN" not in response:
            logger.info("Monitor: %s/%s — CONTINUE", agent.id, phase.id)
            return

        # Parse replan response
        helper_task = ""
        helper_setup: List[Dict[str, Any]] = []
        message_to_agent = ""

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
            elif line.startswith("message_to_agent:"):
                message_to_agent = line[len("message_to_agent:"):].strip()

        if not helper_task:
            logger.warning("Monitor: REPLAN response but no helper_task parsed")
            return

        logger.info("Monitor: REPLAN for %s/%s", agent.id, phase.id)
        logger.info("  Helper task: %s", helper_task[:120])
        logger.info("  Message: %s", message_to_agent[:120])

        self._replanned_agents.add(agent.id)

        self._spawn_helper(helper_task, helper_setup)

        if message_to_agent:
            self.send_message(agent.id, message_to_agent)

    def _spawn_helper(self, task: str, setup: List[Dict[str, Any]]):
        self._helper_count += 1
        helper_id = f"helper_{self._helper_count}"

        helper = AgentPlan(
            id=helper_id,
            task=task,
            phases=[Phase(id="execute", task=task, max_steps=25)],
            setup=setup,
        )

        display_num = self.display_pool.allocate(agent_id=helper_id)
        if display_num is None:
            logger.warning("No display available for helper %s", helper_id)
            return

        helper.display_num = display_num

        with self._lock:
            self._helpers[helper_id] = helper

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

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        self._start_time = time.time()
        num_agents = len(self.plan.agents)
        logger.info("Orchestrator starting: %d agents, %d signals",
                     num_agents, len(self.plan.signals))

        for agent in self.plan.agents.values():
            display_num = self.display_pool.allocate(agent_id=agent.id)
            if display_num is None:
                logger.error("No display available for agent %s", agent.id)
                agent.status = "failed"
                for phase in agent.phases:
                    for sig in phase.signals:
                        self.fail_signal(sig, "no display available")
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

        self._start_monitor()

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

        # Mark timed-out agents
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
                parts.append(f"  {phase['id']}: {phase['status']} - {phase.get('summary', '')[:150]}")
        return "\n".join(parts)
