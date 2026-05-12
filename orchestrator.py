"""Simple orchestrator for multi-agent computer-use tasks.

The orchestrator is the hub. It:
  1. Plans subtasks from a task description
  2. Assigns subtasks to subagents on separate displays
  3. Collects results when subagents finish
  4. Decides what to do next: assign new subtasks, forward data, or finish
  5. Can send instructions to running agents to redirect them

Subagents only act when given a subtask. When they finish, they report
back and wait. The orchestrator decides everything.
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
from typing import Any, Callable, Dict, List, Optional

import anthropic

from agent_utils import COMPUTER_USE_TOOL, _resize_screenshot, parse_computer_use_actions
from display_pool import DisplayPool
from xvfb_display import XvfbDisplay
from setup_executor import SetupExecutor

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class Subtask:
    """A single subtask assigned to one subagent on one display."""
    id: str
    task: str
    setup: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "pending"
    display_num: Optional[int] = None
    result: Optional[Dict[str, Any]] = None
    step_count: int = 0
    latest_response: str = ""


# ------------------------------------------------------------------
# Planning
# ------------------------------------------------------------------

_PLANNER_PROMPT = """\
You are a task decomposition planner for a multi-agent computer-use system.

The system has multiple virtual displays. Each subtask gets its own display and \
agent that works independently. Decompose the task into subtasks that run in parallel.

Output a JSON object:
{{
  "subtasks": [
    {{
      "id": "short_name",
      "task": "What this agent should do. Include ALL info it needs.",
      "setup": [setup actions for the display BEFORE the agent starts]
    }}
  ]
}}

**Setup types** (prepare the display before the agent starts):
- {{"type": "chrome_open_tabs", "parameters": {{"urls_to_open": ["https://..."]}}}}
- {{"type": "launch", "parameters": {{"command": ["app", "arg1", ...]}}}}
- {{"type": "sleep", "parameters": {{"seconds": 3}}}}

**Guidelines**:
- Google Workspace: multiple agents CAN open the same Doc/Sheet/Slides URL \
on different displays and edit collaboratively in real-time.
- Maximize parallelism where work is truly independent.
- Don't over-split: if a single agent can handle everything in ~30 actions, \
return a single subtask.
- Each subtask must be self-contained with all info the agent needs.
- Each agent should include a detailed summary of its results when it completes.

Output ONLY the JSON object, no other text."""


_ORCHESTRATOR_PROMPT = """\
You are the orchestrator for a multi-agent computer-use system. Agents work on \
subtasks across separate virtual displays and report results back to you.

Overall task: {root_task}

Completed agents and their results:
{completed_results}

Currently running agents:
{running_agents}

Available displays (idle, can assign new subtasks): {idle_displays}

Decide what to do. You can take MULTIPLE actions:

1. PEEK <agent_id> — take a screenshot of a running agent's display to see \
what it's doing right now. Use when an agent has been running many steps, \
seems stuck, or you need to verify its progress before deciding.

2. ASSIGN <agent_id>: <task description> — give a new subtask to an idle display.
SETUP: <JSON setup action or "none">

3. MESSAGE <agent_id>: <your instruction> — redirect a running agent, provide \
data from another agent's results, or change its scope.

4. WAIT — agents are still working and no action needed.

5. DONE — the overall task is complete.

You can output multiple actions (e.g. PEEK one agent, MESSAGE another). \
Be decisive."""


def plan_subtasks(
    task_description: str,
    bedrock: Any,
    model: str,
    temperature: float = 0.3,
) -> List[Subtask]:
    """Use LLM to decompose a task into parallel subtasks."""
    messages = [
        {"role": "user", "content": [{"type": "text", "text": f"Task: {task_description}"}]}
    ]

    content_blocks, _ = bedrock.chat(
        messages=messages,
        system=_PLANNER_PROMPT,
        model=model,
        temperature=temperature,
        max_tokens=4096,
    )

    response_text = "".join(
        b.get("text", "") for b in content_blocks
        if isinstance(b, dict) and b.get("type") == "text"
    )

    plan = _parse_json(response_text)
    if not plan or "subtasks" not in plan:
        logger.warning("Planner returned invalid plan, using single-agent fallback")
        plan = {"subtasks": [{"id": "agent_0", "task": task_description, "setup": []}]}

    subtasks = []
    for i, st_data in enumerate(plan["subtasks"]):
        subtask = Subtask(
            id=st_data.get("id", f"agent_{i}"),
            task=st_data.get("task", task_description),
            setup=st_data.get("setup", []),
        )
        subtasks.append(subtask)

    logger.info("Planned %d subtask(s)", len(subtasks))
    for st in subtasks:
        logger.info("  %s: %s", st.id, st.task[:80])

    return subtasks


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
        logger.error("Failed to parse plan JSON: %s", e)
    return None


# ------------------------------------------------------------------
# Worker (CUA agent loop)
# ------------------------------------------------------------------

_COMPLETION = re.compile(r'\bSUBTASK\s+COMPLETE\b', re.IGNORECASE)
_FAILURE = re.compile(r'\bSUBTASK\s+FAILED\b', re.IGNORECASE)
_MAX_STEPS = 200


def _build_system_prompt(display_num: int, password: str) -> str:
    chrome_port = 1337 + display_num
    return (
        "You are a computer-use agent on Ubuntu 22.04 with openbox window manager. "
        f"Password: '{password}'. Home directory: /home/user. "
        "\n\n"
        "Recovery: Press Ctrl+Alt+T to open a terminal.\n\n"
        f"If you launch Chrome from terminal: google-chrome --remote-debugging-port={chrome_port} "
        f"--user-data-dir=/tmp/chrome_display_{display_num} --no-first-run "
        "--no-default-browser-check --disable-default-apps URL\n\n"
        "Your display has been prepared with the necessary applications.\n\n"
        "Your manager may send you messages during execution (shown as [MANAGER]: ...). "
        "Follow their instructions.\n\n"
        "When you complete your task, include a detailed summary of your results "
        "and any key data (values, findings, URLs).\n\n"
        "When done, output SUBTASK COMPLETE followed by a summary.\n"
        "If you cannot complete the task, output SUBTASK FAILED with explanation.\n\n"
        "If setup failed -- empty desktop or wrong app on first screenshot:\n"
        "SUBTASK FAILED: Setup did not work. Display shows [describe what you see].\n\n"
        "Google Docs/Sheets/Slides: multiple agents can open the same URL simultaneously.\n"
        "Google Workspace: Do NOT use Apps Script -- complete tasks through the UI.\n"
        "Google Sheets: Use the Name Box (top-left) to jump to cells.\n\n"
        "Focus only on your assigned task. Do not do extra work beyond what was asked."
    )


def _run_worker(
    subtask: Subtask,
    vm_ip: str,
    server_port: int,
    bedrock: Any,
    model: str,
    output_dir: str,
    password: str,
    orchestrator: Orchestrator,
) -> Dict[str, Any]:
    """Run a CUA agent loop for one subtask. Returns result dict."""
    tag = f"[{subtask.id}]"
    display_num = subtask.display_num or 0
    display = XvfbDisplay(vm_ip, server_port, display_num)
    system_prompt = _build_system_prompt(display_num, password)
    tools: List[Any] = [COMPUTER_USE_TOOL]
    resize_factor = (1920.0 / 1280.0, 1080.0 / 720.0)

    task_output = os.path.join(output_dir, subtask.id)
    os.makedirs(task_output, exist_ok=True)

    messages: List[Dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": f"Your task:\n{subtask.task}"}]}
    ]

    pending_tool_results: List[Dict[str, Any]] = []
    final_response_text = ""
    start_time = time.time()
    step = 0

    for step in range(1, _MAX_STEPS + 1):
        if orchestrator._start_time and (time.time() - orchestrator._start_time) > orchestrator.task_timeout:
            logger.warning("%s Task timeout reached", tag)
            break

        logger.info("%s Step %d", tag, step)

        shot = display.screenshot()
        step_timestamp = time.time()

        if shot:
            _save_screenshot(task_output, step, shot, step_timestamp)
            obs_content = _build_screenshot_observation(step, shot)
        else:
            logger.warning("%s Screenshot failed at step %d", tag, step)
            obs_content: List[Dict[str, Any]] = [
                {"type": "text", "text": f"Step {step}: screenshot unavailable."}
            ]

        for tr in pending_tool_results:
            obs_content.insert(0, tr)
        pending_tool_results.clear()

        pending = orchestrator.get_pending_messages(subtask.id)
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

        subtask.step_count = step
        subtask.latest_response = response_text

        with open(os.path.join(task_output, f"step_{step:03d}_response.txt"), "w") as f:
            f.write(response_text)

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
# Helpers
# ------------------------------------------------------------------

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


def _build_screenshot_observation(step: int, shot: bytes) -> List[Dict[str, Any]]:
    resized = _resize_screenshot(shot)
    return [
        {"type": "text", "text": f"Step {step}: current desktop state."},
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.b64encode(resized).decode(),
            },
        },
    ]


def _save_action(output_dir: str, step: int, tool_input: Dict[str, Any]):
    text = _action_summary(tool_input)
    with open(os.path.join(output_dir, f"step_{step:03d}_action.txt"), "w") as f:
        f.write(text)
    with open(os.path.join(output_dir, f"step_{step:03d}_action.json"), "w") as f:
        json.dump(tool_input, f)


# ------------------------------------------------------------------
# Orchestrator
# ------------------------------------------------------------------

class Orchestrator:
    """Central hub that assigns subtasks, collects results, and coordinates.

    Flow:
      1. Launch initial subtasks on displays (from planner)
      2. Run orchestrator loop: every few seconds, check for completed agents
      3. When an agent completes: collect result, consult LLM for next action
         - ASSIGN: give a new subtask to an idle display
         - MESSAGE: send instructions to a running agent
         - WAIT: do nothing, agents are still working
         - DONE: overall task is complete, stop everything
      4. When all agents idle and orchestrator says DONE, return results
    """

    def __init__(
        self,
        subtasks: List[Subtask],
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
        self._all_subtasks: Dict[str, Subtask] = {}
        self._threads: Dict[str, threading.Thread] = {}
        self._bedrock_clients: Dict[str, Any] = {}
        self._messages: Dict[str, List[str]] = {}
        self._start_time: Optional[float] = None
        self._agent_counter = 0
        self._newly_completed: List[str] = []

        for st in subtasks:
            self._all_subtasks[st.id] = st

    def run(self) -> Dict[str, Any]:
        """Main loop: launch initial subtasks, then monitor and react."""
        self._start_time = time.time()

        orch_bedrock = self.bedrock_factory(
            os.path.join(self.output_dir, "_orchestrator"), "orchestrator"
        )
        with self._lock:
            self._bedrock_clients["orchestrator"] = orch_bedrock

        for subtask in list(self._all_subtasks.values()):
            self._launch_agent(subtask)

        logger.info(
            "Orchestrator started: %d initial subtask(s)", len(self._all_subtasks)
        )

        last_check_steps: Dict[str, int] = {}
        check_interval = 30
        last_periodic_check = time.time()

        while (time.time() - self._start_time) < self.task_timeout:
            time.sleep(5)

            with self._lock:
                running = [
                    st for st in self._all_subtasks.values()
                    if st.status == "running"
                ]
                completed = [
                    st for st in self._all_subtasks.values()
                    if st.status in ("done", "failed")
                ]
                newly = list(self._newly_completed)
                self._newly_completed.clear()

            if not running and not newly:
                logger.info("All agents finished, no new completions to process")
                break

            for st in running:
                logger.info(
                    "[orchestrator] %s: step %d, display :%s",
                    st.id, st.step_count, st.display_num,
                )

            should_decide = False

            if newly:
                for agent_id in newly:
                    st = self._all_subtasks[agent_id]
                    logger.info(
                        "[orchestrator] %s completed (%s, %d steps)",
                        st.id, st.status, st.step_count,
                    )
                should_decide = True

            if running and (time.time() - last_periodic_check) >= check_interval:
                has_progress = any(
                    st.step_count > last_check_steps.get(st.id, 0)
                    for st in running
                )
                if has_progress:
                    should_decide = True
                    for st in running:
                        last_check_steps[st.id] = st.step_count

            if should_decide:
                last_periodic_check = time.time()
                action = self._decide_next(orch_bedrock, running, completed)
                if action == "DONE":
                    logger.info("[orchestrator] LLM says overall task is DONE")
                    break

        # Wait for any still-running threads to finish (short grace period)
        for thread in list(self._threads.values()):
            thread.join(timeout=5.0)

        duration = time.time() - self._start_time

        for st in self._all_subtasks.values():
            t = self._threads.get(st.id)
            if t and t.is_alive():
                logger.warning("%s still running after timeout", st.id)
                st.status = "failed"

        all_done = all(
            st.status == "done" for st in self._all_subtasks.values()
        )
        status = "DONE" if all_done else "FAIL"

        agent_summaries = {}
        for st in self._all_subtasks.values():
            summary = st.result.get("summary", "")[:300] if st.result else ""
            agent_summaries[st.id] = {
                "status": st.status,
                "steps_used": st.step_count,
                "summary": summary,
                "display_num": st.display_num,
            }

        logger.info(
            "Orchestrator finished: %s (%.1fs, %d agents)",
            status, duration, len(self._all_subtasks),
        )

        return {
            "status": status,
            "duration": duration,
            "agents": agent_summaries,
        }

    def _launch_agent(self, subtask: Subtask):
        """Allocate a display and start an agent thread for a subtask."""
        display_num = self.display_pool.allocate(agent_id=subtask.id)
        if display_num is None:
            logger.error("No display available for %s", subtask.id)
            subtask.status = "failed"
            subtask.result = {"status": "FAIL", "summary": "No display available"}
            return

        subtask.display_num = display_num
        thread = threading.Thread(
            target=self._run_agent,
            args=(subtask,),
            daemon=True,
            name=f"agent-{subtask.id}",
        )
        self._threads[subtask.id] = thread
        thread.start()
        logger.info("Launched %s on display :%d", subtask.id, display_num)

    def _run_agent(self, subtask: Subtask):
        """Run setup + worker for one subtask. Notifies orchestrator on completion."""
        tag = f"[{subtask.id}]"
        try:
            subtask.status = "running"

            if subtask.setup and subtask.display_num is not None:
                executor = SetupExecutor(
                    display_num=subtask.display_num, vm_exec=self.vm_exec
                )
                if not executor.execute_config(subtask.setup):
                    logger.warning("%s Setup failed, proceeding anyway", tag)

            agent_output = os.path.join(self.output_dir, subtask.id)
            os.makedirs(agent_output, exist_ok=True)

            bedrock = self.bedrock_factory(agent_output, subtask.id)
            with self._lock:
                self._bedrock_clients[subtask.id] = bedrock

            result = _run_worker(
                subtask=subtask,
                vm_ip=self.vm_ip,
                server_port=self.server_port,
                bedrock=bedrock,
                model=self.model,
                output_dir=self.output_dir,
                password=self.password,
                orchestrator=self,
            )

            subtask.result = result
            subtask.status = "done" if result.get("status") == "DONE" else "failed"
            logger.info("%s Finished: %s (%d steps)", tag, subtask.status, subtask.step_count)

        except Exception as e:
            logger.error("%s Failed: %s", tag, e, exc_info=True)
            subtask.status = "failed"
            subtask.result = {"status": "FAIL", "summary": str(e)}

        finally:
            if subtask.display_num is not None:
                self.display_pool.release(subtask.display_num)
            with self._lock:
                self._newly_completed.append(subtask.id)

    def _decide_next(
        self,
        bedrock: Any,
        running: List[Subtask],
        completed: List[Subtask],
    ) -> str:
        """Ask LLM what to do after an agent completed. Returns action taken."""
        completed_lines = []
        for st in completed:
            summary = st.result.get("summary", "")[:500] if st.result else "(no result)"
            completed_lines.append(f"  [{st.id}] ({st.status}, {st.step_count} steps): {summary}")

        running_lines = []
        for st in running:
            latest = st.latest_response[:200] if st.latest_response else "(no response yet)"
            running_lines.append(
                f"  [{st.id}] display:{st.display_num} step:{st.step_count} "
                f"task: {st.task[:100]}\n    latest: {latest}"
            )

        idle_displays = self.display_pool.get_idle_count()

        prompt = _ORCHESTRATOR_PROMPT.format(
            root_task=self.root_task[:500],
            completed_results="\n".join(completed_lines) if completed_lines else "(none)",
            running_agents="\n".join(running_lines) if running_lines else "(none idle)",
            idle_displays=idle_displays,
        )

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        try:
            content_blocks, _ = bedrock.chat(
                messages=messages,
                system="You are a task orchestrator. Be concise and decisive.",
                model=self.model,
                temperature=0.3,
                max_tokens=800,
            )
        except Exception as e:
            logger.warning("[orchestrator] LLM decision failed: %s", e)
            return "WAIT"

        response = "".join(
            b.get("text", "") for b in content_blocks
            if isinstance(b, dict) and b.get("type") == "text"
        )
        logger.info("[orchestrator] Decision: %s", response[:300])

        result_action = "WAIT"
        peek_ids: List[str] = []

        for line in response.strip().split("\n"):
            line = line.strip()

            if line.startswith("PEEK "):
                agent_id = line[len("PEEK "):].strip()
                if agent_id in self._all_subtasks and self._all_subtasks[agent_id].status == "running":
                    peek_ids.append(agent_id)

            elif line.startswith("ASSIGN "):
                rest = line[len("ASSIGN "):]
                if ":" in rest:
                    agent_id, task_desc = rest.split(":", 1)
                    agent_id = agent_id.strip()
                    task_desc = task_desc.strip()
                    if task_desc:
                        self._assign_new_subtask(agent_id, task_desc, response)

            elif line.startswith("MESSAGE "):
                rest = line[len("MESSAGE "):]
                if ":" in rest:
                    agent_id, msg = rest.split(":", 1)
                    agent_id = agent_id.strip()
                    msg = msg.strip()
                    if agent_id in self._all_subtasks and self._all_subtasks[agent_id].status == "running":
                        self.send_message(agent_id, msg)
                        logger.info("[orchestrator] -> %s: %s", agent_id, msg[:120])

            elif line.strip() == "DONE":
                result_action = "DONE"

        if peek_ids and result_action != "DONE":
            followup = self._handle_peeks(bedrock, peek_ids, running, completed)
            if followup == "DONE":
                result_action = "DONE"

        return result_action

    def _handle_peeks(
        self,
        bedrock: Any,
        peek_ids: List[str],
        running: List[Subtask],
        completed: List[Subtask],
    ) -> str:
        """Take screenshots of peeked agents and ask LLM for follow-up actions."""
        peek_content: List[Dict[str, Any]] = [
            {"type": "text", "text": (
                f"You peeked at {len(peek_ids)} agent(s). "
                "Here are their current screenshots. "
                "Decide: MESSAGE, ASSIGN, WAIT, or DONE."
            )}
        ]

        for agent_id in peek_ids:
            st = self._all_subtasks[agent_id]
            if st.display_num is None:
                continue

            display = XvfbDisplay(self.vm_ip, self.server_port, st.display_num)
            shot = display.screenshot()

            peek_content.append({
                "type": "text",
                "text": f"\n[{agent_id}] display:{st.display_num} step:{st.step_count} task: {st.task[:150]}",
            })

            if shot:
                resized = _resize_screenshot(shot)
                peek_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(resized).decode(),
                    },
                })
                logger.info("[orchestrator] PEEK %s — screenshot captured", agent_id)
            else:
                peek_content.append({
                    "type": "text", "text": "(screenshot failed)"
                })
                logger.warning("[orchestrator] PEEK %s — screenshot failed", agent_id)

        messages = [{"role": "user", "content": peek_content}]

        try:
            content_blocks, _ = bedrock.chat(
                messages=messages,
                system="You are a task orchestrator. You just peeked at agent screenshots. Be concise and decisive.",
                model=self.model,
                temperature=0.3,
                max_tokens=800,
            )
        except Exception as e:
            logger.warning("[orchestrator] PEEK follow-up LLM failed: %s", e)
            return "WAIT"

        response = "".join(
            b.get("text", "") for b in content_blocks
            if isinstance(b, dict) and b.get("type") == "text"
        )
        logger.info("[orchestrator] PEEK decision: %s", response[:300])

        result_action = "WAIT"
        for line in response.strip().split("\n"):
            line = line.strip()

            if line.startswith("MESSAGE "):
                rest = line[len("MESSAGE "):]
                if ":" in rest:
                    agent_id, msg = rest.split(":", 1)
                    agent_id = agent_id.strip()
                    msg = msg.strip()
                    if agent_id in self._all_subtasks and self._all_subtasks[agent_id].status == "running":
                        self.send_message(agent_id, msg)
                        logger.info("[orchestrator] PEEK -> %s: %s", agent_id, msg[:120])

            elif line.startswith("ASSIGN "):
                rest = line[len("ASSIGN "):]
                if ":" in rest:
                    agent_id, task_desc = rest.split(":", 1)
                    agent_id = agent_id.strip()
                    task_desc = task_desc.strip()
                    if task_desc:
                        self._assign_new_subtask(agent_id, task_desc, response)

            elif line.strip() == "DONE":
                result_action = "DONE"

        return result_action

    def _assign_new_subtask(self, agent_id: str, task_desc: str, full_response: str):
        """Create and launch a new subtask on an idle display."""
        setup: List[Dict[str, Any]] = []
        for line in full_response.split("\n"):
            line = line.strip()
            if line.startswith("SETUP:"):
                setup_str = line[len("SETUP:"):].strip()
                if setup_str.lower() != "none":
                    try:
                        parsed = json.loads(setup_str)
                        setup = [parsed] if isinstance(parsed, dict) else parsed
                    except json.JSONDecodeError:
                        pass
                break

        if agent_id in self._all_subtasks:
            self._agent_counter += 1
            agent_id = f"{agent_id}_{self._agent_counter}"

        subtask = Subtask(id=agent_id, task=task_desc, setup=setup)
        with self._lock:
            self._all_subtasks[agent_id] = subtask

        self._launch_agent(subtask)
        logger.info("[orchestrator] Assigned new subtask: %s — %s", agent_id, task_desc[:80])

    def send_message(self, agent_id: str, message: str):
        """Send a coordination message to a running subagent."""
        with self._lock:
            self._messages.setdefault(agent_id, []).append(message)
        logger.info("Message -> %s: %s", agent_id, message[:120])

    def get_pending_messages(self, agent_id: str) -> List[str]:
        """Retrieve and clear pending messages for a subagent."""
        with self._lock:
            return self._messages.pop(agent_id, [])

    def get_all_bedrock_clients(self) -> Dict[str, Any]:
        """Return all bedrock clients for token usage aggregation."""
        with self._lock:
            return dict(self._bedrock_clients)
