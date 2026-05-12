"""Simple orchestrator for multi-agent computer-use tasks.

Breaks a task into subtasks, assigns each to a subagent on its own display,
and monitors progress until completion or timeout.

  Orchestrator
  |- SubAgent A (display :2) -- CUA loop for its subtask
  |- SubAgent B (display :3) -- CUA loop for its subtask
  |- Monitor thread -- logs progress periodically
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
    """Assigns subtasks to subagents on separate displays and monitors progress."""

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
    ):
        self.subtasks = {st.id: st for st in subtasks}
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
        self._threads: Dict[str, threading.Thread] = {}
        self._bedrock_clients: Dict[str, Any] = {}
        self._messages: Dict[str, List[str]] = {}
        self._start_time: Optional[float] = None

    def run(self) -> Dict[str, Any]:
        """Start all subagents and wait for completion."""
        self._start_time = time.time()
        logger.info("Orchestrator starting: %d subtask(s)", len(self.subtasks))

        for subtask in self.subtasks.values():
            display_num = self.display_pool.allocate(agent_id=subtask.id)
            if display_num is None:
                logger.error("No display available for %s", subtask.id)
                subtask.status = "failed"
                subtask.result = {"status": "FAIL", "summary": "No display available"}
                continue

            subtask.display_num = display_num
            thread = threading.Thread(
                target=self._run_agent,
                args=(subtask,),
                daemon=True,
                name=f"agent-{subtask.id}",
            )
            self._threads[subtask.id] = thread
            thread.start()
            logger.info("Started %s on display :%d", subtask.id, display_num)

        monitor = threading.Thread(target=self._monitor, daemon=True, name="monitor")
        monitor.start()

        deadline = self._start_time + self.task_timeout
        while time.time() < deadline:
            all_done = True
            for thread in list(self._threads.values()):
                if thread.is_alive():
                    remaining = max(0.1, deadline - time.time())
                    thread.join(timeout=min(2.0, remaining))
                    if thread.is_alive():
                        all_done = False
            if all_done:
                break

        duration = time.time() - self._start_time

        for subtask in self.subtasks.values():
            t = self._threads.get(subtask.id)
            if t and t.is_alive():
                logger.warning("%s still running after timeout", subtask.id)
                subtask.status = "failed"

        all_done = all(st.status == "done" for st in self.subtasks.values())
        status = "DONE" if all_done else "FAIL"

        agent_summaries = {}
        for st in self.subtasks.values():
            summary = st.result.get("summary", "")[:300] if st.result else ""
            agent_summaries[st.id] = {
                "status": st.status,
                "steps_used": st.step_count,
                "summary": summary,
                "display_num": st.display_num,
            }

        logger.info(
            "Orchestrator finished: %s (%.1fs, %d agents)",
            status, duration, len(self.subtasks),
        )

        return {
            "status": status,
            "duration": duration,
            "agents": agent_summaries,
        }

    def _run_agent(self, subtask: Subtask):
        """Run setup + worker for one subtask."""
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

    def _monitor(self):
        """Periodically log progress of all subagents."""
        while True:
            time.sleep(15)
            with self._lock:
                active = [st for st in self.subtasks.values() if st.status == "running"]
            if not active:
                break
            for st in active:
                logger.info(
                    "[monitor] %s: step %d, status=%s", st.id, st.step_count, st.status
                )

    def send_message(self, agent_id: str, message: str):
        """Send a coordination message to a subagent."""
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
