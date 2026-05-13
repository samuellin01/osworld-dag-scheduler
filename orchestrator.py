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
from typing import Any, Callable, Dict, List, Optional, Tuple

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
    step_history: List[str] = field(default_factory=list)


# ------------------------------------------------------------------
# Planning
# ------------------------------------------------------------------

_PLANNER_PROMPT = """\
You are a task planner for a multi-agent computer-use system.

The system has multiple virtual displays. Create a plan of subtasks. The \
orchestrator will decide when to launch each subtask — not everything needs \
to run in parallel. Subtasks that depend on other subtasks' results should \
be listed but the orchestrator will launch them at the right time.

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
- Don't over-split: if a single agent can handle everything in ~30 actions, \
return a single subtask.
- Each subtask must be self-contained with all info the agent needs.
- Each agent should include a detailed summary of its results when it completes.
- Subtask descriptions should only describe what to DO, not provide answers. \
Agents discover information themselves.
- Documents may have existing content — agents should preserve it, not delete it.

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

Use the provided tools to take actions. You can call multiple tools in one response.

IMPORTANT: Let agents work autonomously. Do NOT micromanage or send messages \
unless an agent is clearly stuck or going completely off-track. Agents scrolling, \
reading files, or taking multiple steps is NORMAL — do not interrupt them. \
Only message an agent if it has been stuck doing the same thing for many steps \
with no progress, or is working on the wrong task entirely.

You can restructure work while agents are running. If you see an agent doing \
something that would be better handled separately, you can MESSAGE it to narrow \
its scope and ASSIGN a new agent for the split-off piece.

Existing template content in documents and spreadsheets should not be modified, \
reformatted, or deleted — it will be exact-matched during evaluation."""


_ORCHESTRATOR_TOOLS = [
    {
        "name": "peek_agent",
        "description": "View a running agent's full conversation history to check its progress. Use when an agent has been running many steps, seems stuck, or you need to verify progress.",
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "ID of the running agent to peek at",
                },
            },
            "required": ["agent_id"],
        },
    },
    {
        "name": "assign_subtask",
        "description": "Assign a new subtask to an idle display. A new agent will be launched to execute it. Include ALL details the agent needs — be very specific about what to do, what to type, what to click.",
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "Short name for the new agent (e.g. 'write_answers', 'edit_doc')",
                },
                "task": {
                    "type": "string",
                    "description": "Complete task description with all details the agent needs",
                },
                "setup": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": (
                        "Setup actions to prepare the display before the agent starts. "
                        "Empty array for none. Available actions:\n"
                        '- {"type": "chrome_open_tabs", "parameters": {"urls_to_open": ["https://..."]}}\n'
                        '- {"type": "launch", "parameters": {"command": ["app", "arg1", ...]}}\n'
                        '- {"type": "sleep", "parameters": {"seconds": 3}}'
                    ),
                    "default": [],
                },
            },
            "required": ["agent_id", "task"],
        },
    },
    {
        "name": "message_agent",
        "description": "Send a message to a running agent to redirect it, provide data, or change its scope. The agent receives it as a [MANAGER] message.",
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "ID of the running agent to message",
                },
                "message": {
                    "type": "string",
                    "description": "The instruction or data to send to the agent",
                },
            },
            "required": ["agent_id", "message"],
        },
    },
    {
        "name": "mark_done",
        "description": "Declare the overall task complete. Use only when all necessary work is finished and verified.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
]


def plan_subtasks(
    task_description: str,
    bedrock: Any,
    model: str,
    temperature: float = 0.3,
    screenshot: Optional[bytes] = None,
) -> List[Subtask]:
    """Use LLM to decompose a task into parallel subtasks."""
    content: List[Dict[str, Any]] = [
        {"type": "text", "text": f"Task: {task_description}"}
    ]

    if screenshot:
        try:
            resized = _resize_screenshot(screenshot)
            content.append({"type": "text", "text": "Current state of the primary display (display :0):"})
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.b64encode(resized).decode(),
                },
            })
            content.append({
                "type": "text",
                "text": "This is the primary display (display :0) which already has apps open "
                        "from environment setup. Do NOT include setup steps for subtasks on this "
                        "display — leave the setup array empty. Only include setup steps for "
                        "subtasks that need a fresh display.",
            })
        except Exception as e:
            logger.warning("Failed to include screenshot in plan: %s", e)

    messages = [{"role": "user", "content": content}]

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


def _clean_agent_id(raw: str) -> str:
    """Extract a clean agent ID from LLM output."""
    cleaned = raw.strip().strip("[]\"'")
    cleaned = re.split(r'[\s\[\]]+', cleaned)[0]
    return cleaned


# ------------------------------------------------------------------
# Worker (CUA agent loop)
# ------------------------------------------------------------------

_COMPLETION = re.compile(r'\bSUBTASK\s+COMPLETE\b', re.IGNORECASE)
_FAILURE = re.compile(r'\bSUBTASK\s+FAILED\b', re.IGNORECASE)
_MAX_STEPS = 200


def _build_system_prompt(display_num: int, password: str) -> str:
    chrome_port = 1337 + display_num
    return (
        "You are a computer-use agent on Ubuntu 22.04 with a desktop environment. "
        f"Password: '{password}'. Home directory: /home/user. "
        "\n\n"
        "Recovery: Press Ctrl+Alt+T to open a terminal.\n\n"
        f"If you launch Chrome from terminal: google-chrome --remote-debugging-port={chrome_port} "
        f"--user-data-dir=/tmp/chrome_display_{display_num} --no-first-run "
        "--no-default-browser-check --disable-default-apps URL\n\n"
        "Your display has been prepared with the necessary applications.\n\n"
        "Your manager may send you messages during execution (shown as [MANAGER]: ...). "
        "Follow their instructions.\n\n"
        "IMPORTANT: Only output SUBTASK COMPLETE after you have FULLY completed "
        "your task — not when you have merely observed or opened something. "
        "If your task says to type, edit, or write something, you must actually "
        "perform those actions BEFORE declaring complete. Seeing a document is not "
        "the same as editing it.\n\n"
        "When done, output SUBTASK COMPLETE followed by a detailed summary of your "
        "results and any key data (values, findings, URLs).\n"
        "If you cannot complete the task, output SUBTASK FAILED with explanation.\n\n"
        "If setup failed -- empty desktop or wrong app on first screenshot:\n"
        "SUBTASK FAILED: Setup did not work. Display shows [describe what you see].\n\n"
        "Google Docs/Sheets/Slides: multiple agents can open the same URL simultaneously.\n"
        "Google Workspace: Do NOT use Apps Script -- complete tasks through the UI.\n"
        "Google Sheets: Use the Name Box (top-left) to jump to cells.\n"
        "Existing template content in documents and spreadsheets is meant to help you "
        "and should not be modified, reformatted, or deleted — it will be exact-matched "
        "during evaluation.\n\n"
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
    prior_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run a CUA agent loop for one subtask. Returns result dict."""
    tag = f"[{subtask.id}]"
    display_num = subtask.display_num or 0
    display = XvfbDisplay(vm_ip, server_port, display_num)
    system_prompt = _build_system_prompt(display_num, password)
    tools: List[Any] = [COMPUTER_USE_TOOL]
    # Display :0 is 1920x1080 (needs resize), Xvfb displays are 1280x720 (1:1)
    if display_num == 0:
        resize_factor = (1920.0 / 1280.0, 1080.0 / 720.0)
        needs_resize = True
    else:
        resize_factor = (1.0, 1.0)
        needs_resize = False

    task_output = os.path.join(output_dir, subtask.id)
    os.makedirs(task_output, exist_ok=True)

    initial_text = f"Your task:\n{subtask.task}"

    if prior_context:
        initial_text += (
            f"\n\nDISPLAY CONTEXT: This display was previously used by agent "
            f"'{prior_context['agent_id']}' which worked on: {prior_context['task']}\n"
            f"Its result: {prior_context['result']}\n"
        )
        if prior_context.get("step_history"):
            initial_text += "Its step-by-step history:\n"
            for entry in prior_context["step_history"]:
                initial_text += f"  {entry}\n"
        initial_text += (
            "\nThe display still has the same windows/apps open from that agent's work. "
            "You can pick up where it left off — do NOT re-open or re-navigate to things "
            "that are already on screen."
        )

    messages: List[Dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": initial_text}]}
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
        subtask.step_history.append(step_entry)

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

    Uses structured tool_use for decisions (peek, assign, message, done).
    Accumulates conversation history so it remembers past decisions.
    Reuses displays from completed agents for new subtasks.
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
        self._decision_count = 0
        self._orch_output_dir = os.path.join(output_dir, "_orchestrator")
        os.makedirs(self._orch_output_dir, exist_ok=True)
        self._orch_history: List[Dict[str, Any]] = []
        self._completed_displays: Dict[str, int] = {}
        self._orchestrator_done = False
        self._done_time: Optional[float] = None
        self._execution_log: List[Dict[str, Any]] = []

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

        # Don't launch anything yet — let the orchestrator decide
        # what to launch first based on the plan
        logger.info(
            "Orchestrator started with plan: %d subtask(s)", len(self._all_subtasks)
        )

        self._work_start_time = time.time()

        check_interval = 45
        last_check_time = 0.0

        while (time.time() - self._start_time) < self.task_timeout:
            time.sleep(3)

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

            pending = [
                st for st in self._all_subtasks.values()
                if st.status == "pending"
            ]

            if not running and not newly and not pending:
                logger.info("[orchestrator] All agents finished")
                break

            trigger = None

            if newly:
                for agent_id in newly:
                    st = self._all_subtasks[agent_id]
                    logger.info(
                        "[orchestrator] %s completed (%s, %d steps)",
                        st.id, st.status, st.step_count,
                    )
                trigger = "completion"

            elif not running and pending:
                trigger = "launch"

            elif running and (time.time() - last_check_time) >= check_interval:
                trigger = "periodic"

            if trigger:
                last_check_time = time.time()
                elapsed = time.time() - self._start_time
                self._decision_count += 1
                step_num = self._decision_count

                status_lines = []
                status_lines.append(f"trigger: {trigger}")
                status_lines.append(f"elapsed: {elapsed:.0f}s")
                status_lines.append(f"running:")
                for st in running:
                    status_lines.append(f"  {st.id}: step {st.step_count}, display :{st.display_num}")
                    logger.info(
                        "[orchestrator] check #%d (%s, %.0fs) — %s: step %d, display :%s",
                        step_num, trigger, elapsed, st.id, st.step_count, st.display_num,
                    )
                status_lines.append(f"completed:")
                for st in completed:
                    status_lines.append(f"  {st.id}: {st.status} ({st.step_count} steps)")
                if newly:
                    status_lines.append(f"newly_completed: {newly}")

                with open(os.path.join(self._orch_output_dir, f"step_{step_num:03d}_context.txt"), "w") as f:
                    f.write("\n".join(status_lines))

                action = self._decide_next(orch_bedrock, running, completed)
                logger.info("[orchestrator] decision #%d action: %s", step_num, action)

                # Release any completed displays that weren't reused this cycle
                with self._lock:
                    for old_agent, old_display in list(self._completed_displays.items()):
                        self.display_pool.release(old_display)
                        logger.info("[orchestrator] Released unused display :%d from %s", old_display, old_agent)
                    self._completed_displays.clear()

                if action == "DONE":
                    self._orchestrator_done = True
                    self._done_time = time.time()
                    self._log_event("done")
                    logger.info("[orchestrator] Overall task is DONE")
                    break

        # Wait for any still-running threads to finish (short grace period)
        for thread in list(self._threads.values()):
            thread.join(timeout=5.0)

        # Latency = work time only (excludes setup, measures until DONE)
        if self._done_time:
            duration = self._done_time - self._work_start_time
        else:
            duration = time.time() - self._work_start_time

        for st in self._all_subtasks.values():
            t = self._threads.get(st.id)
            if t and t.is_alive():
                logger.warning("%s still running after timeout", st.id)
                st.status = "failed"

        if self._orchestrator_done:
            status = "DONE"
        else:
            all_done = all(
                st.status == "done" for st in self._all_subtasks.values()
            )
            status = "DONE" if all_done else "FAIL"

        agent_summaries = {}
        for st in self._all_subtasks.values():
            summary = st.result.get("summary", "") if st.result else ""
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

    def _launch_agent(
        self,
        subtask: Subtask,
        reuse_display: Optional[int] = None,
        prior_context: Optional[Dict[str, Any]] = None,
    ):
        """Allocate a display and start an agent thread for a subtask."""
        if reuse_display is not None:
            display_num = reuse_display
        else:
            display_num = self.display_pool.allocate(agent_id=subtask.id)
            if display_num is None:
                logger.error("No display available for %s", subtask.id)
                subtask.status = "failed"
                subtask.result = {"status": "FAIL", "summary": "No display available"}
                return

        subtask.display_num = display_num
        thread = threading.Thread(
            target=self._run_agent,
            args=(subtask, prior_context),
            daemon=True,
            name=f"agent-{subtask.id}",
        )
        self._threads[subtask.id] = thread
        thread.start()
        logger.info("Launched %s on display :%d", subtask.id, display_num)
        self._log_event("launch", agent_id=subtask.id, display=display_num, task=subtask.task[:200])

    def _run_agent(self, subtask: Subtask, prior_context: Optional[Dict[str, Any]] = None):
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
                logger.info("%s Letting setup settle (5s)...", tag)
                time.sleep(5)

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
                prior_context=prior_context,
            )

            subtask.result = result
            subtask.status = "done" if result.get("status") == "DONE" else "failed"
            logger.info("%s Finished: %s (%d steps)", tag, subtask.status, subtask.step_count)
            summary = result.get("summary", "")[:300] if result else ""
            self._log_event("complete", agent_id=subtask.id, status=subtask.status,
                          steps=subtask.step_count, summary=summary)

        except Exception as e:
            logger.error("%s Failed: %s", tag, e, exc_info=True)
            subtask.status = "failed"
            subtask.result = {"status": "FAIL", "summary": str(e)}

        finally:
            if subtask.display_num is not None:
                with self._lock:
                    self._completed_displays[subtask.id] = subtask.display_num
            with self._lock:
                self._newly_completed.append(subtask.id)

    def _decide_next(
        self,
        bedrock: Any,
        running: List[Subtask],
        completed: List[Subtask],
    ) -> str:
        """Ask LLM what to do via tool use. Returns 'DONE' or 'WAIT'."""
        completed_lines = []
        for st in completed:
            summary = st.result.get("summary", "") if st.result else "(no result)"
            completed_lines.append(f"  [{st.id}] ({st.status}, {st.step_count} steps): {summary}")

        running_lines = []
        for st in running:
            latest = st.latest_response if st.latest_response else "(no response yet)"
            running_lines.append(
                f"  [{st.id}] display:{st.display_num} step:{st.step_count} "
                f"task: {st.task}\n    latest: {latest}"
            )

        idle_displays = self.display_pool.get_idle_count()
        elapsed = time.time() - self._start_time

        pending_lines = []
        for st in self._all_subtasks.values():
            if st.status == "pending":
                pending_lines.append(f"  [{st.id}]: {st.task[:150]}")

        prompt = f"[{elapsed:.0f}s elapsed]\n\n" + _ORCHESTRATOR_PROMPT.format(
            root_task=self.root_task,
            completed_results="\n".join(completed_lines) if completed_lines else "(none)",
            running_agents="\n".join(running_lines) if running_lines else "(none)",
            idle_displays=idle_displays,
        )

        if pending_lines:
            prompt += "\n\nPlanned subtasks (not yet launched — use assign_subtask to launch):\n"
            prompt += "\n".join(pending_lines)

        step_num = self._decision_count

        with open(os.path.join(self._orch_output_dir, f"step_{step_num:03d}_prompt.txt"), "w") as f:
            f.write(prompt)

        self._orch_history.append(
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        )

        try:
            content_blocks, _ = bedrock.chat(
                messages=self._orch_history,
                system="You are a task orchestrator. Be concise and decisive.",
                model=self.model,
                temperature=0.3,
                max_tokens=4096,
                tools=_ORCHESTRATOR_TOOLS,
            )
        except Exception as e:
            logger.warning("[orchestrator] LLM decision failed: %s", e)
            self._orch_history.pop()
            return "WAIT"

        self._orch_history.append({"role": "assistant", "content": content_blocks})

        response_text = "".join(
            b.get("text", "") for b in content_blocks
            if isinstance(b, dict) and b.get("type") == "text"
        )
        logger.info("[orchestrator] Decision #%d: %s", step_num, response_text[:300])

        with open(os.path.join(self._orch_output_dir, f"step_{step_num:03d}_response.txt"), "w") as f:
            f.write(response_text)

        # Process tool calls
        result_action = "WAIT"
        peek_ids: List[str] = []
        actions_taken: List[str] = []
        tool_results: List[Dict[str, Any]] = []

        for block in content_blocks:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue

            tool_name = block.get("name", "")
            tool_id = block.get("id", "")
            tool_input = block.get("input", {})
            agent_id = _clean_agent_id(tool_input.get("agent_id", ""))

            if tool_name == "peek_agent":
                if agent_id in self._all_subtasks and self._all_subtasks[agent_id].status == "running":
                    peek_ids.append(agent_id)
                    actions_taken.append(f"PEEK {agent_id}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": "Peek results will follow.",
                    })
                else:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": f"Agent '{agent_id}' not found or not running.",
                    })

            elif tool_name == "assign_subtask":
                task_desc = tool_input.get("task", "")
                setup = tool_input.get("setup", [])
                if task_desc:
                    self._assign_new_subtask_structured(agent_id, task_desc, setup)
                    actions_taken.append(f"ASSIGN {agent_id}: {task_desc[:80]}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": f"Agent '{agent_id}' launched.",
                })

            elif tool_name == "message_agent":
                msg = tool_input.get("message", "")
                if agent_id in self._all_subtasks and self._all_subtasks[agent_id].status == "running":
                    self.send_message(agent_id, msg)
                    actions_taken.append(f"MESSAGE {agent_id}: {msg[:80]}")
                    logger.info("[orchestrator] -> %s: %s", agent_id, msg[:120])
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": f"Message delivered to '{agent_id}'.",
                    })
                else:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": f"Agent '{agent_id}' not found or not running.",
                    })

            elif tool_name == "mark_done":
                result_action = "DONE"
                actions_taken.append("DONE")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": "Task marked as done.",
                })

        # Ensure EVERY tool_use block has a matching tool_result
        handled_ids = {tr["tool_use_id"] for tr in tool_results}
        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                if block.get("id") not in handled_ids:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block["id"],
                        "content": "Acknowledged.",
                    })

        if tool_results:
            self._orch_history.append({"role": "user", "content": tool_results})

        if not actions_taken:
            actions_taken.append("WAIT")

        with open(os.path.join(self._orch_output_dir, f"step_{step_num:03d}_action.txt"), "w") as f:
            f.write("\n".join(actions_taken))

        if peek_ids and result_action != "DONE":
            followup = self._handle_peeks(bedrock, peek_ids, running, completed, step_num)
            if followup == "DONE":
                result_action = "DONE"

        return result_action

    def _handle_peeks(
        self,
        bedrock: Any,
        peek_ids: List[str],
        running: List[Subtask],
        completed: List[Subtask],
        step_num: int,
    ) -> str:
        """Show peeked agents' full conversation history and ask LLM for follow-up."""
        elapsed = time.time() - self._start_time
        parts = [
            f"[{elapsed:.0f}s elapsed] "
            f"You peeked at {len(peek_ids)} agent(s). "
            "Here is each agent's full conversation history (what it said and did at each step). "
            "Decide: MESSAGE, ASSIGN, WAIT, or DONE.\n"
        ]

        for agent_id in peek_ids:
            st = self._all_subtasks[agent_id]
            parts.append(f"\n--- [{agent_id}] display:{st.display_num} step:{st.step_count} ---")
            parts.append(f"Task: {st.task}")
            if st.step_history:
                parts.append("History:")
                for entry in st.step_history:
                    parts.append(f"  {entry}")
            else:
                parts.append("History: (no steps yet)")
            logger.info("[orchestrator] PEEK %s — %d steps in history", agent_id, len(st.step_history))

        prompt = "\n".join(parts)

        with open(os.path.join(self._orch_output_dir, f"step_{step_num:03d}_peek_prompt.txt"), "w") as f:
            f.write(prompt)

        self._orch_history.append(
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        )

        try:
            content_blocks, _ = bedrock.chat(
                messages=self._orch_history,
                system="You are a task orchestrator reviewing agent progress. Be concise and decisive.",
                model=self.model,
                temperature=0.3,
                max_tokens=4096,
                tools=_ORCHESTRATOR_TOOLS,
            )
        except Exception as e:
            logger.warning("[orchestrator] PEEK follow-up LLM failed: %s", e)
            self._orch_history.pop()
            return "WAIT"

        self._orch_history.append({"role": "assistant", "content": content_blocks})

        response_text = "".join(
            b.get("text", "") for b in content_blocks
            if isinstance(b, dict) and b.get("type") == "text"
        )
        logger.info("[orchestrator] PEEK decision #%d: %s", step_num, response_text[:300])

        with open(os.path.join(self._orch_output_dir, f"step_{step_num:03d}_peek_response.txt"), "w") as f:
            f.write(response_text)

        result_action = "WAIT"
        peek_actions: List[str] = []
        tool_results: List[Dict[str, Any]] = []

        for block in content_blocks:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue

            tool_name = block.get("name", "")
            tool_id = block.get("id", "")
            tool_input = block.get("input", {})
            agent_id = _clean_agent_id(tool_input.get("agent_id", ""))

            if tool_name == "message_agent":
                msg = tool_input.get("message", "")
                if agent_id in self._all_subtasks and self._all_subtasks[agent_id].status == "running":
                    self.send_message(agent_id, msg)
                    peek_actions.append(f"MESSAGE {agent_id}: {msg[:80]}")
                    logger.info("[orchestrator] PEEK -> %s: %s", agent_id, msg[:120])
                tool_results.append({
                    "type": "tool_result", "tool_use_id": tool_id,
                    "content": f"Message delivered to '{agent_id}'.",
                })

            elif tool_name == "assign_subtask":
                task_desc = tool_input.get("task", "")
                setup = tool_input.get("setup", [])
                if task_desc:
                    self._assign_new_subtask_structured(agent_id, task_desc, setup)
                    peek_actions.append(f"ASSIGN {agent_id}: {task_desc[:80]}")
                tool_results.append({
                    "type": "tool_result", "tool_use_id": tool_id,
                    "content": f"Agent '{agent_id}' launched.",
                })

            elif tool_name == "mark_done":
                result_action = "DONE"
                peek_actions.append("DONE")
                tool_results.append({
                    "type": "tool_result", "tool_use_id": tool_id,
                    "content": "Task marked as done.",
                })

        # Ensure EVERY tool_use block has a matching tool_result
        handled_ids = {tr["tool_use_id"] for tr in tool_results}
        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                if block.get("id") not in handled_ids:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block["id"],
                        "content": "Acknowledged.",
                    })

        if tool_results:
            self._orch_history.append({"role": "user", "content": tool_results})

        if not peek_actions:
            peek_actions.append("WAIT")

        with open(os.path.join(self._orch_output_dir, f"step_{step_num:03d}_peek_action.txt"), "w") as f:
            f.write("\n".join(peek_actions))

        return result_action

    def _assign_new_subtask_structured(
        self, agent_id: str, task_desc: str, setup: Optional[List[Dict[str, Any]]] = None
    ):
        """Create and launch a new subtask (structured input from tool use)."""
        if agent_id in self._all_subtasks:
            self._agent_counter += 1
            agent_id = f"{agent_id}_{self._agent_counter}"

        # Try to reuse a display from a recently completed agent (keeps app state)
        reuse_display = None
        prior_context = None
        with self._lock:
            if self._completed_displays:
                reuse_agent, reuse_display = next(iter(self._completed_displays.items()))
                del self._completed_displays[reuse_agent]
                prev_subtask = self._all_subtasks.get(reuse_agent)
                if prev_subtask:
                    prior_context = {
                        "agent_id": reuse_agent,
                        "task": prev_subtask.task,
                        "result": prev_subtask.result.get("summary", "") if prev_subtask.result else "",
                        "step_history": prev_subtask.step_history,
                    }
                logger.info(
                    "[orchestrator] Reusing display :%d from completed agent %s",
                    reuse_display, reuse_agent,
                )

        subtask = Subtask(id=agent_id, task=task_desc, setup=setup or [])
        with self._lock:
            self._all_subtasks[agent_id] = subtask

        self._launch_agent(subtask, reuse_display=reuse_display, prior_context=prior_context)
        self._log_event("assign", agent_id=agent_id, task=task_desc[:200],
                       reused_display=reuse_display)
        logger.info("[orchestrator] Assigned new subtask: %s — %s", agent_id, task_desc[:80])

    def _log_event(self, event_type: str, **kwargs):
        """Record an event to the execution log."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        entry = {"time": round(elapsed, 1), "event": event_type, **kwargs}
        self._execution_log.append(entry)
        # Save after each event so it's always up to date
        log_path = os.path.join(self._orch_output_dir, "execution_log.json")
        try:
            with open(log_path, "w") as f:
                json.dump(self._execution_log, f, indent=2)
        except Exception:
            pass

    def send_message(self, agent_id: str, message: str):
        """Send a coordination message to a running subagent."""
        with self._lock:
            self._messages.setdefault(agent_id, []).append(message)
        self._log_event("message", agent_id=agent_id, message=message[:200])
        logger.info("Message -> %s: %s", agent_id, message[:120])

    def get_pending_messages(self, agent_id: str) -> List[str]:
        """Retrieve and clear pending messages for a subagent."""
        with self._lock:
            return self._messages.pop(agent_id, [])

    def get_all_bedrock_clients(self) -> Dict[str, Any]:
        """Return all bedrock clients for token usage aggregation."""
        with self._lock:
            return dict(self._bedrock_clients)
