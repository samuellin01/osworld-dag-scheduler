"""Generate interactive trajectory HTML visualization for fork-based parallel agent execution."""

import html as html_mod
import json
import pathlib
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple


def parse_bedrock_api_calls(local_path: pathlib.Path) -> List[Dict]:
    """Parse bedrock_api_calls.jsonl to extract action details and timestamps."""
    api_calls_path = local_path / "bedrock_api_calls.jsonl"
    if not api_calls_path.is_file():
        return []

    calls = []
    try:
        with open(api_calls_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    calls.append(json.loads(line))
    except (json.JSONDecodeError, OSError):
        pass

    return calls


def build_timeline_from_api_calls(api_calls: List[Dict], agent_dirs: List[Tuple[str, pathlib.Path]]) -> Dict:
    """Build timeline data from bedrock API calls with actual timestamps."""

    # Track steps per agent
    agent_steps_map = {}  # agent_id -> {step_num -> {'timestamp': ..., 'action': ...}}
    agent_step_counters = {}  # agent_id -> current step number

    start_time = None
    end_time = None

    for call in api_calls:
        # Get timestamps
        request_ts_str = call.get('request_timestamp')
        response_ts_str = call.get('response_timestamp')

        if not response_ts_str:
            continue

        # Parse ISO timestamp
        try:
            response_ts = datetime.fromisoformat(response_ts_str.replace('Z', '+00:00'))
        except:
            continue

        if start_time is None:
            start_time = response_ts
        end_time = response_ts

        relative_time = (response_ts - start_time).total_seconds() if start_time else 0

        # Determine agent ID from system prompt
        request = call.get('request', {})
        system_prompt = request.get('system_prompt', '')
        tools = request.get('tools', [])

        # Worker agents have "You are a worker agent" in system prompt
        # Root has fork_subtask in tools
        if 'You are a worker agent' in system_prompt:
            # This is a worker - need to figure out which one
            # Workers have different display numbers mentioned in system prompt
            # Look for "chrome_display_2" pattern
            display_match = re.search(r'chrome_display_(\d+)', system_prompt)
            if display_match:
                display_num = int(display_match.group(1))
                # Map display to agent ID (display 0 = root, display 2+ = workers)
                if display_num == 0:
                    agent_id = 'root'
                else:
                    # Find which child this is by display number
                    # This is a bit hacky - we'll use display number to infer
                    agent_id = f'root_child_{(display_num - 2) // 2}'  # Approximation
            else:
                agent_id = 'root_child_0'  # Fallback
        else:
            agent_id = 'root'

        # Increment step counter for this agent
        if agent_id not in agent_step_counters:
            agent_step_counters[agent_id] = 0
        agent_step_counters[agent_id] += 1
        step_num = agent_step_counters[agent_id]

        # Extract action details from response content blocks
        action_detail = ""
        tool_name = ""

        response = call.get('response', {})
        content_blocks = response.get('content_blocks', [])

        for block in content_blocks:
            if not isinstance(block, dict):
                continue

            # Text block
            if block.get('type') == 'text':
                text = block.get('text', '').strip()
                if text and not action_detail:
                    # Use first 200 chars of text
                    action_detail = text[:200]

            # Tool use block
            elif block.get('type') == 'tool_use':
                tool_name = block.get('name', '')
                tool_input = block.get('input', {})

                if tool_name == 'computer':
                    action = tool_input.get('action', '')
                    if action == 'key':
                        action_detail = f"Key: {tool_input.get('text', '')}"
                    elif action == 'type':
                        text = tool_input.get('text', '')
                        action_detail = f"Type: {text}"
                    elif action in ['left_click', 'right_click', 'middle_click', 'double_click']:
                        coord = tool_input.get('coordinate', [])
                        action_detail = f"{action.replace('_', ' ').title()} at {coord}"
                    elif action == 'mouse_move':
                        coord = tool_input.get('coordinate', [])
                        action_detail = f"Move to {coord}"
                    elif action == 'screenshot':
                        action_detail = "Screenshot"
                    elif action == 'zoom':
                        region = tool_input.get('region', [])
                        action_detail = f"Zoom to {region}"
                    else:
                        action_detail = f"Computer: {action}"

                elif tool_name == 'bash':
                    command = tool_input.get('command', '')
                    if len(command) > 100:
                        command = command[:100] + "..."
                    action_detail = f"Bash: {command}"

                elif tool_name == 'fork_subtask':
                    subtask = tool_input.get('subtask', '')
                    if len(subtask) > 150:
                        subtask = subtask[:150] + "..."
                    action_detail = f"Fork worker: {subtask}"

                elif tool_name == 'peek_child':
                    child_id = tool_input.get('child_id', '')
                    action_detail = f"Peek at {child_id}"

                elif tool_name == 'message_child':
                    child_id = tool_input.get('child_id', '')
                    message = tool_input.get('message', '')
                    action_detail = f"Message to {child_id}: {message[:50]}"

                else:
                    action_detail = f"Tool: {tool_name}"

                # Prefer tool action over text
                break

        # Store step data
        if agent_id not in agent_steps_map:
            agent_steps_map[agent_id] = {}

        agent_steps_map[agent_id][step_num] = {
            'timestamp': relative_time,
            'action': action_detail,
            'tool': tool_name,
        }

    # Build agent timeline spans (start/end times)
    agent_timeline = {}
    for agent_id, steps_dict in agent_steps_map.items():
        if steps_dict:
            step_nums = sorted(steps_dict.keys())
            start = steps_dict[step_nums[0]]['timestamp']
            end = steps_dict[step_nums[-1]]['timestamp']

            agent_timeline[agent_id] = {
                'start': start,
                'end': end,
                'steps': steps_dict,
            }

    total_duration = (end_time - start_time).total_seconds() if start_time and end_time else 0

    return {
        'agent_timeline': agent_timeline,
        'agent_steps_map': agent_steps_map,
        'total_duration': total_duration,
    }


def generate_trajectory_html(
    local_dir: str,
    task_id: str,
    github_repo: str,
    github_path: str,
    task_type: str,
    config_name: str,
    trial: int,
) -> None:
    """Generate interactive trajectory.html for fork-based agent runs.

    Args:
        local_dir: Local directory containing task results
        task_id: Task ID
        github_repo: GitHub repo for raw image URLs
        github_path: Path prefix in repo
        task_type: Task type (e.g., collaborative)
        config_name: Config name (e.g., fork_parallel)
        trial: Trial number
    """
    local_path = pathlib.Path(local_dir)
    if not local_path.is_dir():
        return

    img_base = (
        f"https://raw.githubusercontent.com/{github_repo}/main"
        f"/{github_path}/{task_type}/{task_id}/{config_name}/trial_{trial}"
    )

    # -- Gather data --------------------------------------------------------

    task_txt = local_path / "task.txt"
    instruction = ""
    if task_txt.is_file():
        instruction = task_txt.read_text(encoding="utf-8", errors="replace").strip()

    result_path = local_path / "result.txt"
    score_str = "N/A"
    if result_path.is_file():
        try:
            score_str = result_path.read_text(encoding="utf-8").strip()
        except OSError:
            pass

    result_json_path = local_path / "result.json"
    result_data: dict = {}
    if result_json_path.is_file():
        try:
            result_data = json.loads(result_json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    # Discover agent directories
    agent_dirs: List[Tuple[str, pathlib.Path]] = []
    for d in sorted(local_path.iterdir()):
        if not d.is_dir():
            continue
        if d.name == "root" or d.name.startswith("root_child_"):
            agent_dirs.append((d.name, d))

    agent_dirs.sort()

    total_agents = len(agent_dirs)
    duration = result_data.get("duration", 0)
    num_agents = result_data.get("num_agents", total_agents)
    forked = result_data.get("forked", num_agents > 1)

    # Parse API calls for timing data
    api_calls = parse_bedrock_api_calls(local_path)
    timeline_data = build_timeline_from_api_calls(api_calls, agent_dirs)

    agent_timeline = timeline_data['agent_timeline']
    agent_steps_map = timeline_data['agent_steps_map']
    total_duration = timeline_data['total_duration'] or duration

    # -- Helper -------------------------------------------------------------

    def esc(text: str) -> str:
        """HTML-escape text."""
        return html_mod.escape(text)

    def fmt_duration(secs: float) -> str:
        """Format duration in seconds to 'Xm Ys'."""
        m, s = divmod(int(secs), 60)
        return f"{m}m {s}s"

    # -- Build agent step data with screenshots -----------------------------

    agent_data = []
    for agent_id, agent_dir in agent_dirs:
        step_files = sorted([f for f in agent_dir.glob("step_*.png")])
        steps = []

        for step_file in step_files:
            m = re.match(r"step_(\d+)\.png$", step_file.name)
            if not m:
                continue
            step_num = int(m.group(1))

            # Get step info from timeline
            step_info = agent_steps_map.get(agent_id, {}).get(step_num, {})
            timestamp = step_info.get('timestamp', 0)
            action = step_info.get('action', '')

            # Screenshot URL
            screenshot_url = f"{img_base}/{agent_id}/step_{step_num:03d}.png"

            steps.append({
                'num': step_num,
                'timestamp': timestamp,
                'action': action,
                'screenshot': screenshot_url,
            })

        # Get agent timeline span
        timeline = agent_timeline.get(agent_id, {})
        start_time = timeline.get('start', 0)
        end_time = timeline.get('end', 0)

        agent_info = result_data.get("agents", {}).get(agent_id, {})
        status = agent_info.get("status", "unknown")
        display_num = agent_info.get("display", "?")

        agent_data.append({
            'id': agent_id,
            'display': display_num,
            'status': status,
            'start': start_time,
            'end': end_time,
            'steps': steps,
        })

    # -- Build action log ---------------------------------------------------

    all_actions = []
    for agent_id, steps_dict in agent_steps_map.items():
        for step_num in sorted(steps_dict.keys()):
            step_info = steps_dict[step_num]
            all_actions.append({
                'timestamp': step_info['timestamp'],
                'agent': agent_id,
                'step': step_num,
                'action': step_info['action'],
            })

    all_actions.sort(key=lambda x: x['timestamp'])

    # -- Build HTML ---------------------------------------------------------

    h: List[str] = []
    h.append("<!DOCTYPE html>")
    h.append("<html lang='en'>")
    h.append("<head>")
    h.append("<meta charset='utf-8'>")
    h.append(f"<title>Trajectory — {esc(task_id)}</title>")
    h.append("<style>")
    h.append("""
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       background: #0d1117; color: #e6edf3; padding: 24px; line-height: 1.6; }
h1 { font-size: 1.4em; margin-bottom: 8px; color: #f0f6fc; }
h2 { font-size: 1.2em; margin: 24px 0 12px 0; color: #f0f6fc; border-bottom: 1px solid #30363d; padding-bottom: 8px; }
.meta { display: flex; flex-wrap: wrap; gap: 12px 24px; margin-bottom: 20px;
        font-size: 0.85em; color: #b1bac4; }
.meta span { background: #161b22; padding: 4px 10px; border-radius: 6px; }
.meta strong { color: #e6edf3; }
.score-pass { color: #56d364; } .score-fail { color: #f85149; }

/* Timeline Scrubber */
.timeline-scrubber {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 16px;
    margin-bottom: 24px;
}
.timeline-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}
.timeline-time {
    font-size: 1.1em;
    color: #58a6ff;
    font-weight: 600;
}
.timeline-track-container {
    margin-bottom: 16px;
}
.timeline-slider {
    width: 100%;
    height: 8px;
    background: #0d1117;
    border-radius: 4px;
    position: relative;
    cursor: pointer;
    margin-bottom: 8px;
}
.timeline-progress {
    height: 100%;
    background: linear-gradient(90deg, #58a6ff 0%, #1f6feb 100%);
    border-radius: 4px;
    position: absolute;
    pointer-events: none;
}
.timeline-bars {
    position: relative;
    height: 40px;
    margin-top: 8px;
}
.timeline-bar {
    position: absolute;
    height: 16px;
    border-radius: 4px;
    cursor: pointer;
}
.timeline-bar.agent-root {
    background: linear-gradient(90deg, #3fb950 0%, #2ea043 100%);
    top: 0;
}
.timeline-bar.agent-child {
    background: linear-gradient(90deg, #58a6ff 0%, #1f6feb 100%);
}
.timeline-bar-label {
    position: absolute;
    font-size: 0.7em;
    color: #fff;
    padding: 2px 6px;
    white-space: nowrap;
    pointer-events: none;
}

/* Display Panels */
.display-panels {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
}
.display-panel {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 12px;
}
.display-panel-header {
    font-size: 0.9em;
    color: #b1bac4;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
}
.display-panel-step {
    font-size: 0.75em;
    color: #8b949e;
}
.display-panel img {
    width: 100%;
    height: auto;
    border-radius: 4px;
    border: 1px solid #30363d;
}
.display-panel-action {
    font-size: 0.8em;
    color: #e6edf3;
    margin-top: 8px;
    padding: 8px;
    background: #0d1117;
    border-radius: 4px;
    font-family: 'SF Mono', Monaco, monospace;
    word-wrap: break-word;
}

/* Action Log */
.action-log {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 16px;
    max-height: 500px;
    overflow-y: auto;
}
.action-item {
    display: grid;
    grid-template-columns: 80px 120px 1fr;
    gap: 12px;
    padding: 8px;
    border-bottom: 1px solid #21262d;
    font-size: 0.85em;
}
.action-item:last-child {
    border-bottom: none;
}
.action-time {
    color: #8b949e;
    font-family: 'SF Mono', Monaco, monospace;
}
.action-agent {
    color: #79c0ff;
}
.action-detail {
    color: #e6edf3;
    word-wrap: break-word;
}

/* Tabs */
.tabs {
    display: flex;
    gap: 8px;
    margin-bottom: 16px;
    border-bottom: 1px solid #30363d;
}
.tab {
    padding: 8px 16px;
    cursor: pointer;
    color: #8b949e;
    border-bottom: 2px solid transparent;
    transition: all 0.2s;
}
.tab:hover {
    color: #e6edf3;
}
.tab.active {
    color: #58a6ff;
    border-bottom-color: #58a6ff;
}
.tab-content {
    display: none;
}
.tab-content.active {
    display: block;
}
""")
    h.append("</style>")
    h.append("</head>")
    h.append("<body>")

    # Header
    h.append(f"<h1>Task {esc(task_id)}</h1>")
    h.append(f"<p style='margin-bottom:12px;color:#b1bac4;font-size:0.9em'>{esc(instruction)}</p>")

    # Meta info
    score_class = "score-pass" if score_str not in ["N/A", "0", "0.0"] else "score-fail"
    cost = result_data.get("cost", 0)
    h.append("<div class='meta'>")
    h.append(f"  <span>Score: <strong class='{score_class}'>{esc(score_str)}</strong></span>")
    h.append(f"  <span>Duration: <strong>{fmt_duration(duration)}</strong></span>")
    h.append(f"  <span>Agents: <strong>{num_agents}</strong> {'(parallelized)' if forked else ''}</span>")
    h.append(f"  <span>Cost: <strong>${cost:.2f}</strong></span>")
    h.append("</div>")

    # Timeline Scrubber
    h.append("<h2>⏱️ Execution Timeline</h2>")
    h.append("<div class='timeline-scrubber'>")
    h.append("  <div class='timeline-header'>")
    h.append("    <div class='timeline-time' id='timeline-time'>0:00</div>")
    h.append(f"    <div style='color:#8b949e;font-size:0.85em'>Total: {fmt_duration(total_duration)}</div>")
    h.append("  </div>")
    h.append("  <div class='timeline-track-container'>")
    h.append("    <div class='timeline-slider' id='timeline-slider'>")
    h.append("      <div class='timeline-progress' id='timeline-progress' style='width: 0%'></div>")
    h.append("    </div>")
    h.append("    <div class='timeline-bars'>")

    # Timeline bars for each agent
    for idx, agent in enumerate(agent_data):
        agent_id = agent['id']
        start_pct = (agent['start'] / total_duration * 100) if total_duration > 0 else 0
        duration_pct = ((agent['end'] - agent['start']) / total_duration * 100) if total_duration > 0 else 0

        is_root = (agent_id == 'root')
        bar_class = 'agent-root' if is_root else 'agent-child'
        top_offset = 0 if is_root else (idx * 20)

        h.append(f"    <div class='timeline-bar {bar_class}' style='left: {start_pct:.1f}%; width: {duration_pct:.1f}%; top: {top_offset}px' title='{esc(agent_id)}'>")
        h.append(f"      <div class='timeline-bar-label'>{esc(agent_id)}</div>")
        h.append("    </div>")

    h.append("    </div>")
    h.append("  </div>")
    h.append("</div>")

    # Display Panels
    h.append("<h2>📺 Displays</h2>")
    h.append("<div class='display-panels'>")

    for agent in agent_data:
        agent_id = agent['id']
        display = agent['display']
        h.append(f"  <div class='display-panel' id='panel-{esc(agent_id)}'>")
        h.append(f"    <div class='display-panel-header'>")
        h.append(f"      <span><strong>{esc(agent_id)}</strong> (Display {esc(str(display))})</span>")
        h.append(f"      <span class='display-panel-step' id='panel-step-{esc(agent_id)}'>Step —</span>")
        h.append(f"    </div>")
        h.append(f"    <img id='panel-img-{esc(agent_id)}' src='' alt='No screenshot' style='display:none'>")
        h.append(f"    <div id='panel-action-{esc(agent_id)}' class='display-panel-action' style='display:none'></div>")
        h.append(f"  </div>")

    h.append("</div>")

    # Tabs
    h.append("<div class='tabs'>")
    h.append("  <div class='tab active' onclick='showTab(\"log\")'>📋 Action Log</div>")
    h.append("</div>")

    # Action Log
    h.append("<div id='tab-log' class='tab-content active'>")
    h.append("  <div class='action-log'>")

    for action in all_actions:
        time_str = fmt_duration(action['timestamp'])
        h.append("    <div class='action-item'>")
        h.append(f"      <div class='action-time'>{esc(time_str)}</div>")
        h.append(f"      <div class='action-agent'>{esc(action['agent'])} step {action['step']}</div>")
        h.append(f"      <div class='action-detail'>{esc(action['action'])}</div>")
        h.append("    </div>")

    h.append("  </div>")
    h.append("</div>")

    # JavaScript for interactivity
    h.append("<script>")

    # Embed agent data as JSON
    h.append("const agentData = " + json.dumps(agent_data) + ";")
    h.append(f"const totalDuration = {total_duration};")

    h.append("""
let currentTime = 0;

function updateDisplays(time) {
    currentTime = time;

    // Update time display
    const mins = Math.floor(time / 60);
    const secs = Math.floor(time % 60);
    document.getElementById('timeline-time').textContent = mins + ':' + (secs < 10 ? '0' : '') + secs;

    // Update progress bar
    const progress = (time / totalDuration) * 100;
    document.getElementById('timeline-progress').style.width = progress + '%';

    // Update each agent display
    agentData.forEach(agent => {
        const agentId = agent.id;
        const imgEl = document.getElementById('panel-img-' + agentId);
        const stepEl = document.getElementById('panel-step-' + agentId);
        const actionEl = document.getElementById('panel-action-' + agentId);

        // Find the latest step at or before current time
        let currentStep = null;
        for (const step of agent.steps) {
            if (step.timestamp <= time) {
                currentStep = step;
            } else {
                break;
            }
        }

        if (currentStep) {
            imgEl.src = currentStep.screenshot;
            imgEl.style.display = 'block';
            stepEl.textContent = 'Step ' + currentStep.num;
            if (currentStep.action) {
                actionEl.textContent = currentStep.action;
                actionEl.style.display = 'block';
            } else {
                actionEl.style.display = 'none';
            }
        } else {
            imgEl.style.display = 'none';
            stepEl.textContent = 'Step —';
            actionEl.style.display = 'none';
        }
    });
}

// Timeline slider interaction
const slider = document.getElementById('timeline-slider');
slider.addEventListener('click', (e) => {
    const rect = slider.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const percent = x / rect.width;
    const time = percent * totalDuration;
    updateDisplays(time);
});

// Initialize at t=0
updateDisplays(0);

function showTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
    document.getElementById('tab-' + tabName).classList.add('active');
    event.target.classList.add('active');
}
""")
    h.append("</script>")

    h.append("</body></html>")

    # Write HTML
    output_path = local_path / "trajectory.html"
    output_path.write_text("\n".join(h), encoding="utf-8")
