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
                    # Use full text (no truncation)
                    action_detail = text

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
                    action_detail = f"Bash: {command}"

                elif tool_name == 'fork_subtask':
                    subtask = tool_input.get('subtask', '')
                    action_detail = f"Fork worker: {subtask}"

                elif tool_name == 'peek_child':
                    child_id = tool_input.get('child_id', '')
                    action_detail = f"Peek at {child_id}"

                elif tool_name == 'message_child':
                    child_id = tool_input.get('child_id', '')
                    message = tool_input.get('message', '')
                    action_detail = f"Message to {child_id}: {message}"

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


def calculate_cost_from_tokens(local_path: pathlib.Path) -> float:
    """Calculate cost from token_usage.json using Opus 4 pricing."""
    token_usage_path = local_path / "token_usage.json"
    if not token_usage_path.is_file():
        return 0.0

    try:
        with open(token_usage_path, 'r', encoding='utf-8') as f:
            usage = json.load(f)

        # Opus 4.6 pricing (as of early 2026)
        # Input: $15 per million tokens
        # Output: $75 per million tokens
        input_tokens = usage.get('input_tokens', 0)
        output_tokens = usage.get('output_tokens', 0)
        cache_read_tokens = usage.get('cache_read_input_tokens', 0)

        # Cache read is cheaper: $1.50 per million
        input_cost = (input_tokens / 1_000_000) * 15.0
        cache_cost = (cache_read_tokens / 1_000_000) * 1.5
        output_cost = (output_tokens / 1_000_000) * 75.0

        total_cost = input_cost + cache_cost + output_cost
        return total_cost

    except (json.JSONDecodeError, OSError):
        return 0.0


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

    # Calculate cost from token usage
    cost = calculate_cost_from_tokens(local_path)

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
    # Calculate timeline height based on number of agents
    timeline_height = 40 + max(0, (total_agents - 1)) * 24

    h.append("<style>")
    h.append(f"""
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
    background: #0d1117;
    color: #e6edf3;
    padding: 24px;
    line-height: 1.6;
}}
h1 {{
    font-size: 1.6em;
    margin-bottom: 12px;
    color: #f0f6fc;
    font-weight: 700;
    letter-spacing: -0.02em;
}}
h2 {{
    font-size: 1.1em;
    margin: 32px 0 16px 0;
    color: #f0f6fc;
    border-bottom: 2px solid #21262d;
    padding-bottom: 8px;
    font-weight: 600;
    letter-spacing: -0.01em;
}}
.meta {{
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-bottom: 24px;
    font-size: 0.9em;
}}
.meta span {{
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    padding: 8px 14px;
    border-radius: 8px;
    border: 1px solid #30363d;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
}}
.meta strong {{ color: #e6edf3; font-weight: 600; }}
.score-pass {{ color: #3fb950; font-weight: 700; }}
.score-fail {{ color: #f85149; font-weight: 700; }}

/* Timeline Scrubber */
.timeline-scrubber {{
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 28px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
}}
.timeline-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
}}
.timeline-time {{
    font-size: 1.4em;
    background: linear-gradient(90deg, #58a6ff 0%, #79c0ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
}}
.timeline-track-container {{
    margin-bottom: 20px;
}}
.timeline-slider {{
    width: 100%;
    height: 20px;
    background: #0d1117;
    border-radius: 10px;
    position: relative;
    cursor: pointer;
    margin-bottom: 16px;
    border: 1px solid #21262d;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.5);
}}
.timeline-progress {{
    height: 100%;
    background: linear-gradient(90deg, #58a6ff 0%, #1f6feb 100%);
    border-radius: 10px;
    position: absolute;
    transition: width 0.1s ease-out;
    box-shadow: 0 0 8px rgba(88, 166, 255, 0.4);
}}
.timeline-knob {{
    position: absolute;
    width: 28px;
    height: 28px;
    background: linear-gradient(135deg, #58a6ff 0%, #1f6feb 100%);
    border: 3px solid #0d1117;
    border-radius: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
    cursor: grab;
    box-shadow: 0 2px 8px rgba(88, 166, 255, 0.6), 0 0 0 4px rgba(88, 166, 255, 0.1);
    transition: transform 0.2s, box-shadow 0.2s;
}}
.timeline-knob:hover {{
    transform: translate(-50%, -50%) scale(1.1);
    box-shadow: 0 4px 12px rgba(88, 166, 255, 0.8), 0 0 0 6px rgba(88, 166, 255, 0.15);
}}
.timeline-knob:active {{
    cursor: grabbing;
    transform: translate(-50%, -50%) scale(0.95);
}}
.timeline-bars {{
    position: relative;
    height: {timeline_height}px;
    margin-top: 12px;
}}
.timeline-bar {{
    position: absolute;
    height: 20px;
    border-radius: 6px;
    cursor: pointer;
    transition: transform 0.2s, box-shadow 0.2s;
    border: 1px solid rgba(255, 255, 255, 0.1);
}}
.timeline-bar:hover {{
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
}}
.timeline-bar.agent-root {{
    background: linear-gradient(90deg, #3fb950 0%, #2ea043 100%);
    box-shadow: 0 2px 8px rgba(63, 185, 80, 0.3);
    top: 0;
}}
.timeline-bar.agent-child {{
    background: linear-gradient(90deg, #58a6ff 0%, #1f6feb 100%);
    box-shadow: 0 2px 8px rgba(88, 166, 255, 0.3);
}}
.timeline-bar-label {{
    position: absolute;
    font-size: 0.7em;
    color: #fff;
    padding: 2px 8px;
    white-space: nowrap;
    pointer-events: none;
    font-weight: 600;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
}}
.timeline-playhead {{
    position: absolute;
    width: 2px;
    height: 100%;
    background: #f85149;
    top: 0;
    pointer-events: none;
    box-shadow: 0 0 8px rgba(248, 81, 73, 0.8);
    transition: left 0.1s ease-out;
}}
.timeline-playhead::before {{
    content: '';
    position: absolute;
    top: -4px;
    left: -3px;
    width: 8px;
    height: 8px;
    background: #f85149;
    border-radius: 50%;
    box-shadow: 0 0 8px rgba(248, 81, 73, 1);
}}

/* Display Panels */
.display-panels {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
    gap: 20px;
    margin-bottom: 28px;
}}
.display-panel {{
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    transition: transform 0.2s, box-shadow 0.2s;
}}
.display-panel:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.5);
}}
.display-panel-header {{
    font-size: 0.9em;
    color: #b1bac4;
    margin-bottom: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}}
.display-panel-header strong {{
    color: #58a6ff;
    font-weight: 600;
}}
.display-panel-step {{
    font-size: 0.75em;
    color: #8b949e;
    font-variant-numeric: tabular-nums;
}}
.display-panel img {{
    width: 100%;
    height: auto;
    border-radius: 8px;
    border: 1px solid #30363d;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
}}
.display-panel-action {{
    font-size: 0.8em;
    color: #e6edf3;
    margin-top: 12px;
    padding: 10px;
    background: #0d1117;
    border-radius: 6px;
    font-family: 'SF Mono', Monaco, 'Courier New', monospace;
    word-wrap: break-word;
    border: 1px solid #21262d;
    line-height: 1.5;
}}

/* Action Log */
.action-log {{
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 16px;
    max-height: 600px;
    overflow-y: auto;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
}}
.action-log::-webkit-scrollbar {{
    width: 12px;
}}
.action-log::-webkit-scrollbar-track {{
    background: #0d1117;
    border-radius: 6px;
}}
.action-log::-webkit-scrollbar-thumb {{
    background: #30363d;
    border-radius: 6px;
    border: 2px solid #0d1117;
}}
.action-log::-webkit-scrollbar-thumb:hover {{
    background: #484f58;
}}
.action-item {{
    display: grid;
    grid-template-columns: 90px 140px 1fr;
    gap: 16px;
    padding: 10px;
    border-bottom: 1px solid #21262d;
    font-size: 0.85em;
    transition: background 0.15s;
}}
.action-item:hover {{
    background: rgba(56, 139, 253, 0.05);
}}
.action-item:last-child {{
    border-bottom: none;
}}
.action-time {{
    color: #8b949e;
    font-family: 'SF Mono', Monaco, monospace;
    font-variant-numeric: tabular-nums;
    font-weight: 500;
}}
.action-agent {{
    color: #79c0ff;
    font-weight: 600;
}}
.action-detail {{
    color: #e6edf3;
    word-wrap: break-word;
    line-height: 1.5;
}}

/* Tabs */
.tabs {{
    display: flex;
    gap: 4px;
    margin-bottom: 16px;
    border-bottom: 2px solid #21262d;
}}
.tab {{
    padding: 10px 18px;
    cursor: pointer;
    color: #8b949e;
    border-bottom: 3px solid transparent;
    transition: all 0.2s;
    font-weight: 500;
    border-radius: 6px 6px 0 0;
}}
.tab:hover {{
    color: #e6edf3;
    background: rgba(56, 139, 253, 0.05);
}}
.tab.active {{
    color: #58a6ff;
    border-bottom-color: #58a6ff;
    background: rgba(56, 139, 253, 0.08);
}}
.tab-content {{
    display: none;
}}
.tab-content.active {{
    display: block;
}}
""")
    h.append("</style>")
    h.append("</head>")
    h.append("<body>")

    # Header
    h.append(f"<h1>Task {esc(task_id)}</h1>")
    h.append(f"<p style='margin-bottom:12px;color:#b1bac4;font-size:0.9em'>{esc(instruction)}</p>")

    # Meta info
    score_class = "score-pass" if score_str not in ["N/A", "0", "0.0"] else "score-fail"
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
    h.append("      <div class='timeline-knob' id='timeline-knob' style='left: 0%'></div>")
    h.append("    </div>")
    h.append("    <div class='timeline-bars' id='timeline-bars'>")

    # Timeline bars for each agent
    for idx, agent in enumerate(agent_data):
        agent_id = agent['id']
        start_pct = (agent['start'] / total_duration * 100) if total_duration > 0 else 0
        duration_pct = ((agent['end'] - agent['start']) / total_duration * 100) if total_duration > 0 else 0

        is_root = (agent_id == 'root')
        bar_class = 'agent-root' if is_root else 'agent-child'
        top_offset = 0 if is_root else (idx * 24)

        h.append(f"    <div class='timeline-bar {bar_class}' style='left: {start_pct:.1f}%; width: {duration_pct:.1f}%; top: {top_offset}px' title='{esc(agent_id)}'>")
        h.append(f"      <div class='timeline-bar-label'>{esc(agent_id)}</div>")
        h.append("    </div>")

    h.append("      <div class='timeline-playhead' id='timeline-playhead' style='left: 0%'></div>")
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
let isDragging = false;

function updateDisplays(time) {
    currentTime = time;

    // Update time display
    const mins = Math.floor(time / 60);
    const secs = Math.floor(time % 60);
    document.getElementById('timeline-time').textContent = mins + ':' + (secs < 10 ? '0' : '') + secs;

    // Update progress bar and knob
    const progress = (time / totalDuration) * 100;
    document.getElementById('timeline-progress').style.width = progress + '%';
    document.getElementById('timeline-knob').style.left = progress + '%';
    document.getElementById('timeline-playhead').style.left = progress + '%';

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
const knob = document.getElementById('timeline-knob');

function seekToPosition(clientX) {
    const rect = slider.getBoundingClientRect();
    const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
    const percent = x / rect.width;
    const time = percent * totalDuration;
    updateDisplays(time);
}

// Click on slider
slider.addEventListener('click', (e) => {
    if (e.target === slider || e.target === document.getElementById('timeline-progress')) {
        seekToPosition(e.clientX);
    }
});

// Drag knob
knob.addEventListener('mousedown', (e) => {
    isDragging = true;
    e.preventDefault();
});

document.addEventListener('mousemove', (e) => {
    if (isDragging) {
        seekToPosition(e.clientX);
    }
});

document.addEventListener('mouseup', () => {
    isDragging = false;
});

// Keyboard navigation
document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowLeft') {
        updateDisplays(Math.max(0, currentTime - 5));
    } else if (e.key === 'ArrowRight') {
        updateDisplays(Math.min(totalDuration, currentTime + 5));
    } else if (e.key === 'Home') {
        updateDisplays(0);
    } else if (e.key === 'End') {
        updateDisplays(totalDuration);
    }
});

// Initialize at first step of first agent
if (agentData.length > 0 && agentData[0].steps.length > 0) {
    updateDisplays(agentData[0].steps[0].timestamp);
}

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
