"""Generate interactive trajectory HTML visualization for fork-based parallel agent execution."""

import html as html_mod
import json
import pathlib
import re
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


def parse_agent_steps(agent_dir: pathlib.Path, agent_id: str) -> List[Dict]:
    """Parse agent steps to extract timing and action details."""
    steps = []

    # Find all step response files
    for f in sorted(agent_dir.iterdir()):
        m = re.match(r"step_(\d+)_response\.txt$", f.name)
        if m:
            step_num = int(m.group(1))
            try:
                response_text = f.read_text(encoding='utf-8', errors='replace')
                steps.append({
                    'step': step_num,
                    'agent': agent_id,
                    'response': response_text[:500],  # First 500 chars
                })
            except OSError:
                pass

    return steps


def calculate_metrics(agent_dirs: List[Tuple[str, pathlib.Path]],
                     result_data: dict) -> Dict:
    """Calculate latency metrics for parallel execution."""
    total_duration = result_data.get("duration", 0)
    num_agents = len(agent_dirs)

    # Count steps per agent
    agent_steps = {}
    max_steps = 0
    total_steps = 0

    for agent_id, agent_dir in agent_dirs:
        step_count = len(list(agent_dir.glob("step_*.png")))
        agent_steps[agent_id] = step_count
        max_steps = max(max_steps, step_count)
        total_steps += step_count

    # Estimate agent durations (assuming ~2s per step)
    step_duration = 2.0  # seconds per step estimate
    agent_durations = {aid: steps * step_duration for aid, steps in agent_steps.items()}

    # Critical path = longest agent
    critical_path_agent = max(agent_durations, key=agent_durations.get) if agent_durations else None
    critical_path_duration = agent_durations.get(critical_path_agent, 0) if critical_path_agent else 0

    # Sequential time = sum of all agent times
    sequential_time = sum(agent_durations.values())

    # Speedup = sequential / actual
    speedup = sequential_time / total_duration if total_duration > 0 else 1.0

    # Efficiency = speedup / num_agents
    efficiency = speedup / num_agents if num_agents > 0 else 0.0

    # Overhead = actual - critical_path
    overhead = total_duration - critical_path_duration

    return {
        'total_duration': total_duration,
        'sequential_time': sequential_time,
        'critical_path_duration': critical_path_duration,
        'critical_path_agent': critical_path_agent,
        'speedup': speedup,
        'efficiency': efficiency,
        'overhead': overhead,
        'agent_steps': agent_steps,
        'agent_durations': agent_durations,
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

    token_usage_path = local_path / "token_usage.json"
    token_usage: dict = {}
    if token_usage_path.is_file():
        try:
            token_usage = json.loads(token_usage_path.read_text(encoding="utf-8"))
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

    # Calculate metrics
    metrics = calculate_metrics(agent_dirs, result_data)

    # -- Helper -------------------------------------------------------------

    def esc(text: str) -> str:
        return html_mod.escape(text)

    def fmt_duration(secs: float) -> str:
        if secs < 60:
            return f"{secs:.0f}s"
        m, s = divmod(int(secs), 60)
        return f"{m}m {s}s"

    # -- Build agent timeline data ------------------------------------------

    agent_timeline_data = []
    agents_info = result_data.get("agents", {})

    for agent_id, agent_dir in agent_dirs:
        step_files = sorted([f for f in agent_dir.glob("step_*.png")])
        n_steps = len(step_files)

        agent_info = agents_info.get(agent_id, {})
        status = agent_info.get("status", "unknown")
        display_num = agent_info.get("display", "?")

        # Estimate duration based on steps
        est_duration = metrics['agent_durations'].get(agent_id, 0)

        agent_timeline_data.append({
            'id': agent_id,
            'display': display_num,
            'status': status,
            'steps': n_steps,
            'duration': est_duration,
        })

    # -- Build HTML ---------------------------------------------------------

    h: List[str] = []
    h.append(f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Trajectory — {esc(task_id)}</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       background: #0d1117; color: #e6edf3; padding: 24px; line-height: 1.6; }}
h1 {{ font-size: 1.4em; margin-bottom: 8px; color: #f0f6fc; }}
h2 {{ font-size: 1.2em; margin: 24px 0 12px 0; color: #f0f6fc; border-bottom: 1px solid #30363d; padding-bottom: 8px; }}
.meta {{ display: flex; flex-wrap: wrap; gap: 12px 24px; margin-bottom: 20px;
        font-size: 0.85em; color: #b1bac4; }}
.meta span {{ background: #161b22; padding: 4px 10px; border-radius: 6px; }}
.meta strong {{ color: #e6edf3; }}
.score-pass {{ color: #56d364; }} .score-fail {{ color: #f85149; }}

/* Metrics Dashboard */
.metrics-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
    margin-bottom: 24px;
}}
.metric-card {{
    background: #161b22;
    padding: 12px;
    border-radius: 6px;
    border: 1px solid #30363d;
}}
.metric-label {{
    font-size: 0.75em;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}
.metric-value {{
    font-size: 1.5em;
    font-weight: 600;
    color: #e6edf3;
    margin-top: 4px;
}}
.metric-subtext {{
    font-size: 0.8em;
    color: #b1bac4;
    margin-top: 4px;
}}

/* Timeline */
.timeline {{
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 16px;
    margin-bottom: 24px;
}}
.timeline-row {{
    display: flex;
    align-items: center;
    margin-bottom: 8px;
    height: 32px;
}}
.timeline-label {{
    width: 150px;
    font-size: 0.85em;
    color: #b1bac4;
    flex-shrink: 0;
}}
.timeline-track {{
    flex: 1;
    height: 24px;
    background: #0d1117;
    border-radius: 4px;
    position: relative;
}}
.timeline-bar {{
    position: absolute;
    height: 100%;
    border-radius: 4px;
    cursor: pointer;
    transition: opacity 0.2s;
}}
.timeline-bar:hover {{
    opacity: 0.8;
}}
.timeline-bar.agent-root {{
    background: linear-gradient(90deg, #3fb950 0%, #2ea043 100%);
}}
.timeline-bar.agent-child {{
    background: linear-gradient(90deg, #58a6ff 0%, #1f6feb 100%);
}}

/* Action Log */
.action-log {{
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 16px;
    margin-bottom: 24px;
    max-height: 400px;
    overflow-y: auto;
}}
.action-item {{
    display: flex;
    gap: 12px;
    padding: 8px;
    border-bottom: 1px solid #21262d;
    font-size: 0.85em;
}}
.action-item:last-child {{
    border-bottom: none;
}}
.action-time {{
    color: #8b949e;
    min-width: 60px;
}}
.action-agent {{
    color: #79c0ff;
    min-width: 100px;
}}
.action-detail {{
    color: #e6edf3;
    flex: 1;
}}

/* Step viewer */
details {{ margin-bottom: 8px; }}
summary {{ cursor: pointer; user-select: none; padding: 8px 12px;
          border-radius: 6px; font-weight: 600; color: #e6edf3; }}
summary:hover {{ background: #1c2128; }}

.tag {{ display: inline-block; font-size: 0.75em; padding: 2px 8px; border-radius: 12px;
       font-weight: 600; vertical-align: middle; margin-left: 8px; }}
.tag-display {{ background: #1f6feb44; color: #79c0ff; }}
.tag-status {{ background: #56d36444; color: #56d364; }}
.tag-status-fail {{ background: #f8514944; color: #f85149; }}

.agent > summary {{ font-size: 0.95em; background: #1c2128; border: 1px solid #30363d; }}
.agent[open] > summary {{ border-bottom-left-radius: 0; border-bottom-right-radius: 0;
                            border-bottom: none; }}
.agent > .agent-body {{ border: 1px solid #30363d; border-top: none;
                           border-radius: 0 0 6px 6px; padding: 10px; margin-bottom: 8px; }}

.step > summary {{ font-size: 0.85em; color: #b1bac4; }}
.step-content {{ padding: 8px 0 8px 16px; }}
.step-content img {{ max-width: 100%; height: auto; border-radius: 6px;
                    border: 1px solid #30363d; margin-bottom: 8px; }}
.step-content pre {{ background: #161b22; padding: 10px; border-radius: 6px;
                    font-size: 0.82em; overflow-x: auto; white-space: pre-wrap;
                    word-break: break-word; color: #e6edf3; }}

/* Tabs */
.tabs {{
    display: flex;
    gap: 8px;
    margin-bottom: 16px;
    border-bottom: 1px solid #30363d;
}}
.tab {{
    padding: 8px 16px;
    cursor: pointer;
    color: #8b949e;
    border-bottom: 2px solid transparent;
    transition: all 0.2s;
}}
.tab:hover {{
    color: #e6edf3;
}}
.tab.active {{
    color: #58a6ff;
    border-bottom-color: #58a6ff;
}}
.tab-content {{
    display: none;
}}
.tab-content.active {{
    display: block;
}}
</style>
</head>
<body>
""")

    # Header
    h.append(f"<h1>Task {esc(task_id)}</h1>\n")
    if instruction:
        h.append(f"<p style='margin-bottom:12px;color:#b1bac4;font-size:0.9em'>{esc(instruction)}</p>\n")

    score_cls = "score-pass" if score_str not in ("N/A", "0.0", "0") else "score-fail"
    h.append("<div class='meta'>\n")
    h.append(f"  <span>Score: <strong class='{score_cls}'>{esc(score_str)}</strong></span>\n")
    h.append(f"  <span>Duration: <strong>{fmt_duration(duration)}</strong></span>\n")
    h.append(f"  <span>Agents: <strong>{num_agents}</strong> ({'parallelized' if forked else 'single agent'})</span>\n")
    cost = token_usage.get("total_cost_usd")
    if cost is not None:
        h.append(f"  <span>Cost: <strong>${cost:.2f}</strong></span>\n")
    h.append("</div>\n")

    # Metrics Dashboard
    h.append("<h2>📊 Performance Metrics</h2>\n")
    h.append("<div class='metrics-grid'>\n")

    h.append("  <div class='metric-card'>\n")
    h.append("    <div class='metric-label'>Speedup</div>\n")
    h.append(f"    <div class='metric-value'>{metrics['speedup']:.2f}x</div>\n")
    h.append(f"    <div class='metric-subtext'>vs sequential ({fmt_duration(metrics['sequential_time'])})</div>\n")
    h.append("  </div>\n")

    h.append("  <div class='metric-card'>\n")
    h.append("    <div class='metric-label'>Efficiency</div>\n")
    h.append(f"    <div class='metric-value'>{metrics['efficiency']*100:.1f}%</div>\n")
    h.append(f"    <div class='metric-subtext'>of {num_agents} agents utilized</div>\n")
    h.append("  </div>\n")

    h.append("  <div class='metric-card'>\n")
    h.append("    <div class='metric-label'>Critical Path</div>\n")
    h.append(f"    <div class='metric-value'>{fmt_duration(metrics['critical_path_duration'])}</div>\n")
    if metrics['critical_path_agent']:
        agent_label = "Root" if metrics['critical_path_agent'] == "root" else metrics['critical_path_agent']
        h.append(f"    <div class='metric-subtext'>{agent_label} ({metrics['agent_steps'][metrics['critical_path_agent']]} steps)</div>\n")
    h.append("  </div>\n")

    h.append("  <div class='metric-card'>\n")
    h.append("    <div class='metric-label'>Overhead</div>\n")
    h.append(f"    <div class='metric-value'>{fmt_duration(metrics['overhead'])}</div>\n")
    h.append("    <div class='metric-subtext'>coordination + idle time</div>\n")
    h.append("  </div>\n")

    h.append("</div>\n")

    # Execution Timeline
    h.append("<h2>⏱️ Execution Timeline</h2>\n")
    h.append("<div class='timeline'>\n")

    max_duration = max([a['duration'] for a in agent_timeline_data]) if agent_timeline_data else 1.0

    for agent_data in agent_timeline_data:
        agent_id = agent_data['id']
        agent_label = "Root" if agent_id == "root" else f"Worker {agent_id}"
        display_label = f"display {agent_data['display']}" if agent_data['display'] != "?" else "?"

        width_pct = (agent_data['duration'] / max_duration) * 100 if max_duration > 0 else 0

        bar_class = "agent-root" if agent_id == "root" else "agent-child"

        h.append("  <div class='timeline-row'>\n")
        h.append(f"    <div class='timeline-label'>{esc(agent_label)} ({agent_data['steps']} steps)</div>\n")
        h.append("    <div class='timeline-track'>\n")
        h.append(f"      <div class='timeline-bar {bar_class}' style='width: {width_pct:.1f}%; left: 0;' ")
        h.append(f"title='{agent_label}: {agent_data['steps']} steps, ~{fmt_duration(agent_data['duration'])} on {display_label}'></div>\n")
        h.append("    </div>\n")
        h.append("  </div>\n")

    h.append("</div>\n")

    # Tabs for different views
    h.append("<div class='tabs'>\n")
    h.append("  <div class='tab active' onclick='showTab(\"steps\")'>📸 Steps</div>\n")
    h.append("  <div class='tab' onclick='showTab(\"log\")'>📋 Action Log</div>\n")
    h.append("</div>\n")

    # Tab: Action Log
    h.append("<div id='tab-log' class='tab-content'>\n")
    h.append("  <div class='action-log'>\n")

    # Build chronological action log from all agents
    all_actions = []
    for agent_id, agent_dir in agent_dirs:
        steps = parse_agent_steps(agent_dir, agent_id)
        for step_data in steps:
            # Extract action type from response
            response = step_data['response']
            action_type = "unknown"
            if "pyautogui.click" in response or "click" in response.lower():
                action_type = "click"
            elif "xdotool type" in response or "type" in response.lower():
                action_type = "type"
            elif "fork_subtask" in response:
                action_type = "fork"
            elif "screenshot" in response.lower():
                action_type = "screenshot"

            all_actions.append({
                'agent': agent_id,
                'step': step_data['step'],
                'action': action_type,
                'detail': response[:100]  # First 100 chars
            })

    # Sort by agent and step (approximation of chronological order)
    all_actions.sort(key=lambda x: (x['agent'], x['step']))

    for i, action in enumerate(all_actions[:50]):  # Limit to 50 for performance
        agent_label = "root" if action['agent'] == "root" else action['agent']
        h.append("    <div class='action-item'>\n")
        h.append(f"      <div class='action-time'>Step {action['step']}</div>\n")
        h.append(f"      <div class='action-agent'>{esc(agent_label)}</div>\n")
        h.append(f"      <div class='action-detail'>{esc(action['detail'])}</div>\n")
        h.append("    </div>\n")

    if len(all_actions) > 50:
        h.append(f"    <div style='text-align:center;padding:12px;color:#8b949e'>... and {len(all_actions) - 50} more actions</div>\n")

    h.append("  </div>\n")
    h.append("</div>\n")

    # Tab: Step Details
    h.append("<div id='tab-steps' class='tab-content active'>\n")

    # Agents — flat list, each collapsible
    for agent_id, agent_dir in agent_dirs:
        step_files: Dict[int, pathlib.Path] = {}
        for f in sorted(agent_dir.iterdir()):
            m = re.match(r"step_(\d+)\.png$", f.name)
            if m:
                step_files[int(m.group(1))] = f

        n_steps = len(step_files)

        agents_info = result_data.get("agents", {})
        agent_info = agents_info.get(agent_id, {})
        status = agent_info.get("status", "unknown")
        display_num = agent_info.get("display", "?")

        disp_tag = ""
        if display_num != "?":
            disp_label = "primary" if display_num == 0 else f"display {display_num}"
            disp_tag = f" <span class='tag tag-display'>{disp_label}</span>"

        status_tag = ""
        if status == "completed":
            status_tag = " <span class='tag tag-status'>COMPLETED</span>"
        elif status == "failed":
            status_tag = " <span class='tag tag-status-fail'>FAILED</span>"

        agent_label = "Root agent" if agent_id == "root" else f"Worker {agent_id}"

        h.append(f"<details class='agent' id='agent-{agent_id}'>\n")
        h.append(
            f"  <summary>{agent_label} ({n_steps} steps)"
            f"{disp_tag}{status_tag}</summary>\n"
        )
        h.append("  <div class='agent-body'>\n")

        for step_num in sorted(step_files.keys()):
            step_file = step_files[step_num]

            h.append(f"    <details class='step' id='step-{agent_id}-{step_num}'>\n")
            h.append(f"      <summary>Step {step_num}</summary>\n")
            h.append("      <div class='step-content'>\n")

            rel = step_file.relative_to(local_path)
            img_url = f"{img_base}/{rel}"
            h.append(f"        <img src='{img_url}' alt='Step {step_num}' loading='lazy'>\n")

            h.append("      </div>\n")
            h.append("    </details>\n")

        h.append("  </div>\n")
        h.append("</details>\n")

    h.append("</div>\n")

    # JavaScript for interactivity
    h.append("""\
<script>
function showTab(tabName) {
  // Hide all tab contents
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));

  // Show selected tab
  document.getElementById('tab-' + tabName).classList.add('active');
  event.target.classList.add('active');
}
</script>
""")

    h.append("</body></html>\n")

    html_path = local_path / "trajectory.html"
    html_path.write_text("".join(h), encoding="utf-8")
    print(f"Generated {html_path} ({total_agents} agents)")
