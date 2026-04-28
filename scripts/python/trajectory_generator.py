"""Generate trajectory HTML visualization for fork-based parallel agent execution."""

import html as html_mod
import json
import pathlib
import re
from typing import Dict, List, Optional


def generate_trajectory_html(
    local_dir: str,
    task_id: str,
    github_repo: str,
    github_path: str,
    task_type: str,
    config_name: str,
    trial: int,
) -> None:
    """Generate trajectory.html for fork-based agent runs.

    Args:
        local_dir: Local directory containing task results (e.g., task_01b269ae-collaborative/)
        task_id: Task ID
        github_repo: GitHub repo for raw image URLs (e.g., samuellin01/memory_experiments_3)
        github_path: Path prefix in repo (e.g., osworld)
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

    # Discover agent directories (root, root_child_0, root_child_1, etc.)
    agent_dirs: List[tuple[str, pathlib.Path]] = []
    print(f"Scanning for agent directories in: {local_path}")
    for d in sorted(local_path.iterdir()):
        print(f"  Found: {d.name} (is_dir={d.is_dir()})")
        if not d.is_dir():
            continue
        # Match agent directories (root, root_child_0, root_child_1, etc.)
        if d.name == "root" or d.name.startswith("root_child_"):
            agent_dirs.append((d.name, d))
            print(f"    -> Matched as agent directory")

    agent_dirs.sort()  # root comes first, then root_child_0, root_child_1, etc.
    print(f"Found {len(agent_dirs)} agent directories: {[name for name, _ in agent_dirs]}")

    total_agents = len(agent_dirs)
    duration = result_data.get("duration", 0)
    num_agents = result_data.get("num_agents", total_agents)
    forked = result_data.get("forked", num_agents > 1)

    # -- Helper -------------------------------------------------------------

    def esc(text: str) -> str:
        return html_mod.escape(text)

    def fmt_duration(secs: float) -> str:
        if secs < 60:
            return f"{secs:.0f}s"
        m, s = divmod(int(secs), 60)
        return f"{m}m {s}s"

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
.meta {{ display: flex; flex-wrap: wrap; gap: 12px 24px; margin-bottom: 20px;
        font-size: 0.85em; color: #b1bac4; }}
.meta span {{ background: #161b22; padding: 4px 10px; border-radius: 6px; }}
.meta strong {{ color: #e6edf3; }}
.score-pass {{ color: #56d364; }} .score-fail {{ color: #f85149; }}

details {{ margin-bottom: 8px; }}
summary {{ cursor: pointer; user-select: none; padding: 8px 12px;
          border-radius: 6px; font-weight: 600; color: #e6edf3; }}
summary:hover {{ background: #1c2128; }}

.tag {{ display: inline-block; font-size: 0.75em; padding: 2px 8px; border-radius: 12px;
       font-weight: 600; vertical-align: middle; margin-left: 8px; }}
.tag-display {{ background: #1f6feb44; color: #79c0ff; }}
.tag-duration {{ background: #30363d; color: #b1bac4; font-weight: 400; }}
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

    # Agents — flat list, each collapsible
    for agent_id, agent_dir in agent_dirs:
        # Read agent log if available
        log_path = agent_dir / "agent.log"
        if not log_path.is_file():
            continue

        # Discover step screenshots
        step_files: Dict[int, pathlib.Path] = {}
        for f in sorted(agent_dir.iterdir()):
            m = re.match(r"step_(\d+)\.png$", f.name)
            if m:
                step_files[int(m.group(1))] = f

        n_steps = len(step_files)

        # Determine status from result.json agents
        agents_info = result_data.get("agents", {})
        agent_info = agents_info.get(agent_id, {})
        status = agent_info.get("status", "unknown")
        display_num = agent_info.get("display", "?")

        # Display tag
        disp_tag = ""
        if display_num != "?":
            disp_label = "primary" if display_num == 0 else f"display {display_num}"
            disp_tag = f" <span class='tag tag-display'>{disp_label}</span>"

        # Status tag
        status_tag = ""
        if status == "completed":
            status_tag = " <span class='tag tag-status'>COMPLETED</span>"
        elif status == "failed":
            status_tag = " <span class='tag tag-status-fail'>FAILED</span>"

        agent_label = "Root agent" if agent_id == "root" else f"Worker {agent_id}"

        h.append(f"<details class='agent'>\n")
        h.append(
            f"  <summary>{agent_label} ({n_steps} steps)"
            f"{disp_tag}{status_tag}</summary>\n"
        )
        h.append("  <div class='agent-body'>\n")

        # Show steps
        for step_num in sorted(step_files.keys()):
            step_file = step_files[step_num]

            h.append(f"    <details class='step'>\n")
            h.append(f"      <summary>Step {step_num}</summary>\n")
            h.append("      <div class='step-content'>\n")

            # Screenshot
            rel = step_file.relative_to(local_path)
            img_url = f"{img_base}/{rel}"
            h.append(f"        <img src='{img_url}' alt='Step {step_num}' loading='lazy'>\n")

            h.append("      </div>\n")
            h.append("    </details>\n")

        h.append("  </div>\n")
        h.append("</details>\n")

    h.append("</body></html>\n")

    html_path = local_path / "trajectory.html"
    html_path.write_text("".join(h), encoding="utf-8")
    print(f"Generated {html_path} ({total_agents} agents)")
