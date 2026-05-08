"""Extract a semantic action DAG from trial output.

Reads step files (actions, responses, timestamps) from a trial's output
directory and produces a structured DAG of semantically labeled actions.

For baseline (single-agent) runs: sequential chain.
For multi-agent runs: parallel branches with dependency edges.

Uses an LLM to convert raw actions (click at [540, 320]) into readable
labels (click [Save button]) while preserving exact step count and timing.

Usage:
    python extract_action_dag.py batch_results/trial_21/task_XXX/ -o dag.json
    python extract_action_dag.py batch_results/trial_21/task_XXX/ --format mermaid
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_SKIP_DIRS = {"_manager", "monitor", "planner"}


@dataclass
class ActionNode:
    agent_id: str
    phase_id: str
    step: int
    timestamp: float
    latency: float
    raw_action: Dict[str, Any]
    raw_action_summary: str
    reasoning: str
    semantic_label: str
    screenshot_path: str


@dataclass
class ActionDAG:
    task: str
    task_dir: str
    total_steps: int
    total_duration: float
    agents: List[Dict[str, Any]]
    nodes: List[ActionNode]
    edges: List[Dict[str, str]]


def _read_step_files(phase_dir: str) -> List[Dict[str, Any]]:
    steps = []
    step_nums = set()
    for f in os.listdir(phase_dir):
        m = re.match(r"step_(\d+)_action\.txt$", f)
        if m:
            step_nums.add(int(m.group(1)))

    for step_num in sorted(step_nums):
        prefix = f"step_{step_num:03d}"

        action_txt = ""
        action_path = os.path.join(phase_dir, f"{prefix}_action.txt")
        if os.path.exists(action_path):
            with open(action_path) as fh:
                action_txt = fh.read().strip()

        action_json = {}
        action_json_path = os.path.join(phase_dir, f"{prefix}_action.json")
        if os.path.exists(action_json_path):
            with open(action_json_path) as fh:
                try:
                    action_json = json.load(fh)
                except json.JSONDecodeError:
                    pass

        response_txt = ""
        response_path = os.path.join(phase_dir, f"{prefix}_response.txt")
        if os.path.exists(response_path):
            with open(response_path) as fh:
                response_txt = fh.read().strip()

        timestamp = 0.0
        ts_path = os.path.join(phase_dir, f"{prefix}_timestamp.txt")
        if os.path.exists(ts_path):
            with open(ts_path) as fh:
                try:
                    timestamp = float(fh.read().strip())
                except ValueError:
                    pass

        screenshot_path = ""
        png_path = os.path.join(phase_dir, f"{prefix}.png")
        if os.path.exists(png_path):
            screenshot_path = png_path

        steps.append({
            "step": step_num,
            "action_summary": action_txt,
            "action_json": action_json,
            "response": response_txt,
            "timestamp": timestamp,
            "screenshot_path": screenshot_path,
        })

    return steps


def _find_agents_and_phases(task_dir: str) -> List[Dict[str, Any]]:
    agents = []

    for entry in sorted(os.listdir(task_dir)):
        agent_dir = os.path.join(task_dir, entry)
        if not os.path.isdir(agent_dir):
            continue
        if entry in _SKIP_DIRS or entry.startswith("."):
            continue
        if entry in ("dag_plan.json", "result.txt", "task.txt"):
            continue

        phases = []
        for phase_entry in sorted(os.listdir(agent_dir)):
            phase_dir = os.path.join(agent_dir, phase_entry)
            if not os.path.isdir(phase_dir):
                continue
            if phase_entry in _SKIP_DIRS:
                continue

            steps = _read_step_files(phase_dir)
            if steps:
                phases.append({
                    "phase_id": phase_entry,
                    "phase_dir": phase_dir,
                    "steps": steps,
                })

        if phases:
            agents.append({
                "agent_id": entry,
                "agent_dir": agent_dir,
                "phases": phases,
            })

    return agents


def _compute_latencies(steps: List[Dict[str, Any]]) -> List[float]:
    latencies = []
    for i, step in enumerate(steps):
        if i == 0:
            latencies.append(0.0)
        else:
            prev_ts = steps[i - 1]["timestamp"]
            curr_ts = step["timestamp"]
            if prev_ts > 0 and curr_ts > 0:
                latencies.append(curr_ts - prev_ts)
            else:
                latencies.append(0.0)
    return latencies


def _relabel_with_llm(
    agents_data: List[Dict[str, Any]],
    task_description: str,
    bedrock: Any,
    model: str,
) -> Dict[str, str]:
    """Send raw action sequence to LLM for semantic relabeling.

    Returns a mapping from "agent_id/phase_id/step_num" -> "semantic label".
    """
    lines = [f"Task: {task_description}", ""]

    for agent in agents_data:
        agent_id = agent["agent_id"]
        for phase in agent["phases"]:
            phase_id = phase["phase_id"]
            for step in phase["steps"]:
                step_num = step["step"]
                key = f"{agent_id}/{phase_id}/{step_num}"

                action = step["action_summary"]
                reasoning = step["response"][:300] if step["response"] else "(no reasoning)"

                lines.append(f"[{key}]")
                lines.append(f"  Action: {action}")
                if step["action_json"]:
                    aj = step["action_json"]
                    if aj.get("action") in ("left_click", "right_click", "double_click") and aj.get("coordinate"):
                        lines.append(f"  Coordinate: {aj['coordinate']}")
                    if aj.get("text"):
                        lines.append(f"  Text: {aj['text'][:200]}")
                lines.append(f"  Agent reasoning: {reasoning}")
                lines.append("")

    raw_log = "\n".join(lines)

    prompt = (
        "Below is a raw action log from a computer-use agent completing a task. "
        "Each entry has a key [agent/phase/step], the raw action (click coordinates, "
        "typed text, key presses), and the agent's reasoning.\n\n"
        "Convert each action into a concise, human-readable semantic label. Rules:\n"
        "- Same number of entries, same keys\n"
        "- For clicks: describe WHAT was clicked, e.g. 'click [Save button]', "
        "'click [cell A1]', 'click [File menu]'\n"
        "- For type: summarize what was typed, e.g. 'type formula =SUM(A1:A10)', "
        "'type terminal command: ls ~/Desktop'\n"
        "- For key: describe the intent, e.g. 'press Ctrl+S to save', "
        "'press Enter to confirm', 'press Ctrl+Alt+T to open terminal'\n"
        "- For scroll: describe context, e.g. 'scroll down in document', "
        "'scroll down in terminal output'\n"
        "- For wait: 'wait for page to load', 'wait for install to finish'\n"
        "- Keep labels under 60 characters\n"
        "- Use the agent's reasoning to understand WHAT the element is\n\n"
        "Output a JSON object mapping each key to its semantic label.\n"
        f"Example: {{\"agent_0/execute/1\": \"open terminal (Ctrl+Alt+T)\"}}\n\n"
        f"Raw log:\n{raw_log}\n\n"
        "Output ONLY the JSON object."
    )

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

    try:
        content_blocks, _ = bedrock.chat(
            messages=messages,
            system="You are a precise labeling assistant. Output only valid JSON.",
            model=model,
            temperature=0.0,
            max_tokens=8192,
        )
    except Exception as e:
        logger.error("LLM relabeling failed: %s", e)
        return {}

    response_text = "".join(
        b.get("text", "") for b in content_blocks
        if isinstance(b, dict) and b.get("type") == "text"
    )

    # Parse JSON from response
    text = response_text.strip()
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        text = match.group(1)
    else:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            text = text[start:end + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse LLM labels: %s", e)
        return {}


def extract_dag(
    task_dir: str,
    relabel: bool = False,
    bedrock: Any = None,
    model: str = "claude-sonnet-4",
) -> ActionDAG:
    """Extract an action DAG from a trial's task output directory."""

    # Read task description
    task_description = ""
    task_file = os.path.join(task_dir, "task.txt")
    if os.path.exists(task_file):
        with open(task_file) as f:
            task_description = f.read().strip()

    # Read DAG plan for dependency info
    dag_plan = {}
    plan_file = os.path.join(task_dir, "dag_plan.json")
    if os.path.exists(plan_file):
        with open(plan_file) as f:
            try:
                dag_plan = json.load(f)
            except json.JSONDecodeError:
                pass

    agents_data = _find_agents_and_phases(task_dir)

    if not agents_data:
        logger.warning("No agent data found in %s", task_dir)
        return ActionDAG(
            task=task_description, task_dir=task_dir,
            total_steps=0, total_duration=0.0,
            agents=[], nodes=[], edges=[],
        )

    # Get semantic labels
    labels: Dict[str, str] = {}
    if relabel and bedrock:
        logger.info("Relabeling %d agents with LLM...", len(agents_data))
        labels = _relabel_with_llm(agents_data, task_description, bedrock, model)
        logger.info("Got %d labels from LLM", len(labels))

    # Build nodes and edges
    nodes: List[ActionNode] = []
    edges: List[Dict[str, str]] = []
    agent_summaries: List[Dict[str, Any]] = []

    all_timestamps = []
    total_steps = 0

    for agent in agents_data:
        agent_id = agent["agent_id"]
        agent_steps = 0

        for phase in agent["phases"]:
            phase_id = phase["phase_id"]
            steps = phase["steps"]
            latencies = _compute_latencies(steps)

            prev_node_id = None
            for i, step in enumerate(steps):
                step_num = step["step"]
                key = f"{agent_id}/{phase_id}/{step_num}"
                node_id = key

                semantic_label = labels.get(key, step["action_summary"])

                node = ActionNode(
                    agent_id=agent_id,
                    phase_id=phase_id,
                    step=step_num,
                    timestamp=step["timestamp"],
                    latency=latencies[i],
                    raw_action=step["action_json"],
                    raw_action_summary=step["action_summary"],
                    reasoning=step["response"][:500],
                    semantic_label=semantic_label,
                    screenshot_path=step["screenshot_path"],
                )
                nodes.append(node)

                if prev_node_id:
                    edges.append({
                        "from": prev_node_id,
                        "to": node_id,
                        "type": "sequential",
                    })
                prev_node_id = node_id

                if step["timestamp"] > 0:
                    all_timestamps.append(step["timestamp"])
                total_steps += 1
                agent_steps += 1

        agent_summaries.append({
            "agent_id": agent_id,
            "total_steps": agent_steps,
            "phases": [p["phase_id"] for p in agent["phases"]],
        })

    # Add dependency edges between agents (from dag_plan)
    dep_edges = _build_dependency_edges(dag_plan, agents_data)
    edges.extend(dep_edges)

    total_duration = 0.0
    if all_timestamps:
        total_duration = max(all_timestamps) - min(all_timestamps)

    return ActionDAG(
        task=task_description,
        task_dir=task_dir,
        total_steps=total_steps,
        total_duration=total_duration,
        agents=agent_summaries,
        nodes=nodes,
        edges=edges,
    )


def _build_dependency_edges(
    dag_plan: Dict[str, Any],
    agents_data: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Build cross-agent dependency edges from the DAG plan."""
    edges = []

    # Map agent_id -> last step key
    agent_last_step: Dict[str, str] = {}
    agent_first_step: Dict[str, str] = {}
    for agent in agents_data:
        agent_id = agent["agent_id"]
        all_steps = []
        for phase in agent["phases"]:
            for step in phase["steps"]:
                all_steps.append(f"{agent_id}/{phase['phase_id']}/{step['step']}")
        if all_steps:
            agent_first_step[agent_id] = all_steps[0]
            agent_last_step[agent_id] = all_steps[-1]

    # Parse depends_on from dag_plan
    for agent_data in dag_plan.get("agents", []):
        agent_id = agent_data.get("id", "")
        for phase in agent_data.get("phases", []):
            for dep_id in phase.get("depends_on", []):
                if dep_id in agent_last_step and agent_id in agent_first_step:
                    edges.append({
                        "from": agent_last_step[dep_id],
                        "to": agent_first_step[agent_id],
                        "type": "dependency",
                        "label": f"{dep_id} -> {agent_id}",
                    })

    return edges


def dag_to_json(dag: ActionDAG) -> Dict[str, Any]:
    return {
        "task": dag.task,
        "task_dir": dag.task_dir,
        "total_steps": dag.total_steps,
        "total_duration": round(dag.total_duration, 2),
        "agents": dag.agents,
        "nodes": [
            {
                "id": f"{n.agent_id}/{n.phase_id}/{n.step}",
                "agent_id": n.agent_id,
                "phase_id": n.phase_id,
                "step": n.step,
                "timestamp": round(n.timestamp, 3),
                "latency": round(n.latency, 2),
                "semantic_label": n.semantic_label,
                "raw_action_summary": n.raw_action_summary,
                "raw_action": n.raw_action,
                "reasoning": n.reasoning,
                "screenshot_path": n.screenshot_path,
            }
            for n in dag.nodes
        ],
        "edges": dag.edges,
    }


def dag_to_mermaid(dag: ActionDAG) -> str:
    lines = ["graph TD"]
    for node in dag.nodes:
        node_id = f"{node.agent_id}_{node.phase_id}_{node.step}".replace("-", "_")
        label = node.semantic_label.replace('"', "'")
        latency_str = f" ({node.latency:.1f}s)" if node.latency > 0 else ""
        lines.append(f'    {node_id}["{node.step}. {label}{latency_str}"]')

    for edge in dag.edges:
        from_id = edge["from"].replace("/", "_").replace("-", "_")
        to_id = edge["to"].replace("/", "_").replace("-", "_")
        if edge["type"] == "dependency":
            lines.append(f"    {from_id} -.->|dep| {to_id}")
        else:
            lines.append(f"    {from_id} --> {to_id}")

    return "\n".join(lines)


def dag_to_table(dag: ActionDAG) -> str:
    lines = []
    lines.append(f"Task: {dag.task[:100]}")
    lines.append(f"Steps: {dag.total_steps}  Duration: {dag.total_duration:.1f}s")
    lines.append(f"Agents: {len(dag.agents)}")
    lines.append("")
    lines.append(f"{'Step':>5}  {'Latency':>8}  {'Agent':<15}  {'Label'}")
    lines.append("-" * 80)

    for node in dag.nodes:
        latency = f"{node.latency:.1f}s" if node.latency > 0 else "-"
        lines.append(f"{node.step:>5}  {latency:>8}  {node.agent_id:<15}  {node.semantic_label}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Extract semantic action DAG from trial output")
    parser.add_argument("task_dir", help="Path to task output directory")
    parser.add_argument("-o", "--output", help="Output file path (default: stdout)")
    parser.add_argument("--format", choices=["json", "mermaid", "table"], default="json")
    parser.add_argument("--relabel", action="store_true", help="Use LLM to generate semantic labels")
    parser.add_argument("--model", default="claude-sonnet-4", help="Model for relabeling")
    parser.add_argument("--region", default="us-east-1", help="AWS region for Bedrock")
    args = parser.parse_args()

    bedrock = None
    if args.relabel:
        from bedrock_client import BedrockClient
        bedrock = BedrockClient(region=args.region)

    dag = extract_dag(
        task_dir=args.task_dir,
        relabel=args.relabel,
        bedrock=bedrock,
        model=args.model,
    )

    if args.format == "json":
        output = json.dumps(dag_to_json(dag), indent=2)
    elif args.format == "mermaid":
        output = dag_to_mermaid(dag)
    elif args.format == "table":
        output = dag_to_table(dag)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        logger.info("Wrote %s to %s", args.format, args.output)
    else:
        print(output)


if __name__ == "__main__":
    main()
