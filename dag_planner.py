"""LLM-based planner for agent-level DAG decomposition.

Decomposes a task into agents with phased step lists:
  - Each agent gets its own display and runs independently
  - Phases execute sequentially within an agent (display state carries over)
  - Cross-agent data flows through agent-level dependencies (depends_on)
  - Agents start ALL independent work immediately, blocking only at dependency points
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from dag_core import AgentPlan, DAGPlan, Phase

logger = logging.getLogger(__name__)


_PLANNER_SYSTEM_PROMPT = """\
You are a task decomposition planner for a multi-agent computer-use system.

The system has multiple virtual displays. Each agent gets its own display and \
runs independently. Your job is to decompose a task into agents that work in \
parallel, with dependencies when one agent needs another's output.

**Key insight**: Separate each agent's work into setup (no cross-agent deps) \
and execute (needs another agent's output). Agents start ALL their independent \
work immediately.

Example:
  Agent A: [open chrome] → [search X] → [extract info] → completes with results
  Agent B: [open google doc] → [navigate to section] → depends_on A → [paste data]

Agent B's first two actions run in parallel with Agent A. B only blocks at \
the depends_on point.

Output a JSON object with this structure:
{{
  "agents": [
    {{
      "id": "short_name",
      "task": "High-level description of this agent's role",
      "setup": [setup actions for the display BEFORE the agent starts],
      "phases": [
        {{
          "id": "phase_name",
          "task": "What to do in this phase. Include ALL info the agent needs.",
          "depends_on": ["agent_id"]
        }}
      ]
    }}
  ]
}}

**Setup types** (prepare the display before the agent starts):
- {{"type": "chrome_open_tabs", "parameters": {{"urls_to_open": ["https://..."]}}}}
- {{"type": "launch", "parameters": {{"command": ["app", "arg1", ...]}}}}
- {{"type": "sleep", "parameters": {{"seconds": 3}}}}

**Dependency rules**:
- depends_on lists agent IDs whose results this phase needs before starting
- A phase with depends_on=[] starts immediately (or after the previous phase)
- When a phase starts, it receives the results from all dependency agents
- Each agent should include a detailed summary of its results when it completes

**Phase rules**:
- Phases within an agent execute sequentially on the SAME display
- The display state (windows, tabs, cursor) carries over between phases
- Put independent work BEFORE dependency points to maximize parallelism

**Guidelines**:
- Google Workspace: multiple agents CAN open the same Doc/Sheet/Slides URL \
on different displays and edit collaboratively in real-time.
- Maximize parallelism where work is truly independent.
- Don't over-split: if a single agent can handle everything in ~30 actions, \
return a single agent with one phase.
- Each agent needs setup. Phases within an agent do NOT need setup \
(display carries over).
- When an agent produces data another needs, the consumer declares \
depends_on that agent. The producer's completion summary is the data.

Output ONLY the JSON object, no other text."""


def plan_dag(
    task_description: str,
    bedrock: Any,
    model: str,
    context: Optional[str] = None,
    temperature: float = 0.3,
) -> Dict[str, Any]:
    """Produce a DAG plan with agents, phases, and dependencies."""
    user_msg = f"Task: {task_description}"
    if context:
        user_msg += f"\n\nAdditional context:\n{context}"

    messages = [{"role": "user", "content": [{"type": "text", "text": user_msg}]}]

    content_blocks, _ = bedrock.chat(
        messages=messages,
        system=_PLANNER_SYSTEM_PROMPT,
        model=model,
        temperature=temperature,
        max_tokens=4096,
    )

    response_text = "".join(
        b.get("text", "") for b in content_blocks
        if isinstance(b, dict) and b.get("type") == "text"
    )

    plan = _parse_plan_json(response_text)
    if not plan or "agents" not in plan:
        logger.warning("Planner returned invalid plan, single-agent fallback")
        plan = _single_agent_fallback(task_description)

    _validate_plan(plan)

    logger.info("DAG plan: %d agents", len(plan["agents"]))
    for agent in plan["agents"]:
        logger.info("  %s: %s (%d phases)",
                     agent["id"], agent["task"][:60], len(agent["phases"]))
        for phase in agent["phases"]:
            logger.info("    %s: depends_on=%s",
                         phase["id"], phase.get("depends_on", []))
    return plan


def convert_plan_to_dag(plan: Dict[str, Any], root_task: str) -> DAGPlan:
    """Convert planner output into a DAGPlan with agents and dependencies."""
    agents = {}

    for agent_data in plan["agents"]:
        agent_id = agent_data["id"]
        phases = []
        for phase_data in agent_data.get("phases", []):
            phase = Phase(
                id=phase_data["id"],
                task=phase_data["task"],
                depends_on=phase_data.get("depends_on", []),
            )
            phases.append(phase)

        agent = AgentPlan(
            id=agent_id,
            task=agent_data["task"],
            phases=phases,
            setup=agent_data.get("setup", []),
        )
        agents[agent_id] = agent

    return DAGPlan(agents=agents, root_task=root_task)


def _single_agent_fallback(task_description: str) -> Dict[str, Any]:
    return {
        "agents": [{
            "id": "agent_0",
            "task": task_description,
            "setup": [],
            "phases": [{
                "id": "execute",
                "task": task_description,
                "depends_on": [],
            }],
        }]
    }


def _parse_plan_json(text: str) -> Optional[Dict[str, Any]]:
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
        logger.debug("Raw text: %s", text[:500])
    return None


def _validate_plan(plan: Dict[str, Any]):
    if "agents" not in plan:
        plan["agents"] = []
        return

    agent_ids = set()
    for i, agent in enumerate(plan["agents"]):
        if "id" not in agent:
            agent["id"] = f"agent_{i}"
        agent_ids.add(agent["id"])
        if "task" not in agent:
            agent["task"] = "Unknown task"
        if "setup" not in agent:
            agent["setup"] = []
        if "phases" not in agent:
            agent["phases"] = [{
                "id": "execute",
                "task": agent["task"],
                "depends_on": [],
            }]

        for j, phase in enumerate(agent["phases"]):
            if "id" not in phase:
                phase["id"] = f"phase_{j}"
            if "task" not in phase:
                phase["task"] = "Unknown phase"
            if "depends_on" not in phase:
                # Convert old awaits/signals format to depends_on
                phase["depends_on"] = phase.get("awaits", [])

    # Validate dependency references
    for agent in plan["agents"]:
        for phase in agent["phases"]:
            phase["depends_on"] = [d for d in phase["depends_on"] if d in agent_ids and d != agent["id"]]
