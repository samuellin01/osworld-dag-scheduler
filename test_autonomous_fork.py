"""Test autonomous fork discovery - agent decides when to fork.

This test gives a task with parallelizable work but does NOT tell the agent
to use fork_subtask. We want to see if the agent:
1. Recognizes the opportunity for parallelism
2. Decides forking is beneficial
3. Uses fork_subtask on its own

Usage:
    python test_autonomous_fork.py --provider-name aws --region us-east-1 --headless
"""

import argparse
import logging
import os
import threading
import time
from typing import Optional

import requests

from agent_runtime import AgentRuntime
from bedrock_client import BedrockClient
from fork_agent import run_fork_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def agent_monitor_thread(
    runtime: AgentRuntime,
    vm_ip: str,
    server_port: int,
    bedrock: BedrockClient,
    model: str,
    output_dir: str,
):
    """Monitor for new agents and spawn threads to run them."""
    logger.info("[Monitor] Started agent monitor thread")

    running_threads = {}  # agent_id -> thread

    while True:
        time.sleep(0.5)  # Check every 500ms

        # Get all agents
        all_agents = runtime.get_all_agents()

        for agent_id, status_dict in all_agents.items():
            status = status_dict["status"]  # This is a string, not enum

            # Skip if already running or completed
            if agent_id in running_threads:
                continue
            if status in ("completed", "failed", "killed"):
                continue

            # Skip if not ready (still initializing)
            if status != "running":
                continue

            # New agent that needs to run
            logger.info(f"[Monitor] Detected new agent {agent_id}, spawning thread")

            agent_output = os.path.join(output_dir, agent_id)
            os.makedirs(agent_output, exist_ok=True)

            # Get agent details
            agent_info = runtime.get_agent_status(agent_id)
            task = agent_info.get("subtask", "")
            parent_id = agent_info.get("parent_id")
            context_summary = agent_info.get("context_summary")

            # Spawn thread to run agent
            thread = threading.Thread(
                target=run_agent_thread,
                args=(
                    agent_id,
                    runtime,
                    vm_ip,
                    server_port,
                    bedrock,
                    model,
                    task,
                    context_summary,
                    agent_output,
                ),
                daemon=True,
            )
            thread.start()
            running_threads[agent_id] = thread

        # Clean up completed threads
        completed = [
            aid for aid, thread in running_threads.items()
            if not thread.is_alive()
        ]
        for aid in completed:
            del running_threads[aid]

        # Check if root is done
        root_status = runtime.get_agent_status("root")
        if root_status and root_status["status"] in ("completed", "failed"):
            logger.info("[Monitor] Root agent finished, stopping monitor")
            break


def run_agent_thread(
    agent_id: str,
    runtime: AgentRuntime,
    vm_ip: str,
    server_port: int,
    bedrock: BedrockClient,
    model: str,
    task: str,
    parent_context: Optional[str],
    output_dir: str,
):
    """Run an agent in a thread."""
    try:
        logger.info(f"[Thread-{agent_id}] Starting agent")
        result = run_fork_agent(
            agent_id=agent_id,
            runtime=runtime,
            vm_ip=vm_ip,
            server_port=server_port,
            bedrock=bedrock,
            model=model,
            task=task,
            parent_context=parent_context,
            max_steps=25,
            temperature=0.7,
            output_dir=output_dir,
        )
        logger.info(f"[Thread-{agent_id}] Finished: {result['status']}")
    except Exception as e:
        logger.error(f"[Thread-{agent_id}] Crashed: {e}", exc_info=True)
        runtime.fail_agent(agent_id, error=str(e))


def main():
    parser = argparse.ArgumentParser(description="Test autonomous fork discovery")
    parser.add_argument("--provider-name", default="aws", help="Provider")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--headless", action="store_true", help="Run headless")
    parser.add_argument("--model", default="claude-opus-4-6", help="Model name")
    parser.add_argument("--output-dir", default="test_autonomous_fork_output", help="Output dir")
    args = parser.parse_args()

    password = "osworld-public-evaluation"
    os.makedirs(args.output_dir, exist_ok=True)

    # Boot VM
    logger.info("Booting VM...")
    from desktop_env.desktop_env import DesktopEnv
    from desktop_env.providers.aws.manager import IMAGE_ID_MAP

    screen_size = (1920, 1080)
    region_map = IMAGE_ID_MAP[args.region]
    ami_id = region_map.get(screen_size, region_map.get((1920, 1080)))

    env = DesktopEnv(
        provider_name=args.provider_name,
        action_space="pyautogui",
        screen_size=screen_size,
        headless=args.headless,
        os_type="Ubuntu",
        client_password=password,
        region=args.region,
        snapshot_name=ami_id,
    )
    env.reset()

    vm_ip = env.vm_ip
    port = env.server_port
    logger.info(f"VM ready at {vm_ip}:{port}")

    exec_url = f"http://{vm_ip}:{port}/setup/execute"

    def vm_exec(cmd: str, timeout: int = 120) -> Optional[dict]:
        try:
            r = requests.post(
                exec_url,
                json={"command": cmd, "shell": True},
                timeout=timeout,
            )
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            logger.error(f"vm_exec failed: {e}")
        return None

    # Wait for VM
    logger.info("Waiting for VM server...")
    for attempt in range(30):
        try:
            r = requests.post(
                exec_url,
                json={"command": "echo ready", "shell": True},
                timeout=10,
            )
            if r.status_code == 200 and r.json().get("returncode") == 0:
                logger.info(f"VM server healthy (waited {attempt * 2}s)")
                break
        except Exception:
            pass
        time.sleep(2)

    # Initialize runtime
    logger.info("\n" + "=" * 60)
    logger.info("Initializing Agent Runtime")
    logger.info("=" * 60)

    runtime = AgentRuntime(vm_exec=vm_exec, num_displays=8, password=password)
    success = runtime.initialize()

    if not success:
        logger.error("Runtime initialization failed!")
        return

    # Initialize Bedrock
    bedrock = BedrockClient(region=args.region, log_dir=args.output_dir)

    # Task WITHOUT explicit fork instruction
    logger.info("\n" + "=" * 60)
    logger.info("Test: Autonomous Fork Discovery")
    logger.info("=" * 60)

    task = (
        "Search for 'Python programming' and 'JavaScript programming' on Google. "
        "For each search term, take a screenshot of the results page. "
        "Report both results when complete."
    )

    logger.info(f"\nTask (no mention of fork or parallel):\n{task}\n")
    logger.info("Will the agent discover it should fork? Let's see...\n")

    # Spawn root agent on display :0
    root_id = runtime.spawn_root_agent(task=task, display_num=0)
    logger.info(f"[{root_id}] Task: {task}")

    # Start agent monitor thread
    monitor_thread = threading.Thread(
        target=agent_monitor_thread,
        args=(runtime, vm_ip, port, bedrock, args.model, args.output_dir),
        daemon=True,
    )
    monitor_thread.start()

    # Wait for root to complete
    logger.info("\n[Main] Waiting for root agent to complete...")

    while True:
        time.sleep(2)
        root_status = runtime.get_agent_status(root_id)
        if not root_status:
            logger.error("Root agent disappeared!")
            break

        status = root_status["status"]  # This is a string
        if status in ("completed", "failed", "killed"):
            logger.info(f"[Main] Root agent finished with status: {status}")
            break

    # Wait a bit for monitor thread to finish
    time.sleep(2)

    # Final status
    logger.info("\n" + "=" * 60)
    logger.info("Test Complete")
    logger.info("=" * 60)

    all_agents = runtime.get_all_agents()

    # Check if agent forked
    num_agents = len(all_agents)
    did_fork = num_agents > 1

    logger.info(f"\n{'✓' if did_fork else '✗'} Agent forked: {did_fork}")
    logger.info(f"  Total agents: {num_agents}")

    for agent_id, info in all_agents.items():
        logger.info(f"  {agent_id}: {info['status']} (display :{info['display_num']})")

    root_info = runtime.get_agent_status(root_id)
    if root_info and root_info.get("result"):
        result = root_info["result"]
        logger.info(f"\nRoot result:")
        logger.info(f"  Status: {result.get('status')}")
        logger.info(f"  Steps: {result.get('steps_used')}")
        logger.info(f"  Duration: {result.get('duration', 0):.1f}s")
        logger.info(f"  Summary: {result.get('summary', '')[:300]}")

    # Token usage
    usage = bedrock.get_token_usage()
    logger.info(f"\nToken usage:")
    logger.info(f"  Input tokens: {usage['total_input_tokens']}")
    logger.info(f"  Output tokens: {usage['total_output_tokens']}")
    logger.info(f"  Cost: ${usage['total_cost_usd']:.4f}")
    logger.info(f"  LLM calls: {usage['num_llm_calls']}")

    logger.info(f"\nScreenshots and logs saved to: {args.output_dir}/")

    # Analysis
    logger.info("\n" + "=" * 60)
    logger.info("Analysis")
    logger.info("=" * 60)

    if did_fork:
        logger.info("✓ SUCCESS: Agent autonomously discovered it should fork!")
        logger.info(f"  Agent forked {num_agents - 1} children without being told")
        logger.info(f"  This demonstrates autonomous parallelism discovery")
    else:
        logger.info("✗ Agent did NOT fork")
        logger.info("  Agent completed task sequentially")
        logger.info("  Possible reasons:")
        logger.info("    - Did not recognize parallelization opportunity")
        logger.info("    - Thought sequential was simpler/faster")
        logger.info("    - Did not understand fork_subtask tool")

    # Shutdown
    runtime.shutdown()

    logger.info("\n" + "=" * 60)
    logger.info("✓ Test complete!")
    logger.info("=" * 60)

    logger.info(f"\nVM still running. VNC: http://{vm_ip}:5910/vnc.html")
    logger.info("Press Ctrl+C to exit")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\nExiting...")


if __name__ == "__main__":
    main()
