"""Test fork-based CUA agent with real LLM calls.

This test runs a real agent that can:
- Take screenshots
- Call Claude via Bedrock
- Execute computer use actions
- Fork children for parallel work
- Send/receive messages

Usage:
    python test_fork_agent.py --provider-name aws --region us-east-1
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
        result = run_fork_agent(
            agent_id=agent_id,
            runtime=runtime,
            vm_ip=vm_ip,
            server_port=server_port,
            bedrock=bedrock,
            model=model,
            task=task,
            parent_context=parent_context,
            max_steps=15,
            temperature=0.7,
            output_dir=output_dir,
        )
        logger.info(f"Agent {agent_id} finished: {result['status']}")
    except Exception as e:
        logger.error(f"Agent {agent_id} crashed: {e}", exc_info=True)
        runtime.fail_agent(agent_id, error=str(e))


def main():
    parser = argparse.ArgumentParser(description="Test fork-based CUA agent")
    parser.add_argument("--provider-name", default="aws", help="Provider")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--headless", action="store_true", help="Run headless")
    parser.add_argument("--model", default="claude-opus-4-6", help="Model name")
    parser.add_argument("--output-dir", default="test_fork_agent_output", help="Output dir")
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

    # Test: Simple task on a single display
    logger.info("\n" + "=" * 60)
    logger.info("Test: Single Agent Task")
    logger.info("=" * 60)

    task = "Open Google Chrome and search for 'hello world'. Take a screenshot showing the search results."

    # Spawn root agent
    root_id = runtime.spawn_root_agent(task=task, display_num=2)  # Use display :2 (easier than :0)

    # Run agent in main thread for this simple test
    root_output = os.path.join(args.output_dir, "root")
    os.makedirs(root_output, exist_ok=True)

    logger.info(f"\n[{root_id}] Starting task: {task}")

    result = run_fork_agent(
        agent_id=root_id,
        runtime=runtime,
        vm_ip=vm_ip,
        server_port=port,
        bedrock=bedrock,
        model=args.model,
        task=task,
        parent_context=None,
        max_steps=10,
        temperature=0.7,
        output_dir=root_output,
    )

    logger.info("\n" + "=" * 60)
    logger.info("Test Complete")
    logger.info("=" * 60)
    logger.info(f"Status: {result['status']}")
    logger.info(f"Steps used: {result['steps_used']}")
    logger.info(f"Duration: {result.get('duration', 0):.1f}s")
    logger.info(f"Summary: {result['summary'][:200]}")

    # Token usage
    usage = bedrock.get_token_usage()
    logger.info(f"\nToken usage:")
    logger.info(f"  Input tokens: {usage['total_input_tokens']}")
    logger.info(f"  Output tokens: {usage['total_output_tokens']}")
    logger.info(f"  Cost: ${usage['total_cost_usd']:.4f}")
    logger.info(f"  LLM calls: {usage['num_llm_calls']}")

    logger.info(f"\nScreenshots and logs saved to: {args.output_dir}/")

    # Shutdown
    runtime.shutdown()

    logger.info("\n" + "=" * 60)
    logger.info("✓ Test passed!")
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
