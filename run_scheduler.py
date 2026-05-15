"""Run resource-aware DAG scheduler on OSWorld benchmark tasks.

Third method: uses fine-grained resource conflict detection to maximize
parallelism.  Compare against baseline (run_baseline_task.py) and
orchestrator (run_orchestrator.py).

Usage:
    python run_scheduler.py --task-id 0 --provider-name aws --region us-east-1 --headless
    python run_scheduler.py --task-name 030eeff7-collaborative --task-type collaborative \
        --provider-name aws --region us-east-1 --headless
"""

import argparse
import json
import logging
import os
import time
from typing import Any, Dict, Optional

import requests

from bedrock_client import BedrockClient
from display_pool import DisplayPool
from scheduler import Scheduler
from run_orchestrator import (
    _process_google_workspace_config,
    aggregate_token_usage,
    load_osworld_tasks,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def run_single_task(task_data, args, output_base):
    """Run a single benchmark task using the resource-aware scheduler."""
    task_data = _process_google_workspace_config(task_data)

    task_id = task_data.get("id", "unknown")
    instruction = task_data.get("instruction", "")

    logger.info("\n" + "=" * 80)
    logger.info(f"Task {task_id}: {instruction[:100]}")
    logger.info("=" * 80)

    output_dir = os.path.join(output_base, f"task_{task_id}")
    os.makedirs(output_dir, exist_ok=True)

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
        client_password="osworld-public-evaluation",
        region=args.region,
        snapshot_name=ami_id,
    )
    env.reset(task_config=task_data)

    vm_ip = env.vm_ip
    port = env.server_port

    def vm_exec(cmd: str, timeout: int = 120) -> Optional[dict]:
        try:
            r = requests.post(
                f"http://{vm_ip}:{port}/setup/execute",
                json={"command": cmd, "shell": True},
                timeout=timeout,
            )
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            logger.error(f"vm_exec failed: {e}")
        return None

    logger.info("Letting environment settle (5s)...")
    time.sleep(5)

    logger.info("Waiting for VM...")
    for _ in range(30):
        try:
            r = requests.post(
                f"http://{vm_ip}:{port}/setup/execute",
                json={"command": "echo ready", "shell": True},
                timeout=10,
            )
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(2)

    display_pool = DisplayPool(
        vm_exec=vm_exec,
        num_displays=8,
        password="osworld-public-evaluation",
        include_primary=True,
    )
    display_pool.initialize()

    from xvfb_display import XvfbDisplay
    primary_display = XvfbDisplay(vm_ip, port, 0)
    initial_screenshot = primary_display.screenshot()

    def bedrock_factory(log_dir: str, agent_id: str) -> BedrockClient:
        return BedrockClient(region=args.region, log_dir=log_dir, agent_id=agent_id)

    scheduler = Scheduler(
        display_pool=display_pool,
        vm_exec=vm_exec,
        bedrock_factory=bedrock_factory,
        model=args.model,
        vm_ip=vm_ip,
        server_port=port,
        output_dir=output_dir,
        task_timeout=1200.0,
        password="osworld-public-evaluation",
        root_task=instruction,
        initial_screenshot=initial_screenshot,
    )

    scheduler_result = scheduler.run()

    all_bedrock_clients = scheduler.get_all_bedrock_clients()
    aggregated_token_usage = aggregate_token_usage(all_bedrock_clients)

    result = {
        "task_id": task_id,
        "instruction": instruction,
        "status": scheduler_result.get("status", "unknown"),
        "num_agents": len(all_bedrock_clients),
        "duration": scheduler_result.get("duration", 0),
        "agents": scheduler_result.get("agents", {}),
        "token_usage": aggregated_token_usage,
    }

    with open(os.path.join(output_dir, "token_usage.json"), "w") as f:
        json.dump(aggregated_token_usage, f, indent=2)

    score = None
    if hasattr(env, "evaluator") and env.evaluator:
        logger.info("Waiting 20s before evaluation...")
        time.sleep(20)
        score = env.evaluate()
        logger.info(f"Benchmark score: {score:.4f}")

        result["score"] = score

        if hasattr(env, "last_eval_details") and env.last_eval_details:
            result["eval_details"] = env.last_eval_details
            with open(os.path.join(output_dir, "eval_details.json"), "w") as f:
                json.dump(env.last_eval_details, f, indent=2)

        with open(os.path.join(output_dir, "result.txt"), "w") as f:
            f.write(f"{score}\n")
    else:
        logger.info("No evaluator configured - skipping evaluation")

    with open(os.path.join(output_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    display_pool.cleanup()

    logger.info(f"\nTask {task_id} completed: {result['status']}")
    logger.info(f"  Agents: {result['num_agents']}")
    logger.info(f"  Duration: {result['duration']:.1f}s")
    logger.info(f"  Cost: ${result['token_usage']['total_cost_usd']:.4f}")
    if score is not None:
        logger.info(f"  Score: {score:.4f}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run OSWorld benchmark with resource-aware DAG scheduler"
    )

    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument("--task-id", type=int, help="Run specific task by index")
    task_group.add_argument("--task-name", type=str, help="Run specific task by UUID")

    parser.add_argument("--num-tasks", type=int, default=1)
    parser.add_argument("--domain", type=str, default=None)
    parser.add_argument("--task-type", default="collaborative",
                        choices=["standard", "collaborative"])
    parser.add_argument("--test-file", type=str, default=None)
    parser.add_argument("--provider-name", default="aws")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--model", default="claude-opus-4-6")
    parser.add_argument("--output-dir", default="benchmark_results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tasks = load_osworld_tasks(task_type=args.task_type, test_file=args.test_file)
    if not tasks:
        logger.error("No tasks found!")
        return

    logger.info(f"Loaded {len(tasks)} tasks")

    if args.domain and args.domain != "all":
        original_count = len(tasks)
        tasks = [t for t in tasks
                 if t.get("snapshot") == args.domain
                 or args.domain in t.get("related_apps", [])]
        logger.info(f"Filtered to {len(tasks)} tasks in domain '{args.domain}' (from {original_count})")
        if not tasks:
            logger.error(f"No tasks found for domain '{args.domain}'")
            return

    if args.task_name:
        matching_tasks = [t for t in tasks if t.get("id") == args.task_name]
        if not matching_tasks:
            logger.error(f"Task '{args.task_name}' not found")
            return
        tasks_to_run = matching_tasks
    elif args.task_id is not None:
        if args.task_id >= len(tasks):
            logger.error(f"Task index {args.task_id} out of range (0-{len(tasks)-1})")
            return
        tasks_to_run = [tasks[args.task_id]]
    else:
        tasks_to_run = tasks[:args.num_tasks]

    results = []
    for i, task in enumerate(tasks_to_run):
        logger.info(f"\n{'='*80}\nRunning task {i+1}/{len(tasks_to_run)}\n{'='*80}")
        result = run_single_task(task, args, args.output_dir)
        results.append(result)

    logger.info("\n" + "=" * 80)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 80)

    total_cost = sum(r["token_usage"]["total_cost_usd"] for r in results)
    num_completed = sum(1 for r in results if r.get("status") == "DONE")

    logger.info(f"Tasks run: {len(results)}")
    logger.info(f"Tasks completed: {num_completed}/{len(results)}")
    logger.info(f"Total cost: ${total_cost:.4f}")

    summary = {
        "num_tasks": len(results),
        "num_completed": num_completed,
        "total_cost": total_cost,
        "results": results,
    }

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
