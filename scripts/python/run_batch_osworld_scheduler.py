"""Batch evaluation script for running the resource-aware DAG scheduler
(run_scheduler.py) across all tasks in a domain and uploading results to GitHub.

Example usage:

    # Run all collaborative tasks (skips tasks already in results.json):
    python scripts/python/run_batch_osworld_scheduler.py

    # Force re-run tasks that already have results:
    python scripts/python/run_batch_osworld_scheduler.py --force

    # Run only specific task IDs:
    python scripts/python/run_batch_osworld_scheduler.py \
        --task_ids 01b269ae-collaborative

    # Dry-run to preview commands without executing:
    python scripts/python/run_batch_osworld_scheduler.py --dry_run

    # Skip GitHub upload:
    python scripts/python/run_batch_osworld_scheduler.py --skip_github_upload
"""

import argparse
import base64
import datetime
import json
import logging
import os
import pathlib
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any

# Import trajectory generator
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
from trajectory_generator import generate_trajectory_html

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

datetime_str: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

os.makedirs("logs", exist_ok=True)

_file_handler = logging.FileHandler(
    os.path.join("logs", f"batch_scheduler-{datetime_str}.log"), encoding="utf-8"
)
_stdout_handler = logging.StreamHandler(sys.stdout)

_file_handler.setLevel(logging.DEBUG)
_stdout_handler.setLevel(logging.INFO)

_formatter = logging.Formatter(
    fmt="[%(asctime)s %(levelname)s %(module)s/%(lineno)d] %(message)s"
)
_file_handler.setFormatter(_formatter)
_stdout_handler.setFormatter(_formatter)

logger.addHandler(_file_handler)
logger.addHandler(_stdout_handler)


# ---------------------------------------------------------------------------
# GitHub upload constants
# ---------------------------------------------------------------------------

_GITHUB_API_BASE = "https://api.github.com"
_GITHUB_MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MB
_PROXY_URL = os.environ.get("HTTPS_PROXY", os.environ.get("HTTP_PROXY", ""))
_REPO_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", ".."))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch evaluation of resource-aware DAG scheduler on OSWorld tasks."
    )

    parser.add_argument(
        "--task-type", type=str, default="collaborative",
        choices=["standard", "collaborative"],
    )
    parser.add_argument("--domain", type=str, default=None)
    parser.add_argument("--task_ids", nargs="+", default=None)

    parser.add_argument("--model", type=str, default="claude-opus-4-6")
    parser.add_argument("--temperature", type=float, default=0.7)

    parser.add_argument("--provider_name", type=str, default="aws")
    parser.add_argument("--region", type=str, default="us-east-1")
    parser.add_argument("--headless", action="store_true")

    parser.add_argument(
        "--github_results_repo", type=str, default="samuellin01/memory_experiments",
    )
    parser.add_argument("--github_results_path", type=str, default="osworld")
    parser.add_argument("--config_name", type=str, default="scheduler")
    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument("--skip_github_upload", action="store_true")

    parser.add_argument("--task_timeout", type=int, default=7200)

    parser.add_argument("--skip_credential_refresh", action="store_true")
    parser.add_argument("--credential_refresh_interval", type=int, default=0)

    parser.add_argument("--force", action="store_true")

    parser.add_argument("--result_dir", type=str, default="./batch_results")
    parser.add_argument("--test_file", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# AWS credential refresh
# ---------------------------------------------------------------------------

def refresh_aws_credentials() -> None:
    cmd = [
        "cloud", "aws", "get-creds", "009160068926",
        "--role", "SSOAdmin", "--duration", "14400",
    ]
    logger.info("Refreshing AWS credentials: %s", " ".join(cmd))
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        logger.error(
            "AWS credential refresh failed: 'cloud' CLI not found. "
            "Pass --skip_credential_refresh if credentials are already set."
        )
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        logger.error("AWS credential refresh failed (rc=%d).\nstdout: %s\nstderr: %s",
                     e.returncode, e.stdout, e.stderr)
        sys.exit(1)

    refreshed: list[str] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line.startswith("export "):
            continue
        rest = line[len("export "):]
        if "=" not in rest:
            continue
        key, _, value = rest.partition("=")
        key, value = key.strip(), value.strip()
        if key:
            os.environ[key] = value
            refreshed.append(key)

    if refreshed:
        logger.info("AWS credentials refreshed: %s", ", ".join(refreshed))
    else:
        logger.warning("No 'export KEY=VALUE' lines found in credential output.")

    creds_path = os.path.join(_REPO_ROOT, "aws_credentials.json")
    try:
        creds = {
            "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID", ""),
            "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
            "AWS_SESSION_TOKEN": os.environ.get("AWS_SESSION_TOKEN", ""),
        }
        with open(creds_path, "w", encoding="utf-8") as fh:
            json.dump(creds, fh, indent=2)
        os.chmod(creds_path, 0o600)
        logger.info("AWS credentials written to %s", creds_path)
    except OSError as exc:
        logger.error("Failed to write credentials file: %s", exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _github_opener():
    if _PROXY_URL:
        return urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": _PROXY_URL, "https": _PROXY_URL})
        )
    return urllib.request.build_opener()


def _github_api_request(opener, url, headers, method="GET", body=None) -> dict:
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    with opener.open(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_existing_results(args: argparse.Namespace) -> dict:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        return {}

    opener = _github_opener()
    url = (f"{_GITHUB_API_BASE}/repos/{args.github_results_repo}/contents"
           f"/{args.github_results_path}/results.json")
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    try:
        resp_data = _github_api_request(opener, url, headers)
        data = json.loads(base64.b64decode(resp_data.get("content", "")).decode("utf-8"))
        return data.get(args.config_name, {}).get(args.task_type, {})
    except Exception as exc:
        logger.warning("Could not fetch existing results.json: %s", exc)
        return {}


def get_existing_trial_numbers(task_id: str, args: argparse.Namespace) -> list[int]:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        return []

    opener = _github_opener()
    url = (f"{_GITHUB_API_BASE}/repos/{args.github_results_repo}/contents"
           f"/{args.github_results_path}/{args.task_type}/{task_id}/{args.config_name}")
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    try:
        items = _github_api_request(opener, url, headers)
        trials = []
        for item in items:
            if item.get("type") == "dir" and item.get("name", "").startswith("trial_"):
                try:
                    trials.append(int(item["name"].replace("trial_", "")))
                except ValueError:
                    pass
        return sorted(trials)
    except Exception:
        return []


def find_next_trial_slots(existing_trials: list[int], num_new: int) -> list[int]:
    if not existing_trials:
        return list(range(1, num_new + 1))
    slots, candidate, existing_set = [], 1, set(existing_trials)
    while len(slots) < num_new:
        if candidate not in existing_set:
            slots.append(candidate)
        candidate += 1
    return slots


def discover_task_ids(task_type: str, test_file: str | None) -> list[str]:
    sys.path.insert(0, _REPO_ROOT)
    from run_orchestrator import load_osworld_tasks
    tasks = load_osworld_tasks(task_type=task_type, test_file=test_file)
    task_ids = [t.get("id") for t in tasks if t.get("id")]
    logger.info("Discovered %d task IDs for '%s'.", len(task_ids), task_type)
    return task_ids


def build_run_cmd(task_id: str, trial_base_dir: str, args: argparse.Namespace) -> list:
    cmd = [
        sys.executable,
        os.path.join(_REPO_ROOT, "run_scheduler.py"),
        "--task-name", task_id,
        "--task-type", args.task_type,
        "--provider-name", args.provider_name,
        "--region", args.region,
        "--model", args.model,
        "--output-dir", trial_base_dir,
    ]
    if args.headless:
        cmd.append("--headless")
    if args.test_file:
        cmd.extend(["--test-file", args.test_file])
    if args.domain:
        cmd.extend(["--domain", args.domain])
    return cmd


def run_subprocess(cmd: list, timeout: int, dry_run: bool, description: str) -> bool:
    logger.info("[CMD] %s: %s", description, " ".join(cmd))
    if dry_run:
        logger.info("[DRY RUN] Would execute: %s", " ".join(cmd))
        return True
    try:
        result = subprocess.run(cmd, timeout=timeout, check=False, capture_output=False)
        if result.returncode == 0:
            logger.info("[COMPLETED] %s (rc=%d)", description, result.returncode)
            return True
        logger.error("[FAILED] %s — rc=%d", description, result.returncode)
        return False
    except subprocess.TimeoutExpired:
        logger.error("[TIMEOUT] %s — exceeded %ds", description, timeout)
        return False
    except Exception as e:
        logger.error("[ERROR] %s — %s: %s", description, type(e).__name__, e)
        return False


# ---------------------------------------------------------------------------
# GitHub upload
# ---------------------------------------------------------------------------

def upload_task_results_to_github(
    local_dir: str, task_id: str, args: argparse.Namespace, trial: int = 1,
) -> None:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        logger.warning("GITHUB_TOKEN not set; skipping upload.")
        return

    if not args.dry_run and not os.path.isdir(local_dir):
        logger.warning("[UPLOAD SKIP] Directory not found: %s", local_dir)
        return

    opener = _github_opener()

    if args.dry_run:
        logger.info("[dry-run] Would upload %s → %s/%s/%s/%s/trial_%d/",
                     local_dir, args.github_results_repo, args.github_results_path,
                     args.task_type, task_id, trial)
        return

    local_path = pathlib.Path(local_dir)
    files = sorted(p for p in local_path.rglob("*") if p.is_file())
    if not files:
        logger.warning("No files to upload for task %s", task_id)
        return

    logger.info("Uploading %d file(s) for task %s", len(files), task_id)

    api_base = f"{_GITHUB_API_BASE}/repos/{args.github_results_repo}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json",
    }
    branch = "main"

    try:
        ref_data = _github_api_request(opener, f"{api_base}/git/ref/heads/{branch}", headers)
        head_sha = ref_data["object"]["sha"]
        commit_data = _github_api_request(opener, f"{api_base}/git/commits/{head_sha}", headers)
        base_tree_sha = commit_data["tree"]["sha"]
    except Exception as exc:
        logger.error("Failed to get HEAD: %s", exc)
        return

    tree_items: list[dict] = []
    for file_path in files:
        if file_path.stat().st_size > _GITHUB_MAX_FILE_BYTES:
            continue
        rel_path = file_path.relative_to(local_path)
        gh_path = (f"{args.github_results_path}/{args.task_type}/{task_id}/"
                   f"{args.config_name}/trial_{trial}/{rel_path}")
        content_b64 = base64.b64encode(file_path.read_bytes()).decode("ascii")
        blob_body = json.dumps({"content": content_b64, "encoding": "base64"}).encode()
        try:
            blob_data = _github_api_request(opener, f"{api_base}/git/blobs", headers,
                                            method="POST", body=blob_body)
            tree_items.append({"path": gh_path, "mode": "100644", "type": "blob",
                              "sha": blob_data["sha"]})
        except Exception as exc:
            logger.error("Failed to create blob for %s: %s", gh_path, exc)

    if not tree_items:
        return

    commit_message = f"Add OSWorld scheduler results: {task_id}/trial_{trial}"
    max_retries = 5
    for attempt in range(max_retries):
        if attempt > 0:
            try:
                ref_data = _github_api_request(opener, f"{api_base}/git/ref/heads/{branch}", headers)
                head_sha = ref_data["object"]["sha"]
                commit_data = _github_api_request(opener, f"{api_base}/git/commits/{head_sha}", headers)
                base_tree_sha = commit_data["tree"]["sha"]
            except Exception:
                return

        try:
            tree_data = _github_api_request(
                opener, f"{api_base}/git/trees", headers, method="POST",
                body=json.dumps({"base_tree": base_tree_sha, "tree": tree_items}).encode(),
            )
            new_commit = _github_api_request(
                opener, f"{api_base}/git/commits", headers, method="POST",
                body=json.dumps({"message": commit_message, "tree": tree_data["sha"],
                                "parents": [head_sha]}).encode(),
            )
            _github_api_request(
                opener, f"{api_base}/git/refs/heads/{branch}", headers, method="PATCH",
                body=json.dumps({"sha": new_commit["sha"]}).encode(),
            )
            logger.info("Committed %d file(s) for task %s", len(tree_items), task_id)
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 422 and attempt < max_retries - 1:
                time.sleep(1.0 * (attempt + 1))
            else:
                logger.error("Failed to push: HTTP %d", exc.code)
                return
        except Exception as exc:
            logger.error("Failed to push: %s", exc)
            return


def update_results_json_on_github(
    task_id: str, score: float | None, args: argparse.Namespace, trial: int = 1,
) -> None:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        return

    opener = _github_opener()
    url = (f"{_GITHUB_API_BASE}/repos/{args.github_results_repo}/contents"
           f"/{args.github_results_path}/results.json")
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json",
    }

    existing_data: dict = {}
    file_sha: str | None = None
    try:
        resp_data = _github_api_request(opener, url, headers)
        file_sha = resp_data.get("sha")
        existing_data = json.loads(base64.b64decode(resp_data.get("content", "")).decode())
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            logger.error("Failed to fetch results.json: HTTP %d", exc.code)
            return
    except Exception as exc:
        logger.error("Failed to fetch results.json: %s", exc)
        return

    cfg = args.config_name
    tt = args.task_type
    existing_data.setdefault(cfg, {}).setdefault(tt, {}).setdefault(task_id, {})
    if not isinstance(existing_data[cfg][tt][task_id], dict):
        existing_data[cfg][tt][task_id] = {"trial_1": existing_data[cfg][tt][task_id]}
    existing_data[cfg][tt][task_id][f"trial_{trial}"] = score

    body: dict[str, Any] = {
        "message": f"Update results.json: {task_id}/{cfg}/trial_{trial} = {score}",
        "content": base64.b64encode(json.dumps(existing_data, indent=2, sort_keys=True).encode()).decode(),
    }
    if file_sha:
        body["sha"] = file_sha

    try:
        _github_api_request(opener, url, headers, method="PUT",
                           body=json.dumps(body).encode())
        logger.info("Updated results.json: %s/%s/trial_%d = %s", cfg, task_id, trial, score)
    except Exception as exc:
        logger.error("Failed to update results.json: %s", exc)


def update_latency_json_on_github(
    task_id: str, duration: float | None, args: argparse.Namespace, trial: int = 1,
) -> None:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        return

    opener = _github_opener()
    url = (f"{_GITHUB_API_BASE}/repos/{args.github_results_repo}/contents"
           f"/{args.github_results_path}/latency_results.json")
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json",
    }

    existing_data: dict = {}
    file_sha: str | None = None
    try:
        resp_data = _github_api_request(opener, url, headers)
        file_sha = resp_data.get("sha")
        existing_data = json.loads(base64.b64decode(resp_data.get("content", "")).decode())
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            return
    except Exception:
        return

    cfg = args.config_name
    tt = args.task_type
    existing_data.setdefault(cfg, {}).setdefault(tt, {}).setdefault(task_id, {})
    if not isinstance(existing_data[cfg][tt][task_id], dict):
        existing_data[cfg][tt][task_id] = {"trial_1": existing_data[cfg][tt][task_id]}
    existing_data[cfg][tt][task_id][f"trial_{trial}"] = duration

    body: dict[str, Any] = {
        "message": f"Update latency_results.json: {task_id}/{cfg}/trial_{trial} = {duration}s",
        "content": base64.b64encode(json.dumps(existing_data, indent=2, sort_keys=True).encode()).decode(),
    }
    if file_sha:
        body["sha"] = file_sha

    try:
        _github_api_request(opener, url, headers, method="PUT",
                           body=json.dumps(body).encode())
        logger.info("Updated latency_results.json: %s/trial_%d = %ss", task_id, trial, duration)
    except Exception as exc:
        logger.error("Failed to update latency_results.json: %s", exc)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    last_credential_refresh: float = 0.0
    if args.skip_credential_refresh:
        logger.info("Skipping AWS credential refresh.")
    else:
        refresh_aws_credentials()
        last_credential_refresh = time.monotonic()

    if args.task_ids:
        task_ids = args.task_ids
        logger.info("Using %d task IDs from --task_ids.", len(task_ids))
    else:
        task_ids = discover_task_ids(args.task_type, args.test_file)

    if not task_ids:
        logger.error("No task IDs to process.")
        sys.exit(1)

    existing_results: dict = {}
    if not args.force and not args.skip_github_upload:
        existing_results = fetch_existing_results(args)
        if existing_results:
            logger.info("Found %d existing results. Use --force to re-run.", len(existing_results))

    logger.info("Starting batch evaluation: %d tasks.", len(task_ids))

    results: dict[str, dict] = {}
    skipped: list[str] = []

    for task_idx, task_id in enumerate(task_ids, start=1):
        logger.info("=== Task %d/%d: %s ===", task_idx, len(task_ids), task_id)

        if task_id in existing_results:
            existing = existing_results[task_id]
            if isinstance(existing, dict) and any(v is not None for v in existing.values()):
                logger.info("SKIP %s — already has results. Use --force to re-run.", task_id)
                skipped.append(task_id)
                continue

        results[task_id] = {"trials": []}

        existing_trials = []
        if not args.skip_github_upload:
            existing_trials = get_existing_trial_numbers(task_id, args)

        trial_slots = find_next_trial_slots(existing_trials, args.num_trials)
        logger.info("Task %s: existing=%s, will run=%s", task_id, existing_trials, trial_slots)

        for trial_num, trial_idx in enumerate(trial_slots, start=1):
            logger.info("--- Trial %d/%d (trial_%d) for %s ---",
                        trial_num, args.num_trials, trial_idx, task_id)

            if not args.skip_credential_refresh:
                if (args.credential_refresh_interval <= 0
                    or (time.monotonic() - last_credential_refresh) >= args.credential_refresh_interval):
                    refresh_aws_credentials()
                    last_credential_refresh = time.monotonic()

            trial_base_dir = os.path.join(os.path.abspath(args.result_dir), f"trial_{trial_idx}")
            os.makedirs(trial_base_dir, exist_ok=True)

            cache_dir = f"cache/{task_id}"
            if os.path.exists(cache_dir) and not args.dry_run:
                shutil.rmtree(cache_dir)
                logger.info("Cleared cache for %s (trial %d)", task_id, trial_idx)

            run_cmd = build_run_cmd(task_id, trial_base_dir, args)
            run_ok = run_subprocess(
                run_cmd, timeout=args.task_timeout, dry_run=args.dry_run,
                description=f"run task {task_id} trial {trial_idx}",
            )

            trial_output_dir = os.path.join(trial_base_dir, f"task_{task_id}")

            trial_score = None
            trial_duration = None
            trial_instruction = None
            trial_error = None

            if not args.dry_run and os.path.isdir(trial_output_dir):
                result_json = os.path.join(trial_output_dir, "result.json")
                if os.path.isfile(result_json):
                    try:
                        with open(result_json) as fh:
                            result_data = json.load(fh)
                            trial_duration = result_data.get("duration")
                            trial_instruction = result_data.get("instruction")

                            if result_data.get("status") == "failed":
                                agent_result = result_data.get("result", {})
                                if isinstance(agent_result, dict) and "error" in agent_result:
                                    error_msg = agent_result["error"]
                                    if "tool_use" in error_msg and "tool_result" in error_msg:
                                        trial_error = "tool_result_missing"
                                    else:
                                        trial_error = "agent_failed"
                    except (json.JSONDecodeError, OSError):
                        pass

                if trial_error is None:
                    result_txt = os.path.join(trial_output_dir, "result.txt")
                    if os.path.isfile(result_txt):
                        try:
                            with open(result_txt) as fh:
                                trial_score = float(fh.read().strip())
                        except (ValueError, OSError):
                            pass

                if trial_instruction:
                    with open(os.path.join(trial_output_dir, "task.txt"), "w") as fh:
                        fh.write(trial_instruction + "\n")

                if os.path.isfile(result_json):
                    try:
                        with open(result_json) as fh:
                            rd = json.load(fh)
                            tu = rd.get("token_usage", {})
                            with open(os.path.join(trial_output_dir, "token_usage.json"), "w") as fh2:
                                json.dump(tu, fh2, indent=2)
                    except (json.JSONDecodeError, OSError):
                        pass

            results[task_id]["trials"].append({
                "trial": trial_idx, "run": run_ok, "score": trial_score,
                "duration": trial_duration, "error": trial_error,
            })

            if not run_ok:
                logger.warning("Task %s trial %d FAILED — skipping upload.", task_id, trial_idx)
                continue

            if not args.dry_run and os.path.isdir(trial_output_dir):
                try:
                    generate_trajectory_html(
                        local_dir=trial_output_dir,
                        task_id=task_id,
                        github_repo=args.github_results_repo,
                        github_path=args.github_results_path,
                        task_type=args.task_type,
                        config_name=args.config_name,
                        trial=trial_idx,
                    )
                except Exception as exc:
                    logger.warning("Failed to generate trajectory HTML: %s", exc)

            if not args.skip_github_upload:
                upload_task_results_to_github(trial_output_dir, task_id, args, trial=trial_idx)
                update_results_json_on_github(task_id, trial_score, args, trial=trial_idx)
                update_latency_json_on_github(task_id, trial_duration, args, trial=trial_idx)

            if trial_error:
                logger.warning("Task %s trial %d ERROR: %s", task_id, trial_idx, trial_error)
            else:
                score_str = f" score={trial_score}" if trial_score is not None else ""
                logger.info("Task %s trial %d COMPLETED.%s", task_id, trial_idx, score_str)

    # Summary
    logger.info("=" * 60)
    logger.info("BATCH EVALUATION SUMMARY (scheduler)")
    logger.info("=" * 60)

    run_success = [t for t, v in results.items() if any(tr["run"] for tr in v["trials"])]
    run_failed = [t for t, v in results.items() if v["trials"] and not any(tr["run"] for tr in v["trials"])]
    all_scores = []
    error_trials = []
    for task_id, v in results.items():
        for trial in v["trials"]:
            if trial.get("error"):
                error_trials.append((task_id, trial["trial"], trial["error"]))
            if trial["score"] is not None:
                all_scores.append(trial["score"])

    if skipped:
        logger.info("Tasks skipped: %d (already have results)", len(skipped))
    logger.info("Tasks run: %d succeeded, %d failed (of %d total)",
                len(run_success), len(run_failed), len(results))

    if error_trials:
        logger.warning("Trials with errors: %d", len(error_trials))
        for tid, tnum, err in error_trials:
            logger.warning("  - %s trial_%d: %s", tid, tnum, err)

    pass_count = 0
    if all_scores:
        avg_score = sum(all_scores) / len(all_scores)
        pass_count = sum(1 for s in all_scores if s > 0)
        logger.info("Scores: %d evaluated, %d passed, avg=%.4f",
                     len(all_scores), pass_count, avg_score)

    if run_failed:
        logger.info("FAILED RUNS:")
        for tid in run_failed:
            logger.info("  - %s", tid)

    summary = {
        "timestamp": datetime.datetime.now().isoformat(),
        "config_name": args.config_name,
        "task_type": args.task_type,
        "model": args.model,
        "total_tasks": len(results),
        "tasks_succeeded": len(run_success),
        "tasks_failed": len(run_failed),
        "tasks_evaluated": len(all_scores),
        "tasks_passed": pass_count,
        "average_score": round(sum(all_scores) / len(all_scores), 4) if all_scores else None,
        "results": results,
    }
    summary_path = os.path.join(os.path.abspath(args.result_dir), "batch_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Batch summary saved to %s", summary_path)

    logger.info("=" * 60)

    if run_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
