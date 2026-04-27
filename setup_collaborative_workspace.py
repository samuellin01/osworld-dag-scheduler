"""One-time setup: Create Google Sheets for all collaborative tasks.

This script creates 27 Google Sheets (one per collaborative task) and saves
their URLs to collaborative_sheet_urls.json. After running this once, you can
run collaborative tasks infinitely without creating new sheets.

Usage:
    python setup_collaborative_sheets.py

The script will:
1. Load collaborative task metadata
2. Create a Google Sheet for each task from its template
3. Set permissions to "anyone with link can edit"
4. Save URLs to collaborative_sheet_urls.json

You'll need:
- oauth_client_secret.json (Google OAuth credentials)
- Internet connection
"""

import argparse
import json
import logging
import os
import time

from google_sheets_oauth import create_sheet_from_template_oauth

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Pre-create Google Sheets for collaborative tasks")
    parser.add_argument(
        "--config-file",
        default="evaluation_examples/collaborative_task_configs.json",
        help="Path to collaborative task configs"
    )
    parser.add_argument(
        "--output-file",
        default="collaborative_sheet_urls.json",
        help="Where to save sheet URLs"
    )
    parser.add_argument(
        "--client-secret",
        default="oauth_client_secret.json",
        help="Path to OAuth client secret"
    )
    parser.add_argument(
        "--token",
        default="oauth_token.pickle",
        help="Path to OAuth token"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay between sheet creations (seconds) to avoid rate limits"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recreate all sheets even if they already exist"
    )
    args = parser.parse_args()

    # Check for OAuth credentials
    if not os.path.exists(args.client_secret):
        logger.error(f"OAuth client secret not found: {args.client_secret}")
        logger.error("Download from Google Cloud Console: APIs & Services > Credentials")
        return

    # Load collaborative task configs
    if not os.path.exists(args.config_file):
        logger.error(f"Config file not found: {args.config_file}")
        return

    logger.info(f"Loading task configs from {args.config_file}")
    with open(args.config_file) as f:
        config_data = json.load(f)

    task_metadata = config_data.get("tasks", {})
    active_tasks = {
        task_id: meta
        for task_id, meta in task_metadata.items()
        if meta.get("status") == "active"
    }

    logger.info(f"Found {len(active_tasks)} active collaborative tasks")

    # Load existing sheet URLs if they exist
    sheet_urls = {}
    if os.path.exists(args.output_file) and not args.force:
        logger.info(f"\nLoading existing sheet URLs from {args.output_file}")
        with open(args.output_file) as f:
            sheet_urls = json.load(f)
        logger.info(f"Found {len(sheet_urls)} existing sheets")
        if len(sheet_urls) == len(active_tasks):
            logger.info("\n✓ All sheets already created!")
            logger.info(f"URLs saved in: {args.output_file}")
            logger.info(f"\nTo recreate all sheets, use: --force")
            logger.info(f"\nYou can now run collaborative tasks:")
            logger.info(f"  python run_benchmark.py --task-type collaborative --num-tasks 10 ...")
            return
        logger.info(f"Will create {len(active_tasks) - len(sheet_urls)} missing sheets")
    elif args.force:
        logger.info(f"\n--force flag set: will recreate all sheets")

    # Load task details to get templates
    total = len(active_tasks)
    created_count = 0
    skipped_count = 0

    for idx, (task_id, metadata) in enumerate(active_tasks.items(), 1):
        logger.info(f"\n{'='*70}")
        logger.info(f"Task {idx}/{total}: {task_id}")
        logger.info(f"{'='*70}")

        # Load full task config
        task_file = f"evaluation_examples/examples/collaborative/{task_id}.json"
        if not os.path.exists(task_file):
            logger.warning(f"Task file not found: {task_file}, skipping")
            continue

        with open(task_file) as f:
            task_data = json.load(f)

        # Find google_sheet_from_template config step
        config_steps = task_data.get("config", [])
        sheet_config = None
        for step in config_steps:
            if step.get("type") == "google_sheet_from_template":
                sheet_config = step
                break

        if not sheet_config:
            logger.warning(f"No google_sheet_from_template found for {task_id}, skipping")
            continue

        params = sheet_config["parameters"]
        template_url = params["template_url"]
        title = params.get("title", f"OSWorld Task {task_id}")

        logger.info(f"Template: {template_url[:80]}...")
        logger.info(f"Title: {title}")

        # Check if sheet already exists (unless --force)
        if task_id in sheet_urls and not args.force:
            logger.info(f"⊙ Already exists: {sheet_urls[task_id]}")
            logger.info(f"   Skipping (use --force to recreate)")
            skipped_count += 1
            continue

        # Create sheet
        try:
            sheet_url = create_sheet_from_template_oauth(
                template_url=template_url,
                client_secret_path=args.client_secret,
                token_path=args.token,
                title=title
            )

            sheet_urls[task_id] = sheet_url
            logger.info(f"✓ Created: {sheet_url}")
            created_count += 1

            # Save progress incrementally
            with open(args.output_file, 'w') as f:
                json.dump(sheet_urls, f, indent=2)

            # Rate limiting
            if idx < total:
                logger.info(f"Waiting {args.delay}s before next sheet...")
                time.sleep(args.delay)

        except Exception as e:
            logger.error(f"✗ Failed to create sheet for {task_id}: {e}")
            continue

    # Final summary
    logger.info(f"\n{'='*70}")
    logger.info("SETUP COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Total sheets: {len(sheet_urls)}/{total}")
    if created_count > 0:
        logger.info(f"  Created: {created_count}")
    if skipped_count > 0:
        logger.info(f"  Skipped (already exist): {skipped_count}")
    logger.info(f"URLs saved to: {args.output_file}")
    logger.info(f"\nYou can now run collaborative tasks infinitely:")
    logger.info(f"  python run_benchmark.py --task-type collaborative --num-tasks 10 ...")


if __name__ == "__main__":
    main()
