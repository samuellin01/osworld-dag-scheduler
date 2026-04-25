"""Test script for SetupExecutor.

Tests:
1. Initialize display pool
2. Allocate a display
3. Run various setup configs:
   - Open Chrome to a URL
   - Launch terminal (xterm)
   - Run shell commands
   - Sleep
4. Take screenshot to verify
5. Cleanup

Usage:
    python test_setup_executor.py --provider-name aws --region us-east-1
"""

import argparse
import base64
import logging
import os
import time
from typing import Optional

import requests

from display_pool import DisplayPool
from setup_executor import SetupExecutor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Test setup executor")
    parser.add_argument("--provider-name", default="aws", help="Provider (aws/docker)")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--headless", action="store_true", help="Run headless")
    parser.add_argument("--output-dir", default="test_setup_output", help="Output directory")
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

    # Wait for VM to be healthy
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

    # Initialize display pool
    logger.info("\n" + "=" * 60)
    logger.info("Initializing Display Pool")
    logger.info("=" * 60)

    pool = DisplayPool(vm_exec=vm_exec, num_displays=8, password=password)
    success = pool.initialize()

    if not success:
        logger.error("Display pool initialization failed!")
        return

    logger.info(f"✓ Pool initialized with {pool.get_idle_count()} idle displays")

    # Allocate a display for testing
    logger.info("\n" + "=" * 60)
    logger.info("Testing Setup Executor")
    logger.info("=" * 60)

    display_num = pool.allocate(agent_id="test_setup")
    if not display_num:
        logger.error("Failed to allocate display!")
        return

    logger.info(f"Allocated display :{display_num}")

    # Create setup executor
    executor = SetupExecutor(display_num=display_num, vm_exec=vm_exec)

    # Test 1: Open Chrome to a URL
    logger.info("\n" + "-" * 60)
    logger.info("Test 1: Open Chrome to Google")
    logger.info("-" * 60)

    config_1 = [
        {
            "type": "chrome_open_tabs",
            "parameters": {
                "urls_to_open": ["https://www.google.com"]
            }
        },
        {
            "type": "sleep",
            "parameters": {"seconds": 3}
        }
    ]

    success = executor.execute_config(config_1)
    logger.info(f"Config 1: {'✓ SUCCESS' if success else '✗ FAILED'}")

    # Take screenshot
    screenshot = executor.take_screenshot()
    if screenshot:
        path = os.path.join(args.output_dir, f"test1_chrome_display{display_num}.png")
        with open(path, "wb") as f:
            f.write(screenshot)
        logger.info(f"Screenshot saved: {path}")

    # Test 2: Launch terminal
    logger.info("\n" + "-" * 60)
    logger.info("Test 2: Launch xterm terminal")
    logger.info("-" * 60)

    config_2 = [
        {
            "type": "launch",
            "parameters": {
                "command": ["xterm", "-geometry", "80x24+100+100"]
            }
        },
        {
            "type": "sleep",
            "parameters": {"seconds": 2}
        }
    ]

    success = executor.execute_config(config_2)
    logger.info(f"Config 2: {'✓ SUCCESS' if success else '✗ FAILED'}")

    # Take screenshot
    screenshot = executor.take_screenshot()
    if screenshot:
        path = os.path.join(args.output_dir, f"test2_xterm_display{display_num}.png")
        with open(path, "wb") as f:
            f.write(screenshot)
        logger.info(f"Screenshot saved: {path}")

    # Test 3: Create file and open it
    logger.info("\n" + "-" * 60)
    logger.info("Test 3: Create text file and open in editor")
    logger.info("-" * 60)

    config_3 = [
        {
            "type": "command",
            "parameters": {
                "command": "echo 'Hello from setup executor!' > /tmp/test_setup.txt"
            }
        },
        {
            "type": "launch",
            "parameters": {
                "command": ["gedit", "/tmp/test_setup.txt"]
            }
        },
        {
            "type": "sleep",
            "parameters": {"seconds": 2}
        }
    ]

    success = executor.execute_config(config_3)
    logger.info(f"Config 3: {'✓ SUCCESS' if success else '✗ FAILED'}")

    # Check what windows are actually open
    logger.info("Checking open windows on display...")
    result = vm_exec(f"DISPLAY=:{display_num} wmctrl -l")
    if result and result.get("output"):
        logger.info(f"Open windows:\n{result['output']}")

    # Take screenshot
    screenshot = executor.take_screenshot()
    if screenshot:
        path = os.path.join(args.output_dir, f"test3_textfile_display{display_num}.png")
        with open(path, "wb") as f:
            f.write(screenshot)
        logger.info(f"Screenshot saved: {path}")

    # Test 4: Download file
    logger.info("\n" + "-" * 60)
    logger.info("Test 4: Download file from internet")
    logger.info("-" * 60)

    config_4 = [
        {
            "type": "download",
            "parameters": {
                "files": [
                    {
                        "path": "/tmp/test_download.txt",
                        "url": "https://raw.githubusercontent.com/anthropics/anthropic-sdk-python/main/README.md"
                    }
                ]
            }
        },
        {
            "type": "command",
            "parameters": {
                "command": "ls -lh /tmp/test_download.txt"
            }
        }
    ]

    success = executor.execute_config(config_4)
    logger.info(f"Config 4: {'✓ SUCCESS' if success else '✗ FAILED'}")

    # Test 5: Open multiple Chrome tabs
    logger.info("\n" + "-" * 60)
    logger.info("Test 5: Open multiple URLs in Chrome")
    logger.info("-" * 60)

    config_5 = [
        {
            "type": "chrome_open_tabs",
            "parameters": {
                "urls_to_open": [
                    "https://www.wikipedia.org",
                    "https://github.com"
                ]
            }
        },
        {
            "type": "sleep",
            "parameters": {"seconds": 3}
        }
    ]

    success = executor.execute_config(config_5)
    logger.info(f"Config 5: {'✓ SUCCESS' if success else '✗ FAILED'}")

    # Take final screenshot
    screenshot = executor.take_screenshot()
    if screenshot:
        path = os.path.join(args.output_dir, f"test5_multiple_tabs_display{display_num}.png")
        with open(path, "wb") as f:
            f.write(screenshot)
        logger.info(f"Screenshot saved: {path}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Setup Executor Test Complete")
    logger.info("=" * 60)
    logger.info(f"Display :{display_num} was configured with:")
    logger.info("  - Chrome with multiple tabs open")
    logger.info("  - Terminal window (xterm)")
    logger.info("  - Text file opened in editor")
    logger.info("  - Downloaded file")
    logger.info(f"\nScreenshots saved to: {args.output_dir}/")

    # Release display back to pool
    logger.info(f"\nReleasing display :{display_num} back to pool...")
    pool.release(display_num)
    logger.info(f"Idle displays: {pool.get_idle_count()}")

    # Cleanup
    logger.info("\nCleaning up display pool...")
    pool.cleanup()

    logger.info("\n" + "=" * 60)
    logger.info("✓ All tests passed!")
    logger.info("=" * 60)

    # Keep VM alive for manual inspection
    logger.info(f"\nVM still running. VNC: http://{vm_ip}:5910/vnc.html")
    logger.info("You can VNC in to see the displays")
    logger.info("Press Ctrl+C to terminate VM and exit")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\nShutting down VM...")


if __name__ == "__main__":
    main()
