"""Wrapper for virtual display interactions via VM controller."""

import base64
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class XvfbDisplay:
    """Screenshot and action execution on a specific DISPLAY number."""

    def __init__(self, vm_ip: str, server_port: int, display_num: int):
        self.vm_ip = vm_ip
        self.server_port = server_port
        self.display_num = display_num
        self.display = f":{display_num}"
        self.exec_url = f"http://{vm_ip}:{server_port}/setup/execute"

    def _shell(self, cmd: str, timeout: int = 60) -> Optional[dict]:
        import requests
        try:
            r = requests.post(
                self.exec_url,
                json={"command": cmd, "shell": True},
                timeout=timeout,
            )
            return r.json() if r.status_code == 200 else None
        except Exception as e:
            logger.warning("[%s] shell failed: %s", self.display, e)
            return None

    def screenshot(self) -> Optional[bytes]:
        """Capture screenshot from this display using multiple fallback methods."""
        tmp = f"/tmp/fork_shot_{self.display.replace(':', '')}.png"

        shot_result = self._shell(
            f"DISPLAY={self.display} python3 -c \""
            f"import pyautogui; pyautogui.screenshot('{tmp}')\" 2>&1"
        )

        check = self._shell(f"test -s {tmp} && echo OK || echo FAIL")
        if not check or "OK" not in check.get("output", ""):
            err = shot_result.get("output", "") if shot_result else "no response"
            logger.warning(
                "[%s] pyautogui screenshot failed: %s -- trying fallbacks",
                self.display, err[:150],
            )
            self._shell(
                f"DISPLAY={self.display} scrot -o {tmp} 2>/dev/null || "
                f"DISPLAY={self.display} import -window root {tmp} 2>/dev/null || "
                f"DISPLAY={self.display} xwd -root -silent 2>/dev/null | "
                f"convert xwd:- {tmp} 2>/dev/null"
            )

        result = self._shell(f"base64 -w0 {tmp}")
        if result and result.get("output"):
            try:
                return base64.b64decode(result["output"].strip())
            except Exception as e:
                logger.warning("[%s] failed to decode screenshot: %s", self.display, e)

        logger.warning("[%s] screenshot failed -- all methods exhausted", self.display)
        return None

    def run_action(self, action_code: str) -> dict:
        """Execute Python action code on this display."""
        import requests
        tmp = f"/tmp/fork_action_{self.display.replace(':', '')}.py"
        full_code = (
            "import pyautogui; import time; pyautogui.FAILSAFE = False\n"
            + action_code
        )
        write_cmd = f"cat > {tmp} << 'ACTIONEOF'\n{full_code}\nACTIONEOF"

        requests.post(
            self.exec_url,
            json={"command": write_cmd, "shell": True},
            timeout=30,
        )

        resp = requests.post(
            self.exec_url,
            json={"command": f"DISPLAY={self.display} python3 {tmp}", "shell": True},
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()
