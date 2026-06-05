"""Manage the GNU Radio BPSK video streaming script as a subprocess.

Uses a PID file so that start/stop/status work correctly even across
different Python processes (e.g. CLI vs FastAPI server).
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

GR_SCRIPT = "/home/njupt/Desktop/VIDEO/bpsk_video_stream.py"
GR_PYTHON = "/usr/bin/python3"

PID_FILE = Path("/tmp/sdr_video_stream.pid")
LOG_FILE = Path("/tmp/sdr_video_stream.log")


def _pid_is_gr_script(pid: int) -> bool:
    """Check whether the given PID belongs to the GR video streaming script."""
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_text()
        return "bpsk_video_stream.py" in cmdline
    except (FileNotFoundError, PermissionError, OSError):
        return False


def _read_pid_file() -> int | None:
    try:
        pid = int(PID_FILE.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None
    if _pid_is_gr_script(pid):
        return pid
    PID_FILE.unlink(missing_ok=True)
    return None


def _write_pid_file(pid: int) -> None:
    PID_FILE.write_text(str(pid))


def _remove_pid_file() -> None:
    PID_FILE.unlink(missing_ok=True)


class VideoStreamController:
    """Launch the GNU Radio video streaming flowgraph as a subprocess.

    State is persisted via /tmp/sdr_video_stream.pid so that different
    Python processes (CLI, FastAPI, MCP) see the same running task.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

    # ── public API ──────────────────────────────────────────────────────

    def get_state(self) -> dict[str, Any]:
        pid = _read_pid_file()
        return {
            "active": pid is not None,
            "pid": pid,
        }

    def start(self) -> dict[str, Any]:
        with self._lock:
            existing_pid = _read_pid_file()
            if existing_pid is not None:
                return {
                    "status": "error",
                    "error": f"视频流任务已在运行中 (PID {existing_pid})，请先停止。",
                    "pid": existing_pid,
                }

            if not os.path.exists(GR_SCRIPT):
                return {
                    "status": "error",
                    "error": f"GNU Radio 脚本不存在: {GR_SCRIPT}",
                }
            if not os.path.exists(GR_PYTHON):
                return {
                    "status": "error",
                    "error": f"GNU Radio Python 解释器不存在: {GR_PYTHON}",
                }

            # Clean up leftover VLC processes
            _kill_vlc()

            env = os.environ.copy()
            if "DISPLAY" not in env:
                env["DISPLAY"] = ":0"
            # Ensure XAUTHORITY is set so Qt/VLC can connect to the X server
            if "XAUTHORITY" not in env:
                xauth_candidate = os.path.expanduser("~/.Xauthority")
                if os.path.exists(xauth_candidate):
                    env["XAUTHORITY"] = xauth_candidate

            # Open log file to capture stderr
            log_fp = LOG_FILE.open("w")

            try:
                proc = subprocess.Popen(
                    [GR_PYTHON, "-u", GR_SCRIPT],
                    env=env,
                    stdout=log_fp,
                    stderr=log_fp,
                    stdin=subprocess.DEVNULL,
                    preexec_fn=os.setsid,
                )
            except Exception as exc:
                log_fp.close()
                return {
                    "status": "error",
                    "error": f"无法启动 GNU Radio 视频流进程: {exc}",
                }

            _write_pid_file(proc.pid)

            # Wait briefly and verify the process is still alive
            time.sleep(2.0)
            if proc.poll() is not None:
                # Process died — read stderr to report what went wrong
                log_fp.close()
                exit_code = proc.returncode
                error_msg = _read_log_tail()
                _remove_pid_file()
                return {
                    "status": "error",
                    "error": (
                        f"GNU Radio 视频流进程启动后立即退出 (退出码 {exit_code})。"
                    ),
                    "stderr_tail": error_msg,
                    "pid": proc.pid,
                }
            log_fp.close()

            return {
                "status": "success",
                "message": (
                    "BPSK 视频流传输已启动。"
                    "GNU Radio Qt 窗口和 VLC 播放器将自动打开。"
                ),
                "pid": proc.pid,
            }

    def stop(self) -> dict[str, Any]:
        with self._lock:
            pid = _read_pid_file()
            if pid is None:
                # Double-check via pgrep as fallback
                pid = _find_gr_process()
                if pid is None:
                    return {"status": "success", "message": "当前没有视频流任务在运行。"}

            # SIGTERM to the process group
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass

            # Wait for graceful exit
            for _ in range(50):
                if not _pid_is_gr_script(pid):
                    break
                time.sleep(0.1)
            else:
                # Force kill if still alive
                try:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
                time.sleep(0.5)

            _remove_pid_file()
            _kill_vlc()

            return {
                "status": "success",
                "message": "视频流任务已停止，VLC 进程已清理。",
            }


# ── module-level helpers ────────────────────────────────────────────────


def _find_gr_process() -> int | None:
    """Scan the process table for a running bpsk_video_stream.py process."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "bpsk_video_stream.py"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip().split()[0])
    except Exception:
        pass
    return None


def _read_log_tail(max_lines: int = 50) -> str:
    """Read the last *max_lines* lines from the stderr log file."""
    try:
        lines = LOG_FILE.read_text().splitlines()
        return "\n".join(lines[-max_lines:])
    except (FileNotFoundError, OSError):
        return "(日志文件不可读)"


def _kill_vlc() -> None:
    try:
        subprocess.run(["killall", "-9", "vlc"], capture_output=True, timeout=3)
    except Exception:
        pass
