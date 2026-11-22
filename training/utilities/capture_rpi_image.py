"""Capture full-resolution JPEGs from a Raspberry Pi CSI camera over SSH.

Run in a virtual environment with Python 3.9+, Tk 8.6+, Pillow and FFmpeg:
    python3 training/utilities/capture_rpi_image.py
    python3 -m pip install Pillow

On Homebrew Python 3.12, install Tk with `brew install python-tk@3.12`.
Use `ssh rpi-camera true` once to verify the host key; unlock an encrypted key
with `ssh-add` before starting. No SSH security configuration is changed.
The Pi needs its usual bash, flock (util-linux), rpicam-vid and rpicam-still.
Only a temporary session lock is created on the Pi, never image files/scripts.
"""

from __future__ import annotations

import io
import os
import select
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import BinaryIO, Optional

try:
    import tkinter as tk
    from tkinter import ttk
    from PIL import Image, ImageTk
except ImportError as exc:
    raise SystemExit(
        f"GUI dependency missing: {exc}\n"
        "Install Pillow in the virtual environment used to run this file:\n"
        "  python3 -m pip install Pillow\n"
        "For Homebrew Python 3.12: brew install python-tk@3.12\n"
        "Then run with /opt/homebrew/bin/python3.12."
    ) from exc


# ============================================================
# Configuration — edit these values, then run this file again.
# ============================================================

SSH_HOST = "rpi-camera"
OUTPUT_DIR = "/Users/whatdojdo./Desktop/클라우드 AioT/AioT_project/CV-Medication-Detection/training/dataset_synth_prep"

STREAM_WIDTH = 1280
STREAM_HEIGHT = 720
STREAM_FPS = 30

CAPTURE_QUALITY = 100
CAPTURE_WARMUP_MS = 1500
CAPTURE_TIMEOUT_SEC = 30
IMAGE_PREFIX = "capture"

SSH_CONNECT_TIMEOUT_SEC = 5
PROCESS_STOP_TIMEOUT_SEC = 3
REMOTE_LEASE_SEC = 8
HEARTBEAT_INTERVAL_SEC = 1
STREAM_START_TIMEOUT_SEC = 20
STREAM_STALL_TIMEOUT_SEC = 10
RECONNECT_ATTEMPTS = 3
RECONNECT_DELAY_SEC = 2
CAMERA_BUSY_RETRIES = 2
MAX_JPEG_BYTES = 128 * 1024 * 1024
GUI_UPDATE_MS = 33


# An inline supervisor owns just its camera child. A stdin heartbeat lease also
# handles half-open SSH connections. The session lock stays held through wait(),
# so a new stream/still cannot acquire the camera before its predecessor exits.
# Closing FD 9 in the camera prevents it from inheriting the supervisor's lock.
REMOTE_WRAPPER = r'''
set -u
session=$1; token=$2; lease=$3; grace=$4; lock_wait=$5
shift 5
child=
child_start=
child_running() {
    [ -n "$child" ] && [ -r "/proc/$child/stat" ] || return 1
    IFS= read -r stat < "/proc/$child/stat" || return 1
    read -r -a fields <<< "${stat##*) }"
    case "${fields[0]}" in Z*|X*) return 1;; esac
    # PPID and start time prevent signalling a reused PID belonging to another job.
    [ "${fields[1]}" = "$$" ] || return 1
    if [ -n "$child_start" ] && [ "${fields[19]}" != "$child_start" ]; then return 1; fi
    child_start=${fields[19]}
    return 0
}
cleanup() {
    trap '' HUP INT TERM
    if [ -n "$child" ]; then
        if child_running; then
            kill -TERM "$child" 2>/dev/null || :
            for ((i=0; i<grace*10; i++)); do
                child_running || break
                sleep 0.1
            done
            if child_running; then kill -KILL "$child" 2>/dev/null || :; fi
        fi
        wait "$child" 2>/dev/null || :
        child=
    fi
    printf '\nRPI_CAMERA_RELEASED:%s\n' "$token" >&2
}
trap cleanup EXIT
trap 'exit 143' HUP INT TERM
exec 9>"$session/lock" || exit 72
if ! flock -w "$lock_wait" 9; then
    printf 'Previous camera session has not released its lock.\n' >&2
    exit 73
fi
"$@" </dev/null 9>&- &
child=$!
last=$SECONDS
while child_running; do
    if IFS= read -r -t 0.2 control; then
        case "$control" in
            PING) last=$SECONDS;;
            STOP) exit 143;;
        esac
    else
        result=$?
        # Bash returns >128 for a read timeout, 1 for control-channel EOF.
        if ((result <= 128)); then exit 143; fi
    fi
    if ((SECONDS - last >= lease)); then
        printf 'Camera heartbeat lease expired.\n' >&2
        exit 124
    fi
done
wait "$child"
result=$?
child=
exit "$result"
'''

REMOTE_FENCE = r'''
set -eu
[ -d "$1" ] || exit 0
exec 9>"$1/lock"
flock -w "$2" 9
if [ "$3" = remove ]; then
    rm -- "$1/lock"
    rmdir -- "$1"
fi
'''


class Cancelled(Exception):
    """The application is shutting down."""


@dataclass
class CameraStatus:
    state: str = "Connecting"
    error: str = ""
    saved_count: int = 0
    last_filename: str = "—"
    resolution: str = "—"


class StderrTail:
    """Continuously drain stderr, retaining a bounded diagnostic tail."""

    def __init__(self, pipe: BinaryIO, token: str = "") -> None:
        self.pipe = pipe
        self.marker = f"RPI_CAMERA_RELEASED:{token}\n".encode() if token else b""
        self.released = threading.Event()
        self._tail = b""
        self._lock = threading.Lock()
        self.thread = threading.Thread(target=self._read, name="stderr-drain")
        self.thread.start()

    def _read(self) -> None:
        try:
            while True:
                chunk = os.read(self.pipe.fileno(), 4096)
                if not chunk:
                    break
                with self._lock:
                    self._tail = (self._tail + chunk)[-16384:]
                    if self.marker and self.marker in self._tail:
                        self.released.set()
        except (OSError, ValueError):
            pass

    def text(self) -> str:
        with self._lock:
            lines = self._tail.decode("utf-8", errors="replace").splitlines()
        return "\n".join(line for line in lines if not line.startswith("RPI_CAMERA_RELEASED:"))[-6000:].strip()

    def close(self) -> None:
        # The owning process has exited, so its stderr is now at EOF.
        self.thread.join(timeout=PROCESS_STOP_TIMEOUT_SEC)
        if not self.thread.is_alive():
            self.pipe.close()


def terminate_process(process: Optional[subprocess.Popen]) -> None:
    """Boundedly reap a local process group created by this utility."""
    if process is None:
        return
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=PROCESS_STOP_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=PROCESS_STOP_TIMEOUT_SEC)


def validate_jpeg(data: bytes) -> tuple[int, int]:
    """Fully decode for validation; never re-encode or change the saved bytes."""
    if not data or not data.startswith(b"\xff\xd8") or not data.endswith(b"\xff\xd9"):
        raise ValueError("Empty, incomplete, or invalid JPEG received.")
    with Image.open(io.BytesIO(data)) as image:
        if image.format != "JPEG":
            raise ValueError("The camera did not return JPEG data.")
        image.verify()
    # JPEG verify() alone does not decode entropy data or detect all truncation.
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        return image.size


def save_jpeg(data: bytes, directory: Path) -> Path:
    """Atomically publish complete original bytes without overwriting files."""
    directory.mkdir(parents=True, exist_ok=True)
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, prefix=".rpi-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        stamp = datetime.now()
        for offset in range(10000):
            timestamp = (stamp + timedelta(microseconds=offset)).strftime("%Y%m%d_%H%M%S_%f")
            destination = directory / f"{IMAGE_PREFIX}_{timestamp}.jpg"
            try:
                # Same-directory hard link is atomic, and fails if the name exists.
                # Unlike replace()/rename(), it never clobbers a previous capture.
                os.link(temporary, destination)
                return destination
            except FileExistsError:
                continue
        raise FileExistsError("Could not allocate a unique capture filename.")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class RPiCameraController:
    """One I/O worker serializes all camera operations; Tk runs independently.

    Capture and stream operations deliberately share this worker, which makes
    simultaneous camera ownership impossible without more worker locks/queues.
    Only immutable frame bytes and status snapshots cross into the GUI thread.
    """

    def __init__(self) -> None:
        self.output_dir = Path(OUTPUT_DIR).expanduser().resolve()
        self._lock = threading.Lock()
        self._closing = threading.Event()
        self._capture = threading.Event()
        self._reconnect = threading.Event()
        self.done = threading.Event()
        self._status = CameraStatus()
        self._frame: Optional[tuple[int, bytes]] = None
        self._sequence = 0
        self._worker: Optional[threading.Thread] = None
        self._remote: Optional[subprocess.Popen] = None
        self._remote_log: Optional[StderrTail] = None
        self._decoder: Optional[subprocess.Popen] = None
        self._decoder_log: Optional[StderrTail] = None
        self._session = f"/tmp/rpi-camera-{uuid.uuid4().hex}"
        self._session_created = False
        self._needs_fence = False
        self._last_ping = 0.0
        self._retry_count = 0

    def _status_update(self, **changes: object) -> None:
        with self._lock:
            self._status = replace(self._status, **changes)

    def status(self) -> CameraStatus:
        with self._lock:
            return replace(self._status)

    def get_latest_frame(self) -> Optional[tuple[int, bytes]]:
        with self._lock:
            return self._frame

    def _spawn(self, args: list[str], *, cleanup: bool = False, **kwargs: object) -> subprocess.Popen:
        # Sharing this lock with close() makes shutdown vs process creation atomic.
        with self._lock:
            if self._closing.is_set() and not cleanup:
                raise Cancelled()
            return subprocess.Popen(args, start_new_session=True, **kwargs)

    def _ssh_args(self, command: list[str]) -> list[str]:
        return [
            "ssh", "-T", "-o", "BatchMode=yes", "-o", "ForwardAgent=no",
            "-o", "ClearAllForwardings=yes", "-o", "ControlMaster=no",
            "-o", "ControlPath=none", "-o", "ConnectionAttempts=1",
            "-o", f"ConnectTimeout={SSH_CONNECT_TIMEOUT_SEC}",
            "-o", "ServerAliveInterval=2", "-o", "ServerAliveCountMax=3",
            SSH_HOST, shlex.join(command),
        ]

    def _short_ssh(self, command: list[str], timeout: float, *, cleanup: bool = False) -> bytes:
        process = self._spawn(self._ssh_args(command), cleanup=cleanup,
                              stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        deadline = time.monotonic() + timeout
        try:
            while True:
                if self._closing.is_set() and not cleanup:
                    raise Cancelled()
                try:
                    output, error = process.communicate(timeout=min(0.2, max(0.01, deadline - time.monotonic())))
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("SSH command timed out; check Wi-Fi and the Pi.")
            if process.returncode:
                detail = error.decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"SSH exited with code {process.returncode}: {detail}")
            return output
        finally:
            terminate_process(process)
            for pipe in (process.stdout, process.stderr):
                if pipe:
                    pipe.close()

    def check_connection(self) -> None:
        """Check executables and create a private, temporary session lock directory."""
        if not shutil.which("ssh"):
            raise RuntimeError("SSH client is missing. Install the macOS command line tools.")
        if not shutil.which("ffmpeg"):
            raise RuntimeError("FFmpeg is missing. Install it with: brew install ffmpeg")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Test actual write access without leaving a file or collecting an image.
        with tempfile.TemporaryFile(dir=self.output_dir):
            pass
        script = (
            'set -eu; for tool in bash flock rpicam-vid rpicam-still; do '
            'command -v "$tool" >/dev/null || { echo "Missing Pi command: $tool" >&2; exit 127; }; '
            'done; test -d /proc/self; '
            'if [ ! -d "$1" ]; then mkdir -m 700 -- "$1"; fi'
        )
        # A cancelled SSH may have created the directory before returning.
        self._session_created = True
        self._short_ssh(["sh", "-c", script, "rpi-check", self._session], SSH_CONNECT_TIMEOUT_SEC + 5)
        if self._needs_fence:
            self._fence()

    def _fence(self, *, remove: bool = False) -> None:
        # Acquiring this lock is explicit proof the old supervisor has finished
        # waiting for its camera child, even if its SSH acknowledgement was lost.
        wait = REMOTE_LEASE_SEC + PROCESS_STOP_TIMEOUT_SEC + 2
        self._short_ssh(
            ["bash", "-c", REMOTE_FENCE, "rpi-fence", self._session, str(wait), "remove" if remove else "keep"],
            SSH_CONNECT_TIMEOUT_SEC + wait + 2, cleanup=remove,
        )
        self._needs_fence = False

    def start_stream(self) -> None:
        """Start the single I/O worker (safe to call from Tk)."""
        with self._lock:
            if self._closing.is_set() or self._worker is not None:
                return
            self._worker = threading.Thread(target=self._run, name="camera-io")
            self._worker.start()

    def request_capture(self) -> bool:
        """Accept one shutter request only while the camera is live."""
        with self._lock:
            if self._closing.is_set() or self._status.state != "Live Streaming" or self._capture.is_set():
                return False
            self._status = replace(self._status, state="Capturing", error="")
            self._capture.set()
            return True

    def reconnect(self) -> None:
        if not self._closing.is_set():
            self._reconnect.set()

    def _start_remote(self, camera_command: list[str]) -> None:
        if self._needs_fence:
            self._fence()
        token = uuid.uuid4().hex
        command = ["bash", "-c", REMOTE_WRAPPER, "rpi-supervisor", self._session, token,
                   str(REMOTE_LEASE_SEC), str(PROCESS_STOP_TIMEOUT_SEC),
                   str(REMOTE_LEASE_SEC + PROCESS_STOP_TIMEOUT_SEC + 2), *camera_command]
        self._remote = self._spawn(self._ssh_args(command), stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        self._remote_log = StderrTail(self._remote.stderr, token)
        self._last_ping = 0
        self._heartbeat()

    def _heartbeat(self) -> None:
        if self._closing.is_set():
            raise Cancelled()
        if time.monotonic() - self._last_ping < HEARTBEAT_INTERVAL_SEC:
            return
        if self._remote is None or self._remote.poll() is not None:
            return
        try:
            self._remote.stdin.write(b"PING\n")
            self._last_ping = time.monotonic()
        except (BrokenPipeError, OSError):
            pass  # stdout EOF and the exit code supply a fuller diagnostic.

    def _open_stream(self) -> None:
        self._start_remote([
            "rpicam-vid", "-t", "0", "-n", "--inline", "--width", str(STREAM_WIDTH),
            "--height", str(STREAM_HEIGHT), "--framerate", str(STREAM_FPS),
            "--codec", "h264", "-o", "-",
        ])
        self._decoder = self._spawn([
            "ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostdin",
            "-fflags", "nobuffer", "-flags", "low_delay", "-probesize", "32768",
            "-analyzeduration", "0", "-threads", "1", "-f", "h264", "-i", "pipe:0",
            "-an", "-sn", "-vf", f"scale={STREAM_WIDTH}:{STREAM_HEIGHT}",
            "-pix_fmt", "rgb24", "-f", "rawvideo", "-flush_packets", "1", "pipe:1",
        ], stdin=self._remote.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        self._remote.stdout.close()  # FFmpeg owns its copy of this pipe.
        self._decoder_log = StderrTail(self._decoder.stderr)

    def _read_stream(self) -> None:
        frame_size = STREAM_WIDTH * STREAM_HEIGHT * 3
        pending = bytearray()
        started = last_frame = time.monotonic()
        received_frame = False
        while not self._closing.is_set() and not self._capture.is_set():
            self._heartbeat()
            readable, _, _ = select.select([self._decoder.stdout], [], [], 0.1)
            if readable:
                chunk = os.read(self._decoder.stdout.fileno(), min(262144, frame_size - len(pending)))
                if not chunk:
                    raise RuntimeError(self._diagnostic("Video stream ended (EOF)."))
                pending.extend(chunk)
                if len(pending) == frame_size:
                    with self._lock:
                        self._sequence += 1
                        self._frame = (self._sequence, bytes(pending))
                        if self._status.state in ("Connecting", "Reconnecting"):
                            self._status = replace(self._status, state="Live Streaming")
                    pending.clear()
                    last_frame = time.monotonic()
                    received_frame = True
                    if last_frame - started >= STREAM_STALL_TIMEOUT_SEC:
                        self._retry_count = 0
            limit = STREAM_STALL_TIMEOUT_SEC if received_frame else STREAM_START_TIMEOUT_SEC
            if time.monotonic() - last_frame > limit:
                raise TimeoutError(self._diagnostic("No video frames received before the timeout."))
        if self._closing.is_set():
            raise Cancelled()

    def _diagnostic(self, message: str) -> str:
        parts = [message]
        for process, log in ((self._remote, self._remote_log), (self._decoder, self._decoder_log)):
            if process is not None and process.poll() is not None:
                parts.append(f"Process exit code: {process.returncode}")
            if log is not None and log.text():
                parts.append(log.text())
        detail = "\n".join(parts)
        if "Permission denied" in detail or "Host key verification" in detail:
            detail += "\nVerify with `ssh rpi-camera true`; register/unlock your key with ssh-add."
        return detail

    def stop_stream(self) -> None:
        """Stop the owned camera child, confirm its wait(), then reap FFmpeg/SSH."""
        remote, decoder = self._remote, self._decoder
        try:
            if remote is not None:
                if remote.poll() is None:
                    try:
                        remote.stdin.write(b"STOP\n")
                        remote.stdin.close()
                    except (BrokenPipeError, OSError):
                        pass
                    drain = decoder.stdout if decoder is not None else remote.stdout
                    deadline = time.monotonic() + PROCESS_STOP_TIMEOUT_SEC + 3
                    # Keep the decoder flowing while rpicam shuts down; otherwise
                    # a full video pipe can prevent camera shutdown from finishing.
                    while remote.poll() is None and time.monotonic() < deadline:
                        if drain is not None and not drain.closed:
                            ready, _, _ = select.select([drain], [], [], 0.1)
                            if ready:
                                if not os.read(drain.fileno(), 262144):
                                    drain = None
                        else:
                            time.sleep(0.05)
                terminate_process(remote)
                if self._remote_log:
                    self._remote_log.close()
                    self._needs_fence = not self._remote_log.released.is_set()
        finally:
            terminate_process(decoder)
            if self._decoder_log:
                self._decoder_log.close()
            for process in (remote, decoder):
                if process:
                    for pipe in (process.stdin, process.stdout, process.stderr):
                        if pipe and not pipe.closed:
                            pipe.close()
            self._remote = self._decoder = None
            self._remote_log = self._decoder_log = None

    def capture_image(self) -> None:
        """Receive a separate full-resolution still, validate it, and save bytes."""
        for attempt in range(CAMERA_BUSY_RETRIES + 1):
            if self._closing.is_set():
                raise Cancelled()
            try:
                self._start_remote([
                    "rpicam-still", "-n", "-t", str(CAPTURE_WARMUP_MS),
                    "-q", str(CAPTURE_QUALITY), "-o", "-",
                ])
                pending = bytearray()
                deadline = time.monotonic() + CAPTURE_TIMEOUT_SEC
                while True:
                    self._heartbeat()
                    if time.monotonic() >= deadline:
                        raise TimeoutError(self._diagnostic("Still capture timed out."))
                    ready, _, _ = select.select([self._remote.stdout], [], [], 0.1)
                    if not ready:
                        continue
                    chunk = os.read(self._remote.stdout.fileno(), 262144)
                    if not chunk:
                        break
                    pending.extend(chunk)
                    if len(pending) > MAX_JPEG_BYTES:
                        raise ValueError("JPEG exceeds MAX_JPEG_BYTES; check the camera output.")
                remaining = max(0.01, deadline - time.monotonic())
                try:
                    code = self._remote.wait(timeout=min(PROCESS_STOP_TIMEOUT_SEC, remaining))
                except subprocess.TimeoutExpired as exc:
                    raise TimeoutError("Camera supervisor did not finish after JPEG EOF.") from exc
                self._remote_log.close()
                if code:
                    raise RuntimeError(self._diagnostic(f"Still capture failed (exit {code})."))
                if not self._remote_log.released.is_set():
                    raise RuntimeError("Camera exit acknowledgement was not received.")
                self._status_update(state="Saving")
                data = bytes(pending)
                width, height = validate_jpeg(data)
                if self._closing.is_set():
                    raise Cancelled()
                destination = save_jpeg(data, self.output_dir)
                with self._lock:
                    self._status = replace(self._status, saved_count=self._status.saved_count + 1,
                                           last_filename=destination.name, resolution=f"{width} × {height}", error="")
                return
            except (RuntimeError, TimeoutError) as exc:
                message = str(exc).lower()
                busy = any(word in message for word in ("busy", "in use", "failed to acquire", "pipeline handler in use"))
                if not busy or attempt >= CAMERA_BUSY_RETRIES:
                    raise
                self._status_update(error=f"Camera busy; retry {attempt + 1}/{CAMERA_BUSY_RETRIES}.\n{exc}")
                self.stop_stream()
                if self._closing.wait(RECONNECT_DELAY_SEC):
                    raise Cancelled()
            finally:
                self.stop_stream()

    def _run(self) -> None:
        try:
            while not self._closing.is_set():
                try:
                    self.check_connection()
                    self._open_stream()
                    self._read_stream()
                    self.stop_stream()
                    if self._capture.is_set():
                        try:
                            self.capture_image()
                        except Cancelled:
                            raise
                        except Exception as exc:
                            self._status_update(state="Error", error=f"Capture/save failed: {exc}")
                        finally:
                            self._capture.clear()
                        self._status_update(state="Reconnecting")
                        self._retry_count = 0
                except Cancelled:
                    break
                except Exception as exc:
                    detail = self._diagnostic(str(exc))
                    self.stop_stream()
                    self._capture.clear()
                    self._retry_count += 1
                    self._status_update(state="Disconnected", error=detail)
                    if self._retry_count <= RECONNECT_ATTEMPTS:
                        self._status_update(state="Reconnecting")
                        if self._closing.wait(RECONNECT_DELAY_SEC):
                            break
                    else:
                        self._status_update(state="Error")
                        while not self._closing.is_set():
                            if self._reconnect.wait(0.2):
                                break
                        self._reconnect.clear()
                        self._retry_count = 0
                        self._status_update(state="Reconnecting", error="")
        finally:
            try:
                self.stop_stream()
                if self._session_created or self._needs_fence:
                    self._fence(remove=True)
            except Exception as exc:
                detail = f"Remote cleanup could not be confirmed: {exc}"
                self._status_update(error=detail)
                print(detail, file=sys.stderr)
            finally:
                self._status_update(state="Disconnected")
                self.done.set()

    def close(self) -> None:
        """Request shutdown without blocking Tk; the worker performs cleanup."""
        with self._lock:
            self._closing.set()
            if self._worker is None:
                self.done.set()


class CameraGUI:
    """Tk widgets and ImageTk objects are created/updated only on this thread."""

    def __init__(self, controller: RPiCameraController) -> None:
        self.controller = controller
        self.root = tk.Tk()
        self.root.title("RPi Camera · Image Collection")
        self.root.geometry("1100x820")
        self.root.minsize(640, 480)
        self._closing = False
        self._sequence = -1
        self._source: Optional[Image.Image] = None
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._view_size = (0, 0)
        self._last_status: Optional[CameraStatus] = None

        content = ttk.Frame(self.root, padding=12)
        content.pack(fill="both", expand=True)
        self.preview = tk.Canvas(content, bg="#17191d", highlightthickness=0)
        self.preview.pack(fill="both", expand=True)
        self._preview_item = self.preview.create_image(0, 0, anchor="center")
        self._placeholder = self.preview.create_text(20, 20, anchor="nw", fill="white", text="Connecting to camera…")

        toolbar = ttk.Frame(content, padding=(0, 10))
        toolbar.pack(fill="x")
        self.capture_button = ttk.Button(toolbar, text="Capture (Space)", command=self.on_capture, state="disabled")
        self.capture_button.pack(side="left")
        self.reconnect_button = ttk.Button(toolbar, text="Reconnect", command=self.controller.reconnect, state="disabled")
        self.reconnect_button.pack(side="left", padx=10)
        self.capture_button.bind("<space>", self.on_capture)
        self.reconnect_button.bind("<space>", self.on_capture)
        self.status_text = tk.StringVar(value="Connecting")
        ttk.Label(toolbar, textvariable=self.status_text).pack(side="right")
        self.info_text = tk.StringVar(value="Saved this session: 0   |   Last image: —   |   Resolution: —")
        ttk.Label(content, textvariable=self.info_text).pack(anchor="w")
        directory = ttk.Label(content, text=f"Save directory: {controller.output_dir}", wraplength=1000)
        directory.pack(anchor="w", pady=4)
        self.error_text = tk.StringVar()
        ttk.Label(content, textvariable=self.error_text, foreground="#b42318", wraplength=1000).pack(anchor="w", fill="x")
        self.root.bind("<space>", self.on_capture)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(GUI_UPDATE_MS, self.update_frame)
        self.root.after(100, self.update_status)

    def on_capture(self, event: Optional[tk.Event] = None) -> str:
        if not self._closing:
            self.controller.request_capture()
        return "break"  # Avoid also activating a focused ttk button with Space.

    def update_frame(self) -> None:
        if self._closing:
            return
        frame = self.controller.get_latest_frame()
        changed = frame is not None and frame[0] != self._sequence
        if changed:
            self._sequence, data = frame
            self._source = Image.frombytes("RGB", (STREAM_WIDTH, STREAM_HEIGHT), data)
        size = (max(1, self.preview.winfo_width()), max(1, self.preview.winfo_height()))
        if self._source is not None and (changed or size != self._view_size):
            self._view_size = size
            shown = self._source.copy()
            shown.thumbnail(size, Image.Resampling.BILINEAR)
            self._photo = ImageTk.PhotoImage(shown)
            self.preview.coords(self._preview_item, size[0] / 2, size[1] / 2)
            self.preview.itemconfigure(self._preview_item, image=self._photo)
            self.preview.itemconfigure(self._placeholder, state="hidden")
        self.root.after(GUI_UPDATE_MS, self.update_frame)

    def update_status(self) -> None:
        if self._closing:
            return
        status = self.controller.status()
        if status != self._last_status:
            self._last_status = status
            self.status_text.set(status.state)
            self.info_text.set(f"Saved this session: {status.saved_count}   |   Last image: {status.last_filename}   |   Resolution: {status.resolution}")
            self.error_text.set(status.error)
            self.capture_button.configure(state="normal" if status.state == "Live Streaming" else "disabled")
            self.reconnect_button.configure(state="normal" if status.state in ("Disconnected", "Error") else "disabled")
        self.root.after(100, self.update_status)

    def on_close(self) -> None:
        if self._closing:
            return
        self._closing = True
        self.capture_button.configure(state="disabled")
        self.reconnect_button.configure(state="disabled")
        self.status_text.set("Disconnecting · cleaning up camera processes…")
        self.controller.close()
        self.root.after(50, self._wait_for_close)

    def _wait_for_close(self) -> None:
        if self.controller.done.is_set():
            self.root.destroy()
        else:
            self.root.after(100, self._wait_for_close)

    def run(self) -> None:
        self.controller.start_stream()
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.on_close()
            self.root.mainloop()
        finally:
            self.controller.close()
            if self.controller._worker is not None:
                self.controller._worker.join()


def validate_configuration() -> None:
    integers = (STREAM_WIDTH, STREAM_HEIGHT, STREAM_FPS, CAPTURE_QUALITY,
                CAPTURE_WARMUP_MS, SSH_CONNECT_TIMEOUT_SEC, PROCESS_STOP_TIMEOUT_SEC,
                REMOTE_LEASE_SEC, RECONNECT_ATTEMPTS, CAMERA_BUSY_RETRIES,
                MAX_JPEG_BYTES, GUI_UPDATE_MS)
    if any(type(value) is not int for value in integers):
        raise ValueError("Dimensions, FPS, quality, warmup, SSH/stop/lease seconds, retry counts and size limits must be integers.")
    positive = (STREAM_WIDTH, STREAM_HEIGHT, STREAM_FPS, CAPTURE_TIMEOUT_SEC,
                SSH_CONNECT_TIMEOUT_SEC, PROCESS_STOP_TIMEOUT_SEC, REMOTE_LEASE_SEC,
                HEARTBEAT_INTERVAL_SEC, STREAM_START_TIMEOUT_SEC, STREAM_STALL_TIMEOUT_SEC,
                RECONNECT_DELAY_SEC, MAX_JPEG_BYTES, GUI_UPDATE_MS)
    if any(value <= 0 for value in positive):
        raise ValueError("Dimensions, rates, timeouts and size limits must be positive.")
    if not 1 <= CAPTURE_QUALITY <= 100 or CAPTURE_WARMUP_MS < 1:
        raise ValueError("Use JPEG quality 1–100 and a positive camera warmup time.")
    if REMOTE_LEASE_SEC <= HEARTBEAT_INTERVAL_SEC * 3:
        raise ValueError("REMOTE_LEASE_SEC must exceed three heartbeat intervals.")
    if RECONNECT_ATTEMPTS < 0 or CAMERA_BUSY_RETRIES < 0:
        raise ValueError("Retry counts cannot be negative.")
    if not SSH_HOST or SSH_HOST.startswith("-") or any(char.isspace() for char in SSH_HOST):
        raise ValueError("SSH_HOST must be a single SSH host alias.")
    if not IMAGE_PREFIX or Path(IMAGE_PREFIX).name != IMAGE_PREFIX or IMAGE_PREFIX in (".", ".."):
        raise ValueError("IMAGE_PREFIX must be a filename prefix without directory separators.")


def main() -> None:
    try:
        validate_configuration()
        if sys.platform == "darwin" and tk.TkVersion < 8.6:
            raise RuntimeError(
                "macOS system Tk 8.5 is unsupported. Use Python with Tk 8.6 or newer.\n"
                "For the installed Homebrew Python 3.12:\n"
                "  brew install python-tk@3.12\n"
                "  /opt/homebrew/bin/python3.12 -m venv ~/.venvs/rpi-camera\n"
                "  ~/.venvs/rpi-camera/bin/python -m pip install Pillow\n"
                "  ~/.venvs/rpi-camera/bin/python training/utilities/capture_rpi_image.py"
            )
        controller = RPiCameraController()
        CameraGUI(controller).run()
    except (OSError, ValueError, RuntimeError, tk.TclError) as exc:
        raise SystemExit(f"Camera utility could not start: {exc}") from exc


if __name__ == "__main__":
    main()
