"""macOS screen capture for the SIA Shell.

Uses the built-in `screencapture` CLI so no third-party dependency is needed.
Requires **Screen Recording** permission the first time it runs; the helper
below detects the absence and tells the user how to grant it.
"""

from __future__ import annotations

import shutil
import subprocess
from io import BytesIO
from pathlib import Path

from .capture import Capture


class ScreenCapturePermissionError(RuntimeError):
    pass


def _screencapture_available() -> bool:
    return shutil.which("screencapture") is not None


def check_macos_permission() -> None:
    """Raise if macOS screen-recording permission is missing.

    `screencapture -x` silently produces a blank image when permission is
    denied, so we test by reading the raw pixel bytes of a screenshot.
    A valid non-blank capture has at least one non-zero byte.
    """
    if not _screencapture_available():
        raise RuntimeError("screencapture not found; this helper only works on macOS")

    proc = subprocess.run(
        ["screencapture", "-x", "-P", "-"],
        capture_output=True,
    )
    data = proc.stdout
    if not data or all(b == 0 for b in data[:1024]):
        raise ScreenCapturePermissionError(
            "macOS screen-recording permission denied. "
            "Open System Settings > Privacy & Security > Screen Recording, "
            "and allow the terminal/IDE running SIA. Then re-run."
        )


class MacosCapture:
    """Capture the main display via `screencapture -x -P -`.

    `-x` plays the screenshot sound (not silent), `-P` captures the primary
    display, `-` writes PNG to stdout.
    """

    def __init__(self) -> None:
        if not _screencapture_available():
            raise RuntimeError("screencapture not found; this helper only works on macOS")
        check_macos_permission()
        self.width = 0
        self.height = 0

    def _read_png(self) -> bytes:
        proc = subprocess.run(
            ["screencapture", "-x", "-P", "-"],
            capture_output=True,
            check=True,
        )
        return proc.stdout

    def grab(self) -> dict:
        data = self._read_png()
        try:
            from PIL import Image
            img = Image.open(BytesIO(data))  # type: ignore[name-defined]
        except Exception:
            raise RuntimeError("PIL is needed to parse the PNG produced by screencapture")
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        self.width, self.height = img.size
        return {
            "width": self.width,
            "height": self.height,
            "channels": 4,
            "buffer": img.tobytes(),
        }

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        data = self._read_png()
        out.write_bytes(data)
        return out


def default_capture() -> Capture:
    """Return macOS real capture if available, else fall back to MSS, then stub."""
    try:
        return MacosCapture()
    except ScreenCapturePermissionError:
        raise
    except Exception:
        try:
            from .capture import MssCapture
            return MssCapture()
        except Exception:
            from .capture import CaptureStub
            return CaptureStub()
