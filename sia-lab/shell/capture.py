"""Screen capture for the SIA Shell embodiment.

Provides a real implementation using mss on Linux/Windows/macOS and keeps
CaptureStub for headless/CI environments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class Capture(Protocol):
    def grab(self) -> dict:
        ...

    def save(self, path: str | Path) -> Path:
        ...


class CaptureStub:
    """Stub screen capturer that generates a synthetic screenshot."""

    def __init__(self, width: int = 1280, height: int = 720) -> None:
        self.width = width
        self.height = height

    def grab(self) -> dict:
        size = self.width * self.height * 4
        return {
            "width": self.width,
            "height": self.height,
            "channels": 4,
            "buffer": bytes(size),
        }

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(self.grab()["buffer"])
        return out


class MssCapture:
    """Real screen capture via mss (cross-platform).

    Returns an RGBA buffer and dimensions. Requires `mss` to be installed:
    `pip install mss`.
    """

    def __init__(self) -> None:
        try:
            import mss  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("mss is required for real capture: pip install mss") from exc
        self._mss = mss.mss()
        mon = self._mss.monitors[0]
        self.width = mon["width"]
        self.height = mon["height"]

    def grab(self) -> dict:
        import mss  # type: ignore
        with mss.mss() as sct:
            shot = sct.grab(sct.monitors[0])
        return {
            "width": shot.width,
            "height": shot.height,
            "channels": 4,
            "buffer": shot.raw,
        }

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        data = self.grab()
        try:
            from PIL import Image
        except ImportError:  # pragma: no cover
            out.write_bytes(data["buffer"])
            return out
        # ponytail: PIL is only used for saving; keep buffer raw in the API.
        Image.frombytes("RGBA", (data["width"], data["height"]), data["buffer"]).save(out)
        return out


def default_capture() -> Capture:
    """Use real capture if mss is available, otherwise fall back to stub."""
    try:
        return MssCapture()
    except Exception:  # ponytail: broad fallback so headless/CI still runs
        return CaptureStub()
