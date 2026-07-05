"""Screen capture stub for the SIA Shell embodiment."""

from __future__ import annotations

from pathlib import Path


class CaptureStub:
    """Stub screen capturer that generates a synthetic screenshot.

    Real implementation will use mss / PIL on Linux and platform-specific
    APIs elsewhere. The stub returns a fixed-resolution buffer and a path.
    """

    def __init__(self, width: int = 1280, height: int = 720) -> None:
        self.width = width
        self.height = height

    def grab(self) -> dict:
        """Return a synthetic RGBA buffer plus dimensions."""
        size = self.width * self.height * 4
        return {
            "width": self.width,
            "height": self.height,
            "channels": 4,
            "buffer": bytes(size),
        }

    def save(self, path: str | Path) -> Path:
        """Stub: pretend to save and return the path."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(self.grab()["buffer"])
        return out
