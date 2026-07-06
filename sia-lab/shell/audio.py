"""Local audio I/O for the SIA Shell embodiment (Linux).

Recording uses ffmpeg with alsa/pulse if available. Playback also uses ffmpeg.
Real STT/TTS models are intentionally not hard-coded; callers provide the
model. This file only handles raw audio bytes.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path


def _has(cmd: str) -> bool:
    return shutil.which(cmd) is not None


class FfmpegAudioRecorder:
    """Record raw PCM audio via ffmpeg.

    Defaults to 16 kHz mono s16 PCM, which most local STT/TTS engines expect.
    """

    def __init__(self, sample_rate: int = 16000, channels: int = 1, duration: int = 5) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.duration = duration

    def record(self) -> bytes:
        if not _has("ffmpeg"):
            raise RuntimeError("ffmpeg not found; install it to record audio")
        backend = "pulse" if _has("pactl") else ("alsa" if _has("arecord") else "default")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out = Path(f.name)
        cmd = [
            "ffmpeg", "-y",
            "-f", backend,
            "-i", "default",
            "-ar", str(self.sample_rate),
            "-ac", str(self.channels),
            "-acodec", "pcm_s16le",
            "-t", str(self.duration),
            str(out),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return out.read_bytes()
        finally:
            out.unlink(missing_ok=True)

    def stream(self) -> Iterator[bytes]:
        # ponytail: one-shot recording is enough for a P2 smoke test.
        yield self.record()


class FfmpegPlayer:
    """Play raw/PCM/WAV audio via ffmpeg."""

    def play(self, audio: bytes, fmt: str = "s16le", sample_rate: int = 22050, channels: int = 1) -> None:
        if not _has("ffmpeg"):
            raise RuntimeError("ffmpeg not found; install it to play audio")
        with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as f:
            raw = Path(f.name)
            raw.write_bytes(audio)
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-f", fmt,
                    "-ar", str(sample_rate),
                    "-ac", str(channels),
                    "-i", str(raw),
                    "-f", "pulse" if _has("pactl") else "alsa",
                    "default",
                ],
                check=True,
                capture_output=True,
            )
        finally:
            raw.unlink(missing_ok=True)


def record_audio(duration: int = 5) -> bytes:
    """Convenience one-liner for P2 demos."""
    return FfmpegAudioRecorder(duration=duration).record()
