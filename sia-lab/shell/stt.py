"""Streaming STT stub for the SIA Shell embodiment."""

from __future__ import annotations

from collections.abc import Iterator


class StreamingSTTStub:
    """Stub streaming speech-to-text.

    Real implementation will stream audio chunks to a local Whisper / Sarvam
    edge model. The stub yields canned transcripts from an iterable.
    """

    def __init__(self, transcripts: list[str] | None = None) -> None:
        self.transcripts = transcripts or []
        self._idx = 0

    def stream(self, audio_chunks: Iterator[bytes]) -> Iterator[str]:
        """Yield one transcript per non-empty audio chunk."""
        for chunk in audio_chunks:
            if not chunk:
                continue
            text = self.transcripts[self._idx] if self._idx < len(self.transcripts) else ""
            self._idx += 1
            if text:
                yield text

    def reset(self) -> None:
        self._idx = 0
