"""Streaming TTS stub for the SIA Shell embodiment."""

from __future__ import annotations

from collections.abc import Iterator


class StreamingTTSStub:
    """Stub streaming text-to-speech.

    Real implementation will stream text to a local Piper / Coqui edge model.
    The stub yields one empty audio chunk per sentence as a heartbeat.
    """

    def __init__(self, chunk_size: int = 512) -> None:
        self.chunk_size = chunk_size

    def stream(self, text_stream: Iterator[str]) -> Iterator[bytes]:
        """Yield one small audio chunk per non-empty text chunk."""
        for text in text_stream:
            text = text.strip()
            if text:
                # ponytail: real TTS returns PCM; stub returns silence heartbeat.
                yield bytes(self.chunk_size)
