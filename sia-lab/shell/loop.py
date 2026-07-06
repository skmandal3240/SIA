"""see-screen → reason → point/act → speak loop for SIA Shell."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .capture import CaptureStub
from .stt import StreamingSTTStub
from .tts import StreamingTTSStub
from .dispatcher import Dispatcher, ToolResult, ActionResult


@dataclass
class Turn:
    screenshot: dict
    transcript: str
    response: str
    tools: list[ToolResult] = field(default_factory=list)
    actions: list[ActionResult] = field(default_factory=list)
    spoken: list[bytes] = field(default_factory=list)
    context: str = ""


class ShellLoop:
    """One-turn see-screen → reason → point/act → speak loop.

    The `reason` callable receives the screenshot buffer, transcript, and an
    optional context summary and returns a text response plus any tool calls.
    The dispatcher turns tags into on-screen actions and tool calls into tool
    results. The TTS stub streams the response text to audio.

    P4: A TokenCake working memory is optionally maintained across turns; each
    turn is recorded as an episode.
    """

    def __init__(
        self,
        capture: CaptureStub | None = None,
        stt: StreamingSTTStub | None = None,
        tts: StreamingTTSStub | None = None,
        dispatcher: Dispatcher | None = None,
        reason: Callable[[dict, str, str], tuple[str, list[dict]]] | None = None,
        context: str = "",
        memory: Any | None = None,
    ) -> None:
        self.capture = capture or CaptureStub()
        self.stt = stt or StreamingSTTStub()
        self.tts = tts or StreamingTTSStub()
        self.dispatcher = dispatcher or Dispatcher()
        self.reason = reason or self._default_reason
        self.context = context
        self.memory = memory

    def _default_reason(self, screenshot: dict, transcript: str, context: str) -> tuple[str, list[dict]]:
        # ponytail: echo-style stub so the loop is testable without a model.
        return f"SIA saw {screenshot['width']}x{screenshot['height']} and heard: {transcript} (context: {context}) [POINT:100,200:submit_button:screen0]", []

    def run_once(self, audio_chunks: list[bytes] | None = None) -> Turn:
        """Run one full loop turn from audio input to spoken output."""
        screenshot = self.capture.grab()

        transcript = ""
        for text in self.stt.stream(iter(audio_chunks or [b"tap"])):
            transcript = text

        memory_context = self.context
        # ponytail: if a TokenCake is attached, summarize recent user turns as context.
        if self.memory is not None and hasattr(self.memory, "to_messages"):
            msgs = self.memory.to_messages()
            if msgs:
                memory_context = " | ".join(f"{m['role']}: {m['content']}" for m in msgs[-3:])

        response, tool_calls = self.reason(screenshot, transcript, memory_context)
        tools, actions = self.dispatcher.dispatch(response, tool_calls)

        # Record this turn in memory.
        if self.memory is not None:
            if hasattr(self.memory, "add"):
                self.memory.add("user", transcript)
                self.memory.add("assistant", response)

        spoken: list[bytes] = []
        for chunk in self.tts.stream(iter([response])):
            spoken.append(chunk)

        return Turn(
            screenshot=screenshot,
            transcript=transcript,
            response=response,
            tools=tools,
            actions=actions,
            spoken=spoken,
            context=memory_context,
        )
