"""SIA Shell embodiment — vertical slice (stubs for Linux)."""

from .capture import CaptureStub
from .stt import StreamingSTTStub
from .tts import StreamingTTSStub
from .tag_parser import parse_action_tags, ActionTag
from .dispatcher import Dispatcher, ToolResult, ActionResult
from .loop import ShellLoop

__all__ = [
    "CaptureStub",
    "StreamingSTTStub",
    "StreamingTTSStub",
    "parse_action_tags",
    "ActionTag",
    "Dispatcher",
    "ToolResult",
    "ActionResult",
    "ShellLoop",
]
