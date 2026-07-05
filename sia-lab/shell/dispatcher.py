"""Shared dispatcher for API tool-calls and on-screen actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .tag_parser import ActionTag, parse_action_tags


@dataclass(frozen=True)
class ToolResult:
    tool: str
    args: dict
    output: str


@dataclass(frozen=True)
class ActionResult:
    tag: ActionTag
    ok: bool
    message: str


def _default_tool_runner(tool: str, args: dict) -> str:
    return f"stub:{tool}({args})"


def _default_action_runner(tag: ActionTag) -> str:
    return f"executed:{tag.kind}({tag.x},{tag.y})"


class Dispatcher:
    """Routes LLM outputs to tool calls or on-screen actions.

    The dispatcher extracts action tags from the model response, runs any
    registered tool calls, and dispatches on-screen actions through a
    configurable runner. It is intentionally tiny: the loop calls it once
    per reasoning cycle.
    """

    def __init__(
        self,
        tool_runner: Callable[[str, dict], str] | None = None,
        action_runner: Callable[[ActionTag], str] | None = None,
    ) -> None:
        self.tool_runner = tool_runner or _default_tool_runner
        self.action_runner = action_runner or _default_action_runner

    def dispatch(self, response: str, tool_calls: list[dict] | None = None) -> tuple[list[ToolResult], list[ActionResult]]:
        """Dispatch tool calls and parse+run action tags from a response."""
        tools: list[ToolResult] = []
        for call in tool_calls or []:
            tool = call.get("tool", "unknown")
            args = call.get("args", {})
            tools.append(ToolResult(tool=tool, args=args, output=self.tool_runner(tool, args)))

        actions: list[ActionResult] = []
        for tag in parse_action_tags(response):
            try:
                msg = self.action_runner(tag)
                actions.append(ActionResult(tag=tag, ok=True, message=msg))
            except Exception as exc:  # ponytail: keep loop alive, surface error
                actions.append(ActionResult(tag=tag, ok=False, message=str(exc)))

        return tools, actions
