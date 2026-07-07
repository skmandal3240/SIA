#!/usr/bin/env python3
"""Reusable SIA action parser.

Extracts tool calls and screen actions from the model's tagged output.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ToolCall:
    tool: str
    arguments: dict


@dataclass(frozen=True)
class ScreenAction:
    kind: str
    x: float
    y: float
    label: str
    screen: str


def parse_sia_text(text: str) -> tuple[list[ToolCall], list[ScreenAction], str]:
    """Return (tool_calls, screen_actions, stripped_text)."""
    tools: list[ToolCall] = []
    actions: list[ScreenAction] = []

    # Tool grammar: <|sia_tool|>NAME<|sia_call|>{...}<|sia_endcall|>
    for m in re.finditer(
        r"<\|sia_tool\|\>(\w+)<\|sia_call\|\>(\{.*?\})<\|sia_endcall\|\>",
        text,
        flags=re.DOTALL,
    ):
        try:
            tools.append(ToolCall(tool=m.group(1), arguments=json.loads(m.group(2))))
        except json.JSONDecodeError:
            pass

    # Fallback: bare JSON object/array.
    if not tools:
        try:
            obj = json.loads(text.strip())
            if isinstance(obj, list):
                obj = obj[0] if obj else None
            if isinstance(obj, dict) and "tool" in obj:
                tools.append(ToolCall(tool=obj["tool"], arguments=obj.get("arguments", {})))
        except json.JSONDecodeError:
            pass

    # Screen grammar: <|sia_screen|><|sia_action|>POINT:x,y:label:screen<|sia_endaction|>
    for m in re.finditer(
        r"POINT:([0-9.]+),([0-9.]+):([^:]+):([^\s<]+)",
        text,
    ):
        try:
            actions.append(
                ScreenAction(
                    kind="POINT",
                    x=float(m.group(1)),
                    y=float(m.group(2)),
                    label=m.group(3),
                    screen=m.group(4),
                )
            )
        except ValueError:
            pass

    stripped = re.sub(
        r"<\|sia_(tool|call|endcall|action|endaction|screen)\|\>",
        "",
        text,
    )
    return tools, actions, stripped.strip()


def main() -> int:
    sample = (
        '<|sia_tool|>set_alarm<|sia_call|>{"time":"07:00"}<|sia_endcall|> '
        '<|sia_screen|><|sia_action|>POINT:0.12,0.34:submit_button:screen0<|sia_endaction|>'
    )
    tools, actions, stripped = parse_sia_text(sample)
    assert len(tools) == 1 and tools[0].tool == "set_alarm"
    assert len(actions) == 1 and actions[0].label == "submit_button"
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
