"""On-screen action tag parser.

Tags have the form:

    [POINT:x,y:label:screenN]
    [CLICK:x,y:screenN]
    [TYPE:x,y:text:screenN]
    [SCROLL:x,y:dx,dy:screenN]

The final :screenN field is optional. Unknown tag types are captured so the
loop can report them instead of silently dropping them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActionTag:
    kind: str
    x: float
    y: float
    label: str | None
    screen: int
    raw: str


_ALLOWED_KINDS = {"POINT", "CLICK", "TYPE", "SCROLL"}


def _split_payload(parts: list[str], kind: str) -> tuple[str | None, int]:
    """Return (label_or_text, scroll_dx, scroll_dy) and screen index."""
    screen = 0
    payload_parts = parts[2:]
    if payload_parts:
        last = payload_parts[-1]
        if last.startswith("screen") and last[6:].isdigit():
            screen = int(last[6:])
            payload_parts = payload_parts[:-1]

    payload = ":".join(payload_parts) if payload_parts else None
    if kind == "SCROLL" and payload and "," in payload:
        # ponytail: scroll payload is dx,dy; keep it as a label for dispatch.
        payload = f"scroll:{payload}"
    return payload, screen


def parse_action_tags(text: str) -> list[ActionTag]:
    """Parse all action tags in a model/response string."""
    out: list[ActionTag] = []
    start = 0
    while True:
        bracket_open = text.find("[", start)
        if bracket_open == -1:
            break
        bracket_close = text.find("]", bracket_open)
        if bracket_close == -1:
            break
        raw = text[bracket_open : bracket_close + 1]
        body = text[bracket_open + 1 : bracket_close]
        parts = body.split(":")
        if len(parts) >= 2 and parts[0] in _ALLOWED_KINDS:
            try:
                kind = parts[0]
                if "," in parts[1]:
                    x, y = parts[1].split(",")
                    payload, screen = _split_payload(parts, kind)
                else:
                    # ponytail: tolerate [POINT:x:y:label] from imprecise small models.
                    x, y = parts[1], parts[2]
                    payload, screen = _split_payload([kind, f"{x},{y}"] + parts[3:], kind)
                out.append(
                    ActionTag(
                        kind=kind,
                        x=float(x),
                        y=float(y),
                        label=payload,
                        screen=screen,
                        raw=raw,
                    )
                )
            except (ValueError, IndexError):
                pass  # malformed; skip
        start = bracket_close + 1
    return out
