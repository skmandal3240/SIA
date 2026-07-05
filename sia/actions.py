"""Deterministic action parser."""
import json
import re


# ponytail: regex parser is enough for a prototype; swap to model-native
# tool-calling once the trained model is strong enough.
def parse_action(text: str) -> dict | None:
    t = text.lower()
    patterns = [
        (
            "set_alarm",
            r"(?:wake me up|set an? alarm|alarm).*?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?",
            lambda m: {
                "time": f"{int(m[1]) % 12 + (12 if (m[3] or '').lower() == 'pm' else 0):02d}:{m[2] or '00'}",
                "label": "alarm",
            },
        ),
        (
            "send_message",
            r"(?:text|message|send a message to)\s+(\S+)(?:\s+(?:that|saying|to))?\s+(.+)",
            lambda m: {"contact": m[1], "message": m[2]},
        ),
        (
            "toggle_setting",
            r"(turn on|turn off|enable|disable|toggle)\s+(wifi|bluetooth|flashlight|airplane mode)",
            lambda m: {"setting": m[2].replace(" ", "_"), "state": "on" if m[1] in {"turn on", "enable"} else "off"},
        ),
        (
            "open_maps",
            r"(?:navigate to|directions to|open maps to|take me to)\s+(.+)",
            lambda m: {"location": m[1]},
        ),
        (
            "create_event",
            r"(?:schedule|create|add)\s+(?:a\s+)?\w*\s*(?:meeting|event|reminder|call)\b(?:\s+(?:for\s+)?(.+))?",
            lambda m: {"title": m[1] or "scheduled event", "start_time": "tomorrow 10:00", "duration_minutes": 60},
        ),
    ]
    for name, pat, args in patterns:
        m = re.match(pat, t)
        if m:
            return {"name": name, "arguments": args(m)}
    return None


def extract_tool_calls(text: str) -> list:
    # ponytail: parse any <tool_call> tags the model may emit.
    out = []
    for m in re.finditer(r"\u003ctool_call\u003e(.*?)\u003c/tool_call\u003e", text, re.S):
        try:
            out.append(json.loads(m.group(1)))
        except json.JSONDecodeError:
            out.append({"raw": m.group(1)})
    return out


if __name__ == "__main__":
    cases = [
        ("Wake me up at 7 AM", "set_alarm"),
        ("Text Ravi I am running late", "send_message"),
        ("Turn off bluetooth", "toggle_setting"),
        ("Directions to Patna airport", "open_maps"),
        ("Schedule a team call tomorrow", "create_event"),
        ("What is the weather?", None),
    ]
    for text, expected in cases:
        got = parse_action(text)
        assert (got["name"] if got else None) == expected, f"failed: {text}"
    print("Action parser OK")
