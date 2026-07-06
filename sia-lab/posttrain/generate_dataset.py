#!/usr/bin/env python3
"""Generate a larger synthetic device-actions dataset for P1 retraining.

Targets 1000 examples across alarm, message, maps, brightness, toggle, and
on-screen action verbs. The grammar uses the existing SIA tool/action tokens.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

TOOLS = ["set_alarm", "send_message", "open_maps", "set_brightness", "toggle_setting"]
CONTACTS = ["Mummy", "Papa", "Rahul", "Sneha", "Rohit", "Priya", "Boss"]
TIMES = ["06:00", "06:30", "07:00", "07:30", "08:00", "09:00", "17:30", "21:00"]
PLACES = ["Patna", "Bihta", "Patna Junction", "Airport", "office", "home"]
BODIES = ["I am busy", "I am running late", "Call me", "Reached", "On my way"]
SETTINGS = ["bluetooth", "wifi", "hotspot", "airplane_mode", "flashlight"]
LABELS = ["alarm_icon", "send_button", "search_bar", "submit_button", "toggle"]


def _system() -> dict:
    return {
        "role": "system",
        "content": (
            "You are SIA, a private on-device AI companion. "
            "When the user wants a device action, emit exactly one SIA tool call: "
            "<|sia_tool|>TOOL_NAME<|sia_call|>{\"arg\":\"value\"}<|sia_endcall|>. "
            "Do not explain. Do not refuse. No extra text."
        ),
    }


def _sample_alarm() -> dict:
    t = random.choice(TIMES)
    queries = [
        f"Wake me up at {t}",
        f"Set an alarm for {t}",
        f"Alarm lagao {t} baje",
    ]
    return {
        "messages": [
            _system(),
            {"role": "user", "content": random.choice(queries)},
            {"role": "assistant", "content": f'<|sia_tool|>set_alarm<|sia_call|>{{"time":"{t}"}}<|sia_endcall|>'},
        ]
    }


def _sample_message() -> dict:
    c = random.choice(CONTACTS)
    b = random.choice(BODIES)
    queries = [
        f"Text {c} that {b}",
        f"Send {c} a message: {b}",
        f"{c} ko WhatsApp karo: {b}",
    ]
    return {
        "messages": [
            _system(),
            {"role": "user", "content": random.choice(queries)},
            {"role": "assistant", "content": f'<|sia_tool|>send_message<|sia_call|>{{"contact":"{c}","body":"{b}"}}<|sia_endcall|>'},
        ]
    }


def _sample_maps() -> dict:
    p = random.choice(PLACES)
    return {
        "messages": [
            _system(),
            {"role": "user", "content": f"Open maps for {p}"},
            {"role": "assistant", "content": f'<|sia_tool|>open_maps<|sia_call|>{{"query":"{p}"}}<|sia_endcall|>'},
        ]
    }


def _sample_brightness() -> dict:
    level = random.randint(10, 90)
    return {
        "messages": [
            _system(),
            {"role": "user", "content": f"Make screen {level}% bright"},
            {"role": "assistant", "content": f'<|sia_tool|>set_brightness<|sia_call|>{{"level":"{level}%"}}<|sia_endcall|>'},
        ]
    }


def _sample_toggle() -> dict:
    s = random.choice(SETTINGS)
    state = random.choice(["on", "off"])
    return {
        "messages": [
            _system(),
            {"role": "user", "content": f"Turn {state} {s}"},
            {"role": "assistant", "content": f'<|sia_tool|>toggle_setting<|sia_call|>{{"setting":"{s}","state":"{state}"}}<|sia_endcall|>'},
        ]
    }


def _sample_screen_action() -> dict:
    label = random.choice(LABELS)
    x, y = round(random.uniform(0.05, 0.95), 2), round(random.uniform(0.05, 0.95), 2)
    return {
        "messages": [
            _system(),
            {"role": "user", "content": f"Tap the {label.replace('_', ' ')} on the screen"},
            {"role": "assistant", "content": f'<|sia_screen|><|sia_action|>POINT:{x},{y}:{label}:screen0<|sia_endaction|>'},
        ]
    }


_BUILDERS = [_sample_alarm, _sample_message, _sample_maps, _sample_brightness, _sample_toggle, _sample_screen_action]


def generate(n: int = 1000, seed: int = 42) -> list[dict]:
    random.seed(seed)
    return [random.choice(_BUILDERS)() for _ in range(n)]


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--train", default="sia-lab/posttrain/data/device_actions_train.json")
    parser.add_argument("--val", default="sia-lab/posttrain/data/device_actions_val.json")
    parser.add_argument("--val-size", type=int, default=100)
    args = parser.parse_args(argv)

    out_train = Path(args.train)
    out_val = Path(args.val)
    out_train.parent.mkdir(parents=True, exist_ok=True)

    all_samples = generate(args.n + args.val_size)
    train = all_samples[args.val_size:]
    val = all_samples[:args.val_size]

    out_train.write_text(json.dumps(train, indent=2, ensure_ascii=False))
    out_val.write_text(json.dumps(val, indent=2, ensure_ascii=False))
    print(f"wrote {len(train)} train + {len(val)} val samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
