#!/usr/bin/env python3
"""P2 shell smoke test on macOS: see screen, reason with local model, speak.

Run from repo root on a Mac with screen-recording permission granted:
    python3 sia-lab/shell/smoke_p2_macos.py

If permission is missing, the script prints the System Settings path and exits 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sia-lab"))

from shell.capture_macos import MacosCapture, ScreenCapturePermissionError
from shell.capture import CaptureStub
from shell.stt import StreamingSTTStub
from shell.tts import StreamingTTSStub
from shell.dispatcher import Dispatcher
from shell.loop import ShellLoop
from shell.reason_ollama import reason_with_ollama


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    model_tag = argv[0] if argv else "sia-p0"
    transcript = "set an alarm for seven am and point to the submit button"

    try:
        capture = MacosCapture()
    except ScreenCapturePermissionError as exc:
        print(exc)
        return 1
    except Exception as exc:
        print(f"macOS capture unavailable ({exc}); using stub")
        capture = CaptureStub()

    turn = ShellLoop(
        capture=capture,
        stt=StreamingSTTStub(transcripts=[transcript]),
        tts=StreamingTTSStub(),
        dispatcher=Dispatcher(),
        reason=lambda ss, tx, ctx: reason_with_ollama(ss, tx, context=ctx, model_tag=model_tag),
    ).run_once(audio_chunks=[b"chunk"])

    print(f"Model: {model_tag}")
    print(f"Screen: {turn.screenshot['width']}x{turn.screenshot['height']}")
    print(f"Transcript: {turn.transcript}")
    print(f"Response: {turn.response}")
    print(f"Tools: {[(t.tool, t.args) for t in turn.tools]}")
    print(f"Actions: {[(a.tag.kind, a.tag.x, a.tag.y, a.tag.label) for a in turn.actions]}")
    print(f"Audio chunks spoken: {len(turn.spoken)}")

    return 0 if turn.screenshot and turn.transcript == transcript else 1


if __name__ == "__main__":
    raise SystemExit(main())
