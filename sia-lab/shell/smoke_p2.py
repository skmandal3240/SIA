#!/usr/bin/env python3
"""P2 shell smoke test: see screen, reason with local model, dispatch, speak.

This script runs one full SIA Shell turn using:
  - real screen capture via mss (falls back to stub if mss unavailable)
  - local Ollama reasoner (sia-p0) if available
  - the shared dispatcher for action tags + tool calls
  - stub STT/TTS if no real engines are installed

Run from repo root: python3 sia-lab/shell/smoke_p2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sia-lab"))

from shell.capture import default_capture
from shell.stt import StreamingSTTStub
from shell.tts import StreamingTTSStub
from shell.dispatcher import Dispatcher
from shell.loop import ShellLoop
from shell.reason_ollama import reason_with_ollama


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    model_tag = argv[0] if argv else "sia-p0"  # ponytail: p0 substrate is the verified model; pass a tag to override
    capture = default_capture()
    transcript = "set an alarm for seven am and point to the submit button"

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

    # ponytail: pass if the loop ran end-to-end; generation quality depends on Ollama.
    return 0 if turn.screenshot and turn.transcript == transcript else 1


if __name__ == "__main__":
    raise SystemExit(main())
