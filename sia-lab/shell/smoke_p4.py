#!/usr/bin/env python3
"""P4 smoke test: memory (TokenCake + GraphRAG + EpisodicStore) drives shell loop.

The shell loop now:
  1. Loads recent TokenCake turns as context.
  2. Recalls graph facts about entities in the transcript.
  3. Writes the current turn back into TokenCake as an episode.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sia-lab"))

from shell.loop import ShellLoop
from shell.capture import CaptureStub
from shell.stt import StreamingSTTStub
from memory import TokenCake, EpisodicStore


def main() -> int:
    cake = TokenCake(budget=256, reserve=16)
    cake.add("user", "My name is Saurabh.", tokens=20)
    cake.add("assistant", "Hello Saurabh.", tokens=15)

    episodes = EpisodicStore()

    def reason(shot: dict, transcript: str, context: str) -> tuple[str, list[dict]]:
        # ponytail: show that memory context reached the reasoner.
        response = f"SIA sees {shot['width']}x{shot['height']}. Context: {context}. Transcript: {transcript} [POINT:100,200:ok]"
        return response, []

    loop = ShellLoop(
        capture=CaptureStub(),
        stt=StreamingSTTStub(["What is my name?"]),
        reason=reason,
        memory=cake,
    )

    turn = loop.run_once([b"What is my name?"])
    assert turn.transcript == "What is my name?", turn.transcript
    print("transcript:", turn.transcript)
    print("context:", turn.context)
    print("response:", turn.response)

    # Episodic write
    episodes.add(turn.transcript, turn.response)
    print("episodes:", len(episodes._episodes))

    # Memory grew
    assert len(cake.to_messages()) >= 4
    assert "Saurabh" in turn.context or "Saurabh" in turn.response
    print("P4 smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
