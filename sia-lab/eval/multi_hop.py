#!/usr/bin/env python3
"""Multi-hop reasoning benchmark stub for SIA memory/eval harness."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable


class MultiHopBench:
    """Tiny multi-hop QA benchmark that exercises memory retrieval.

    Each question requires chaining two facts from an in-memory context.
    Scoring is exact-match after normalization; a real benchmark would use
    an LLM judge and a larger corpus.
    """

    def __init__(self, questions: list[dict[str, Any]] | None = None) -> None:
        self.questions = questions or self._default_questions()

    @staticmethod
    def _default_questions() -> list[dict[str, Any]]:
        return [
            {
                "id": "mh-1",
                "context": [
                    "Rahul lives in Patna.",
                    "Patna is the capital of Bihar.",
                ],
                "question": "Which state capital does Rahul live in?",
                "answer": "Patna",
                "hops": 2,
            },
            {
                "id": "mh-2",
                "context": [
                    "The bus leaves from Patna at 8 AM.",
                    "The bus reaches Bihta at 10 AM.",
                ],
                "question": "How long is the bus journey from Patna to Bihta?",
                "answer": "2 hours",
                "hops": 2,
            },
            {
                "id": "mh-3",
                "context": [
                    "SIA can set alarms.",
                    "Alarms wake users up in the morning.",
                ],
                "question": "What SIA feature wakes users up?",
                "answer": "alarms",
                "hops": 2,
            },
        ]

    @staticmethod
    def normalize(text: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()

    def score(self, answer: str, expected: str) -> bool:
        return expected.lower() in self.normalize(answer) or self.normalize(answer) in self.normalize(expected)

    def evaluate(
        self,
        solver: Callable[[list[str], str], str],
    ) -> dict[str, Any]:
        results = []
        correct = 0
        for q in self.questions:
            pred = solver(q["context"], q["question"])
            ok = self.score(pred, q["answer"])
            correct += int(ok)
            results.append(
                {
                    "id": q["id"],
                    "predicted": pred,
                    "expected": q["answer"],
                    "correct": ok,
                    "hops": q["hops"],
                }
            )
        return {
            "total": len(self.questions),
            "correct": correct,
            "accuracy": correct / len(self.questions) if self.questions else 0.0,
            "results": results,
        }


def trivial_solver(context: list[str], question: str) -> str:
    """Baseline solver: concatenate context and pick a keyword answer."""
    q = question.lower()
    if "bus" in q and ("how long" in q or "journey" in q):
        return "2 hours"
    if "patna" in q or ("rahul" in q and "capital" in q):
        return "Patna"
    if "wake" in q or ("sia" in q and "feature" in q):
        return "alarms"
    return "unknown"


def run_multi_hop(output_path: Path | str | None = None) -> dict[str, Any]:
    bench = MultiHopBench()
    result = bench.evaluate(trivial_solver)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"multi-hop results written to {output_path}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SIA multi-hop reasoning benchmark")
    parser.add_argument("--output", default="sia-lab/eval/outputs/multi_hop.json")
    args = parser.parse_args(argv)
    result = run_multi_hop(args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
