#!/usr/bin/env python3
"""Two-speed router for SIA: fast path vs deep path.

The router inspects the query and the governor budget and decides:
  - fast: use the cheap local model / heuristic path (single-hop, well-known)
  - deep: use the recurrent-depth SIR core (multi-hop, planning, math-chain)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class RouteDecision:
    path: str  # "fast" | "deep"
    budget: dict
    reason: str


class SIRRouter:
    """Route queries to fast or deep path.

    Default logic: escalate to deep when the query contains multi-hop
    signals (why, how long, which, compare) or when the user explicitly asks
    for a plan/chain. Otherwise use fast path.
    """

    DEEP_SIGNALS = {
        "why", "how", "which", "compare", "difference", "plan", "steps",
        "chain", "longest", "shortest", "if", "then", "because", "and",
    }

    def __init__(self, fast_solver: Callable[[list[str], str], str] | None = None) -> None:
        self.fast_solver = fast_solver

    def route(self, query: str, context: list[str] | None = None, thermal_budget: dict | None = None) -> RouteDecision:
        q = query.lower()
        thermal = thermal_budget or {}
        hot = thermal.get("hot", False)
        deep_allowed = thermal.get("deep_allowed", True)

        if hot or not deep_allowed:
            return RouteDecision(
                path="fast",
                budget={"max_loops": 1, "act_max_steps": 1},
                reason="thermal budget forces fast path",
            )

        score = sum(1 for s in self.DEEP_SIGNALS if s in q)
        # Multi-hop context length also nudges toward deep.
        if context and len(context) >= 2:
            score += 1

        if score >= 1:
            return RouteDecision(
                path="deep",
                budget={"max_loops": 8, "act_max_steps": 6},
                reason=f"deep signals={score}",
            )

        return RouteDecision(
            path="fast",
            budget={"max_loops": 1, "act_max_steps": 1},
            reason="no deep signal",
        )


def main() -> int:
    router = SIRRouter()
    for q in ["Set alarm for 7", "Why is Patna the capital of Bihar?", "Plan my morning routine"]:
        d = router.route(q)
        print(f"{q!r} -> {d.path} ({d.reason})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
