#!/usr/bin/env python3
"""P5 end-to-end smoke: swarm + distillation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from swarm import SwarmRuntime, SwarmNode, SwarmTask
from distill import DistillationCycle


def main() -> int:
    # Operational swarm
    nodes = [
        SwarmNode("alpha", "precise", lambda p, c: "Patna"),
        SwarmNode("beta", "cautious", lambda p, c: "Patna"),
    ]
    swarm = SwarmRuntime(nodes)
    task = SwarmTask("p5-t1", "Where does Rahul live?", ["Rahul lives in Patna."])
    result = swarm.run(task)
    assert result["consensus"] == "Patna"
    print("swarm consensus:", result["consensus"])

    # Distillation cycle
    cycle = DistillationCycle(n_students=2)
    distill_result = cycle.run(
        [
            "Which state capital does Rahul live in?",
            "How long is the bus journey from Delhi to Agra?",
        ],
        "Which state capital does Rahul live in?",
    )
    assert distill_result["consensus"] == "Patna"
    print("distillation consensus:", distill_result["consensus"])
    print("P5 smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
