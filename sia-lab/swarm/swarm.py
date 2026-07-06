#!/usr/bin/env python3
"""Minimal P5 swarm runtime: N=2 operational loop.

Ponytail: real swarm would use message passing over a network. This file
implements the in-process operational API: nodes share a task, produce
individual responses, and a simple consensus reduces them to one answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class SwarmTask:
    task_id: str
    prompt: str
    context: list[str] = field(default_factory=list)


@dataclass
class SwarmNode:
    node_id: str
    persona: str
    solver: Callable[[str, list[str]], str]


class SwarmRuntime:
    """Operational swarm: N nodes solve a task, then a reducer picks consensus."""

    def __init__(self, nodes: list[SwarmNode], reducer: Callable[[list[str]], str] | None = None) -> None:
        self.nodes = nodes
        self.reducer = reducer or self._majority_vote

    @staticmethod
    def _majority_vote(answers: list[str]) -> str:
        # ponytail: exact-match majority; fallback to first answer.
        counts: dict[str, int] = {}
        for a in answers:
            counts[a] = counts.get(a, 0) + 1
        if not counts:
            return ""
        return max(counts.items(), key=lambda kv: kv[1])[0]

    def run(self, task: SwarmTask) -> dict:
        answers = []
        for node in self.nodes:
            answers.append(node.solver(task.prompt, task.context))
        consensus = self.reducer(answers)
        return {
            "task_id": task.task_id,
            "consensus": consensus,
            "answers": {node.node_id: ans for node, ans in zip(self.nodes, answers)},
            "n_nodes": len(self.nodes),
        }


def demo() -> int:
    def solver_a(prompt: str, ctx: list[str]) -> str:
        if "capital" in prompt.lower():
            return "Patna"
        return "yes"

    def solver_b(prompt: str, ctx: list[str]) -> str:
        if "capital" in prompt.lower():
            return "Patna"
        return "yes"

    nodes = [
        SwarmNode("node-1", "precise", solver_a),
        SwarmNode("node-2", "cautious", solver_b),
    ]
    swarm = SwarmRuntime(nodes)
    task = SwarmTask("t1", "Which state capital does Rahul live in?", ["Rahul lives in Patna.", "Patna is the capital of Bihar."])
    result = swarm.run(task)
    print(result)
    assert result["consensus"] == "Patna"
    return 0


if __name__ == "__main__":
    raise SystemExit(demo())
