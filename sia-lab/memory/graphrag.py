#!/usr/bin/env python3
"""GraphRAG semantic store stub for SIA memory."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class GraphRAGStub:
    """Stub for a semantic graph memory layer.

    Real implementation would build a knowledge graph from chunked memories,
    embed entities/relations, and run multi-hop retrieval. This version stores
    (subject, predicate, object) triples and exposes the same API surface so
    callers can integrate without waiting for a full graph engine.
    """

    def __init__(self) -> None:
        self._triples: list[tuple[str, str, str]] = []

    def add_triple(self, subject: str, predicate: str, obj: str) -> None:
        self._triples.append((subject, predicate, obj))

    def query(self, entity: str) -> list[dict[str, str]]:
        """Return triples where entity appears as subject or object."""
        return [
            {"subject": s, "predicate": p, "object": o}
            for s, p, o in self._triples
            if entity.lower() in (s.lower(), o.lower())
        ]

    def multi_hop(self, start: str, hops: int = 2) -> list[list[dict[str, str]]]:
        """Return simple paths of length <= hops from start."""
        paths: list[list[dict[str, str]]] = []
        for s, p, o in self._triples:
            if start.lower() in s.lower():
                paths.append([{"subject": s, "predicate": p, "object": o}])
                if hops > 1:
                    for s2, p2, o2 in self._triples:
                        if o.lower() in s2.lower() and (s2, p2, o2) != (s, p, o):
                            paths.append([
                                {"subject": s, "predicate": p, "object": o},
                                {"subject": s2, "predicate": p2, "object": o2},
                            ])
        return paths

    def dump(self) -> dict[str, Any]:
        return {
            "schema": "sia-graphrag-stub/v0",
            "triples": [
                {"subject": s, "predicate": p, "object": o}
                for s, p, o in self._triples
            ],
        }

    def load(self, data: dict[str, Any]) -> None:
        self._triples = [
            (t["subject"], t["predicate"], t["object"])
            for t in data.get("triples", [])
        ]

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.dump(), indent=2, ensure_ascii=False))

    @classmethod
    def from_file(cls, path: Path | str) -> "GraphRAGStub":
        stub = cls()
        stub.load(json.loads(Path(path).read_text()))
        return stub


def demo() -> None:
    g = GraphRAGStub()
    g.add_triple("SIA", "has_feature", "alarm")
    g.add_triple("alarm", "used_for", "wake_up")
    q = g.query("alarm")
    assert len(q) == 2, q
    hops = g.multi_hop("SIA", hops=2)
    assert len(hops) == 2, hops
    print("GraphRAGStub demo passed:", len(hops), "paths")


if __name__ == "__main__":
    demo()
