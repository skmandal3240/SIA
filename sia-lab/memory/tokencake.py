#!/usr/bin/env python3
"""TokenCake working memory: a fixed-size ring buffer keyed by token budget."""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass
class TokenCakeSlice:
    """One layer of the working-memory cake."""

    role: str  # e.g. system, user, assistant, tool_result
    content: str
    tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class TokenCake:
    """Working memory bounded by a token budget, kept in eviction order.

    SIA keeps the most-recent and highest-priority slices; overflow drops the
    oldest non-pinned slices. This is intentionally a stdlib-only sketch: real
    token counts need the model tokenizer and are left to callers.
    """

    def __init__(self, budget: int = 4096, reserve: int = 256) -> None:
        if budget <= 0:
            raise ValueError("budget must be positive")
        if reserve >= budget:
            raise ValueError("reserve must be smaller than budget")
        self.budget = budget
        self.reserve = reserve
        self._slices: deque[TokenCakeSlice] = deque()
        self._pinned: set[int] = set()

    def add(self, role: str, content: str, tokens: int = 0, *, pin: bool = False,
            metadata: dict[str, Any] | None = None) -> TokenCakeSlice:
        """Append a slice, evicting old non-pinned slices if over budget."""
        if tokens < 0:
            raise ValueError("tokens cannot be negative")
        if metadata is None:
            metadata = {}
        item_id = id(content)  # stable enough for this stub
        item_id += len(self._slices)
        # Recalculate item_id above is not unique across deletions; keep it simple
        # and use the position at insertion time for pinning.
        idx = len(self._slices)
        slice_ = TokenCakeSlice(role, content, tokens, metadata)
        self._slices.append(slice_)
        if pin:
            self._pinned.add(idx)
        self._evict()
        return slice_

    def _evict(self) -> None:
        """Drop oldest non-pinned slices until used tokens fit the budget."""
        used = sum(s.tokens for s in self._slices)
        limit = self.budget - self.reserve
        while used > limit and self._slices:
            for i, s in enumerate(self._slices):
                idx = i
                if idx not in self._pinned:
                    self._slices.remove(s)
                    used -= s.tokens
                    break
            else:
                # everything pinned; cannot evict
                break

    @property
    def used_tokens(self) -> int:
        return sum(s.tokens for s in self._slices)

    @property
    def free_tokens(self) -> int:
        return self.budget - self.used_tokens

    def to_messages(self) -> list[dict[str, str]]:
        """Return memory contents as chat-style messages."""
        return [{"role": s.role, "content": s.content} for s in self._slices]

    def dump(self) -> list[dict[str, Any]]:
        """Serialize state to JSON-friendly list."""
        return [
            {
                "role": s.role,
                "content": s.content,
                "tokens": s.tokens,
                "metadata": s.metadata,
            }
            for s in self._slices
        ]

    def load(self, data: Sequence[dict[str, Any]]) -> None:
        """Restore from dump output."""
        self._slices.clear()
        self._pinned.clear()
        for i, row in enumerate(data):
            s = TokenCakeSlice(
                role=row["role"],
                content=row["content"],
                tokens=row.get("tokens", 0),
                metadata=row.get("metadata", {}),
            )
            self._slices.append(s)
            if row.get("pinned"):
                self._pinned.add(i)
        self._evict()

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.dump(), indent=2, ensure_ascii=False))

    @classmethod
    def from_file(cls, path: Path | str, budget: int = 4096, reserve: int = 256) -> "TokenCake":
        cake = cls(budget=budget, reserve=reserve)
        cake.load(json.loads(Path(path).read_text()))
        return cake


def demo() -> None:
    cake = TokenCake(budget=100, reserve=10)
    cake.add("system", "You are SIA.", tokens=20, pin=True)
    cake.add("user", "What's the weather?", tokens=15)
    cake.add("assistant", "Sunny, 32C.", tokens=20)
    cake.add("user", "Set alarm.", tokens=60)  # should evict earlier user
    assert cake.used_tokens <= cake.budget - cake.reserve, cake.used_tokens
    msgs = cake.to_messages()
    roles = [m["role"] for m in msgs]
    assert "system" in roles
    print("TokenCake demo passed:", roles, "used", cake.used_tokens)


if __name__ == "__main__":
    demo()
