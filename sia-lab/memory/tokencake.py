#!/usr/bin/env python3
"""TokenCake working memory: a fixed-size ring buffer keyed by token budget."""
from __future__ import annotations

import json
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


@dataclass
class TokenCakeSlice:
    """One layer of the working-memory cake."""

    role: str  # e.g. system, user, assistant, tool_result
    content: str
    tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    pinned: bool = False


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

    def add(self, role: str, content: str, tokens: int = 0, *, pin: bool = False,
            metadata: dict[str, Any] | None = None) -> TokenCakeSlice:
        """Append a slice, evicting old non-pinned slices if over budget."""
        if tokens < 0:
            raise ValueError("tokens cannot be negative")
        if metadata is None:
            metadata = {}
        slice_ = TokenCakeSlice(role, content, tokens, metadata, pinned=pin)
        self._slices.append(slice_)
        self._evict()
        return slice_

    def _evict(self) -> None:
        """Drop oldest non-pinned slices until used tokens fit the budget."""
        used = sum(s.tokens for s in self._slices)
        limit = self.budget - self.reserve
        while used > limit and self._slices:
            # Pin state travels with the slice, so eviction stays correct even
            # after earlier slices have been dropped and positions shifted.
            for s in self._slices:
                if not s.pinned:
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
                "pinned": s.pinned,
            }
            for s in self._slices
        ]

    def load(self, data: Sequence[dict[str, Any]]) -> None:
        """Restore from dump output."""
        self._slices.clear()
        for row in data:
            s = TokenCakeSlice(
                role=row["role"],
                content=row["content"],
                tokens=row.get("tokens", 0),
                metadata=row.get("metadata", {}),
                pinned=bool(row.get("pinned", False)),
            )
            self._slices.append(s)
        self._evict()

    def save(self, path: Path | str, keystore: "DeviceKeystore | None" = None) -> None:
        """Persist to disk, encrypted at rest through the device keystore."""
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from safety.crypto import DeviceKeystore

        ks = keystore or DeviceKeystore()
        payload = json.dumps(self.dump(), ensure_ascii=False).encode("utf-8")
        Path(path).write_bytes(ks.encrypt(payload))

    @classmethod
    def from_file(cls, path: Path | str, budget: int = 4096, reserve: int = 256,
                   keystore: "DeviceKeystore | None" = None) -> "TokenCake":
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from safety.crypto import DeviceKeystore

        ks = keystore or DeviceKeystore()
        cake = cls(budget=budget, reserve=reserve)
        payload = ks.decrypt(Path(path).read_bytes())
        cake.load(json.loads(payload))
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

    # A pin on a non-first slice must survive eviction even after earlier
    # slices are dropped and positions shift (regression for index-based pins).
    cake2 = TokenCake(budget=100, reserve=10)
    cake2.add("user", "throwaway one", tokens=30)
    cake2.add("tool_result", "IMPORTANT fact", tokens=20, pin=True)
    cake2.add("user", "throwaway two", tokens=40)
    cake2.add("user", "throwaway three", tokens=40)  # forces eviction of unpinned
    contents = [m["content"] for m in cake2.to_messages()]
    assert "IMPORTANT fact" in contents, contents
    # Round-trip the pin through dump/load.
    restored = TokenCake(budget=100, reserve=10)
    restored.load(cake2.dump())
    assert any(s.pinned and s.content == "IMPORTANT fact" for s in restored._slices)

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "tokencake.enc"
        cake2.save(p)
        assert b"IMPORTANT fact" not in p.read_bytes(), "save() must not write plaintext content"
        restored2 = TokenCake.from_file(p, budget=100, reserve=10)
        restored2_contents = [m["content"] for m in restored2.to_messages()]
        assert "IMPORTANT fact" in restored2_contents, restored2_contents

    print("TokenCake pin demo passed:", contents)


if __name__ == "__main__":
    demo()
