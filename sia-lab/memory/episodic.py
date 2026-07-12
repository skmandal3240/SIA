#!/usr/bin/env python3
"""Episodic memory store with TTL and vector-similarity search stub."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from safety.crypto import DeviceKeystore


@dataclass
class Episode:
    """One memory episode."""

    key: str
    content: str
    created_at: float
    ttl_seconds: float
    embedding: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_expired(self, now: float | None = None) -> bool:
        if self.ttl_seconds <= 0:
            return False
        now = time.time() if now is None else now
        return now - self.created_at > self.ttl_seconds


class EpisodicStore:
    """Key-value episodic memory with optional TTL and cosine-similarity stub."""

    def __init__(self, embed_fn: Callable[[str], list[float]] | None = None) -> None:
        self._episodes: dict[str, Episode] = {}
        # ponytail: stub embedding is random-ish hash vector; swap for real model.
        self._embed_fn = embed_fn or self._hash_embedding

    @staticmethod
    def _hash_embedding(text: str, dim: int = 16) -> list[float]:
        """Return a deterministic but low-quality embedding vector."""
        h = hashlib.sha256(text.encode()).digest()
        vec = []
        for i in range(dim):
            b = h[i % len(h)]
            vec.append((b / 255.0) * 2 - 1)
        return vec

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def add(self, key: str, content: str, ttl_seconds: float = 0,
            metadata: dict[str, Any] | None = None) -> Episode:
        if metadata is None:
            metadata = {}
        ep = Episode(
            key=key,
            content=content,
            created_at=time.time(),
            ttl_seconds=ttl_seconds,
            embedding=self._embed_fn(content),
            metadata=metadata,
        )
        self._episodes[key] = ep
        return ep

    def get(self, key: str, now: float | None = None) -> Episode | None:
        ep = self._episodes.get(key)
        if ep is None:
            return None
        if ep.is_expired(now):
            del self._episodes[key]
            return None
        return ep

    def expire(self) -> list[str]:
        """Remove expired entries and return removed keys."""
        now = time.time()
        expired = [k for k, ep in self._episodes.items() if ep.is_expired(now)]
        for k in expired:
            del self._episodes[k]
        return expired

    def search(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        """Return top_k episodic keys by cosine similarity to query embedding."""
        qvec = self._embed_fn(query)
        scored: list[tuple[str, float]] = []
        self.expire()
        for ep in self._episodes.values():
            score = self._cosine(qvec, ep.embedding)
            scored.append((ep.key, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def dump(self) -> list[dict[str, Any]]:
        return [
            {
                "key": ep.key,
                "content": ep.content,
                "created_at": ep.created_at,
                "ttl_seconds": ep.ttl_seconds,
                "embedding": ep.embedding,
                "metadata": ep.metadata,
            }
            for ep in self._episodes.values()
        ]

    def load(self, data: list[dict[str, Any]]) -> None:
        self._episodes.clear()
        for row in data:
            ep = Episode(
                key=row["key"],
                content=row["content"],
                created_at=row["created_at"],
                ttl_seconds=row["ttl_seconds"],
                embedding=row.get("embedding", []),
                metadata=row.get("metadata", {}),
            )
            self._episodes[ep.key] = ep

    def save(self, path: Path | str, keystore: "DeviceKeystore | None" = None) -> None:
        """Persist to disk, encrypted at rest through the device keystore."""
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from safety.crypto import DeviceKeystore

        ks = keystore or DeviceKeystore()
        payload = json.dumps(self.dump(), ensure_ascii=False).encode("utf-8")
        Path(path).write_bytes(ks.encrypt(payload))

    @classmethod
    def from_file(cls, path: Path | str, keystore: "DeviceKeystore | None" = None) -> "EpisodicStore":
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from safety.crypto import DeviceKeystore

        ks = keystore or DeviceKeystore()
        store = cls()
        payload = ks.decrypt(Path(path).read_bytes())
        store.load(json.loads(payload))
        return store


def demo() -> None:
    store = EpisodicStore()
    store.add("alarm", "Set alarm for 7 AM", ttl_seconds=3600, metadata={"tool": "set_alarm"})
    store.add("weather", "Asked about Patna weather", ttl_seconds=0.01)
    time.sleep(0.05)
    assert store.get("weather") is None, "TTL expiry should remove weather"
    assert store.get("alarm") is not None
    results = store.search("alarm clock morning")
    assert results[0][0] == "alarm", results

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "episodic.enc"
        store.save(p)
        assert b"alarm" not in p.read_bytes(), "save() must not write plaintext content"
        restored = EpisodicStore.from_file(p)
        assert restored.get("alarm") is not None

    print("EpisodicStore demo passed:", results[:2])


if __name__ == "__main__":
    demo()
