#!/usr/bin/env python3
"""OTA adapter manifest stub.

Holds metadata for hot-swap LoRA adapters: name, version, checksum, download
URL. Production would fetch and verify signatures; this is the repo-side
manifest format.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class AdapterManifestEntry:
    name: str
    version: str
    base_model: str
    checksum_sha256: str
    url: str
    size_bytes: int


class OTAManifest:
    """Read/write the SIA adapter OTA manifest."""

    def __init__(self, entries: list[AdapterManifestEntry] | None = None) -> None:
        self.entries = entries or []

    def add(self, entry: AdapterManifestEntry) -> None:
        self.entries.append(entry)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "sia-ota-manifest/v0",
            "adapters": [asdict(e) for e in self.entries],
        }

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path | str) -> "OTAManifest":
        data = json.loads(Path(path).read_text())
        entries = [AdapterManifestEntry(**e) for e in data.get("adapters", [])]
        return cls(entries)

    def find(self, name: str) -> AdapterManifestEntry | None:
        for e in self.entries:
            if e.name == name:
                return e
        return None


def main() -> int:
    manifest = OTAManifest()
    manifest.add(AdapterManifestEntry(
        name="device-actions",
        version="v1.0.0",
        base_model="liquidai/LFM2.5-1.2B-Instruct",
        checksum_sha256="0" * 64,
        url="https://example.com/sia/device-actions-v1.gguf",
        size_bytes=0,
    ))
    manifest.save("/tmp/sia_ota_manifest.json")
    loaded = OTAManifest.load("/tmp/sia_ota_manifest.json")
    print(loaded.find("device-actions"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
