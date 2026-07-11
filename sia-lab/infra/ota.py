#!/usr/bin/env python3
"""OTA adapter manifest and updater.

Holds metadata for hot-swap LoRA adapters (name, version, checksum, download
URL) and applies updates: download, verify, and atomically install. A
download that fails checksum or size verification is discarded without ever
touching the existing installed adapter -- an in-progress or bad update can
never leave a device with a broken (or silently corrupted) adapter.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
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


class OTAUpdateError(Exception):
    """Raised when an OTA adapter download or verification fails."""


class OTAUpdater:
    """Downloads, verifies, and installs adapters described by manifest entries.

    Download and install are strictly separated by checksum/size
    verification: a partially downloaded or corrupted file is written to a
    scratch path and only atomically renamed into install_dir once it
    checks out, so a bad or interrupted update never clobbers a working
    installed adapter.
    """

    def __init__(self, install_dir: Path | str, chunk_size: int = 1 << 20) -> None:
        self.install_dir = Path(install_dir)
        self.chunk_size = chunk_size

    def _record_path(self, name: str) -> Path:
        return self.install_dir / f"{name}.installed.json"

    def installed_version(self, name: str) -> str | None:
        record_path = self._record_path(name)
        if not record_path.exists():
            return None
        return json.loads(record_path.read_text()).get("version")

    def needs_update(self, entry: AdapterManifestEntry) -> bool:
        return self.installed_version(entry.name) != entry.version

    def apply(self, entry: AdapterManifestEntry) -> Path:
        """Fetch entry.url, verify it, and install it. Returns the installed path.

        Raises OTAUpdateError -- leaving any existing install untouched --
        if the download fails, or the checksum/size don't match the manifest.
        """
        self.install_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.install_dir / f".{entry.name}-{entry.version}.download"
        final_path = self.install_dir / f"{entry.name}-{entry.version}.adapter"

        digest = hashlib.sha256()
        size = 0
        try:
            with urllib.request.urlopen(entry.url, timeout=30) as resp, tmp_path.open("wb") as f:
                while True:
                    chunk = resp.read(self.chunk_size)
                    if not chunk:
                        break
                    digest.update(chunk)
                    size += len(chunk)
                    f.write(chunk)
        except OSError as e:
            tmp_path.unlink(missing_ok=True)
            raise OTAUpdateError(f"download failed for {entry.name} {entry.version}: {e}") from e

        actual_checksum = digest.hexdigest()
        if entry.checksum_sha256 and actual_checksum != entry.checksum_sha256:
            tmp_path.unlink(missing_ok=True)
            raise OTAUpdateError(
                f"checksum mismatch for {entry.name} {entry.version}: "
                f"expected {entry.checksum_sha256}, got {actual_checksum}"
            )
        if entry.size_bytes and size != entry.size_bytes:
            tmp_path.unlink(missing_ok=True)
            raise OTAUpdateError(
                f"size mismatch for {entry.name} {entry.version}: "
                f"expected {entry.size_bytes} bytes, got {size}"
            )

        tmp_path.replace(final_path)  # atomic rename on the same filesystem
        self._record_path(entry.name).write_text(json.dumps(
            {**asdict(entry), "installed_path": str(final_path)}, indent=2))
        return final_path


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

    # Exercise the updater end-to-end against a local fake adapter blob --
    # there is no real hosted adapter yet, but the download/verify/install
    # mechanism itself is real and fully testable via a file:// URL.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        fake_adapter = tmp / "device-actions-v1.0.0.bin"
        fake_adapter.write_bytes(b"fake adapter weights v1.0.0")
        checksum = hashlib.sha256(fake_adapter.read_bytes()).hexdigest()
        entry = AdapterManifestEntry(
            name="device-actions",
            version="v1.0.0",
            base_model="liquidai/LFM2.5-1.2B-Instruct",
            checksum_sha256=checksum,
            url=fake_adapter.as_uri(),
            size_bytes=fake_adapter.stat().st_size,
        )

        install_dir = tmp / "installed"
        updater = OTAUpdater(install_dir)
        assert updater.needs_update(entry)
        installed_path = updater.apply(entry)
        assert installed_path.read_bytes() == fake_adapter.read_bytes()
        assert not updater.needs_update(entry)
        print(f"OTA update applied: {installed_path.name}")

        # A corrupted/mismatched download must be rejected without touching
        # the already-installed adapter.
        bad_entry = AdapterManifestEntry(
            name="device-actions",
            version="v1.1.0",
            base_model="liquidai/LFM2.5-1.2B-Instruct",
            checksum_sha256="0" * 64,  # wrong on purpose
            url=fake_adapter.as_uri(),
            size_bytes=fake_adapter.stat().st_size,
        )
        try:
            updater.apply(bad_entry)
            raise AssertionError("apply() should reject a checksum mismatch")
        except OTAUpdateError:
            pass
        assert updater.installed_version("device-actions") == "v1.0.0", \
            "a failed update must not disturb the existing install"
        assert not (install_dir / ".device-actions-v1.1.0.download").exists(), \
            "a failed download must not leave partial files behind"
        print("OTA update correctly rejected a corrupted download")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
