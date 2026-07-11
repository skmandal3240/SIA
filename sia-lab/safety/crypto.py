#!/usr/bin/env python3
"""Device-local encryption-at-rest for SIA's persisted state.

SIA never sends raw on-device data to the network (see safety/privacy.py).
This module protects it *at rest* on the device itself: every store that
persists to disk (audit log, episodic memory, working memory, graph memory)
encrypts through a single shared DeviceKeystore. Deleting that one key file
(``shred()``) makes every store encrypted with it permanently unreadable --
this is the primitive a factory reset relies on to make personalized data
unrecoverable without a slow overwrite-every-byte wipe.
"""
from __future__ import annotations

import os
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken  # noqa: F401  (re-exported for callers)

DEFAULT_KEY_PATH = Path(os.environ.get("SIA_HOME", str(Path.home() / ".sia"))) / "device.key"


class DeviceKeystore:
    """Holds (and lazily generates) the symmetric key that protects every
    SIA store persisted to this device."""

    def __init__(self, key_path: Path | str = DEFAULT_KEY_PATH) -> None:
        self.key_path = Path(key_path)
        self._fernet: Fernet | None = None

    def _load_or_create_key(self) -> bytes:
        if self.key_path.exists():
            return self.key_path.read_bytes()
        key = Fernet.generate_key()
        self.key_path.parent.mkdir(parents=True, exist_ok=True)
        self.key_path.write_bytes(key)
        # Owner read/write only -- this file is the single point of failure
        # for every encrypted store on the device.
        os.chmod(self.key_path, 0o600)
        return key

    @property
    def fernet(self) -> Fernet:
        if self._fernet is None:
            self._fernet = Fernet(self._load_or_create_key())
        return self._fernet

    def encrypt(self, plaintext: bytes) -> bytes:
        return self.fernet.encrypt(plaintext)

    def decrypt(self, token: bytes) -> bytes:
        return self.fernet.decrypt(token)

    def shred(self) -> bool:
        """Irrecoverably destroy the device key.

        Every store encrypted with this key becomes permanently unreadable
        ciphertext. This is the "factory reset" primitive: it is fast (one
        file unlink) and does not require overwriting the -- potentially
        large -- encrypted payloads themselves.
        """
        self._fernet = None
        if self.key_path.exists():
            self.key_path.unlink()
            return True
        return False


def demo() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        key_path = Path(tmp) / "device.key"
        ks = DeviceKeystore(key_path)
        token = ks.encrypt(b"sensitive on-device content")
        assert token != b"sensitive on-device content"
        assert ks.decrypt(token) == b"sensitive on-device content"

        # A second keystore instance pointed at the same file transparently
        # shares the persisted key (simulates a process restart before shred).
        ks2 = DeviceKeystore(key_path)
        assert ks2.decrypt(token) == b"sensitive on-device content"

        ks.shred()

        # A *fresh* keystore instance (nothing cached in memory) created
        # after shred gets a brand-new key and cannot read data encrypted
        # under the destroyed one -- this is the guarantee a factory reset
        # relies on. (An already-running process holding the old key in
        # memory, like ks2 above, can still decrypt until it restarts --
        # shred revokes future access, not already-loaded key material.)
        ks3 = DeviceKeystore(key_path)
        try:
            ks3.decrypt(token)
            raise AssertionError("decrypt should fail after shred + key rotation")
        except InvalidToken:
            pass

    print("DeviceKeystore demo passed: encrypt/decrypt round-trip + shred makes data unrecoverable")


if __name__ == "__main__":
    demo()
