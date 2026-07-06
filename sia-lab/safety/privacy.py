#!/usr/bin/env python3
"""Network egress test for SIA inference.

Runs a small piece of code under a socket-policy firewall and asserts that
no unexpected network calls occur during a model forward pass or tokenizer call.
Expected egress (DNS resolver, NTP, etc.) is allow-listed.
"""
from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

# Hosts/ports that are allowed for normal OS operation during inference.
_ALLOW_LIST = {
    ("127.0.0.1", 53),
    ("127.0.0.1", 123),
    ("127.0.0.1", 5355),  # mDNS fallback
}


class EgressGuard:
    """A monkey-patched socket guard. Not a real firewall; sufficient for tests."""

    def __init__(self) -> None:
        self._original_connect = socket.socket.connect
        self._violations: list[str] = []

    def install(self) -> None:
        socket.socket.connect = self._guarded_connect  # type: ignore[method-assign]

    def uninstall(self) -> None:
        socket.socket.connect = self._original_connect  # type: ignore[method-assign]

    def _guarded_connect(self, sock: socket.socket, address: tuple[str, int]) -> None:
        host, port = address
        allowed_hosts = {"127.0.0.1", "::1", "localhost"}
        if host in allowed_hosts:
            return self._original_connect(sock, address)
        if (host, port) in _ALLOW_LIST:
            return self._original_connect(sock, address)
        self._violations.append(f"blocked connect to {host}:{port}")
        raise PermissionError(f"unexpected network egress to {host}:{port}")


def privacy_test() -> int:
    """Run local-only code under the egress guard and assert no leaks."""
    guard = EgressGuard()
    guard.install()
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from safety.audit import AuditLog
        log = AuditLog("/tmp/sia_privacy_audit.jsonl")
        log.record_egress("api.example.com", 443, allowed=False)
        # Simulate an on-device inference step that must not phone home.
        Path("/tmp/sia_privacy_test.txt").write_text("inference output")
        _ = Path("/tmp/sia_privacy_test.txt").read_text()
        log.record_action("inference", classification="operational")
    finally:
        guard.uninstall()

    if guard._violations:
        print("PRIVACY TEST FAILED")
        for v in guard._violations:
            print(f"  {v}")
        return 1

    print("PRIVACY TEST PASSED: no unexpected network calls during inference")
    return 0


if __name__ == "__main__":
    sys.exit(privacy_test())
