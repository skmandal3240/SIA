#!/usr/bin/env python3
"""DPDP audit logger for SIA.

Records every cloud egress and device action with timestamp, classification,
and retention metadata. Stored locally on device; supports user-triggered
erasure.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class AuditEvent:
    timestamp: float
    event_type: str  # egress | action | access | erasure
    description: str
    classification: str  # PII | sensitive | operational
    allowed: bool
    retention_days: int


class AuditLog:
    """Append-only local audit log."""

    def __init__(self, path: Path | str = "/tmp/sia_audit.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: AuditEvent) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")

    def record_egress(self, host: str, port: int, allowed: bool) -> None:
        self.log(AuditEvent(
            timestamp=time.time(),
            event_type="egress",
            description=f"{host}:{port}",
            classification="operational" if allowed else "sensitive",
            allowed=allowed,
            retention_days=30 if allowed else 90,
        ))

    def record_action(self, action: str, classification: str = "operational") -> None:
        self.log(AuditEvent(
            timestamp=time.time(),
            event_type="action",
            description=action,
            classification=classification,
            allowed=True,
            retention_days=7,
        ))

    def erase(self) -> None:
        self.path.write_text("")
        self.log(AuditEvent(
            timestamp=time.time(),
            event_type="erasure",
            description="user-triggered erasure",
            classification="operational",
            allowed=True,
            retention_days=0,
        ))

    def read(self, limit: int = 100) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text().strip().splitlines()
        return [json.loads(ln) for ln in lines[-limit:]]


def main() -> int:
    log = AuditLog("/tmp/sia_audit_demo.jsonl")
    log.record_egress("api.example.com", 443, allowed=False)
    log.record_action("set_alarm")
    events = log.read()
    print(f"audit events: {len(events)}")
    log.erase()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
