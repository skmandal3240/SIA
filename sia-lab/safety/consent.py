#!/usr/bin/env python3
"""Consent and retention metadata for DPDP compliance.

Every user-derived record in SIA must carry:
  - data_classification: PII | sensitive | operational
  - consent: granted | denied | withdrawn
  - retention_days: TTL for this record (0 = indefinite)
  - consent_timestamp: when consent was given/withdrawn

This module provides a ConsentRecord dataclass and helpers to enforce
retention and consent at read/write time (TRD-SR-004/005).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConsentRecord:
    """DPDP consent + retention metadata attached to every user record."""
    data_classification: str = "operational"  # PII | sensitive | operational
    consent: str = "granted"  # granted | denied | withdrawn
    retention_days: int = 0  # 0 = indefinite
    consent_timestamp: float = field(default_factory=time.time)
    user_id: str = "default"

    def is_expired(self, now: float | None = None) -> bool:
        if self.retention_days <= 0:
            return False
        now = time.time() if now is None else now
        age_seconds = now - self.consent_timestamp
        return age_seconds > self.retention_days * 86400

    def is_accessible(self) -> bool:
        return self.consent == "granted"

    def withdraw(self) -> None:
        self.consent = "withdrawn"

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_classification": self.data_classification,
            "consent": self.consent,
            "retention_days": self.retention_days,
            "consent_timestamp": self.consent_timestamp,
            "user_id": self.user_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConsentRecord":
        return cls(
            data_classification=data.get("data_classification", "operational"),
            consent=data.get("consent", "granted"),
            retention_days=data.get("retention_days", 0),
            consent_timestamp=data.get("consent_timestamp", time.time()),
            user_id=data.get("user_id", "default"),
        )


def check_access(consent: ConsentRecord, now: float | None = None) -> bool:
    """Gate: return True if the record can be read (consent granted + not expired)."""
    if not consent.is_accessible():
        return False
    if consent.is_expired(now):
        return False
    return True


def demo() -> None:
    # Active consent
    c1 = ConsentRecord(data_classification="PII", retention_days=30)
    assert check_access(c1)
    assert c1.is_accessible()

    # Expired
    c2 = ConsentRecord(data_classification="PII", retention_days=1,
                       consent_timestamp=time.time() - 2 * 86400)
    assert not check_access(c2)
    assert c2.is_expired()

    # Withdrawn
    c3 = ConsentRecord(data_classification="sensitive")
    c3.withdraw()
    assert not check_access(c3)

    # Round-trip
    c4 = ConsentRecord.from_dict(c1.to_dict())
    assert c4.data_classification == c1.data_classification
    assert c4.consent == c1.consent

    print("consent/retention demo passed")


if __name__ == "__main__":
    demo()