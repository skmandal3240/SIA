#!/usr/bin/env python3
"""P1 action-adapter SFT wrapper: thin shim over train_gcp.py.

Original sft.py had stale TRL call sites. Rather than maintain two trainers,
this delegates to the GCP trainer, which works on any GPU (not just GCP).
"""
from __future__ import annotations

import sys

from train_gcp import main

if __name__ == "__main__":
    raise SystemExit(main())
