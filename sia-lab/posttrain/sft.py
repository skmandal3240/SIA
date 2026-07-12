#!/usr/bin/env python3
"""P1 action-adapter SFT wrapper: thin shim over train_gcp.py.

Defaults to --dry-run so `make ci` never accidentally starts GPU training.
Pass --run for a real training run on a GPU machine.
"""
from __future__ import annotations

from train_gcp import main

if __name__ == "__main__":
    # ponytail: default to dry-run. Use `python3 sft.py --run` for real training.
    import sys as _sys
    _args = _sys.argv[1:]
    if "--run" not in _args and "--dry-run" not in _args:
        _args.append("--dry-run")
    # Remove --run if present (train_gcp doesn't understand it; --dry-run absence = real run)
    _args = [a for a in _args if a != "--run"]
    raise SystemExit(main(_args))