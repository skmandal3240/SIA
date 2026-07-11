#!/usr/bin/env python3
"""P1 action-adapter SFT wrapper: thin shim over train_gcp.py.

Original sft.py had stale TRL call sites. Rather than maintain two trainers,
this delegates to the GCP trainer, which works on any GPU (not just GCP).

CLI contract preserved from the original sft.py (and what `make smoke`
relies on): bare invocation is a safe, dependency-light dry-run; real
training is opt-in via --run, e.g.:
    python3 sia-lab/posttrain/sft.py --run --base unsloth/Llama-3.2-1B-Instruct
train_gcp.py's own default is the opposite (real run unless --dry-run is
passed), so --run is translated to the absence of --dry-run here rather than
forwarded as-is.
"""
from __future__ import annotations

import sys

from train_gcp import main

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--run" in argv:
        argv = [a for a in argv if a != "--run"]
    else:
        argv = ["--dry-run", *argv]
    raise SystemExit(main(argv))
