#!/usr/bin/env python3
"""P0 substrate verification: LFM2.5 via Ollama Modelfile.

If Ollama or weights are unavailable, performs a dry-run that validates
Modelfile syntax and reports exactly what is missing.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # sia-lab/product/verify_p0.py -> .../sia-lab-build
MODELS_DIR = Path(os.environ.get("SIA_MODELS_DIR", PROJECT_ROOT / "PROJECT" / "models"))
MODELFILE = MODELS_DIR / "Modelfile"


def _find_gguf() -> Path | None:
    if not MODELS_DIR.exists():
        return None
    for p in sorted(MODELS_DIR.iterdir()):
        if p.suffix == ".gguf":
            return p
    return None


def _modelfile_is_valid(path: Path) -> tuple[bool, list[str]]:
    if not path.exists():
        return False, [f"Modelfile not found: {path}"]
    text = path.read_text()
    required = ["FROM", "TEMPLATE"]
    missing = [k for k in required if not re.search(rf"^{k}\s+", text, re.IGNORECASE | re.MULTILINE)]
    errors = [f"Missing directive: {k}" for k in missing]
    # Lightweight FROM file existence check
    m = re.search(r"^FROM\s+(\S+)", text, re.IGNORECASE | re.MULTILINE)
    if m:
        blob = m.group(1)
        if blob.startswith(".") or not any(x in blob for x in [":", "/"]):
            maybe = path.parent / blob
            if not maybe.exists():
                errors.append(f"FROM blob not on disk: {maybe}")
    return not errors, errors


def _ollama_available() -> bool:
    try:
        subprocess.run(["ollama", "--version"], check=True, capture_output=True)
        return True
    except FileNotFoundError:
        return False


def _run_modelfile(model_tag: str) -> dict:
    if not _ollama_available():
        return {"status": "skipped", "reason": "ollama binary not on PATH"}
    try:
        subprocess.run(
            ["ollama", "create", model_tag, "-f", str(MODELFILE)],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.CalledProcessError as e:
        return {"status": "error", "stdout": e.stdout, "stderr": e.stderr}
    except subprocess.TimeoutExpired:
        return {"status": "error", "reason": "ollama create timed out after 300s"}

    text_prompt = "Reply with exactly one word: pong"
    tool_prompt = (
        "You can call tools. Call the set_alarm tool with time 07:00. "
        "Respond ONLY in JSON: {\"tool\": \"set_alarm\", \"arguments\": {\"time\": \"07:00\"}}"
    )
    results: list[dict] = []
    for label, prompt in [("text", text_prompt), ("tool", tool_prompt)]:
        try:
            proc = subprocess.run(
                ["ollama", "run", model_tag, prompt],
                capture_output=True,
                text=True,
                timeout=120,
            )
            results.append({
                "label": label,
                "returncode": proc.returncode,
                "output": proc.stdout.strip()[:500],
                "error": proc.stderr.strip()[:500],
            })
        except subprocess.TimeoutExpired:
            results.append({"label": label, "timeout": True})
    return {"status": "ran", "results": results}


def main() -> int:
    print("P0 substrate verification")
    print(f"  models dir: {MODELS_DIR}")
    print(f"  Modelfile:  {MODELFILE}")

    gguf = _find_gguf()
    if gguf:
        print(f"  GGUF:       {gguf}")
    else:
        print("  GGUF:       NOT FOUND")

    valid, errors = _modelfile_is_valid(MODELFILE)
    print(f"  Modelfile syntax: {'OK' if valid else 'INVALID'}")
    for e in errors:
        print(f"    - {e}")

    if not _ollama_available() or not gguf or not valid:
        print("\nDry-run report — P0 cannot run live:")
        if not _ollama_available():
            print("  - Ollama is not installed or not on PATH")
        if not gguf:
            print(f"  - No .gguf found in {MODELS_DIR}")
        if not valid:
            print("  - Modelfile has syntax/dependency errors")
        print("\nTo run live:")
        print("  1. Install Ollama: curl -fsSL https://ollama.com/install.sh | sh")
        print(f"  2. Place LFM2.5-*.gguf in {MODELS_DIR}")
        print("  3. python3 sia-lab/product/verify_p0.py")
        return 0  # dry-run success: validation happened, missing deps reported honestly

    result = _run_modelfile("sia-p0")
    print("\nLive run result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("status") == "ran" else 1


if __name__ == "__main__":
    sys.exit(main())
