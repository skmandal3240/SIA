#!/usr/bin/env bash
# SIA installer: creates an isolated venv and installs runtime dependencies.
#
# Usage:
#   ./scripts/install.sh
#   ./bin/sia            # run the full stack once installed
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${SIA_VENV_DIR:-$REPO_ROOT/.venv}"

find_python() {
    for candidate in python3.12 python3.11 python3.10 python3; do
        if command -v "$candidate" >/dev/null 2>&1; then
            echo "$candidate"
            return 0
        fi
    done
    echo "error: no python3 (>=3.10) found on PATH" >&2
    exit 1
}

PYTHON_BIN="$(find_python)"
PY_VERSION="$("$PYTHON_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
echo "==> using $PYTHON_BIN (Python $PY_VERSION)"

echo "==> creating virtual environment at $VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"

VENV_PY="$VENV_DIR/bin/python"
"$VENV_PY" -m pip install --quiet --upgrade pip
echo "==> installing dependencies from requirements.txt"
"$VENV_PY" -m pip install --quiet -r "$REPO_ROOT/requirements.txt"

echo "==> verifying install (sia-lab/safety/crypto.py smoke check)"
"$VENV_PY" "$REPO_ROOT/sia-lab/safety/crypto.py"

echo ""
echo "SIA installed. Run it with:"
echo "  ./bin/sia"
