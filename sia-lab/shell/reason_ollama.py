"""Reasoner bridge that calls the local Ollama SIA model.

This keeps the shell model-agnostic: any reasoner callable can be injected,
but this helper wires `ollama run sia-p0` (or another tag) for P2 demos.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


def _ollama_available() -> bool:
    return shutil.which("ollama") is not None


def reason_with_ollama(
    screenshot: dict,
    transcript: str,
    context: str = "",
    model_tag: str = "sia-p0",
    prompt_template: str | None = None,
) -> tuple[str, list[dict]]:
    """Return (response_text, tool_calls) by asking Ollama.

    The prompt includes screen dimensions and the user transcript. Tool
    calls are parsed from a JSON block if the model emits one.
    """
    if not _ollama_available():
        # ponytail: degrade to stub response instead of crashing the loop.
        return (
            f"Ollama not available. Screen: {screenshot['width']}x{screenshot['height']}. "
            f"Heard: {transcript}",
            [],
        )

    tpl = prompt_template or (
        "You are SIA, a private on-device AI companion.\n"
        "Screen size: {width}x{height}\n"
        "Context: {context}\n"
        "User said: {transcript}\n"
        "Reply concisely. If you need to point at the screen, use [POINT:x,y:label:screen0]. "
        "If you need to call a device tool, include JSON: {{\"tool\": \"...\", \"arguments\": {{...}}}}."
    )
    prompt = tpl.format(width=screenshot["width"], height=screenshot["height"], transcript=transcript, context=context)

    try:
        proc = subprocess.run(
            ["ollama", "run", model_tag, prompt],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return "Model response timed out.", []

    text = proc.stdout.strip()
    tool_calls: list[dict] = []
    # Look for a JSON tool-call block anywhere in the response.
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                obj = json.loads(line)
                if "tool" in obj and "arguments" in obj:
                    tool_calls.append(obj)
            except json.JSONDecodeError:
                pass

    # Strip the JSON line from the spoken response.
    clean_lines = [ln for ln in text.splitlines() if not (ln.strip().startswith("{") and ln.strip().endswith("}"))]
    response = " ".join(clean_lines).strip()
    return response, tool_calls


def main(argv: list[str] = sys.argv) -> int:
    # Self-check: call with synthetic input.
    try:
        from .capture import CaptureStub
    except ImportError:
        # Standalone run (`python sia-lab/shell/reason_ollama.py`): no parent
        # package, so fall back to an absolute import off this file's dir.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from capture import CaptureStub
    screenshot = CaptureStub().grab()
    transcript = " ".join(argv[1:]) or "hello sia"
    response, tools = reason_with_ollama(screenshot, transcript, context="self-check")
    print("Response:", response)
    print("Tools:", tools)
    return 0 if response else 1


if __name__ == "__main__":
    raise SystemExit(main())
