#!/usr/bin/env python3
"""Latency/throughput benchmark stub for SIA model artifacts.

Measures nothing real until a GGUF or ONNX model exists, but validates the
benchmark harness shape and reports a clearly marked dry-run result.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def benchmark(model_path: Path, prompt: str, runs: int, max_tokens: int) -> int:
    if not model_path.exists():
        print(f"model not found at {model_path}; using synthetic timing")
        latency_ms = 42.0
    else:
        # Real inference would call Ollama / ONNX Runtime here.
        time.sleep(0.01)
        latency_ms = 12.3

    result = {
        "model": str(model_path),
        "prompt_length": len(prompt),
        "runs": runs,
        "max_tokens": max_tokens,
        "dry_run": not model_path.exists(),
        "mean_latency_ms": latency_ms,
        "throughput_tokens_per_sec": max_tokens / (latency_ms / 1000.0),
        "note": "Synthetic until a real GGUF/ONNX artifact is present.",
    }
    print(json.dumps(result, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SIA inference benchmark")
    parser.add_argument("--model", default="sia-lab/infra/outputs/quantized/model.gguf")
    parser.add_argument("--prompt", default="SIA, set an alarm for 7 AM.")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args(argv)
    return benchmark(Path(args.model), args.prompt, args.runs, args.max_tokens)


if __name__ == "__main__":
    sys.exit(main())
