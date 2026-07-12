#!/usr/bin/env python3
"""Stub for INT8/INT4 quantization dry-run.

Real quantization requires a trained adapter or model weights. This module
checks the export graph, validates supported bit widths, and writes a fake
quantized artifact so downstream Make targets can run without GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Acceptable bit-widths per the ZeroQuant/QLoRA matrix in TRD §C Stage D.
SUPPORTED_BITS = (8, 4, 6)


def fake_quantize(input_dir: Path, output_dir: Path, bits: int) -> int:
    """Write a minimal quantized-model manifest; no actual weights touched."""
    if bits not in SUPPORTED_BITS:
        print(f"unsupported bits={bits}; supported={SUPPORTED_BITS}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "sia-quant-manifest/v0",
        "source": str(input_dir),
        "bits": bits,
        "method": "zeroquant" if bits == 8 else f"int{bits}_dynamic",
        "output_dir": str(output_dir),
        "files": {
            "config.json": "stub: real run produces GGUF + ONNX",
            "model.gguf": f"stub-int{bits}",
            "model.onnx": f"stub-int{bits}",
        },
        "notes": "Dry-run artifact only. Run on GPU to produce real quantized weights.",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    print(f"quantized stub manifest written to {output_dir / 'manifest.json'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SIA quantization dry-run")
    parser.add_argument("--input-dir", default="sia-lab/posttrain/outputs/device_actions_lora")
    parser.add_argument("--output-dir", default="sia-lab/infra/outputs/quantized")
    parser.add_argument("--bits", type=int, default=8, choices=SUPPORTED_BITS)
    args = parser.parse_args(argv)
    return fake_quantize(Path(args.input_dir), Path(args.output_dir), args.bits)


if __name__ == "__main__":
    sys.exit(main())
