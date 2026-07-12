#!/usr/bin/env python3
"""Merge the trained P1 LoRA into the base model and export to Ollama GGUF.

If unsloth is available, use its GGUF export path. Otherwise, export merged
HF weights and print the llama.cpp conversion command.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE_MODEL = "unsloth/Llama-3.2-1B-Instruct"
ADAPTER_DIR = Path("sia-lab/posttrain/outputs/device_actions_lora/checkpoint-3")
MERGED_DIR = Path("sia-lab/posttrain/outputs/p1_merged")


def merge(adapter_dir: Path, merged_dir: Path) -> None:
    import torch
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    print(f"loading merged model from adapter {adapter_dir}")
    # AutoPeftModel handles special-token embedding resize automatically.
    model = AutoPeftModelForCausalLM.from_pretrained(
        str(adapter_dir),
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ponytail: merge LoRA weights into base so we can export a single artifact
    print("merging LoRA into base")
    merged = model.merge_and_unload()

    merged_dir.mkdir(parents=True, exist_ok=True)
    print(f"saving merged model to {merged_dir}")
    merged.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print("merge complete")


def export_gguf(merged_dir: Path, gguf_path: Path) -> bool:
    try:
        from unsloth import FastLanguageModel  # type: ignore
        print("using unsloth GGUF export")
        model, tokenizer = FastLanguageModel.from_pretrained(str(merged_dir))
        model.save_pretrained_gguf(str(gguf_path.parent), tokenizer, quantization_method="q4_k_m")
        default = gguf_path.parent / "unsloth.Q4_K_M.gguf"
        if default.exists() and not gguf_path.exists():
            default.rename(gguf_path)
        return gguf_path.exists()
    except Exception as exc:  # pragma: no cover
        print(f"unsloth GGUF export failed: {exc}")

    print("\nFallback: convert merged HF weights with llama.cpp")
    print("  git clone https://github.com/ggerganov/llama.cpp")
    print(f"  python3 llama.cpp/convert_hf_to_gguf.py {merged_dir} --outfile {gguf_path} --outtype q4_k_m")
    return False


def write_modelfile(gguf_path: Path) -> None:
    gguf_path.parent.mkdir(parents=True, exist_ok=True)
    modelfile = gguf_path.parent / "Modelfile"
    text = f"""FROM ./{gguf_path.name}

TEMPLATE \"\"\"{{ if .System }}<|system|>
{{ .System }}</s>{{ end }}{{ if .Prompt }}<|user|>
{{ .Prompt }}</s>{{ end }}<|assistant|>
{{ .Response }}</s>\"\"\"

SYSTEM You are SIA, a private on-device AI companion. You can set alarms, send messages, control settings, navigate, and manage calendar events. Emit tool calls as SIA action tags when needed.

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 16384
"""
    modelfile.write_text(text)
    print(f"Modelfile written to {modelfile}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge and export P1 LoRA")
    parser.add_argument("--adapter", default=str(ADAPTER_DIR))
    parser.add_argument("--merged", default=str(MERGED_DIR))
    parser.add_argument("--gguf", default="sia-lab/posttrain/outputs/p1_merged/sia-p1.gguf")
    parser.add_argument("--skip-merge", action="store_true")
    args = parser.parse_args(argv)

    adapter_dir = Path(args.adapter)
    merged_dir = Path(args.merged)
    gguf_path = Path(args.gguf)

    if not args.skip_merge:
        merge(adapter_dir, merged_dir)

    ok = export_gguf(merged_dir, gguf_path)
    write_modelfile(gguf_path)

    if ok:
        print(f"\nGGUF ready: {gguf_path}")
        print(f"Create Ollama model: cd {gguf_path.parent} && ollama create sia-p1 -f Modelfile")
    else:
        print(f"\nMerged HF weights ready: {merged_dir}")
        print("GGUF export needs llama.cpp conversion step above.")
    return 0 if (ok or merged_dir.exists()) else 1


if __name__ == "__main__":
    sys.exit(main())
