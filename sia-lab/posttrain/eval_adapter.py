#!/usr/bin/env python3
"""Evaluate the trained P1 device-actions LoRA on a held-out validation set.

Measures structured function-call accuracy: tool name + arguments JSON.
CPU inference is slow; set SIA_EVAL_FAST=1 to evaluate 5 examples only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "unsloth/Llama-3.2-1B-Instruct"
ADAPTER_DIR = Path("sia-lab/posttrain/outputs/device_actions_lora/checkpoint-3")
FAST = os.environ.get("SIA_EVAL_FAST", "0") == "1"


def parse_action(text: str) -> dict | None:
    """Extract SIA tool call from generated or expected text."""
    # Try SIA grammar first
    m = re.search(r"<\|sia_tool\|\>(\w+)<\|sia_call\|\>(\{.*?\})<\|sia_endcall\|\>", text)
    if m:
        try:
            return {"tool": m.group(1), "arguments": json.loads(m.group(2))}
        except json.JSONDecodeError:
            return None
    # Fallback: bare JSON object array or single object
    try:
        obj = json.loads(text.strip())
        if isinstance(obj, list) and obj:
            obj = obj[0]
        if isinstance(obj, dict) and "tool" in obj:
            return {"tool": obj["tool"], "arguments": obj.get("arguments", {})}
    except json.JSONDecodeError:
        pass
    return None


def load_data(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate(adapter_dir: Path, val_path: Path, output_path: Path) -> dict:
    if not adapter_dir.exists():
        raise FileNotFoundError(f"adapter not found: {adapter_dir}")

    print(f"loading tokenizer from {adapter_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"loading base model {BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    # ponytail: adapter added special tokens, so resize base embeddings to match adapter vocab
    if len(tokenizer) != model.config.vocab_size:
        print(f"resizing embeddings from {model.config.vocab_size} to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    print(f"loading adapter {adapter_dir}")
    model = PeftModel.from_pretrained(model, str(adapter_dir))

    val_data = load_data(val_path)
    if FAST:
        val_data = val_data[:5]

    correct = 0
    total = len(val_data)
    per_item = []

    for i, sample in enumerate(val_data):
        messages = sample["messages"]
        expected_text = messages[-1]["content"]
        expected = parse_action(expected_text)
        prompt_messages = messages[:-1]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated_raw = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=False)
        generated = parse_action(generated_raw)

        ok = False
        if expected and generated:
            ok = expected["tool"] == generated["tool"] and expected["arguments"] == generated["arguments"]

        if ok:
            correct += 1
        per_item.append({
            "id": i,
            "user": messages[-2]["content"],
            "expected_text": expected_text,
            "generated_raw": generated_raw,
            "expected": expected,
            "generated": generated,
            "match": ok,
        })
        print(f"[{i+1}/{total}] match={ok} | expected={expected} | generated={generated}")

    accuracy = correct / total if total else 0.0
    result = {
        "adapter_dir": str(adapter_dir),
        "val_path": str(val_path),
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "per_item": per_item,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\naccuracy: {accuracy:.2%} ({correct}/{total})")
    print(f"results written to {output_path}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate SIA P1 LoRA")
    parser.add_argument("--adapter", default=str(ADAPTER_DIR))
    parser.add_argument("--val", default="sia-lab/posttrain/data/device_actions_val.json")
    parser.add_argument("--output", default="sia-lab/posttrain/outputs/eval_result.json")
    args = parser.parse_args(argv)

    evaluate(Path(args.adapter), Path(args.val), Path(args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
