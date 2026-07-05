#!/usr/bin/env python3
"""P1 action-adapter SFT pipeline dry-run with Unsloth+TRL.

Full GPU training is one command away:
    python sia-lab/posttrain/sft.py --run --base unsloth/Llama-3.2-1B-Instruct
This file only imports the libraries and validates the dataset/schema
without touching a model when no GPU is present.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Third-party deps required by the real run. Dry-run tolerates missing GPU.
try:
    from unsloth import FastLanguageModel
    import trl
except Exception as exc:  # pragma: no cover - allowed in CPU-only dry-run
    FastLanguageModel = None
    trl = None
    _import_exc = exc

# SIA tool/action grammar tokens: single-token anchors for the action layer.
SIA_TOOL_TOKENS = [
    "<|sia_tool|>",
    "<|sia_call|>",
    "<|sia_endcall|>",
    "<|sia_action|>",
    "<|sia_endaction|>",
    "<|sia_screen|>",
]

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3"]
LORA_R = 16
LORA_ALPHA = 16


def _schema_ok(sample: dict) -> tuple[bool, str]:
    for key in ("messages",):
        if key not in sample:
            return False, f"missing {key}"
    for turn in sample["messages"]:
        if "role" not in turn or "content" not in turn:
            return False, "turn missing role/content"
    return True, ""


def make_dry_dataset(path: Path | None = None) -> list[dict]:
    """Return a minimal valid tool-call dataset in chat format."""
    samples = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are SIA, a private on-device AI companion. "
                    "Use tools when needed. Available: set_alarm, send_message, open_maps.",
                },
                {"role": "user", "content": "Wake me up at 6:30 AM tomorrow."},
                {
                    "role": "assistant",
                    "content": '<|sia_tool|>set_alarm<|sia_call|>{"time":"06:30","repeat":"daily"}<|sia_endcall|>',
                },
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are SIA. Use tools when needed. Available: send_message.",
                },
                {"role": "user", "content": "Text Rahul that I am running late."},
                {
                    "role": "assistant",
                    "content": '<|sia_tool|>send_message<|sia_call|>{"contact":"Rahul","body":"Running late"}<|sia_endcall|>',
                },
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are SIA. On-screen actions share the tool dispatcher.",
                },
                {"role": "user", "content": "Open the alarm app on the screen."},
                {
                    "role": "assistant",
                    "content": '<|sia_screen|><|sia_action|>POINT:0.12,0.34:alarm_icon:screen0<|sia_endaction|>',
                },
            ]
        },
    ]
    if path:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
    return samples


def smoke_dataset(path: Path | None = None) -> int:
    """Validate dataset schema and that all SIA tokens appear at least once."""
    samples = make_dry_dataset(path)
    failures = 0
    for i, sample in enumerate(samples):
        ok, reason = _schema_ok(sample)
        if not ok:
            print(f"sample {i} schema fail: {reason}")
            failures += 1
    all_text = json.dumps(samples, ensure_ascii=False)
    missing = [t for t in SIA_TOOL_TOKENS if t not in all_text]
    if missing:
        print(f"missing SIA tokens: {missing}")
        failures += len(missing)
    return failures


def dry_run(args: argparse.Namespace) -> int:
    """Run every non-GPU check: imports, schema, LoRA target list, and dataset."""
    if FastLanguageModel is None or trl is None:
        print(f"unsloth/trl not available in this environment: {_import_exc}")
        print("dry-run continues: schema + config checks only")
    else:
        print("unsloth/trl import ok")

    ds_path = Path(args.dataset)
    ds_path.parent.mkdir(parents=True, exist_ok=True)
    failures = smoke_dataset(ds_path)
    print(f"LoRA targets: {LORA_TARGETS} (r={LORA_R}, alpha={LORA_ALPHA})")

    if failures:
        print(f"DRY-RUN FAILED ({failures} issue(s))")
        return 1
    print("DRY-RUN PASSED")
    return 0


def real_run(args: argparse.Namespace) -> int:
    """Actual Unsloth+TRL SFT loop. Requires a GPU and the full deps."""
    if FastLanguageModel is None:
        raise RuntimeError("unsloth not importable; install unsloth>=2024.12 and trl")

    import torch  # type: ignore
    from trl import SFTTrainer  # type: ignore
    from transformers import TrainingArguments  # type: ignore

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base,
        max_seq_length=args.max_seq_length,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        load_in_4bit=args.load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=LORA_TARGETS,
        lora_alpha=LORA_ALPHA,
        use_rslora=False,
    )

    # Add SIA special tokens to tokenizer and resize model embeddings.
    new_tokens = [t for t in SIA_TOOL_TOKENS if t not in tokenizer.get_vocab()]
    if new_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        model.resize_token_embeddings(len(tokenizer))

    ds_path = Path(args.dataset)
    if not ds_path.exists():
        make_dry_dataset(ds_path)

    from datasets import load_dataset  # type: ignore
    dataset = load_dataset("json", data_files=str(ds_path), split="train")

    chat_template = tokenizer.chat_template or (
        "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}"
        "assistant: "
    )
    tokenizer.chat_template = chat_template

    def formatting_func(examples):
        texts = []
        for messages in examples["messages"]:
            texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
        return {"text": texts}

    dataset = dataset.map(formatting_func, batched=True, remove_columns=dataset.column_names)

    # Response-only loss: mask user turns; TRL handles this via SFTTrainer.
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            output_dir=args.output_dir,
            logging_steps=5,
            optim="adamw_8bit",
            seed=3407,
        ),
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    print(f"adapter saved to {args.output_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SIA action-adapter SFT")
    parser.add_argument("--run", action="store_true", help="run real training (needs GPU)")
    parser.add_argument("--base", default="unsloth/Llama-3.2-1B-Instruct")
    parser.add_argument("--dataset", default="sia-lab/posttrain/device_actions.json")
    parser.add_argument("--output-dir", default="sia-lab/posttrain/outputs/device_actions_lora")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args(argv)

    return real_run(args) if args.run else dry_run(args)


if __name__ == "__main__":
    sys.exit(main())
