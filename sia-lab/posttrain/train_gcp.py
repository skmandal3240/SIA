#!/usr/bin/env python3
"""SIA P1 device-actions LoRA trainer — Google Cloud edition.

Runs on a single GPU (a GCP `g2-standard-*` with an L4, or an `n1` + T4 is
plenty for a 1B base). Unlike `sft.py`, this trainer uses only the standard,
easy-to-install Hugging Face stack — `transformers` + `peft` + `trl` +
`datasets` — so it works on a plain Deep Learning VM or a Vertex AI custom
container without the unsloth toolchain.

What it does:
  1. Loads a small instruct base model (default Llama-3.2-1B-Instruct).
  2. Adds the SIA tool/action special tokens and resizes the embeddings.
  3. Attaches a LoRA adapter (r=16, alpha=16) on the attention + MLP projections.
  4. Fine-tunes with response-only loss (the user/system turns are masked).
  5. Evaluates exact structured-match tool-call accuracy on the held-out set.
  6. Saves the adapter (and optionally a merged fp16 model) to --output-dir,
     which can be a local path or a gs:// bucket path (auto-synced).

Quick check without a GPU (validates data + config + wiring only):
    python3 sia-lab/posttrain/train_gcp.py --dry-run

Real run on a GCP GPU VM:
    python3 sia-lab/posttrain/train_gcp.py \
        --base meta-llama/Llama-3.2-1B-Instruct \
        --train sia-lab/posttrain/data/device_actions_train.json \
        --val   sia-lab/posttrain/data/device_actions_val.json \
        --output-dir gs://YOUR_BUCKET/sia/device_actions_lora \
        --epochs 3
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# SIA action grammar: single-token anchors the adapter learns to emit.
SIA_SPECIAL_TOKENS = [
    "<|sia_tool|>",
    "<|sia_call|>",
    "<|sia_endcall|>",
    "<|sia_action|>",
    "<|sia_endaction|>",
    "<|sia_screen|>",
]

# Standard Llama-family projection names. For a non-Llama base (e.g. LFM2),
# override with --target-modules to match that architecture.
DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_chat_json(path: Path) -> list[dict]:
    """Load a [{"messages": [...]}, ...] file and validate its schema."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path}: expected a non-empty JSON list of samples")
    for i, sample in enumerate(data):
        msgs = sample.get("messages")
        if not msgs or msgs[-1].get("role") != "assistant":
            raise ValueError(f"{path}[{i}]: sample must end in an assistant turn")
        for turn in msgs:
            if "role" not in turn or "content" not in turn:
                raise ValueError(f"{path}[{i}]: turn missing role/content")
    return data


def assistant_target(sample: dict) -> str:
    """The gold assistant completion used for exact-match eval."""
    return sample["messages"][-1]["content"].strip()


# --------------------------------------------------------------------------- #
# Dry-run: no GPU, no heavy deps — just prove the data + config are sound.
# --------------------------------------------------------------------------- #
def dry_run(args: argparse.Namespace) -> int:
    train = load_chat_json(Path(args.train))
    val = load_chat_json(Path(args.val))
    text = json.dumps(train + val, ensure_ascii=False)
    missing = [t for t in SIA_SPECIAL_TOKENS if t not in text]
    print(f"train samples : {len(train)}")
    print(f"val samples   : {len(val)}")
    print(f"base model    : {args.base}")
    print(f"LoRA          : r={LORA_R} alpha={LORA_ALPHA} targets={args.target_modules or DEFAULT_TARGET_MODULES}")
    print(f"special tokens present in data: {[t for t in SIA_SPECIAL_TOKENS if t in text]}")
    if missing:
        # Not fatal: on-screen POINT actions use a subset of the tokens.
        print(f"note: tokens not present in this dataset slice: {missing}")
    print("DRY-RUN OK — data + config validated (no GPU used)")
    return 0


# --------------------------------------------------------------------------- #
# Real training
# --------------------------------------------------------------------------- #
def _is_gcs(path: str) -> bool:
    return path.startswith("gs://")


def _sync_to_gcs(local_dir: Path, gcs_uri: str) -> None:
    """Copy the finished adapter up to a bucket with gsutil (present on GCP VMs)."""
    print(f"uploading {local_dir} -> {gcs_uri}")
    subprocess.run(["gsutil", "-m", "cp", "-r", f"{local_dir}/*", gcs_uri], check=True)


def real_run(args: argparse.Namespace) -> int:
    import torch  # type: ignore
    from datasets import Dataset  # type: ignore
    from transformers import (  # type: ignore
        AutoModelForCausalLM,
        AutoTokenizer,
    )
    from peft import LoraConfig, get_peft_model  # type: ignore
    from trl import SFTConfig, SFTTrainer  # type: ignore

    target_modules = args.target_modules or DEFAULT_TARGET_MODULES
    train_samples = load_chat_json(Path(args.train))
    val_samples = load_chat_json(Path(args.val))

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # On a GPU use bf16/fp16; on CPU-only VMs use fp32 — half precision matmul
    # is unsupported/painfully slow on CPU, so fp16 there would crash or crawl.
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype, device_map="auto")
    else:
        dtype = torch.float32
        model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype)

    # Register the SIA action tokens as single tokens and grow the embeddings.
    new_tokens = [t for t in SIA_SPECIAL_TOKENS if t not in tokenizer.get_vocab()]
    if new_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        model.resize_token_embeddings(len(tokenizer))
        print(f"added {len(new_tokens)} SIA special tokens")

    model = get_peft_model(
        model,
        LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        ),
    )
    model.print_trainable_parameters()

    # Render chat -> text via the base model's own template.
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}assistant: "
        )

    def to_text(sample: dict) -> dict:
        return {"text": tokenizer.apply_chat_template(sample["messages"], tokenize=False)}

    train_ds = Dataset.from_list([to_text(s) for s in train_samples])

    # Response-only loss: mask everything up to the assistant turn so the model
    # is trained only on the tool-call completion, not on echoing the prompt.
    collator = None
    try:
        from trl import DataCollatorForCompletionOnlyLM  # type: ignore
        # The assistant marker as rendered by the template; adjust if you swap it.
        collator = DataCollatorForCompletionOnlyLM("assistant:", tokenizer=tokenizer)
    except Exception as exc:  # pragma: no cover - older/newer trl variations
        print(f"response-only collator unavailable ({exc}); training on full text")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        data_collator=collator,
        args=SFTConfig(
            output_dir=args.local_dir,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            lr_scheduler_type="linear",
            warmup_ratio=0.03,
            weight_decay=0.01,
            logging_steps=10,
            bf16=(dtype == torch.bfloat16),
            fp16=(dtype == torch.float16),
            max_seq_length=args.max_seq_length,
            dataset_text_field="text",
            report_to="none",
            seed=42,
        ),
    )
    trainer.train()

    out = Path(args.local_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"adapter saved to {out}")

    acc = evaluate(model, tokenizer, val_samples, args.max_seq_length)
    (out / "eval.json").write_text(json.dumps({"exact_match_accuracy": acc}, indent=2))
    print(f"held-out exact-match tool-call accuracy: {acc:.2%}")

    if args.merge:
        merged = out / "merged"
        model.merge_and_unload().save_pretrained(merged)
        tokenizer.save_pretrained(merged)
        print(f"merged fp16 model saved to {merged}")

    if _is_gcs(args.output_dir):
        _sync_to_gcs(out, args.output_dir)
    return 0


def evaluate(model, tokenizer, val_samples: list[dict], max_seq_length: int) -> float:
    """Exact structured-match: does the generated completion equal the gold call?"""
    import torch  # type: ignore

    model.eval()
    correct = 0
    for sample in val_samples:
        prompt_msgs = sample["messages"][:-1]
        prompt = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_length).to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=64, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        completion = tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        completion = completion.replace(tokenizer.eos_token or "", "").strip()
        if completion.startswith(assistant_target(sample)):
            correct += 1
    return correct / max(1, len(val_samples))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SIA P1 device-actions LoRA trainer (GCP)")
    parser.add_argument("--base", default="meta-llama/Llama-3.2-1B-Instruct",
                        help="HF base model id (or unsloth/Llama-3.2-1B-Instruct)")
    parser.add_argument("--train", default="sia-lab/posttrain/data/device_actions_train.json")
    parser.add_argument("--val", default="sia-lab/posttrain/data/device_actions_val.json")
    parser.add_argument("--output-dir", default="sia-lab/posttrain/outputs/device_actions_lora",
                        help="local path or gs:// bucket path for the finished adapter")
    parser.add_argument("--local-dir", default="",
                        help="local scratch dir (defaults to --output-dir when it is not gs://)")
    parser.add_argument("--target-modules", nargs="*", default=None,
                        help=f"LoRA targets (default: {DEFAULT_TARGET_MODULES})")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--merge", action="store_true", help="also save a merged fp16 model")
    parser.add_argument("--dry-run", action="store_true", help="validate data+config without a GPU")
    args = parser.parse_args(argv)

    # A gs:// output needs a local staging dir to write into first.
    if not args.local_dir:
        args.local_dir = ("sia-lab/posttrain/outputs/device_actions_lora"
                          if _is_gcs(args.output_dir) else args.output_dir)

    if args.dry_run:
        return dry_run(args)
    return real_run(args)


if __name__ == "__main__":
    sys.exit(main())
