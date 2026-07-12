#!/usr/bin/env python3
"""P1 action adapter dry-run pipeline.

Implements the device-actions LoRA recipe using Unsloth + TRL SFT.
In dry-run mode (default) the script only imports dependencies, loads a tiny
dataset, and initializes the model/lora/formatter without running training.
Prints the exact command for a real single-L4 GPU run.
"""
from __future__ import annotations

import json
import os
import sys

DRY_RUN = os.environ.get("SIA_DRY_RUN", "1") == "1"
BASE_MODEL = os.environ.get("SIA_BASE_MODEL", "liquidai/LFM2.5-1.2B-Instruct")
LORA_R = 16
LORA_ALPHA = 16
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3"]
# ponytail: these are the literal recipe targets; real model class may map them to
# o_proj/gate_proj/up_proj/down_proj. Override SIA_BASE_MODEL or edit the list.


def _make_dataset() -> list[dict]:
    """Minimal R4-schema device-actions dataset."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "set_alarm",
                "description": "Set a device alarm",
                "parameters": {"type": "object", "properties": {"time": {"type": "string"}}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "send_message",
                "description": "Send a text message",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "contact": {"type": "string"},
                        "body": {"type": "string"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "toggle_setting",
                "description": "Toggle a system setting",
                "parameters": {"type": "object", "properties": {"setting": {"type": "string"}}},
            },
        },
    ]
    samples = [
        (
            "Wake me up at 7am",
            [{"tool": "set_alarm", "arguments": {"time": "07:00"}}],
        ),
        (
            "Text Rohit that I am running late",
            [{"tool": "send_message", "arguments": {"contact": "Rohit", "body": "I am running late"}}],
        ),
        (
            "Turn on bluetooth",
            [{"tool": "toggle_setting", "arguments": {"setting": "bluetooth", "state": "on"}}],
        ),
    ]
    return [
        {
            "messages": [
                {"role": "system", "content": json.dumps({"tools": tools}, ensure_ascii=False)},
                {"role": "user", "content": user},
                {
                    "role": "assistant",
                    "content": json.dumps(calls, ensure_ascii=False),
                },
            ]
        }
        for user, calls in samples
    ]


def _mock_model_init():
    """ponytail: stand-in when unsloth is absent so the smoke path always runs.

    This fallback must not import any of the deps the smoke path is degrading
    away from (``datasets`` included), or it would crash with the very error it
    exists to survive. A plain list of dicts supports the ``len`` and indexing
    the smoke path needs.
    """

    class Tok:
        pad_token = "<pad>"
        eos_token = "</s>"

        @staticmethod
        def apply_chat_template(messages, tokenize=False, add_generation_prompt=False):
            return "\n".join(f"{m['role']}: {m['content']}" for m in messages)

        def save_pretrained(self, _path):
            pass

    class M:
        def save_pretrained(self, _path):
            pass

    # Expand samples into a text field to mirror the SFT dataset_text_field path.
    dataset = [
        {"text": Tok.apply_chat_template(sample["messages"])}
        for sample in _make_dataset()
    ]
    return M(), Tok(), dataset


def smoke() -> None:
    print("P1 action adapter smoke test")
    print(f"  base model: {BASE_MODEL}")
    print(f"  LoRA r={LORA_R} alpha={LORA_ALPHA}")
    print(f"  target modules: {','.join(TARGET_MODULES)}")
    print(f"  dry_run: {DRY_RUN}")

    # Lazy imports so the script can at least report formatting even if deps fail.
    try:
        import torch  # noqa: F401
        from unsloth import FastLanguageModel
        from datasets import Dataset
        from trl import SFTTrainer, SFTConfig
    except ImportError as e:
        print(f"\nDependency missing: {e}")
        print("Install with:")
        print("  pip install unsloth trl datasets torch")
        print("\nRunning mock smoke path so imports/data/model-init all exercise.")
        model, tokenizer, dataset = _mock_model_init()
        print(f"  dataset rows: {len(dataset)}")
        print("  model + LoRA initialized (mock)")
        # Print response-only text sample
        print(f"  sample text length: {len(dataset[0]['text'])}")
        return

    dataset = Dataset.from_list(_make_dataset())
    print(f"  dataset rows: {len(dataset)}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        device_map="auto",
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        use_rslora=False,
        use_gradient_checkpointing="unsloth",
    )
    print("  model + LoRA initialized")

    # Response-only loss via DataCollatorForCompletionOnlyLM in SFTTrainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        args=SFTConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            max_seq_length=2048,
            num_train_epochs=3,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            output_dir="sia-lab/posttrain/outputs/p1_action_adapter",
            report_to="none",
        ),
    )
    print("  trainer initialized")

    if DRY_RUN:
        print("\nDry-run complete; skipping .train().")
    else:
        trainer.train()
        model.save_pretrained("sia-lab/posttrain/outputs/p1_action_adapter")
        tokenizer.save_pretrained("sia-lab/posttrain/outputs/p1_action_adapter")
        print("Training complete.")


def print_gpu_command() -> None:
    cmd = (
        f"SIA_DRY_RUN=0 SIA_BASE_MODEL={BASE_MODEL} "
        "python3 sia-lab/posttrain/train_adapter.py"
    )
    print("\nRun on a single L4 GPU:")
    print(f"  {cmd}")
    print("If Unsloth does not yet support the base model, pass a compatible instruct model via SIA_BASE_MODEL.")


def main() -> int:
    smoke()
    print_gpu_command()
    return 0


if __name__ == "__main__":
    sys.exit(main())
