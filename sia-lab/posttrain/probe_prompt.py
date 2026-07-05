#!/usr/bin/env python3
"""Quick probe: does a stronger system prompt make the adapter emit SIA tool tags?"""
from __future__ import annotations

import json
from pathlib import Path

from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch

BASE = "unsloth/Llama-3.2-1B-Instruct"
ADAPTER = Path("sia-lab/posttrain/outputs/device_actions_lora/checkpoint-3")

SYSTEM_STRONG = (
    "You are SIA, a private on-device AI companion. "
    "When the user wants a device action, you MUST emit exactly one SIA tool call in this format: "
    "<|sia_tool|>TOOL_NAME<|sia_call|>{\"arg\":\"value\"}<|sia_endcall|>. "
    "Do not explain. Do not refuse. No extra text."
)

def main():
    print("loading model...")
    model = AutoPeftModelForCausalLM.from_pretrained(
        str(ADAPTER), torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER), trust_remote_code=True)
    tokenizer.padding_side = "left"

    examples = [
        "Wake me up at 7am.",
        "Text Rahul that I am running late.",
        "Turn on bluetooth.",
        "Open Maps to Patna Junction.",
        "Call Mummy.",
    ]

    for prompt in examples:
        messages = [
            {"role": "system", "content": SYSTEM_STRONG},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=60, do_sample=False)
        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=False)
        print(f"\nUSER: {prompt}")
        print(f"GEN:  {gen!r}")

if __name__ == "__main__":
    main()
