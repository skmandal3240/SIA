# P1 Action-Adapter Evaluation Summary

Evaluation run on the trained LoRA checkpoint in `sia-lab/posttrain/outputs/device_actions_lora/checkpoint-3/`.

## Method

- Base model: `unsloth/Llama-3.2-1B-Instruct`
- Adapter: SIA device-actions LoRA (r=16, alpha=16)
- Val set: 40 held-out examples from `sia-lab/posttrain/data/device_actions_val.json`
- Metric: exact structured match on tool name + JSON arguments

## Current result

- **Fast smoke (5 examples): 0% accuracy (0/5)**
- Full 40-example evaluation: running

## Observations

The merged model and Ollama GGUF both load and generate text, but the model does not emit the trained `<|sia_tool|>...<|sia_endcall|>` format. Instead it gives generic refusals or explanations.

This indicates the adapter is undertrained for the custom action grammar. Likely causes:

1. Only 200 synthetic training examples.
2. Base model's safety/refusal alignment dominates.
3. No explicit "MUST use SIA tool format" instruction during training.

## Next step to reach 95% gate

Retrain with:
- Larger, more diverse dataset (target 1,000+ examples).
- Stronger system prompt: "You MUST emit tool calls in SIA format. Do not explain. Do not refuse."
- Response-only loss on the SIA action tags.
- More epochs or higher rank if needed.

Training command ready:

```bash
python3 sia-lab/posttrain/sft.py --run --base unsloth/Llama-3.2-1B-Instruct
```

Run this on a Colab T4 (free), Kaggle, or RunPod L4 (~$0.50/run).

## Artifacts

- Merged HF weights: `sia-lab/posttrain/outputs/p1_merged/`
- Ollama GGUF: `sia-lab/posttrain/outputs/p1_merged/sia-p1.gguf`
- Ollama model: `sia-p1`
