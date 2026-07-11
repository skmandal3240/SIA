# Device-Actions LoRA Adapter

This directory tracks the metadata and recipe for the SIA Phase 1 LoRA adapter.

The actual 1.1 GB checkpoint is too large for Git. Download it from the Google Drive link in the README, or re-train it with:

```bash
python3 sia-lab/posttrain/sft.py --base unsloth/Llama-3.2-1B-Instruct
```

## Adapter config

- Base model: `unsloth/Llama-3.2-1B-Instruct`
- LoRA rank: 16
- LoRA alpha: 16
- Targets: `w1`, `w2`, `w3`, `q_proj`, `k_proj`, `v_proj`, `out_proj`, `in_proj`
- Epochs: 3
- Learning rate: 2e-4

## Download

Google Drive: https://drive.google.com/drive/folders/1r1R97vPXw0SYT___39nNhMqi8pGmPGM2

Place contents into `sia-lab/posttrain/outputs/device_actions_lora/`.
