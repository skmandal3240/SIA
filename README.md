# SIA

On-device AI companion built from scratch for privacy-first personal automation.

## Repository layout

```
SIA/
├── sia/
│   ├── __init__.py
│   ├── model.py      # tiny transformer language model from scratch
│   ├── train.py      # training loop
│   ├── inference.py  # text generation
│   ├── tokenizer.py  # character-level tokenizer
│   └── actions.py    # deterministic action parser
├── data/
│   └── device_actions.jsonl
├── train_local.py    # entry point: train on CPU/MPS/GPU
├── generate.py       # entry point: run inference
├── requirements.txt
└── README.md
```

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python3 train_local.py
```

Trains a small transformer on `data/device_actions.jsonl`. Saves to `checkpoints/`.

## Generate

```bash
python3 generate.py --prompt "Set an alarm for 6 AM"
```

## Note

This is a from-scratch model for learning and demonstration. For production
SIA on-device inference we will later load trained weights into Ollama/MLX,
but this repo contains the model itself, owned fully by SIA.
