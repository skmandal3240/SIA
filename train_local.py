#!/usr/bin/env python3
"""Train SIA from scratch."""
from sia.train import train

if __name__ == "__main__":
    train(
        data_path="data/device_actions.jsonl",
        dim=256,
        n_layers=4,
        n_heads=4,
        max_len=128,
        epochs=100,
        batch_size=4,
        lr=3e-4,
        checkpoint_dir="checkpoints",
    )
