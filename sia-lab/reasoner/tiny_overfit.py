#!/usr/bin/env python3
"""Tiny-corpus CPU overfit demo for the SIA deep SIR reasoner.

Trains a gate-sized model on a hand-written Indic-English reasoning corpus
and prints loss curve + sample generation. Proves that:
  - forward/backward works end-to-end,
  - recurrent-depth loop, MoE routing, MLA cache, ACT halting all fire,
  - spectral radius stays below 1,
  - the model can overfit a tiny corpus (loss decreases).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from reasoner import SIRConfig, SIRReasoner

CORPUS = [
    "SIA is a private on-device AI for India.",
    "User: Wake me up at 6 AM. Assistant: I will set the alarm.",
    "User: Text Rahul I am late. Assistant: Sending message to Rahul.",
    "The answer to 2 plus 3 is 5.",
    "Bihar ka sapna apna ghar hai.",
    "Honesty builds trust and trust builds community.",
]


def encode(text: str, vocab: dict[str, int], max_len: int = 64) -> torch.Tensor:
    tokens = [vocab.get(ch, vocab["<unk>"]) for ch in text[:max_len]]
    tokens += [vocab["<pad>"]] * (max_len - len(tokens))
    return torch.tensor(tokens, dtype=torch.long)


def build_vocab(corpus: list[str]) -> dict[str, int]:
    chars = sorted(set("".join(corpus)))
    vocab: dict[str, int] = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    for ch in chars:
        if ch not in vocab:
            vocab[ch] = len(vocab)
    return vocab


def train(corpus: list[str] = CORPUS, steps: int = 400, lr: float = 5e-3) -> SIRReasoner:
    torch.manual_seed(42)
    vocab = build_vocab(corpus)
    inv = {i: c for c, i in vocab.items()}
    cfg = SIRConfig(
        vocab_size=len(vocab),
        dim=128,
        n_heads=4,
        n_layers=1,
        depth_per_layer=2,
        moe_freq=1,
        n_experts=4,
        top_k=1,
        mlp_hidden=256,
        act_max_steps=4,
    )
    model = SIRReasoner(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    seqs = torch.stack([encode(t, vocab) for t in corpus])
    targets = torch.stack([torch.cat([s[1:], torch.tensor([vocab["<pad>"]])]) for s in seqs])

    losses: list[float] = []
    for step in range(steps):
        opt.zero_grad()
        _, info = model(seqs, targets=targets)
        loss = info["loss"]
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
        if step % 80 == 0 or step == steps - 1:
            print(
                f"step {step:03d} loss={loss.item():.3f} "
                f"rho={info['spectral_radius']:.3f} depth={info['avg_depth']:.2f}"
            )

    start = torch.tensor([[vocab["<bos>"]]])
    gen = model.generate(start, max_new=40, temperature=0.8)[0].tolist()
    text = "".join(inv.get(t, "?") for t in gen)
    print(f"\nfinal loss: {losses[-1]:.3f} (first: {losses[0]:.3f})")
    print(f"sample generation: {text!r}")

    assert info["spectral_radius"] < 1.0, "spectral radius should stay below 1"
    assert losses[-1] < losses[0] * 0.5, f"model did not overfit: {losses[-1]} vs {losses[0]}"
    print("\ntiny-overfit gate passed")
    return model


if __name__ == "__main__":
    train()
