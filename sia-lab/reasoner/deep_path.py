#!/usr/bin/env python3
"""Deep-path wrapper around SIRReasoner.

Ponytail: this file wires the tiny from-scratch reasoner as a stand-in deep
core. The real P3 milestone is up-cycling LFM2.5 weights into an RDT-MoE
reasoner; that needs GPU training and is marked below. For repo completeness,
this wrapper exposes the expected interface and accepts a governor budget.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
# `memory` is a package under sia-lab/, so its parent (sia-lab) must be on the
# path — not sia-lab/memory — for `from memory import ...` to resolve when this
# module is run standalone.
sys.path.insert(0, str(ROOT.parent))

from reasoner import SIRConfig, SIRReasoner
from memory import GraphRAGStub, EpisodicStore


class DeepPath:
    """Answer a question over a context using the recurrent-depth core.

    Optionally retrieves semantic graph triples and recent episodes related
    to the question and injects them into the prompt.
    """

    def __init__(
        self,
        cfg: SIRConfig | None = None,
        checkpoint_path: str | None = None,
        graph: GraphRAGStub | None = None,
        episodes: EpisodicStore | None = None,
    ) -> None:
        self.cfg = cfg or SIRConfig(
            vocab_size=128,
            dim=128,
            n_heads=4,
            n_layers=1,
            depth_per_layer=2,
            n_experts=4,
            top_k=1,
            mlp_hidden=256,
            act_max_steps=4,
        )
        self.model = SIRReasoner(self.cfg)
        self.graph = graph or GraphRAGStub()
        self.episodes = episodes or EpisodicStore()
        if checkpoint_path:
            # ponytail: placeholder for real LFM2.5 up-cycled checkpoint.
            pass
        self._vocab: dict[str, int] | None = None
        self._inv: dict[int, str] | None = None

    def _ensure_vocab(self, text: str) -> None:
        if self._vocab is not None:
            return
        chars = sorted(set(text))
        self._vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
        for ch in chars:
            if ch not in self._vocab:
                self._vocab[ch] = len(self._vocab)
        self._inv = {i: c for c, i in self._vocab.items()}
        # ponytail: expand model embedding if new vocab is larger than config.
        if len(self._vocab) > self.cfg.vocab_size:
            self.cfg = SIRConfig(**{**self.cfg.__dict__, "vocab_size": len(self._vocab)})
            self.model = SIRReasoner(self.cfg)

    def _encode(self, text: str, max_len: int = 128) -> torch.Tensor:
        tokens = [self._vocab.get(ch, self._vocab["<unk>"]) for ch in text[:max_len]]
        tokens += [self._vocab["<pad>"]] * (max_len - len(tokens))
        return torch.tensor([tokens], dtype=torch.long)

    def _decode(self, tokens: torch.Tensor) -> str:
        assert self._inv is not None
        return "".join(self._inv.get(int(t), "?") for t in tokens[0])

    def answer(self, context: list[str], question: str, budget: dict | None = None) -> str:
        """Generate an answer conditioned on context + question."""
        self._ensure_vocab(" ".join(context) + " " + question)
        assert self._vocab is not None and self._inv is not None

        # P4 memory: inject graph triples and recent episodes related to the question.
        memory_context = ""
        for entity in question.split():
            triples = self.graph.query(entity)
            if triples:
                memory_context += "; ".join(f"{t['subject']} {t['predicate']} {t['object']}" for t in triples) + ". "
        # ponytail: recall last 3 episodes as a cheap stand-in for semantic search.
        recent = sorted(self.episodes._episodes.values(), key=lambda e: e.created_at, reverse=True)[:3]
        if recent:
            memory_context += "Recent: " + " | ".join(ep.content for ep in recent) + ". "

        prompt = "Context: " + " ".join(context) + "\n"
        if memory_context:
            prompt += "Memory: " + memory_context + "\n"
        prompt += "Question: " + question + "\nAnswer: "
        idx = self._encode(prompt)

        # Governor budget: reduce depth when hot/low-battery.
        if budget:
            self.cfg = SIRConfig(**{**self.cfg.__dict__, "act_max_steps": budget.get("act_max_steps", self.cfg.act_max_steps)})
            self.model = SIRReasoner(self.cfg)
            self._vocab = None  # ponytail: vocab will be rebuilt for new size if needed
            self._inv = None

        input_len = idx.shape[1]
        self.model.eval()
        with torch.no_grad():
            out = self.model.generate(idx, max_new=20, temperature=0.8)
        # Rebuild vocab for decode in case model was recreated.
        self._ensure_vocab(" ".join(context) + " " + question)
        assert self._inv is not None
        # Decode only the newly generated tokens; the input was padded to a
        # fixed length, so slicing the decoded string by prompt char-count is
        # wrong (pad/special ids decode to multi-char names).
        answer = self._decode(out[:, input_len:])
        # Stop at newline, eos, or padding.
        answer = answer.split("\n")[0].split("<eos>")[0].split("<pad>")[0].strip()
        return answer or "unknown"


def main() -> int:
    deep = DeepPath()
    print(deep.answer(["Rahul lives in Patna.", "Patna is the capital of Bihar."], "Which state capital does Rahul live in?"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
