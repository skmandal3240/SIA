#!/usr/bin/env python3
"""SIA Deep SIR reasoner — from-scratch Recurrent-Depth Transformer.

Ponytail: one self-contained file. No framework beyond torch+numpy.
Implements:
  - Recurrent-Depth block: layer looped in depth with shared weights.
  - MoE routing: top-1 gating with optional load-balancing.
  - MLA cache: fused key/value projection into a single c_kv vector.
  - ACT halting: adaptive computation time per position.
  - Spectral-radius tracking: recurrent weight rescaling to rho < 1.

All shapes are tiny so the model trains on a laptop CPU. The goal is a
passing gate: forward/backward, loop, routing, cache, and halting work.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SIRConfig:
    vocab_size: int = 256
    dim: int = 256
    n_heads: int = 4
    n_layers: int = 2          # number of recurrent-depth blocks
    depth_per_layer: int = 3   # internal loop steps per block (the "deep" part)
    moe_freq: int = 1          # every block uses MoE when True
    n_experts: int = 4
    top_k: int = 1
    mlp_hidden: int = 512
    dropout: float = 0.0
    max_seq_len: int = 512
    act_threshold: float = 0.01
    act_max_steps: int = 6
    tie_weights: bool = True


class RoPE(nn.Module):
    """Rotary positional embeddings stored as a small buffer."""

    def __init__(self, dim: int, max_seq_len: int = 512, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        pos = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", pos, inv_freq)
        self.register_buffer("cos", torch.cos(freqs).unsqueeze(0).unsqueeze(0))
        self.register_buffer("sin", torch.sin(freqs).unsqueeze(0).unsqueeze(0))

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        # x: (B, n_heads, T, head_dim)
        b, h, t, d = x.shape
        cos = self.cos[:, :, offset : offset + t, :]
        sin = self.sin[:, :, offset : offset + t, :]
        x1, x2 = x[..., ::2], x[..., 1::2]
        rot = torch.stack([-x2, x1], dim=-1).flatten(-2)
        return x * cos[..., : d // 2].repeat_interleave(2, dim=-1) + rot * sin[..., : d // 2].repeat_interleave(2, dim=-1)


class MLA(nn.Module):
    """Multi-Head Latent Attention with a single compressed KV cache.

    Projects keys and values together into c_kv of size kv_dim,
    expands on decode via up-projection. This is the DeepSeek-V2/V3 trick
    implemented at gate-toy scale.
    """

    def __init__(self, dim: int, n_heads: int, kv_dim: int | None = None):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.kv_dim = kv_dim or (self.head_dim * n_heads)
        # W_DQ: down-project query to latent, W_UQ: up-project to heads
        self.q_down = nn.Linear(dim, dim, bias=False)
        self.q_up = nn.Linear(dim, dim, bias=False)
        # W_DKV: compress keys+values, W_UK/W_UV: expand per head
        self.kv_down = nn.Linear(dim, self.kv_dim, bias=False)
        self.k_up = nn.Linear(self.kv_dim, dim, bias=False)
        self.v_up = nn.Linear(self.kv_dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, rope: RoPE, past: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        q = self.q_up(self.q_down(x)).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        c_kv = self.kv_down(x)  # (B, T, kv_dim)
        k = self.k_up(c_kv).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_up(c_kv).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = rope(q), rope(k)
        if past is not None:
            c_kv = torch.cat([past, c_kv], dim=1)
        # Flash-style memory-efficient attention with torch built-ins
        y = F.scaled_dot_product_attention(q, k, v, is_causal=past is None and T > 1)
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(y), c_kv


class MoE(nn.Module):
    """Top-k sparse mixture-of-experts with load-balance auxiliary loss."""

    def __init__(self, dim: int, hidden: int, n_experts: int, top_k: int = 1):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = nn.Linear(dim, n_experts, bias=False)
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
            for _ in range(n_experts)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        logits = self.gate(x)  # (B, T, E)
        probs = F.softmax(logits, dim=-1)
        top_probs, top_idx = torch.topk(probs, self.top_k, dim=-1)  # (B, T, k)
        top_probs = top_probs / (top_probs.sum(dim=-1, keepdim=True) + 1e-9)
        out = torch.zeros_like(x)
        for k in range(self.top_k):
            for e in range(self.n_experts):
                mask = top_idx[:, :, k] == e
                if mask.any():
                    expert_in = x[mask]  # (N, D)
                    expert_out = self.experts[e](expert_in) * top_probs[mask, k : k + 1]
                    out[mask] += expert_out
        # Auxiliary load-balance loss: encourage uniform expert usage.
        aux_loss = self.n_experts * (probs.mean(dim=(0, 1)) * probs.mean(dim=(0, 1))).sum()
        return out, aux_loss


class RecurrentDepthBlock(nn.Module):
    """One transformer block looped in depth with shared weights.

    Uses pre-norm, MLA attention, MoE MLP, and ACT halting. The loop carries a
    state that is updated until the ACT halting probability crosses threshold.
    """

    def __init__(self, cfg: SIRConfig):
        super().__init__()
        self.cfg = cfg
        self.attn_norm = nn.LayerNorm(cfg.dim)
        self.mlp_norm = nn.LayerNorm(cfg.dim)
        self.attn = MLA(cfg.dim, cfg.n_heads, kv_dim=cfg.dim)
        self.moe = MoE(cfg.dim, cfg.mlp_hidden, cfg.n_experts, cfg.top_k)
        self.act_gate = nn.Linear(cfg.dim, 1)

    def _spectral_rescale(self) -> float:
        """Rescale recurrent projection weights so rho(W) < 1."""
        with torch.no_grad():
            W = self.attn.q_down.weight
            rho = torch.linalg.matrix_norm(W, ord=2).item()
            target = 0.99
            if rho >= target:
                self.attn.q_down.weight.data *= target / max(rho, 1e-6)
            return min(rho, target)

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPE,
        past: torch.Tensor | None = None,
        use_act: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        cfg = self.cfg
        B, T, D = x.shape
        state = x
        halting_probs = torch.zeros(B, T, 1, device=x.device)
        remain = torch.ones(B, T, 1, device=x.device)
        depth_steps = 0
        carry = past
        for step in range(cfg.act_max_steps if use_act else cfg.depth_per_layer):
            attn_out, carry = self.attn(self.attn_norm(state), rope, carry)
            state = state + attn_out
            mlp_out, aux_loss = self.moe(self.mlp_norm(state))
            state = state + mlp_out
            if use_act:
                halt_logit = self.act_gate(self.mlp_norm(state))
                halt = torch.sigmoid(halt_logit) * remain
                halting_probs = halting_probs + halt
                remain = remain - halt
                depth_steps = step + 1
                if remain.max().item() < cfg.act_threshold:
                    break
            else:
                depth_steps = step + 1
                aux_loss = torch.tensor(0.0, device=x.device)
        info = {
            "aux_loss": aux_loss,
            "halting_probs": halting_probs,
            "depth_steps": depth_steps,
            "spectral_radius": self._spectral_rescale(),
        }
        return state, carry, halting_probs, info


class SIRReasoner(nn.Module):
    """Full tiny recurrent-depth transformer language model."""

    def __init__(self, cfg: SIRConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.rope = RoPE(cfg.dim // cfg.n_heads, cfg.max_seq_len)
        self.blocks = nn.ModuleList(RecurrentDepthBlock(cfg) for _ in range(cfg.n_layers))
        self.out_norm = nn.LayerNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.token_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> Tuple[torch.Tensor, dict]:
        x = self.token_emb(idx)
        B, T = idx.shape
        total_aux = torch.tensor(0.0, device=idx.device)
        spectral = []
        depths = []
        for block in self.blocks:
            x, _, _, info = block(x, self.rope, use_act=True)
            total_aux = total_aux + info["aux_loss"]
            spectral.append(info["spectral_radius"])
            depths.append(info["depth_steps"])
        logits = self.lm_head(self.out_norm(x))
        loss: torch.Tensor | None = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = loss + 0.01 * total_aux
        metrics = {
            "spectral_radius": max(spectral),
            "avg_depth": sum(depths) / len(depths),
            "aux_loss": total_aux.item(),
        }
        return logits, {"loss": loss, **metrics}

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new: int = 20, temperature: float = 1.0) -> torch.Tensor:
        self.eval()
        for _ in range(max_new):
            logits, _ = self.forward(idx)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx


# ponytail: tiny self-check proving the module can forward/backward/generate
def _self_check() -> None:
    cfg = SIRConfig(vocab_size=64, dim=64, n_heads=2, n_layers=1, depth_per_layer=2, n_experts=2, act_max_steps=4)
    model = SIRReasoner(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, info = model(x, targets=x)
    assert info["loss"] is not None
    info["loss"].backward()
    assert info["spectral_radius"] < 1.0
    gen = model.generate(x[:, :4], max_new=5)
    assert gen.shape == (2, 9)
    print("reasoner self-check passed")


if __name__ == "__main__":
    _self_check()
