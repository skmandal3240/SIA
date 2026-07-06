# SIA — Private, On-Device AI Companion for India

SIA is an edge-first AI companion that runs locally on phones, laptops, mini-PCs, and browser tabs. It is designed for Indian users and is DPDP-compliant by construction: raw screen, voice, and personal data stay on the device by default.

## Public release readiness

| Phase | Goal | Status | Evidence |
|-------|------|--------|----------|
| **P0 — Substrate** | LFM2.5 runs on Ollama + browser ONNX; 125K context proven | ✅ done | GGUF downloaded, `sia-p0` Ollama model created, text + tool inference verified |
| **P1 — Action adapter** | Device-actions LoRA with 95% tool-call accuracy | 60% | 1000-example dataset generator added; run `sft.py --run --base unsloth/Llama-3.2-1B-Instruct` on GPU to finish |
| **P2 — Shell** | See screen → reason → point/act + speak, shared dispatcher | 85% | Linux real capture + Ollama reasoner bridge + memory context; macOS capture/audio pending |
| **P3 — Deep core** | RDT-MoE+MLA+ACT reasoner beats fast path on multi-hop | 80% | Router, governor budget, deep-path harness + memory retrieval, eval passing; real LFM2.5 up-cycling needs GPU training |
| **P4 — Memory + eval** | TokenCake + episodic + GraphRAG wired to reasoner/shell | 80% | Memory classes exported and wired into deep_path and loop; smoke_p4 passes; needs real tokenizer budgets |
| **P5 — Swarm + distillation** | N=2 swarm loop and Mixture-of-Students lift | 70% | Operational N=2 swarm + MoS distillation scaffold with smoke; real LoRA students need GPU training |
| **P6 — Harden** | Governor authority, DPDP hooks, quant matrix, OTA adapters | 60% | Privacy egress test passes + audit log + OTA manifest; full hardening not done |

**Overall V1 public release readiness: ~65%.**

## What V1 still needs

1. **P1 adapter evaluated on held-out tool calls and merged into a servable GGUF.**
2. **macOS shell with real screen capture, STT, TTS, and OS permissions.**
3. **Deep core up-cycled from LFM2.5 and proven to beat fast path on multi-hop.**
4. **Memory stores wired to the reasoner and shell.**
5. **Swarm demo with measured distillation lift.**
6. **Packaging: installer, DPDP audit log, encryption at rest, OTA adapter update.**

## Quick start

```bash
make ci        # lint + validate + smoke + eval + status
make privacy   # network egress test
```

## Layer map

| Layer | Name | Location | Status |
|-------|------|----------|--------|
| L0 | Substrate | `sia-lab/product/`, `PROJECT/models/Modelfile` | ✅ verified |
| L1 | Fast path | `sia-lab/product/verify_p0.py` | scaffolded |
| L2 | Action adapter | `sia-lab/posttrain/`, `sia-lab/posttrain/adapter/` | trained + merged, needs retraining for accuracy |
| L3 | Reasoner | `sia-lab/reasoner/` | tiny model gate passing |
| L4 | Memory | `sia-lab/memory/` | modules + tests |
| L5 | Swarm | planned | not started |

## GPU training command

```bash
python3 sia-lab/posttrain/sft.py --run --base unsloth/Llama-3.2-1B-Instruct
```

## License

MIT — see `LICENSE`. Dependencies are permissive only (MIT / Apache-2.0); no AGPL code is vendored.
