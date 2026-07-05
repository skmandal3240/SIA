# SIA — Private, On-Device AI Companion for India

SIA is an edge-first AI companion that runs locally on phones, laptops, mini-PCs, and browser tabs. It is designed for Indian users and is DPDP-compliant by construction: raw screen, voice, and personal data stay on the device by default.

## Public release readiness

| Phase | Goal | Status | Evidence |
|-------|------|--------|----------|
| **P0 — Substrate** | LFM2.5 runs on Ollama + browser ONNX; 125K context proven | 80% | Modelfile exists, P0 dry-run passes, live GGUF not yet placed |
| **P1 — Action adapter** | Device-actions LoRA with 95% held-out tool-call accuracy | 90% | 200-example dataset, Unsloth SFT pipeline fixed, LoRA trained and downloaded |
| **P2 — Shell** | See screen → reason → point/act + speak, shared dispatcher | 70% | Linux shell stubs, dispatcher, tag parser, tests passing; real macOS capture/audio pending |
| **P3 — Deep core** | RDT-MoE+MLA+ACT reasoner beats fast path on multi-hop | 60% | Tiny from-scratch model overfits on CPU; up-cycle from LFM2.5 not done |
| **P4 — Memory + eval** | TokenCake + episodic + GraphRAG + governor tests | 60% | All modules and tests in repo, not yet wired to live reasoner |
| **P5 — Swarm + distillation** | N=2 swarm loop and Mixture-of-Students lift | 0% | Not started |
| **P6 — Harden** | Governor authority, DPDP hooks, quant matrix, OTA adapters | 50% | Privacy egress test passes; full hardening not done |

**Overall V1 public release readiness: ~65%.**

## What V1 still needs

1. **Real LFM2.5 GGUF in `PROJECT/models/` and verified Ollama run.**
2. **P1 adapter evaluated on held-out tool calls and merged into a servable GGUF.**
3. **macOS shell with real screen capture, STT, TTS, and OS permissions.**
4. **Deep core up-cycled from LFM2.5 and proven to beat fast path on multi-hop.**
5. **Memory stores wired to the reasoner and shell.**
6. **Swarm demo with measured distillation lift.**
7. **Packaging: installer, DPDP audit log, encryption at rest, OTA adapter update.**

## Quick start

```bash
make ci        # lint + validate + smoke + eval + status
make privacy   # network egress test
```

## Layer map

| Layer | Name | Location | Status |
|-------|------|----------|--------|
| L0 | Substrate | `sia-lab/product/`, `PROJECT/models/Modelfile` | dry-run passing |
| L1 | Fast path | `sia-lab/product/verify_p0.py` | scaffolded |
| L2 | Action adapter | `sia-lab/posttrain/`, `sia-lab/posttrain/adapter/` | trained, needs merge |
| L3 | Reasoner | `sia-lab/reasoner/` | tiny model gate passing |
| L4 | Memory | `sia-lab/memory/` | modules + tests |
| L5 | Swarm | planned | not started |

## GPU training command

```bash
python3 sia-lab/posttrain/sft.py --run --base unsloth/Llama-3.2-1B-Instruct
```

## License

MIT — see `LICENSE`. Dependencies are permissive only (MIT / Apache-2.0); no AGPL code is vendored.
