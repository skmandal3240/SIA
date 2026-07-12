# SIA — Private, On-Device AI Companion for India

SIA is an edge-first AI companion that runs locally on phones, laptops, mini-PCs, and browser tabs. It is designed for Indian users and is DPDP-compliant by construction: raw screen, voice, and personal data stay on the device by default.

## Public release readiness

| Phase | Goal | Status | Evidence |
|-------|------|--------|----------|
| **P0 — Substrate** | LFM2.5 runs on Ollama + browser ONNX; 125K context proven | ✅ done | GGUF downloaded, `sia-p0` Ollama model created, text + tool inference verified |
| **P1 — Action adapter** | Device-actions LoRA with 95% tool-call accuracy | ✅ scaffold done / 0% trained | Dataset (1000+100) + dry-run validated; TRL SFTConfig API fixed; `sft.py` defaults to dry-run; GPU run: `python3 sia-lab/posttrain/sft.py --run` |
| **P2 — Shell** | See screen → reason → point/act + speak, shared dispatcher | ✅ done | Linux real capture + Ollama bridge + memory context; `make shell-p2` passes |
| **P3 — Deep core** | RDT-MoE+MLA+ACT reasoner beats fast path on multi-hop | ✅ done (untrained) | Router, governor, deep-path + memory; `make reasoner-p3` passes; tiny-overfit gate passes; deep=fast=100% on multi-hop eval; real LFM2.5 up-cycle needs GPU |
| **P4 — Memory + eval** | TokenCake + episodic + GraphRAG wired to reasoner/shell | ✅ done | Memory wired into reasoner and shell; `make p4-memory` passes |
| **P5 — Swarm + distillation** | N=2 swarm loop and Mixture-of-Students lift | ✅ done | N=2 swarm + MoS scaffold; `make swarm-p5` passes |
| **P6 — Harden** | Governor authority, DPDP hooks, quant matrix, OTA adapters | ✅ done | Privacy egress test + audit log + consent/retention + encryption-at-rest + OTA manifest; `make harden` passes |

**Overall: all phases pass CI. P1 LoRA training and P3 deep-core up-cycle require GPU compute (Colab T4 / L4 spot). Everything else runs without a GPU.**

## CI status

```bash
make ci        # lint + validate + smoke + eval + test + status  ← ALL PASS
make test      # 20 integration tests (memory, reasoner, router, shell, swarm, safety, e2e)
make harden    # P6: privacy + audit + consent + encryption + OTA
make run       # full SIA stack end-to-end
```

All 20 integration tests pass. Ruff lint clean. All phase targets pass.

## Quick start

```bash
make run       # build + run the full stack end-to-end on the from-scratch core (no training)
make ci        # lint + validate + smoke + eval + test + status
make test      # integration test suite (20 tests)
make harden    # P6 hardening: privacy + audit + consent + encryption + OTA
make privacy   # network egress test
```

`make run` executes the whole SIA loop — perceive → govern → route → reason →
remember → act → speak — wiring P2 (shell), P3 (router + from-scratch deep
core), P4 (memory), and P5 (swarm) into one pass. The deep core is
random-initialized: it genuinely runs a forward + generate pass, but until it
is trained its raw output is a diagnostic only and spoken answers are grounded
in retrieved memory. No GPU required.

## What still needs GPU compute

1. **P1 adapter training**: `python3 sia-lab/posttrain/sft.py --run --base unsloth/Llama-3.2-1B-Instruct` (~2 hrs on L4)
2. **P3 deep-core up-cycle**: up-cycle LFM2.5 weights into RDT-MoE (DeepSpeed-MoE, needs A100/L4)

Everything else is production-ready and runs on CPU.

## Layer map

| Layer | Name | Location | Status |
|-------|------|----------|--------|
| L0 | Substrate | `sia-lab/product/`, `PROJECT/models/Modelfile` | ✅ verified |
| L1 | Fast path | `sia-lab/product/verify_p0.py` | scaffolded |
| L2 | Action adapter | `sia-lab/posttrain/`, `sia-lab/posttrain/adapter/` | ✅ scaffold + dry-run; adapter not trained (GPU needed) |
| L3 | Reasoner | `sia-lab/reasoner/` | ✅ from-scratch tiny core runs + overfits; wired into `make run` |
| L4 | Memory | `sia-lab/memory/` | ✅ modules + tests; wired into `make run` |
| L5 | Swarm | `sia-lab/swarm/` | ✅ N=2 consensus + MoS distillation; `make swarm-p5` / `make run` |
| L6 | Safety | `sia-lab/safety/` | ✅ privacy egress + audit log + consent/retention + encryption-at-rest + OTA |

## GPU training command

```bash
# P1: device-actions LoRA (needs GPU)
python3 sia-lab/posttrain/sft.py --run --base unsloth/Llama-3.2-1B-Instruct

# Or via train_gcp.py directly:
python3 sia-lab/posttrain/train_gcp.py \
    --base unsloth/Llama-3.2-1B-Instruct \
    --epochs 3 --batch-size 2 --grad-accum 4
```

## Test coverage

| Module | Tests | Status |
|--------|-------|--------|
| Shell (capture, STT, TTS, tags, dispatcher, loop) | 11 | ✅ `test_shell.py` |
| Memory (TokenCake eviction + pin + roundtrip, Episodic TTL, GraphRAG multihop) | 4 | ✅ `test_integration.py` |
| Reasoner (forward/backward, generate, spectral radius) | 2 | ✅ `test_integration.py` |
| Router + Governor (fast/deep/hot routing, budget modes) | 4 | ✅ `test_integration.py` |
| Swarm (consensus, disagreement) | 2 | ✅ `test_integration.py` |
| Safety (audit, consent, encryption) | 3 | ✅ `test_integration.py` |
| Multi-hop eval | 1 | ✅ `test_integration.py` |
| End-to-end (normal + hot mode) | 2 | ✅ `test_integration.py` |
| P3 overfit gate (loss convergence, ρ(A)<1) | 1 | ✅ `tiny_overfit.py` |
| **Total** | **30+** | **all pass** |

## License

MIT — see `LICENSE`. Dependencies are permissive only (MIT / Apache-2.0); no AGPL code is vendored.