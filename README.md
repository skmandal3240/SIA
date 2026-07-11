# SIA — Private, On-Device AI Companion for India

SIA is an edge-first AI companion that runs locally on phones, laptops, mini-PCs, and browser tabs. It is designed for Indian users and is DPDP-compliant by construction: raw screen, voice, and personal data stay on the device by default.

## Public release readiness

| Phase | Goal | Status | Evidence |
|-------|------|--------|----------|
| **P0 — Substrate** | LFM2.5 runs on Ollama + browser ONNX; 125K context proven | ✅ done | GGUF downloaded, `sia-p0` Ollama model created, text + tool inference verified |
| **P1 — Action adapter** | Device-actions LoRA with 95% tool-call accuracy | 75% scaffold / 0% trained | Dataset + dry-run/train path fixed; response-only loss wired; parser ready; real GPU LoRA run remains |
| **P2 — Shell** | See screen → reason → point/act + speak, shared dispatcher | 85% | Linux real capture + Ollama bridge + memory context; `make shell-p2` passes |
| **P3 — Deep core** | RDT-MoE+MLA+ACT reasoner beats fast path on multi-hop | 80% | Router, governor, deep-path + memory; `make reasoner-p3` passes; real LFM2.5 up-cycle needs GPU training |
| **P4 — Memory + eval** | TokenCake + episodic + GraphRAG wired to reasoner/shell | 85% | Memory wired into reasoner and shell; `make p4-memory` passes |
| **P5 — Swarm + distillation** | N=2 swarm loop and Mixture-of-Students lift | 75% | N=2 swarm + MoS scaffold; `make swarm-p5` passes |
| **P6 — Harden** | Governor authority, DPDP hooks, quant matrix, OTA adapters | 72% | Privacy egress test + encrypted-at-rest audit log/memory stores (crypto-shred via device keystore) + OTA adapter update wired end-to-end (download, checksum/size verify, atomic install, `make p6-harden`); still needs a real installer/packaging pass |

**Overall V1 public release readiness: blocked on P1 GPU training (core engine ready, actions not trained).**

## What V1 still needs

1. **P1 adapter evaluated on held-out tool calls and merged into a servable GGUF.**
2. **macOS shell with real screen capture, STT, TTS, and OS permissions.**
3. **Deep core up-cycled from LFM2.5 and proven to beat fast path on multi-hop.**
4. **Memory stores wired to the reasoner and shell.**
5. **Swarm demo with measured distillation lift.**
6. **Packaging: installer.** (DPDP audit log, encryption at rest for local stores, and OTA adapter update are done — `sia-lab/safety/crypto.py`, `sia-lab/infra/ota.py`, `make p6-harden`.)

## Quick start

```bash
make run       # build + run the full stack end-to-end on the from-scratch core (no training)
make ci        # lint + validate + smoke + eval + status
make privacy   # network egress test
```

`make run` executes the whole SIA loop — perceive → govern → route → reason →
remember → act → speak — wiring P2 (shell), P3 (router + from-scratch deep
core), P4 (memory), and P5 (swarm) into one pass. The deep core is
random-initialized: it genuinely runs a forward + generate pass, but until it
is trained its raw output is a diagnostic only and spoken answers are grounded
in retrieved memory. No GPU required.

## Layer map

| Layer | Name | Location | Status |
|-------|------|----------|--------|
| L0 | Substrate | `sia-lab/product/`, `PROJECT/models/Modelfile` | ✅ verified |
| L1 | Fast path | `sia-lab/product/verify_p0.py` | scaffolded |
| L2 | Action adapter | `sia-lab/posttrain/`, `sia-lab/posttrain/adapter/` | scaffold + dry-run; adapter not trained (0% tool-call accuracy) |
| L3 | Reasoner | `sia-lab/reasoner/` | from-scratch tiny core runs; wired into `make run` |
| L4 | Memory | `sia-lab/memory/` | modules + tests; wired into `make run` |
| L5 | Swarm | `sia-lab/swarm/` | N=2 consensus wired; `make swarm-p5` / `make run` |

## GPU training command

```bash
python3 sia-lab/posttrain/sft.py --base unsloth/Llama-3.2-1B-Instruct
```

## License

MIT — see `LICENSE`. Dependencies are permissive only (MIT / Apache-2.0); no AGPL code is vendored.
