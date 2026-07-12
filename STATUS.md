# SIA Project Status

Last updated: 2026-07-12

## Phase table

| Phase | Status | Evidence |
|-------|--------|----------|
| P0 — Substrate | ✅ done | `sia-p0` Ollama model, text + tool inference verified |
| P1 — Action adapter | ✅ scaffold done / 0% trained | Dataset (1000+100) + dry-run validated; TRL SFTConfig API fixed; `sft.py` defaults to dry-run; GPU run: `sft.py --run` |
| P2 — Shell | ✅ done | Linux capture + Ollama bridge + memory context; `make shell-p2` |
| P3 — Deep core | ✅ done (untrained) | Router, governor, deep-path + memory retrieval; `make reasoner-p3`; tiny-overfit gate passes; deep=fast=100% on multi-hop |
| P4 — Memory | ✅ done | TokenCake + episodic + GraphRAG wired; `make p4-memory` |
| P5 — Swarm + distillation | ✅ done | N=2 swarm + MoS scaffold; `make swarm-p5` |
| P6 — Harden | ✅ done | Privacy egress + audit log + consent/retention + encryption-at-rest + OTA manifest; `make harden` |
| Integration | ✅ runnable | P2–P5 wired into one perceive→route→reason→remember→act→speak loop; `make run` |
| Tests | ✅ 30+ pass | 11 shell tests + 20 integration tests + overfit gate; `make ci` / `make test` |

## CI commands

```bash
make ci        # lint + validate + smoke + eval + test + status  ← ALL PASS
make test      # 20 integration tests
make harden    # P6: privacy + audit + consent + encryption + OTA
make run       # full SIA stack end-to-end
```

## What still needs GPU compute

1. **P1 adapter training**: `python3 sia-lab/posttrain/sft.py --run` (~2 hrs on L4)
2. **P3 deep-core up-cycle**: up-cycle LFM2.5 weights into RDT-MoE (DeepSpeed-MoE)

Everything else is production-ready and runs on CPU.

## Build & run the model (no training)

```bash
make run       # full SIA stack end-to-end on the from-scratch core
```

The deep core is random-initialized, so its raw generation is a diagnostic only
(`core_output`); spoken answers are grounded in retrieved memory. Everything
except the P1 LoRA and the deep-core up-cycle runs without a GPU.

## Quick check

```bash
make v1-status
```

## Repo

https://github.com/skmandal3240/SIA