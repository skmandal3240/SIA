# Technical Requirements Document (TRD)
# SIA — On-Device Swarm-Brain Foundation

---

## 1. Document Control

| Field | Value |
|-------|-------|
| Document title | SIA Foundation — Technical Requirements Document |
| Document ID | SIA-TRD-001 |
| Version | 1.0 |
| Status | Draft for review |
| Author | SIA Engineering (solo founder) |
| Classification | Confidential — internal |
| Date | 18 Jun 2026 |
| Supersedes | — |
| Related docs | SIA-ARCH-001 (Foundation Architecture); SIR Identity Charter; TokenCake Spec; Swarm-Distillation Loop Spec; Thermal Management Spec; DPDP Compliance Note |

### 1.1 Revision history
| Ver | Date | Author | Change |
|-----|------|--------|--------|
| 0.1 | 18 Jun 2026 | SIA Eng | Initial architecture synthesis (SIA-ARCH-001) |
| 1.0 | 18 Jun 2026 | SIA Eng | First formal TRD; requirements baselined |

### 1.2 Approval
| Role | Name | Decision | Date |
|------|------|----------|------|
| Founder / Tech Lead | — | Pending | — |

---

## 2. Introduction

### 2.1 Purpose
This TRD specifies the **technical requirements** for the SIA foundation: the substrate model, the SIR reasoner, memory, action, embodiment, swarm, the training/distillation/export pipeline, and the resource governor. It is the authoritative, testable requirement baseline against which the foundation is built and accepted. It refines, and is subordinate to, the architecture in SIA-ARCH-001.

### 2.2 Scope

**In scope**
- Foundation model selection, serving, and on-device deployment (L0).
- SIR two-path reasoner: router, fast path, deep looped-MoE core (L1).
- Memory subsystem: working/KV, episodic/temporal, semantic/graph (L2).
- On-device action/tool layer via hot-swap LoRA adapters (L3).
- Perception/embodiment shell: screen vision, voice I/O, on-screen action (L4).
- Swarm-brain runtime and swarm→node distillation (L5).
- Cloud training/distillation/quantization/export pipeline.
- Resource & thermal governance; DPDP compliance posture.

**Out of scope (this TRD)**
- Application-level product UIs beyond the companion shell.
- ASTRO/creative/business domain *content* (only their tool schemas as adapters).
- Commercial/go-to-market, pricing, and grant applications.
- Hardware procurement beyond compute-class assumptions.

### 2.3 Intended audience
Engineering (model, systems, edge, infra), QA, and the founder as approver.

### 2.4 Definitions & acronyms
See §15 Glossary. Key: **SIA** swarm-brain system; **SIR** on-device reasoner node; **RDT** Recurrent-Depth Transformer; **MoE** Mixture-of-Experts; **MLA** Multi-Latent Attention; **ACT** Adaptive Computation Time; **LoRA** Low-Rank Adaptation; **DPDP** Digital Personal Data Protection Act (India).

### 2.5 References
| Ref | Source | Use |
|-----|--------|-----|
| R1 | LiquidAI LFM2.5 (Ollama library; HF `LiquidAI/LFM2.5-*`) | Substrate model (DEP) |
| R2 | kyegomez/OpenMythos | RDT-MoE reasoning core (BLUEPRINT) |
| R3 | deepspeedai/DeepSpeed (v0.19.x, DeepSpeed-MoE, ZeRO++, ZeroQuant, Mixture-of-Students) | Train/distill/quant (DEP) |
| R4 | Mandark-droid/LFM2.5-1.2B-Instruct-mobile-actions | Action LoRA recipe (BLUEPRINT) |
| R5 | sitammeur/lfm2.5-thinking-web | Browser/WebGPU deployment (PATTERN) |
| R6 | farzaa/clicky | Embodiment shell (PATTERN) |
| R7 | 666ghj/MiroFish (OASIS/CAMEL, GraphRAG) | Swarm runtime (PATTERN) |
| S1 | DPDP Act 2023 (India) | Compliance |
| S2 | ONNX Runtime Web / WebGPU; GGUF/Ollama | Runtimes |

### 2.6 Conventions
- Requirement IDs: `TRD-<TYPE>-<SUBSYS>-<NNN>`. TYPE ∈ {FR, NFR, IF, DR, SR}. SUBSYS ∈ {SUB, SIR, MEM, ACT, EMB, SWM, TDP, GOV, SYS}.
- Priority (MoSCoW): **M** Must, **S** Should, **C** Could, **W** Won't (this release).
- Normative verbs: *shall* (mandatory), *should* (recommended), *may* (optional).
- Each requirement carries: statement, priority, source ref, verification method (T=test, D=demo, A=analysis, I=inspection).

---

## 3. System Overview

### 3.1 Context
SIA is a privacy-first, on-device AI that perceives a user's device, reasons (shallow or deep as needed), acts via tools and on-screen control, and participates in a swarm whose collective experience is distilled back into each node. Default execution is local; the network is opt-in.

### 3.2 Architectural summary
Five layers over a substrate, with a two-speed reasoner:

```
L5 Swarm-brain ── L4 Embodiment ── L3 Action ── L2 Memory ── L1 SIR ── L0 Substrate
                                                          built by DeepSpeed (cloud)
```
SIR routes each query to a **fast path** (LFM2.5 direct) or a **deep path** (RDT looped-MoE core). Loop depth `T` and ACT halting are governed by device thermal/battery/latency budget.

### 3.3 Design decisions (baselined; override-able)
| ID | Decision | Rationale | Status |
|----|----------|-----------|--------|
| DD-1 | Deep core is **up-cycled from LFM2.5** weights (dense→MoE), not trained from random init | Inherits Liquid pretraining; ~10–50× cheaper than scratch | Baselined (confirm) |
| DD-2 | **Device floor = LFM2.5-1.2B-Thinking**; deep RDT-MoE runs on Node-Capable tier only; phones may receive a distilled looped *student* later | Fits phone RAM/thermal; avoids shipping full MoE to weak devices | Baselined (confirm) |
| DD-3 | MiroFish used as **re-implemented pattern** (OASIS/CAMEL + GraphRAG), not as ingested AGPL code | Avoids AGPL-3.0 copyleft on SIA | Baselined (confirm) |
| DD-4 | Deep-core attention = **MLA** (compressed KV latent) | Smallest KV cache for on-device RAM | Baselined |
| DD-5 | Action skills shipped as **hot-swap LoRA adapters**, never base retraining | Flat edge RAM; OTA-updatable | Baselined |
| DD-6 | Single **resource governor** drives loop depth, KV cap, path bias, offload | Unified thermal/latency control | Baselined |

### 3.4 Assumptions & dependencies
- **A1** LFM2.5 weights (8B-A1B, 1.2B-Instruct, 1.2B-Thinking-ONNX) remain available under their license.
- **A2** Cloud/VPS = Hostinger + burst GPU (L4/A100-class spot) for training/distillation.
- **A3** Edge runtimes available: Ollama (GGUF) on native; WebGPU + ONNX Runtime Web in target browsers.
- **A4** OpenMythos, DeepSpeed, clicky, mobile-actions usable under their licenses (MIT/Apache/other-permissive).
- **D1** PyTorch ≥ 2.0; CUDA/ROCm toolchain for DeepSpeed ops.

### 3.5 Constraints
- **C1** Edge device floor (phone): operate within ~1–2 GB model RAM and a skin-temperature thermal envelope.
- **C2** DPDP: raw PII/screen/voice must not leave device on the default path.
- **C3** Budget: grant-first; per-run compute kept lean (ZeRO++/ZeroQuant).
- **C4** Solo-founder build velocity: prefer copy-able blueprints (R2, R4) over novel research.

---

## 4. Functional Requirements

### 4.1 L0 — Substrate / model serving (SUB)
| ID | Pri | Requirement | Ref | V |
|----|-----|-------------|-----|---|
| TRD-FR-SUB-001 | M | The system *shall* serve LFM2.5-8B-A1B on the Node-Capable tier via GGUF/Ollama. | R1 | T |
| TRD-FR-SUB-002 | M | The system *shall* serve LFM2.5-1.2B-Instruct on the Edge-Standard tier. | R1 | T |
| TRD-FR-SUB-003 | M | The system *shall* serve LFM2.5-1.2B-Thinking (ONNX) in-browser via Transformers.js + ONNX Runtime Web + WebGPU with no server inference. | R1,R5 | T |
| TRD-FR-SUB-004 | M | The substrate *shall* expose native tool-calling and a 125K-token context window on Node-Capable. | R1 | T |
| TRD-FR-SUB-005 | S | The system *shall* select tier automatically from device capability (RAM, accelerator, thermal class). | — | T |
| TRD-FR-SUB-006 | M | The same logical model *shall* be packaged for two runtimes (GGUF native, ONNX browser) from one export pipeline. | R1,R5 | A |

### 4.2 L1 — SIR Reasoner (SIR)

**Router**
| ID | Pri | Requirement | Ref | V |
|----|-----|-------------|-----|---|
| TRD-FR-SIR-001 | M | SIR *shall* expose a single inference interface that internally routes each query to `fast` or `deep`. | — | T |
| TRD-FR-SIR-002 | M | The router *shall* default to `fast` and escalate to `deep` on multi-hop/planning/math-chain intent or explicit user request. | R2 | T |
| TRD-FR-SIR-003 | M | The router *shall* accept a budget `{max_loops, latency_ms, thermal_headroom}` from the governor (§4.8) and respect it. | — | T |

**Fast path**
| ID | Pri | Requirement | Ref | V |
|----|-----|-------------|-----|---|
| TRD-FR-SIR-010 | M | The fast path *shall* answer via LFM2.5 direct (no looping) for tool-calls, instruction following, and single-hop tasks. | R1 | T |
| TRD-FR-SIR-011 | M | The fast path *shall* handle ≥ 80% of production queries without escalation. | — | A |

**Deep path — RDT looped-MoE core (BLUEPRINT R2)**
| ID | Pri | Requirement | Ref | V |
|----|-----|-------------|-----|---|
| TRD-FR-SIR-020 | M | The deep core *shall* implement the structure Prelude → looped Recurrent Block (×T) → Coda. | R2 | I |
| TRD-FR-SIR-021 | M | The recurrent update *shall* be `h_{t+1} = A·h_t + B·e + Transformer(h_t, e)` with input `e` re-injected every loop. | R2 | I |
| TRD-FR-SIR-022 | M | Injection parameter A *shall* be constructed as `A = Diag(-exp(log_A))` (discretized) to guarantee spectral radius ρ(A) < 1. | R2 | T |
| TRD-FR-SIR-023 | M | The FFN *shall* be a fine-grained MoE with routed experts (top-`n_experts_per_tok`) plus ≥1 always-on shared expert. | R2 | I |
| TRD-FR-SIR-024 | M | Attention *shall* be MLA (compressed KV latent) per DD-4. | R2 | I |
| TRD-FR-SIR-025 | M | The core *shall* support ACT halting that stops looping per-position on convergence (no fixed-T forcing). | R2 | T |
| TRD-FR-SIR-026 | S | The core *shall* inject a loop-index (RoPE-like) embedding so each iteration is a distinct computational phase. | R2 | I |
| TRD-FR-SIR-027 | C | The core *may* apply depth-wise LoRA per loop for added expressiveness at minimal parameter cost. | R2 | I |
| TRD-FR-SIR-028 | S | The core *shall* support continuous depth-wise batching (per-sequence early exit within a batch). | R2 | T |
| TRD-FR-SIR-029 | M | Initial deep-core config *shall* baseline on `mythos_1b`/`mythos_3b` scale (dim 2048–3072, 64 experts, `n_experts_per_tok`=2, `n_shared_experts`=1, `max_loop_iters`=16). | R2 | I |
| TRD-FR-SIR-030 | M | The deep core *shall* run on the Node-Capable tier; phone execution of the full MoE core is **W** (DD-2). | — | A |

### 4.3 L2 — Memory subsystem (MEM)
| ID | Pri | Requirement | Ref | V |
|----|-----|-------------|-----|---|
| TRD-FR-MEM-001 | M | The working-memory/KV tier *shall* be managed by TokenCake. | Internal | T |
| TRD-FR-MEM-002 | M | KV state *shall* use MLA-style compressed latents to minimize on-device RAM. | R2 | T |
| TRD-FR-MEM-003 | M | An episodic/temporal store *shall* record time-stamped events with recency/decay and TTL hooks. | R7 | T |
| TRD-FR-MEM-004 | M | A semantic store *shall* maintain an entity-relationship graph (GraphRAG) supporting multi-hop retrieval. | R7 | T |
| TRD-FR-MEM-005 | M | The deep path *shall* be able to read GraphRAG-retrieved subgraphs as part of the injected input `e`. | R2,R7 | T |
| TRD-FR-MEM-006 | M | Memory eviction *shall* be driven by the same governor budget as loop depth (single controller). | — | T |
| TRD-FR-MEM-007 | M | Every graph node and episodic record *shall* carry consent + retention metadata (see §8). | S1 | I |

### 4.4 L3 — Action / Tool layer (ACT) (BLUEPRINT R4)
| ID | Pri | Requirement | Ref | V |
|----|-----|-------------|-----|---|
| TRD-FR-ACT-001 | M | SIR *shall* emit tool calls as structured JSON `[{"name","arguments"}]` given tool definitions in context. | R4 | T |
| TRD-FR-ACT-002 | M | Action skills *shall* ship as LoRA adapters trained per the R4 recipe (Unsloth+TRL SFT, r=16/α=16, response-only loss, targets q/k/v/out/in/w1/w2/w3). | R4 | I |
| TRD-FR-ACT-003 | M | Only the active domain's adapter *shall* be loaded at runtime to keep edge RAM flat. | DD-5 | T |
| TRD-FR-ACT-004 | M | The system *shall* ship a `device-actions` adapter (alarms, calls, messages, calendar, maps, system toggles). | R4 | T |
| TRD-FR-ACT-005 | S | The system *shall* support additional adapters: `astro-ops`, `creative-ops`, `swarm-ops`. | — | T |
| TRD-FR-ACT-006 | M | Tool-call JSON *shall* be dispatched over an MCP/skills bus shared with on-screen actions (§4.5). | R6 | T |
| TRD-FR-ACT-007 | S | Adapters *shall* be OTA-updatable independent of the base model. | — | T |

### 4.5 L4 — Embodiment / perception shell (EMB) (PATTERN R6)
| ID | Pri | Requirement | Ref | V |
|----|-----|-------------|-----|---|
| TRD-FR-EMB-001 | M | The shell *shall* capture screen content on demand and provide it to SIR (vision input). | R6 | T |
| TRD-FR-EMB-002 | M | The shell *shall* support voice input via streaming speech-to-text. | R6 | T |
| TRD-FR-EMB-003 | M | The shell *shall* support voice output via streaming text-to-speech. | R6 | T |
| TRD-FR-EMB-004 | M | SIR *shall* emit on-screen action tags (minimum `[POINT:x,y:label:screenN]`) parsed and executed by the shell. | R6 | T |
| TRD-FR-EMB-005 | S | The action-tag vocabulary *shall* extend beyond pointing to tap/scroll/type/gesture for mobile. | R6 | T |
| TRD-FR-EMB-006 | M | On-screen actions and API tool-calls *shall* share one dispatcher (§4.4). | R6 | A |
| TRD-FR-EMB-007 | M | The shell *shall* present as an always-available companion (e.g., menu-bar/overlay) without blocking the host UI. | R6 | D |
| TRD-FR-EMB-008 | M | The shell *shall* request and respect OS permissions (mic, screen capture, accessibility) explicitly. | R6 | I |

### 4.6 L5 — Swarm-brain (SWM) (PATTERN R7)
| ID | Pri | Requirement | Ref | V |
|----|-----|-------------|-----|---|
| TRD-FR-SWM-001 | M | The swarm runtime *shall* implement the 5-stage loop: Graph build → Environment setup → Simulation → Report → Deep interaction. | R7 | D |
| TRD-FR-SWM-002 | M | The runtime *shall* support multi-agent simulation with per-agent personas, independent memory, and behavioral logic (OASIS/CAMEL pattern, re-implemented per DD-3). | R7 | T |
| TRD-FR-SWM-003 | M | The runtime *shall* support an **operational** mode (real SIR nodes delegating/sharing/consensus) and a **simulation** mode (rehearsal sandbox). | R7 | D |
| TRD-FR-SWM-004 | M | Swarm experience (operational + simulated) *shall* be poolable as a corpus for node distillation (§4.7). | R3,R7 | A |
| TRD-FR-SWM-005 | S | The runtime *shall* support God-view variable injection to perturb a simulation. | R7 | D |
| TRD-FR-SWM-006 | M | The swarm *shall not* incorporate AGPL-licensed source; only re-implemented patterns (DD-3). | R7 | I |

### 4.7 Training / distillation / export pipeline (TDP) (DEP R3)
| ID | Pri | Requirement | Ref | V |
|----|-----|-------------|-----|---|
| TRD-FR-TDP-001 | M | The pipeline *shall* stand up the RDT-MoE core by **up-cycling LFM2.5 weights** (DD-1) using DeepSpeed-MoE with Hybrid Tensor-Expert-Data parallelism and ZeRO++. | R2,R3 | T |
| TRD-FR-TDP-002 | M | The pipeline *shall* perform reasoning/loop-scaling training (scale mean recurrence + data, not parameters). | R2 | A |
| TRD-FR-TDP-003 | M | The pipeline *shall* distill the cloud RDT-MoE + pooled swarm corpus into a smaller edge student via **DeepSpeed Mixture-of-Students** (the swarm-distillation loop). | R3 | T |
| TRD-FR-TDP-004 | M | The pipeline *shall* quantize edge artifacts via ZeroQuant (INT8 baseline; INT4/FP6 evaluated). | R3 | T |
| TRD-FR-TDP-005 | M | The pipeline *shall* export GGUF (Ollama) and ONNX (Transformers.js/WebGPU) plus separate LoRA action adapters. | R1,R5 | T |
| TRD-FR-TDP-006 | S | The pipeline *shall* be re-runnable as a scheduled distillation cycle that measurably improves the on-device SIR. | R3 | T |

### 4.8 Resource & thermal governor (GOV)
| ID | Pri | Requirement | Ref | V |
|----|-----|-------------|-----|---|
| TRD-FR-GOV-001 | M | A single governor *shall* read battery%, SoC/skin temperature, free RAM, latency SLA, and network state. | Internal | T |
| TRD-FR-GOV-002 | M | The governor *shall* output `{max_loops T, ACT_threshold, KV_cache_cap, path_bias, offload?}`. | — | T |
| TRD-FR-GOV-003 | M | Under hot/low-battery, the governor *shall* force fast path, T→1, and a tight KV cap. | — | T |
| TRD-FR-GOV-004 | S | Under cool/charging, the governor *shall* permit deep path with T up to 16 and a larger KV cap. | R2 | T |
| TRD-FR-GOV-005 | S | Under an SLA-critical request, the governor *shall* cap T to meet latency and *may* offload the deep path to the VPS. | — | T |
| TRD-FR-GOV-006 | M | Compute *shall* degrade smoothly (via ACT + depth-wise batching), not via a hard cutoff. | R2 | A |

---

## 5. Non-Functional Requirements

### 5.1 Performance / latency (PERF)
| ID | Pri | Requirement | V |
|----|-----|-------------|---|
| TRD-NFR-PERF-001 | M | Fast-path first-token latency *shall* be < 1 s on GPU/NPU edge and ≤ 5 s on CPU (Edge-Standard). | T |
| TRD-NFR-PERF-002 | M | A `device-actions` function call *shall* resolve in ≤ 5 s typical on Edge-Standard (per R4 observed ~1–5 s). | T |
| TRD-NFR-PERF-003 | S | Deep-path throughput *shall* benefit ≥ 2× from continuous depth-wise batching versus fixed-depth serving when handling concurrent swarm requests. | T |
| TRD-NFR-PERF-004 | M | Browser (WebGPU) inference of LFM2.5-Thinking *shall* stream tokens interactively (no full-response stall). | T |

### 5.2 Resource footprint (RES)
| ID | Pri | Requirement | V |
|----|-----|-------------|---|
| TRD-NFR-RES-001 | M | Edge-Standard model footprint *shall* fit ~1–2 GB; Node-Capable LFM2.5-8B-A1B ~5.2 GB GGUF. | I |
| TRD-NFR-RES-002 | M | Loading an action adapter *shall* add ≤ ~1% parameters (per R4, ~0.94%) and *shall not* materially raise steady-state RAM beyond the active adapter. | A |
| TRD-NFR-RES-003 | S | MLA KV compression *shall* reduce KV-cache RAM versus full-K/V GQA for equivalent context. | T |

### 5.3 Thermal (THM)
| ID | Pri | Requirement | V |
|----|-----|-------------|---|
| TRD-NFR-THM-001 | M | Sustained on-device inference *shall* remain within the device thermal envelope under governor control without thermal shutdown. | T |
| TRD-NFR-THM-002 | M | The governor *shall* react to thermal headroom within one inference cycle. | T |

### 5.4 Scalability (SCA)
| ID | Pri | Requirement | V |
|----|-----|-------------|---|
| TRD-NFR-SCA-001 | S | Effective capability *shall* scale with loop depth and adapters, not base parameter count (storage-cheap, compute-elastic). | A |
| TRD-NFR-SCA-002 | S | The swarm runtime *shall* support adding nodes without redesign of the node model. | A |

### 5.5 Reliability (REL)
| ID | Pri | Requirement | V |
|----|-----|-------------|---|
| TRD-NFR-REL-001 | M | Deep-core training *shall* be numerically stable at high learning rate by construction (ρ(A)<1). | T |
| TRD-NFR-REL-002 | M | The core *shall* avoid "overthinking" degradation by halting via ACT before divergence. | T |
| TRD-NFR-REL-003 | S | If the deep path fails or times out, SIR *shall* fall back to a fast-path answer. | T |

### 5.6 Portability (POR)
| ID | Pri | Requirement | V |
|----|-----|-------------|---|
| TRD-NFR-POR-001 | M | The foundation *shall* deploy across native (Ollama/GGUF) and browser (ONNX/WebGPU) from one pipeline. | T |
| TRD-NFR-POR-002 | S | Cloud training *shall* run on NVIDIA and *should* tolerate AMD/other accelerators supported by DeepSpeed. | A |

### 5.7 Maintainability (MNT)
| ID | Pri | Requirement | V |
|----|-----|-------------|---|
| TRD-NFR-MNT-001 | M | Skills *shall* evolve via adapter swaps and scheduled distillation, not base re-training. | I |
| TRD-NFR-MNT-002 | S | Each layer *shall* expose a documented interface (§6) so layers are independently replaceable. | I |

### 5.8 Observability (OBS)
| ID | Pri | Requirement | V |
|----|-----|-------------|---|
| TRD-NFR-OBS-001 | M | Every cloud egress and every executed action tag *shall* be logged at the gateway/shell. | I |
| TRD-NFR-OBS-002 | S | Router decisions, loop counts (T), and ACT halts *shall* be traceable per request for tuning. | T |

### 5.9 Usability (USA)
| ID | Pri | Requirement | V |
|----|-----|-------------|---|
| TRD-NFR-USA-001 | S | The companion shell *shall* surface a thinking/working state during deep-path reasoning (per R5 thinking-indicator pattern). | D |
| TRD-NFR-USA-002 | S | Voice and on-screen interaction *shall* feel continuous (push-to-talk, streamed speech/pointing). | D |

---

## 6. Interface Requirements (IF)

| ID | Pri | Interface | Contract | Ref | V |
|----|-----|-----------|----------|-----|---|
| TRD-IF-001 | M | SIR inference API | `infer(query, context) -> stream(text | tool_call | action_tag)`; routing internal | — | T |
| TRD-IF-002 | M | Router→Governor | `route(query,ctx) -> {path, budget{max_loops,latency_ms,thermal_headroom}}` | — | T |
| TRD-IF-003 | M | Memory API | `read(query)->{kv, subgraph, episodes}`, `write(episode)`, `evict(budget)` | R7 | T |
| TRD-IF-004 | M | Tool/Action bus | accepts `{name,arguments}` JSON and `[POINT/tap/scroll/type:…]` tags; one dispatcher | R4,R6 | T |
| TRD-IF-005 | M | Action-tag grammar | `[POINT:x,y:label:screenN]` (+ extended verbs); deterministic parse | R6 | T |
| TRD-IF-006 | M | Model runtime | GGUF via Ollama; ONNX via Transformers.js/ORT-Web/WebGPU | R1,R5 | T |
| TRD-IF-007 | M | Gateway/Worker | key custody + cloud-API proxy; only path that holds secrets | R6 | I |
| TRD-IF-008 | S | Swarm RPC | node task delegation, memory share, consensus messages | R7 | T |
| TRD-IF-009 | M | Export artifacts | `{gguf, onnx, lora_adapters[]}` with version + checksum | R3 | I |

---

## 7. Data Requirements (DR)

| ID | Pri | Requirement | Ref | V |
|----|-----|-------------|-----|---|
| TRD-DR-001 | M | Tool-call data format *shall* be JSON: system=tool-definitions, user=NL, assistant=`[{"name","arguments"}]` (R4 schema). | R4 | I |
| TRD-DR-002 | M | Action-tag format *shall* follow the deterministic grammar in TRD-IF-005. | R6 | I |
| TRD-DR-003 | M | The semantic store *shall* model entities, relationships, personas, and world/temporal state (graph schema). | R7 | I |
| TRD-DR-004 | M | Episodic records *shall* be time-stamped and carry decay/TTL. | R7 | T |
| TRD-DR-005 | M | All user-derived records *shall* carry a data-classification tag (PII / sensitive / operational) and consent + retention metadata. | S1 | I |
| TRD-DR-006 | M | Action-adapter training data *shall* be stored with provenance (source dataset/version, e.g. `google/mobile-actions`). | R4 | I |
| TRD-DR-007 | S | Export artifacts *shall* record lineage: base model, distillation run, quant scheme, adapter set. | R3 | I |

---

## 8. Security & Compliance Requirements (SR)

| ID | Pri | Requirement | Ref | V |
|----|-----|-------------|-----|---|
| TRD-SR-001 | M | Default inference *shall* be on-device; raw screen, voice, and PII *shall not* leave the device on the fast path. | S1 | T |
| TRD-SR-002 | M | Network access *shall* be opt-in and routed only through the gateway Worker. | R6,S1 | T |
| TRD-SR-003 | M | API keys/secrets *shall* reside only in the gateway Worker, never in client binaries. | R6 | I |
| TRD-SR-004 | M | Memory stores *shall* implement TTL and user-triggered erasure (right to erasure). | S1 | T |
| TRD-SR-005 | M | Graph/episodic records *shall* enforce consent + retention metadata at read/write. | S1 | T |
| TRD-SR-006 | M | All cloud egress and executed actions *shall* be auditable (TRD-NFR-OBS-001). | S1 | I |
| TRD-SR-007 | S | Data at rest (memory stores) *shall* be encrypted on device. | S1 | T |
| TRD-SR-008 | M | SIA source *shall not* include AGPL-licensed code (license hygiene; DD-3). | R7 | I |

---

## 9. Technology Stack (definitive)

| Concern | Choice | Source |
|---------|--------|--------|
| Substrate model | LFM2.5 (8B-A1B / 1.2B-Instruct / 1.2B-Thinking-ONNX) | R1 |
| Deep reasoning core | RDT looped-MoE (Prelude/Recurrent/Coda, MLA, MoE, ACT) | R2 |
| Training / distill / quant | DeepSpeed-MoE, ZeRO++, Mixture-of-Students, ZeroQuant | R3 |
| Action adapters | LoRA via Unsloth + TRL (PEFT) | R4 |
| Native runtime | Ollama (GGUF) | R1/S2 |
| Browser runtime | Transformers.js + ONNX Runtime Web + WebGPU | R5/S2 |
| Embodiment shell | Screen capture + STT + TTS + action-tag executor (clicky pattern) | R6 |
| Gateway | Cloudflare Worker (key custody, proxy) | R6 |
| Swarm runtime | OASIS/CAMEL-pattern multi-agent + GraphRAG (re-implemented) | R7 |
| Working memory | TokenCake (+ MLA KV) | Internal/R2 |
| Cloud/VPS | Hostinger + burst GPU | A2 |

---

## 10. Performance Budgets & SLAs

| Metric | Edge-Standard (1.2B) | Node-Capable (8B-A1B) | Browser (Thinking ONNX) |
|--------|----------------------|------------------------|--------------------------|
| Fast-path first token | ≤ 5 s CPU / < 1 s NPU | < 1 s GPU | interactive stream |
| Action call (device-actions) | ≤ 5 s typical | ≤ 3 s | n/a |
| Deep-path max loops T | governor-capped (hot→1) | up to 16 (cool/charging) | low/none |
| Model RAM | ~1–2 GB | ~5.2 GB | ~1–2 GB |
| Concurrency benefit | — | ≥ 2× via depth-wise batching | — |
| Fast-path traffic share | — | ≥ 80% | — |

---

## 11. Deployment & Environment Requirements

| ID | Pri | Requirement | V |
|----|-----|-------------|---|
| TRD-FR-SYS-001 | M | Edge plane *shall* run LFM2.5 + SIR fast path + active LoRA + shell; deep path only where RAM/thermal permit. | T |
| TRD-FR-SYS-002 | M | Browser/PWA plane *shall* run LFM2.5-Thinking ONNX via WebGPU, zero-server. | T |
| TRD-FR-SYS-003 | M | Cloud/VPS plane *shall* host the full RDT-MoE core, swarm sim, and the DeepSpeed pipeline. | T |
| TRD-FR-SYS-004 | M | Gateway plane (Worker) *shall* mediate any cloud API access and hold all secrets. | I |
| TRD-FR-SYS-005 | S | Edge↔VPS offload of the deep path *shall* be possible under SLA-critical/low-headroom conditions. | T |

---

## 12. Verification & Acceptance Criteria (by phase)

| Phase | Exit criteria (acceptance) | Covers |
|-------|----------------------------|--------|
| **P0 Substrate** | LFM2.5 runs on phone (Ollama) and browser (ONNX/WebGPU); tool-call + 125K ctx demonstrated on Node-Capable. | SUB-001..006, PERF-001/004, POR-001 |
| **P1 Action** | `device-actions` adapter trained per R4; ≥ 95% function-call accuracy on SIA tool set; ≤ 5 s typical. | ACT-001..004, PERF-002, RES-002 |
| **P2 Shell** | End-to-end "see screen → reason → point/act + speak" on one platform; shared dispatcher verified. | EMB-001..008, IF-004/005 |
| **P3 Deep core** | RDT-MoE up-cycled from LFM2.5; router live; deep path beats fast path on a multi-hop eval; governor throttles T under thermal load; stable training (ρ(A)<1). | SIR-020..030, GOV-001..006, REL-001/002 |
| **P4 Swarm+distill** | Swarm 5-stage loop runs (operational + sim); a Mixture-of-Students cycle measurably lifts on-device SIR vs baseline. | SWM-001..006, TDP-001..006, SCA-002 |
| **P5 Harden** | Operates within thermal/battery SLA; DPDP hooks (locality, TTL, erasure, audit) pass; quant matrix shipped; OTA adapters. | THM-*, SR-001..008, MNT-001 |

---

## 13. Traceability Matrix (requirement → reference → phase → verify)

| Requirement group | Ref | Phase | Verify |
|-------------------|-----|-------|--------|
| SUB-001..006 | R1, R5 | P0 | T/A |
| SIR-001..011 (router/fast) | R1, R2 | P0/P3 | T/A |
| SIR-020..030 (deep core) | R2 | P3 | I/T |
| MEM-001..007 | R2, R7 | P2/P3 | T/I |
| ACT-001..007 | R4 | P1 | T/I |
| EMB-001..008 | R6 | P2 | T/D |
| SWM-001..006 | R7 | P4 | T/D |
| TDP-001..006 | R2, R3 | P3/P4 | T/A |
| GOV-001..006 | internal | P3/P5 | T/A |
| NFR PERF/RES/THM | R1, R2, R4 | P0–P5 | T/A |
| SR-001..008 | S1, R6 | P5 | T/I |

---

## 14. Risks & Mitigations

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|-----------|------------|
| RK-1 | Up-cycle (dense→MoE) underperforms vs from-scratch | Med | Med | Validate on multi-hop eval at P3; fall back to OpenMythos scratch training for a small tier if needed (DD-1 review) |
| RK-2 | Looped training instability | High | Low | LTI ρ(A)<1 by construction (R2); ACT halting; monitor spectral radius |
| RK-3 | Phone can't run any deep reasoning | Med | Med | DD-2: phones use LFM2.5-Thinking; optional distilled looped student later |
| RK-4 | AGPL contamination from MiroFish | High | Low | DD-3: re-implement pattern only; license inspection at P4 (SR-008) |
| RK-5 | WebGPU availability/perf on target browsers | Med | Med | Provide Ollama/native fallback (POR-001); feature-detect WebGPU |
| RK-6 | Edge action accuracy below 95% on SIA tools | Med | Low | R4 recipe reached 100% on Android tools; expand/curate SIA tool dataset; response-only loss |
| RK-7 | Thermal/latency SLA missed under load | Med | Med | Governor + depth-wise batching + VPS offload (GOV-005) |
| RK-8 | Compute budget overrun | Med | Med | ZeRO++ + ZeroQuant; grant-first; spot GPUs (§17 of ARCH) |
| RK-9 | Upstream model/license change (LFM2.5) | Med | Low | Pin versions; keep export pipeline runtime-agnostic |

---

## 15. Glossary

| Term | Meaning |
|------|---------|
| SIA | Swarm-brain Integration Architecture — the overall system |
| SIR | On-device reasoner node (the brain), runs fast + deep paths |
| Substrate | The base foundation model (LFM2.5 family) |
| Fast path | LFM2.5 direct inference, no looping |
| Deep path | RDT looped-MoE reasoning core |
| RDT | Recurrent-Depth Transformer (looped: Prelude/Recurrent/Coda) |
| MoE | Mixture-of-Experts (routed + shared experts) |
| MLA | Multi-Latent Attention (compressed KV latent) |
| GQA | Grouped-Query Attention (alt. attention) |
| ACT | Adaptive Computation Time (learned loop halting) |
| ρ(A) | Spectral radius of the loop injection matrix; must be < 1 |
| LoRA | Low-Rank Adaptation (skill adapters) |
| TokenCake | SIA working-memory/KV manager |
| GraphRAG | Graph-structured retrieval-augmented memory |
| Mixture-of-Students | DeepSpeed MoE→dense distillation method |
| ZeroQuant | DeepSpeed post-training quantization (INT4/INT8/FP6) |
| Governor | Single resource/thermal controller |
| Node-Capable / Edge-Standard / Edge-Think | Device tiers (§5 ARCH) |
| DPDP | Digital Personal Data Protection Act, India |

---

## 16. Appendices

### Appendix A — Deep-core baseline config (from R2 `mythos_1b`/`mythos_3b`)
```
attn_type        = "mla"            # DD-4
n_experts        = 64 (1B) / 128 (3B)
n_experts_per_tok= 2
n_shared_experts = 1
expert_dim       = 2048 (1B) / 4096 (3B)
dim              = 2048 (1B) / 3072 (3B)
max_loop_iters   = 16              # ACT halts earlier
loop_index_embed = on
halting          = ACT
stability        = A = Diag(-exp(log_A)), ρ(A) < 1
depth_lora       = optional
```

### Appendix B — Action-adapter recipe (from R4, proven)
```
base    = LiquidAI/LFM2.5-1.2B-Instruct
method  = Unsloth + TRL SFT, response-only loss
lora    = r=16, alpha=16, dropout=0
targets = q_proj,k_proj,v_proj,out_proj,in_proj,w1,w2,w3   (~0.94% params)
train   = 3 epochs, lr 2e-4 linear, AdamW-8bit, seq 2048
hw/cost = ~2 hrs on 1× L4  (≈ ₹300–₹700 spot)
data    = {system: tool-defs JSON, user: NL, assistant: [{"name","arguments"}]}
target  = ≥ 95% held-out function-call accuracy on SIA tool set
```

### Appendix C — Pipeline stages (from R3)
```
A  up-cycle LFM2.5 → RDT-MoE     (DeepSpeed-MoE + ZeRO++, TED parallelism)
B  loop-scaling / reasoning RL   (scale recurrence + data, not params)
C  swarm distillation            (Mixture-of-Students → edge student)
D  quantize                      (ZeroQuant INT8 baseline; INT4/FP6 eval)
E  export                        (GGUF + ONNX + LoRA adapters, versioned)
```

### Appendix D — Open items requiring founder decision
1. Confirm DD-1 (up-cycle vs scratch).
2. Confirm DD-2 (phone deep-path policy / device floor).
3. Confirm DD-3 (MiroFish re-implement vs adopt + AGPL acceptance).
4. Swarm transport: VPS mesh vs P2P vs hybrid (affects SWM/IF-008).
5. SIR full-name/scope confirmation for glossary.

---

*End of SIA-TRD-001 v1.0*
