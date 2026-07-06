# Product Requirements Document (PRD)
# SIA — The Private On-Device AI Companion

---

## 1. Document Control

| Field | Value |
|-------|-------|
| Document title | SIA — Product Requirements Document |
| Document ID | SIA-PRD-001 |
| Version | 1.0 |
| Status | Draft for review |
| Author | SIA Product (solo founder) |
| Classification | Confidential — internal |
| Date | 18 Jun 2026 |
| Related docs | SIA-ARCH-001 (Foundation Architecture); SIA-TRD-001 (Technical Requirements); SIR Identity Charter; DPDP Compliance Note |

### 1.1 Revision history
| Ver | Date | Author | Change |
|-----|------|--------|--------|
| 1.0 | 18 Jun 2026 | SIA Product | First PRD; product scope, personas, metrics, roadmap baselined |

### 1.2 Relationship to other docs
This PRD defines **what** SIA delivers and **for whom**. The **TRD (SIA-TRD-001)** specifies **how** it is built. Where useful, product requirements below cite the TRD requirement IDs they depend on.

---

## 2. Executive Summary

SIA is a **private, always-present AI companion that lives on your device** — it sees your screen, listens and talks, reasons (lightly or deeply as the moment demands), and *acts* on your behalf through device functions and on-screen control. Unlike cloud assistants, SIA runs **on-device by default**: your screen, voice, and personal data don't leave the device for everyday use. SIA nodes also form a **swarm** whose collective experience is distilled back into each device — so every SIA gets smarter over time **without anyone's private data being shared**.

The wedge: **privacy you can verify + an assistant that actually does things, built India-first for a DPDP world.** Cloud assistants are capable but custodial; on-device assistants are private but usually shallow. SIA is private *and* capable — small in storage, deep on demand.

---

## 3. Problem & Opportunity

### 3.1 The problem
- **Cloud assistants are custodial.** Every screenshot, voice clip, and document is shipped to a third-party cloud. For privacy-conscious users and for India's DPDP regime, that's a liability, not a feature.
- **Most assistants chat but don't act.** They produce text; the user still does the clicking, typing, and app-switching.
- **On-device AI is usually too shallow** to handle genuinely hard reasoning — or it drains the battery trying.
- **Assistants don't compound.** They don't get meaningfully smarter from collective use without harvesting personal data.

### 3.2 The opportunity
A new class of foundation models (LFM2.5-class, sparse-active, edge-native) plus looped-MoE reasoning makes it possible to run a *capable* assistant locally, spend deep compute only when needed, and improve the on-device model through privacy-preserving swarm distillation. India-first positioning aligns with DPDP and with a massive on-device-capable device base.

---

## 4. Product Goals & Non-Goals

### 4.1 Goals (G)
| ID | Goal |
|----|------|
| G1 | Deliver an assistant that **runs on-device by default** — everyday use needs no cloud and ships no personal data off-device. |
| G2 | Make SIA **act, not just answer** — complete real device tasks and on-screen actions. |
| G3 | Provide **two-speed intelligence** — instant for the everyday, deep reasoning when it matters, without wrecking battery/thermals. |
| G4 | Be a **trustworthy, embodied companion** — sees the screen, talks, points/acts, always available, never intrusive. |
| G5 | **Compound over time** via swarm distillation — every node improves without private data leaving devices. |
| G6 | Be **India-first and DPDP-aligned**, brandable globally. |

### 4.2 Non-goals (this release line) (NG)
| ID | Non-goal |
|----|----------|
| NG1 | Not a cloud-first chatbot; cloud is opt-in, not the default. |
| NG2 | Not a full third-party app marketplace at launch (skills/adapters are first-party + curated initially). |
| NG3 | Not shipping the full deep reasoning core to low-end phones at launch (see DD-2 in TRD). |
| NG4 | Not building domain *content* products here (ASTRO/creative are internal dogfood customers, not PRD scope). |

---

## 5. Success Metrics

### 5.1 North Star
**Daily on-device tasks completed per active user** — captures real value delivery, the act-don't-chat promise, and the on-device/privacy commitment in a single number.

### 5.2 KPIs & targets
| Area | Metric | Target (v1.0) |
|------|--------|---------------|
| Value | Daily tasks completed / active user (North Star) | ≥ 5 |
| Privacy (guardrail) | % of intents resolved fully on-device (default path) | ≥ 95% |
| Action quality | Device-action success rate | ≥ 95% (TRD-NFR-PERF / ACT) |
| Reasoning | Deep-path "win rate" vs fast path on hard-task eval | deep > fast, measurable (TRD P3) |
| Speed | Fast-path first-token latency | < 1 s NPU/GPU; ≤ 5 s CPU (TRD-NFR-PERF-001) |
| Trust | Thermal/battery SLA compliance (no overheating, acceptable drain) | 100% under governor (TRD-NFR-THM) |
| Compounding | Capability lift per swarm-distillation cycle | measurable uplift vs baseline (TRD-TDP-006) |
| Retention | D7 / D30 active retention | establish baseline → improve |
| Efficiency | Marginal cost per on-device inference | ≈ ₹0 |
| Activation | Time-to-first-completed-action (new user) | < 5 min |

---

## 6. Target Users & Personas

| Persona | Who | Core need | Why SIA |
|---------|-----|-----------|---------|
| **Aarav — Privacy-conscious professional** (India) | Works on sensitive docs; wary of US-cloud data flows | An assistant that doesn't exfiltrate his screen/voice/data | On-device default; DPDP-aligned; verifiable locality |
| **Meera — The Delegator** | Busy multitasker on phone + laptop | Wants tasks *done*, not described | Acts on-screen + device functions; companion shell |
| **Dev — The Builder** | Developer/tinkerer | Wants to extend the assistant with custom skills | Hot-swap LoRA adapters; skills/tool bus; open patterns |
| **Internal — ASTRO Ops & ToneRoom** (dogfood) | Founder's own ventures | Domain actions (VTOL telemetry, creative pipeline) | `astro-ops` / `creative-ops` adapters; swarm sim for rehearsal |

**Primary persona for v1.0:** Aarav (privacy) + Meera (action). Dev is v-Next. Internal dogfooding runs throughout.

---

## 7. Jobs To Be Done (JTBD)

- **J1** *When I'm working on my screen,* I want help that can **see what I see and act**, *so I* don't copy-paste context or switch apps constantly.
- **J2** *When I hand off a task,* I want it **done on my device without my data leaving**, *so I* stay private and compliant.
- **J3** *When a problem is genuinely hard,* I want **deeper thinking on demand**, *so I* get real answers — but not at the cost of my battery on every trivial query.
- **J4** *When I use my assistant daily,* I want it to **get better over time**, *so I* feel it learning — without it harvesting my personal data.
- **J5** *(Builder)* *When I have a repetitive workflow,* I want to **teach SIA a new skill**, *so I* can automate it on-device.

---

## 8. Product Principles (pillars)

1. **Private by default.** On-device is the default; the network is opt-in and visible. (G1, TRD-SR-001/002)
2. **Acts, doesn't just chat.** Every interaction can end in a completed action. (G2)
3. **Spend compute only when it's worth it.** Fast by default; deep on demand; thermally graceful. (G3, TRD-GOV)
4. **Present, not intrusive.** Always reachable companion; never hijacks the screen. (G4, TRD-EMB-007)
5. **Gets smarter together, privately.** Swarm distillation, not data harvesting. (G5)
6. **Small in storage, deep on demand.** Capability from loops + adapters, not bloat. (TRD-NFR-SCA)

---

## 9. Use Cases / User Stories

Each story: actor, narrative, acceptance criteria, TRD dependency.

**UC-1 — On-screen help that acts** *(Meera)*
> "Looking at a form on my screen, I ask SIA to fill it; it sees the screen and points/acts to complete it."
- **Accept:** SIA captures the screen, identifies fields, and executes on-screen actions to completion; no screen data leaves device. → TRD-FR-EMB-001/004, TRD-FR-ACT-006, TRD-SR-001
- **Priority:** Now (P2)

**UC-2 — Voice task on the phone** *(Meera)*
> "I say 'remind me to call Ravi at 6 and text him I'm running late'; SIA sets the reminder and drafts/sends the text."
- **Accept:** Voice→intent→correct function calls (`set_alarm`/reminder, `send_message`) executed; ≥95% accuracy; ≤5 s. → TRD-FR-ACT-001/004, TRD-FR-EMB-002, TRD-NFR-PERF-002
- **Priority:** Now (P1)

**UC-3 — Private everyday Q&A** *(Aarav)*
> "I ask about a sensitive document open on my screen; the answer is computed on-device."
- **Accept:** Inference runs locally; no PII/screen egress on default path; fast-path latency met. → TRD-FR-SUB-004, TRD-SR-001, TRD-NFR-PERF-001
- **Priority:** Now (P0/P2)

**UC-4 — Deep reasoning on demand** *(Aarav/Meera)*
> "I give SIA a multi-step planning problem; it 'thinks harder' and returns a reasoned plan."
- **Accept:** Router escalates to deep path; visible thinking state; deep path outperforms fast path on the eval; governor respects thermal budget. → TRD-FR-SIR-002/020, TRD-NFR-USA-001, TRD-FR-GOV-003/004
- **Priority:** Next (P3)

**UC-5 — Don't overheat my phone** *(all)*
> "When my phone is hot or low on battery, SIA stays responsive without making it worse."
- **Accept:** Under hot/low-battery, deep path is suppressed (T→1), responses still useful; no thermal shutdown. → TRD-FR-GOV-003, TRD-NFR-THM-001
- **Priority:** Now→Next (P3/P5)

**UC-6 — Browser-only, zero-install** *(Aarav/Dev)*
> "I use SIA in my browser with no server; the model runs locally via WebGPU."
- **Accept:** LFM2.5-Thinking runs client-side; streams tokens; no server inference. → TRD-FR-SUB-003, TRD-NFR-PERF-004
- **Priority:** Now (P0)

**UC-7 — Teach a new skill** *(Dev)*
> "I add a custom skill; SIA gains a new action domain via an adapter."
- **Accept:** A new LoRA adapter loads and is invocable; only the active adapter is resident. → TRD-FR-ACT-002/003/007
- **Priority:** Next (P1/P4)

**UC-8 — It learns from the swarm** *(all)*
> "After a while, SIA handles things it used to struggle with — without me sending my data anywhere."
- **Accept:** A swarm-distillation cycle measurably lifts on-device capability vs baseline; user data not shared. → TRD-FR-TDP-003/006, TRD-FR-SWM-004
- **Priority:** Later (P4)

**UC-9 — Rehearse-the-future (internal/advanced)** *(ASTRO Ops)*
> "I simulate a mission/scenario with many agents and inject variables to test outcomes."
- **Accept:** Swarm runs the 5-stage sim with personas; God-view variable injection works; a report is produced. → TRD-FR-SWM-001/003/005
- **Priority:** Later (P4)

---

## 10. Feature Requirements (prioritized)

### 10.1 NOW — MVP (maps to TRD P0–P2)
| ID | Feature | Persona | TRD link |
|----|---------|---------|----------|
| PF-N1 | On-device model running (native + browser/WebGPU) | Aarav | SUB-001..006 |
| PF-N2 | Companion shell: sees screen, voice in/out, points/acts | Meera | EMB-001..008 |
| PF-N3 | Device actions (alarms, calls, messages, calendar, maps, toggles) | Meera | ACT-001..004 |
| PF-N4 | Private everyday Q&A (on-device default, no PII egress) | Aarav | SUB-004, SR-001 |
| PF-N5 | Fast, responsive everyday latency | all | NFR-PERF-001 |
| PF-N6 | Visible "thinking/working" state | all | NFR-USA-001 |

### 10.2 NEXT (maps to TRD P3–P4)
| ID | Feature | Persona | TRD link |
|----|---------|---------|----------|
| PF-X1 | Two-speed reasoning (router + deep looped-MoE core) | Aarav/Meera | SIR-001..030 |
| PF-X2 | Resource/thermal governor (deep on demand, graceful) | all | GOV-001..006 |
| PF-X3 | Extended on-screen action vocabulary (tap/scroll/type/gesture) | Meera | EMB-005 |
| PF-X4 | Custom skill adapters (builder-facing) | Dev | ACT-002/007 |
| PF-X5 | Swarm runtime (operational delegation + simulation) | internal | SWM-001..003 |

### 10.3 LATER (maps to TRD P4–P5)
| ID | Feature | Persona | TRD link |
|----|---------|---------|----------|
| PF-L1 | Swarm distillation → on-device gets smarter (privately) | all | TDP-003/006, SWM-004 |
| PF-L2 | Full DPDP hardening (TTL, erasure, on-device encryption, audit) | Aarav | SR-001..008 |
| PF-L3 | Rehearse-the-future simulation product (advanced/internal) | ASTRO Ops | SWM-005 |
| PF-L4 | Edge↔VPS offload for SLA-critical deep tasks | Meera | SYS-005 |
| PF-L5 | Distilled looped student for phones (optional) | all | DD-2 review |

---

## 11. MVP Definition

**MVP = "A private companion that sees, talks, and does."**

Concretely (NOW features PF-N1…N6): SIA runs LFM2.5 on-device (and in-browser via WebGPU), presents an always-available companion shell that can see the screen and use voice, completes core device actions with ≥95% accuracy, answers everyday questions locally without shipping personal data, and shows a thinking state. Deep reasoning, the governor, custom adapters, and the swarm are explicitly **post-MVP**.

**MVP success bar:** a privacy-conscious user (Aarav) and a delegator (Meera) can complete ≥5 on-device tasks/day, with ≥95% of intents resolved locally, action success ≥95%, time-to-first-action < 5 min.

---

## 12. User Experience & Key Flows

### 12.1 Companion presence
- Always-available, non-blocking (menu-bar/overlay on desktop; persistent assistant surface on mobile). Never seizes the screen. (TRD-FR-EMB-007)
- Explicit, first-run permission grants (mic, screen capture, accessibility) with plain-language rationale. (TRD-FR-EMB-008)

### 12.2 Core interaction loop
```
User speaks / types / shares screen
  → SIA perceives (screen frame + voice)
  → SIR reasons (fast by default; deep if hard)
  → SIA responds: speaks + points/acts + completes the task
  → episode remembered locally; (later) pooled to swarm privately
```

### 12.3 Trust & transparency cues
- A clear **on-device indicator** (this ran locally) vs an explicit **cloud prompt** when a task would need the network. (G1, TRD-SR-002)
- A visible **thinking state** during deep reasoning so the user understands when SIA is "working harder." (TRD-NFR-USA-001)
- Battery/thermal-aware behavior is **felt as responsiveness**, not surfaced as errors. (TRD-FR-GOV-006)

---

## 13. Product-level Non-Functional Expectations

These are the user-facing *promises* the TRD NFRs must uphold.

| Promise | Backed by |
|---------|-----------|
| "It's private — your data stays on your device by default." | TRD-SR-001/002, NFR-OBS-001 |
| "It's fast — everyday tasks feel instant." | TRD-NFR-PERF-001/002 |
| "It won't cook your phone or kill your battery." | TRD-NFR-THM-001/002, GOV |
| "It works without a server — even in the browser." | TRD-FR-SUB-003, POR-001 |
| "It actually finishes tasks." | TRD-FR-ACT/EMB |
| "It gets smarter over time, privately." | TRD-FR-TDP-003/006 |

---

## 14. Release Plan / Roadmap

| Release | Theme | Product scope | TRD phase | Audience |
|---------|-------|---------------|-----------|----------|
| **v0.1 (internal/alpha)** | Substrate on device | PF-N1, PF-N4 | P0 | Dogfood |
| **v0.5 (closed beta)** | It does things | + PF-N3 (actions), PF-N5 | P1 | Aarav/Meera (closed) |
| **v1.0 (MVP launch)** | Private companion that sees, talks, does | + PF-N2 (shell), PF-N6 | P2 | India-first GA |
| **v1.5** | Thinks harder, on demand | + PF-X1, PF-X2 | P3 | GA |
| **v2.0** | Builder + swarm | + PF-X3/X4/X5, PF-L1 (first distill) | P4 | + Dev |
| **v2.5** | Hardened & compounding | + PF-L2/L4/L5, PF-L3 | P5 | GA |

---

## 15. Dependencies & Assumptions

- **DEP/ASSUMPTIONS inherited from TRD §3.4** (LFM2.5 availability, edge runtimes, DeepSpeed toolchain, permissive licenses for R2/R4/R6).
- **Product assumption PA-1:** Target devices have an on-device-capable accelerator or sufficient CPU/RAM for at least the 1.2B tier.
- **Product assumption PA-2:** Privacy/locality is a real purchase driver for the India-first segment (DPDP tailwind).
- **Product assumption PA-3:** Curated first-party skills are sufficient at launch; open skill ecosystem follows.

---

## 16. Risks (product / adoption)

| ID | Risk | Impact | Mitigation |
|----|------|--------|------------|
| PR-1 | Users don't perceive on-device privacy as worth a capability trade-off | High | Lead with action value (Meera) + verifiable privacy cues; benchmark capability vs cloud on everyday tasks |
| PR-2 | Device fragmentation → inconsistent performance | Med | Tiered model (8B-A1B / 1.2B / Thinking) + native + WebGPU fallback (TRD-POR-001) |
| PR-3 | "Acts on my screen" feels risky/creepy | Med | Explicit permissions, on-device indicator, undo/confirm on consequential actions |
| PR-4 | Deep reasoning value not obvious to users | Med | Visible thinking state; reserve deep path for clearly hard tasks; show before/after |
| PR-5 | Swarm-learning misread as data harvesting | High | Clear "your data never leaves" messaging; distillation operates on pooled/derived, not raw personal data |
| PR-6 | Scope creep delays MVP | Med | Hard MVP line (§11); deep/governor/swarm strictly post-MVP |
| PR-7 | Battery/thermal complaints | Med | Governor as a first-class product behavior; conservative defaults |

---

## 17. Pricing & Packaging (directional, India-first)

- **Free tier:** on-device companion + core device actions (privacy + action value; drives adoption). Marginal inference cost ≈ ₹0 supports a generous free tier.
- **Pro (₹, monthly):** deep reasoning unlocked, custom skills/adapters, priority distillation updates, advanced shell actions.
- **Builder/Dev:** skill SDK + adapter tooling.
- **Internal/Enterprise (later):** on-prem swarm, domain adapters (ASTRO/creative), compliance pack.
- Funding posture: **grant-first** for the platform build (per founder strategy); revenue via Pro/Builder.

*Detailed pricing belongs in a separate business/GTM doc; included here only as packaging intent.*

---

## 18. Open Questions

1. **Primary launch surface** — phone-first, desktop-first, or browser-first for v1.0? (affects shell build order)
2. **Device floor** — minimum supported device tier for GA (ties to TRD DD-2).
3. **Consequential-action policy** — which on-screen actions require explicit confirm/undo?
4. **Skill ecosystem timing** — first-party-only at launch vs early builder access (PA-3).
5. **Privacy proof** — how do we let users *verify* on-device execution (indicator, audit view, attestation)?
6. **Swarm participation** — opt-in vs default-on (with clear privacy framing) for distillation contribution.

---

## 19. Glossary (brief — see TRD §15 for full)

| Term | Meaning |
|------|---------|
| SIA | The product: private on-device AI companion (and the swarm of nodes) |
| SIR | The on-device reasoner (fast + deep paths) |
| Fast / Deep path | Instant LFM2.5 inference vs deeper looped-MoE reasoning |
| Companion shell | The see/talk/point-and-act surface (embodiment layer) |
| Adapter (skill) | A hot-swappable LoRA giving SIA a new action domain |
| Swarm distillation | Privacy-preserving way the on-device model improves from collective use |
| DPDP | India's Digital Personal Data Protection Act |

---

## 20. Appendix — Product → TRD traceability

| Product feature | TRD requirements |
|-----------------|------------------|
| PF-N1 substrate | SUB-001..006, SUB-003 (browser) |
| PF-N2 shell | EMB-001..008, IF-004/005 |
| PF-N3 actions | ACT-001..004, ACT-006 |
| PF-N4 private Q&A | SUB-004, SR-001/002 |
| PF-N5 latency | NFR-PERF-001 |
| PF-N6 thinking state | NFR-USA-001 |
| PF-X1 two-speed reasoning | SIR-001..030 |
| PF-X2 governor | GOV-001..006, NFR-THM |
| PF-X3 action vocab | EMB-005 |
| PF-X4 custom adapters | ACT-002/003/007 |
| PF-X5 swarm runtime | SWM-001..003 |
| PF-L1 distillation | TDP-003/006, SWM-004 |
| PF-L2 DPDP hardening | SR-001..008 |
| PF-L4 offload | SYS-005 |

---

*End of SIA-PRD-001 v1.0*
