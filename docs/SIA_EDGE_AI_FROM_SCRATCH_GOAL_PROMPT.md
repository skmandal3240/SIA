# SIA Model Build — Goal Prompt (Fable 5)

I'm building SIA, a private, on-device AI companion for India (DPDP-compliant, privacy-by-locality). The foundation architecture is frozen in this project's docs, but the model workspace is still scaffolding. I need it turned into a real, buildable, published codebase: the SIA tokenizer implemented (informed by Sarvam AI's Indic tokenizer work), the model build pipeline runnable end to end, and everything pushed to a GitHub repo that grant reviewers and future collaborators can point at. With that in mind, here is the mission.

## Ground truth (read before building)

- `SIA_Foundation_Architecture.md` — the frozen spec: LFM2.5 substrate (L0), fast/deep SIR reasoner (L1), LoRA action adapters (L3), the DeepSpeed train → distill → quantize → export pipeline (§8), build phases P0–P5 (§15), and the license posture for dependencies (§16.5).
- `PROJECT/sia-lab/` — the model workspace. `pretrain/tokenizer/__init__.py` imports `sia_tokenizer.py` which **does not exist yet** — that stub defines the contract you must implement (`SIA_TOKENIZER`, `VocabSpec`, `load_or_train`, `adapt_base_tokenizer`). Also: `posttrain/sft.py`, `infra/{quantize,benchmark}.py`, `safety/`, `product/`.
- `PROJECT/models/` — LFM2.5-8B-A1B GGUF (5.2 GB) with an Ollama Modelfile.
- `PROJECT/Makefile` — `make ci` runs lint + validate + smoke + status; `make privacy` is the network-egress test.
- `PROJECT/docs/{PRD,SRS,TRD}.md` — product constraints.

## Workstream 1 — Study Sarvam AI's tokenizer, then implement SIA's

Sarvam AI (https://www.sarvam.ai) builds sovereign Indic LLMs. Their tokenizer achieves much lower fertility (tokens per word) on Indic scripts than Western tokenizers — which is exactly what SIA needs for Hindi/Bhojpuri/Indian-English users on thermally constrained devices, where every wasted token is wasted battery.

1. Research their published models and materials — the website, engineering blog, and Hugging Face `sarvamai/` org (Sarvam-1, Sarvam-2B, Sarvam-M). Extract the tokenizer design facts: vocab size, script coverage, corpus balance, measured fertility per language, and the license of each tokenizer artifact. Write the findings with sources to `sia-lab/pretrain/tokenizer/SARVAM_RESEARCH.md` — design facts, not marketing.
2. Implement `sia-lab/pretrain/tokenizer/sia_tokenizer.py` satisfying the existing `__init__.py` contract. Two acceptable routes: adapt Sarvam's own tokenizer via `adapt_base_tokenizer` **if and only if its license permits reuse** (verify first; record the license decision in the research memo), or train a SentencePiece/BPE tokenizer with Sarvam-style Indic corpus balance via `load_or_train`. Either way, include SIA's special tool/action tokens.
3. Benchmark fertility on Hindi, Bhojpuri/Bihari-language, Indian-English, and code samples against the LFM2.5 tokenizer and one Western baseline (e.g. GPT-4/Llama tokenizer). Commit the benchmark harness and the results table.
4. Be explicit in the docs about where this tokenizer applies: it serves the from-scratch pretrain path and future distilled students (§8 Stage C). It is **not** a drop-in swap for the pretrained LFM2.5 GGUF — changing the tokenizer under pretrained weights breaks the embedding table. Document the real integration path (vocab adaptation / embedding re-init at distillation time).

## Workstream 2 — Make the model build pipeline real

Per the architecture's build phases, get as far as this machine allows and leave the rest ready-to-run:

- **P0 substrate:** verify LFM2.5 runs locally through the Ollama Modelfile (text generation + a tool-call), and record the result.
- **P1 action adapter:** the device-actions LoRA per the mobile-actions recipe (Unsloth + TRL SFT, LoRA r=16/α=16 on q/k/v/out/in/w1/w2/w3, response-only loss). Full training needs a GPU this machine likely lacks — if so, deliver the complete pipeline with a passing dry-run/smoke test and document the single GPU command left to run. Do not fake training metrics.
- Wire everything into the Makefile so `make ci` passes clean, including `make privacy`.

## Workstream 3 — Publish to GitHub

Create a repo for the model codebase (sia-lab plus whichever demo/verification dirs belong with it — use judgment and record the boundary in the README). Requirements: **private** by default; `.gitignore` excludes model weights (`*.gguf`), venvs, `node_modules`, `__pycache__`, and datasets over a few MB; a README that orients a new engineer in one read (what SIA is, the layer map, how to run `make ci`, the tokenizer benchmark results); a LICENSE consistent with §16.5 of the architecture doc (permissive deps only — never vendor AGPL code such as MiroFish; re-implement patterns instead). Use `gh` to create and push, then tag `v0.1.0`.

## Operating rules

When you have enough information to act, act. Do not re-derive facts already established in the conversation, re-litigate a decision the user has already made, or narrate options you will not pursue in user-facing messages. If you are weighing a choice, give a recommendation, not an exhaustive survey. This does not apply to thinking blocks.

Don't add features, refactor, or introduce abstractions beyond what the task requires. A bug fix doesn't need surrounding cleanup and a one-shot operation usually doesn't need a helper. Don't design for hypothetical future requirements: do the simplest thing that works well. Avoid premature abstraction and half-finished implementations. Don't add error handling, fallbacks, or validation for scenarios that cannot happen. Trust internal code and framework guarantees. Only validate at system boundaries (user input, external APIs). Don't use feature flags or backwards-compatibility shims when you can just change the code.

You are operating autonomously. The user is not watching in real time and cannot answer questions mid-task, so asking "Want me to…?" or "Shall I…?" will block the work. For reversible actions that follow from the original request, proceed without asking. Offering follow-ups after the task is done is fine; asking permission after already discussing with the user before doing the work is not. Before ending your turn, check your last paragraph. If it is a plan, an analysis, a question, a list of next steps, or a promise about work you have not done ("I'll…", "let me know when…"), do that work now with tool calls. End your turn only when the task is complete or you are blocked on input only the user can provide.

Before reporting progress, audit each claim against a tool result from this session. Only report work you can point to evidence for; if something is not yet verified, say so explicitly. Report outcomes faithfully: if tests fail, say so with the output; if a step was skipped, say that; when something is done and verified, state it plainly without hedging.

Pause for the user only when the work genuinely requires them: a destructive or irreversible action, a real scope change, or input that only they can provide. If you hit one of these, ask and end the turn, rather than ending on a promise.

Establish a method for checking your own work at the end of each workstream. Run it after each workstream completes, verifying your work with subagents against `SIA_Foundation_Architecture.md` and this prompt before moving to the next.

Keep working lessons in `sia-lab/.notes/`. Store one lesson per file with a one-line summary at the top. Record corrections and confirmed approaches alike, including why they mattered. Don't save what the repo or chat history already records; update an existing note rather than creating a duplicate; delete notes that turn out to be wrong.

Terse shorthand is fine between tool calls (that's you thinking out loud, and brevity there is good). Your final summary is different: it's for a reader who didn't see any of that. If you've been working for a while without the user watching, your final message is their first look at any of it. Write it as a re-grounding, not a continuation of your working thread: the outcome first, then the one or two things you need from them, each explained as if new. When you write the summary at the end, drop the working shorthand. Write complete sentences. Spell out terms. Don't use arrow chains, hyphen-stacked compounds, or labels you made up earlier. When you mention files, commits, flags, or other identifiers, give each one its own plain-language clause. Open with the outcome: one sentence on what happened or what you found. If you have to choose between short and clear, choose clear.
