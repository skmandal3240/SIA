# SIA — Private, On-Device AI Companion for India

SIA is an edge-first AI companion that runs locally on phones, laptops, mini-PCs, and browser tabs. It is designed for Indian users and is DPDP-compliant by construction: raw screen, voice, and personal data stay on the device by default.

## Layer map

| Layer | Name | What lives here | Status |
|-------|------|-----------------|--------|
| L0 | Substrate | LFM2.5 base model served via GGUF (Ollama) and ONNX (browser / WebGPU) | P0 dry-run passing |
| L1 | Fast path | Direct LFM2.5 inference for single-turn and chat queries | scaffolded |
| L2 | Action adapter | LoRA device-actions adapter (alarms, calls, messages, maps, toggles) | P1 dry-run passing |
| L3 | Reasoner | SIR two-speed reasoner: fast path + deep recurrent-depth core | planned |
| L4 | Memory | TokenCake working memory + episodic store + GraphRAG semantic store | planned |
| L5 | Swarm | Multi-node delegation, simulation, and Mixture-of-Students distillation | planned |

## Quick start

```bash
make ci        # lint + validate + smoke tests + repo status
make privacy   # network egress test: asserts no unexpected calls during inference
make lint      # ruff + basic formatting checks
make validate  # schema checks for tokenizer, SFT dataset, and manifests
make smoke     # non-GPU dry runs of posttrain, quantize, and benchmark
make status    # print which artifacts exist and what remains dry-run
```

## Tokenizer benchmark results

The tokenizer work is in `sia-lab/pretrain/tokenizer/`. See `BENCHMARK_RESULTS.md` for the full fertility table.

Quick summary (tokens per word, lower is better):

| language | SIA | LFM2.5 | Western baseline (Llama) |
|----------|-----|--------|--------------------------|
| hindi | 4.95 | 6.63 | mock |
| bhojpuri_bihari | 4.46 | 5.70 | mock |
| indian_english | 3.64 | 1.35 | mock |
| code | 4.52 | 2.42 | mock |

SIA's SentencePiece tokenizer is trained on a synthetic Indic-English-code corpus and is designed to be rebuilt at pretraining / distillation time. It is not a drop-in swap for the pretrained LFM2.5 weights.

## Repository structure

```
SIA/
├── sia-lab/
│   ├── pretrain/tokenizer/    # SIA tokenizer + Sarvam research + benchmark
│   ├── pretrain/corpus/       # Indic training samples
│   ├── posttrain/             # action adapter SFT (Unsloth+TRL dry-run)
│   ├── infra/                 # quantization + benchmark stubs
│   ├── safety/                # privacy / network-egress tests
│   └── product/               # P0 substrate verification
├── PROJECT/models/            # Ollama Modelfile (weights excluded from git)
├── Makefile                   # make ci, make privacy, etc.
├── LICENSE
└── README.md
```

## License

MIT — see `LICENSE`. Dependencies are permissive only (MIT / Apache-2.0); no AGPL code is vendored.

## GPU training command

The P1 action-adapter full training is left as one documented command:

```bash
python3 sia-lab/posttrain/sft.py --run --base unsloth/Llama-3.2-1B-Instruct
```

This requires a CUDA GPU with `unsloth` and `trl` installed, roughly 2 hours on a single L4.
