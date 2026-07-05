# SIA Model Workspace

This repository contains the model workspace for SIA, a private on-device AI companion for India.

## Structure

- `sia-lab/` — model build pipeline and safety checks
  - `pretrain/tokenizer/` — Indic-first tokenizer implementation and benchmarks
  - `posttrain/` — action-adapter SFT pipeline (Unsloth + TRL)
  - `infra/` — quantization, export, and inference benchmark scripts
  - `safety/` — network egress / privacy tests
  - `product/` — product documentation placeholder
- `docs/` — foundation architecture and technical requirements (TRD)
- `models/` — LFM2.5 GGUF artifacts (not committed; see `.gitignore`)
- `Makefile` — `make ci` runs lint, validate, smoke, and status

## Quick start

```bash
make ci
make privacy
```

See `README.md` for the full orientation.

## License

MIT — see `LICENSE`.
