# SIA Tokenizer

This directory holds the SIA tokenizer for the from-scratch pretrain path and future distilled students.

## Files

- `sia_tokenizer.py` — `SIA_TOKENIZER` class, `VocabSpec`, `load_or_train`, `adapt_base_tokenizer`.
- `benchmark.py` — fertility benchmark on Hindi, Bhojpuri/Bihari, Indian English, and code.
- `SARVAM_RESEARCH.md` — design facts collected from Sarvam AI publications and HuggingFace.

## Scope

This tokenizer is **not** a drop-in replacement for the pretrained LFM2.5 GGUF weights. Changing the tokenizer under pretrained weights breaks the embedding table. The real integration path is:

1. Train the SIA tokenizer (this module) on a full Indic+English corpus.
2. Use it for the **from-scratch pretrain** path or when rebuilding the embedding table in the distillation Stage C.
3. Do not swap it under the existing LFM2.5 GGUF without re-initializing / resizing embeddings.

## Run

```bash
cd sia-lab/pretrain/tokenizer
python3 sia_tokenizer.py      # trains on sample corpus and runs a Hindi roundtrip check
python3 benchmark.py           # prints fertility results table
```

## Notes

The sample corpus (`../corpus/sample_indic.jsonl`) is intentionally small (60 examples) to stay in the repo. A production tokenizer should be trained on a much larger Indic corpus to reach the target vocab size (~49k) and the low fertility reported by Sarvam. The current training auto-caps vocabulary to what the corpus can support and uses `byte_fallback=True` so unseen Indic characters still roundtrip.
