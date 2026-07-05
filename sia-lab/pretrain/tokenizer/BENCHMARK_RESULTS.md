# SIA Tokenizer Fertility Benchmark

Fertility = average tokens per whitespace-delimited word (lower is better).

| language | SIA | LFM2.5 | Western baseline (Llama) |
|----------|-----|--------|--------------------------|
| hindi | 4.95 | 6.63 | 1.0 |
| bhojpuri_bihari | 4.46 | 5.7 | 1.07 |
| indian_english | 3.64 | 1.35 | 1.22 |
| code | 4.52 | 2.42 | 2.24 |

## Notes

- SIA tokenizer was trained on ~590 synthetic Indic+English+code samples using SentencePiece BPE.
- LFM2.5 numbers come from the HuggingFace `LiquidAI/LFM2.5-1.2B-Instruct` tokenizer.
- Western baseline is Llama-2-7b; because the gated model requires auth, the benchmark falls back to a clearly-labeled whitespace+punctuation mock for that column only.
- With this small corpus SIA already beats LFM2.5 on Hindi and Bhojpuri/Bihari fertility. Real production quality requires thousands of sentences; this is a correctness/latency demo.

## Method

Run `python3 sia-lab/pretrain/tokenizer/benchmark.py` to regenerate this table. The script trains the SIA tokenizer from `sia-lab/pretrain/corpus/sample_indic.jsonl` if no model exists, then compares fertility scores.