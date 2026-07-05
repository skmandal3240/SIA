# Sarvam AI tokenizer analysis for SIA

## What Sarvam AI did

Sarvam AI built **OpenHathi**, a Hindi-first LLM based on **Llama 2 7B**.

Their key tokenizer decision:
- Extended the Llama 2 tokenizer with **~48,000 additional tokens**.
- These new tokens cover Devanagari characters, Hindi words, and common Hinglish patterns.
- This lets the model represent Hindi/Hinglish text with far fewer tokens than a pure English tokenizer.

Result: faster inference, lower cost, and better comprehension for Indian languages.

## Why this matters for SIA

SIA targets Indian users who mix Hindi and English. A tokenizer that understands Hindi morphemes is better than a generic English tokenizer.

## What I implemented in SIA

File: `sia/sarvam_tokenizer.py`

It uses **SentencePiece BPE**, the same underlying technology as Llama / OpenHathi, and trains a tokenizer on a combined English + Hindi/Hinglish corpus.

## Example output

```text
Input:  6 बजे अलार्म लगाओ
Tokens: [19, 141, 170, 163]
Decoded: 6 बजे अलार्म लगाओ
Vocab size: 256
```

The Hindi sentence is encoded in just **4 tokens** because the tokenizer learned whole Hindi words/morphemes.

## Next step

To make SIA truly strong for Indian languages:
1. Collect 1,000+ Hindi/Hinglish device-action examples.
2. Retrain `sia_tokenizer.model` with `vocab_size=8,000–16,000`.
3. Use this tokenizer in `sia/model.py` and retrain the SIA model.

This mirrors Sarvam's approach without copying any of their proprietary code.
