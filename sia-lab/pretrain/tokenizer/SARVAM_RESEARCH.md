# Sarvam AI Tokenizer Research — Design Facts for SIA

Research goal: extract concrete, buildable tokenizer design facts from Sarvam AI's published models and materials to inform the SIA Indic tokenizer.

## Models and tokenizer artifacts

| Model | Parameter scale | Vocabulary size | Reported Indic coverage | Reported corpus / balance | License of artifact | Source |
|-------|-----------------|-----------------|---------------------------|---------------------------|---------------------|--------|
| Sarvam-1 (released Oct 2024) | 2 B | **68,096** (64k active + 4,096 reserved) | 10 Indic languages: Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Oriya, Punjabi, Tamil, Telugu | 2 T tokens total for Sarvam-1; final Sarvam-2B planned for **4 T tokens with 2 T English + 2 T Indic** | **Sarvam non-commercial license** (custom; not OSI-permissive; reuse requires explicit grant) | https://www.sarvam.ai/blogs/sarvam-1, https://huggingface.co/sarvamai/sarvam-1 |
| Sarvam-2B (early checkpoints, e.g. sarvam-2b-v0.5) | 2 B | Not independently verified in public docs; believed to share the Sarvam-1 tokenizer family | 10 Indic languages + English | 2 T tokens (early checkpoint); final planned for 4 T equal English/Indic | **Sarvam non-commercial license / "other"** on HF | https://huggingface.co/sarvamai/sarvam-2b-v0.5, https://huggingface.co/sarvamai/sarvam-1-v0.5 |
| Sarvam-M | 24 B | Not independently verified | 11 Indian languages including Hindi | Fine-tuned from Mistral-Small-3.1-24B-Base for Indic | **Apache-2.0** (per third-party aggregator; verify license file before reuse) | https://www.sarvam.ai/blogs/sarvam-m, https://www.runlocalai.co/models/sarvamai-sarvam-m |
| Sarvam-30B / Sarvam-105B (released Mar 2026) | 30 B / 105 B MoE | Not independently verified; public claim: "outperforms other open-source tokenizers in Indic fertility" | 22 official Indian languages (per press coverage) | MoE from-scratch pretraining | **Apache-2.0** for 105B per HF metadata (per third-party aggregator; verify license file before reuse) | https://www.sarvam.ai/blogs/sarvam-30b-105b, https://huggingface.co/sarvamai/sarvam-105b |

## Fertility claims

Sarvam-1's tokenizer is reported to achieve **fertility rates of 1.4–2.1 tokens per word across all supported Indic languages**, roughly matching English efficiency. This compares with Western/general tokenizers that are reported to use **4–8 tokens per word for Indic languages** in Sarvam's own marketing materials and third-party coverage.

*Sources:*
- https://www.sarvam.ai/blogs/sarvam-1 ("fertility rates of 1.4-2.1 across all supported languages")
- https://huggingface.co/sarvamai/sarvam-1-v0.5 ("average fertility score of ~2")
- https://indiaai.gov.in/article/sarvam-ai-unveils-sarvam-1-optimized-language-model-for-indian-languages
- https://www.sarvam.ai/blogs/sarvam-30b-105b ("outperforms other open-source tokenizers in encoding Indic text efficiently, as measured by the fertility score")

## License decision for SIA

**Do not adapt Sarvam's own tokenizer weights directly.** The released Sarvam-1 / Sarvam-2B tokenizer artifacts are under the **Sarvam non-commercial license**, which is not an OSI-permissive license and requires an explicit license grant for reuse beyond the authorized purposes. That conflicts with SIA's goal of a clean, permissive, grant-review-friendly codebase. The Sarvam-M / 30B / 105B artifacts have been listed with Apache-2.0 metadata on third-party aggregators, but the canonical Hugging Face license files must be checked before any adaptation.

**Chosen path for SIA:** train a new SentencePiece BPE tokenizer with Sarvam-style design facts (Indic-heavy corpus balance, vocab ~49k–68k, special tool/action tokens) rather than adapting Sarvam's restricted weights. This keeps the license clean and the embedding table compatible with SIA's from-scratch pretrain and future distilled students.

## Design facts to adopt

1. **Vocabulary size:** ~49k–68k is the practical range for Indic+English edge models. Sarvam-1 uses 64k active + 4k reserved = 68,096. SIA targets **49,152** as a minimum viable edge vocabulary (saves embedding-table RAM) with room to grow to 65k.
2. **Script coverage:** Devanagari (Hindi, Bhojpuri/Bihari languages), Latin (English, code), and common Indic punctuation/numerals. Future coverage can expand to Bengali, Tamil, Telugu, etc.
3. **Corpus balance:** equal-parts English and Indic tokens is the stated Sarvam-2B target. SIA adopts a **50/50 English–Indic text balance with an oversampling of device-action Hinglish** because the product surface is on-device actions.
4. **Fertility target:** < 2.5 tokens per Indic word on clean Indic text; < 2.0 on Indian-English/Hinglish.
5. **Special tokens:** reserve IDs for SIA tool-call grammar (`<|tool|>`, `<|action|>`, `<|point|>`, etc.) and action tags (`[POINT:x,y:label:screenN]`). This makes the action grammar single-token where useful.

## Scope note (TRD §8 Stage C)

This tokenizer is intended for:
- The **from-scratch pretrain path** (SIA base model trained with this vocabulary).
- **Distilled students** where the embedding table is rebuilt in Stage C.

It is **not** a drop-in replacement for the pretrained LFM2.5 GGUF weights, because changing the tokenizer under a trained embedding table breaks the embedding lookup. The integration path for LFM2.5-based tiers is vocabulary adaptation / embedding re-initialization at distillation time, not tokenizer swapping.
