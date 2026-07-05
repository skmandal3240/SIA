# Tokenizer build lesson

When training a SentencePiece tokenizer on a tiny corpus, a large target vocabulary causes a fatal "vocabulary size too high" error. Cap the vocabulary to what the corpus can support (unique tokens + specials) and enable `byte_fallback=True` so Indic characters not represented by BPE merges still encode/decode losslessly.
