"""Stub __init__.py so the tokenizer directory is importable as a package.

The goal prompt references an existing __init__.py contract with
SIA_TOKENIZER, VocabSpec, load_or_train, and adapt_base_tokenizer. We keep
those exports minimal and point to the implementation in sia_tokenizer.py.
"""

from .sia_tokenizer import (
    SIA_TOKENIZER,
    VocabSpec,
    adapt_base_tokenizer,
    load_or_train,
)

__all__ = ["SIA_TOKENIZER", "VocabSpec", "load_or_train", "adapt_base_tokenizer"]
