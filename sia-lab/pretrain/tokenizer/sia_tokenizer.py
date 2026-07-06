"""SIA tokenizer module.

Implements the contract expected by sia-lab/pretrain/tokenizer/__init__.py:
- VocabSpec dataclass
- SIA_TOKENIZER class with encode/decode and special-token helpers
- load_or_train(train=True) to build a SentencePiece BPE tokenizer
- adapt_base_tokenizer(base_tokenizer_path, new_tokens) placeholder

This tokenizer is for the SIA from-scratch pretrain path and distilled students
(embedding table rebuilt). It is NOT a drop-in swap for the pretrained LFM2.5
GGUF weights.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import sentencepiece as spm

# Repo-relative anchors. This file lives at
# sia-lab/pretrain/tokenizer/sia_tokenizer.py, so parents[1] is sia-lab/pretrain.
_PRETRAIN_DIR = Path(__file__).resolve().parents[1]
_DEFAULT_CORPUS = _PRETRAIN_DIR / "corpus" / "sample_indic.jsonl"
# Training writes into a scratch build dir so the committed tokenizer artifacts
# in this source tree are never clobbered.
_DEFAULT_WORK_DIR = Path(tempfile.gettempdir()) / "sia-lab-build" / "tokenizer"


@dataclass
class VocabSpec:
    """Minimal vocab metadata used by the tokenizer contract."""

    vocab_size: int
    pad_id: int
    eos_id: int
    bos_id: int
    unk_id: int


# SIA special tokens. Keep the list small and stable; IDs are allocated by
# SentencePiece after the BPE vocabulary, before the normal control symbols.
SIA_SPECIAL_TOKENS: list[str] = [
    "<|sia|>",          # 0 / system marker
    "<|user|>",         # user turn marker
    "<|assistant|>",   # assistant turn marker
    "<|tool|>",         # tool-call start
    "<|action|>",       # generic action start
    "<|point|>",         # screen point action start
    "<|endofaction|>",  # action end
    "<|think|>",         # deep-reasoning start
    "<|endthink|>",     # deep-reasoning end
]

# Indic-ish script detection regex (Devanagari + common Indic punctuation).
INDIC_RE = re.compile(r"[\u0900-\u097F]")


def _is_indic(text: str) -> bool:
    return bool(INDIC_RE.search(text))


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


class SIA_TOKENIZER:
    """Wrapper around a SentencePiece model with SIA special tokens."""

    def __init__(self, model_path: str | Path, spec: VocabSpec | None = None):
        self.model_path = Path(model_path)
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(str(self.model_path))
        self.spec = spec or self._infer_spec()
        self.special = {tok: self.sp.piece_to_id(tok) for tok in SIA_SPECIAL_TOKENS}

    def _infer_spec(self) -> VocabSpec:
        return VocabSpec(
            vocab_size=self.sp.vocab_size(),
            pad_id=self.sp.pad_id(),
            eos_id=self.sp.eos_id(),
            bos_id=self.sp.bos_id(),
            unk_id=self.sp.unk_id(),
        )

    # --- contract helpers -------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return self.spec.vocab_size

    @property
    def pad_id(self) -> int:
        return self.spec.pad_id

    @property
    def eos_id(self) -> int:
        return self.spec.eos_id

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Encode text to token IDs."""
        ids = self.sp.EncodeAsIds(_normalize_whitespace(text))
        if add_bos and self.spec.bos_id >= 0:
            ids = [self.spec.bos_id] + ids
        if add_eos and self.spec.eos_id >= 0:
            ids = ids + [self.spec.eos_id]
        return ids

    def decode(self, ids: Sequence[int], skip_special: bool = False) -> str:
        """Decode token IDs back to text."""
        return self.sp.DecodeIds(list(ids))

    def encode_batch(self, texts: Iterable[str]) -> list[list[int]]:
        return [self.encode(t) for t in texts]

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))

    def tokenize(self, text: str) -> list[str]:
        return self.sp.EncodeAsPieces(_normalize_whitespace(text))

    def get_vocab(self) -> dict[str, int]:
        return {self.sp.IdToPiece(i): i for i in range(self.sp.vocab_size())}

    def special_token_id(self, token: str) -> int:
        return self.special[token]

    # --- metadata -----------------------------------------------------------
    def fertility(self, text: str) -> float:
        """Tokens per whitespace-delimited word."""
        words = _normalize_whitespace(text).split()
        if not words:
            return 0.0
        return len(self.encode(text)) / len(words)

    def compression_ratio(self, text: str) -> float:
        """Characters per token."""
        ids = self.encode(text)
        if not ids:
            return 0.0
        return len(text) / len(ids)


def build_training_texts(jsonl_path: str | Path, out_txt_path: str | Path) -> None:
    """Convert a JSONL corpus to a flat text file for SentencePiece training."""
    out_txt_path = Path(out_txt_path)
    out_txt_path.parent.mkdir(parents=True, exist_ok=True)
    with Path(jsonl_path).open("r", encoding="utf-8") as fin, out_txt_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            obj = json.loads(line)
            # Accept common keys; fall back to string value.
            text = obj.get("text") or obj.get("content") or obj.get("message")
            if isinstance(text, dict):
                text = text.get("content") or text.get("text")
            if text:
                fout.write(_normalize_whitespace(str(text)) + "\n")


def _write_dummy_vocab(model_prefix: str, vocab_size: int) -> None:
    """Create a minimal vocab file so SentencePiece has a deterministic seed."""
    vocab_path = Path(f"{model_prefix}.vocab")
    with vocab_path.open("w", encoding="utf-8") as f:
        f.write("\u003cunk\u003e\t0\n")
        f.write("\u003cs\u003e\t0\n")
        f.write("\u003c/e\u003e\t0\n")
        f.write("\u003cpad\u003e\t0\n")
        for i in range(4, vocab_size):
            f.write(f"\u003cunused{i}\u003e\t0\n")


def train_sentencepiece(
    input_txt: str | Path,
    model_prefix: str | Path,
    vocab_size: int | None = None,
    character_coverage: float = 0.9999,
    num_threads: int = 8,
) -> Path:
    """Train a SentencePiece BPE model and add SIA special tokens."""
    model_prefix = Path(model_prefix)
    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    user_symbols = ",".join(SIA_SPECIAL_TOKENS)

    # ponytail: tiny corpus cannot support 49k vocab; cap to min(8192, unique tokens).
    with Path(input_txt).open("r", encoding="utf-8") as f:
        corpus = f.read()
    unique_tokens = len(set(corpus.split()))
    target = min(vocab_size or 8192, max(512, unique_tokens + len(SIA_SPECIAL_TOKENS) + 64))

    spm.SentencePieceTrainer.train(
        input=str(input_txt),
        model_prefix=str(model_prefix),
        vocab_size=target,
        model_type="bpe",
        character_coverage=character_coverage,
        num_threads=num_threads,
        split_by_whitespace=True,
        split_by_unicode_script=True,
        split_by_number=True,
        max_sentencepiece_length=16,
        shuffle_input_sentence=True,
        seed_sentencepiece_size=100000,
        num_sub_iterations=2,
        max_sentence_length=2048,
        pad_id=3,
        eos_id=2,
        bos_id=1,
        unk_id=0,
        user_defined_symbols=user_symbols,
        # ponytail: byte_fallback lets unseen Indic chars roundtrip as bytes.
        byte_fallback=True,
        # ponytail: disable NFKC normalization so Indic matras stay intact.
        normalization_rule_name="identity",
    )
    return Path(f"{model_prefix}.model")


def load_or_train(
    corpus_jsonl: str | Path | None = None,
    work_dir: str | Path = _DEFAULT_WORK_DIR,
    vocab_size: int = 49152,
    train: bool = True,
) -> SIA_TOKENIZER:
    """Load an existing SIA tokenizer or train one from a JSONL corpus."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    model_file = work_dir / "sia_tokenizer.model"
    spec_file = work_dir / "sia_tokenizer_spec.json"

    if model_file.exists() and not train:
        spec = VocabSpec(**json.loads(spec_file.read_text())) if spec_file.exists() else None
        return SIA_TOKENIZER(model_file, spec)

    if corpus_jsonl is None:
        # ponytail: default to shipped sample corpus when no path given.
        corpus_jsonl = _DEFAULT_CORPUS

    txt_file = work_dir / "train.txt"
    build_training_texts(corpus_jsonl, txt_file)

    model_path = train_sentencepiece(
        txt_file, work_dir / "sia_tokenizer", vocab_size=vocab_size
    )

    tok = SIA_TOKENIZER(model_path)
    spec_file.write_text(json.dumps(tok.spec.__dict__, indent=2), encoding="utf-8")
    return tok


def adapt_base_tokenizer(base_tokenizer_path: str, new_tokens: list[str]) -> dict:
    """Placeholder: describe how to adapt a base HF tokenizer to SIA tokens.

    We do not vendor or modify restricted Sarvam weights here. For a permissive
    base tokenizer, the real implementation would:
      1. Load the base tokenizer.
      2. Add new_tokens (SIA special + any missing Indic pieces).
      3. Resize model embeddings.
      4. Return the new tokenizer + embedding resize metadata.
    This stub returns a manifest so callers know the integration path.
    """
    return {
        "base_tokenizer_path": base_tokenizer_path,
        "new_tokens": new_tokens,
        "status": "stub",
        "note": "Embedding table must be resized before use; see SARVAM_RESEARCH.md license note.",
    }


def demo() -> None:
    """Self-check that the tokenizer can encode/decode Hindi and special tokens."""
    tok = load_or_train(train=True)
    hindi = "कर्नाटक की राजधानी क्या है?"
    ids = tok.encode(hindi)
    back = tok.decode(ids)
    assert back.strip() == hindi.strip(), f"roundtrip failed: {back!r} != {hindi!r}"

    for special in SIA_SPECIAL_TOKENS:
        sid = tok.special_token_id(special)
        assert sid >= 0, f"missing special token {special}"

    hinglish = "bhai alarm 6 baje ke liye set kar do"
    print("Hindi roundtrip OK:", back)
    print("Hinglish tokens:", tok.tokenize(hinglish))
    print("Fertility Hindi:", round(tok.fertility(hindi), 2))
    print("Vocab size:", tok.vocab_size)


if __name__ == "__main__":
    demo()
