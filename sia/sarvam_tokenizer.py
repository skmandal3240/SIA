"""Sarvam-style SentencePiece tokenizer wrapper.

Sarvam AI extended the Llama2 tokenizer with ~48k extra tokens to cover Indic
(Hindi/Hinglish) text. For SIA we follow the same idea: start from an existing
byte-pair tokenizer (we build our own SentencePiece model) and train it on a
combined English + Indic corpus so one sub-word covers common Hindi morphemes.

This file is a drop-in replacement for sia.tokenizer.CharTokenizer:

    from sia.sarvam_tokenizer import SarvamTokenizer
    tok = SarvamTokenizer()
    tok.train(["hindi english mixed text here"], "sia_tokenizer.model")
    ids = tok.encode("Set alarm for 6 AM")
    text = tok.decode(ids)
"""
import tempfile
from pathlib import Path

import sentencepiece as spm


class SarvamTokenizer:
    # ponytail: wraps SentencePiece. Train once, then ship sia_tokenizer.model.
    def __init__(self, model_path: str | None = None):
        self.sp = None
        if model_path and Path(model_path).exists():
            self.load(model_path)

    def train(self, texts: list[str], vocab_size: int = 8000, model_prefix: str = "sia_tokenizer"):
        """Train a SentencePiece model on the given texts."""
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as f:
            for line in texts:
                f.write(line.strip() + "\n")
            corpus_path = f.name

        spm.SentencePieceTrainer.train(
            input=corpus_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=0.9995,
            num_threads=4,
            max_sentence_length=2048,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
        )
        self.load(f"{model_prefix}.model")

    def load(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

    def encode(self, text: str) -> list[int]:
        if self.sp is None:
            raise RuntimeError("Tokenizer not loaded. Train or load a model first.")
        return self.sp.encode(text, out_type=int)

    def decode(self, ids: list[int]) -> str:
        if self.sp is None:
            raise RuntimeError("Tokenizer not loaded. Train or load a model first.")
        return self.sp.decode(ids)

    def vocab_size(self) -> int:
        return self.sp.vocab_size()


if __name__ == "__main__":
    sample = [
        "Set an alarm for 6 AM",
        "Wake me up at 7:30 AM tomorrow",
        "Text Ravi I am running late",
        "Turn off bluetooth",
        "Directions to Patna airport",
        "Schedule a team call tomorrow",
        "6 बजे अलार्म लगाओ",
        "रवि को मैसेज करो कि मैं लेट हूँ",
        "ब्लूटूथ बंद करो",
        "पटना एयरपोर्ट का रास्ता दिखाओ",
        "कल मीटिंग शेड्यूल करो",
    ]
    tok = SarvamTokenizer()
    tok.train(sample, vocab_size=256, model_prefix="sia_tokenizer")
    ids = tok.encode("6 बजे अलार्म लगाओ")
    print("ids:", ids)
    print("decoded:", tok.decode(ids))
    print("vocab_size:", tok.vocab_size())
