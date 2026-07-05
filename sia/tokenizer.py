"""Character-level tokenizer."""


class CharTokenizer:
    # ponytail: char-level avoids SentencePiece/Byte-Pair dependencies.
    # Upgrade to BPE once data grows beyond ~1k unique tokens.
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.pad_id = 0

    def fit(self, texts):
        chars = sorted(set("".join(texts)))
        # reserve 0 for pad, 1 for unk
        self.char_to_idx = {c: i + 2 for i, c in enumerate(chars)}
        self.idx_to_char = {i + 2: c for i, c in enumerate(chars)}
        self.char_to_idx["<pad>"] = 0
        self.idx_to_char[0] = "<pad>"
        self.char_to_idx["<unk>"] = 1
        self.idx_to_char[1] = "<unk>"

    def encode(self, text):
        return [self.char_to_idx.get(c, 1) for c in text]

    def decode(self, ids):
        return "".join(self.idx_to_char.get(i, "") for i in ids)

    def vocab_size(self):
        return len(self.char_to_idx)


if __name__ == "__main__":
    tok = CharTokenizer()
    tok.fit(["hello world"])
    ids = tok.encode("hello world")
    assert tok.decode(ids) == "hello world"
    print("Tokenizer OK", tok.vocab_size())
