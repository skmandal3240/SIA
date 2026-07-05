"""Training loop for the SIA model."""
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sia.model import SIAModel
from sia.tokenizer import CharTokenizer


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def format_example(ex):
    return f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['output']}"


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=128):
        self.tokens = []
        for text in texts:
            ids = tokenizer.encode(text)
            for i in range(0, len(ids) - 1, max_len):
                chunk = ids[i : i + max_len + 1]
                if len(chunk) < 2:
                    continue
                self.tokens.append(chunk)
        self.max_len = max_len

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        ids = self.tokens[idx]
        x = ids[:-1] + [0] * (self.max_len - len(ids) + 1)
        y = ids[1:] + [0] * (self.max_len - len(ids) + 1)
        return torch.tensor(x[: self.max_len]), torch.tensor(y[: self.max_len])


def train(
    data_path,
    vocab_size=None,
    dim=256,
    n_layers=4,
    n_heads=4,
    max_len=128,
    epochs=50,
    batch_size=4,
    lr=3e-4,
    device=None,
    checkpoint_dir="checkpoints",
):
    device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    examples = load_jsonl(data_path)
    texts = [format_example(ex) for ex in examples]

    tokenizer = CharTokenizer()
    tokenizer.fit(texts)

    dataset = TextDataset(texts, tokenizer, max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if vocab_size is None:
        vocab_size = tokenizer.vocab_size()

    model = SIAModel(vocab_size, dim=dim, n_layers=n_layers, n_heads=n_heads, max_len=max_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(loader))

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for x, y in tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=0)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        print(f"epoch {epoch + 1}: loss={avg:.4f}")

    save_path = checkpoint_dir / "sia_model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab": tokenizer.char_to_idx,
            "config": {"dim": dim, "n_layers": n_layers, "n_heads": n_heads, "max_len": max_len},
        },
        save_path,
    )
    print(f"saved {save_path}")


if __name__ == "__main__":
    train("data/device_actions.jsonl", epochs=5, batch_size=2)
