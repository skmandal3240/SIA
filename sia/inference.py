"""Inference/generation for the trained SIA model."""
import torch

from sia.actions import parse_action
from sia.model import SIAModel
from sia.tokenizer import CharTokenizer


class SIA:
    def __init__(self, checkpoint_path, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        ckpt = torch.load(checkpoint_path, map_location=device)
        self.tokenizer = CharTokenizer()
        self.tokenizer.char_to_idx = ckpt["vocab"]
        self.tokenizer.idx_to_char = {v: k for k, v in ckpt["vocab"].items()}
        cfg = ckpt["config"]
        self.model = SIAModel(self.tokenizer.vocab_size(), **cfg).to(device)
        state = {k: v for k, v in ckpt["model_state"].items() if not k.endswith(".mask")}
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        self.device = device

    def respond(self, prompt, max_new_tokens=80):
        text = f"### Instruction:\n{prompt}\n\n### Response:\n"
        ids = [self.tokenizer.encode(text)]
        x = torch.tensor(ids, device=self.device)
        out = self.model.generate(x, max_new_tokens=max_new_tokens, temperature=0.8, top_k=20)
        return self.tokenizer.decode(out[0].tolist())


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/sia_model.pt")
    parser.add_argument("--prompt", default="Set an alarm for 6 AM")
    args = parser.parse_args()

    sia = SIA(args.checkpoint)
    response = sia.respond(args.prompt)
    print("=" * 40)
    print(response)
    print("=" * 40)
    action = parse_action(response) or parse_action(args.prompt)
    if action:
        print("\nParsed action:", action)
        print("(mock execution only)")


if __name__ == "__main__":
    main()
