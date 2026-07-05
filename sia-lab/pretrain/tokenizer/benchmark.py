"""Tokenizer fertility benchmark for SIA.

Compares the SIA tokenizer against the LFM2.5 tokenizer and one Western
baseline on Hindi, Bhojpuri/Bihari, Indian-English, and code samples.

If `transformers` is not installed, the HF baselines are mocked with a
clearly-labeled fallback tokenizer that splits on whitespace+punctuation.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Callable

from sia_tokenizer import SIA_TOKENIZER, load_or_train


def _words(text: str) -> list[str]:
    return [w for w in re.split(r"[\s\n\r\t]+", text.strip()) if w]


def _fertility(tok_fn: Callable[[str], int], text: str) -> float:
    ws = _words(text)
    if not ws:
        return 0.0
    return tok_fn(text) / len(ws)


# ---------------------------------------------------------------------------
# Benchmark samples
# ---------------------------------------------------------------------------
BENCH_SAMPLES = {
    "hindi": [
        "भारत एक विशाल देश है जिसमें कई भाषाएँ बोली जाती हैं।",
        "कर्नाटक की राजधानी बेंगलुरु है।",
        "मुझे सुबह छह बजे उठना है।",
        "कृपया मेरे लिए एक अलार्म सेट कर दीजिए।",
        "हिंदी भाषा में डिजिटल सहायक बहुत उपयोगी हो सकता है।",
    ],
    "bhojpuri_bihari": [
        "हमार बिहार बड़ा प्यारा जगह बा।",
        "का हाल बा?",
        "लइका लोग स्कूल जाइत बाड़न।",
        "हमके बाजार जाय के बा।",
        "चल घूमे चलल जाइब।",
    ],
    "indian_english": [
        "Kindly do the needful and revert back by EOD.",
        "My phone battery is about to die, please call later.",
        "Let's prepone the meeting to morning.",
        "Send me the OTP on WhatsApp.",
        "I will give a missed call once I reach.",
    ],
    "code": [
        "def set_alarm(hour: int, minute: int, label: str = '') -> bool:",
        "import torch\nmodel = torch.nn.Linear(128, 64)",
        "for item in items:\n    if item.active:\n        process(item)",
        "curl -X POST https://api.example.com/v1/call -H 'Content-Type: application/json'",
        "const navigate = (lat, lon) => maps.open({ latitude: lat, longitude: lon });",
    ],
}


def _make_hf_baseline(name: str, model_name: str) -> dict:
    """Try to load a HF tokenizer; return mock fallback if unavailable."""
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        def count_fn(text: str) -> int:
            return len(tok(text, add_special_tokens=False)["input_ids"])

        return {"name": name, "count": count_fn, "kind": "hf", "model": model_name}
    except Exception as exc:

        def mock_count(text: str) -> int:
            # ponytail: mock baseline clearly labeled; not a real tokenizer.
            pieces = re.split(r"([\s\n\r\t.,;:!?()\[\]{}'\"|\-=+/\\<>>&@#$%^*]+)", text)
            return len([p for p in pieces if p.strip()])

        return {
            "name": f"{name} (MOCK)",
            "count": mock_count,
            "kind": "mock",
            "model": model_name,
            "error": str(exc),
        }


def run_benchmark(sia_tokenizer_path: str | None = None) -> dict:
    """Run fertility benchmark and return results structure."""
    tok = load_or_train(
        work_dir=Path(__file__).parent,
        train=not bool(sia_tokenizer_path),
    )

    baselines = [
        _make_hf_baseline("LFM2.5", "LiquidAI/LFM2.5-1.2B-Instruct"),
        _make_hf_baseline("Western baseline (Llama)", "meta-llama/Llama-2-7b-hf"),
    ]

    rows = []
    for lang, samples in BENCH_SAMPLES.items():
        sia_scores = [_fertility(tok.count_tokens, s) for s in samples]
        row = {
            "language": lang,
            "samples": len(samples),
            "sia": round(statistics.mean(sia_scores), 2),
        }
        for base in baselines:
            base_scores = [_fertility(base["count"], s) for s in samples]
            row[base["name"]] = round(statistics.mean(base_scores), 2)
        rows.append(row)

    return {"rows": rows, "baselines": [b["name"] for b in baselines]}


def print_table(results: dict) -> None:
    rows = results["rows"]
    cols = ["language", "sia"] + [b for b in results["baselines"]]
    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    header = " | ".join(c.center(widths[c]) for c in cols)
    sep = "-" * len(header)
    print(header)
    print(sep)
    for r in rows:
        print(" | ".join(str(r.get(c, "")).center(widths[c]) for c in cols))


def main() -> None:
    results = run_benchmark()
    print("# SIA Tokenizer Fertility Benchmark")
    print("Fertility = average tokens per whitespace-delimited word (lower is better).\n")
    print_table(results)
    print("\n## Baseline metadata")
    for base in ["LFM2.5", "Western baseline (Llama)"]:
        print(f"- {base}: HuggingFace model if available, otherwise a clearly-labeled mock.")

    out_path = Path(__file__).parent / "benchmark_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults written to: {out_path}")


if __name__ == "__main__":
    main()
