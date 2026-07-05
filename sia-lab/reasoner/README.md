# SIA Deep SIR Reasoner

Ponytail gate: a from-scratch Recurrent-Depth Transformer that proves the
mechanisms work on CPU before scaling up.

## Files

- `reasoner.py` — core model (MLA cache, MoE routing, ACT halting, spectral-radius tracking).
- `tiny_overfit.py` — CPU demo that overfits a tiny corpus.

## Run

```bash
python3 sia-lab/reasoner/reasoner.py          # self-check
python3 sia-lab/reasoner/tiny_overfit.py      # tiny overfit gate
```

## What it proves

1. Forward and backward pass run end-to-end.
2. Recurrent-depth block loops over shared weights.
3. MoE router dispatches tokens to top-1 experts.
4. MLA compresses key/value into a single cache vector.
5. ACT halts early when the per-position probability crosses threshold.
6. Spectral radius of recurrent weights is forced below 1.
7. Model overfits a tiny Indic-English corpus on CPU.

## Scaling notes

- This is intentionally gate-sized. Real SIR would use multi-query KV down-projection,
  grouped-query attention, and flash-attention kernels.
- Generation here is auto-regressive with simple multinomial sampling; a production
  reasoner needs a two-speed router (fast path vs deep core).
