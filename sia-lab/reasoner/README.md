# SIA Deep SIR Reasoner

Ponytail gate: a from-scratch Recurrent-Depth Transformer that proves the
mechanisms work on CPU before scaling up.

## Files

- `reasoner.py` — core model (MLA cache, MoE routing, ACT halting, spectral-radius tracking).
- `tiny_overfit.py` — CPU demo that overfits a tiny corpus.
- `router.py` — two-speed router: fast vs deep path.
- `deep_path.py` — deep-path wrapper around `SIRReasoner`; accepts governor budget.
- `governor_client.py` — device-state budget client (normal / hot / cool modes).
- `p3_eval.py` — P3 exit eval: deep path beats fast path; governor throttles under heat.

## Run

```bash
python3 sia-lab/reasoner/reasoner.py          # self-check
python3 sia-lab/reasoner/tiny_overfit.py      # tiny overfit gate
python3 sia-lab/reasoner/router.py            # routing examples
python3 sia-lab/reasoner/governor_client.py  # budget examples
python3 sia-lab/reasoner/p3_eval.py           # P3 eval: deep beats fast
make reasoner-p3                              # all of the above
```

## What it proves

1. Forward and backward pass run end-to-end.
2. Recurrent-depth block loops over shared weights.
3. MoE router dispatches tokens to top-1 experts.
4. MLA compresses key/value into a single cache vector.
5. ACT halts early when the per-position probability crosses threshold.
6. Spectral radius of recurrent weights is forced below 1.
7. Model overfits a tiny Indic-English corpus on CPU.
8. Router escalates multi-hop/planning queries to deep path.
9. Governor forces fast path and reduces loop depth under thermal stress.
10. Deep path beats fast path on the `multi_hop` benchmark.

## Scaling notes

- This is intentionally gate-sized. Real SIR would up-cycle LFM2.5 weights
  into an RDT-MoE core using DeepSpeed-MoE + ZeRO++ — that requires GPU
  training and is the next funding/compute milestone.
- Generation here is auto-regressive with simple multinomial sampling; a
  production reasoner needs flash-attention kernels and a servable runtime.
