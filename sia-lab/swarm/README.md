# SIA Swarm + Distillation (P5)

P5 adds a lightweight Mixture-of-Students (MoS) distillation cycle on top of
the operational N=2 swarm loop.

## Files

- `swarm.py` — operational swarm: N nodes solve a task and reducer returns consensus.
- `distill.py` — MoS scaffold: teacher labels, students train on splits, swarm votes.
- `p5_eval.py` — end-to-end P5 smoke test.

## Run

```bash
make swarm-p5
```

## Production gap

The stubs use deterministic heuristics. The real pipeline will use:
- `ollama run sia-p0` as teacher for pseudo-labels.
- LoRA students fine-tuned on Colab T4 from `sia-lab/posttrain/sft.py`.
- Majority-vote consensus exported to the OTA manifest.
