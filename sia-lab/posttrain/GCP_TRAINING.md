# Training the SIA P1 Action Adapter on Google Cloud

This trains the **device-actions LoRA** — the adapter that teaches the base
model to emit SIA tool calls (`<|sia_tool|>set_alarm<|sia_call|>{...}<|sia_endcall|>`)
and on-screen actions. It is the one model whose training turns SIA's measured
tool-call accuracy from 0% into something usable.

- **Trainer:** [`train_gcp.py`](train_gcp.py) — plain `transformers` + `peft` + `trl` (no unsloth).
- **Data:** [`data/device_actions_train.json`](data/device_actions_train.json) (1000) + [`data/device_actions_val.json`](data/device_actions_val.json) (100).
- **Deps:** [`requirements-train.txt`](requirements-train.txt).
- **Target:** ≥95% exact structured-match on held-out tool calls.

The dataset is chat-format JSON: each record is `{"messages": [system, user, assistant]}`
where the assistant turn is the gold SIA tool/action string. Regenerate or
resize it any time with `python3 sia-lab/posttrain/generate_dataset.py --n 2000`.

---

## Option 0 — One command (laptop-side driver)

[`run_on_gcp.sh`](run_on_gcp.sh) does the whole thing from your laptop:
provisions an L4 VM in `asia-south1`, trains via a boot-time startup script,
waits for the adapter to land in the bucket, and deletes the VM afterward. It
only needs an authenticated `gcloud` CLI.

```bash
gcloud auth login
bash sia-lab/posttrain/run_on_gcp.sh
```

Overrides via env vars: `EPOCHS=5`, `KEEP_VM=1` (don't auto-delete),
`NO_WAIT=1` (launch and return), `BASE_MODEL=...`, `ZONE=...`. The project and
bucket are baked in (`sia-edge-prod` / `sia-artifacts`); override with
`PROJECT_ID=` / `BUCKET=` if they change.

The options below are the manual equivalents if you prefer to drive it yourself.

---

## Option A — Deep Learning VM with a GPU (simplest)

A single **L4** (`g2-standard-8`) or **T4** (`n1-standard-8` + 1×T4) trains the
1B base in a few minutes.

```bash
# 1. Create a GPU VM from a CUDA Deep Learning image (torch preinstalled).
gcloud compute instances create sia-p1-train \
  --zone=us-central1-a \
  --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1 \
  --maintenance-policy=TERMINATE \
  --image-family=common-cu121-debian-11 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB

# 2. SSH in.
gcloud compute ssh sia-p1-train --zone=us-central1-a

# --- on the VM ---
# 3. Get the code + data.
git clone https://github.com/skmandal3240/SIA.git && cd SIA

# 4. Install training deps (torch is already on the DLVM image).
pip install -r sia-lab/posttrain/requirements-train.txt

# 5. (Gated base models only) authenticate to Hugging Face.
huggingface-cli login   # needed for meta-llama/*; skip for unsloth/* mirrors

# 6. Validate data + config without spending GPU time.
python3 sia-lab/posttrain/train_gcp.py --dry-run

# 7. Train, evaluate, and push the adapter to a bucket.
python3 sia-lab/posttrain/train_gcp.py \
  --base meta-llama/Llama-3.2-1B-Instruct \
  --output-dir gs://YOUR_BUCKET/sia/device_actions_lora \
  --epochs 3 --merge
```

`--output-dir gs://…` stages the adapter locally, then `gsutil`-syncs it up.
Use a local path instead to keep it on the VM's disk.

**Remember to delete the VM when done** so it stops billing:

```bash
gcloud compute instances delete sia-p1-train --zone=us-central1-a
```

---

## Option B — Vertex AI custom training job (managed, no VM to babysit)

Run the same script as a managed job; Vertex provisions and tears down the GPU.

```bash
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=sia-p1-lora \
  --worker-pool-spec=machine-type=g2-standard-8,replica-count=1,accelerator-type=NVIDIA_L4,accelerator-count=1,container-image-uri=us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-gpu.2-3.py310:latest,local-package-path=.,script=sia-lab/posttrain/train_gcp.py \
  --args=--base=meta-llama/Llama-3.2-1B-Instruct,--output-dir=gs://YOUR_BUCKET/sia/device_actions_lora,--epochs=3,--merge
```

Set `HF_TOKEN` in the job environment (or pass a public `unsloth/*` base) so a
gated base model can be pulled. Point `--output-dir` at a bucket so the artifact
survives the job.

---

## Base-model note

`train_gcp.py` defaults to Llama's standard projection names
(`q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`). If you train a
non-Llama base (for example an LFM2 mirror whose modules are
`w1,w2,w3,in_proj,out_proj`), pass the matching set:

```bash
python3 sia-lab/posttrain/train_gcp.py --base <lfm2-repo> \
  --target-modules w1 w2 w3 q_proj k_proj v_proj out_proj in_proj
```

---

## After training — use the adapter

1. Download it: `gsutil -m cp -r gs://YOUR_BUCKET/sia/device_actions_lora sia-lab/posttrain/outputs/`.
2. Check `outputs/device_actions_lora/eval.json` for the held-out accuracy.
3. Merge + export to a servable GGUF for on-device inference:
   `python3 sia-lab/posttrain/merge_and_export.py` (needs `peft` installed).
4. If accuracy is below target, the recipe's next knobs are: more epochs,
   `--target-modules` set to r=32 via editing `LORA_R`, and a larger dataset
   (`generate_dataset.py --n 4000`).
