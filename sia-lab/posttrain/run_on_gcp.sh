#!/usr/bin/env bash
#
# run_on_gcp.sh — laptop-side driver that trains the SIA P1 device-actions LoRA
# on Google Cloud end to end: it provisions an L4 GPU VM in asia-south1 (Mumbai),
# runs the training via a boot-time startup script, waits for the artifact to
# land in your bucket, and (by default) deletes the VM so it stops billing.
#
# Prereqs on your laptop: the Google Cloud CLI (`gcloud`) installed and
# authenticated (`gcloud auth login`). Nothing else — all heavy work runs on
# the VM.
#
# Usage:
#   bash sia-lab/posttrain/run_on_gcp.sh              # provision, train, wait, auto-delete
#   EPOCHS=5 bash sia-lab/posttrain/run_on_gcp.sh     # override epochs
#   KEEP_VM=1 bash sia-lab/posttrain/run_on_gcp.sh    # leave the VM running afterwards
#   NO_WAIT=1 bash sia-lab/posttrain/run_on_gcp.sh    # kick it off and return immediately
#
set -euo pipefail

# --------------------------------------------------------------------------- #
# Config — baked in for this project. Override any via environment variables.
# --------------------------------------------------------------------------- #
PROJECT_ID="${PROJECT_ID:-sia-edge-prod}"
BUCKET="${BUCKET:-sia-artifacts}"
REGION="${REGION:-asia-south1}"
ZONE="${ZONE:-asia-south1-a}"
VM="${VM:-sia-p1-train}"
# GPU=1 (default) uses an L4 GPU VM; GPU=0 uses a CPU-only VM, which needs NO
# GPU quota — the fallback when GPUS_ALL_REGIONS is 0 on a fresh project.
GPU="${GPU:-1}"
if [ "$GPU" = "0" ]; then
  MACHINE_TYPE="${MACHINE_TYPE:-e2-standard-8}"   # 8 vCPU / 32 GB, no accelerator
  ACCELERATOR=""
else
  MACHINE_TYPE="${MACHINE_TYPE:-g2-standard-8}"
  ACCELERATOR="${ACCELERATOR:-type=nvidia-l4,count=1}"
fi
BASE_MODEL="${BASE_MODEL:-unsloth/Llama-3.2-1B-Instruct}"
REPO="${REPO:-https://github.com/skmandal3240/SIA.git}"
EPOCHS="${EPOCHS:-3}"
OUTPUT_URI="gs://${BUCKET}/sia/device_actions_lora"

KEEP_VM="${KEEP_VM:-0}"      # 1 = do not auto-delete the VM after training
NO_WAIT="${NO_WAIT:-0}"      # 1 = return right after launching; do not poll
WAIT_TIMEOUT="${WAIT_TIMEOUT:-3600}"   # seconds to wait for the artifact
AUTO_DELETE=$([ "$KEEP_VM" = "1" ] && echo false || echo true)

# Boot image. Google retires Deep Learning VM image families periodically, so
# rather than hardcode one (which breaks when it's removed), resolve the newest
# available CUDA family at runtime. Override with IMAGE_FAMILY=... if you want a
# specific one.
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"
IMAGE_FAMILY="${IMAGE_FAMILY:-}"

log() { printf '\033[1;36m[run_on_gcp]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[run_on_gcp] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #
command -v gcloud >/dev/null 2>&1 || die "gcloud CLI not found. Install: https://cloud.google.com/sdk/docs/install"
gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | grep -q . \
  || die "No active gcloud account. Run: gcloud auth login"

log "project=$PROJECT_ID  zone=$ZONE  compute=$([ "$GPU" = "0" ] && echo "CPU ($MACHINE_TYPE)" || echo "GPU ($ACCELERATOR)")  base=$BASE_MODEL  epochs=$EPOCHS"
log "artifact will be written to: $OUTPUT_URI"
gcloud config set project "$PROJECT_ID" >/dev/null

# --------------------------------------------------------------------------- #
# Bucket
# --------------------------------------------------------------------------- #
if gcloud storage buckets describe "gs://$BUCKET" >/dev/null 2>&1; then
  log "bucket gs://$BUCKET exists"
else
  log "creating bucket gs://$BUCKET in $REGION"
  gcloud storage buckets create "gs://$BUCKET" --location="$REGION" --uniform-bucket-level-access
fi

# --------------------------------------------------------------------------- #
# Startup script that runs on the VM at boot (config values are expanded here;
# runtime bits like $(date) are escaped so they run on the VM, not the laptop).
# --------------------------------------------------------------------------- #
STARTUP_FILE="$(mktemp)"
trap 'rm -f "$STARTUP_FILE"' EXIT
cat > "$STARTUP_FILE" <<EOF
#!/bin/bash
set -xe
exec > /var/log/sia-train.log 2>&1
echo "[sia] startup begin \$(date)"

# Ensure the NVIDIA driver is present (Deep Learning images ship an installer).
if [ -x /opt/deeplearning/install-driver.sh ]; then /opt/deeplearning/install-driver.sh || true; fi
nvidia-smi || true

cd /root
rm -rf SIA
git clone ${REPO}
cd SIA

pip install -r sia-lab/posttrain/requirements-train.txt
python3 sia-lab/posttrain/train_gcp.py --dry-run

python3 sia-lab/posttrain/train_gcp.py \\
  --base ${BASE_MODEL} \\
  --train sia-lab/posttrain/data/device_actions_train.json \\
  --val   sia-lab/posttrain/data/device_actions_val.json \\
  --output-dir ${OUTPUT_URI} \\
  --epochs ${EPOCHS} --merge

echo "[sia] training complete \$(date)"
echo "done \$(date)" | gsutil cp - ${OUTPUT_URI}/_SUCCESS || true

if [ "${AUTO_DELETE}" = "true" ]; then
  echo "[sia] deleting self to stop billing"
  gcloud compute instances delete ${VM} --zone=${ZONE} --quiet || true
fi
EOF

# --------------------------------------------------------------------------- #
# Create the VM
# --------------------------------------------------------------------------- #
if gcloud compute instances describe "$VM" --zone="$ZONE" >/dev/null 2>&1; then
  die "VM '$VM' already exists in $ZONE. Delete it first: gcloud compute instances delete $VM --zone=$ZONE"
fi

# Resolve the newest available Deep Learning VM image family (these get
# retired/renamed, so we look one up instead of trusting a hardcoded name).
# GPU runs use a CUDA family (common-cu*); CPU runs use a CPU family (common-cpu*).
if [ -z "$IMAGE_FAMILY" ]; then
  if [ "$GPU" = "0" ]; then img_pat='^common-cpu'; else img_pat='^common-cu[0-9]+'; fi
  log "resolving latest Deep Learning VM image family (~$img_pat) in $IMAGE_PROJECT..."
  IMAGE_FAMILY=$(gcloud compute images list \
    --project="$IMAGE_PROJECT" \
    --filter="family~'${img_pat}'" \
    --format="value(family)" 2>/dev/null | sort -Vu | tail -n1)
fi
[ -n "$IMAGE_FAMILY" ] || die "could not resolve an image family in $IMAGE_PROJECT. List options with: gcloud compute images list --project $IMAGE_PROJECT --format=\"value(family)\" | sort -u"
log "boot image family: $IMAGE_FAMILY ($IMAGE_PROJECT)"

# Assemble create args; GPU-only flags are added conditionally so a CPU run
# needs no accelerator and no GPU quota.
create_args=(
  "$VM"
  --project="$PROJECT_ID"
  --zone="$ZONE"
  --machine-type="$MACHINE_TYPE"
  --image-family="$IMAGE_FAMILY"
  --image-project="$IMAGE_PROJECT"
  --boot-disk-size=100GB
  --scopes=https://www.googleapis.com/auth/cloud-platform
  --metadata-from-file=startup-script="$STARTUP_FILE"
)
if [ "$GPU" = "0" ]; then
  log "creating CPU VM '$VM' ($MACHINE_TYPE, no GPU quota needed) — training starts via the boot script"
else
  create_args+=(
    --accelerator="$ACCELERATOR"
    --maintenance-policy=TERMINATE
    --provisioning-model=STANDARD
    --metadata=install-nvidia-driver=True
  )
  log "creating L4 GPU VM '$VM' — training starts via the boot script"
fi
gcloud compute instances create "${create_args[@]}"

log "VM created. Training runs on boot; logs at /var/log/sia-train.log on the VM."
log "Tail live:  gcloud compute ssh $VM --zone=$ZONE --command 'sudo tail -f /var/log/sia-train.log'"

if [ "$NO_WAIT" = "1" ]; then
  log "NO_WAIT set — not polling. Artifact will appear at: $OUTPUT_URI"
  exit 0
fi

# --------------------------------------------------------------------------- #
# Wait for the artifact to land in the bucket
# --------------------------------------------------------------------------- #
log "waiting up to ${WAIT_TIMEOUT}s for training to finish (polling $OUTPUT_URI/_SUCCESS)"
elapsed=0
while ! gcloud storage ls "$OUTPUT_URI/_SUCCESS" >/dev/null 2>&1; do
  if [ "$elapsed" -ge "$WAIT_TIMEOUT" ]; then
    die "timed out after ${WAIT_TIMEOUT}s. Check the VM log or re-run with a larger WAIT_TIMEOUT."
  fi
  sleep 30
  elapsed=$((elapsed + 30))
  printf '.'
done
printf '\n'

log "training finished. Held-out accuracy:"
gcloud storage cat "$OUTPUT_URI/eval.json" || log "(eval.json not found; check the VM log)"
log "adapter artifacts: $OUTPUT_URI/"
if [ "$AUTO_DELETE" = "true" ]; then
  log "the VM deleted itself; nothing left billing except the bucket objects."
else
  log "KEEP_VM set — remember to delete it: gcloud compute instances delete $VM --zone=$ZONE"
fi
