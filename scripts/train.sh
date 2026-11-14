#!/usr/bin/env bash
# Train MedJev-4B on the pooled training split of the four benchmarks.
#   bash scripts/train.sh
# Override any default from the environment, for example OUTPUT_DIR=... bash scripts/train.sh
set -euo pipefail
[ -f .env ] && set -a && . ./.env && set +a

DATA_ROOT=${DATA_ROOT:-$MEDJEV_DATA}
OUTPUT_DIR=${OUTPUT_DIR:-$MEDJEV_WORK/runs/medjev-4b}
SEED=${SEED:-42}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
LR=${LR:-1e-4}

python -m medjev.train \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --seed "$SEED" \
  --global-batch-size "$GLOBAL_BATCH_SIZE" \
  --lr "$LR" \
  --weight-decay "${WEIGHT_DECAY:-0.01}" \
  --clip-grad-norm "${CLIP_GRAD_NORM:-1.0}" \
  --checkpoint-every "${CHECKPOINT_EVERY:-250}"
