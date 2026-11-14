#!/usr/bin/env bash
# Score the four test splits with one forward pass per request.
#   bash scripts/evaluate.sh                       # the released checkpoint
#   bash scripts/evaluate.sh runs/medjev-4b/final  # a checkpoint you trained
set -euo pipefail
[ -f .env ] && set -a && . ./.env && set +a

CHECKPOINT=${1:-${CHECKPOINT:-$MEDJEV_MODELS/MedJev-4B}}
DATA_ROOT=${DATA_ROOT:-$MEDJEV_DATA}
OUTPUT_DIR=${OUTPUT_DIR:-$MEDJEV_WORK/eval/$(basename "$CHECKPOINT")}

python -m medjev.evaluate \
  --data-root "$DATA_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  ${DATASETS:+--datasets $DATASETS}
