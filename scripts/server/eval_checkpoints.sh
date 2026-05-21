#!/usr/bin/env bash
# Phase 4: holdout eval + checkpoint sweep; pick best adapter
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source .venv/bin/activate

STAGE2_ROOT="${STAGE2_ROOT:-workspaces/stage2_adapt}"
HOLDOUT="$STAGE2_ROOT/stage2_holdout.jsonl"
GPU_ID="${GPU_ID:-0}"

if [[ ! -f "$HOLDOUT" ]]; then
  echo "Missing holdout: $HOLDOUT"
  echo "Run the full pipeline first (scripts/server/run_full_pipeline.sh)."
  exit 1
fi

python scripts/modular_pipeline/eval_runner.py \
  --input "$HOLDOUT" \
  --lora-adapter-path "$STAGE2_ROOT/final_adapter" \
  --split-name val \
  --gpu-id "$GPU_ID" \
  --eval-report "$STAGE2_ROOT/holdout_eval_final.json" \
  --vllm-quantization none \
  --vllm-load-format auto

python scripts/modular_pipeline/eval_runner.py \
  --input "$HOLDOUT" \
  --checkpoint-dir "$STAGE2_ROOT" \
  --split-name val \
  --gpu-id "$GPU_ID" \
  --eval-report "$STAGE2_ROOT/holdout_checkpoint_sweep.json" \
  --vllm-quantization none \
  --vllm-load-format auto

python scripts/server/pick_best_checkpoint.py --stage2-root "$STAGE2_ROOT"
