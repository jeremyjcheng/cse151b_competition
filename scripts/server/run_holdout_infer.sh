#!/usr/bin/env bash
# Save holdout model outputs for error analysis / curation (Phase 3).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

STAGE2_ROOT="${STAGE2_ROOT:-workspaces/stage2_adapt_v3}"
GPU_ID="${GPU_ID:-0}"

if [[ -f "$STAGE2_ROOT/best_adapter.txt" ]]; then
  ADAPTER="$(tr -d '\n' < "$STAGE2_ROOT/best_adapter.txt")"
else
  ADAPTER="$STAGE2_ROOT/final_adapter"
fi

HOLDOUT="$STAGE2_ROOT/stage2_holdout.jsonl"
if [[ ! -f "$HOLDOUT" ]]; then
  echo "Missing $HOLDOUT"
  exit 1
fi

"$PY" scripts/modular_pipeline/modular_pipeline.py \
  --input "$HOLDOUT" \
  --lora-adapter-path "$ADAPTER" \
  --output-dir results \
  --no-eval \
  --gpu-id "$GPU_ID" \
  --vllm-quantization none \
  --vllm-load-format auto

# Rename to stable path for curate_data
STEM="$(basename "$HOLDOUT" .jsonl)"
if [[ -f "results/${STEM}_outputs.jsonl" ]]; then
  cp "results/${STEM}_outputs.jsonl" results/holdout_outputs.jsonl
  echo "Wrote results/holdout_outputs.jsonl"
fi
