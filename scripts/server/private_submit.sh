#!/usr/bin/env bash
# Phase 5: verify LoRA then private inference
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

GPU_ID="${GPU_ID:-0}"
STAGE2_ROOT="${STAGE2_ROOT:-workspaces/stage2_adapt}"

if [[ -f "$STAGE2_ROOT/best_adapter.txt" ]]; then
  ADAPTER="$(tr -d '\n' < "$STAGE2_ROOT/best_adapter.txt")"
else
  ADAPTER="$STAGE2_ROOT/final_adapter"
fi

if [[ ! -d "$ADAPTER" ]]; then
  echo "Adapter not found: $ADAPTER"
  exit 1
fi

"$PY" scripts/modular_pipeline/verify_lora_vllm.py \
  --lora-adapter-path "$ADAPTER" \
  --gpu-id "$GPU_ID" \
  --vllm-quantization none \
  --vllm-load-format auto

"$PY" scripts/modular_pipeline/modular_pipeline.py \
  --input private \
  --lora-adapter-path "$ADAPTER" \
  --gpu-id "$GPU_ID" \
  --vllm-quantization none \
  --vllm-load-format auto
