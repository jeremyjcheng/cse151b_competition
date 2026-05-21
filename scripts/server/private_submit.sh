#!/usr/bin/env bash
# Phase 5: verify LoRA then private inference
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source .venv/bin/activate

GPU_ID="${GPU_ID:-0}"
STAGE2_ROOT="${STAGE2_ROOT:-workspaces/stage2_adapt}"

if [[ -f "$STAGE2_ROOT/best_adapter.txt" ]]; then
  ADAPTER="$(tr -d '\n' < "$STAGE2_ROOT/best_adapter.txt")"
  echo "Using best checkpoint from sweep: $ADAPTER"
else
  ADAPTER="$STAGE2_ROOT/final_adapter"
  echo "Using final_adapter: $ADAPTER"
fi

if [[ ! -d "$ADAPTER" ]]; then
  echo "Adapter not found: $ADAPTER"
  exit 1
fi

if [[ ! -f data/private.jsonl ]]; then
  echo "WARNING: data/private.jsonl missing; inference may fail."
fi

python scripts/modular_pipeline/verify_lora_vllm.py \
  --lora-adapter-path "$ADAPTER" \
  --gpu-id "$GPU_ID" \
  --vllm-quantization none \
  --vllm-load-format auto

python scripts/modular_pipeline/modular_pipeline.py \
  --input private \
  --lora-adapter-path "$ADAPTER" \
  --gpu-id "$GPU_ID" \
  --vllm-quantization none \
  --vllm-load-format auto

echo "Private outputs should be under results/ (see modular_pipeline logs)."
