#!/usr/bin/env bash
# Phase 4: enable MCQ self-consistency (3 samples) and re-eval holdout.
# Temporarily patches setting via env — set MCQ_SELF_CONSISTENCY_SAMPLES=3 in setting.py for permanent use.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

STAGE2_ROOT="${STAGE2_ROOT:-workspaces/stage2_adapt_v3}"
HOLDOUT="${HOLDOUT:-$STAGE2_ROOT/stage2_holdout.jsonl}"

if [[ -f "$STAGE2_ROOT/best_adapter.txt" ]]; then
  ADAPTER="$(tr -d '\n' < "$STAGE2_ROOT/best_adapter.txt")"
else
  ADAPTER="$STAGE2_ROOT/final_adapter"
fi

echo "Enable MCQ_SELF_CONSISTENCY_SAMPLES=3 in scripts/modular_pipeline/setting.py, then run:"
echo ""
echo "  $PY scripts/modular_pipeline/eval_runner.py \\"
echo "    --input $HOLDOUT \\"
echo "    --lora-adapter-path $ADAPTER \\"
echo "    --split-name val_sc \\"
echo "    --vllm-quantization none --vllm-load-format auto \\"
echo "    --eval-report $STAGE2_ROOT/holdout_eval_self_consistency.json"
