#!/usr/bin/env bash
# Phase 0: eval ladder on stage2 holdout (base vs stage1 vs stage2).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

HOLDOUT="${HOLDOUT:-workspaces/stage2_adapt_v2/stage2_holdout.jsonl}"
STAGE1="${STAGE1:-workspaces/stage1_reasoning/checkpoint-step-100}"
STAGE2_ROOT="${STAGE2_ROOT:-workspaces/stage2_adapt_v2}"
GPU_ID="${GPU_ID:-0}"
OUT_DIR="${ROOT}/results/baseline_ladder"
mkdir -p "$OUT_DIR"

if [[ ! -f "$HOLDOUT" ]]; then
  echo "Missing holdout: $HOLDOUT"
  exit 1
fi

VLLM_ARGS=(--vllm-quantization none --vllm-load-format auto --gpu-id "$GPU_ID")

_run_eval() {
  local label="$1"
  local adapter="${2:-}"
  local report="$OUT_DIR/eval_${label}.json"
  echo ""
  echo "========== $label =========="
  local cmd=(
    "$PY" scripts/modular_pipeline/eval_runner.py
    --input "$HOLDOUT"
    --split-name "val"
    --eval-report "$report"
    "${VLLM_ARGS[@]}"
  )
  if [[ -n "$adapter" ]]; then
    cmd+=(--lora-adapter-path "$adapter")
  fi
  "${cmd[@]}"
}

_run_eval "base" ""
if [[ -d "$STAGE1" ]]; then
  _run_eval "stage1_step100" "$STAGE1"
else
  echo "Skip stage1: $STAGE1 not found"
fi

if [[ -f "$STAGE2_ROOT/best_adapter.txt" ]]; then
  BEST="$(tr -d '\n' < "$STAGE2_ROOT/best_adapter.txt")"
elif [[ -f "$STAGE2_ROOT/holdout_checkpoint_sweep.csv" ]]; then
  BEST_NAME="$("$PY" -c "
import csv
from pathlib import Path
rows = list(csv.DictReader(open('${STAGE2_ROOT}/holdout_checkpoint_sweep.csv')))
rows.sort(key=lambda r: float(r.get('validation_accuracy_pct') or 0), reverse=True)
print(rows[0]['checkpoint'] if rows else '')
")"
  BEST="${STAGE2_ROOT}/${BEST_NAME}"
else
  BEST="${STAGE2_ROOT}/final_adapter"
fi

if [[ -d "$BEST" ]]; then
  _run_eval "stage2_best" "$BEST"
else
  echo "Skip stage2_best: $BEST not found"
fi

echo ""
echo "Ladder reports written under: $OUT_DIR"
"$PY" - <<PY
import json
from pathlib import Path
out = Path("${OUT_DIR}")
for p in sorted(out.glob("eval_*.json")):
    data = json.loads(p.read_text())
    m = data.get("metrics") or {}
    acc = m.get("validation_accuracy_pct", "n/a")
    print(f"  {p.stem}: {acc}% overall")
PY
