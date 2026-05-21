#!/usr/bin/env bash
# Print path to Stage 1 adapter: final_adapter, or latest checkpoint-step-*.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STAGE1_ROOT="${STAGE1_ROOT:-$ROOT/workspaces/stage1_reasoning}"

if [[ -d "$STAGE1_ROOT/final_adapter" ]]; then
  echo "$STAGE1_ROOT/final_adapter"
  exit 0
fi

latest=""
best_step=-1
for d in "$STAGE1_ROOT"/checkpoint-step-*; do
  [[ -d "$d" ]] || continue
  step="${d##*checkpoint-step-}"
  if [[ "$step" =~ ^[0-9]+$ ]] && (( step > best_step )); then
    best_step=$step
    latest="$d"
  fi
done

if [[ -n "$latest" ]]; then
  echo "Note: final_adapter missing; using checkpoint-step-${best_step}." >&2
  echo "$latest"
  exit 0
fi

echo "ERROR: No Stage 1 adapter under $STAGE1_ROOT." >&2
exit 1
