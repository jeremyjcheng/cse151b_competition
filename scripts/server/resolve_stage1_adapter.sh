#!/usr/bin/env bash
# Print path to Stage 1 adapter (prefers stage1_reasoning_v2, then stage1_reasoning).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

_resolve_latest_checkpoint() {
  local stage_root="$1"
  local latest="" best_step=-1
  if [[ -d "$stage_root/final_adapter" ]]; then
    echo "$stage_root/final_adapter"
    return 0
  fi
  for d in "$stage_root"/checkpoint-step-*; do
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
    return 0
  fi
  return 1
}

for root in \
  "${STAGE1_ROOT:-$ROOT/workspaces/stage1_reasoning_v2}" \
  "$ROOT/workspaces/stage1_reasoning_v2" \
  "$ROOT/workspaces/stage1_reasoning"; do
  if _resolve_latest_checkpoint "$root"; then
    exit 0
  fi
done

echo "ERROR: No Stage 1 adapter under workspaces/stage1_reasoning*." >&2
exit 1
