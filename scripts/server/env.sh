#!/usr/bin/env bash
# Activate conda env "vllm" only — never uses project .venv.
# Usage: source scripts/server/env.sh
set -euo pipefail

if [[ -n "${_CSE151B_ENV_LOADED:-}" ]]; then
  return 0 2>/dev/null || true
fi

_SERVER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROOT="${ROOT:-$(cd "${_SERVER_DIR}/../.." && pwd)}"
cd "$ROOT"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-vllm}"

# Old .venv on PATH breaks conda — strip it before activating vllm.
if [[ -n "${VIRTUAL_ENV:-}" && "$VIRTUAL_ENV" == *".venv"* ]]; then
  echo "Clearing stale VIRTUAL_ENV (.venv): $VIRTUAL_ENV"
  deactivate 2>/dev/null || true
  unset VIRTUAL_ENV
fi

if [[ -d "${ROOT}/.venv" ]]; then
  echo "ERROR: ${ROOT}/.venv exists and can hijack your shell/IDE."
  echo "  Run: bash scripts/server/remove_venv.sh"
  echo "  Then: conda activate $CONDA_ENV_NAME"
  exit 1
fi

_activate_conda() {
  if [[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV_NAME" ]]; then
    return 0
  fi
  if ! command -v conda &>/dev/null; then
    return 1
  fi
  # shellcheck disable=SC1091
  eval "$(conda shell.bash hook 2>/dev/null)" || true
  conda activate "$CONDA_ENV_NAME"
  [[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV_NAME" ]]
}

if ! _activate_conda; then
  echo "ERROR: Could not activate conda env '$CONDA_ENV_NAME'."
  echo ""
  echo "  conda env list"
  echo "  conda activate $CONDA_ENV_NAME"
  echo "  # then re-run your command"
  exit 1
fi

export PY="$(command -v python)"

# Block .venv python even if it somehow appears on PATH.
case "$PY" in
  *"/.venv/"*)
    echo "ERROR: Python still points at .venv: $PY"
    echo "  Run: bash scripts/server/remove_venv.sh && conda activate $CONDA_ENV_NAME"
    exit 1
    ;;
esac

if [[ "$(basename "${CONDA_PREFIX:-}")" != "$CONDA_ENV_NAME" ]]; then
  echo "ERROR: CONDA_PREFIX is not $CONDA_ENV_NAME (got: ${CONDA_PREFIX:-unset})"
  exit 1
fi

export PYTHONUNBUFFERED=1
export _CSE151B_ENV_LOADED=1
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "Python:    $PY"
