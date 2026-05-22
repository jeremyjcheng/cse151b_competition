#!/usr/bin/env bash
# Activate conda vllm in your current shell (bash or zsh).
#
#   source scripts/server/activate_vllm.sh
#
# Do NOT run:  bash scripts/server/activate_vllm.sh

# zsh: $0 is the shell name when sourced
if [[ -n "${ZSH_VERSION:-}" ]]; then
  if [[ "${ZSH_EVAL_CONTEXT:-}" != *:file:* ]]; then
    echo "Run with:  source scripts/server/activate_vllm.sh" >&2
    return 1
  fi
elif [[ -n "${BASH_VERSION:-}" ]]; then
  if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Run with:  source scripts/server/activate_vllm.sh" >&2
    exit 1
  fi
fi

if [[ -n "${ZSH_VERSION:-}" ]]; then
  _ACTIVATE_SCRIPT="${(%):-%x}"
else
  _ACTIVATE_SCRIPT="${BASH_SOURCE[0]}"
fi
SCRIPT_DIR="$(cd "$(dirname "${_ACTIVATE_SCRIPT}")" && pwd)"

# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh" || return 1

echo "Ready. Verify with: bash scripts/server/diagnose_vllm.sh"
