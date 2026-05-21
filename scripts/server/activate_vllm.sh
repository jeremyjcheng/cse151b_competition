#!/usr/bin/env bash
# Manual activation helper — run:  source scripts/server/activate_vllm.sh
# (Must use source, not bash, so conda stays active in your shell.)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Run with source, not bash:"
  echo "  source scripts/server/activate_vllm.sh"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"
echo "Ready. Verify with: bash scripts/server/test_installs.sh"
