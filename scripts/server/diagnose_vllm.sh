#!/usr/bin/env bash
# vLLM + CUDA check. Run in a subshell (safe — does not close your terminal):
#   bash scripts/server/diagnose_vllm.sh
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run env setup in bash subshell; never source env.sh from your interactive zsh.
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh" || exit 1

echo "=== Python / conda ==="
echo "PY=$PY"
echo "CONDA_PREFIX=${CONDA_PREFIX:-unset}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-unset}"

echo ""
echo "=== libcudart ==="
found=0
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  for d in ${LD_LIBRARY_PATH//:/ }; do
    for lib in libcudart.so.13 libcudart.so.12 libcudart.so; do
      if [[ -f "${d}/${lib}" ]]; then
        echo "  OK ${d}/${lib}"
        found=1
      fi
    done
  done
fi
if [[ "$found" -eq 0 ]]; then
  echo "  No libcudart on LD_LIBRARY_PATH — try: module load cuda/13.0"
fi

echo ""
echo "=== vLLM import ==="
if ! "$PY" - <<'PY'
import sys
print("executable:", sys.executable)
for p in sys.path[:8]:
    if ".local" in p:
        print("WARNING: user-site on path:", p)
from vllm import LLM
from vllm.lora.request import LoRARequest
import vllm
print("vllm:", vllm.__file__)
print("vllm version:", getattr(vllm, "__version__", "?"))
print("OK")
PY
then
  echo "FAILED vLLM import. Fix conda env vllm + CUDA, then retry." >&2
  exit 1
fi

echo ""
echo "PASSED diagnose_vllm"
