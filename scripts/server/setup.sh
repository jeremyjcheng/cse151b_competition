#!/usr/bin/env bash
# Phase 0: one-time server setup (from Stage 1-2 accuracy plan)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

echo "Project root: $ROOT"

mkdir -p workspaces logs results data

if [[ ! -f data/public.jsonl ]]; then
  echo "WARNING: data/public.jsonl is missing."
  echo "  Place competition public.jsonl at: $ROOT/data/public.jsonl"
  echo "  Private set (for final infer): $ROOT/data/private.jsonl"
fi

if [[ ! -d .venv ]]; then
  echo "Creating virtualenv..."
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip

UNAME="$(uname -s)"
if [[ "$UNAME" == "Linux" ]] && command -v nvidia-smi &>/dev/null; then
  echo "Installing full CUDA stack from requirements.txt ..."
  pip install -r requirements.txt
  pip install 'vllm>=0.10.0'
else
  echo "Non-CUDA host ($UNAME): installing core training deps only."
  echo "  Run this script on your Linux GPU server for the full stack."
  pip install \
    torch transformers peft datasets tqdm accelerate bitsandbytes \
    sentencepiece safetensors huggingface_hub pyyaml
fi

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

if command -v nvidia-smi &>/dev/null; then
  nvidia-smi
else
  echo "nvidia-smi not found (CPU-only host or no NVIDIA driver)."
fi

echo "Setup complete. Activate with: source .venv/bin/activate"
