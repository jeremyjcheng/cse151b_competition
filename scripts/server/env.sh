#!/usr/bin/env bash
# Activate conda env "vllm" — source this file (bash or zsh).
# Never uses project .venv. Does NOT use set -e (would close zsh when sourced).
#
# Usage:  source scripts/server/env.sh
#         source scripts/server/activate_vllm.sh

# Resolve this file's directory (bash + zsh when sourced).
if [[ -n "${ZSH_VERSION:-}" ]]; then
  _ENV_SCRIPT="${(%):-%x}"
elif [[ -n "${BASH_VERSION:-}" ]]; then
  _ENV_SCRIPT="${BASH_SOURCE[0]}"
else
  _ENV_SCRIPT="$0"
fi
_SERVER_DIR="$(cd "$(dirname "${_ENV_SCRIPT}")" && pwd)"

_env_die() {
  echo "$*" >&2
  return 1
}

if [[ -n "${_CSE151B_ENV_LOADED:-}" ]]; then
  return 0 2>/dev/null || true
fi

export ROOT="${ROOT:-$(cd "${_SERVER_DIR}/../.." && pwd)}"
if ! cd "$ROOT" 2>/dev/null; then
  _env_die "Cannot cd to project root: $ROOT"
  return 1
fi

CONDA_ENV_NAME="${CONDA_ENV_NAME:-vllm}"
export PYTHONNOUSERSITE=1

if [[ -n "${VIRTUAL_ENV:-}" && "$VIRTUAL_ENV" == *".venv"* ]]; then
  echo "Clearing stale VIRTUAL_ENV (.venv): $VIRTUAL_ENV"
  deactivate 2>/dev/null || true
  unset VIRTUAL_ENV
fi

if [[ -d "${ROOT}/.venv" ]]; then
  echo "ERROR: ${ROOT}/.venv exists and can hijack your shell/IDE."
  echo "  Run: bash scripts/server/remove_venv.sh"
  _env_die "Remove .venv first."
  return 1
fi

_find_conda_base() {
  if command -v conda >/dev/null 2>&1; then
    local base
    base="$(conda info --base 2>/dev/null)" || return 1
    if [[ -n "$base" && -f "${base}/etc/profile.d/conda.sh" ]]; then
      echo "$base"
      return 0
    fi
  fi
  local candidate
  for candidate in /opt/conda "${HOME}/miniconda3" "${HOME}/anaconda3" "${HOME}/miniforge3"; do
    if [[ -f "${candidate}/etc/profile.d/conda.sh" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

_activate_conda() {
  if [[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV_NAME" ]]; then
    return 0
  fi
  local conda_base
  conda_base="$(_find_conda_base)" || return 1
  # shellcheck disable=SC1091
  source "${conda_base}/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV_NAME" || return 1
  [[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV_NAME" ]]
}

_activate_conda_direct() {
  local conda_base env_python
  conda_base="$(_find_conda_base)" || return 1
  env_python="${conda_base}/envs/${CONDA_ENV_NAME}/bin/python"
  if [[ ! -x "$env_python" ]]; then
    return 1
  fi
  export CONDA_PREFIX="${conda_base}/envs/${CONDA_ENV_NAME}"
  export CONDA_DEFAULT_ENV="$CONDA_ENV_NAME"
  export PATH="${CONDA_PREFIX}/bin:${conda_base}/bin:${PATH}"
  hash -r 2>/dev/null || true
  return 0
}

if ! _activate_conda; then
  echo "Note: conda activate failed. Trying direct env path..."
  if ! _activate_conda_direct; then
    echo "ERROR: Could not activate conda env '$CONDA_ENV_NAME'."
    echo "  source /opt/conda/etc/profile.d/conda.sh"
    echo "  conda env list"
    echo "  conda activate $CONDA_ENV_NAME"
    return 1
  fi
fi

export PY="$(command -v python)"

case "$PY" in
  *"/.venv/"*)
    echo "ERROR: Python still points at .venv: $PY"
    echo "  Run: bash scripts/server/remove_venv.sh"
    return 1
    ;;
esac

if [[ "$(basename "${CONDA_PREFIX:-}")" != "$CONDA_ENV_NAME" ]]; then
  echo "WARNING: CONDA_PREFIX is not $CONDA_ENV_NAME (got: ${CONDA_PREFIX:-unset})"
  echo "  Python: $PY"
fi

if [[ "$PY" != *"/envs/${CONDA_ENV_NAME}/"* && "$PY" != "${CONDA_PREFIX}/bin/python" ]]; then
  echo "WARNING: python may not be from env $CONDA_ENV_NAME: $PY"
  echo "  source /opt/conda/etc/profile.d/conda.sh && conda activate $CONDA_ENV_NAME"
fi

_setup_cuda_lib_path() {
  local -a lib_dirs=()
  local d pip_cuda_lib added=0

  if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
    lib_dirs+=("${CONDA_PREFIX}/lib")
  fi
  for d in /usr/local/cuda/lib64 /usr/local/cuda-13/lib64 /usr/local/cuda-12/lib64; do
    [[ -d "$d" ]] && lib_dirs+=("$d")
  done

  pip_cuda_lib="$("$PY" - <<'PY' 2>/dev/null || true
import os
candidates = []
prefix = os.environ.get("CONDA_PREFIX")
if prefix:
    candidates.append(os.path.join(prefix, "lib"))
try:
    import nvidia.cuda_runtime  # type: ignore
    candidates.append(os.path.join(os.path.dirname(nvidia.cuda_runtime.__file__), "lib"))
except Exception:
    pass
for base in __import__("site").getsitepackages():
    for sub in ("nvidia/cuda_runtime/lib", "nvidia/cudnn/lib"):
        p = os.path.join(base, sub)
        if os.path.isdir(p):
            candidates.append(p)
for d in candidates:
    for name in ("libcudart.so.13", "libcudart.so.12", "libcudart.so"):
        if os.path.isfile(os.path.join(d, name)):
            print(d)
            raise SystemExit(0)
PY
)" || true

  if [[ -n "$pip_cuda_lib" ]]; then
    lib_dirs+=("$pip_cuda_lib")
  fi

  for d in "${lib_dirs[@]}"; do
    [[ -d "$d" ]] || continue
    if [[ -f "${d}/libcudart.so.13" || -f "${d}/libcudart.so.12" || -f "${d}/libcudart.so" ]]; then
      if [[ ":${LD_LIBRARY_PATH:-}:" != *":${d}:"* ]]; then
        export LD_LIBRARY_PATH="${d}:${LD_LIBRARY_PATH:-}"
        added=1
      fi
    fi
  done
  if [[ "$added" -eq 1 ]]; then
    echo "CUDA libs prepended to LD_LIBRARY_PATH (first: ${LD_LIBRARY_PATH%%:*})"
  fi
}

_setup_cuda_lib_path || true

export PYTHONUNBUFFERED=1
export _CSE151B_ENV_LOADED=1
echo "Conda env: ${CONDA_DEFAULT_ENV:-$CONDA_ENV_NAME}"
echo "Python:    $PY"
echo "PYTHONNOUSERSITE=1"
