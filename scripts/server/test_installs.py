#!/usr/bin/env python3
"""Test that the active env (e.g. conda vllm) has packages needed for training + eval."""

from __future__ import annotations

import importlib
import sys


def _check(name: str, import_path: str | None = None) -> tuple[bool, str]:
    path = import_path or name
    try:
        mod = importlib.import_module(path)
        ver = getattr(mod, "__version__", "?")
        return True, f"{name}: OK ({ver})"
    except Exception as exc:
        return False, f"{name}: FAIL ({exc})"


def main() -> int:
    import os

    print(f"Python: {sys.executable}")
    print(f"Version: {sys.version}")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        print("WARNING: PYTHONNOUSERSITE is not 1 — ~/.local may shadow conda packages.")
        print("  Use: source scripts/server/activate_vllm.sh")
    import site as _site

    user_site = getattr(_site, "USER_SITE", "")
    if user_site and user_site in sys.path:
        print(f"WARNING: user-site on sys.path: {user_site}")
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if conda_env:
        print(f"CONDA_DEFAULT_ENV: {conda_env}")
    else:
        print("CONDA_DEFAULT_ENV: (not set — are you in conda vllm?)")

    if "/.venv/" in sys.executable:
        print("ERROR: Using .venv Python — run: source scripts/server/activate_vllm.sh")
        return 1

    expected = os.environ.get("CONDA_ENV_NAME", "vllm")
    if not conda_env:
        print(f"ERROR: Not in a conda env. Run: conda activate {expected}")
        return 1
    if conda_env != expected:
        print(f"ERROR: expected conda env '{expected}', got '{conda_env}'")
        return 1
    print()

    required = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("peft", "peft"),
        ("datasets", "datasets"),
        ("accelerate", "accelerate"),
        ("bitsandbytes", "bitsandbytes"),
        ("tqdm", "tqdm"),
        ("huggingface_hub", "huggingface_hub"),
    ]
    optional = [
        ("vllm", "vllm"),
        ("sentencepiece", "sentencepiece"),
        ("safetensors", "safetensors"),
    ]

    if "/.local/" in sys.executable or "envs/vllm" not in sys.executable.replace("\\", "/"):
        print(
            "WARNING: Python is not clearly from conda env vllm — "
            "run: source scripts/server/activate_vllm.sh"
        )

    failed = 0
    for name, path in required:
        ok, msg = _check(name, path)
        print(msg)
        if not ok:
            failed += 1

    print()
    print("--- optional ---")
    for name, path in optional:
        ok, msg = _check(name, path)
        print(msg)

    print()
    try:
        import torch

        cuda = torch.cuda.is_available()
        print(f"torch.cuda.is_available(): {cuda}")
        if cuda:
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            print("WARNING: CUDA not available (use a GPU node / load cuda module)")
            failed += 1
    except Exception as exc:
        print(f"torch CUDA check: FAIL ({exc})")
        failed += 1

    print()
    print("--- vLLM LoRA (eval / checkpoint sweep) ---")
    try:
        from vllm.lora.request import LoRARequest  # noqa: F401

        print("vllm.lora.request.LoRARequest: OK")
    except Exception as exc:
        print(f"vllm.lora.request.LoRARequest: FAIL ({exc})")
        failed += 1

    print()
    if failed:
        print(f"FAILED: {failed} required check(s).")
        print("  conda activate vllm")
        print("  bash scripts/server/install_into_vllm.sh")
        return 1

    print("PASSED: environment ready for Stage 1/2 pipeline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
