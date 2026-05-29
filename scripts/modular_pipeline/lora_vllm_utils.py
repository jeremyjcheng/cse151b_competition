"""Shared helpers for vLLM LoRA adapter paths and engine configuration."""

from __future__ import annotations

import json
from pathlib import Path

from settings import LORA_R, VLLM_MAX_LORA_RANK


def normalize_vllm_optional(value: str | None) -> str | None:
    """Map CLI sentinels like 'none' to None so vLLM omits quantization/load_format."""
    if value is None:
        return None
    stripped = value.strip()
    if not stripped or stripped.lower() in {"none", "null", "false"}:
        return None
    return stripped


def validate_lora_adapter_dir(adapter_path: str | Path) -> Path:
    """Resolve adapter path and ensure PEFT artifacts exist."""
    resolved = Path(adapter_path).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"LoRA adapter directory not found: {resolved}")

    config_path = resolved / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Missing adapter_config.json in LoRA adapter directory: {resolved}"
        )

    weight_candidates = (
        resolved / "adapter_model.safetensors",
        resolved / "adapter_model.bin",
    )
    if not any(path.is_file() for path in weight_candidates):
        raise FileNotFoundError(
            "Missing adapter weights (adapter_model.safetensors or adapter_model.bin) "
            f"in {resolved}"
        )

    return resolved


def read_adapter_lora_rank(adapter_dir: Path) -> int | None:
    """Read LoRA rank from adapter_config.json when present."""
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.is_file():
        return None
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    rank = cfg.get("r", cfg.get("lora_r"))
    if rank is None:
        return None
    return int(rank)


def resolve_max_lora_rank(adapter_dir: Path | None = None) -> int:
    """Pick max_lora_rank for vLLM from adapter config, bounded below by defaults."""
    floor = max(LORA_R, VLLM_MAX_LORA_RANK)
    if adapter_dir is None:
        return floor
    adapter_rank = read_adapter_lora_rank(adapter_dir)
    if adapter_rank is None:
        return floor
    return max(floor, adapter_rank)


def check_vllm_version(min_version: str) -> None:
    """Warn when installed vLLM is below the recommended minimum for Qwen3 LoRA."""
    try:
        from packaging.version import Version
    except ImportError:
        return

    try:
        import vllm  # type: ignore
    except ImportError:
        return

    installed = getattr(vllm, "__version__", "0.0.0")
    if Version(installed) < Version(min_version):
        print(
            f"Warning: vLLM {installed} is below recommended {min_version} for Qwen3 LoRA. "
            "Upgrade vLLM if LoRA appears ignored or you see Transformers fallback logs."
        )
