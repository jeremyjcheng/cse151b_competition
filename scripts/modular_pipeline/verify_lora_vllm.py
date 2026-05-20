"""Smoke test: confirm vLLM applies a LoRA adapter (base vs adapter outputs differ).

Quantization fallback matrix (run manually if this script fails):
  A) Default bitsandbytes + LoRA + max_lora_rank from adapter
  B) --vllm-quantization none --vllm-load-format auto
  C) Upgrade vLLM to >= setting.VLLM_MIN_VERSION (Qwen3 native LoRA)
  D) Lower gpu_memory_utilization in setting.py if B OOMs

Requires GPU, vLLM, and a valid PEFT adapter directory.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

from cli_utils import resolve_input_path
from lora_vllm_utils import validate_lora_adapter_dir
from model_pipeline import ModularPipeline
from prompting import build_free_user, build_mcq_user
from settings import (
    MAX_TOKENS_FREE,
    MAX_TOKENS_MCQ,
    SYSTEM_PROMPT_FREE,
    SYSTEM_PROMPT_MCQ,
    VLLM_ENFORCE_EAGER,
    VLLM_MIN_VERSION,
)


def _discover_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "data").exists():
            return candidate
    return start.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify vLLM LoRA changes model outputs vs base-only inference.",
    )
    parser.add_argument(
        "--lora-adapter-path",
        required=True,
        help="Path to PEFT final_adapter directory.",
    )
    parser.add_argument("--gpu-id", default="0", help="CUDA_VISIBLE_DEVICES value.")
    parser.add_argument(
        "--input",
        default="public",
        help="Dataset for sample prompts (default: public).",
    )
    parser.add_argument(
        "--vllm-quantization",
        default=None,
        help="Optional vLLM quantization override (use 'none' to disable).",
    )
    parser.add_argument(
        "--vllm-load-format",
        default=None,
        help="Optional vLLM load_format override.",
    )
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override vLLM enforce_eager (False enables CUDA graphs).",
    )
    return parser.parse_args()


def _pick_samples(data: list[dict]) -> list[tuple[str, dict]]:
    """Return up to one MCQ and one free-form example."""
    samples: list[tuple[str, dict]] = []
    for item in data:
        if item.get("options") and not any(kind == "mcq" for kind, _ in samples):
            samples.append(("mcq", item))
        elif not item.get("options") and not any(kind == "free" for kind, _ in samples):
            samples.append(("free", item))
        if len(samples) >= 2:
            break
    if not samples:
        raise SystemExit("No usable examples found in input dataset.")
    return samples


def _generate_one(
    pipe: ModularPipeline,
    kind: str,
    item: dict,
) -> str:
    if kind == "mcq":
        user = build_mcq_user(item["question"], item["options"])
        rows = pipe._generate_batch(
            [SYSTEM_PROMPT_MCQ],
            [user],
            max_new_tokens=MAX_TOKENS_MCQ,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            repetition_penalty=1.0,
            do_sample=False,
            think_budget=0,
            enable_thinking=False,
        )
    else:
        user = build_free_user(item["question"])
        rows = pipe._generate_batch(
            [SYSTEM_PROMPT_FREE],
            [user],
            max_new_tokens=min(256, MAX_TOKENS_FREE),
            temperature=0.1,
            top_p=0.9,
            top_k=10,
            repetition_penalty=1.05,
            do_sample=True,
            think_budget=0,
            enable_thinking=True,
        )
    return rows[0]["raw"]


def _unload_pipeline(pipe: ModularPipeline | None) -> None:
    if pipe is None:
        return
    del pipe
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def main() -> None:
    args = parse_args()
    here = Path(__file__).resolve().parent
    root = _discover_project_root(here)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    adapter_path = validate_lora_adapter_dir(args.lora_adapter_path)
    input_path = resolve_input_path(args.input, root)
    with open(input_path, encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    samples = _pick_samples(data)
    print(f"Using {len(samples)} fixed prompt(s) from {input_path}")
    print(f"Adapter: {adapter_path}")
    print(f"Recommended vLLM >= {VLLM_MIN_VERSION}")

    enforce_eager = VLLM_ENFORCE_EAGER if args.enforce_eager is None else args.enforce_eager
    vllm_kwargs = dict(
        gpu_id=args.gpu_id,
        vllm_quantization=args.vllm_quantization,
        vllm_load_format=args.vllm_load_format,
        enforce_eager=enforce_eager,
    )
    print(f"enforce_eager={enforce_eager}")

    base_outputs: dict[str, str] = {}
    print("\n=== Base model (no LoRA) ===")
    base_pipe = ModularPipeline(lora_adapter_path=None, **vllm_kwargs)
    try:
        for kind, item in samples:
            base_outputs[kind] = _generate_one(base_pipe, kind, item)
            preview = base_outputs[kind][:200].replace("\n", " ")
            print(f"  [{kind}] {preview!r}...")
    finally:
        _unload_pipeline(base_pipe)

    lora_outputs: dict[str, str] = {}
    print("\n=== LoRA adapter ===")
    lora_pipe = ModularPipeline(lora_adapter_path=str(adapter_path), **vllm_kwargs)
    try:
        for kind, item in samples:
            lora_outputs[kind] = _generate_one(lora_pipe, kind, item)
            preview = lora_outputs[kind][:200].replace("\n", " ")
            print(f"  [{kind}] {preview!r}...")
    finally:
        _unload_pipeline(lora_pipe)

    print("\n=== Comparison ===")
    any_diff = False
    for kind, _item in samples:
        base_text = base_outputs[kind]
        lora_text = lora_outputs[kind]
        same = base_text == lora_text
        print(f"  [{kind}] identical={same}")
        if not same:
            any_diff = True

    if not any_diff:
        raise SystemExit(
            "FAIL: LoRA outputs matched base on all prompts — adapter may not be applied. "
            "Try --vllm-quantization none --vllm-load-format auto or upgrade vLLM."
        )

    print("PASS: At least one prompt differed between base and LoRA runs.")


if __name__ == "__main__":
    main()
