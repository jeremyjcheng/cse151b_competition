#!/usr/bin/env python3
"""Minimal LoRA smoke test: one short vLLM generate() with an adapter.

Bypasses the full dataset pipeline to isolate hangs in vLLM init vs first generate.

Example:
  python scripts/modular_pipeline/lora_smoke_test.py \\
    --lora-adapter-path artifacts/lora_best_v1/v2/stage2_adapt/final_adapter \\
    --gpu-id 0

  # Test without bitsandbytes:
  python scripts/modular_pipeline/lora_smoke_test.py \\
    --lora-adapter-path artifacts/.../final_adapter --no-bitsandbytes
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

from lora_vllm_utils import validate_lora_adapter_dir
from settings import MODEL_ID, VLLM_MIN_VERSION


def _discover_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "data").exists():
            return candidate
    return start.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-prompt vLLM LoRA smoke test.")
    parser.add_argument("--lora-adapter-path", required=True)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument(
        "--no-bitsandbytes",
        action="store_true",
        help="Use full-precision loading (no bitsandbytes quantization).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Keep small (default 32) so a hang is obvious quickly.",
    )
    parser.add_argument(
        "--vllm-enforce-eager",
        action="store_true",
        help="Pass enforce_eager=True to vLLM (skips CUDA graphs / may avoid compile hangs).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    here = Path(__file__).resolve().parent
    root = _discover_project_root(here)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    adapter = validate_lora_adapter_dir(args.lora_adapter_path)
    vllm_quantization = None if args.no_bitsandbytes else "bitsandbytes"
    vllm_load_format = "auto" if args.no_bitsandbytes else "bitsandbytes"

    print(f"Recommended vLLM >= {VLLM_MIN_VERSION}")
    print(f"Adapter: {adapter}")
    print(f"quantization={vllm_quantization!r} load_format={vllm_load_format!r}")
    print(f"max_new_tokens={args.max_new_tokens}")

    from model_pipeline import ModularPipeline

    print("\n[smoke] Building ModularPipeline (vLLM init)...", flush=True)
    pipe = ModularPipeline(
        gpu_id=args.gpu_id,
        lora_adapter_path=str(adapter),
        vllm_quantization=vllm_quantization,
        vllm_load_format=vllm_load_format,
        enforce_eager=args.vllm_enforce_eager,
    )
    print("[smoke] ModularPipeline ready.", flush=True)

    prompt = "What is 2+2? Reply with one short sentence."
    try:
        chat = pipe.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except TypeError:
        chat = f"<|user|>\n{prompt}\n<|assistant|>\n"

    from vllm import SamplingParams

    sampling = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0)
    gen_kwargs: dict = dict(sampling_params=sampling, use_tqdm=True)
    if pipe._lora_request is not None:
        gen_kwargs["lora_request"] = pipe._lora_request

    print("\n[smoke] BEFORE llm.generate (single prompt)", flush=True)
    print(f"  lora_request={pipe._lora_request!r}", flush=True)
    outputs = pipe.llm.generate([chat], **gen_kwargs)
    print("[smoke] AFTER llm.generate", flush=True)

    text = outputs[0].outputs[0].text
    print(f"\n[smoke] Output ({len(text)} chars):\n{text[:500]}")

    del pipe
    gc.collect()


if __name__ == "__main__":
    main()
