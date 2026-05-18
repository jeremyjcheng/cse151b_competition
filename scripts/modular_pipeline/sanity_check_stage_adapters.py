"""Quick sanity check comparing base, Stage-1, and Stage-2 adapter outputs.

This is intentionally lightweight: it checks a few fixed public examples to catch
obvious Stage-2 overfitting regressions before full inference.
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
from settings import MAX_TOKENS_FREE, MAX_TOKENS_MCQ, SYSTEM_PROMPT_FREE, SYSTEM_PROMPT_MCQ
from text_processing import extract_boxed


def _discover_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "data").exists():
            return candidate
    return start.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare base vs Stage-1 vs Stage-2 outputs on a few public samples.",
    )
    parser.add_argument("--stage1-adapter-path", required=True, help="Path to Stage-1 final_adapter.")
    parser.add_argument("--stage2-adapter-path", required=True, help="Path to Stage-2 final_adapter.")
    parser.add_argument("--gpu-id", default="0", help="CUDA_VISIBLE_DEVICES value.")
    parser.add_argument("--input", default="public", help="Dataset split/path used for sanity checks.")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of deterministic samples to compare.",
    )
    parser.add_argument("--vllm-quantization", default=None, help="Optional vLLM quantization override.")
    parser.add_argument("--vllm-load-format", default=None, help="Optional vLLM load_format override.")
    return parser.parse_args()


def _pick_samples(data: list[dict], num_samples: int) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    seen_mcq = False
    seen_free = False

    for item in data:
        is_mcq = bool(item.get("options"))
        if is_mcq and not seen_mcq:
            out.append(("mcq", item))
            seen_mcq = True
        elif (not is_mcq) and not seen_free:
            out.append(("free", item))
            seen_free = True
        if len(out) >= num_samples:
            return out

    for item in data:
        tag = "mcq" if item.get("options") else "free"
        pair = (tag, item)
        if pair not in out:
            out.append(pair)
        if len(out) >= num_samples:
            break
    return out


def _generate_one(pipe: ModularPipeline, kind: str, item: dict) -> str:
    if kind == "mcq":
        user_prompt = (
            f"Q: {item['question']}\n\nOptions:\n"
            + "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(item["options"]))
            + "\n\nReturn exactly one final boxed option, like \\boxed{A}."
        )
        rows = pipe._generate_batch(
            [SYSTEM_PROMPT_MCQ],
            [user_prompt],
            max_new_tokens=min(512, MAX_TOKENS_MCQ),
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            repetition_penalty=1.0,
            do_sample=False,
            think_budget=0,
            enable_thinking=False,
        )
    else:
        rows = pipe._generate_batch(
            [SYSTEM_PROMPT_FREE],
            [item["question"]],
            max_new_tokens=min(512, MAX_TOKENS_FREE),
            temperature=0.1,
            top_p=0.9,
            top_k=10,
            repetition_penalty=1.05,
            do_sample=True,
            think_budget=0,
            enable_thinking=True,
        )
    return rows[0]["raw"]


def _release(pipe: ModularPipeline | None) -> None:
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


def _run_model(tag: str, adapter_path: str | None, samples: list[tuple[str, dict]], kwargs: dict) -> dict[int, str]:
    print(f"\n=== {tag} ===")
    outputs: dict[int, str] = {}
    pipe = ModularPipeline(lora_adapter_path=adapter_path, **kwargs)
    try:
        for idx, (kind, item) in enumerate(samples):
            raw = _generate_one(pipe, kind, item)
            outputs[idx] = raw
            boxed = extract_boxed(raw)
            preview = raw[:160].replace("\n", " ")
            print(f"[{idx}] kind={kind} boxed={boxed!r} preview={preview!r}...")
    finally:
        _release(pipe)
    return outputs


def main() -> None:
    args = parse_args()
    stage1_path = validate_lora_adapter_dir(args.stage1_adapter_path)
    stage2_path = validate_lora_adapter_dir(args.stage2_adapter_path)

    here = Path(__file__).resolve().parent
    root = _discover_project_root(here)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    input_path = resolve_input_path(args.input, root)
    with open(input_path, encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    samples = _pick_samples(data, max(1, args.num_samples))
    print(f"Sanity samples: {len(samples)} from {input_path}")

    model_kwargs = {
        "gpu_id": args.gpu_id,
        "vllm_quantization": args.vllm_quantization,
        "vllm_load_format": args.vllm_load_format,
    }

    base = _run_model("Base model", None, samples, model_kwargs)
    stage1 = _run_model("Stage 1 adapter", str(stage1_path), samples, model_kwargs)
    stage2 = _run_model("Stage 2 adapter", str(stage2_path), samples, model_kwargs)

    print("\n=== Stage comparison summary ===")
    stage2_same_as_stage1 = 0
    stage2_same_as_base = 0
    for idx in range(len(samples)):
        s2_eq_s1 = stage2[idx] == stage1[idx]
        s2_eq_base = stage2[idx] == base[idx]
        print(f"[{idx}] stage2==stage1: {s2_eq_s1} | stage2==base: {s2_eq_base}")
        stage2_same_as_stage1 += int(s2_eq_s1)
        stage2_same_as_base += int(s2_eq_base)

    # Soft warning only: this is a heuristic sanity check, not a strict evaluation metric.
    if stage2_same_as_base == len(samples):
        print("Warning: Stage 2 matched base on all sanity samples (adaptation may be too weak).")
    if stage2_same_as_stage1 == 0:
        print("Warning: Stage 2 differed on every sample from Stage 1 (adaptation may be too strong).")


if __name__ == "__main__":
    main()
