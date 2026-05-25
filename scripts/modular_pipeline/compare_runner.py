"""Side-by-side base-vs-LoRA inference with a separated diagnostic / deploy
winner pipeline.

This entry point is intentionally additive on top of `modular_pipeline.py`:
the existing 0.491 base run path (`runner.main`) is not changed and not
imported here. The default compare mode is `sequential` which builds the
base engine first with `enforce_eager=False` so the base pass bit-matches
the previous baseline.

Output files (under `--output-dir`):

- ``{stem}_base_outputs.jsonl``  -- base pass, one record per item
- ``{stem}_lora_outputs.jsonl``  -- LoRA pass, one record per item
- ``{stem}_comparison.jsonl``    -- merged with both winners
- ``{stem}_summary.md``          -- truth metrics + deploy metrics +
  confusion matrix + recommendation
- ``{stem}_deploy_outputs.jsonl`` (when ``--gated``) -- one record per
  item using `deploy_response`
- ``{stem}_deploy_submission.csv`` (when ``--gated``) -- the gated CSV
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
from functools import partial
from pathlib import Path
from typing import Any

from tqdm import tqdm

from cli_utils import (
    apply_subset_caps,
    apply_vllm_cli_overrides,
    build_run_stem,
    resolve_input_path,
)
from evaluation import _load_project_judger, _safe_auto_judge
from formatting_diagnostics import score_output
from lora_gate import decide_deployable_winner, score_diagnostic_winner
from settings import (
    FREE_BATCH_SIZE,
    MAX_TOKENS_FREE,
    MAX_TOKENS_MCQ,
    MAX_TOKENS_MCQ_FINAL,
    MCQ_BATCH_SIZE,
)
from text_processing import extract_boxed, truncate_after_first_valid_mcq_box


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run base and LoRA inference on the same items and produce a "
            "comparison JSONL, a summary report, and (optionally) a "
            "deploy-gated submission CSV. Defaults to sequential mode so "
            "the base pass matches the existing 0.491 baseline."
        ),
    )
    parser.add_argument(
        "--input",
        default="public",
        help=(
            "'public' (default; enables truth metrics if `answer` is present), "
            "'private', or a path to a .jsonl file."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for outputs. Default: <project_root>/results/compare.",
    )
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument(
        "--lora-adapter-path",
        required=True,
        help="Path to the trained LoRA adapter directory.",
    )
    parser.add_argument("--vllm-quantization", default=None)
    parser.add_argument("--vllm-load-format", default=None)
    parser.add_argument("--no-bitsandbytes", action="store_true")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16"),
        default=None,
    )
    parser.add_argument(
        "--vllm-enforce-eager",
        action="store_true",
        help=(
            "Force enforce_eager=True on the base pass too. By default the "
            "base pass uses the project default (False) so it matches the "
            "0.491 baseline; the LoRA pass always uses True per the vLLM "
            "0.8.5 LoRA-hang workaround."
        ),
    )
    parser.add_argument(
        "--inference-backend",
        choices=("vllm", "peft"),
        default="vllm",
        help=(
            "Inference engine for the LoRA pass. Default vllm. peft is "
            "supported for both passes only when LoRA is loaded."
        ),
    )
    parser.add_argument("--limit-mcq", type=int, default=None)
    parser.add_argument("--limit-free", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument(
        "--save-raw-output",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--mcq-max-new-tokens",
        type=int,
        default=MAX_TOKENS_MCQ,
    )
    parser.add_argument(
        "--mcq-final-max-new-tokens",
        type=int,
        default=MAX_TOKENS_MCQ_FINAL,
    )
    parser.add_argument(
        "--free-max-new-tokens",
        type=int,
        default=MAX_TOKENS_FREE,
    )
    parser.add_argument(
        "--lora-only",
        action="store_true",
        help=(
            "Run only the LoRA inference pass (skip base). Writes "
            "{stem}_lora_outputs.jsonl only; no comparison report. "
            "Score with compare_runs.py on that file."
        ),
    )
    parser.add_argument(
        "--compare-mode",
        choices=("sequential", "shared"),
        default="sequential",
        help=(
            "sequential (default): build base engine, run, tear down, then "
            "build a separate LoRA engine. Slower but base bit-matches "
            "0.491. shared: one engine with enable_lora=True; toggle the "
            "adapter per request (faster but base inherits enforce_eager=True)."
        ),
    )
    parser.add_argument(
        "--gated",
        action="store_true",
        help=(
            "Also write a deploy-gated submission CSV using "
            "decide_deployable_winner() per item."
        ),
    )
    parser.add_argument(
        "--lora-gate-strict",
        action="store_true",
        help=(
            "Tighten the deploy gate: only return LoRA when its confidence "
            "score beats base by >= 2 and (if MCQ) its boxed letter is a "
            "valid option."
        ),
    )
    parser.add_argument(
        "--no-eval-comparison",
        action="store_true",
        help=(
            "Skip judger-based truth_winner computation even when gold is "
            "present. The deploy_winner column is still produced."
        ),
    )
    parser.add_argument(
        "--judge-timeout-s",
        type=float,
        default=1.0,
        help=(
            "Per-item sympy judger timeout during comparison (default 1.0s). "
            "Lower is faster but may mark hard items wrong. Use "
            "--no-eval-comparison to skip judger entirely."
        ),
    )

    args = parser.parse_args()
    return apply_vllm_cli_overrides(args)


def _discover_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "data").exists():
            return candidate
    return start.parent


def _load_done_ids(output_path: Path) -> set:
    done_ids: set = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("id") is not None:
                        done_ids.add(rec["id"])
                except json.JSONDecodeError:
                    pass
    return done_ids


def _load_records_by_id(output_path: Path) -> dict:
    records_by_id: dict = {}
    if not output_path.exists():
        return records_by_id
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records_by_id[rec["id"]] = rec
            except json.JSONDecodeError:
                pass
    return records_by_id


def _write_records(
    file_obj,
    chunk: list[dict],
    solved_batch: list[dict],
    *,
    pass_label: str,
    save_raw_output: bool = True,
) -> None:
    for item, solved in zip(chunk, solved_batch):
        rec = {
            "id": item.get("id"),
            "is_mcq": bool(item.get("options")),
            "response": solved["response"],
            "meta": solved["meta"],
            "gate_pass": pass_label,
        }
        if save_raw_output:
            rec["raw"] = solved.get("raw")
        file_obj.write(json.dumps(rec) + "\n")
    file_obj.flush()
    os.fsync(file_obj.fileno())


def _run_pipeline_pass(
    pipe,
    data: list[dict],
    output_path: Path,
    *,
    pass_label: str,
    save_raw_output: bool,
) -> None:
    done_ids = _load_done_ids(output_path)
    if done_ids:
        print(f"[{pass_label}] found {len(done_ids)} completed records; resuming.")
    remaining = [item for item in data if item.get("id") not in done_ids]
    mcq_items = [item for item in remaining if item.get("options")]
    free_items = [item for item in remaining if not item.get("options")]
    print(
        f"[{pass_label}] remaining: total={len(remaining)} "
        f"mcq={len(mcq_items)} free={len(free_items)}"
    )

    with open(output_path, "a") as f:
        for start in tqdm(
            range(0, len(mcq_items), MCQ_BATCH_SIZE),
            desc=f"[{pass_label}] MCQ",
        ):
            chunk = mcq_items[start : start + MCQ_BATCH_SIZE]
            solved_batch = pipe.solve_mcq_batch(chunk)
            _write_records(
                f, chunk, solved_batch,
                pass_label=pass_label,
                save_raw_output=save_raw_output,
            )
        for start in tqdm(
            range(0, len(free_items), FREE_BATCH_SIZE),
            desc=f"[{pass_label}] free",
        ):
            chunk = free_items[start : start + FREE_BATCH_SIZE]
            solved_batch = pipe.solve_free_batch(chunk)
            _write_records(
                f, chunk, solved_batch,
                pass_label=pass_label,
                save_raw_output=save_raw_output,
            )


def _teardown_pipeline(pipe) -> None:
    try:
        if getattr(pipe, "llm", None) is not None:
            try:
                del pipe.llm
            except Exception:
                pipe.llm = None
        if getattr(pipe, "_peft_engine", None) is not None:
            try:
                del pipe._peft_engine
            except Exception:
                pipe._peft_engine = None
    finally:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _run_sequential(
    args: argparse.Namespace,
    data: list[dict],
    base_outputs_path: Path,
    lora_outputs_path: Path,
) -> None:
    from model_pipeline import ModularPipeline

    if getattr(args, "lora_only", False):
        print("[sequential] --lora-only: skipping base pass.")
    else:
        base_done = _load_done_ids(base_outputs_path)
        base_remaining = [item for item in data if item.get("id") not in base_done]
    if not getattr(args, "lora_only", False) and base_remaining:
        print("[sequential] === BASE PASS ===")
        base_pipe = ModularPipeline(
            gpu_id=args.gpu_id,
            lora_adapter_path=None,
            vllm_quantization=args.vllm_quantization,
            vllm_load_format=args.vllm_load_format,
            enforce_eager=True if args.vllm_enforce_eager else None,
            inference_backend="vllm",
            mcq_max_new_tokens=args.mcq_max_new_tokens,
            mcq_final_max_new_tokens=args.mcq_final_max_new_tokens,
            free_max_new_tokens=args.free_max_new_tokens,
        )
        try:
            _run_pipeline_pass(
                base_pipe, data, base_outputs_path,
                pass_label="base",
                save_raw_output=args.save_raw_output,
            )
        finally:
            _teardown_pipeline(base_pipe)
            del base_pipe
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
    elif not getattr(args, "lora_only", False):
        print("[sequential] base pass already complete; skipping.")

    lora_done = _load_done_ids(lora_outputs_path)
    lora_remaining = [item for item in data if item.get("id") not in lora_done]
    if lora_remaining:
        print("[sequential] === LORA PASS ===")
        lora_pipe = ModularPipeline(
            gpu_id=args.gpu_id,
            lora_adapter_path=args.lora_adapter_path,
            vllm_quantization=args.vllm_quantization,
            vllm_load_format=args.vllm_load_format,
            enforce_eager=True,
            inference_backend=args.inference_backend,
            mcq_max_new_tokens=args.mcq_max_new_tokens,
            mcq_final_max_new_tokens=args.mcq_final_max_new_tokens,
            free_max_new_tokens=args.free_max_new_tokens,
        )
        try:
            _run_pipeline_pass(
                lora_pipe, data, lora_outputs_path,
                pass_label="lora",
                save_raw_output=args.save_raw_output,
            )
        finally:
            _teardown_pipeline(lora_pipe)
            del lora_pipe
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
    else:
        print("[sequential] lora pass already complete; skipping.")


def _run_shared(
    args: argparse.Namespace,
    data: list[dict],
    base_outputs_path: Path,
    lora_outputs_path: Path,
) -> None:
    from model_pipeline import ModularPipeline

    pipe = ModularPipeline(
        gpu_id=args.gpu_id,
        lora_adapter_path=args.lora_adapter_path,
        vllm_quantization=args.vllm_quantization,
        vllm_load_format=args.vllm_load_format,
        enforce_eager=True,
        inference_backend="vllm",
        mcq_max_new_tokens=args.mcq_max_new_tokens,
        mcq_final_max_new_tokens=args.mcq_final_max_new_tokens,
        free_max_new_tokens=args.free_max_new_tokens,
    )

    try:
        if getattr(args, "lora_only", False):
            print("[shared] --lora-only: skipping base pass.")
        base_done = _load_done_ids(base_outputs_path)
        if not getattr(args, "lora_only", False) and any(
            item.get("id") not in base_done for item in data
        ):
            print("[shared] === BASE PASS (lora_active=False) ===")
            pipe.set_lora_active(False)
            _run_pipeline_pass(
                pipe, data, base_outputs_path,
                pass_label="base",
                save_raw_output=args.save_raw_output,
            )
        else:
            print("[shared] base pass already complete; skipping.")

        lora_done = _load_done_ids(lora_outputs_path)
        if any(item.get("id") not in lora_done for item in data):
            print("[shared] === LORA PASS (lora_active=True) ===")
            pipe.set_lora_active(True)
            _run_pipeline_pass(
                pipe, data, lora_outputs_path,
                pass_label="lora",
                save_raw_output=args.save_raw_output,
            )
        else:
            print("[shared] lora pass already complete; skipping.")
    finally:
        _teardown_pipeline(pipe)
        del pipe
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _build_comparison_rows(
    args: argparse.Namespace,
    data: list[dict],
    base_records: dict,
    lora_records: dict,
    *,
    use_judger: bool,
    judge_timeout_s: float = 1.0,
) -> tuple[list[dict], Any]:
    """Combine the two pass JSONLs into per-item comparison rows.

    Returns (rows, judger_instance). The judger instance is None when
    gold-based scoring was skipped.
    """
    judger_instance = None
    if use_judger:
        print(
            "Loading Judger for truth metrics (sympy import can take 30-60s on "
            "first use)..."
        )
        JudgerCls = _load_project_judger()
        if JudgerCls is not None:
            try:
                judger_instance = JudgerCls(strict_extract=False)
                print("Judger ready.")
            except Exception as exc:
                print(f"Could not init Judger for comparison: {exc}")
                judger_instance = None

    rows: list[dict] = []
    missing_ids: list = []
    judge_fn = partial(_safe_auto_judge, timeout_s=float(judge_timeout_s))

    for item in tqdm(data, desc="Comparing base vs LoRA"):
        item_id = item.get("id")
        is_mcq = bool(item.get("options"))
        options = list(item.get("options") or [])
        labels = [chr(65 + i) for i in range(len(options))] if is_mcq else []

        base_rec = base_records.get(item_id)
        lora_rec = lora_records.get(item_id)
        if base_rec is None or lora_rec is None:
            missing_ids.append(item_id)
            continue

        base_raw = str(base_rec.get("raw") or base_rec.get("meta", {}).get("raw") or "")
        lora_raw = str(lora_rec.get("raw") or lora_rec.get("meta", {}).get("raw") or "")
        base_response = str(base_rec.get("response") or "")
        lora_response = str(lora_rec.get("response") or "")
        base_meta = base_rec.get("meta") or {}
        lora_meta = lora_rec.get("meta") or {}

        max_new_tokens = (
            args.mcq_max_new_tokens if is_mcq else args.free_max_new_tokens
        )

        # Score the ORIGINAL raw output so repeated boxes / repeated phrases
        # show up as signals against LoRA. The MCQ safety net in
        # decide_deployable_winner will independently use
        # extract_first_valid_letter on lora_raw, which is already first-
        # match biased, so we do not need to pre-truncate for that purpose.
        base_score = score_output(
            base_raw,
            base_response,
            is_mcq=is_mcq,
            labels=labels,
            max_new_tokens=max_new_tokens,
            n_tokens=int(base_meta.get("n_tokens", 0) or 0),
            pre_trunc_n_tokens=base_meta.get("pre_trunc_n_tokens"),
            generation_hit_max=base_meta.get("generation_hit_max"),
        )
        lora_score = score_output(
            lora_raw,
            lora_response,
            is_mcq=is_mcq,
            labels=labels,
            max_new_tokens=max_new_tokens,
            n_tokens=int(lora_meta.get("n_tokens", 0) or 0),
            pre_trunc_n_tokens=lora_meta.get("pre_trunc_n_tokens"),
            generation_hit_max=lora_meta.get("generation_hit_max"),
        )

        # Pre-truncate only for the MCQ safety extraction path so the gate
        # never picks up junk text after the first valid letter; the raw
        # field stored on the row is the unmodified pipeline raw.
        if is_mcq and labels:
            lora_raw_for_gate = truncate_after_first_valid_mcq_box(lora_raw, labels)
        else:
            lora_raw_for_gate = lora_raw

        deploy = decide_deployable_winner(
            base_response,
            lora_response,
            lora_raw_for_gate,
            base_score=base_score,
            lora_score=lora_score,
            is_mcq=is_mcq,
            labels=labels,
            strict=args.lora_gate_strict,
        )

        row: dict[str, Any] = {
            "id": item_id,
            "is_mcq": is_mcq,
            "options": options if is_mcq else None,
            "base_response": base_response,
            "lora_response": lora_response,
            "base_raw": base_raw,
            "lora_raw": lora_raw,
            "base_extracted_answer": extract_boxed(base_response),
            "lora_extracted_answer": extract_boxed(lora_response),
            "base_format_score": base_score,
            "lora_format_score": lora_score,
            "base_meta": base_meta,
            "lora_meta": lora_meta,
            "deploy_winner": deploy["deploy_winner"],
            "deploy_response": deploy["deploy_response"],
            "deploy_reason": deploy["reason"],
            "deploy_meta": deploy["deploy_meta"],
        }

        gold = item.get("answer")
        if gold is not None and judger_instance is not None:
            row["gold"] = gold
            truth = score_diagnostic_winner(
                base_response,
                lora_response,
                gold=gold,
                options=options,
                judger=judger_instance,
                safe_auto_judge=judge_fn,
            )
            row["truth_winner"] = truth["truth_winner"]
            row["base_correct"] = truth["base_correct"]
            row["lora_correct"] = truth["lora_correct"]
            row["truth_reason"] = truth["reason"]

            gold_list = gold if isinstance(gold, list) else [gold]
            options_per_slot = [options or []] * len(gold_list)
            row["deploy_correct"] = bool(
                judge_fn(
                    judger_instance,
                    pred=deploy["deploy_response"],
                    gold=gold_list,
                    options_per_slot=options_per_slot,
                )
            )

        rows.append(row)

    if missing_ids:
        print(
            f"Warning: {len(missing_ids)} items missing from one of the passes; "
            "they were skipped in the comparison."
        )

    return rows, judger_instance


def _acc(correct: int, total: int) -> float:
    return (correct / total * 100.0) if total else 0.0


def _bucket(rows: list[dict], is_mcq: bool) -> list[dict]:
    return [r for r in rows if bool(r["is_mcq"]) == is_mcq]


def _write_summary(
    path: Path,
    rows: list[dict],
    args: argparse.Namespace,
    *,
    input_path: Path,
    base_outputs_path: Path,
    lora_outputs_path: Path,
) -> None:
    n_total = len(rows)
    mcq_rows = _bucket(rows, True)
    free_rows = _bucket(rows, False)
    n_mcq = len(mcq_rows)
    n_free = len(free_rows)

    has_truth = any("truth_winner" in r for r in rows)

    lines: list[str] = []
    lines.append("# Base vs LoRA comparison report")
    lines.append("")
    lines.append("## 1. Run info")
    lines.append("")
    lines.append(f"- input: `{input_path}`")
    lines.append(f"- compare_mode: `{args.compare_mode}`")
    lines.append(f"- adapter: `{args.lora_adapter_path}`")
    lines.append(f"- sample_seed: {args.sample_seed}")
    lines.append(f"- limit_mcq: {args.limit_mcq}")
    lines.append(f"- limit_free: {args.limit_free}")
    lines.append(f"- mcq_max_new_tokens: {args.mcq_max_new_tokens}")
    lines.append(f"- free_max_new_tokens: {args.free_max_new_tokens}")
    lines.append(f"- lora_gate_strict: {args.lora_gate_strict}")
    lines.append(f"- gated_submission: {args.gated}")
    lines.append(f"- total_items: {n_total} (mcq={n_mcq}, free={n_free})")
    lines.append(f"- base_outputs: `{base_outputs_path}`")
    lines.append(f"- lora_outputs: `{lora_outputs_path}`")
    lines.append("")

    if has_truth:
        lines.append("## 2. Truth metrics (gold-aware, public only)")
        lines.append("")
        base_correct = sum(1 for r in rows if r.get("base_correct"))
        lora_correct = sum(1 for r in rows if r.get("lora_correct"))
        lines.append(
            f"- base_acc overall: {base_correct}/{n_total} ({_acc(base_correct, n_total):.2f}%)"
        )
        lines.append(
            f"- lora_acc overall: {lora_correct}/{n_total} ({_acc(lora_correct, n_total):.2f}%)"
        )
        for label, items in (("MCQ", mcq_rows), ("free-form", free_rows)):
            if not items:
                continue
            bc = sum(1 for r in items if r.get("base_correct"))
            lc = sum(1 for r in items if r.get("lora_correct"))
            lines.append(
                f"- {label}: base={bc}/{len(items)} ({_acc(bc, len(items)):.2f}%), "
                f"lora={lc}/{len(items)} ({_acc(lc, len(items)):.2f}%)"
            )
        truth_counts: dict[str, int] = {}
        for r in rows:
            tw = r.get("truth_winner")
            if tw is None:
                continue
            truth_counts[tw] = truth_counts.get(tw, 0) + 1
        lines.append("")
        lines.append(
            f"- truth_winner=lora (LoRA helps): {truth_counts.get('lora', 0)}"
        )
        lines.append(
            f"- truth_winner=base (LoRA harms): {truth_counts.get('base', 0)}"
        )
        lines.append(
            f"- truth_winner=tie_correct: {truth_counts.get('tie_correct', 0)}"
        )
        lines.append(
            f"- truth_winner=tie_wrong: {truth_counts.get('tie_wrong', 0)}"
        )
        lines.append("")

    lines.append("## 3. Deploy metrics (gold-blind)")
    lines.append("")
    deploy_lora = sum(1 for r in rows if r["deploy_winner"] == "lora")
    deploy_base = n_total - deploy_lora
    downgrades = sum(
        1 for r in rows if r["deploy_meta"].get("downgraded_lora_to_base")
    )
    lines.append(f"- deploy_winner=base: {deploy_base}/{n_total}")
    lines.append(f"- deploy_winner=lora: {deploy_lora}/{n_total}")
    lines.append(
        f"- MCQ downgrades (LoRA picked but no valid letter): {downgrades}"
    )
    lines.append("")

    failure_counts: dict[str, int] = {}
    for r in rows:
        for reason in r["lora_format_score"].get("mandatory_fail_reasons", []):
            failure_counts[reason] = failure_counts.get(reason, 0) + 1
    if failure_counts:
        lines.append("Top LoRA hard-fail reasons:")
        for reason, count in sorted(failure_counts.items(), key=lambda kv: -kv[1])[:8]:
            lines.append(f"- {reason}: {count}")
        lines.append("")
    else:
        lines.append("No LoRA hard-fail reasons recorded.")
        lines.append("")

    soft_phrase = sum(
        1 for r in rows if r["lora_format_score"].get("repeated_phrase_after_box", 0) > 0
    )
    soft_boxes = sum(
        1 for r in rows if r["lora_format_score"].get("repeated_boxed_answers", 0) > 0
    )
    lines.append(
        f"LoRA repeated_boxed_answers (>=1 dup): {soft_boxes}/{n_total}"
    )
    lines.append(
        f"LoRA repeated_phrase_after_box (>=1 match): {soft_phrase}/{n_total}"
    )
    lines.append("")

    if has_truth:
        lines.append("## 4. Deploy-vs-truth confusion (public only)")
        lines.append("")
        cells: dict[tuple, int] = {}
        for r in rows:
            tw = r.get("truth_winner")
            if tw is None:
                continue
            cells[(r["deploy_winner"], tw)] = cells.get((r["deploy_winner"], tw), 0) + 1
        lines.append("| deploy \\ truth | lora | base | tie_correct | tie_wrong |")
        lines.append("|---|---|---|---|---|")
        for dw in ("lora", "base"):
            lines.append(
                f"| **{dw}** | "
                f"{cells.get((dw, 'lora'), 0)} | "
                f"{cells.get((dw, 'base'), 0)} | "
                f"{cells.get((dw, 'tie_correct'), 0)} | "
                f"{cells.get((dw, 'tie_wrong'), 0)} |"
            )
        lines.append("")
        lines.append(
            "- `deploy=lora,truth=base` means the gate took a bad LoRA "
            "output -- counts heavily against shipping LoRA."
        )
        lines.append(
            "- `deploy=base,truth=lora` means the gate missed a LoRA win "
            "(LoRA might be good but the formatting gate cannot recover it)."
        )
        lines.append("")

        lines.append("## 5. Strategy accuracy on public (decides shipping)")
        lines.append("")
        base_correct_total = sum(1 for r in rows if r.get("base_correct"))
        lora_correct_total = sum(1 for r in rows if r.get("lora_correct"))
        deploy_correct_total = sum(1 for r in rows if r.get("deploy_correct"))
        lines.append(
            f"- base_only_acc: {base_correct_total}/{n_total} "
            f"({_acc(base_correct_total, n_total):.2f}%)"
        )
        lines.append(
            f"- lora_only_acc: {lora_correct_total}/{n_total} "
            f"({_acc(lora_correct_total, n_total):.2f}%)"
        )
        lines.append(
            f"- deploy_acc:    {deploy_correct_total}/{n_total} "
            f"({_acc(deploy_correct_total, n_total):.2f}%)"
        )
        deploy_lora_truth_base = cells.get(("lora", "base"), 0)
        deploy_lora_truth_lora = cells.get(("lora", "lora"), 0)
        use_lora = (
            deploy_correct_total > base_correct_total
            and deploy_lora_truth_base <= deploy_lora_truth_lora
        )
        lines.append("")
        if use_lora:
            lines.append("**RECOMMENDATION: USE LoRA**")
        else:
            lines.append("**RECOMMENDATION: DO NOT USE LoRA**")
        lines.append("")
        lines.append(
            "The recommendation is intentionally driven by deploy_acc vs "
            "base_only_acc, not lora_only_acc. A LoRA the formatting gate "
            "cannot reliably select is still unsafe to ship."
        )
        lines.append("")
    else:
        lines.append("Gold answers not available -- skipping truth metrics.")
        lines.append("Deploy outputs were still produced; review them manually.")
        lines.append("")

    path.write_text("\n".join(lines))


def _write_gated_outputs(
    rows: list[dict],
    *,
    deploy_outputs_path: Path,
    submission_path: Path,
) -> None:
    with open(deploy_outputs_path, "w") as f:
        for row in rows:
            rec = {
                "id": row["id"],
                "is_mcq": row["is_mcq"],
                "response": row["deploy_response"],
                "deploy_winner": row["deploy_winner"],
                "deploy_reason": row["deploy_reason"],
                "deploy_meta": row["deploy_meta"],
            }
            f.write(json.dumps(rec) + "\n")

    with open(submission_path, "w", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["id", "response"])
        for row in rows:
            response = str(row["deploy_response"]).replace("\r\n", "\n").replace("\r", "\n")
            writer.writerow([row["id"], response])


def main() -> None:
    args = _parse_args()

    here = Path(__file__).resolve().parent
    root = _discover_project_root(here)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    input_path = resolve_input_path(args.input, root)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else root / "results" / "compare"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = build_run_stem(input_path.stem, args)
    base_outputs_path = output_dir / f"{stem}_base_outputs.jsonl"
    lora_outputs_path = output_dir / f"{stem}_lora_outputs.jsonl"
    comparison_path = output_dir / f"{stem}_comparison.jsonl"
    summary_path = output_dir / f"{stem}_summary.md"
    deploy_outputs_path = output_dir / f"{stem}_deploy_outputs.jsonl"
    submission_path = output_dir / f"{stem}_deploy_submission.csv"

    with open(input_path) as f:
        data = [json.loads(line) for line in f]
    print(f"Loaded {len(data)} questions from {input_path}")

    has_answers = any("answer" in item for item in data)
    use_judger = has_answers and not args.no_eval_comparison

    data = apply_subset_caps(
        data,
        limit_mcq=args.limit_mcq,
        limit_free=args.limit_free,
        seed=args.sample_seed,
    )

    if args.compare_mode == "sequential":
        _run_sequential(args, data, base_outputs_path, lora_outputs_path)
    else:
        _run_shared(args, data, base_outputs_path, lora_outputs_path)

    if args.lora_only:
        print(f"LoRA-only run complete: {lora_outputs_path.resolve()}")
        print(
            "Score with: python scripts/modular_pipeline/compare_runs.py "
            f"--input public --limit-mcq {args.limit_mcq} --limit-free {args.limit_free} "
            f"--sample-seed {args.sample_seed} --runs {lora_outputs_path}"
        )
        return

    print("Loading saved base/LoRA outputs...")
    base_records = _load_records_by_id(base_outputs_path)
    lora_records = _load_records_by_id(lora_outputs_path)
    print(
        f"Loaded {len(base_records)} base + {len(lora_records)} lora records; "
        f"building comparison for {len(data)} items "
        f"(judger={'on' if use_judger else 'off'})..."
    )

    rows, _judger = _build_comparison_rows(
        args,
        data,
        base_records,
        lora_records,
        use_judger=use_judger,
        judge_timeout_s=args.judge_timeout_s,
    )

    with open(comparison_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote comparison to {comparison_path.resolve()}")

    _write_summary(
        summary_path,
        rows,
        args,
        input_path=input_path,
        base_outputs_path=base_outputs_path,
        lora_outputs_path=lora_outputs_path,
    )
    print(f"Wrote summary to {summary_path.resolve()}")

    if args.gated:
        _write_gated_outputs(
            rows,
            deploy_outputs_path=deploy_outputs_path,
            submission_path=submission_path,
        )
        print(f"Wrote deploy outputs to {deploy_outputs_path.resolve()}")
        print(f"Wrote deploy submission to {submission_path.resolve()}")


if __name__ == "__main__":
    main()
