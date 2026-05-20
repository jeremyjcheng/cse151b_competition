#!/usr/bin/env python3
"""Evaluation runner with JSON reports, latency tracking, and checkpoint sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from cli_utils import (
    apply_subset_caps,
    build_run_stem,
    discover_adapter_checkpoints,
    parse_eval_args,
    resolve_input_path,
)
from evaluation import compute_evaluation_metrics
from model_pipeline import ModularPipeline
from settings import FREE_BATCH_SIZE, MCQ_BATCH_SIZE


def _discover_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "data").exists():
            return candidate
    return start.parent


def _load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(ordered) - 1)
    if f == c:
        return ordered[f]
    return ordered[f] + (ordered[c] - ordered[f]) * (k - f)


def _unload_pipeline(pipe: ModularPipeline) -> None:
    if hasattr(pipe, "llm"):
        del pipe.llm
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_inference_with_latency(
    pipe: ModularPipeline,
    data: list[dict],
    *,
    save_raw_output: bool = True,
) -> tuple[dict, dict]:
    """Run batched inference and return (records_by_id, latency_summary)."""
    mcq_items = [item for item in data if item.get("options")]
    free_items = [item for item in data if not item.get("options")]

    records_by_id: dict = {}
    per_item_latency_s: list[float] = []
    batch_latencies_s: list[float] = []

    t_all_start = time.perf_counter()

    for start in tqdm(range(0, len(mcq_items), MCQ_BATCH_SIZE), desc="MCQ batches"):
        chunk = mcq_items[start : start + MCQ_BATCH_SIZE]
        t0 = time.perf_counter()
        solved_batch = pipe.solve_mcq_batch(chunk)
        dt = time.perf_counter() - t0
        batch_latencies_s.append(dt)
        per_item = dt / max(len(chunk), 1)
        for item, solved in zip(chunk, solved_batch):
            per_item_latency_s.append(per_item)
            records_by_id[item.get("id")] = {
                "id": item.get("id"),
                "is_mcq": True,
                "response": solved["response"],
                "meta": solved.get("meta") or {},
                "latency_s": per_item,
            }
            if save_raw_output:
                records_by_id[item["id"]]["raw"] = solved.get("raw")

    for start in tqdm(range(0, len(free_items), FREE_BATCH_SIZE), desc="Free-form batches"):
        chunk = free_items[start : start + FREE_BATCH_SIZE]
        t0 = time.perf_counter()
        solved_batch = pipe.solve_free_batch(chunk)
        dt = time.perf_counter() - t0
        batch_latencies_s.append(dt)
        per_item = dt / max(len(chunk), 1)
        for item, solved in zip(chunk, solved_batch):
            per_item_latency_s.append(per_item)
            records_by_id[item.get("id")] = {
                "id": item.get("id"),
                "is_mcq": False,
                "response": solved["response"],
                "meta": solved.get("meta") or {},
                "latency_s": per_item,
            }
            if save_raw_output:
                records_by_id[item["id"]]["raw"] = solved.get("raw")

    total_s = time.perf_counter() - t_all_start
    n_items = len(per_item_latency_s)

    latency_summary = {
        "total_wall_s": total_s,
        "num_items": n_items,
        "questions_per_sec": (n_items / total_s) if total_s > 0 else 0.0,
        "latency_p50_s": _percentile(per_item_latency_s, 50),
        "latency_p95_s": _percentile(per_item_latency_s, 95),
        "latency_mean_s": statistics.mean(per_item_latency_s) if per_item_latency_s else 0.0,
        "batch_latency_mean_s": statistics.mean(batch_latencies_s) if batch_latencies_s else 0.0,
        "per_item_latency_s": per_item_latency_s,
    }
    return records_by_id, latency_summary


def _write_eval_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote eval report: {path.resolve()}")


def _write_leaderboard_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote checkpoint leaderboard: {path.resolve()}")


def evaluate_adapter(
    *,
    data: list[dict],
    adapter_path: str | None,
    args: argparse.Namespace,
    split_name: str,
    checkpoint_label: str,
) -> dict:
    """Run inference + scoring for one adapter checkpoint."""
    has_answers = any("answer" in item for item in data)

    pipe = ModularPipeline(
        gpu_id=args.gpu_id,
        lora_adapter_path=adapter_path,
        vllm_quantization=args.vllm_quantization,
        vllm_load_format=args.vllm_load_format,
        enforce_eager=args.enforce_eager,
    )
    try:
        records_by_id, latency = run_inference_with_latency(
            pipe,
            data,
            save_raw_output=args.save_raw_output,
        )
    finally:
        _unload_pipeline(pipe)

    metrics: dict = {}
    if has_answers:
        metrics = compute_evaluation_metrics(data, records_by_id, verbose=True)
    else:
        print("No ground-truth answers in input; skipping judger scoring.")

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "split_name": split_name,
        "input": str(args.input),
        "checkpoint": checkpoint_label,
        "lora_adapter_path": adapter_path,
        "enforce_eager": args.enforce_eager,
        "vllm_quantization": args.vllm_quantization,
        "vllm_load_format": args.vllm_load_format,
        "latency": {
            k: v for k, v in latency.items() if k != "per_item_latency_s"
        },
        "metrics": metrics,
    }
    return report


def main() -> None:
    args = parse_eval_args()

    here = Path(__file__).resolve().parent
    root = _discover_project_root(here)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    input_path = resolve_input_path(args.input, root)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    data = _load_jsonl(input_path)
    data = apply_subset_caps(
        data,
        limit_mcq=args.limit_mcq,
        limit_free=args.limit_free,
        seed=args.sample_seed,
    )
    print(f"Loaded {len(data)} items from {input_path}")

    split_name = args.split_name or input_path.stem
    output_dir = Path(args.output_dir) if args.output_dir else root / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    reports: list[dict] = []
    leaderboard_rows: list[dict] = []

    if args.checkpoint_dir:
        ckpt_root = Path(args.checkpoint_dir).resolve()
        adapters = discover_adapter_checkpoints(ckpt_root)
        if not adapters:
            raise SystemExit(f"No adapter checkpoints found under {ckpt_root}")
        print(f"Evaluating {len(adapters)} checkpoints under {ckpt_root}")
        for adapter_path in adapters:
            label = adapter_path.name
            print("\n" + "=" * 60)
            print(f"Checkpoint: {label}")
            report = evaluate_adapter(
                data=data,
                adapter_path=str(adapter_path),
                args=args,
                split_name=split_name,
                checkpoint_label=label,
            )
            reports.append(report)
            metrics = report.get("metrics") or {}
            if metrics and not metrics.get("error"):
                leaderboard_rows.append(
                    {
                        "checkpoint": label,
                        "validation_accuracy_pct": metrics.get("validation_accuracy_pct", 0),
                        "mcq_accuracy_pct": metrics.get("mcq_accuracy_pct", 0),
                        "mcq_exact_match_pct": metrics.get("mcq_exact_match_pct", 0),
                        "format_ok_rate_pct": metrics.get("format_ok_rate_pct", 0),
                        "extraction_failures": metrics.get("extraction_failures", 0),
                        "reasoning_failures": metrics.get("reasoning_failures", 0),
                        "questions_per_sec": report.get("latency", {}).get("questions_per_sec", 0),
                        "latency_p50_s": report.get("latency", {}).get("latency_p50_s", 0),
                    }
                )
        leaderboard_rows.sort(
            key=lambda r: float(r.get("validation_accuracy_pct") or 0),
            reverse=True,
        )
        lb_path = (
            Path(args.eval_report).with_suffix(".csv")
            if args.eval_report
            else output_dir / f"eval_{split_name}_{timestamp}_leaderboard.csv"
        )
        _write_leaderboard_csv(lb_path, leaderboard_rows)
    else:
        label = Path(args.lora_adapter_path).name if args.lora_adapter_path else "base"
        report = evaluate_adapter(
            data=data,
            adapter_path=args.lora_adapter_path,
            args=args,
            split_name=split_name,
            checkpoint_label=label,
        )
        reports.append(report)

    combined = {
        "split_name": split_name,
        "input": str(input_path),
        "num_items": len(data),
        "runs": reports,
    }
    if args.eval_report:
        report_path = Path(args.eval_report)
        if report_path.suffix.lower() != ".json":
            report_path = report_path.with_suffix(".json")
    else:
        report_path = output_dir / f"eval_{split_name}_{timestamp}.json"
    _write_eval_report(report_path, combined)


if __name__ == "__main__":
    main()
