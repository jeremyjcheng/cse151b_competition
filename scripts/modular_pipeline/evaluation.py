"""Judger-based evaluation helpers."""

from __future__ import annotations

import importlib.util
import signal
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm

from text_processing import (
    extract_all_boxed,
    extract_submission,
    extract_valid_letter,
    iter_boxed_spans,
)


def _has_exactly_one_valid_mcq_box(text: str, labels: list[str]) -> bool:
    """True when text has exactly one boxed span and it is a valid option letter."""
    if not labels:
        return False
    boxed_spans = iter_boxed_spans(text)
    if len(boxed_spans) != 1:
        return False
    inner = boxed_spans[0][2].strip()
    letter = extract_valid_letter(f"\\boxed{{{inner}}}", labels)
    return bool(letter)


def _normalize_gold_list(answer) -> list:
    if isinstance(answer, list):
        return answer
    return [answer]


def _mcq_exact_match(pred: str, gold, labels: list[str]) -> bool:
    if not labels:
        return False
    pred_letter = extract_valid_letter(pred, labels)
    if not pred_letter:
        return False
    gold_list = _normalize_gold_list(gold)
    for g in gold_list:
        g_letter = extract_valid_letter(str(g), labels) or str(g).strip().upper()
        if pred_letter == g_letter:
            return True
    return False


def _free_exact_match(pred: str, gold) -> bool:
    from text_processing import _normalize_free_inner

    pred_boxed = extract_all_boxed(pred)
    if not pred_boxed:
        return False
    pred_val = _normalize_free_inner(pred_boxed[-1].strip())
    gold_list = _normalize_gold_list(gold)
    return any(_normalize_free_inner(str(g).strip()) == pred_val for g in gold_list)


class _JudgeTimeout(Exception):
    """Raised when a single judger call exceeds the time budget."""


def _alarm_handler(signum, frame):
    del signum, frame
    raise _JudgeTimeout()


def _safe_auto_judge(
    judger,
    pred: str,
    gold: list,
    options_per_slot: list,
    timeout_s: float = 2.0,
) -> bool:
    """Run `judger.auto_judge` with a per-item timeout and safe fallback."""
    if "\\boxed" not in pred:
        return False

    previous_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.setitimer(signal.ITIMER_REAL, timeout_s)
        return bool(judger.auto_judge(pred=pred, gold=gold, options=options_per_slot))
    except (_JudgeTimeout, Exception):
        return False
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _load_project_judger() -> Optional[type]:
    """Load `Judger` from this project's `judger.py` explicitly by path."""
    judger_path = Path(__file__).resolve().parents[2] / "judger.py"
    if not judger_path.exists():
        print(f"Could not locate project judger at: {judger_path}")
        return None

    try:
        spec = importlib.util.spec_from_file_location("project_judger", judger_path)
        if spec is None or spec.loader is None:
            print(f"Could not create import spec for: {judger_path}")
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, "Judger", None)
    except Exception as exc:
        print(f"Could not import project judger from {judger_path}: {exc}")
        return None


def _is_format_valid(pred: str, *, is_mcq: bool, labels: list[str]) -> bool:
    boxed_values = extract_all_boxed(pred)
    if not boxed_values or len(boxed_values) != 1:
        return False
    if is_mcq and labels:
        return bool(extract_valid_letter(pred, labels))
    return bool(boxed_values[0].strip())


def compute_evaluation_metrics(
    data: list[dict],
    records_by_id: dict,
    *,
    verbose: bool = True,
) -> dict[str, Any]:
    """Score predictions and return a metrics dictionary."""
    Judger = _load_project_judger()
    if Judger is None:
        return {"error": "judger_unavailable"}

    try:
        judger = Judger(strict_extract=False)
    except Exception as exc:
        return {"error": f"judger_init_failed: {exc}"}

    mcq_total = mcq_correct = mcq_em_correct = 0
    free_total = free_correct = free_em_correct = 0
    malformed_missing_box = 0
    malformed_multi_box = 0
    malformed_invalid_mcq = 0
    format_valid_total = 0
    raw_multi_box = 0
    malformed_flag_meta = 0
    extractor_counts: dict[str, int] = {}
    mcq_raw_strict_valid = 0
    mcq_recovered_valid = 0
    mcq_empty_boxed = 0
    mcq_invalid_boxed = 0
    mcq_generation_hit_max = 0
    mcq_finalizer_used = 0
    reasoning_failures = 0
    extraction_failures = 0

    for item in tqdm(data, desc="Scoring with Judger", disable=not verbose):
        answer = item.get("answer")
        if answer is None:
            continue

        rec = records_by_id.get(item.get("id"))
        if rec is None:
            continue

        pred = rec.get("response", "")
        meta = rec.get("meta") or {}
        boxed_values = extract_all_boxed(pred)
        if not boxed_values:
            malformed_missing_box += 1
        if len(boxed_values) > 1:
            malformed_multi_box += 1

        n_in_raw = meta.get("boxed_count_in_raw")
        if n_in_raw is None:
            raw_text = rec.get("raw") or ""
            n_in_raw = len(extract_all_boxed(raw_text))
        if n_in_raw > 1:
            raw_multi_box += 1

        if meta.get("malformed_output"):
            malformed_flag_meta += 1

        ep = str(meta.get("extractor_path") or "")
        if ep:
            extractor_counts[ep] = extractor_counts.get(ep, 0) + 1

        gold = _normalize_gold_list(answer)
        options_per_slot = [item.get("options", [])] * len(gold)

        is_mcq = bool(item.get("options"))
        labels = [chr(65 + i) for i in range(len(item.get("options", [])))] if is_mcq else []
        if is_mcq and boxed_values:
            if labels and not extract_valid_letter(pred, labels):
                malformed_invalid_mcq += 1

        format_ok = _is_format_valid(pred, is_mcq=is_mcq, labels=labels)
        format_valid_total += int(format_ok)

        ok = _safe_auto_judge(
            judger,
            pred=pred,
            gold=gold,
            options_per_slot=options_per_slot,
        )

        if format_ok and not ok:
            reasoning_failures += 1
        if not format_ok:
            extraction_failures += 1

        if is_mcq:
            mcq_total += 1
            mcq_correct += int(ok)
            mcq_em_correct += int(_mcq_exact_match(pred, gold, labels))
            raw_text = str(rec.get("raw") or meta.get("raw") or "")
            final_has_valid_letter = bool(labels and extract_valid_letter(pred, labels))
            raw_is_strict_valid = _has_exactly_one_valid_mcq_box(raw_text, labels)
            final_is_strict_valid = _has_exactly_one_valid_mcq_box(pred, labels)

            mcq_recovered_valid += int(final_has_valid_letter)
            mcq_raw_strict_valid += int(raw_is_strict_valid)
            mcq_generation_hit_max += int(bool(meta.get("generation_hit_max")))
            mcq_finalizer_used += int(bool(meta.get("finalizer_used")))
            mcq_empty_boxed += int(any(not str(v).strip() for v in boxed_values))
            mcq_invalid_boxed += int(bool(boxed_values) and not final_is_strict_valid)
        else:
            free_total += 1
            free_correct += int(ok)
            free_em_correct += int(_free_exact_match(pred, gold))

    overall_total = mcq_total + free_total
    overall_correct = mcq_correct + free_correct

    def acc(correct: int, total: int) -> float:
        return (correct / total * 100.0) if total else 0.0

    metrics: dict[str, Any] = {
        "mcq_total": mcq_total,
        "mcq_correct": mcq_correct,
        "mcq_accuracy_pct": acc(mcq_correct, mcq_total),
        "mcq_exact_match": mcq_em_correct,
        "mcq_exact_match_pct": acc(mcq_em_correct, mcq_total),
        "free_total": free_total,
        "free_correct": free_correct,
        "free_accuracy_pct": acc(free_correct, free_total),
        "free_exact_match": free_em_correct,
        "free_exact_match_pct": acc(free_em_correct, free_total),
        "overall_total": overall_total,
        "overall_correct": overall_correct,
        "validation_accuracy_pct": acc(overall_correct, overall_total),
        "format_valid_total": format_valid_total,
        "format_ok_rate_pct": acc(format_valid_total, overall_total),
        "malformed_missing_box": malformed_missing_box,
        "malformed_multi_box": malformed_multi_box,
        "malformed_invalid_mcq": malformed_invalid_mcq,
        "raw_multi_box": raw_multi_box,
        "malformed_flag_meta": malformed_flag_meta,
        "reasoning_failures": reasoning_failures,
        "extraction_failures": extraction_failures,
        "extractor_path_counts": extractor_counts,
        "mcq_raw_strict_valid": mcq_raw_strict_valid,
        "mcq_recovered_valid": mcq_recovered_valid,
        "mcq_empty_boxed": mcq_empty_boxed,
        "mcq_invalid_boxed": mcq_invalid_boxed,
        "mcq_generation_hit_max": mcq_generation_hit_max,
        "mcq_finalizer_used": mcq_finalizer_used,
    }

    if verbose:
        _print_metrics(metrics)

    return metrics


def _print_metrics(metrics: dict[str, Any]) -> None:
    if metrics.get("error"):
        print(f"Evaluation error: {metrics['error']}")
        return

    mcq_total = metrics["mcq_total"]
    mcq_correct = metrics["mcq_correct"]
    free_total = metrics["free_total"]
    free_correct = metrics["free_correct"]
    overall_total = metrics["overall_total"]
    overall_correct = metrics["overall_correct"]
    format_valid_total = metrics["format_valid_total"]

    def acc(correct: int, total: int) -> float:
        return (correct / total * 100.0) if total else 0.0

    print("=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"  MCQ        : {mcq_correct:4d} / {mcq_total:4d}  ({metrics['mcq_accuracy_pct']:.2f}%)")
    print(
        f"  MCQ EM     : {metrics['mcq_exact_match']:4d} / {mcq_total:4d}  "
        f"({metrics['mcq_exact_match_pct']:.2f}%)"
    )
    print(f"  Free-form  : {free_correct:4d} / {free_total:4d}  ({metrics['free_accuracy_pct']:.2f}%)")
    print(
        f"  Free EM    : {metrics['free_exact_match']:4d} / {free_total:4d}  "
        f"({metrics['free_exact_match_pct']:.2f}%)"
    )
    print(
        f"  Overall    : {overall_correct:4d} / {overall_total:4d}  "
        f"({metrics['validation_accuracy_pct']:.2f}%)"
    )
    print(
        f"  Format OK  : {format_valid_total:4d} / {overall_total:4d}  "
        f"({metrics['format_ok_rate_pct']:.2f}%)"
    )
    print(
        f"  Failures   : reasoning={metrics['reasoning_failures']}, "
        f"extraction={metrics['extraction_failures']}"
    )
    print(
        f"  Malformed  : missing_box={metrics['malformed_missing_box']}, "
        f"multiple_box={metrics['malformed_multi_box']}, "
        f"invalid_mcq_letter={metrics['malformed_invalid_mcq']}"
    )
    print(
        f"  Diagnostics: raw_multi_boxed_spans={metrics['raw_multi_box']}, "
        f"meta_malformed_flag={metrics['malformed_flag_meta']}"
    )
    if mcq_total:
        raw_strict_rate = acc(metrics["mcq_raw_strict_valid"], mcq_total)
        recovered_rate = acc(metrics["mcq_recovered_valid"], mcq_total)
        recovered_gap = recovered_rate - raw_strict_rate
        print(
            "  MCQ quality : "
            f"raw_strict_valid={metrics['mcq_raw_strict_valid']}/{mcq_total} ({raw_strict_rate:.2f}%), "
            f"recovered_valid={metrics['mcq_recovered_valid']}/{mcq_total} ({recovered_rate:.2f}%), "
            f"recovered_raw_gap={recovered_gap:.2f} pts"
        )
        print(
            "  MCQ format  : "
            f"empty_boxed={metrics['mcq_empty_boxed']}/{mcq_total} "
            f"({acc(metrics['mcq_empty_boxed'], mcq_total):.2f}%), "
            f"invalid_boxed={metrics['mcq_invalid_boxed']}/{mcq_total} "
            f"({acc(metrics['mcq_invalid_boxed'], mcq_total):.2f}%)"
        )
        print(
            "  MCQ runtime : "
            f"generation_hit_max={metrics['mcq_generation_hit_max']}/{mcq_total} "
            f"({acc(metrics['mcq_generation_hit_max'], mcq_total):.2f}%), "
            f"finalizer_used={metrics['mcq_finalizer_used']}/{mcq_total} "
            f"({acc(metrics['mcq_finalizer_used'], mcq_total):.2f}%)"
        )
    extractor_counts = metrics.get("extractor_path_counts") or {}
    if extractor_counts:
        top_paths = sorted(extractor_counts.items(), key=lambda x: -x[1])[:12]
        paths_str = ", ".join(f"{k}={v}" for k, v in top_paths)
        print(f"  Extractor paths (top): {paths_str}")
    print("=" * 50)


def is_format_valid(pred: str, *, is_mcq: bool, labels: list[str]) -> bool:
    """Public helper: True when pred has exactly one valid boxed answer."""
    return _is_format_valid(pred, is_mcq=is_mcq, labels=labels)


def evaluate_with_judger(data: list[dict], records_by_id: dict) -> dict[str, Any]:
    """Score predictions against gold answers using judger.Judger.auto_judge."""
    return compute_evaluation_metrics(data, records_by_id, verbose=True)
