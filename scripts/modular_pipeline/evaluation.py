"""Judger-based evaluation helpers."""

import importlib.util
import signal
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from formatting_diagnostics import score_output
from text_processing import extract_all_boxed, extract_valid_letter, iter_boxed_spans


def _has_exactly_one_valid_mcq_box(text: str, labels: list[str]) -> bool:
    """True when text has exactly one boxed span and it is a valid option letter."""
    if not labels:
        return False
    boxed_spans = iter_boxed_spans(text)
    if len(boxed_spans) != 1:
        return False
    inner = boxed_spans[0][2].strip()
    # Route through MCQ extraction logic on a synthetic one-box text to
    # support wrappers like \boxed{\text{A}} while avoiding phrase matching.
    letter = extract_valid_letter(f"\\boxed{{{inner}}}", labels)
    return bool(letter)


class _JudgeTimeout(Exception):
    """Raised when a single judger call exceeds the time budget."""


def _alarm_handler(signum, frame):
    del signum, frame
    raise _JudgeTimeout()


def _safe_auto_judge(judger, pred: str, gold: list, options_per_slot: list, timeout_s: float = 2.0) -> bool:
    """Run `judger.auto_judge` with a per-item timeout and safe fallback."""
    if "\\boxed" not in pred:
        # Most malformed outputs without a final boxed answer are low quality and
        # can trigger expensive parsing paths in symbolic equivalence checks.
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


def evaluate_with_judger(data: list[dict], records_by_id: dict) -> None:
    """Score predictions against gold answers using judger.Judger.auto_judge."""
    Judger = _load_project_judger()
    if Judger is None:
        print("Could not import Judger for evaluation.")
        return

    try:
        judger = Judger(strict_extract=False)
    except Exception as exc:
        print(f"Could not initialize Judger for evaluation: {exc}")
        return

    mcq_total = mcq_correct = 0
    free_total = free_correct = 0
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
    mcq_raw_was_post_truncated = 0
    mcq_training_echo = 0
    mcq_raw_letter_recovered = 0
    mcq_guessed_letter_used = 0
    mcq_avg_tokens_total = 0
    mcq_avg_tokens_count = 0
    mcq_repetition_loop_detected = 0
    mcq_boxed_count_gt1_raw = 0
    recovery_path_counts: dict[str, int] = {}
    extractor_total: dict[str, int] = {}
    extractor_correct: dict[str, int] = {}

    for item in tqdm(data, desc="Scoring with Judger"):
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

        gold = answer if isinstance(answer, list) else [answer]
        options_per_slot = [item.get("options", [])] * len(gold)

        is_mcq = bool(item.get("options"))
        labels = [chr(65 + i) for i in range(len(item.get("options", [])))] if is_mcq else []
        if is_mcq and boxed_values:
            if labels and not extract_valid_letter(pred, labels):
                malformed_invalid_mcq += 1

        ok = _safe_auto_judge(
            judger,
            pred=pred,
            gold=gold,
            options_per_slot=options_per_slot,
        )

        if is_mcq:
            mcq_total += 1
            mcq_correct += int(ok)
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
            mcq_raw_was_post_truncated += int(bool(meta.get("raw_was_post_truncated")))
            mcq_training_echo += int(bool(meta.get("training_echo_detected")))
            mcq_raw_letter_recovered += int(bool(meta.get("raw_letter_recovered")))
            mcq_guessed_letter_used += int(bool(meta.get("guessed_letter_used")))
            mcq_avg_tokens_total += int(meta.get("n_tokens") or 0)
            mcq_avg_tokens_count += 1
            mcq_boxed_count_gt1_raw += int(int(meta.get("boxed_count_in_raw") or 0) > 1)

            rp = str(meta.get("raw_letter_recovery_path") or "")
            if rp:
                recovery_path_counts[rp] = recovery_path_counts.get(rp, 0) + 1

            diag = score_output(
                raw=raw_text,
                response=pred,
                is_mcq=True,
                labels=labels,
                n_tokens=int(meta.get("n_tokens") or 0),
                pre_trunc_n_tokens=meta.get("pre_trunc_n_tokens"),
                generation_hit_max=bool(meta.get("generation_hit_max")),
            )
            mcq_repetition_loop_detected += int(
                int(diag.get("repeated_boxed_answers", 0)) >= 2
                or int(diag.get("repeated_phrase_after_box", 0)) >= 2
            )

            path = str(meta.get("extractor_path") or "unknown")
            extractor_total[path] = extractor_total.get(path, 0) + 1
            extractor_correct[path] = extractor_correct.get(path, 0) + int(ok)
        else:
            free_total += 1
            free_correct += int(ok)

        format_valid_total += int(
            bool(boxed_values)
            and len(boxed_values) == 1
            and (not is_mcq or not labels or bool(extract_valid_letter(pred, labels)))
        )

    overall_total = mcq_total + free_total
    overall_correct = mcq_correct + free_correct

    def acc(correct: int, total: int) -> float:
        return (correct / total * 100.0) if total else 0.0

    print("=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"  MCQ        : {mcq_correct:4d} / {mcq_total:4d}  ({acc(mcq_correct, mcq_total):.2f}%)")
    print(f"  Free-form  : {free_correct:4d} / {free_total:4d}  ({acc(free_correct, free_total):.2f}%)")
    print(f"  Overall    : {overall_correct:4d} / {overall_total:4d}  ({acc(overall_correct, overall_total):.2f}%)")
    print(
        f"  Format OK  : {format_valid_total:4d} / {overall_total:4d}  "
        f"({acc(format_valid_total, overall_total):.2f}%)"
    )
    print(
        f"  Malformed  : missing_box={malformed_missing_box}, "
        f"multiple_box={malformed_multi_box}, invalid_mcq_letter={malformed_invalid_mcq}"
    )
    print(
        f"  Diagnostics: raw_multi_boxed_spans={raw_multi_box}, "
        f"meta_malformed_flag={malformed_flag_meta}"
    )
    if mcq_total:
        raw_strict_rate = acc(mcq_raw_strict_valid, mcq_total)
        recovered_rate = acc(mcq_recovered_valid, mcq_total)
        recovered_gap = recovered_rate - raw_strict_rate
        print(
            "  MCQ quality : "
            f"raw_strict_valid={mcq_raw_strict_valid}/{mcq_total} ({raw_strict_rate:.2f}%), "
            f"recovered_valid={mcq_recovered_valid}/{mcq_total} ({recovered_rate:.2f}%), "
            f"recovered_raw_gap={recovered_gap:.2f} pts"
        )
        print(
            "  MCQ format  : "
            f"empty_boxed={mcq_empty_boxed}/{mcq_total} ({acc(mcq_empty_boxed, mcq_total):.2f}%), "
            f"invalid_boxed={mcq_invalid_boxed}/{mcq_total} ({acc(mcq_invalid_boxed, mcq_total):.2f}%)"
        )
        print(
            "  MCQ runtime : "
            f"generation_hit_max={mcq_generation_hit_max}/{mcq_total} "
            f"({acc(mcq_generation_hit_max, mcq_total):.2f}%), "
            f"finalizer_used={mcq_finalizer_used}/{mcq_total} ({acc(mcq_finalizer_used, mcq_total):.2f}%)"
        )
        print(
            "  MCQ diagnostics: "
            f"avg_tokens={(mcq_avg_tokens_total / mcq_avg_tokens_count) if mcq_avg_tokens_count else 0.0:.1f}, "
            f"raw_post_truncated={mcq_raw_was_post_truncated}/{mcq_total} "
            f"({acc(mcq_raw_was_post_truncated, mcq_total):.2f}%), "
            f"raw_letter_recovered={mcq_raw_letter_recovered}/{mcq_total} "
            f"({acc(mcq_raw_letter_recovered, mcq_total):.2f}%)"
        )
        print(
            "  MCQ safeguards: "
            f"guessed_letter_used={mcq_guessed_letter_used}/{mcq_total} "
            f"({acc(mcq_guessed_letter_used, mcq_total):.2f}%), "
            f"training_echo={mcq_training_echo}/{mcq_total} ({acc(mcq_training_echo, mcq_total):.2f}%), "
            f"repetition_loop_detected={mcq_repetition_loop_detected}/{mcq_total} "
            f"({acc(mcq_repetition_loop_detected, mcq_total):.2f}%)"
        )
        print(
            "  MCQ raw shape : "
            f"boxed_count_in_raw>1={mcq_boxed_count_gt1_raw}/{mcq_total} "
            f"({acc(mcq_boxed_count_gt1_raw, mcq_total):.2f}%)"
        )
    if extractor_counts:
        top_paths = sorted(extractor_counts.items(), key=lambda x: -x[1])[:12]
        paths_str = ", ".join(f"{k}={v}" for k, v in top_paths)
        print(f"  Extractor paths (top): {paths_str}")
    if recovery_path_counts:
        top_recovery = sorted(recovery_path_counts.items(), key=lambda x: -x[1])[:8]
        rec_str = ", ".join(f"{k}={v}" for k, v in top_recovery)
        print(f"  Raw letter recovery paths: {rec_str}")
    if extractor_total:
        ranked = sorted(extractor_total.items(), key=lambda x: -x[1])[:10]
        parts = []
        for path, total in ranked:
            correct = extractor_correct.get(path, 0)
            parts.append(f"{path}={correct}/{total} ({acc(correct, total):.1f}%)")
        print("  Extractor path accuracy: " + ", ".join(parts))
    print("=" * 50)
