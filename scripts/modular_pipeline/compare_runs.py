"""Compare modular pipeline runs with question-level judged metrics."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from cli_utils import resolve_input_path
from evaluation import _safe_auto_judge
from text_processing import extract_all_boxed, extract_valid_letter


def _discover_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "data").exists():
            return candidate
    return start.parent


def _load_project_judger():
    judger_path = Path(__file__).resolve().parents[2] / "judger.py"
    spec = importlib.util.spec_from_file_location("project_judger", judger_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import judger from {judger_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Judger


def _load_input_data(input_path: Path) -> list[dict]:
    with open(input_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _resolve_run_output(run_path: Path) -> Path:
    if run_path.is_file():
        return run_path
    ordered = sorted(run_path.glob("*_outputs_ordered.jsonl"))
    if ordered:
        return ordered[-1]
    flat = sorted(run_path.glob("*_outputs.jsonl"))
    if flat:
        return flat[-1]
    raise FileNotFoundError(f"No output JSONL found in {run_path}")


def _load_records_by_id(output_path: Path) -> dict:
    records_by_id: dict = {}
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("id") is not None:
                records_by_id[rec["id"]] = rec
    return records_by_id


def evaluate_run_path(run_path: Path, *, input_path: Path) -> dict:
    output_path = _resolve_run_output(run_path)
    data = _load_input_data(input_path)
    records_by_id = _load_records_by_id(output_path)
    Judger = _load_project_judger()
    judger = Judger(strict_extract=False)

    mcq_total = mcq_correct = 0
    free_total = free_correct = 0
    format_ok = 0
    malformed_missing_box = 0
    malformed_multi_box = 0
    malformed_output_count = 0
    empty_boxed_count = 0
    truncation_count = 0
    guessed_letter_count = 0
    fallback_count = 0
    mcq_valid_boxed = 0
    mcq_total_with_pred = 0
    total_generated_tokens = 0
    generated_token_samples = 0
    extractor_counts: dict[str, int] = {}
    per_question: dict[int, bool] = {}

    for item in data:
        if "answer" not in item:
            continue
        qid = item.get("id")
        rec = records_by_id.get(qid)
        if rec is None:
            continue
        pred = str(rec.get("response") or "")
        meta = rec.get("meta") or {}
        n_tokens = meta.get("n_tokens")
        if isinstance(n_tokens, (int, float)):
            total_generated_tokens += int(n_tokens)
            generated_token_samples += 1
        boxed_values = extract_all_boxed(pred)
        if not boxed_values:
            malformed_missing_box += 1
        if len(boxed_values) > 1:
            malformed_multi_box += 1
        if any(not str(x).strip() for x in boxed_values):
            empty_boxed_count += 1
        if meta.get("malformed_output"):
            malformed_output_count += 1
        if meta.get("generation_hit_max"):
            truncation_count += 1
        if meta.get("guessed_letter_used"):
            guessed_letter_count += 1
        if meta.get("fallback_used"):
            fallback_count += 1

        ep = str(meta.get("extractor_path") or "")
        if ep:
            extractor_counts[ep] = extractor_counts.get(ep, 0) + 1

        gold = item["answer"] if isinstance(item["answer"], list) else [item["answer"]]
        options_per_slot = [item.get("options", [])] * len(gold)
        ok = _safe_auto_judge(judger, pred=pred, gold=gold, options_per_slot=options_per_slot)
        per_question[int(qid)] = bool(ok)

        is_mcq = bool(item.get("options"))
        labels = [chr(65 + i) for i in range(len(item.get("options", [])))] if is_mcq else []
        if is_mcq:
            mcq_total += 1
            mcq_correct += int(ok)
            mcq_total_with_pred += 1
            if len(boxed_values) == 1 and bool(extract_valid_letter(pred, labels)):
                mcq_valid_boxed += 1
        else:
            free_total += 1
            free_correct += int(ok)

        format_ok += int(
            bool(boxed_values)
            and len(boxed_values) == 1
            and (not is_mcq or not labels or bool(extract_valid_letter(pred, labels)))
        )

    total = mcq_total + free_total
    correct = mcq_correct + free_correct

    def _acc(c: int, t: int) -> float:
        return (100.0 * c / t) if t else 0.0

    return {
        "run_path": str(run_path),
        "output_path": str(output_path),
        "total": total,
        "overall_correct": correct,
        "overall_acc": _acc(correct, total),
        "mcq_total": mcq_total,
        "mcq_correct": mcq_correct,
        "mcq_acc": _acc(mcq_correct, mcq_total),
        "free_total": free_total,
        "free_correct": free_correct,
        "free_acc": _acc(free_correct, free_total),
        "format_ok_count": format_ok,
        "format_ok_acc": _acc(format_ok, total),
        "malformed_missing_box": malformed_missing_box,
        "malformed_multi_box": malformed_multi_box,
        "malformed_output_count": malformed_output_count,
        "empty_boxed_count": empty_boxed_count,
        "truncation_count": truncation_count,
        "guessed_letter_count": guessed_letter_count,
        "fallback_count": fallback_count,
        "mcq_valid_boxed_count": mcq_valid_boxed,
        "mcq_valid_boxed_rate": _acc(mcq_valid_boxed, mcq_total_with_pred),
        "avg_generated_tokens": (
            float(total_generated_tokens) / float(generated_token_samples)
            if generated_token_samples
            else 0.0
        ),
        "extractor_counts": extractor_counts,
        "per_question_correct": per_question,
    }


def _paired_delta(base: dict, candidate: dict) -> dict:
    common = sorted(set(base["per_question_correct"]) & set(candidate["per_question_correct"]))
    win = sum(
        1
        for qid in common
        if candidate["per_question_correct"][qid] and not base["per_question_correct"][qid]
    )
    loss = sum(
        1
        for qid in common
        if base["per_question_correct"][qid] and not candidate["per_question_correct"][qid]
    )
    return {"paired_n": len(common), "wins": win, "losses": loss, "net": win - loss}


def _format_summary(results: list[dict]) -> str:
    lines = []
    lines.append("# Run Comparison")
    lines.append("")
    lines.append(
        "| run | total | overall | mcq | free | mcq_valid_box | empty_boxed | malformed | truncated | guessed | fallback | avg_tokens |"
    )
    lines.append(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for res in results:
        run_name = Path(res["run_path"]).name
        lines.append(
            f"| {run_name} | {res['total']} | {res['overall_acc']:.2f}% | {res['mcq_acc']:.2f}% | "
            f"{res['free_acc']:.2f}% | {res['mcq_valid_boxed_rate']:.2f}% | "
            f"{res['empty_boxed_count']} | {res['malformed_output_count']} | "
            f"{res['truncation_count']} | {res['guessed_letter_count']} | "
            f"{res['fallback_count']} | {res['avg_generated_tokens']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def _do_no_harm_verdict(base: dict, new: dict) -> tuple[str, str]:
    n = min(base.get("total", 0), new.get("total", 0))
    allowed_delta = max(2, int(0.03 * n)) if n > 0 else 0

    acc_drop = new.get("overall_acc", 0.0) < (base.get("overall_acc", 0.0) - 0.05)
    malformed_regression = new.get("malformed_output_count", 0) > base.get("malformed_output_count", 0) + allowed_delta
    empty_regression = new.get("empty_boxed_count", 0) > base.get("empty_boxed_count", 0) + allowed_delta
    trunc_regression = new.get("truncation_count", 0) > base.get("truncation_count", 0) + allowed_delta

    if acc_drop or malformed_regression or empty_regression or trunc_regression:
        return "regressive", (
            "Accuracy or reliability metrics regressed beyond tolerance "
            f"(allowed_delta={allowed_delta})."
        )

    acc_gain = new.get("overall_acc", 0.0) > (base.get("overall_acc", 0.0) + 0.05)
    reliability_not_worse = (
        new.get("malformed_output_count", 0) <= base.get("malformed_output_count", 0) + allowed_delta
        and new.get("empty_boxed_count", 0) <= base.get("empty_boxed_count", 0) + allowed_delta
        and new.get("truncation_count", 0) <= base.get("truncation_count", 0) + allowed_delta
    )
    if acc_gain and reliability_not_worse:
        return "additive", "Accuracy improved without meaningful reliability regressions."
    return "neutral", "No material regression detected, but gains are limited."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare modular pipeline runs.")
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Run directories or output jsonl files.",
    )
    parser.add_argument(
        "--input",
        default="public",
        help="'public', 'private', or explicit .jsonl path.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional markdown output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    here = Path(__file__).resolve().parent
    root = _discover_project_root(here)
    input_path = resolve_input_path(args.input, root)

    results = [
        evaluate_run_path(Path(run).resolve(), input_path=input_path)
        for run in args.runs
    ]
    markdown = _format_summary(results)
    print(markdown, end="")

    if len(results) >= 2:
        base = results[0]
        print("## Paired Deltas vs First Run")
        for cand in results[1:]:
            delta = _paired_delta(base, cand)
            print(
                f"- {Path(cand['run_path']).name}: "
                f"wins={delta['wins']}, losses={delta['losses']}, net={delta['net']}, "
                f"paired_n={delta['paired_n']}"
            )
        final = results[-1]
        verdict, reason = _do_no_harm_verdict(base, final)
        print("")
        print("## Do-No-Harm Verdict")
        print(f"- verdict: {verdict}")
        print(f"- rationale: {reason}")

    if args.out:
        out_path = Path(args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(markdown)
        print(f"Wrote summary to {out_path}")


if __name__ == "__main__":
    main()
