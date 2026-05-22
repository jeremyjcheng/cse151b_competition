"""Build filtered synthetic MCQ reasoning traces for Stage 2B distillation.

This script:
1) Generates multiple MCQ candidates from a teacher adapter.
2) Filters/normalizes traces with strict quality checks.
3) Writes accepted + rejected JSONL logs.
4) Emits an optional train-replay JSONL compatible with train_lora._load_base_replay_examples.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from cli_utils import apply_subset_caps, resolve_input_path
from evaluation import _safe_auto_judge
from settings import MODEL_ID
from text_processing import extract_all_boxed, extract_valid_letter, iter_boxed_spans


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


def _token_count_approx(text: str) -> int:
    return len(re.findall(r"\S+", str(text)))


def _looks_truncated(raw: str) -> bool:
    text = str(raw or "").strip()
    if not text:
        return True
    if text.endswith(("\\boxed{", "\\box", "\\", "Therefore", "However")):
        return True
    if text.count("{") > text.count("}"):
        return True
    return False


def _has_contradictory_letters(raw: str, labels: list[str]) -> bool:
    upper = str(raw or "").upper()
    if not upper:
        return False
    valid = {x.upper() for x in labels}
    hits: list[str] = []
    patterns = (
        r"\bTHE\s+ANSWER\s+IS\s+([A-Z])\b",
        r"\bANSWER\s+IS\s+([A-Z])\b",
        r"\bOPTION\s+([A-Z])\b",
        r"\bCHOICE\s+([A-Z])\b",
        r"\b([A-Z])\s+IS\s+CORRECT\b",
    )
    for pattern in patterns:
        for m in re.findall(pattern, upper):
            m = m.strip().upper()
            if m in valid:
                hits.append(m)
    return len(set(hits)) > 1


def _has_repetition_loop(raw: str) -> bool:
    text = str(raw or "").strip()
    if not text:
        return False
    # Repeated lines/paragraphs.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 10:
        half = len(lines) // 2
        if lines[:half] == lines[half : half + half]:
            return True
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paras) >= 2 and paras[-1].lower() == paras[-2].lower():
        return True

    # "answer is X" repeated many times.
    if len(re.findall(r"\b(?:answer|correct answer)\s+is\b", text, flags=re.IGNORECASE)) >= 4:
        return True

    # Numeric loops: very long single-digit runs (e.g., many zeros).
    if re.search(r"(\d)\1{30,}", text):
        return True
    return False


def _strip_all_boxed(text: str) -> str:
    return re.sub(r"\\boxed\s*\{[^{}]*\}", "", str(text or ""))


def _normalize_to_single_final_box(raw: str, letter: str) -> str:
    body = _strip_all_boxed(raw).strip()
    if body:
        return f"{body}\n\n\\boxed{{{letter}}}"
    return f"\\boxed{{{letter}}}"


def _truncate_after_final_box_for_letter(text: str, letter: str) -> str:
    spans = iter_boxed_spans(text)
    want = letter.strip().upper()
    if not spans:
        return text.strip()
    cut = None
    for _s, end, inner in spans:
        if inner.strip().upper() == want:
            cut = end
    if cut is None:
        cut = spans[-1][1]
    return text[:cut].rstrip()


def _sample_params(sample_index: int, default_temperature: float, default_top_p: float) -> tuple[float, float, int]:
    # Moderate exploration near the requested range.
    grid = [
        (max(0.0, default_temperature - 0.1), default_top_p, 40),
        (default_temperature, default_top_p, 50),
        (min(1.2, default_temperature + 0.1), default_top_p, 60),
        (default_temperature, min(1.0, default_top_p + 0.03), 50),
    ]
    return grid[sample_index % len(grid)]


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build synthetic MCQ reasoning traces for Stage 2B.")
    parser.add_argument("--input", default="public", help="'public', 'private', or explicit .jsonl path.")
    parser.add_argument("--output-path", required=True, help="Accepted synthetic MCQ traces JSONL.")
    parser.add_argument("--rejected-output-path", required=True, help="Rejected candidates JSONL.")
    parser.add_argument(
        "--all-candidates-output-path",
        default=None,
        help="Optional path for all generated candidates before filtering.",
    )
    parser.add_argument(
        "--train-replay-output-path",
        default=None,
        help="Optional train replay JSONL output compatible with train_lora base replay loader.",
    )
    parser.add_argument("--gpu-id", default="0", help="CUDA_VISIBLE_DEVICES value.")
    parser.add_argument("--lora-adapter-path", required=True, help="Teacher adapter path (Stage 1).")
    parser.add_argument("--limit-mcq", type=int, default=50, help="Number of MCQ questions for pilot.")
    parser.add_argument("--sample-seed", type=int, default=0, help="Subset/sampling seed.")
    parser.add_argument("--num-samples-per-question", type=int, default=4, help="Candidates per question.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Base temperature for candidate generation.")
    parser.add_argument("--top-p", type=float, default=0.9, dest="top_p", help="Base top-p for candidate generation.")
    parser.add_argument("--mcq-max-new-tokens", type=int, default=2048, help="MCQ max new tokens.")
    parser.add_argument("--min-raw-tokens", type=int, default=64, help="Minimum candidate token length.")
    parser.add_argument("--max-raw-tokens", type=int, default=900, help="Maximum candidate token length.")
    parser.add_argument("--keep-multiple-per-question", action="store_true", help="Keep multiple accepted traces per question.")
    parser.add_argument("--max-accepted-per-question", type=int, default=1, help="Maximum accepted traces per question.")
    parser.add_argument("--allow-finalizer", action="store_true", help="Allow finalizer-used rows (default rejected).")
    parser.add_argument("--allow-fallback", action="store_true", help="Allow fallback/guessed rows (default rejected).")
    parser.add_argument("--vllm-enforce-eager", action="store_true", help="Enable vLLM eager mode.")

    parser.add_argument(
        "--blend-synthetic-ratio",
        type=float,
        default=0.8,
        help="Target synthetic MCQ ratio for --train-replay-output-path.",
    )
    parser.add_argument(
        "--blend-frq-ratio",
        type=float,
        default=0.2,
        help="Target FRQ ratio for --train-replay-output-path.",
    )
    parser.add_argument(
        "--frq-replay-path",
        default=None,
        help="Optional FRQ replay JSONL path to mix into train replay output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from model_pipeline import ModularPipeline

    here = Path(__file__).resolve().parent
    root = _discover_project_root(here)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    input_path = resolve_input_path(args.input, root)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")
    with open(input_path, encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    data = apply_subset_caps(
        data,
        limit_mcq=args.limit_mcq,
        limit_free=0,
        seed=args.sample_seed,
    )
    data = [row for row in data if row.get("options") and row.get("answer") is not None]
    if not data:
        raise SystemExit("No supervised MCQ rows found in selected subset.")

    Judger = _load_project_judger()
    judger = Judger(strict_extract=False)

    pipe = ModularPipeline(
        gpu_id=args.gpu_id,
        lora_adapter_path=args.lora_adapter_path,
        mcq_max_new_tokens=args.mcq_max_new_tokens,
        enforce_eager=True if args.vllm_enforce_eager else None,
    )

    output_path = Path(args.output_path)
    if not output_path.is_absolute():
        output_path = root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rejected_path = Path(args.rejected_output_path)
    if not rejected_path.is_absolute():
        rejected_path = root / rejected_path
    rejected_path.parent.mkdir(parents=True, exist_ok=True)

    all_path = None
    if args.all_candidates_output_path:
        all_path = Path(args.all_candidates_output_path)
        if not all_path.is_absolute():
            all_path = root / all_path
        all_path.parent.mkdir(parents=True, exist_ok=True)

    train_replay_path = None
    if args.train_replay_output_path:
        train_replay_path = Path(args.train_replay_output_path)
        if not train_replay_path.is_absolute():
            train_replay_path = root / train_replay_path
        train_replay_path.parent.mkdir(parents=True, exist_ok=True)

    rejected_counts: Counter = Counter()
    accepted_rows: list[dict] = []
    rejected_rows: list[dict] = []
    all_rows: list[dict] = []
    accepted_by_qid: dict[int, list[dict]] = {}

    total_candidates = 0
    max_keep = max(1, int(args.max_accepted_per_question))
    if not args.keep_multiple_per_question:
        max_keep = 1

    for item in tqdm(data, desc="Synthetic MCQ distill"):
        qid = int(item.get("id"))
        labels = [chr(65 + i) for i in range(len(item.get("options") or []))]
        gold = str(item.get("answer", "")).strip()
        gold_letter = extract_valid_letter(gold, labels)
        if not gold_letter:
            # Fallback for plain "A" style labels.
            compact = gold.strip().strip("()[]{}").upper()
            gold_letter = compact if compact in labels else ""
        if not gold_letter:
            rejected_counts["gold_letter_unmapped"] += 1
            continue

        for sample_index in range(int(args.num_samples_per_question)):
            total_candidates += 1
            temperature, top_p, top_k = _sample_params(sample_index, args.temperature, args.top_p)
            sampling_seed = int(args.sample_seed) * 100_000 + qid * 97 + sample_index * 13 + 7
            solved = pipe.solve_mcq_batch(
                [item],
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                sampling_seed=sampling_seed,
            )[0]

            response = str(solved.get("response") or "")
            raw = str(solved.get("raw") or "")
            meta = solved.get("meta") or {}
            boxed_values = extract_all_boxed(raw)
            n_tokens = int(meta.get("n_tokens") or _token_count_approx(raw))
            pre_trunc_tokens = int(meta.get("pre_trunc_n_tokens") or n_tokens)
            extracted_letter = extract_valid_letter(raw, labels)
            judged_correct = bool(
                _safe_auto_judge(
                    judger,
                    pred=response,
                    gold=[item["answer"]],
                    options_per_slot=[item.get("options", [])],
                )
            )

            candidate = {
                "id": qid,
                "question": item.get("question"),
                "options": item.get("options"),
                "is_mcq": True,
                "prompt": "",
                "target": f"\\boxed{{{gold_letter}}}",
                "gold_letter": gold_letter,
                "extracted_letter": extracted_letter,
                "accepted_raw": raw,
                "raw_candidate": raw,
                "response": response,
                "sample_index": sample_index,
                "source": "stage1_synthetic_mcq_reasoning",
                "filter_passed": False,
                "filter_reason": "",
                "metadata": {
                    "n_tokens": n_tokens,
                    "pre_trunc_n_tokens": pre_trunc_tokens,
                    "generation_hit_max": bool(meta.get("generation_hit_max")),
                    "finalizer_used": bool(meta.get("finalizer_used")),
                    "fallback_used": bool(meta.get("fallback_used")),
                    "guessed_letter_used": bool(meta.get("guessed_letter_used")),
                    "malformed_output": bool(meta.get("malformed_output")),
                    "extractor_path": str(meta.get("extractor_path") or ""),
                    "boxed_count_in_raw": int(meta.get("boxed_count_in_raw") or len(boxed_values)),
                    "raw_was_post_truncated": bool(meta.get("raw_was_post_truncated", False)),
                    "raw_letter_recovered": bool(meta.get("raw_letter_recovered")),
                    "raw_letter_recovery_path": str(meta.get("raw_letter_recovery_path") or ""),
                    "judged_correct": judged_correct,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "sampling_seed": sampling_seed,
                },
            }
            all_rows.append(candidate)

            reject_reason = None
            if not judged_correct:
                reject_reason = "judger_incorrect"
            elif _looks_truncated(raw):
                reject_reason = "looks_truncated"
            elif bool(meta.get("generation_hit_max")):
                reject_reason = "generation_hit_max"
            elif bool(meta.get("malformed_output")):
                reject_reason = "malformed_output"
            elif bool(meta.get("finalizer_used")) and not args.allow_finalizer:
                reject_reason = "finalizer_used"
            elif bool(meta.get("fallback_used")) and not args.allow_fallback:
                reject_reason = "fallback_used"
            elif bool(meta.get("guessed_letter_used")) and not args.allow_fallback:
                reject_reason = "guessed_letter_used"
            elif not boxed_values:
                reject_reason = "missing_boxed_answer"
            elif any(not str(v).strip() for v in boxed_values):
                reject_reason = "empty_boxed_answer"
            elif n_tokens < int(args.min_raw_tokens):
                reject_reason = "raw_too_short"
            elif n_tokens > int(args.max_raw_tokens):
                reject_reason = "raw_too_long"
            elif _has_repetition_loop(raw):
                reject_reason = "repetition_loop"
            elif _has_contradictory_letters(raw, labels):
                reject_reason = "contradictory_letters"
            elif not extracted_letter:
                reject_reason = "invalid_mcq_letter"
            elif extracted_letter.upper() != gold_letter.upper():
                reject_reason = "mcq_letter_mismatch_gold"

            if reject_reason:
                rejected_counts[reject_reason] += 1
                candidate["filter_reason"] = reject_reason
                rejected_rows.append(candidate)
                continue

            trimmed = _truncate_after_final_box_for_letter(raw, extracted_letter)
            normalized_raw = _normalize_to_single_final_box(trimmed, extracted_letter)
            normalized_response = f"\\boxed{{{extracted_letter}}}"
            normalized_tokens = _token_count_approx(normalized_raw)
            accepted = dict(candidate)
            accepted["accepted_raw"] = normalized_raw
            accepted["raw_candidate"] = normalized_raw
            accepted["response"] = normalized_response
            accepted["extracted_letter"] = extracted_letter
            accepted["filter_passed"] = True
            accepted["filter_reason"] = ""
            accepted["metadata"] = dict(candidate["metadata"])
            accepted["metadata"]["n_tokens"] = normalized_tokens
            accepted["metadata"]["pre_trunc_n_tokens"] = normalized_tokens
            accepted["metadata"]["boxed_count_in_raw"] = len(extract_all_boxed(normalized_raw))
            accepted_by_qid.setdefault(qid, []).append(accepted)

    for qid, rows in accepted_by_qid.items():
        del qid
        rows.sort(key=lambda r: int((r.get("metadata") or {}).get("n_tokens", 10**9)))
        accepted_rows.extend(rows[:max_keep])

    with open(output_path, "w", encoding="utf-8") as f:
        for row in accepted_rows:
            f.write(json.dumps(row) + "\n")

    with open(rejected_path, "w", encoding="utf-8") as f:
        for row in rejected_rows:
            f.write(json.dumps(row) + "\n")

    if all_path is not None:
        with open(all_path, "w", encoding="utf-8") as f:
            for row in all_rows:
                f.write(json.dumps(row) + "\n")

    if train_replay_path is not None:
        syn_ratio = max(0.0, float(args.blend_synthetic_ratio))
        frq_ratio = max(0.0, float(args.blend_frq_ratio))
        total_ratio = syn_ratio + frq_ratio
        if total_ratio <= 0:
            raise SystemExit("blend ratios must sum to > 0 when --train-replay-output-path is set.")

        frq_pool: list[dict] = []
        if frq_ratio > 0:
            if not args.frq_replay_path:
                raise SystemExit("--frq-replay-path is required when --blend-frq-ratio > 0.")
            frq_path = Path(args.frq_replay_path)
            if not frq_path.is_absolute():
                frq_path = root / frq_path
            frq_rows = _load_jsonl(frq_path)
            for row in frq_rows:
                is_mcq = bool(row.get("is_mcq", bool(row.get("options"))))
                if is_mcq:
                    continue
                raw = str(row.get("raw") or (row.get("meta") or {}).get("raw") or "").strip()
                response = str(row.get("response") or "").strip()
                if not raw and not response:
                    continue
                if bool((row.get("meta") or {}).get("malformed_output")):
                    continue
                frq_pool.append(row)

        random.Random(args.sample_seed).shuffle(accepted_rows)
        random.Random(args.sample_seed + 1).shuffle(frq_pool)

        if accepted_rows:
            total_target = len(accepted_rows)
            n_syn = max(1, int(round(total_target * (syn_ratio / total_ratio))))
            n_syn = min(n_syn, len(accepted_rows))
        else:
            n_syn = 0

        if frq_ratio > 0:
            n_frq = int(round(max(1, n_syn) * (frq_ratio / max(syn_ratio, 1e-9))))
        else:
            n_frq = 0
        n_frq = min(n_frq, len(frq_pool))

        replay_rows: list[dict] = []
        for row in accepted_rows[:n_syn]:
            m = row.get("metadata") or {}
            replay_rows.append(
                {
                    "id": row.get("id"),
                    "is_mcq": True,
                    "question": row.get("question"),
                    "options": row.get("options"),
                    "answer": row.get("gold_letter"),
                    "response": row.get("response"),
                    "raw": row.get("raw_candidate"),
                    "meta": {
                        "n_tokens": int(m.get("n_tokens", _token_count_approx(str(row.get("raw_candidate") or "")))),
                        "generation_hit_max": bool(m.get("generation_hit_max", False)),
                        "malformed_output": False,
                        "fallback_used": bool(m.get("fallback_used", False)),
                        "guessed_letter_used": bool(m.get("guessed_letter_used", False)),
                        "extractor_path": str(m.get("extractor_path") or ""),
                        "source": "synthetic_mcq_reasoning",
                    },
                }
            )

        for row in frq_pool[:n_frq]:
            meta = row.get("meta") or {}
            replay_rows.append(
                {
                    "id": row.get("id"),
                    "is_mcq": False,
                    "question": row.get("question"),
                    "response": row.get("response"),
                    "raw": row.get("raw") or meta.get("raw") or row.get("response"),
                    "meta": {
                        "n_tokens": int(meta.get("n_tokens") or meta.get("total_n_tokens") or 0),
                        "generation_hit_max": bool(meta.get("generation_hit_max", False)),
                        "malformed_output": bool(meta.get("malformed_output", False)),
                        "source": "frq_replay",
                    },
                }
            )

        random.Random(args.sample_seed + 2).shuffle(replay_rows)
        with open(train_replay_path, "w", encoding="utf-8") as f:
            for row in replay_rows:
                f.write(json.dumps(row) + "\n")

    summary = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "rejected_output_path": str(rejected_path),
        "all_candidates_output_path": str(all_path) if all_path else "",
        "train_replay_output_path": str(train_replay_path) if train_replay_path else "",
        "source_model": MODEL_ID,
        "source_lora_adapter_path": args.lora_adapter_path,
        "questions_used": len(data),
        "total_candidates_generated": total_candidates,
        "accepted_count": len(accepted_rows),
        "rejected_count": len(rejected_rows),
        "acceptance_rate": (len(accepted_rows) / total_candidates) if total_candidates else 0.0,
        "rejected_counts_by_reason": dict(rejected_counts),
    }
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"Wrote accepted synthetic traces: {output_path}")
    print(f"Wrote rejected synthetic traces: {rejected_path}")
    if all_path is not None:
        print(f"Wrote all candidates: {all_path}")
    if train_replay_path is not None:
        print(f"Wrote train replay blend: {train_replay_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
