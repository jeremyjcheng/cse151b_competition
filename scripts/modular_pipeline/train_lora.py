"""Custom LoRA fine-tuning entrypoint for the modular pipeline."""

import json
import math
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)

from cli_utils import (
    apply_subset_caps,
    parse_train_args,
    resolve_input_path,
    split_train_holdout,
    write_jsonl,
)
from prompting import (
    build_adapt_train_free_user,
    build_adapt_train_mcq_user,
    build_reasoning_train_user,
)
from settings import (
    ADAPT_DEFAULT_LEARNING_RATE,
    ADAPT_DEFAULT_MAX_STEPS,
    MODEL_ID,
    REASONING_DEFAULT_LEARNING_RATE,
    REASONING_DEFAULT_MAX_STEPS,
    STAGE2_DEFAULT_HOLDOUT_FRACTION,
)
from text_processing import ensure_boxed, extract_all_boxed, extract_boxed, extract_valid_letter


def _discover_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "data").exists():
            return candidate
    return start.parent


def _load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _sample_cap(items: list[dict], max_items: int | None, seed: int) -> list[dict]:
    if max_items is None or max_items >= len(items):
        return items
    rng = random.Random(seed)
    picked = sorted(rng.sample(range(len(items)), max_items))
    return [items[i] for i in picked]


def _new_dataset_stats(name: str) -> dict:
    return {
        "name": name,
        "loaded": 0,
        "accepted": 0,
        "skipped": 0,
        "skip_reasons": Counter(),
        "schema": {},
    }


def _skip(stats: dict, reason: str) -> None:
    stats["skipped"] += 1
    stats["skip_reasons"][reason] += 1


def _finalize_stats(stats: dict, accepted: int) -> dict:
    stats["accepted"] = accepted
    stats["skip_reasons"] = dict(stats["skip_reasons"])
    return stats


def _print_dataset_schema(name: str, split: str, ds, first_row: dict | None) -> dict:
    features = getattr(ds, "features", None)
    if hasattr(features, "keys"):
        feature_keys = list(features.keys())
        feature_types = {k: str(features[k]) for k in feature_keys}
    elif isinstance(features, dict):
        feature_keys = list(features.keys())
        feature_types = {k: str(v) for k, v in features.items()}
    else:
        feature_keys = []
        feature_types = {}
    first_keys = sorted(list(first_row.keys())) if first_row else []
    schema = {
        "split": split,
        "row_count": len(ds),
        "feature_keys": feature_keys,
        "feature_types": feature_types,
        "first_row_keys": first_keys,
    }
    print(
        f"[{name}] schema split={split} rows={len(ds)} "
        f"features={feature_keys} first_row_keys={first_keys}"
    )
    return schema


def _build_strict_mcq_prompt(question: str, options: list[str]) -> str:
    labels = [chr(65 + i) for i in range(len(options))]
    opts_text = "\n".join(f"{lbl}. {opt}" for lbl, opt in zip(labels, options))
    return (
        f"Question:\n{question}\n\n"
        f"Options:\n{opts_text}\n\n"
        "Think briefly and choose the correct option. End with exactly one boxed letter."
    )


def _normalize_options(raw_options) -> list[str]:
    if not isinstance(raw_options, (list, tuple)):
        return []
    out = [str(x).strip() for x in raw_options if str(x).strip()]
    return out


def _safe_map_answer_to_letter(
    answer_value,
    *,
    options: list[str],
) -> tuple[str, str]:
    labels = [chr(65 + i) for i in range(len(options))]
    if not labels:
        return "", "missing_options"
    if answer_value is None:
        return "", "missing_answer"

    if isinstance(answer_value, bool):
        return "", "ambiguous_bool_answer"

    if isinstance(answer_value, (int, float)):
        idx = int(answer_value)
        if 0 <= idx < len(options):
            return labels[idx], ""
        return "", "answer_index_out_of_range"

    answer_text = str(answer_value).strip()
    if not answer_text:
        return "", "empty_answer"

    # Common MCQ labels like "d", "(C)", "[B]", "{A}".
    compact = answer_text.strip().strip("()[]{}").strip().upper()
    if len(compact) == 1 and compact in labels:
        return compact, ""

    if answer_text.isdigit():
        idx = int(answer_text)
        if 0 <= idx < len(options):
            return labels[idx], ""
        return "", "answer_index_out_of_range"

    letter = extract_valid_letter(answer_text, labels)
    if letter:
        return letter, ""

    exact_matches = [
        i
        for i, opt in enumerate(options)
        if answer_text.lower() == str(opt).strip().lower()
    ]
    if len(exact_matches) == 1:
        return labels[exact_matches[0]], ""
    if len(exact_matches) > 1:
        return "", "ambiguous_answer_text_duplicate_choice"

    return "", "answer_unmapped"


def _is_strict_single_boxed_letter_target(target: str, letter: str) -> bool:
    text = str(target).strip()
    boxed = extract_all_boxed(text)
    if len(boxed) != 1:
        return False
    if boxed[0].strip().upper() != letter.upper():
        return False
    return bool(re.fullmatch(rf"\\boxed\{{{re.escape(letter.upper())}\}}", text))


def _is_valid_mcq_full_trace_target(target: str, letter: str) -> bool:
    text = str(target).strip()
    if len(text) < 40:
        return False
    boxed = extract_all_boxed(text)
    if len(boxed) != 1:
        return False
    return boxed[-1].strip().upper() == letter.upper()


def _build_mcq_training_target(
    letter: str,
    *,
    mode: str,
    rationale: str = "",
) -> str:
    letter = letter.upper()
    if mode == "letter_only":
        target = f"\\boxed{{{letter}}}"
        if not _is_strict_single_boxed_letter_target(target, letter):
            raise ValueError(f"invalid letter-only MCQ target for letter={letter}")
        return target

    body = str(rationale).strip()
    if len(body) < 40:
        body = (
            "Let's solve this step by step and compare the listed options carefully.\n\n"
            "After checking the choices, the correct option letter is:"
        )
    return _enforce_single_final_boxed(body, fallback_answer=letter)


def _top_reasons(skip_reasons: dict, max_items: int = 5) -> str:
    if not skip_reasons:
        return "none"
    ranked = sorted(skip_reasons.items(), key=lambda kv: (-kv[1], kv[0]))
    return ", ".join(f"{k}={v}" for k, v in ranked[:max_items])


def _strip_boxed(text: str) -> str:
    return re.sub(r"\\boxed\s*\{[^{}]*\}", "", text)


def _enforce_single_final_boxed(response: str, fallback_answer: str = "") -> str:
    cleaned = str(response or "").strip()
    boxed_values = extract_all_boxed(cleaned)
    final_boxed = boxed_values[-1] if boxed_values else ""
    if not final_boxed:
        final_boxed = str(fallback_answer).strip()

    body = _strip_boxed(cleaned).strip()
    if final_boxed:
        if body:
            return f"{body}\n\n\\boxed{{{final_boxed}}}"
        return f"\\boxed{{{final_boxed}}}"

    ensured = ensure_boxed(cleaned)
    value = extract_boxed(ensured).strip()
    body = _strip_boxed(ensured).strip()
    if body:
        return f"{body}\n\n\\boxed{{{value}}}"
    return f"\\boxed{{{value}}}"


def _normalize_mcq_answer(item: dict) -> str:
    options = item.get("options") or []
    labels = [chr(65 + i) for i in range(len(options))]
    if not labels:
        return ""

    answer_text = str(item.get("answer", "")).strip()
    if not answer_text:
        return ""

    letter = extract_valid_letter(answer_text, labels)
    if letter:
        return letter

    answer_lower = answer_text.lower()
    for lbl, opt in zip(labels, options):
        if answer_lower == str(opt).strip().lower():
            return lbl

    if len(answer_text) == 1 and answer_text.upper() in labels:
        return answer_text.upper()
    return ""


def _normalize_free_answer(item: dict) -> str:
    answer_value = item.get("answer")
    if answer_value is None:
        return ""
    if isinstance(answer_value, (list, tuple)):
        answer_text = ", ".join(str(v).strip() for v in answer_value)
    else:
        answer_text = str(answer_value).strip()
    if not answer_text:
        return ""
    return _enforce_single_final_boxed("", fallback_answer=answer_text)


def _extract_reasoning_solution_from_row(row: dict) -> str:
    for key in (
        "response",
        "solution",
        "generated_solution",
        "output",
        "assistant",
        "completion",
        "cot",
        "rationale",
        "explanation",
    ):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _extract_final_answer_from_row(row: dict) -> str:
    for key in ("expected_answer", "final_answer", "answer", "target", "label"):
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            text = ", ".join(str(v).strip() for v in value if str(v).strip())
        else:
            text = str(value).strip()
        if text:
            return text
    return ""


def _load_metamathqa_examples(max_examples: int | None, seed: int) -> tuple[list[dict], dict]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(
            "Stage `reasoning` requires `datasets`. Install with `pip install datasets`."
        ) from exc

    ds = load_dataset("meta-math/MetaMathQA", split="train")
    stats = _new_dataset_stats("metamathqa")
    stats["schema"] = _print_dataset_schema("metamathqa", "train", ds, dict(ds[0]) if len(ds) else None)
    rows = [dict(row) for row in ds]
    stats["loaded"] = len(rows)
    rows = _sample_cap(rows, max_examples, seed)

    examples: list[dict] = []
    for row in rows:
        problem = _extract_question_from_row(row)
        solution = _extract_reasoning_solution_from_row(row)
        fallback = _extract_final_answer_from_row(row)
        if not problem or not solution:
            _skip(stats, "missing_problem_or_solution")
            continue
        target = _enforce_single_final_boxed(solution, fallback_answer=fallback)
        examples.append(
            {
                "prompt": build_reasoning_train_user(problem),
                "target": target,
                "example_type": "frq",
                "system_prompt": (
                    "You are an expert competition mathematician. "
                    "Provide concise step-by-step reasoning and finish with exactly one final \\boxed{...}."
                ),
                "source": "metamathqa",
            }
        )
    return examples, _finalize_stats(stats, len(examples))


def _load_numinamath_cot_examples(
    max_examples: int | None,
    seed: int,
) -> tuple[list[dict], dict]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(
            "Stage `reasoning` requires `datasets`. Install with `pip install datasets`."
        ) from exc

    ds = load_dataset("AI-MO/NuminaMath-CoT", split="train")
    stats = _new_dataset_stats("numinamath_cot")
    stats["schema"] = _print_dataset_schema(
        "numinamath_cot", "train", ds, dict(ds[0]) if len(ds) else None
    )
    rows = [dict(row) for row in ds]
    stats["loaded"] = len(rows)
    rows = _sample_cap(rows, max_examples, seed)

    examples: list[dict] = []
    for row in rows:
        problem = _extract_question_from_row(row)
        solution = _extract_reasoning_solution_from_row(row)
        fallback = _extract_final_answer_from_row(row)
        if not problem or not solution:
            _skip(stats, "missing_problem_or_solution")
            continue
        target = _enforce_single_final_boxed(solution, fallback_answer=fallback)
        examples.append(
            {
                "prompt": build_reasoning_train_user(problem),
                "target": target,
                "example_type": "frq",
                "system_prompt": (
                    "You are an expert competition mathematician. "
                    "Provide concise step-by-step reasoning and finish with exactly one final \\boxed{...}."
                ),
                "source": "numinamath_cot",
            }
        )
    return examples, _finalize_stats(stats, len(examples))


def _load_openmath_examples(max_examples: int | None, seed: int) -> tuple[list[dict], dict]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(
            "Stage `reasoning` requires `datasets`. Install with `pip install datasets`."
        ) from exc

    ds = load_dataset("unsloth/OpenMathReasoning-mini", split="cot")
    stats = _new_dataset_stats("openmath")
    stats["schema"] = _print_dataset_schema("openmath", "cot", ds, dict(ds[0]) if len(ds) else None)
    rows = [dict(row) for row in ds]
    stats["loaded"] = len(rows)
    rows = _sample_cap(rows, max_examples, seed)
    examples: list[dict] = []
    for row in rows:
        problem = str(row.get("problem", "")).strip()
        solution = str(row.get("generated_solution", "")).strip()
        expected = str(row.get("expected_answer", "")).strip()
        if not problem or not solution:
            _skip(stats, "missing_problem_or_solution")
            continue
        target = _enforce_single_final_boxed(solution, fallback_answer=expected)
        examples.append(
            {
                "prompt": build_reasoning_train_user(problem),
                "target": target,
                "example_type": "frq",
                "system_prompt": (
                    "You are an expert competition mathematician. "
                    "Provide concise step-by-step reasoning and finish with exactly one final \\boxed{...}."
                ),
                "source": "openmath",
            }
        )
    return examples, _finalize_stats(stats, len(examples))


def _load_hendrycks_examples(
    configs: list[str],
    max_examples: int | None,
    seed: int,
) -> tuple[list[dict], dict]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(
            "Stage `reasoning` requires `datasets`. Install with `pip install datasets`."
        ) from exc

    merged: list[dict] = []
    stats = _new_dataset_stats("hendrycks")
    first_schema_printed = False
    for cfg in configs:
        ds = load_dataset("EleutherAI/hendrycks_math", cfg, split="train")
        if not first_schema_printed:
            stats["schema"] = _print_dataset_schema(
                "hendrycks", f"{cfg}:train", ds, dict(ds[0]) if len(ds) else None
            )
            first_schema_printed = True
        for row in ds:
            merged.append(dict(row))

    stats["loaded"] = len(merged)
    merged = _sample_cap(merged, max_examples, seed)
    examples: list[dict] = []
    for row in merged:
        problem = str(row.get("problem", "")).strip()
        solution = str(row.get("solution", "")).strip()
        if not problem or not solution:
            _skip(stats, "missing_problem_or_solution")
            continue
        target = _enforce_single_final_boxed(solution)
        examples.append(
            {
                "prompt": build_reasoning_train_user(problem),
                "target": target,
                "example_type": "frq",
                "system_prompt": (
                    "You are an expert competition mathematician. "
                    "Provide concise step-by-step reasoning and finish with exactly one final \\boxed{...}."
                ),
                "source": "hendrycks",
            }
        )
    return examples, _finalize_stats(stats, len(examples))


def _extract_options_from_row(row: dict) -> list[str]:
    candidates = [
        row.get("options"),
        row.get("choices"),
        row.get("answer_choices"),
    ]
    for cand in candidates:
        opts = _normalize_options(cand)
        if opts:
            return opts

    letter_keys = [k for k in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"] if row.get(k) is not None]
    if letter_keys:
        return [str(row[k]).strip() for k in letter_keys if str(row[k]).strip()]
    return []


def _extract_question_from_row(row: dict) -> str:
    for key in ("question", "problem", "prompt", "query"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _extract_answer_from_row(row: dict):
    for key in ("answer", "Answer", "label", "correct_label", "target", "correct_answer"):
        if key in row and row.get(key) is not None:
            return row.get(key)
    return None


def _load_math_mc_examples(
    max_examples: int | None,
    seed: int,
    *,
    mcq_target_mode: str = "letter_only",
) -> tuple[list[dict], dict]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(
            "MCQ stages require `datasets`. Install with `pip install datasets`."
        ) from exc

    ds = load_dataset("XiangPan/math-mc", split="train")
    stats = _new_dataset_stats("math_mc")
    stats["schema"] = _print_dataset_schema("math_mc", "train", ds, dict(ds[0]) if len(ds) else None)
    rows = [dict(row) for row in ds]
    stats["loaded"] = len(rows)
    rows = _sample_cap(rows, max_examples, seed)

    examples: list[dict] = []
    for row in rows:
        question = _extract_question_from_row(row)
        # math-mc has both per-letter columns and a `choices` list; try both layouts.
        option_candidates: list[list[str]] = []
        letter_cols = [k for k in ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J") if row.get(k) is not None]
        if letter_cols:
            opts_from_letters = [str(row[k]).strip() for k in letter_cols if str(row[k]).strip()]
            if len(opts_from_letters) >= 2:
                option_candidates.append(opts_from_letters)
        opts_from_choices = _normalize_options(row.get("choices"))
        if len(opts_from_choices) >= 2:
            option_candidates.append(opts_from_choices)
        opts_from_fallback = _extract_options_from_row(row)
        if len(opts_from_fallback) >= 2:
            option_candidates.append(opts_from_fallback)

        # De-duplicate option candidates while preserving order.
        seen_option_tuples: set[tuple[str, ...]] = set()
        options: list[str] = []
        for cand in option_candidates:
            key = tuple(cand)
            if key in seen_option_tuples:
                continue
            seen_option_tuples.add(key)
            options = cand
            break

        # Try multiple answer fields because math-mc mixes conventions.
        answer_candidates = []
        for key in ("answer", "Answer", "label", "correct_label", "target", "correct_answer"):
            if key in row and row.get(key) is not None and str(row.get(key)).strip():
                answer_candidates.append(row.get(key))

        if not question:
            _skip(stats, "missing_question")
            continue
        if len(options) < 2:
            _skip(stats, "missing_or_short_options")
            continue
        if len(set(opt.lower() for opt in options)) != len(options):
            _skip(stats, "duplicate_options")
            continue

        letter = ""
        reason = "missing_answer"
        for answer_value in answer_candidates:
            letter, reason = _safe_map_answer_to_letter(answer_value, options=options)
            if letter:
                break
        if not letter:
            _skip(stats, reason or "unmapped_answer")
            continue

        rationale = ""
        if mcq_target_mode == "full_trace":
            try:
                idx = [chr(65 + i) for i in range(len(options))].index(letter.upper())
                rationale = (
                    "Let's work through the problem and compare each option.\n\n"
                    f"Option {letter} matches the correct answer: {options[idx]}"
                )
            except ValueError:
                rationale = ""

        try:
            target = _build_mcq_training_target(
                letter,
                mode=mcq_target_mode,
                rationale=rationale,
            )
        except ValueError:
            _skip(stats, "invalid_mcq_target_shape")
            continue
        if mcq_target_mode == "full_trace" and not _is_valid_mcq_full_trace_target(
            target, letter
        ):
            _skip(stats, "invalid_mcq_full_trace_target")
            continue
        if mcq_target_mode == "letter_only" and not _is_strict_single_boxed_letter_target(
            target, letter
        ):
            _skip(stats, "invalid_mcq_target_shape")
            continue

        examples.append(
            {
                "prompt": _build_strict_mcq_prompt(question, options),
                "target": target,
                "example_type": "mcq",
                "answer_letter": letter,
                "system_prompt": (
                    "You are solving a multiple-choice math question. "
                    "Reason step by step and end with exactly one final boxed option letter."
                ),
                "source": "math_mc",
            }
        )

    if stats["loaded"] > 0 and not examples:
        raise SystemExit(
            "math-mc schema detected but no rows could be safely mapped. "
            f"features={stats['schema'].get('feature_keys')} "
            f"first_row_keys={stats['schema'].get('first_row_keys')} "
            f"top_skip_reasons={_top_reasons(dict(stats['skip_reasons']))}"
        )
    return examples, _finalize_stats(stats, len(examples))


def _load_compmath_mcq_examples(
    max_examples: int | None,
    seed: int,
    *,
    mcq_target_mode: str = "letter_only",
) -> tuple[list[dict], dict]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(
            "MCQ stages require `datasets`. Install with `pip install datasets`."
        ) from exc

    ds = load_dataset("biancaraimondi/CompMath-MCQ", split="test")
    stats = _new_dataset_stats("compmath_mcq")
    stats["schema"] = _print_dataset_schema(
        "compmath_mcq", "test", ds, dict(ds[0]) if len(ds) else None
    )
    rows = [dict(row) for row in ds]
    stats["loaded"] = len(rows)
    rows = _sample_cap(rows, max_examples, seed)

    examples: list[dict] = []
    for row in rows:
        question = str(row.get("question", "")).strip()
        options = _normalize_options(row.get("options"))
        answer_value = row.get("correct_label", None)

        if not question:
            _skip(stats, "missing_question")
            continue
        if len(options) < 2:
            _skip(stats, "missing_or_short_options")
            continue
        if len(set(opt.lower() for opt in options)) != len(options):
            _skip(stats, "duplicate_options")
            continue

        letter, reason = _safe_map_answer_to_letter(answer_value, options=options)
        if not letter:
            _skip(stats, reason or "unmapped_answer")
            continue

        rationale = ""
        if mcq_target_mode == "full_trace":
            idx = None
            if isinstance(answer_value, int) and 0 <= answer_value < len(options):
                idx = answer_value
            elif str(answer_value).strip().isdigit():
                i = int(str(answer_value).strip())
                if 0 <= i < len(options):
                    idx = i
            if idx is not None:
                lbl = chr(65 + idx)
                rationale = (
                    "Let's analyze the problem and eliminate incorrect options.\n\n"
                    f"Option {lbl} is correct: {options[idx]}"
                )

        try:
            target = _build_mcq_training_target(
                letter,
                mode=mcq_target_mode,
                rationale=rationale,
            )
        except ValueError:
            _skip(stats, "invalid_mcq_target_shape")
            continue
        if mcq_target_mode == "full_trace" and not _is_valid_mcq_full_trace_target(
            target, letter
        ):
            _skip(stats, "invalid_mcq_full_trace_target")
            continue
        if mcq_target_mode == "letter_only" and not _is_strict_single_boxed_letter_target(
            target, letter
        ):
            _skip(stats, "invalid_mcq_target_shape")
            continue

        examples.append(
            {
                "prompt": _build_strict_mcq_prompt(question, options),
                "target": target,
                "example_type": "mcq",
                "answer_letter": letter,
                "system_prompt": (
                    "You are solving a multiple-choice math question. "
                    "Reason step by step and end with exactly one final boxed option letter."
                ),
                "source": "compmath_mcq",
            }
        )

    if stats["loaded"] > 0 and not examples:
        raise SystemExit(
            "CompMath-MCQ schema detected but no rows could be safely mapped. "
            f"features={stats['schema'].get('feature_keys')} "
            f"first_row_keys={stats['schema'].get('first_row_keys')} "
            f"top_skip_reasons={_top_reasons(dict(stats['skip_reasons']))}"
        )
    return examples, _finalize_stats(stats, len(examples))


def _load_reference_questions_by_id(root: Path) -> dict:
    out = {}
    for split_name in ("public", "private"):
        path = root / "data" / f"{split_name}.jsonl"
        if not path.exists():
            continue
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    qid = row.get("id")
                    if qid is not None:
                        out[qid] = row
        except Exception:
            continue
    return out


def _load_base_replay_examples(
    *,
    replay_path: Path,
    root: Path,
    max_examples: int | None,
    seed: int,
    mcq_target_mode: str = "letter_only",
    mcq_replay_min_tokens: int = 32,
) -> tuple[list[dict], dict]:
    if not replay_path.exists():
        raise SystemExit(f"--base-replay-path not found: {replay_path}")

    stats = _new_dataset_stats("base_replay")
    reference = _load_reference_questions_by_id(root)
    rows = _load_jsonl(replay_path)
    stats["loaded"] = len(rows)
    stats["schema"] = {
        "path": str(replay_path),
        "row_count": len(rows),
        "first_row_keys": sorted(list(rows[0].keys())) if rows else [],
    }
    if not rows:
        return [], _finalize_stats(stats, 0)
    if "response" not in rows[0]:
        raise SystemExit(
            f"Replay file schema incompatible: expected key `response`, got keys={stats['schema']['first_row_keys']}"
        )

    rows = _sample_cap(rows, max_examples, seed)
    examples: list[dict] = []
    for row in rows:
        response = str(row.get("response", "")).strip()
        meta = row.get("meta") or {}
        raw_trace = str(row.get("raw") or meta.get("raw") or "").strip()
        if not response and not raw_trace:
            _skip(stats, "missing_response")
            continue

        if any(k in row for k in ("base_correct", "deploy_correct", "is_correct", "correct")):
            is_correct = bool(
                row.get("base_correct", row.get("deploy_correct", row.get("is_correct", row.get("correct"))))
            )
            if not is_correct:
                _skip(stats, "marked_incorrect")
                continue
        else:
            boxed = extract_all_boxed(response)
            if not boxed:
                _skip(stats, "clean_filter_missing_box")
                continue
            if any(not str(x).strip() for x in boxed):
                _skip(stats, "clean_filter_empty_boxed")
                continue
            if len(boxed) > 1:
                _skip(stats, "clean_filter_multiple_boxed")
                continue
            if meta.get("malformed_output"):
                _skip(stats, "clean_filter_malformed")
                continue
            if meta.get("generation_hit_max"):
                _skip(stats, "clean_filter_truncated")
                continue
            if meta.get("guessed_letter_used"):
                _skip(stats, "clean_filter_guessed_letter")
                continue
            if meta.get("fallback_used"):
                _skip(stats, "clean_filter_fallback")
                continue

        qid = row.get("id")
        ref_item = reference.get(qid, {})
        question = str(row.get("question") or ref_item.get("question") or "").strip()
        options = row.get("options")
        if options is None:
            options = ref_item.get("options")
        options = _normalize_options(options)
        is_mcq = bool(row.get("is_mcq", bool(options)))
        if not question:
            _skip(stats, "missing_question_for_replay")
            continue

        if is_mcq:
            if len(options) < 2:
                _skip(stats, "missing_options_for_replay_mcq")
                continue
            labels = [chr(65 + i) for i in range(len(options))]
            gold_letter = _normalize_mcq_answer(ref_item) if ref_item.get("answer") else ""
            trace_for_letter = raw_trace or response
            letter = extract_valid_letter(trace_for_letter, labels)
            if not letter and gold_letter:
                letter = gold_letter
            if not letter:
                _skip(stats, "invalid_replay_mcq_letter")
                continue
            if gold_letter and letter.upper() != gold_letter.upper():
                _skip(stats, "replay_letter_mismatch_gold")
                continue

            if mcq_target_mode == "full_trace":
                n_tok = meta.get("n_tokens") or meta.get("total_n_tokens")
                if isinstance(n_tok, (int, float)) and n_tok < mcq_replay_min_tokens:
                    _skip(stats, "replay_trace_too_short")
                    continue
                if not raw_trace:
                    _skip(stats, "replay_missing_raw_trace")
                    continue
                target = _enforce_single_final_boxed(raw_trace, fallback_answer=letter)
                if not _is_valid_mcq_full_trace_target(target, letter):
                    _skip(stats, "invalid_replay_mcq_full_trace_target")
                    continue
            else:
                target = _build_mcq_training_target(letter, mode="letter_only")
                if not _is_strict_single_boxed_letter_target(target, letter):
                    _skip(stats, "invalid_replay_mcq_target_shape")
                    continue
            prompt = _build_strict_mcq_prompt(question, options)
            example_type = "mcq"
        else:
            boxed = extract_boxed(response).strip()
            if not boxed:
                _skip(stats, "invalid_replay_frq_boxed")
                continue
            target = _enforce_single_final_boxed("", fallback_answer=boxed)
            prompt = build_adapt_train_free_user(question)
            example_type = "frq"

        examples.append(
            {
                "prompt": prompt,
                "target": target,
                "example_type": example_type,
                "system_prompt": (
                    "You are solving competition math questions. "
                    "Preserve stable behavior and finish with exactly one final \\boxed{...}."
                ),
                "source": "base_replay",
            }
        )

    return examples, _finalize_stats(stats, len(examples))


def _build_adapt_examples(
    input_path: Path,
    *,
    limit_mcq: int | None,
    limit_free: int | None,
    sample_seed: int,
    train_on_full_chat: bool,
    final_answer_only: bool,
    freeze_reasoning_style: bool,
    holdout_fraction: float,
    holdout_seed: int,
    holdout_output_path: Path | None,
) -> list[dict]:
    raw_data = _load_jsonl(input_path)
    raw_data = apply_subset_caps(
        raw_data,
        limit_mcq=limit_mcq,
        limit_free=limit_free,
        seed=sample_seed,
    )
    supervised = [item for item in raw_data if item.get("answer") is not None]
    if not supervised:
        raise SystemExit("No supervised samples found. Training data must include `answer` fields.")

    if holdout_fraction > 0:
        train_supervised, holdout_supervised = split_train_holdout(
            supervised,
            holdout_fraction=holdout_fraction,
            seed=holdout_seed,
        )
        if holdout_output_path is not None:
            write_jsonl(holdout_output_path, holdout_supervised)
            print(
                f"Stage 2 holdout saved to {holdout_output_path} "
                f"({len(holdout_supervised)} items). Do not train on this file."
            )
        print(
            f"Stage 2 train/holdout split: {len(train_supervised)} train, "
            f"{len(holdout_supervised)} holdout (fraction={holdout_fraction:.2f}, seed={holdout_seed})"
        )
        print(
            "Use holdout (or a fresh public subset) for local scoring while tuning; "
            "reserve full public scoring for rare final checks before private submit."
        )
        supervised = train_supervised

    examples: list[dict] = []
    if freeze_reasoning_style:
        # Conservative Stage-2 prompt: preserve Stage-1 reasoning ability and only adapt output format.
        system_prompt = (
            "You are solving competition math questions. "
            "Keep reasoning style stable and avoid copying training-template wording. "
            "End with exactly one final \\boxed{...}."
        )
    else:
        system_prompt = (
            "You are solving competition math questions. "
            "Follow the required output format and end with exactly one final \\boxed{...}."
        )

    for item in supervised:
        if item.get("options"):
            letter = _normalize_mcq_answer(item)
            if not letter:
                continue
            target = f"\\boxed{{{letter}}}"
            if train_on_full_chat and not final_answer_only:
                target = (
                    "Compute the answer and compare to options carefully.\n"
                    f"Final answer: \\boxed{{{letter}}}"
                )
            prompt = build_adapt_train_mcq_user(item["question"], item["options"])
        else:
            target = _normalize_free_answer(item)
            if not target:
                continue
            if train_on_full_chat and not final_answer_only:
                target = (
                    "Solve the problem concisely and report only one final boxed answer.\n"
                    f"{target}"
                )
            prompt = build_adapt_train_free_user(item["question"])

        # Conservative default: Stage 2 supervises only final-answer formatting.
        if final_answer_only:
            target = _enforce_single_final_boxed("", fallback_answer=extract_boxed(target))

        examples.append(
            {
                "prompt": prompt,
                "target": _enforce_single_final_boxed(target),
                "example_type": "mcq" if item.get("options") else "frq",
                "system_prompt": system_prompt,
                "source": "competition_adapt",
            }
        )
    return examples


def _mix_examples_with_mcq_weight(
    examples: list[dict],
    *,
    mcq_weight: float,
    seed: int,
) -> list[dict]:
    if mcq_weight <= 0:
        raise SystemExit("--mcq-example-weight must be > 0.")
    if not examples or abs(mcq_weight - 1.0) < 1e-12:
        return list(examples)

    rng = random.Random(seed)
    mcq = [ex for ex in examples if ex.get("example_type") == "mcq"]
    frq = [ex for ex in examples if ex.get("example_type") != "mcq"]
    if not mcq:
        return list(examples)

    weighted_mcq: list[dict] = []
    if mcq_weight > 1.0:
        int_part = int(math.floor(mcq_weight))
        frac = mcq_weight - int_part
        weighted_mcq.extend(mcq * int_part)
        if frac > 0:
            n_extra = int(round(len(mcq) * frac))
            if n_extra > 0:
                idxs = sorted(rng.sample(range(len(mcq)), min(n_extra, len(mcq))))
                weighted_mcq.extend([mcq[i] for i in idxs])
    else:
        n_keep = int(round(len(mcq) * mcq_weight))
        if n_keep > 0:
            idxs = sorted(rng.sample(range(len(mcq)), min(n_keep, len(mcq))))
            weighted_mcq.extend([mcq[i] for i in idxs])

    mixed = frq + weighted_mcq
    rng.shuffle(mixed)
    return mixed


def _print_dataset_samples(name: str, examples: list[dict], n: int = 3) -> None:
    if not examples:
        print(f"[samples:{name}] no samples")
        return
    k = min(max(3, n), 5, len(examples))
    print(f"[samples:{name}] showing {k} examples")
    for i, ex in enumerate(examples[:k], start=1):
        print(f"--- sample {i} ({name}) ---")
        print(ex.get("prompt", "").strip())
        if ex.get("example_type") == "mcq":
            print(f"Mapped answer letter: {extract_boxed(str(ex.get('target', ''))).upper()}")
        print("Assistant target:")
        print(str(ex.get("target", "")).strip())


def _tokenize_example(
    tokenizer,
    prompt: str,
    target: str,
    system_prompt: str,
    max_seq_len: int,
) -> dict[str, torch.Tensor]:
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = prompt_text + target

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    if len(full_ids) > max_seq_len:
        n_drop = len(full_ids) - max_seq_len
        full_ids = full_ids[n_drop:]
        prompt_len = max(0, len(prompt_ids) - n_drop)
    else:
        prompt_len = len(prompt_ids)

    attention_mask = [1] * len(full_ids)
    labels = full_ids.copy()
    for i in range(min(prompt_len, len(labels))):
        labels[i] = -100

    return {
        "input_ids": torch.tensor(full_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def _collate_batch(tokenizer, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    input_ids = [x["input_ids"] for x in batch]
    attention_mask = [x["attention_mask"] for x in batch]
    labels = [x["labels"] for x in batch]
    pad_id = tokenizer.pad_token_id

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_mask,
        batch_first=True,
        padding_value=0,
    )
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _apply_stage_hparam_defaults(args) -> None:
    if args.stage in {"reasoning", "mixed_reasoning_mcq"}:
        reasoning_dataset_selected = any(
            (
                getattr(args, "include_metamathqa", False),
                getattr(args, "include_numinamath_cot", False),
                getattr(args, "include_openmath", False),
                getattr(args, "include_hendrycks", False),
            )
        )
        if not reasoning_dataset_selected:
            args.include_metamathqa = True
            args.include_numinamath_cot = True
            print(
                "No reasoning dataset flag provided; defaulting to "
                "--include-metamathqa and --include-numinamath-cot."
            )

    if args.stage == "reasoning":
        if args.learning_rate >= 2e-4:
            args.learning_rate = REASONING_DEFAULT_LEARNING_RATE
        if args.max_steps == 500:
            args.max_steps = REASONING_DEFAULT_MAX_STEPS
        if not args.train_on_full_chat:
            print("Stage `reasoning` enables --train-on-full-chat by default.")
            args.train_on_full_chat = True
        return

    if args.stage in {"mcq", "mixed_reasoning_mcq"}:
        if args.learning_rate >= 2e-4:
            args.learning_rate = ADAPT_DEFAULT_LEARNING_RATE
        if args.max_steps == 500:
            args.max_steps = REASONING_DEFAULT_MAX_STEPS if args.stage == "mixed_reasoning_mcq" else ADAPT_DEFAULT_MAX_STEPS
        if args.train_on_full_chat:
            print(
                f"Stage `{args.stage}` defaults to concise targets; "
                "disabling --train-on-full-chat to protect base behavior."
            )
            args.train_on_full_chat = False
        if getattr(args, "mcq_target_mode", "letter_only") == "full_trace":
            print(
                "MCQ target mode `full_trace`: use --include-base-replay with base JSONL `raw` "
                "for real chain-of-thought. Hub MCQ rows use short rationale stubs only."
            )
            if not args.include_base_replay:
                print(
                    "Warning: --mcq-target-mode full_trace without --include-base-replay "
                    "will not learn real thinking traces from external MCQ datasets."
                )
        return

    if args.stage == "adapt":
        if args.learning_rate >= 2e-4:
            args.learning_rate = ADAPT_DEFAULT_LEARNING_RATE
        if args.max_steps == 500:
            args.max_steps = ADAPT_DEFAULT_MAX_STEPS
        if args.stage2_holdout_fraction is None:
            args.stage2_holdout_fraction = STAGE2_DEFAULT_HOLDOUT_FRACTION
        if args.stage2_final_answer_only and args.train_on_full_chat:
            print(
                "Stage `adapt` uses --stage2-final-answer-only by default; "
                "disabling --train-on-full-chat to reduce public-label memorization."
            )
            args.train_on_full_chat = False
        if args.stage2_freeze_reasoning_style:
            print(
                "Stage `adapt` is in conservative mode: preserving Stage-1 reasoning "
                "style while learning competition answer formatting."
            )


def main() -> None:
    args = parse_train_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    _set_seed(args.seed)
    _apply_stage_hparam_defaults(args)

    here = Path(__file__).resolve().parent
    root = _discover_project_root(here)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_stats: dict[str, dict] = {}
    dataset_examples: dict[str, list[dict]] = {}
    if args.stage in {"reasoning", "mcq", "mixed_reasoning_mcq"}:
        if args.stage == "reasoning" and (args.include_math_mc or args.include_compmath_mcq):
            raise SystemExit(
                "Stage `reasoning` is FRQ-only. Use --stage mcq or --stage mixed_reasoning_mcq for MCQ datasets."
            )
        if args.stage == "mcq" and not (args.include_math_mc or args.include_compmath_mcq):
            raise SystemExit(
                "Stage `mcq` requires at least one MCQ dataset. Pass --include-math-mc and/or --include-compmath-mcq."
            )

        all_examples: list[dict] = []
        if args.include_metamathqa:
            print("Loading meta-math/MetaMathQA")
            metamath_examples, metamath_stats = _load_metamathqa_examples(
                max_examples=args.max_metamathqa_examples,
                seed=args.sample_seed,
            )
            dataset_examples["metamathqa"] = metamath_examples
            dataset_stats["metamathqa"] = metamath_stats
            all_examples.extend(metamath_examples)

        if args.include_numinamath_cot:
            print("Loading AI-MO/NuminaMath-CoT")
            numina_examples, numina_stats = _load_numinamath_cot_examples(
                max_examples=args.max_numinamath_cot_examples,
                seed=args.sample_seed,
            )
            dataset_examples["numinamath_cot"] = numina_examples
            dataset_stats["numinamath_cot"] = numina_stats
            all_examples.extend(numina_examples)

        if args.include_openmath:
            print("Loading unsloth/OpenMathReasoning-mini (split=cot)")
            openmath_examples, openmath_stats = _load_openmath_examples(
                max_examples=args.max_openmath_examples,
                seed=args.sample_seed,
            )
            dataset_examples["openmath"] = openmath_examples
            dataset_stats["openmath"] = openmath_stats
            all_examples.extend(openmath_examples)

        if args.include_hendrycks:
            print("Loading EleutherAI/hendrycks_math")
            hendrycks_examples, hendrycks_stats = _load_hendrycks_examples(
                configs=args.hendrycks_configs,
                max_examples=args.max_hendrycks_examples,
                seed=args.sample_seed,
            )
            dataset_examples["hendrycks"] = hendrycks_examples
            dataset_stats["hendrycks"] = hendrycks_stats
            all_examples.extend(hendrycks_examples)

        if args.include_math_mc:
            print("Loading XiangPan/math-mc")
            math_mc_examples, math_mc_stats = _load_math_mc_examples(
                max_examples=args.max_math_mc_examples,
                seed=args.sample_seed,
                mcq_target_mode=args.mcq_target_mode,
            )
            dataset_examples["math_mc"] = math_mc_examples
            dataset_stats["math_mc"] = math_mc_stats
            all_examples.extend(math_mc_examples)

        if args.include_compmath_mcq:
            print("Loading biancaraimondi/CompMath-MCQ")
            compmath_examples, compmath_stats = _load_compmath_mcq_examples(
                max_examples=args.max_compmath_mcq_examples,
                seed=args.sample_seed,
                mcq_target_mode=args.mcq_target_mode,
            )
            dataset_examples["compmath_mcq"] = compmath_examples
            dataset_stats["compmath_mcq"] = compmath_stats
            all_examples.extend(compmath_examples)

        if args.include_base_replay:
            if not args.base_replay_path:
                raise SystemExit("--include-base-replay requires --base-replay-path.")
            replay_path = Path(args.base_replay_path)
            if not replay_path.is_absolute():
                replay_path = root / replay_path
            replay_examples, replay_stats = _load_base_replay_examples(
                replay_path=replay_path,
                root=root,
                max_examples=args.max_base_replay_examples,
                seed=args.sample_seed,
                mcq_target_mode=args.mcq_target_mode,
                mcq_replay_min_tokens=args.mcq_replay_min_tokens,
            )
            dataset_examples["base_replay"] = replay_examples
            dataset_stats["base_replay"] = replay_stats
            all_examples.extend(replay_examples)

        if args.stage == "reasoning":
            all_examples = [ex for ex in all_examples if ex.get("example_type") == "frq"]
        elif args.stage == "mcq":
            all_examples = [ex for ex in all_examples if ex.get("example_type") == "mcq"]

        all_examples = _mix_examples_with_mcq_weight(
            all_examples,
            mcq_weight=args.mcq_example_weight,
            seed=args.sample_seed,
        )
        random.Random(args.sample_seed).shuffle(all_examples)

        if args.print_dataset_samples:
            for name, samples in dataset_examples.items():
                _print_dataset_samples(name, samples, n=3)

        if not all_examples:
            raise SystemExit(
                f"Stage `{args.stage}` produced no training samples. "
                "Check dataset flags, schema compatibility, and skip-reason logs."
            )
    else:
        input_path = resolve_input_path(args.input, root)
        if not input_path.exists():
            raise SystemExit(f"Input file not found: {input_path}")
        print(f"Loading competition adaptation data from {input_path}")
        holdout_path = output_dir / "stage2_holdout.jsonl"
        all_examples = _build_adapt_examples(
            input_path,
            limit_mcq=args.limit_mcq,
            limit_free=args.limit_free,
            sample_seed=args.sample_seed,
            train_on_full_chat=args.train_on_full_chat,
            final_answer_only=args.stage2_final_answer_only,
            freeze_reasoning_style=args.stage2_freeze_reasoning_style,
            holdout_fraction=float(args.stage2_holdout_fraction),
            holdout_seed=args.stage2_holdout_seed,
            holdout_output_path=holdout_path,
        )
        print(f"Adaptation examples: {len(all_examples)}")
        dataset_examples["competition_adapt"] = all_examples
        adapt_stats = _new_dataset_stats("competition_adapt")
        adapt_stats["loaded"] = len(all_examples)
        adapt_stats["accepted"] = len(all_examples)
        adapt_stats["skip_reasons"] = {}
        dataset_stats["competition_adapt"] = adapt_stats

    mcq_count = sum(1 for ex in all_examples if ex.get("example_type") == "mcq")
    frq_count = sum(1 for ex in all_examples if ex.get("example_type") != "mcq")
    if mcq_count and frq_count:
        adapter_strategy = "mixed"
    elif mcq_count:
        adapter_strategy = "mcq-only"
    else:
        adapter_strategy = "frq-only"

    print("=" * 60)
    print("Dataset summary")
    print("=" * 60)
    for ds_name, st in dataset_stats.items():
        print(
            f"[{ds_name}] loaded={st.get('loaded', 0)} accepted={st.get('accepted', 0)} "
            f"skipped={st.get('skipped', 0)} top_skip_reasons={_top_reasons(st.get('skip_reasons', {}))}"
        )
    print(
        f"Totals: mcq={mcq_count} frq={frq_count} final={len(all_examples)} mode={adapter_strategy}"
    )
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
    )

    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

    base_model = prepare_model_for_kbit_training(base_model)
    if args.resume_from_adapter:
        print(f"Resuming from adapter: {args.resume_from_adapter}")
        model = PeftModel.from_pretrained(
            base_model,
            args.resume_from_adapter,
            is_trainable=True,
        )
    else:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)

    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    tokenized_dataset = []
    skipped = 0
    for ex in all_examples:
        tok = _tokenize_example(
            tokenizer=tokenizer,
            prompt=ex["prompt"],
            target=ex["target"],
            system_prompt=ex["system_prompt"],
            max_seq_len=args.max_seq_len,
        )
        if torch.any(tok["labels"] != -100):
            tokenized_dataset.append(tok)
        else:
            skipped += 1
    if not tokenized_dataset:
        raise SystemExit("No valid tokenized samples after preprocessing.")
    print(f"Tokenized {len(tokenized_dataset)} samples (skipped {skipped})")

    loader = DataLoader(
        tokenized_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: _collate_batch(tokenizer, batch),
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    try:
        import bitsandbytes as bnb

        optimizer = bnb.optim.PagedAdamW8bit(
            trainable_params,
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=args.weight_decay,
        )
        print("Using bitsandbytes PagedAdamW8bit optimizer")
    except Exception:
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=args.weight_decay,
        )
        print("bitsandbytes optimizer unavailable; using torch.optim.AdamW")

    warmup_steps = int(args.max_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=args.max_steps,
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    accum_counter = 0
    running_loss = 0.0
    epoch = 0

    pbar = tqdm(total=args.max_steps, desc=f"LoRA training ({args.stage})")
    while global_step < args.max_steps:
        epoch += 1
        for batch in loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum_steps
            loss.backward()
            accum_counter += 1
            running_loss += float(outputs.loss.item())

            if accum_counter % args.grad_accum_steps != 0:
                continue

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            pbar.update(1)

            if global_step % 10 == 0:
                avg_loss = running_loss / 10.0
                lr_now = scheduler.get_last_lr()[0]
                msg = f"step={global_step} epoch={epoch} loss={avg_loss:.4f} lr={lr_now:.2e}"
                # tqdm.write + flush so nohup logs show loss (plain print gets buffered/hidden).
                pbar.write(msg)
                print(msg, flush=True)
                running_loss = 0.0

            if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                ckpt_dir = output_dir / f"checkpoint-step-{global_step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                pbar.write(f"Saved adapter checkpoint to {ckpt_dir}")
                print(f"Saved adapter checkpoint to {ckpt_dir}", flush=True)

            if global_step >= args.max_steps:
                break
    pbar.close()

    final_adapter_dir = output_dir / "final_adapter"
    final_adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)
    print(f"Saved final adapter to {final_adapter_dir}")

    if args.save_final_merged:
        merged_dir = output_dir / "merged_model"
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"Saved merged full model to {merged_dir}")

    config_payload = dict(vars(args))
    config_payload["adapter_strategy"] = adapter_strategy
    config_payload["include_base_replay"] = bool(args.include_base_replay)
    config_payload["base_replay_path"] = args.base_replay_path
    config_payload["base_replay_count"] = int(dataset_stats.get("base_replay", {}).get("accepted", 0))
    config_payload["base_replay_filter_stats"] = dataset_stats.get("base_replay", {})
    config_payload["dataset_stats"] = dataset_stats
    config_payload["final_mcq_count"] = mcq_count
    config_payload["final_frq_count"] = frq_count
    config_payload["final_total_count"] = len(all_examples)
    config_payload["resumed_from_adapter"] = bool(args.resume_from_adapter)
    config_payload["do_no_harm_eval_command"] = (
        "python scripts/modular_pipeline/compare_runs.py "
        "--input public "
        "--runs results/base_mcq_eval results/stage1_lora_mcq_eval results/stage2_mcq_lora_eval"
    )
    config_payload["recommended_next_eval_command"] = (
        "python scripts/modular_pipeline/compare_runner.py "
        "--input public --compare-mode sequential --gated "
        "--lora-adapter-path artifacts/lora_clean_v1/stage2_mcq_adapt/final_adapter"
    )

    config_path = output_dir / "train_config.json"
    with open(config_path, "w") as f:
        json.dump(config_payload, f, indent=2, sort_keys=True)
    print(f"Saved train config to {config_path}")


if __name__ == "__main__":
    main()
