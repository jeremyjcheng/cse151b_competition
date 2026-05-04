"""
Modular batched inference for Qwen3-4B-Thinking.

Dual-mode runner:
- `--input private` (default) runs on data/private.jsonl (the leaderboard test
  set with no ground-truth answers) and emits a submission CSV.
- `--input public` runs on data/public.jsonl (training set with answers) and,
  in addition to the submission CSV, scores responses with judger.Judger and
  prints MCQ / free-form / overall accuracy.
- `--input <path>` accepts an arbitrary .jsonl following the same schema.

Outputs are written under results/ and namespaced by input stem so public and
private runs do not collide.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import csv
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LogitsProcessor,
)

MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"

# MCQ settings
MAX_TOKENS_MCQ = 1024
THINK_BUDGET_MCQ = 512
MAX_TOKENS_MCQ_FINAL = 256
TEMP_MCQ = 0.0
TOP_P_MCQ = 1.0
TOP_K_MCQ = 0
REP_PEN_MCQ = 1.10

# Free-form settings
MAX_TOKENS_FREE = 768
TEMP_FREE = 0.1
TOP_P_FREE = 0.9
TOP_K_FREE = 10
REP_PEN_FREE = 1.10
THINK_BUDGET_FREE = 384

MCQ_BATCH_SIZE = 8
FREE_BATCH_SIZE = 2

SYSTEM_PROMPT_MCQ = (
    "Solve the multiple-choice math problem. "
    "Compute the result first. "
    "Compare your result to each option. "
    "Choose the closest matching option. "
    "Output exactly one letter from the listed valid choices; never output "
    "the option text itself. "
    "End with exactly one final answer in the form \\boxed{X}."
)

SYSTEM_PROMPT_FREE = (
    "You are an expert mathematician solving a timed exam. "
    "Solve step by step, but keep the solution concise. "
    "End with exactly one final \\boxed{...}. Do not box intermediate sub-answers. "
    "If the question contains multiple [ANS] slots, output exactly that many "
    "values, in order and comma-separated, inside that single \\boxed{...}, "
    "e.g. \\boxed{3, 7}. "
    "Stop once the final boxed answer is written."
)

MCQ_FEWSHOT = (
    "Example:\n"
    "Q: What is 2+3?\n"
    "A. 4\nB. 5\nC. 6\nD. 7\n\n"
    "Compute: 2+3=5.\n"
    "Compare: option B is 5.\n"
    "\\boxed{B}\n\n"
)


def build_mcq_user(question: str, options: list[str]) -> str:
    labels = [chr(65 + i) for i in range(len(options))]
    valid_letters = ", ".join(labels)
    opts_text = "\n".join(
        f"{lbl}. {str(opt).strip()}" for lbl, opt in zip(labels, options)
    )

    return (
        f"{MCQ_FEWSHOT}"
        f"Q: {question}\n\n"
        f"Options:\n{opts_text}\n\n"
        f"Valid choices: [{valid_letters}].\n"
        "Compute the answer first, compare it to the options, "
        "then output the final answer as \\boxed{X}."
    )


def count_ans_slots(question: str) -> int:
    return question.count("[ANS]")


def build_free_user(question: str) -> str:
    n_slots = count_ans_slots(question)
    if n_slots >= 2:
        return (
            f"{question}\n\n"
            f"This question has {n_slots} answer slots. "
            f"Output exactly {n_slots} values, in order and comma-separated, "
            f"inside one final \\boxed{{...}}."
        )
    return question


class BatchBudgetForcingProcessor(LogitsProcessor):
    def __init__(self, end_think_token_id: int, think_budget: int, input_width: int):
        self.end_think_token_id = end_think_token_id
        self.think_budget = think_budget
        self.input_width = input_width

    def __call__(self, input_ids, scores):
        if self.think_budget <= 0:
            return scores

        n_generated = input_ids.shape[1] - self.input_width

        if n_generated < self.think_budget:
            return scores

        gen_region = input_ids[:, self.input_width:]

        for row_idx in range(input_ids.shape[0]):
            row_tokens = gen_region[row_idx].tolist()
            already_ended_think = self.end_think_token_id in row_tokens

            if not already_ended_think:
                scores[row_idx, :] = float("-inf")
                scores[row_idx, self.end_think_token_id] = 0.0

        return scores


def extract_all_boxed(text: str) -> list[str]:
    """Brace-balanced extraction of every \\boxed{...} occurrence.

    Unlike a regex with [^{}], this correctly handles nested braces such as
    \\boxed{\\frac{1}{2}}.
    """
    out: list[str] = []
    i = 0
    while True:
        idx = text.find("\\boxed{", i)
        if idx < 0:
            break
        brace_start = idx + len("\\boxed{")
        depth = 1
        j = brace_start
        while j < len(text) and depth > 0:
            ch = text[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            j += 1
        if depth == 0:
            out.append(text[brace_start:j - 1].strip())
            i = j
        else:
            break
    return out


def extract_boxed(text: str) -> str:
    matches = extract_all_boxed(text)
    return matches[-1] if matches else ""


def clean_special_tokens(text: str) -> str:
    text = text.replace("<|im_end|>", "")
    text = text.replace("<|endoftext|>", "")
    return text.strip()


def extract_valid_letter(text: str, labels: list[str]) -> str:
    valid_set = set(labels)
    upper = text.upper()

    boxed = extract_boxed(upper).strip().upper()
    if boxed in valid_set:
        return boxed

    patterns = [
        r"\\BOXED\{\s*([A-Z])\s*\}",
        r"OPTION\s+([A-Z])",
        r"CHOICE\s+([A-Z])",
        r"ANSWER\s+IS\s+([A-Z])",
        r"FINAL\s+ANSWER\s+IS\s+([A-Z])",
        r"CORRESPONDS\s+TO\s+OPTION\s+([A-Z])",
        r"MATCH(?:ES)?\s+OPTION\s+([A-Z])",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, upper)
        for m in reversed(matches):
            if m in valid_set:
                return m

    return ""


_LATEX_BLOCK_PATTERNS = [
    re.compile(r"\\\((.+?)\\\)", re.DOTALL),
    re.compile(r"\\\[(.+?)\\\]", re.DOTALL),
    re.compile(r"\$\$(.+?)\$\$", re.DOTALL),
    re.compile(r"\$(.+?)\$", re.DOTALL),
]


def _last_latex_block(text: str) -> str:
    for pat in _LATEX_BLOCK_PATTERNS:
        matches = pat.findall(text)
        if matches:
            return matches[-1].strip()
    return ""


def _last_answer_phrase(text: str) -> str:
    for pat in (
        r"(?:final\s+)?answer\s+is[:\s]+([^\n\.]+)",
        r"(?:therefore|thus|so|hence)[,]?\s+([^\n\.]+)",
        r"=\s*([^\n=]+?)\s*$",
    ):
        matches = re.findall(pat, text, flags=re.IGNORECASE | re.MULTILINE)
        if matches:
            value = matches[-1].strip().strip(".,:; \t")
            if value:
                return value

    nums = re.findall(r"-?\d+(?:\.\d+)?(?:/-?\d+(?:\.\d+)?)?", text)
    return nums[-1] if nums else ""


def ensure_boxed(response: str) -> str:
    """Guarantee the response contains a final \\boxed{...} for the judger.

    If the model omitted one, append a best-effort fallback derived from the
    last LaTeX block or the last "answer is X" / "= X" phrase in the visible
    (post-</think>) portion of the response.
    """
    if extract_all_boxed(response):
        return response

    visible = response
    think_end = visible.rfind("</think>")
    if think_end >= 0:
        visible = visible[think_end + len("</think>"):].strip()
    if not visible:
        visible = response

    fallback = _last_latex_block(visible) or _last_answer_phrase(visible)
    if fallback:
        return response.rstrip() + f"\n\n\\boxed{{{fallback}}}"
    return response.rstrip() + "\n\n\\boxed{}"


class ModularPipeline:
    def __init__(self, gpu_id: str = "0", model_id: str = MODEL_ID):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=False,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "left"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        self.llm = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map="auto",
        )

        eot_ids = self.tokenizer.encode("</think>", add_special_tokens=False)
        if len(eot_ids) == 1:
            self.end_think_token_id = eot_ids[0]
        else:
            self.end_think_token_id = self.tokenizer.convert_tokens_to_ids("</think>")

        # Lazy-loaded judger used for MCQ option-text matching only.
        # Sentinel False marks an attempted-and-failed import so we don't retry.
        self._judger = None

    def _get_judger(self):
        if self._judger is False:
            return None
        if self._judger is None:
            try:
                from judger import Judger
                self._judger = Judger(strict_extract=False)
            except Exception as exc:
                print(f"Warning: option-text Judger unavailable ({exc}); skipping that fallback.")
                self._judger = False
                return None
        return self._judger

    def _option_match_letter(
        self,
        primary_raw: str,
        options: list[str],
        labels: list[str],
    ) -> str:
        """Last-ditch MCQ fallback: extract a candidate answer from the model's
        reasoning and pick the option whose text is mathematically equal to it.
        """
        judger = self._get_judger()
        if judger is None:
            return ""

        try:
            pred = judger.extract_ans(primary_raw)
        except Exception:
            pred = extract_boxed(primary_raw)

        if not pred:
            return ""

        for letter, option_text in zip(labels, options):
            try:
                if judger.is_equal(pred, str(option_text).strip()):
                    return letter
            except Exception:
                continue

        return ""

    def _make_chat(self, system_prompt: str, user_prompt: str) -> str:
        return self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _generate_batch(
        self,
        system_prompts: list[str],
        user_prompts: list[str],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        do_sample: bool,
        think_budget: int,
    ) -> list[dict]:
        chats = [
            self._make_chat(system, user)
            for system, user in zip(system_prompts, user_prompts)
        ]

        inputs = self.tokenizer(
            chats,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=16384,
        ).to(self.llm.device)

        input_width = inputs["input_ids"].shape[1]

        logits_processors = [
            BatchBudgetForcingProcessor(
                end_think_token_id=self.end_think_token_id,
                think_budget=think_budget,
                input_width=input_width,
            )
        ]

        gen_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": repetition_penalty,
            "logits_processor": logits_processors,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if do_sample:
            gen_kwargs.update(
                {
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                }
            )

        with torch.no_grad():
            output_ids = self.llm.generate(**gen_kwargs)

        results = []

        for i in range(output_ids.shape[0]):
            new_tokens = output_ids[i, input_width:]
            raw = self.tokenizer.decode(new_tokens, skip_special_tokens=False).strip()
            raw = clean_special_tokens(raw)

            results.append(
                {
                    "raw": raw,
                    "n_tokens": int(new_tokens.shape[0]),
                }
            )

        return results

    def solve_mcq_batch(self, items: list[dict]) -> list[dict]:
        if not items:
            return []

        user_prompts = [
            build_mcq_user(item["question"], item["options"])
            for item in items
        ]
        system_prompts = [SYSTEM_PROMPT_MCQ] * len(items)

        primary_outputs = self._generate_batch(
            system_prompts,
            user_prompts,
            max_new_tokens=MAX_TOKENS_MCQ,
            temperature=TEMP_MCQ,
            top_p=TOP_P_MCQ,
            top_k=TOP_K_MCQ,
            repetition_penalty=REP_PEN_MCQ,
            do_sample=False,
            think_budget=THINK_BUDGET_MCQ,
        )

        solved = [None] * len(items)
        finalizer_items = []
        finalizer_indices = []
        primary_raws = []

        for idx, (item, out) in enumerate(zip(items, primary_outputs)):
            raw = out["raw"]
            labels = [chr(65 + i) for i in range(len(item["options"]))]
            letter = extract_valid_letter(raw, labels)

            primary_raws.append(raw)

            if letter:
                if extract_boxed(raw).strip().upper() != letter:
                    raw = raw.rstrip() + f"\n\n\\boxed{{{letter}}}"

                solved[idx] = {
                    "response": raw,
                    "raw": raw,
                    "meta": {
                        "is_mcq": True,
                        "n_tokens": out["n_tokens"],
                        "boxed": letter,
                        "finalizer_used": False,
                        "option_match_used": False,
                    },
                }
            else:
                finalizer_items.append(item)
                finalizer_indices.append(idx)

        if finalizer_items:
            finalizer_system_prompts = [
                "You are selecting the final answer for a multiple-choice problem. "
                "Use the previous reasoning and the options. "
                "Output ONLY one valid letter inside \\boxed{}, nothing else."
            ] * len(finalizer_items)

            finalizer_user_prompts = []

            for item, original_idx in zip(finalizer_items, finalizer_indices):
                labels = [chr(65 + i) for i in range(len(item["options"]))]
                valid_letters = ", ".join(labels)
                opts_text = "\n".join(
                    f"{lbl}. {str(opt).strip()}"
                    for lbl, opt in zip(labels, item["options"])
                )

                finalizer_user_prompts.append(
                    f"Question:\n{item['question']}\n\n"
                    f"Options:\n{opts_text}\n\n"
                    f"Previous reasoning:\n{primary_raws[original_idx]}\n\n"
                    f"Valid choices: [{valid_letters}].\n"
                    "Choose the option that best matches the reasoning. "
                    "Output ONLY \\boxed{X}."
                )

            finalizer_outputs = self._generate_batch(
                finalizer_system_prompts,
                finalizer_user_prompts,
                max_new_tokens=MAX_TOKENS_MCQ_FINAL,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                repetition_penalty=1.0,
                do_sample=False,
                think_budget=0,
            )

            for original_idx, fout in zip(finalizer_indices, finalizer_outputs):
                item = items[original_idx]
                labels = [chr(65 + i) for i in range(len(item["options"]))]
                letter = extract_valid_letter(fout["raw"], labels)
                option_match_used = False

                if not letter:
                    matched = self._option_match_letter(
                        primary_raws[original_idx],
                        item["options"],
                        labels,
                    )
                    if matched:
                        letter = matched
                        option_match_used = True

                if not letter:
                    # Last resort. With both finalizer and option-text match
                    # exhausted this should be rare.
                    letter = labels[0]

                raw = primary_raws[original_idx].rstrip() + f"\n\n\\boxed{{{letter}}}"

                solved[original_idx] = {
                    "response": raw,
                    "raw": raw,
                    "meta": {
                        "is_mcq": True,
                        "n_tokens": (
                            primary_outputs[original_idx]["n_tokens"]
                            + fout["n_tokens"]
                        ),
                        "boxed": letter,
                        "finalizer_used": True,
                        "option_match_used": option_match_used,
                    },
                }

        return solved

    def solve_free_batch(self, items: list[dict]) -> list[dict]:
        if not items:
            return []

        user_prompts = [build_free_user(item["question"]) for item in items]
        system_prompts = [SYSTEM_PROMPT_FREE] * len(items)

        outputs = self._generate_batch(
            system_prompts,
            user_prompts,
            max_new_tokens=MAX_TOKENS_FREE,
            temperature=TEMP_FREE,
            top_p=TOP_P_FREE,
            top_k=TOP_K_FREE,
            repetition_penalty=REP_PEN_FREE,
            do_sample=True,
            think_budget=THINK_BUDGET_FREE,
        )

        solved = []

        for out in outputs:
            raw = out["raw"]
            response = ensure_boxed(raw)

            solved.append(
                {
                    "response": response,
                    "raw": raw,
                    "meta": {
                        "is_mcq": False,
                        "n_tokens": out["n_tokens"],
                        "boxed": extract_boxed(response),
                        "boxed_fallback_used": response != raw,
                    },
                }
            )

        return solved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Modular batched inference for Qwen3-4B-Thinking.",
    )
    parser.add_argument(
        "--input",
        default="private",
        help=("'private' (default), 'public', or a path to a .jsonl file. "
              "'private' targets the leaderboard test set; 'public' enables "
              "judger-based local accuracy reporting."),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output files. Default: <project root>/results",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip judger evaluation even if the input has 'answer' fields.",
    )
    parser.add_argument(
        "--gpu-id",
        default="0",
        help="CUDA_VISIBLE_DEVICES value passed through to the pipeline.",
    )
    parser.add_argument(
        "--limit-mcq",
        type=int,
        default=None,
        help=("Cap the number of MCQ items processed (random subset, fixed by "
              "--sample-seed). Default: no cap."),
    )
    parser.add_argument(
        "--limit-free",
        type=int,
        default=None,
        help=("Cap the number of free-form items processed (random subset, fixed "
              "by --sample-seed). Default: no cap."),
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="Seed for the random subset selection used by --limit-mcq / --limit-free.",
    )
    return parser.parse_args()


def resolve_input_path(arg_value: str, root: Path) -> Path:
    if arg_value == "private":
        return root / "data" / "private.jsonl"
    if arg_value == "public":
        return root / "data" / "public.jsonl"

    p = Path(arg_value)
    return p if p.is_absolute() else (root / p)


def apply_subset_caps(
    data: list[dict],
    *,
    limit_mcq: int | None,
    limit_free: int | None,
    seed: int,
) -> list[dict]:
    """Return a deterministic per-type-capped subset of `data`.

    Sampling is seeded so re-running with the same seed produces the same
    subset. Within each type the relative input order is preserved so the
    submission CSV still follows the original `id` ordering.
    """
    if limit_mcq is None and limit_free is None:
        return data

    import random as _random
    rng = _random.Random(seed)

    mcq_indices = [i for i, item in enumerate(data) if item.get("options")]
    free_indices = [i for i, item in enumerate(data) if not item.get("options")]

    if limit_mcq is not None and limit_mcq < len(mcq_indices):
        mcq_pick = sorted(rng.sample(mcq_indices, limit_mcq))
    else:
        mcq_pick = mcq_indices

    if limit_free is not None and limit_free < len(free_indices):
        free_pick = sorted(rng.sample(free_indices, limit_free))
    else:
        free_pick = free_indices

    keep = sorted(set(mcq_pick) | set(free_pick))
    selected = [data[i] for i in keep]

    print(
        f"Subset: {len(mcq_pick)}/{len(mcq_indices)} MCQ + "
        f"{len(free_pick)}/{len(free_indices)} free-form "
        f"(seed={seed}) -> {len(selected)} items"
    )
    return selected


def build_run_stem(input_stem: str, args: argparse.Namespace) -> str:
    """Append a `_mcq{N}_free{N}_seed{S}` suffix when caps are active.

    Sampled runs therefore write to different files than the full pass and
    cannot accidentally pollute its resume state or submission CSV.
    """
    parts: list[str] = []
    if args.limit_mcq is not None:
        parts.append(f"mcq{args.limit_mcq}")
    if args.limit_free is not None:
        parts.append(f"free{args.limit_free}")
    if not parts:
        return input_stem
    parts.append(f"seed{args.sample_seed}")
    return f"{input_stem}_{'_'.join(parts)}"


def evaluate_with_judger(data: list[dict], records_by_id: dict) -> None:
    """Score predictions against gold answers using judger.Judger.auto_judge.

    Skipped silently per-item if the item has no 'answer' field. Prints MCQ /
    free-form / overall accuracy.
    """
    try:
        from judger import Judger
    except Exception as exc:
        print(f"Could not import Judger for evaluation: {exc}")
        return

    j = Judger(strict_extract=False)

    mcq_total = mcq_correct = 0
    free_total = free_correct = 0

    for item in tqdm(data, desc="Scoring with Judger"):
        ans = item.get("answer")
        if ans is None:
            continue

        rec = records_by_id.get(item.get("id"))
        if rec is None:
            continue

        pred = rec.get("response", "")
        gold = ans if isinstance(ans, list) else [ans]
        options_per_slot = [item.get("options", [])] * len(gold)

        try:
            ok = bool(j.auto_judge(pred=pred, gold=gold, options=options_per_slot))
        except Exception:
            ok = False

        if item.get("options"):
            mcq_total += 1
            mcq_correct += int(ok)
        else:
            free_total += 1
            free_correct += int(ok)

    overall_total = mcq_total + free_total
    overall_correct = mcq_correct + free_correct

    def acc(c: int, t: int) -> float:
        return (c / t * 100.0) if t else 0.0

    print("=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(
        f"  MCQ        : {mcq_correct:4d} / {mcq_total:4d}  "
        f"({acc(mcq_correct, mcq_total):.2f}%)"
    )
    print(
        f"  Free-form  : {free_correct:4d} / {free_total:4d}  "
        f"({acc(free_correct, free_total):.2f}%)"
    )
    print(
        f"  Overall    : {overall_correct:4d} / {overall_total:4d}  "
        f"({acc(overall_correct, overall_total):.2f}%)"
    )
    print("=" * 50)


if __name__ == "__main__":
    args = parse_args()

    _here = Path(__file__).resolve().parent
    ROOT = _here if (_here / "data").exists() else _here.parent

    # Make `from judger import Judger` resolvable from the project root.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    input_path = resolve_input_path(args.input, ROOT)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = build_run_stem(input_path.stem, args)
    output_path = output_dir / f"{stem}_outputs.jsonl"
    ordered_output_path = output_dir / f"{stem}_outputs_ordered.jsonl"
    submission_path = output_dir / f"{stem}_submission.csv"

    with open(input_path) as f:
        data = [json.loads(line) for line in f]

    print(f"Loaded {len(data)} questions from {input_path}")
    has_answers = any("answer" in item for item in data)

    data = apply_subset_caps(
        data,
        limit_mcq=args.limit_mcq,
        limit_free=args.limit_free,
        seed=args.sample_seed,
    )

    done_ids = set()
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

    print(f"Found {len(done_ids)} completed records in {output_path}")

    remaining_data = [item for item in data if item.get("id") not in done_ids]
    print(f"Remaining questions to solve: {len(remaining_data)}")

    if remaining_data:
        pipe = ModularPipeline(gpu_id=args.gpu_id)

        mcq_items = [item for item in remaining_data if item.get("options")]
        free_items = [item for item in remaining_data if not item.get("options")]

        print(f"Remaining MCQ questions: {len(mcq_items)}")
        print(f"Remaining free-form questions: {len(free_items)}")

        def write_records(f, chunk: list[dict], solved_batch: list[dict]) -> None:
            for item, solved in zip(chunk, solved_batch):
                rec = {
                    "id": item.get("id"),
                    "is_mcq": bool(item.get("options")),
                    "response": solved["response"],
                    "meta": solved["meta"],
                }
                f.write(json.dumps(rec) + "\n")

            f.flush()
            os.fsync(f.fileno())

        with open(output_path, "a") as f:
            for start in tqdm(
                range(0, len(mcq_items), MCQ_BATCH_SIZE),
                desc="Solving MCQ batches",
            ):
                chunk = mcq_items[start:start + MCQ_BATCH_SIZE]
                solved_batch = pipe.solve_mcq_batch(chunk)
                write_records(f, chunk, solved_batch)

            for start in tqdm(
                range(0, len(free_items), FREE_BATCH_SIZE),
                desc="Solving free-form batches",
            ):
                chunk = free_items[start:start + FREE_BATCH_SIZE]
                solved_batch = pipe.solve_free_batch(chunk)
                write_records(f, chunk, solved_batch)

        print(f"Saved incremental outputs to {output_path.resolve()}")
    else:
        print("Nothing left to solve.")

    records_by_id = {}

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

    missing = [item.get("id") for item in data if item.get("id") not in records_by_id]

    if missing:
        print(f"Run incomplete: {len(missing)} questions still missing.")
        print("Rerun this script and it will resume.")
        raise SystemExit(1)

    with open(ordered_output_path, "w") as f:
        for item in data:
            rec = records_by_id[item.get("id")]
            f.write(json.dumps(rec) + "\n")

    print(f"Saved ordered outputs to {ordered_output_path.resolve()}")

    # CSV format mirrors data/sample_submission.csv exactly: header is
    # `id,response`, records are separated by \r\n, inner newlines stay as \n
    # only, and quoting is minimal (only fields with commas/quotes/newlines
    # are wrapped). We strip stray \r characters from the model output so a
    # cell never contains a \r\n that would conflict with the record
    # terminator.
    with open(submission_path, "w", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["id", "response"])

        for item in data:
            rec = records_by_id[item.get("id")]
            response = str(rec.get("response", "")).replace("\r\n", "\n").replace("\r", "\n")
            writer.writerow([rec["id"], response])

    print(f"Saved submission CSV to {submission_path.resolve()}")

    if has_answers and not args.no_eval:
        evaluate_with_judger(data, records_by_id)
