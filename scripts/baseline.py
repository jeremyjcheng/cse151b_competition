"""
Baseline file
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
DATA_PATH = "cse151b_competition/data/public.jsonl"
OUTPUT_PATH = "results/baseline_results.jsonl"
MAX_TOKENS = 2048
BATCH_SIZE = 8
GPU_ID = "0"

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}."
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)


def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for a question."""
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(
            f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options)
        )
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    return SYSTEM_PROMPT_MATH, question


# ---------------------------------------------------------------------------
# Answer extraction / scoring 
# ---------------------------------------------------------------------------
def extract_letter(text: str) -> str:
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


def score_mcq(response: str, gold_letter: str) -> bool:
    return extract_letter(response) == gold_letter.strip().upper()


def load_completed(output_path):
    completed = {}
    path = Path(output_path)

    if not path.exists():
        return completed

    with open(path) as f:
        for line in f:
            try:
                row = json.loads(line)
                completed[row["id"]] = row
            except Exception:
                continue

    return completed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="CSE 151B baseline inference")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--gpu", type=str, default=GPU_ID)
    parser.add_argument("--data", type=str, default=DATA_PATH)
    parser.add_argument("--output", type=str, default=OUTPUT_PATH)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # -- Load dataset -------------------------------------------------------
    data = [json.loads(line) for line in open(args.data)]
    n_mcq = sum(bool(d.get("options")) for d in data)
    n_free = sum(not d.get("options") for d in data)
    print(f"Loaded {len(data)} questions  ({n_mcq} MCQ, {n_free} free-form)")

    # -- Load model ---------------------------------------------------------
    print(f"Loading model {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
    )
    print("Model loaded.")

    # -- Batched inference --------------------------------------------------
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    completed = load_completed(out_path)
    print(f"Found {len(completed)} completed results. Resuming...")

    remaining_data = [item for item in data if item.get("id") not in completed]

    responses = []
    total_batches = (len(remaining_data) + args.batch_size - 1) // args.batch_size
    start = time.time()

    pbar = tqdm(total=len(data), initial=len(completed), desc="Inference", unit="q")

    for batch_idx in range(total_batches):
        lo = batch_idx * args.batch_size
        hi = min(lo + args.batch_size, len(remaining_data))
        batch = remaining_data[lo:hi]

        prompts = []
        for item in batch:
            system, user = build_prompt(item["question"], item.get("options"))
            prompt_text = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt_text)

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=16384,
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                repetition_penalty=1.0,
                do_sample=True,
            )

        input_len = inputs["input_ids"].shape[1]
        batch_records = []

        for item, out in zip(batch, output_ids):
            new_tokens = out[input_len:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            record = {
                "id": item.get("id"),
                "is_mcq": bool(item.get("options")),
                "gold": item.get("answer"),
                "response": response,
                "correct": None,
            }

            batch_records.append(record)
            responses.append(response)

        with open(out_path, "a") as f:
            for record in batch_records:
                f.write(json.dumps(record) + "\n")
                f.flush()

        pbar.update(len(batch))
        elapsed = time.time() - start
        pbar.set_postfix(
            elapsed=f"{elapsed / 60:.1f}m",
            rate=f"{len(responses) / elapsed:.1f}q/s" if elapsed > 0 else "N/A",
        )

    elapsed = time.time() - start
    pbar.close()
    print(f"\nInference complete. {len(responses)} responses in {elapsed / 60:.1f} min.")

    # -- Scoring ------------------------------------------------------------
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "cse151b_competition"))
    from judger import Judger

    judger = Judger(strict_extract=False)

    merged = load_completed(out_path)
    results = []
    for item in tqdm(data, total=len(data), desc="Scoring"):
        rid = item.get("id")
        rec = merged.get(rid)
        if rec is None:
            continue
        response = rec["response"]
        is_mcq = bool(item.get("options"))
        gold = item["answer"]

        if is_mcq:
            correct = score_mcq(response, str(gold))
        else:
            gold_list = gold if isinstance(gold, list) else [gold]
            try:
                correct = judger.auto_judge(
                    pred=response,
                    gold=gold_list,
                    options=[[]] * len(gold_list),
                )
            except Exception:
                correct = False

        results.append(
            {
                "id": item.get("id"),
                "is_mcq": is_mcq,
                "gold": gold,
                "response": response,
                "correct": correct,
            }
        )

    # -- Summary ------------------------------------------------------------
    mcq_res = [r for r in results if r["is_mcq"]]
    free_res = [r for r in results if not r["is_mcq"]]

    def acc(subset):
        return sum(r["correct"] for r in subset) / len(subset) * 100 if subset else 0.0

    print("=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(
        f"  MCQ        : {sum(r['correct'] for r in mcq_res):4d} / {len(mcq_res):4d}  ({acc(mcq_res):.2f}%)"
    )
    print(
        f"  Free-form  : {sum(r['correct'] for r in free_res):4d} / {len(free_res):4d}  ({acc(free_res):.2f}%)"
    )
    print(
        f"  Overall    : {sum(r['correct'] for r in results):4d} / {len(results):4d}  ({acc(results):.2f}%)"
    )
    print("=" * 50)

    # Results are appended during inference (correct=None until scored separately).
    # Do not write out_path with mode "w" here — it would overwrite resumed JSONL.


if __name__ == "__main__":
    main()
