"""
modular batched inference for Qwen3-4B-Thinking.
Saves raw model outputs for public.jsonl and creates submission.csv.
"""

from __future__ import annotations

import os
import re
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

MAX_TOKENS_MCQ = 1024
TEMP_MCQ = 0.0
TOP_P_MCQ = 1.0
TOP_K_MCQ = 0
REP_PEN_MCQ = 1.15
THINK_BUDGET_MCQ = 512

MAX_TOKENS_FREE = 2048
TEMP_FREE = 0.2
TOP_P_FREE = 0.9
TOP_K_FREE = 10
REP_PEN_FREE = 1.10
THINK_BUDGET_FREE = 1024

MCQ_BATCH_SIZE = 12
FREE_BATCH_SIZE = 2

SYSTEM_PROMPT_MCQ = (
    "You are solving a multiple-choice math exam under time pressure. "
    "Solve briefly and choose the best option. "
    "If your computed value does not exactly match an option, pick the closest match. "
    "Put the final selected letter inside \\boxed{}, e.g. \\boxed{A}. "
    "Stop once the boxed answer is written."
)

SYSTEM_PROMPT_FREE = (
    "You are an expert mathematician solving a timed exam. "
    "Solve step by step, but keep the solution concise. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside one \\boxed{}, "
    "e.g. \\boxed{3, 7}. "
    "Stop once the final boxed answer is written."
)

MCQ_FEWSHOT = (
    "Example:\n"
    "Q: What is 2+3?\n"
    "A. 4\nB. 5\nC. 6\nD. 7\n\n"
    "Reasoning: 2+3=5, so the correct option is B.\n"
    "\\boxed{B}\n\n"
    "Now solve the following problem.\n\n"
)


def build_mcq_user(question: str, options: list[str]) -> str:
    labels = [chr(65 + i) for i in range(len(options))]
    opts_text = "\n".join(
        f"{lbl}. {str(opt).strip()}" for lbl, opt in zip(labels, options)
    )

    return (
        f"{MCQ_FEWSHOT}"
        f"{question}\n\n"
        f"Options:\n{opts_text}\n\n"
        "Put your final answer as \\boxed{X}, where X is one letter from A-J."
    )


def build_free_user(question: str) -> str:
    return question


class BatchBudgetForcingProcessor(LogitsProcessor):
    def __init__(self, end_think_token_id: int, think_budget: int, input_width: int):
        self.end_think_token_id = end_think_token_id
        self.think_budget = think_budget
        self.input_width = input_width

    def __call__(self, input_ids, scores):
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


def extract_boxed(text: str) -> str:
    matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    return matches[-1].strip() if matches else ""


class ModularPipeline:
    def __init__(self, gpu_id: str = "0", model_id: str = MODEL_ID):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

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

        outputs = self._generate_batch(
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

        solved = []

        for out in outputs:
            raw = out["raw"]
            solved.append(
                {
                    "response": raw,
                    "raw": raw,
                    "meta": {
                        "is_mcq": True,
                        "n_tokens": out["n_tokens"],
                        "boxed": extract_boxed(raw),
                    },
                }
            )

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
            solved.append(
                {
                    "response": raw,
                    "raw": raw,
                    "meta": {
                        "is_mcq": False,
                        "n_tokens": out["n_tokens"],
                        "boxed": extract_boxed(raw),
                    },
                }
            )

        return solved


if __name__ == "__main__":
    _here = Path(__file__).resolve().parent
    ROOT = _here if (_here / "data" / "public.jsonl").exists() else _here.parent

    input_path = ROOT / "data" / "public.jsonl"
    output_path = ROOT / "results" / "public_outputs.jsonl"
    ordered_output_path = ROOT / "results" / "public_outputs_ordered.jsonl"
    submission_path = ROOT / "submission.csv"

    with open(input_path) as f:
        data = [json.loads(line) for line in f]

    print(f"Loaded {len(data)} questions from {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

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

    if not remaining_data:
        print("Nothing left to solve.")
    else:
        pipe = ModularPipeline(gpu_id="0")

        mcq_items = [item for item in remaining_data if item.get("options")]
        free_items = [item for item in remaining_data if not item.get("options")]

        print(f"Remaining MCQ questions: {len(mcq_items)}")
        print(f"Remaining free-form questions: {len(free_items)}")

        def write_records(f, chunk: list[dict], solved_batch: list[dict]) -> None:
            for item, solved in zip(chunk, solved_batch):
                rec = {
                    "id": item.get("id"),
                    "is_mcq": bool(item.get("options")),
                    "response": solved["raw"],
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
    else:
        with open(ordered_output_path, "w") as f:
            for item in data:
                rec = records_by_id[item.get("id")]
                f.write(json.dumps(rec) + "\n")

        print(f"Saved ordered outputs to {ordered_output_path.resolve()}")

        with open(submission_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "response"])

            for item in data:
                rec = records_by_id[item.get("id")]
                writer.writerow([rec["id"], rec["response"]])

        print(f"Saved submission CSV to {submission_path.resolve()}")