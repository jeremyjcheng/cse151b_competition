"""
Modular batched inference for Qwen3-4B-Thinking using 4-bit quantization.
Run this file directly or import ModularPipeline from scripts/CLIs.
"""

from __future__ import annotations

import os
import re
import json
from pathlib import Path
from typing import Any, Optional

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LogitsProcessor,
)

MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"

# Faster MCQ settings
MAX_TOKENS_MCQ = 2048
TEMP_MCQ = 0.0
TOP_P_MCQ = 1.0
TOP_K_MCQ = 0
REP_PEN_MCQ = 1.15
MCQ_NO_THINK = False
THINK_BUDGET_MCQ = 1024
FALLBACK_MAX_TOKENS = 16

# Free-form settings
MAX_TOKENS_FREE = 4096
TEMP_FREE = 0.4
TOP_P_FREE = 0.95
TOP_K_FREE = 20
REP_PEN_FREE = 1.05
THINK_BUDGET_FREE = 2048

MCQ_BATCH_SIZE = 16
FREE_BATCH_SIZE = 4

SYSTEM_PROMPT_MCQ = (
    "You are solving a multiple-choice math exam under time pressure. "
    "Solve briefly and choose the best option. "
    "If your computed value does not exactly match an option, pick the closest match. "
    "Return your final output as exactly one line in this format: FINAL: X "
    "where X is a single option letter from A to J. "
    "No explanation, no extra words, no punctuation after the letter."
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
    "FINAL: B\n\n"
    "Now solve the following problem.\n\n"
)


def build_mcq_user(question: str, options: list[str], no_think: bool) -> str:
    labels = [chr(65 + i) for i in range(len(options))]
    opts_text = "\n".join(
        f"{lbl}. {str(opt).strip()}" for lbl, opt in zip(labels, options)
    )
    suffix = "\n/no_think" if no_think else ""

    return (
        f"{MCQ_FEWSHOT}"
        f"{question}\n\n"
        f"Options:\n{opts_text}\n\n"
        "Output format: FINAL: X\n"
        "X must be one letter from A-J.\n"
        "Do not output any other text."
        f"{suffix}"
    )


def build_free_user(question: str) -> str:
    return question


class BatchBudgetForcingProcessor(LogitsProcessor):
    """
    Batch-safe thinking budget processor.

    It checks each row separately. Once a row has generated </think>, it stops forcing
    that row. If a row reaches the thinking budget without </think>, it forces </think>.
    """

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


def strip_thinking(text: str) -> str:
    if "</think>" in text:
        return text.rsplit("</think>", 1)[-1].strip()
    if "<think>" in text:
        return ""
    return text.strip()


def clean_visible(text: str) -> str:
    text = strip_thinking(text)
    text = re.sub(r"<\|[^|]+\|>", "", text)
    return text.strip()


def extract_mcq_letter(text: str) -> str:
    m = re.search(r"FINAL\s*:\s*([A-J])\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(r"\\boxed\{\s*([A-J])\s*\}", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(r"\b([A-J])\b", text.upper())
    return m.group(1) if m else ""


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

        # Important for decoder-only batched generation
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
            build_mcq_user(item["question"], item["options"], no_think=MCQ_NO_THINK)
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

        solved = []
        fallback_items = []
        fallback_indices = []

        for idx, (item, out) in enumerate(zip(items, primary_outputs)):
            response = clean_visible(out["raw"])
            letter = extract_mcq_letter(response)

            if letter:
                solved.append(
                    {
                        "response": f"FINAL: {letter}",
                        "raw": out["raw"],
                        "meta": {
                            "is_mcq": True,
                            "n_tokens": out["n_tokens"],
                            "fallback_used": False,
                        },
                    }
                )
            else:
                solved.append(None)
                fallback_items.append(item)
                fallback_indices.append(idx)

        # Batched fallback for MCQs that failed extraction
        if fallback_items:
            fb_system_prompts = [
                "You output a single uppercase letter from A to J. Nothing else."
            ] * len(fallback_items)

            fb_user_prompts = []
            for item in fallback_items:
                labels = [chr(65 + i) for i in range(len(item["options"]))]
                opts_text = "\n".join(
                    f"{lbl}. {str(opt).strip()}"
                    for lbl, opt in zip(labels, item["options"])
                )
                fb_user_prompts.append(
                    f"{item['question']}\n\nOptions:\n{opts_text}\n\nAnswer:\n/no_think"
                )

            fb_outputs = self._generate_batch(
                fb_system_prompts,
                fb_user_prompts,
                max_new_tokens=FALLBACK_MAX_TOKENS,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                repetition_penalty=1.0,
                do_sample=False,
                think_budget=THINK_BUDGET_MCQ,
            )

            for original_idx, fb_out in zip(fallback_indices, fb_outputs):
                fb_resp = clean_visible(fb_out["raw"])
                letter = extract_mcq_letter(fb_resp)

                primary_tokens = primary_outputs[original_idx]["n_tokens"]
                final_response = f"FINAL: {letter}" if letter else fb_resp

                solved[original_idx] = {
                    "response": final_response,
                    "raw": primary_outputs[original_idx]["raw"],
                    "meta": {
                        "is_mcq": True,
                        "n_tokens": primary_tokens + fb_out["n_tokens"],
                        "fallback_used": True,
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
            response = clean_visible(out["raw"])
            solved.append(
                {
                    "response": response,
                    "raw": out["raw"],
                    "meta": {
                        "is_mcq": False,
                        "n_tokens": out["n_tokens"],
                        "boxed": extract_boxed(response),
                    },
                }
            )

        return solved

    def solve_item(self, item: dict) -> dict:
        if item.get("options"):
            return self.solve_mcq_batch([item])[0]
        return self.solve_free_batch([item])[0]

    def solve_batch(self, items: list[dict]) -> list[dict]:
        """
        True batching.

        This splits MCQ and free-form items because they use different decoding settings,
        then restores the original order.
        """
        mcq_pairs = [(i, item) for i, item in enumerate(items) if item.get("options")]
        free_pairs = [(i, item) for i, item in enumerate(items) if not item.get("options")]

        solved = [None] * len(items)

        if mcq_pairs:
            mcq_indices, mcq_items = zip(*mcq_pairs)
            mcq_solved = self.solve_mcq_batch(list(mcq_items))
            for idx, ans in zip(mcq_indices, mcq_solved):
                solved[idx] = ans

        if free_pairs:
            free_indices, free_items = zip(*free_pairs)
            free_solved = self.solve_free_batch(list(free_items))
            for idx, ans in zip(free_indices, free_solved):
                solved[idx] = ans

        return solved


if __name__ == "__main__":
    _here = Path(__file__).resolve().parent
    ROOT = _here if (_here / "data" / "public.jsonl").exists() else _here.parent

    input_path = ROOT / "data" / "public.jsonl"
    output_path = ROOT / "results" / "modular_outputs.jsonl"

    with open(input_path) as f:
        data = [json.loads(line) for line in f]

    print(f"Loaded {len(data)} questions from {input_path}")

    pipe = ModularPipeline(gpu_id="0")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mcq_items = [item for item in data if item.get("options")]
    free_items = [item for item in data if not item.get("options")]

    responses_by_id = {}

    with open(output_path, "w") as f:
        # MCQ batches
        for start in tqdm(
            range(0, len(mcq_items), MCQ_BATCH_SIZE),
            desc="Solving MCQ batches",
        ):
            chunk = mcq_items[start:start + MCQ_BATCH_SIZE]
            solved_batch = pipe.solve_mcq_batch(chunk)

            for item, solved in zip(chunk, solved_batch):
                rec = {
                    "id": item.get("id"),
                    "response": solved["response"],
                    "meta": solved["meta"],
                }
                responses_by_id[item.get("id")] = rec

        # Free-form batches
        for start in tqdm(
            range(0, len(free_items), FREE_BATCH_SIZE),
            desc="Solving free-form batches",
        ):
            chunk = free_items[start:start + FREE_BATCH_SIZE]
            solved_batch = pipe.solve_free_batch(chunk)

            for item, solved in zip(chunk, solved_batch):
                rec = {
                    "id": item.get("id"),
                    "response": solved["response"],
                    "meta": solved["meta"],
                }
                responses_by_id[item.get("id")] = rec

        # Save in original order
        for item in data:
            rec = responses_by_id[item.get("id")]
            f.write(json.dumps(rec) + "\n")

        f.flush()

    print(f"Saved {len(data)} outputs to {output_path.resolve()}")