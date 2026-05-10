"""Model loading and batched solving logic."""

import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LogitsProcessor

from prompting import build_free_user, build_mcq_user
from settings import (
    FINAL_ANSWER_CUE_WINDOW_CHARS,
    MAX_TOKENS_FREE,
    MAX_TOKENS_MCQ,
    MAX_TOKENS_MCQ_FINAL,
    MIN_TOKENS_BEFORE_BOXED_STOP,
    MODEL_ID,
    NO_REPEAT_NGRAM_SIZE_FREE,
    NO_REPEAT_NGRAM_SIZE_MCQ,
    NO_REPEAT_NGRAM_SIZE_MCQ_FINAL,
    POST_BOX_PATIENCE_TOKENS_FREE,
    POST_BOX_PATIENCE_TOKENS_MCQ,
    REP_PEN_FREE,
    REP_PEN_MCQ,
    REP_PEN_MCQ_FINAL,
    SYSTEM_PROMPT_FREE,
    SYSTEM_PROMPT_MCQ,
    TEMP_FREE,
    TEMP_MCQ,
    THINK_BUDGET_FREE,
    THINK_BUDGET_MCQ,
    TOP_K_FREE,
    TOP_K_MCQ,
    TOP_P_FREE,
    TOP_P_MCQ,
)
from text_processing import (
    canonicalize_free_response_with_meta,
    clean_special_tokens,
    extract_all_boxed,
    extract_boxed,
    extract_first_valid_letter,
    iter_boxed_spans,
    mcq_canonical_response,
)

_FINAL_CUE_RE = re.compile(
    r"(?:final\s+answer|therefore|thus|hence)\b",
    re.IGNORECASE,
)


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

        gen_region = input_ids[:, self.input_width :]
        for row_idx in range(input_ids.shape[0]):
            row_tokens = gen_region[row_idx].tolist()
            if self.end_think_token_id not in row_tokens:
                scores[row_idx, :] = float("-inf")
                scores[row_idx, self.end_think_token_id] = 0.0
        return scores


class SmartBoxedStopProcessor(LogitsProcessor):
    """Force EOS after a likely-final \\boxed{{...}}, not the first intermediate span."""

    def __init__(
        self,
        tokenizer,
        input_width: int,
        eos_token_id: int,
        *,
        min_tokens_before_stop: int,
        post_box_patience_tokens: int,
        cue_window_chars: int,
        mcq_valid_per_row: list[set[str]] | None,
    ):
        self.tokenizer = tokenizer
        self.input_width = input_width
        self.eos_token_id = eos_token_id
        self.min_tokens_before_stop = min_tokens_before_stop
        self.post_box_patience_tokens = post_box_patience_tokens
        self.cue_window_chars = cue_window_chars
        self.mcq_valid_per_row = mcq_valid_per_row

    def _should_force_eos_free(self, partial_clean: str, spans: list[tuple[int, int, str]]) -> bool:
        start, end, _value = spans[-1]
        after = partial_clean[end:]
        after_ids = self.tokenizer.encode(after, add_special_tokens=False)
        if len(after_ids) > self.post_box_patience_tokens:
            return True
        if not after.strip():
            return True
        ctx_start = max(0, start - self.cue_window_chars)
        context = partial_clean[ctx_start:start]
        if _FINAL_CUE_RE.search(context):
            return True
        return False

    def _should_force_eos_mcq(self, row_idx: int, partial_clean: str, spans: list[tuple[int, int, str]]) -> bool:
        valid_set = self.mcq_valid_per_row[row_idx] if self.mcq_valid_per_row else set()

        first_valid_end: int | None = None
        for start, end, val in spans:
            inner = val.strip().upper()
            if inner in valid_set:
                first_valid_end = end
                break

        if first_valid_end is not None:
            after = partial_clean[first_valid_end:]
            after_ids = self.tokenizer.encode(after, add_special_tokens=False)
            if len(after_ids) > self.post_box_patience_tokens:
                return True
            if not after.strip():
                return True
            if self.post_box_patience_tokens == 0 and len(after_ids) > 0:
                return True
            return False

        start, end, _value = spans[-1]
        after = partial_clean[end:]
        after_ids = self.tokenizer.encode(after, add_special_tokens=False)
        if len(after_ids) > self.post_box_patience_tokens:
            return True
        if not after.strip():
            return True
        ctx_start = max(0, start - self.cue_window_chars)
        context = partial_clean[ctx_start:start]
        if _FINAL_CUE_RE.search(context):
            return True
        return False

    def __call__(self, input_ids, scores):
        for row_idx in range(input_ids.shape[0]):
            row_tokens = input_ids[row_idx, self.input_width :]
            if row_tokens.numel() == 0:
                continue

            n_generated = int(row_tokens.shape[0])
            if n_generated < self.min_tokens_before_stop:
                continue

            partial = self.tokenizer.decode(row_tokens, skip_special_tokens=False)
            partial_clean = clean_special_tokens(partial)
            spans = iter_boxed_spans(partial_clean)
            if not spans:
                continue

            if self.mcq_valid_per_row is not None:
                ok = self._should_force_eos_mcq(row_idx, partial_clean, spans)
            else:
                ok = self._should_force_eos_free(partial_clean, spans)

            if ok:
                scores[row_idx, :] = float("-inf")
                scores[row_idx, self.eos_token_id] = 0.0
        return scores


class ModularPipeline:
    def __init__(
        self,
        gpu_id: str = "0",
        model_id: str = MODEL_ID,
        lora_adapter_path: str | None = None,
    ):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)

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
        if lora_adapter_path:
            try:
                from peft import PeftModel
            except Exception as exc:
                raise RuntimeError(
                    "LoRA adapter path was provided, but PEFT is unavailable. "
                    "Install `peft` to load adapters."
                ) from exc
            self.llm = PeftModel.from_pretrained(self.llm, lora_adapter_path)

        eot_ids = self.tokenizer.encode("</think>", add_special_tokens=False)
        if len(eot_ids) == 1:
            self.end_think_token_id = eot_ids[0]
        else:
            self.end_think_token_id = self.tokenizer.convert_tokens_to_ids("</think>")

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

    def _option_match_letter(self, primary_raw: str, options: list[str], labels: list[str]) -> str:
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
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
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
        force_smart_boxed_stop: bool = False,
        mcq_valid_per_row: list[set[str]] | None = None,
        post_box_patience_tokens: int = POST_BOX_PATIENCE_TOKENS_FREE,
        no_repeat_ngram_size: int = 0,
    ) -> list[dict]:
        chats = [self._make_chat(system, user) for system, user in zip(system_prompts, user_prompts)]
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
        if force_smart_boxed_stop:
            logits_processors.append(
                SmartBoxedStopProcessor(
                    tokenizer=self.tokenizer,
                    input_width=input_width,
                    eos_token_id=self.tokenizer.eos_token_id,
                    min_tokens_before_stop=MIN_TOKENS_BEFORE_BOXED_STOP,
                    post_box_patience_tokens=post_box_patience_tokens,
                    cue_window_chars=FINAL_ANSWER_CUE_WINDOW_CHARS,
                    mcq_valid_per_row=mcq_valid_per_row,
                )
            )

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
            gen_kwargs.update({"temperature": temperature, "top_p": top_p, "top_k": top_k})
        if no_repeat_ngram_size and no_repeat_ngram_size > 0:
            gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size

        with torch.no_grad():
            output_ids = self.llm.generate(**gen_kwargs)

        results: list[dict] = []
        for i in range(output_ids.shape[0]):
            new_tokens = output_ids[i, input_width:]
            raw = self.tokenizer.decode(new_tokens, skip_special_tokens=False).strip()
            raw = clean_special_tokens(raw)
            results.append({"raw": raw, "n_tokens": int(new_tokens.shape[0])})
        return results

    @staticmethod
    def _mcq_build_meta(
        *,
        primary_raw: str,
        response: str,
        n_tokens: int,
        finalizer_used: bool,
        option_match_used: bool,
        extractor_path: str,
        fallback_used: bool,
        malformed: bool,
        malformed_reason: str,
        finalizer_n_tokens: int = 0,
        boxed_in_raw: list[str],
    ) -> dict:
        return {
            "is_mcq": True,
            "output_type": "mcq",
            "raw": primary_raw,
            "n_tokens": n_tokens + finalizer_n_tokens,
            "boxed": extract_boxed(response),
            "cleaned_response": response,
            "raw_was_truncated": response != primary_raw,
            "finalizer_used": finalizer_used,
            "option_match_used": option_match_used,
            "extractor_path": extractor_path,
            "fallback_used": fallback_used,
            "malformed_output": malformed,
            "malformed_reason": malformed_reason,
            "boxed_count_in_raw": len(boxed_in_raw),
            "stop_policy": "smart_boxed",
        }

    def solve_mcq_batch(self, items: list[dict]) -> list[dict]:
        if not items:
            return []

        user_prompts = [build_mcq_user(item["question"], item["options"]) for item in items]
        system_prompts = [SYSTEM_PROMPT_MCQ] * len(items)
        mcq_valid = [
            {chr(65 + j) for j in range(len(item["options"]))} for item in items
        ]

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
            force_smart_boxed_stop=True,
            mcq_valid_per_row=mcq_valid,
            post_box_patience_tokens=POST_BOX_PATIENCE_TOKENS_MCQ,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE_MCQ,
        )

        solved = [None] * len(items)
        finalizer_items: list[dict] = []
        finalizer_indices: list[int] = []
        primary_raws: list[str] = []

        for idx, (item, out) in enumerate(zip(items, primary_outputs)):
            raw = out["raw"]
            labels = [chr(65 + i) for i in range(len(item["options"]))]
            letter = extract_first_valid_letter(raw, labels)
            primary_raws.append(raw)
            boxed_in_raw = extract_all_boxed(raw)

            if letter:
                response = mcq_canonical_response(letter)
                solved[idx] = {
                    "response": response,
                    "raw": raw,
                    "meta": self._mcq_build_meta(
                        primary_raw=raw,
                        response=response,
                        n_tokens=out["n_tokens"],
                        finalizer_used=False,
                        option_match_used=False,
                        extractor_path="mcq_first_valid_letter",
                        fallback_used=False,
                        malformed=False,
                        malformed_reason="",
                        boxed_in_raw=boxed_in_raw,
                    ),
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
            finalizer_user_prompts: list[str] = []

            for item, original_idx in zip(finalizer_items, finalizer_indices):
                labels = [chr(65 + i) for i in range(len(item["options"]))]
                valid_letters = ", ".join(labels)
                opts_text = "\n".join(
                    f"{lbl}. {str(opt).strip()}" for lbl, opt in zip(labels, item["options"])
                )
                finalizer_user_prompts.append(
                    f"Question:\n{item['question']}\n\n"
                    f"Options:\n{opts_text}\n\n"
                    f"Previous reasoning:\n{primary_raws[original_idx]}\n\n"
                    f"Valid choices: [{valid_letters}].\n"
                    "Choose the option that best matches the reasoning. "
                    "Output ONLY \\boxed{X}."
                )

            fin_valid = [
                {chr(65 + j) for j in range(len(item["options"]))} for item in finalizer_items
            ]

            finalizer_outputs = self._generate_batch(
                finalizer_system_prompts,
                finalizer_user_prompts,
                max_new_tokens=MAX_TOKENS_MCQ_FINAL,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                repetition_penalty=REP_PEN_MCQ_FINAL,
                do_sample=False,
                think_budget=0,
                force_smart_boxed_stop=True,
                mcq_valid_per_row=fin_valid,
                post_box_patience_tokens=POST_BOX_PATIENCE_TOKENS_MCQ,
                no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE_MCQ_FINAL,
            )

            for original_idx, fout in zip(finalizer_indices, finalizer_outputs):
                item = items[original_idx]
                labels = [chr(65 + i) for i in range(len(item["options"]))]
                letter = extract_first_valid_letter(fout["raw"], labels)
                option_match_used = False
                raw = primary_raws[original_idx]
                boxed_in_raw = extract_all_boxed(raw)

                if not letter:
                    matched = self._option_match_letter(raw, item["options"], labels)
                    if matched:
                        letter = matched
                        option_match_used = True

                if not letter:
                    response = mcq_canonical_response("")
                    solved[original_idx] = {
                        "response": response,
                        "raw": raw,
                        "meta": self._mcq_build_meta(
                            primary_raw=raw,
                            response=response,
                            n_tokens=primary_outputs[original_idx]["n_tokens"] + fout["n_tokens"],
                            finalizer_used=True,
                            option_match_used=option_match_used,
                            extractor_path="none",
                            fallback_used=True,
                            malformed=True,
                            malformed_reason="no_valid_mcq_letter",
                            finalizer_n_tokens=fout["n_tokens"],
                            boxed_in_raw=boxed_in_raw,
                        ),
                    }
                    continue

                response = mcq_canonical_response(letter)
                path = "mcq_finalizer_letter" if not option_match_used else "mcq_option_match_judger"
                solved[original_idx] = {
                    "response": response,
                    "raw": raw,
                    "meta": self._mcq_build_meta(
                        primary_raw=raw,
                        response=response,
                        n_tokens=primary_outputs[original_idx]["n_tokens"] + fout["n_tokens"],
                        finalizer_used=True,
                        option_match_used=option_match_used,
                        extractor_path=path,
                        fallback_used=False,
                        malformed=False,
                        malformed_reason="",
                        finalizer_n_tokens=fout["n_tokens"],
                        boxed_in_raw=boxed_in_raw,
                    ),
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
            force_smart_boxed_stop=True,
            mcq_valid_per_row=None,
            post_box_patience_tokens=POST_BOX_PATIENCE_TOKENS_FREE,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE_FREE,
        )

        solved: list[dict] = []
        for out in outputs:
            raw = out["raw"]
            response, extract_meta = canonicalize_free_response_with_meta(raw)
            boxed_in_raw = extract_meta.get("boxed_candidates") or extract_all_boxed(raw)
            solved.append(
                {
                    "response": response,
                    "raw": raw,
                    "meta": {
                        "is_mcq": False,
                        "output_type": "free",
                        "raw": raw,
                        "n_tokens": out["n_tokens"],
                        "boxed": extract_boxed(response),
                        "cleaned_response": response,
                        "raw_was_truncated": response.strip() != raw.strip(),
                        "boxed_fallback_used": bool(extract_meta.get("fallback_used")),
                        "extractor_path": extract_meta.get("extractor_path", "free"),
                        "boxed_count_in_raw": extract_meta.get("boxed_count_in_raw", len(boxed_in_raw)),
                        "selected_boxed_index": extract_meta.get("selected_boxed_index"),
                        "cue_matched": extract_meta.get("cue_matched", False),
                        "malformed_output": extract_meta.get("malformed_output", False),
                        "malformed_reason": extract_meta.get("malformed_reason", ""),
                        "stop_policy": "smart_boxed",
                    },
                }
            )
        return solved
