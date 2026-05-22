"""Model loading and batched solving logic.

Inference backend is vLLM instead of HuggingFace generate + LogitsProcessor.
The pipeline keeps prompting and answer extraction mostly unchanged, but adds
safer MCQ extraction and stronger finalizer recovery.

LoRA debugging:
  - First llm.generate() logs [debug] BEFORE/AFTER markers (see _generate_batch).
  - Use modular_pipeline.py or compare_runner.py for isolation checks.
  - Try --no-bitsandbytes or --inference-backend peft if vLLM LoRA hangs.
"""

import os
import re
import sys
from pathlib import Path
from typing import Any, Literal

from transformers import AutoTokenizer

from lora_vllm_utils import (
    check_vllm_version,
    normalize_vllm_optional,
    resolve_max_lora_rank,
    validate_lora_adapter_dir,
)
from prompting import build_free_user, build_mcq_user
from settings import (
    ENABLE_THINKING_MCQ_PRIMARY,
    VLLM_ENFORCE_EAGER,
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
    TOP_K_FREE,
    TOP_K_MCQ,
    TOP_P_FREE,
    TOP_P_MCQ,
    VLLM_GPU_MEMORY_UTILIZATION,
    VLLM_LOAD_FORMAT,
    VLLM_MAX_LORA_RANK,
    VLLM_MAX_LORAS,
    VLLM_MAX_MODEL_LEN,
    VLLM_MAX_NUM_BATCHED_TOKENS,
    VLLM_MAX_NUM_SEQS,
    VLLM_MIN_VERSION,
    VLLM_QUANTIZATION,
)
from text_processing import (
    _mcq_letter_from_boxed_inner,
    canonicalize_free_response_with_meta,
    clean_special_tokens,
    extract_boxed,
    extract_all_boxed,
    extract_mcq_letter_from_option_phrases,
    extract_tail_mcq_letter,
    extract_valid_letter,
    iter_boxed_spans,
    mcq_canonical_response,
    visible_answer_after_think_tags,
)

_FINAL_CUE_RE = re.compile(r"(?:final\s+answer|therefore|thus|hence)\b", re.IGNORECASE)
_SYSTEM_PROMPT_MCQ_FINALIZER = (
    "You are an MCQ extraction assistant. Do not solve the problem again. "
    "Read the provided raw model output and extract only the selected option letter. "
    "Return exactly one boxed option letter like \\boxed{A}. "
    "Do not output anything after the boxed letter."
)

_TRAINING_ECHO_PATTERNS = (
    "Solve this problem concisely",
    "Solve the problem concisely",
    "report only one final boxed answer",
    "\\begin{solution}",
)

_MCQ_UNCERTAINTY_RE = re.compile(
    r"\b(?:none\s+of\s+the\s+options|closest\s+option|might\s+be\s+a\s+mistake|"
    r"not\s+sure|cannot\s+determine|unclear)\b",
    re.IGNORECASE,
)

_VLLM_ARG_UNSET = object()


def _has_training_echo(text: str) -> bool:
    """Detect memorized training-template text from broken adapt LoRA outputs."""
    head = text[:500]
    return any(pattern in head for pattern in _TRAINING_ECHO_PATTERNS)


def _extract_last_valid_letter(text: str, labels: list[str]) -> str:
    """Return the last valid boxed MCQ letter in text.

    This is safer than first-boxed extraction because broken LoRA runs can emit
    fake early boxed letters before real reasoning.
    """
    valid = {label.strip().upper() for label in labels}
    candidates: list[str] = []

    for _start, _end, inner in iter_boxed_spans(text):
        cand = _mcq_letter_from_boxed_inner(inner, valid)
        if cand in valid:
            candidates.append(cand)

    return candidates[-1] if candidates else ""


def _should_force_mcq_finalizer(raw_text: str, labels: list[str]) -> bool:
    """Return True when primary MCQ extraction should defer to finalizer."""
    spans = iter_boxed_spans(raw_text)
    if not spans:
        return True

    valid = {str(label).strip().upper() for label in labels}
    valid_letter_spans = 0
    nonletter_boxed = 0

    for _start, _end, inner in spans:
        if _mcq_letter_from_boxed_inner(inner, valid):
            valid_letter_spans += 1
        elif str(inner or "").strip():
            nonletter_boxed += 1

    # Multiple boxed segments or boxed non-letters usually indicate noisy raw
    # output where direct "last valid letter" can be brittle.
    if len(spans) != 1:
        return True
    if nonletter_boxed > 0:
        return True
    if valid_letter_spans != 1:
        return True

    if _MCQ_UNCERTAINTY_RE.search(raw_text):
        return True
    return False


def _apply_vllm_quantization_kwargs(
    llm_kwargs: dict[str, Any],
    *,
    quantization: str | None,
    load_format: str | None,
) -> None:
    """Set or omit vLLM quantization/load_format keys."""
    if quantization is not None:
        llm_kwargs["quantization"] = quantization
    if load_format is not None:
        llm_kwargs["load_format"] = load_format


class ModularPipeline:
    def __init__(
        self,
        gpu_id: str = "0",
        model_id: str = MODEL_ID,
        lora_adapter_path: str | None = None,
        vllm_quantization: str | None = None,
        vllm_load_format: str | None = None,
        enforce_eager: bool | None = None,
        inference_backend: Literal["vllm", "peft"] = "vllm",
        mcq_max_new_tokens: int | None = None,
        mcq_final_max_new_tokens: int | None = None,
        free_max_new_tokens: int | None = None,
    ):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

        self.model_id = model_id
        self.inference_backend = inference_backend
        self._generate_call_count = 0
        self._peft_engine = None
        self.llm = None
        self.mcq_max_new_tokens = (
            MAX_TOKENS_MCQ if mcq_max_new_tokens is None else mcq_max_new_tokens
        )
        self.mcq_final_max_new_tokens = (
            MAX_TOKENS_MCQ_FINAL
            if mcq_final_max_new_tokens is None
            else mcq_final_max_new_tokens
        )
        self.free_max_new_tokens = (
            MAX_TOKENS_FREE if free_max_new_tokens is None else free_max_new_tokens
        )

        resolved_adapter: Path | None = None
        if lora_adapter_path:
            resolved_adapter = validate_lora_adapter_dir(lora_adapter_path)
        self.lora_adapter_path = str(resolved_adapter) if resolved_adapter else None

        if inference_backend == "peft":
            if resolved_adapter is None:
                raise ValueError(
                    "--inference-backend peft requires --lora-adapter-path"
                )
            from peft_inference import PeftGenerateEngine

            self._peft_engine = PeftGenerateEngine(
                model_id=model_id,
                lora_adapter_path=self.lora_adapter_path,
            )
            self.tokenizer = self._peft_engine.tokenizer
            self._lora_request_obj = None
            self._lora_active = False
            return

        check_vllm_version(VLLM_MIN_VERSION)
        quantization_override = (
            normalize_vllm_optional(vllm_quantization)
            if vllm_quantization is not None
            else _VLLM_ARG_UNSET
        )
        load_format_override = (
            normalize_vllm_optional(vllm_load_format)
            if vllm_load_format is not None
            else _VLLM_ARG_UNSET
        )

        self.vllm_quantization = (
            VLLM_QUANTIZATION
            if quantization_override is _VLLM_ARG_UNSET
            else quantization_override
        )
        self.vllm_load_format = (
            VLLM_LOAD_FORMAT
            if load_format_override is _VLLM_ARG_UNSET
            else load_format_override
        )

        if self.vllm_quantization is None and load_format_override is _VLLM_ARG_UNSET:
            self.vllm_load_format = "auto"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=False,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        try:
            from vllm import LLM  # type: ignore
            from vllm.lora.request import LoRARequest  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "vLLM is required for inference. Install `vllm` and optionally `bitsandbytes`."
            ) from exc

        llm_kwargs: dict[str, Any] = dict(
            model=model_id,
            enable_prefix_caching=False,
            gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
            max_model_len=VLLM_MAX_MODEL_LEN,
            trust_remote_code=True,
            max_num_seqs=VLLM_MAX_NUM_SEQS,
            max_num_batched_tokens=VLLM_MAX_NUM_BATCHED_TOKENS,
            enforce_eager=(
                VLLM_ENFORCE_EAGER if enforce_eager is None else enforce_eager
            ),
        )
        _apply_vllm_quantization_kwargs(
            llm_kwargs,
            quantization=self.vllm_quantization,
            load_format=self.vllm_load_format,
        )

        self._lora_request_obj = None
        self._lora_active = False
        self._max_lora_rank = VLLM_MAX_LORA_RANK
        if resolved_adapter is not None:
            self._max_lora_rank = resolve_max_lora_rank(resolved_adapter)
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_lora_rank"] = self._max_lora_rank
            llm_kwargs["max_loras"] = VLLM_MAX_LORAS
            self._lora_request_obj = LoRARequest("adapter", 1, str(resolved_adapter))
            self._lora_active = True

        eff_eager = llm_kwargs["enforce_eager"]
        print("Initializing vLLM with:")
        print(f"  enforce_eager={eff_eager}")
        print(f"  model_id={model_id}")
        print(f"  lora_adapter_path={self.lora_adapter_path}")
        print(f"  gpu_memory_utilization={VLLM_GPU_MEMORY_UTILIZATION}")
        print(f"  max_model_len={VLLM_MAX_MODEL_LEN}")
        print(f"  max_num_seqs={VLLM_MAX_NUM_SEQS}")
        print(f"  max_num_batched_tokens={VLLM_MAX_NUM_BATCHED_TOKENS}")
        print(f"  quantization={self.vllm_quantization}")
        print(f"  load_format={self.vllm_load_format}")
        print(f"  mcq_max_new_tokens={self.mcq_max_new_tokens}")
        print(f"  mcq_final_max_new_tokens={self.mcq_final_max_new_tokens}")
        print(f"  free_max_new_tokens={self.free_max_new_tokens}")
        if resolved_adapter is not None:
            print("  enable_lora=True")
            print(f"  max_lora_rank={self._max_lora_rank}")
            print(f"  max_loras={VLLM_MAX_LORAS}")
            print(
                "  LoRA diagnostics: watch vLLM startup logs for SupportsLoRA, "
                "Qwen3ForCausalLM, or 'falling back to Transformers' — fallback can "
                "mean LoRA is not applied natively."
            )
            if not eff_eager and "qwen3" in model_id.lower():
                try:
                    from packaging.version import Version
                    import vllm  # type: ignore

                    installed = getattr(vllm, "__version__", "0.0.0")
                    if Version(installed) < Version(VLLM_MIN_VERSION):
                        print(
                            "Warning: Qwen3 + LoRA + enforce_eager=False on vLLM "
                            f"{installed} may hang on first generate. Try "
                            "--vllm-enforce-eager or upgrade vLLM."
                        )
                except Exception:
                    pass

        self.llm = LLM(**llm_kwargs)
        print(
            "vLLM engine ready. First generate() may spend time on torch.compile "
            "(high CPU, 0% GPU is normal until it finishes).",
            flush=True,
        )
        self._judger: Any = None

    def set_lora_active(self, enabled: bool) -> None:
        """Toggle the LoRA adapter for subsequent `generate` calls.

        Used by the comparison runner to alternate between a "base" pass
        (`enabled=False`) and a "lora" pass (`enabled=True`) on a single
        engine. Default behaviour is unchanged: `enabled` is True whenever
        an adapter was provided at init.

        Raises
        ------
        RuntimeError
            If `enabled=True` is requested but no adapter was loaded at
            init time (there is nothing to activate).
        """
        if enabled and self._lora_request_obj is None:
            raise RuntimeError(
                "Cannot enable LoRA: no adapter was provided at "
                "ModularPipeline init time."
            )
        self._lora_active = bool(enabled)

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

    def _make_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        enable_thinking: bool = True,
    ) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            if not enable_thinking:
                user_prompt = user_prompt + "\n\n/no_think"
            return self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )

    @staticmethod
    def _build_vllm_sampling_params(
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        do_sample: bool,
        no_repeat_ngram_size: int = 0,
        seed: int | None = None,
    ):
        """Map HF-style do_sample to vLLM SamplingParams.

        vLLM has no do_sample flag. Greedy decoding is temperature=0.
        """
        from vllm import SamplingParams  # type: ignore

        if do_sample:
            eff_temp = temperature
            eff_top_p = top_p
            eff_top_k = top_k if top_k > 0 else -1
        else:
            eff_temp = 0.0
            eff_top_p = 1.0
            eff_top_k = -1

        sampling_kwargs: dict[str, Any] = dict(
            max_tokens=max_tokens,
            temperature=eff_temp,
            top_p=eff_top_p,
            top_k=eff_top_k,
            min_p=0.0,
            presence_penalty=0.0,
            repetition_penalty=repetition_penalty,
        )
        if seed is not None:
            sampling_kwargs["seed"] = int(seed)

        if no_repeat_ngram_size and no_repeat_ngram_size > 0:
            sampling_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size

        try:
            return SamplingParams(**sampling_kwargs)
        except TypeError:
            sampling_kwargs.pop("no_repeat_ngram_size", None)
            return SamplingParams(**sampling_kwargs)

    def _truncate_for_smart_boxed_stop(
        self,
        *,
        raw_clean: str,
        mcq_valid: set[str] | None,
        post_box_patience_tokens: int,
    ) -> str:
        """Post-hoc truncate to a likely-final boxed span.

        This does not save generation time. It only cleans decoded text after vLLM finishes.
        """
        spans = iter_boxed_spans(raw_clean)
        if not spans:
            return raw_clean

        if mcq_valid is not None:
            valid_upper = {str(x).strip().upper() for x in mcq_valid}

            valid_spans: list[tuple[int, int, bool]] = []
            for start, end, inner in spans:
                cand = _mcq_letter_from_boxed_inner(inner, valid_upper)
                if not cand:
                    continue

                prefix = raw_clean[:end]
                prefix_tokens = len(self.tokenizer.encode(prefix, add_special_tokens=False))
                if prefix_tokens < MIN_TOKENS_BEFORE_BOXED_STOP:
                    continue

                cue_window = max(0, int(FINAL_ANSWER_CUE_WINDOW_CHARS))
                context_start = max(0, start - cue_window)
                context = raw_clean[context_start:start]
                cue_hit = bool(_FINAL_CUE_RE.search(context))
                valid_spans.append((start, end, cue_hit))

            if valid_spans:
                # Prefer a cue-aligned final answer box; otherwise keep the last
                # valid boxed letter so mid-reasoning tentative choices do not win.
                cue_spans = [span for span in valid_spans if span[2]]
                chosen = cue_spans[-1] if cue_spans else valid_spans[-1]
                _start, end, _cue_hit = chosen
                if post_box_patience_tokens > 0:
                    suffix = raw_clean[end:]
                    suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
                    keep_tokens = suffix_tokens[: post_box_patience_tokens]
                    if keep_tokens:
                        suffix_keep = self.tokenizer.decode(keep_tokens, skip_special_tokens=True)
                        return (raw_clean[:end] + suffix_keep).rstrip()
                return raw_clean[:end].rstrip()

            # Do not truncate MCQ output to a random non-letter box.
            return raw_clean.rstrip()

        # Free-form: keep up to the last complete boxed span.
        return raw_clean[: spans[-1][1]].rstrip()

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
        enable_thinking: bool = True,
        force_smart_boxed_stop: bool = False,
        mcq_valid_per_row: list[set[str]] | None = None,
        post_box_patience_tokens: int = POST_BOX_PATIENCE_TOKENS_FREE,
        no_repeat_ngram_size: int = 0,
        sampling_seed: int | None = None,
    ) -> list[dict]:
        # vLLM cannot enforce your old token-level think_budget logic.
        del think_budget

        chats = [
            self._make_chat(system, user, enable_thinking=enable_thinking)
            for system, user in zip(system_prompts, user_prompts)
        ]

        sampling_params = self._build_vllm_sampling_params(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            no_repeat_ngram_size=no_repeat_ngram_size,
            seed=sampling_seed,
        )

        llm_kwargs: dict[str, Any] = dict(
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        if self._peft_engine is not None:
            completion_texts = self._peft_engine.generate_texts(
                chats,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
            )
            results: list[dict] = []
            for i, completion_text in enumerate(completion_texts):
                raw_before_trunc = clean_special_tokens(completion_text).strip()
                raw_clean = raw_before_trunc
                if force_smart_boxed_stop:
                    mcq_valid = mcq_valid_per_row[i] if mcq_valid_per_row is not None else None
                    raw_clean = self._truncate_for_smart_boxed_stop(
                        raw_clean=raw_clean,
                        mcq_valid=mcq_valid,
                        post_box_patience_tokens=post_box_patience_tokens,
                    )
                pre_trunc_tokens = len(
                    self.tokenizer.encode(raw_before_trunc, add_special_tokens=False)
                )
                post_trunc_tokens = len(
                    self.tokenizer.encode(raw_clean, add_special_tokens=False)
                )
                results.append(
                    {
                        "raw": raw_clean,
                        "raw_before_trunc": raw_before_trunc,
                        "n_tokens": int(post_trunc_tokens),
                        "pre_trunc_n_tokens": int(pre_trunc_tokens),
                        "generation_hit_max": bool(pre_trunc_tokens >= max_new_tokens),
                        "raw_was_post_truncated": raw_clean.strip()
                        != raw_before_trunc.strip(),
                    }
                )
            return results

        if self._lora_active and self._lora_request_obj is not None:
            llm_kwargs["lora_request"] = self._lora_request_obj

        self._generate_call_count += 1
        is_first_generate = self._generate_call_count == 1
        if is_first_generate:
            print(
                f"[debug] BEFORE llm.generate #{self._generate_call_count} "
                f"(batch_size={len(chats)}, max_new_tokens={max_new_tokens})",
                flush=True,
            )
            lora_request_attached = (
                self._lora_active and self._lora_request_obj is not None
            )
            print(
                f"[debug]   lora_enabled={lora_request_attached} "
                f"(adapter_loaded={self._lora_request_obj is not None}, "
                f"active={self._lora_active})",
                flush=True,
            )
            if lora_request_attached:
                print(
                    f"[debug]   lora_request={self._lora_request_obj!r}",
                    flush=True,
                )
            print(
                f"[debug]   quantization={self.vllm_quantization!r} "
                f"load_format={self.vllm_load_format!r}",
                flush=True,
            )
            sys.stdout.flush()

        request_outputs = self.llm.generate(chats, **llm_kwargs)

        if is_first_generate:
            print(
                f"[debug] AFTER llm.generate #{self._generate_call_count}",
                flush=True,
            )
            sys.stdout.flush()

        results: list[dict] = []
        for i, req_out in enumerate(request_outputs):
            completion_text = req_out.outputs[0].text
            raw_before_trunc = clean_special_tokens(completion_text).strip()
            raw_clean = raw_before_trunc

            if force_smart_boxed_stop:
                mcq_valid = mcq_valid_per_row[i] if mcq_valid_per_row is not None else None
                raw_clean = self._truncate_for_smart_boxed_stop(
                    raw_clean=raw_clean,
                    mcq_valid=mcq_valid,
                    post_box_patience_tokens=post_box_patience_tokens,
                )

            pre_trunc_tokens = len(self.tokenizer.encode(raw_before_trunc, add_special_tokens=False))
            post_trunc_tokens = len(self.tokenizer.encode(raw_clean, add_special_tokens=False))

            results.append(
                {
                    "raw": raw_clean,
                    "raw_before_trunc": raw_before_trunc,
                    "n_tokens": int(post_trunc_tokens),
                    "pre_trunc_n_tokens": int(pre_trunc_tokens),
                    "generation_hit_max": bool(pre_trunc_tokens >= max_new_tokens),
                    "raw_was_post_truncated": raw_clean.strip() != raw_before_trunc.strip(),
                }
            )

        return results

    def _mcq_build_meta(
        self,
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
        training_echo_detected: bool = False,
        pre_trunc_n_tokens: int | None = None,
        generation_hit_max: bool | None = None,
        raw_was_post_truncated: bool = False,
        confidence_tier: str = "",
        guessed_letter_used: bool = False,
        finalizer_extractor_path: str = "",
    ) -> dict:
        raw_recovery_path = ""
        for candidate_path in (extractor_path, finalizer_extractor_path):
            if "raw_option_phrase" in candidate_path:
                raw_recovery_path = candidate_path
                break

        return {
            "is_mcq": True,
            "output_type": "mcq",
            "model_id": self.model_id,
            "lora_adapter_path": self.lora_adapter_path,
            "raw": primary_raw,
            "n_tokens": n_tokens,
            "finalizer_n_tokens": finalizer_n_tokens,
            "total_n_tokens": n_tokens + finalizer_n_tokens,
            "pre_trunc_n_tokens": pre_trunc_n_tokens,
            "generation_hit_max": generation_hit_max,
            "raw_was_post_truncated": raw_was_post_truncated,
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
            "training_echo_detected": training_echo_detected,
            "confidence_tier": confidence_tier,
            "guessed_letter_used": guessed_letter_used,
            "finalizer_extractor_path": finalizer_extractor_path,
            "raw_letter_recovered": bool(raw_recovery_path),
            "raw_letter_recovery_path": raw_recovery_path,
            "stop_policy": "smart_boxed_post",
        }

    def _best_effort_mcq_guess(
        self,
        *,
        primary_raw: str,
        primary_raw_before_trunc: str,
        finalizer_raw: str,
        options: list[str],
        labels: list[str],
        training_echo_detected: bool,
    ) -> tuple[str, bool, str, str]:
        """Return (letter, option_match_used, extractor_path, confidence_tier)."""
        letter = _extract_last_valid_letter(finalizer_raw, labels)
        if letter:
            return letter, False, "mcq_finalizer_last_valid_letter", "boxed_high"

        letter = extract_tail_mcq_letter(finalizer_raw, labels)
        if letter:
            return letter, False, "mcq_finalizer_tail_phrase", "answer_phrase_medium"

        letter = extract_mcq_letter_from_option_phrases(finalizer_raw, labels)
        if letter:
            return letter, False, "mcq_finalizer_raw_option_phrase", "answer_phrase_medium"

        for blob in (visible_answer_after_think_tags(finalizer_raw), finalizer_raw):
            letter = extract_valid_letter(blob, labels) if blob else ""
            if letter:
                return letter, False, "mcq_finalizer_phrase_fulltext", "answer_phrase_medium"

        if not training_echo_detected:
            letter = _extract_last_valid_letter(primary_raw, labels)
            if letter:
                return letter, False, "mcq_last_valid_letter", "boxed_high"

            letter = extract_tail_mcq_letter(primary_raw, labels)
            if letter:
                return letter, False, "mcq_primary_tail_phrase", "answer_phrase_medium"

            letter = extract_mcq_letter_from_option_phrases(primary_raw, labels)
            if letter:
                return letter, False, "mcq_raw_option_phrase", "answer_phrase_medium"

            for blob in (visible_answer_after_think_tags(primary_raw), primary_raw):
                letter = extract_valid_letter(blob, labels) if blob else ""
                if letter:
                    return letter, False, "mcq_primary_phrase_fulltext", "answer_phrase_medium"

            if primary_raw_before_trunc.strip() != primary_raw.strip():
                letter = _extract_last_valid_letter(primary_raw_before_trunc, labels)
                if letter:
                    return letter, False, "mcq_pre_trunc_last_valid_letter", "boxed_high"

                letter = extract_mcq_letter_from_option_phrases(primary_raw_before_trunc, labels)
                if letter:
                    return letter, False, "mcq_pre_trunc_raw_option_phrase", "answer_phrase_medium"

                for blob in (
                    visible_answer_after_think_tags(primary_raw_before_trunc),
                    primary_raw_before_trunc,
                ):
                    letter = extract_valid_letter(blob, labels) if blob else ""
                    if letter:
                        return letter, False, "mcq_pre_trunc_phrase_fulltext", "answer_phrase_medium"

            combo = f"{primary_raw}\n\n{finalizer_raw}"
            letter = extract_mcq_letter_from_option_phrases(combo, labels)
            if letter:
                return letter, False, "mcq_combined_raw_option_phrase", "answer_phrase_medium"

            for blob in (visible_answer_after_think_tags(combo), combo):
                letter = extract_valid_letter(blob, labels) if blob else ""
                if letter:
                    return letter, False, "mcq_combined_phrase_fulltext", "answer_phrase_medium"

                letter = extract_tail_mcq_letter(blob, labels) if blob else ""
                if letter:
                    return letter, False, "mcq_combined_tail_phrase", "answer_phrase_medium"

            if primary_raw_before_trunc.strip() != primary_raw.strip():
                combo_pre = f"{primary_raw_before_trunc}\n\n{finalizer_raw}"
                for blob in (visible_answer_after_think_tags(combo_pre), combo_pre):
                    letter = extract_valid_letter(blob, labels) if blob else ""
                    if letter:
                        return letter, False, "mcq_combined_pre_trunc_phrase", "answer_phrase_medium"

            matched = self._option_match_letter(combo, options, labels)
            if matched:
                return matched, True, "mcq_option_match_judger", "option_match_medium"

        return "", False, "mcq_abstain_no_signal", "none"

    def solve_mcq_batch(
        self,
        items: list[dict],
        *,
        do_sample: bool = False,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        sampling_seed: int | None = None,
    ) -> list[dict]:
        if not items:
            return []
        print(
            f"[debug] solve_mcq_batch: batch_size={len(items)} "
            f"mcq_max_new_tokens={self.mcq_max_new_tokens}",
            flush=True,
        )

        user_prompts = [build_mcq_user(item["question"], item["options"]) for item in items]
        system_prompts = [SYSTEM_PROMPT_MCQ] * len(items)

        mcq_valid = [
            {chr(65 + j) for j in range(len(item["options"]))}
            for item in items
        ]

        primary_outputs = self._generate_batch(
            system_prompts,
            user_prompts,
            max_new_tokens=self.mcq_max_new_tokens,
            temperature=TEMP_MCQ if temperature is None else float(temperature),
            top_p=TOP_P_MCQ if top_p is None else float(top_p),
            top_k=TOP_K_MCQ if top_k is None else int(top_k),
            repetition_penalty=REP_PEN_MCQ,
            do_sample=bool(do_sample),
            think_budget=0,
            enable_thinking=ENABLE_THINKING_MCQ_PRIMARY,
            force_smart_boxed_stop=True,
            mcq_valid_per_row=mcq_valid,
            post_box_patience_tokens=POST_BOX_PATIENCE_TOKENS_MCQ,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE_MCQ,
            sampling_seed=sampling_seed,
        )

        solved: list[dict | None] = [None] * len(items)
        finalizer_items: list[dict] = []
        finalizer_indices: list[int] = []
        primary_raws: list[str] = []

        primary_pre_truncs: list[str] = []

        for idx, (item, out) in enumerate(zip(items, primary_outputs)):
            raw = out["raw"]
            raw_pre = str(out.get("raw_before_trunc") or raw)
            labels = [chr(65 + i) for i in range(len(item["options"]))]
            boxed_in_raw = extract_all_boxed(raw)
            echo = _has_training_echo(raw)

            letter = "" if echo else _extract_last_valid_letter(raw, labels)
            extractor_path_primary = "mcq_last_valid_letter"
            if not letter and not echo:
                for blob in (visible_answer_after_think_tags(raw), raw):
                    if not blob:
                        continue
                    letter = extract_mcq_letter_from_option_phrases(blob, labels)
                    if letter:
                        extractor_path_primary = "mcq_raw_option_phrase"
                        break
                    letter = extract_valid_letter(blob, labels)
                    if letter:
                        extractor_path_primary = "mcq_primary_phrase_fulltext"
                        break
            if not letter and not echo and raw_pre.strip() != raw.strip():
                letter = _extract_last_valid_letter(raw_pre, labels)
                if letter:
                    extractor_path_primary = "mcq_primary_pre_trunc_last_valid_letter"
            if not letter and not echo and raw_pre.strip() != raw.strip():
                for blob in (visible_answer_after_think_tags(raw_pre), raw_pre):
                    if not blob:
                        continue
                    letter = extract_mcq_letter_from_option_phrases(blob, labels)
                    if letter:
                        extractor_path_primary = "mcq_primary_pre_trunc_raw_option_phrase"
                        break
                    letter = extract_valid_letter(blob, labels)
                    if letter:
                        extractor_path_primary = "mcq_primary_pre_trunc_phrase"
                        break

            primary_raws.append(raw)
            primary_pre_truncs.append(raw_pre)

            if letter:
                force_finalizer = _should_force_mcq_finalizer(raw, labels)
                if not force_finalizer:
                    response = mcq_canonical_response(letter)
                    phrase_only = extractor_path_primary in (
                        "mcq_primary_phrase_fulltext",
                        "mcq_primary_pre_trunc_phrase",
                    )
                    solved[idx] = {
                        "response": response,
                        "raw": raw,
                        "meta": self._mcq_build_meta(
                            primary_raw=raw,
                            response=response,
                            n_tokens=out["n_tokens"],
                            finalizer_used=False,
                            option_match_used=False,
                            extractor_path=extractor_path_primary,
                            fallback_used=phrase_only,
                            malformed=False,
                            malformed_reason="",
                            boxed_in_raw=boxed_in_raw,
                            training_echo_detected=echo,
                            pre_trunc_n_tokens=out.get("pre_trunc_n_tokens"),
                            generation_hit_max=out.get("generation_hit_max"),
                            raw_was_post_truncated=bool(out.get("raw_was_post_truncated", False)),
                            confidence_tier=(
                                "boxed_high"
                                if extractor_path_primary
                                in ("mcq_last_valid_letter", "mcq_primary_pre_trunc_last_valid_letter")
                                else "answer_phrase_medium"
                            ),
                            guessed_letter_used=phrase_only,
                            finalizer_extractor_path="",
                        ),
                    }
                else:
                    finalizer_items.append(item)
                    finalizer_indices.append(idx)
            else:
                finalizer_items.append(item)
                finalizer_indices.append(idx)

        if finalizer_items:
            print(
                f"[debug] solve_mcq_batch: finalizer_items={len(finalizer_items)} "
                f"mcq_final_max_new_tokens={self.mcq_final_max_new_tokens}",
                flush=True,
            )
            finalizer_system_prompts = [_SYSTEM_PROMPT_MCQ_FINALIZER] * len(finalizer_items)

            finalizer_user_prompts: list[str] = []

            for item, original_idx in zip(finalizer_items, finalizer_indices):
                labels = [chr(65 + i) for i in range(len(item["options"]))]
                valid_letters = ", ".join(labels)
                opts_text = "\n".join(
                    f"{lbl}. {str(opt).strip()}"
                    for lbl, opt in zip(labels, item["options"])
                )
                raw_for_finalizer = primary_pre_truncs[original_idx]

                finalizer_user_prompts.append(
                    "Do not solve this question. Only extract a letter from the raw output.\n\n"
                    f"Question:\n{item['question']}\n\n"
                    f"Options:\n{opts_text}\n\n"
                    f"Valid choices: [{valid_letters}].\n\n"
                    "Raw model output to extract from:\n"
                    f"{raw_for_finalizer}\n\n"
                    "Return exactly one boxed letter: \\boxed{X}, where X is one valid choice. "
                    "If raw includes a boxed non-letter value (e.g. \\boxed{601}), map that value "
                    "to the matching option text and return that option letter. "
                    "If multiple letters appear, use the final supported choice in the raw text. "
                    "Do not output explanation or any extra text."
                )

            fin_valid = [
                {chr(65 + j) for j in range(len(item["options"]))}
                for item in finalizer_items
            ]

            finalizer_outputs = self._generate_batch(
                finalizer_system_prompts,
                finalizer_user_prompts,
                max_new_tokens=self.mcq_final_max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                repetition_penalty=REP_PEN_MCQ_FINAL,
                do_sample=False,
                think_budget=0,
                enable_thinking=False,
                force_smart_boxed_stop=False,
                mcq_valid_per_row=fin_valid,
                post_box_patience_tokens=0,
                no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE_MCQ_FINAL,
            )

            for original_idx, fout in zip(finalizer_indices, finalizer_outputs):
                item = items[original_idx]
                labels = [chr(65 + i) for i in range(len(item["options"]))]

                raw = primary_raws[original_idx]
                boxed_in_raw = extract_all_boxed(raw)
                echo = _has_training_echo(raw)
                letter, option_match_used, path, confidence_tier = self._best_effort_mcq_guess(
                    primary_raw=raw,
                    primary_raw_before_trunc=primary_pre_truncs[original_idx],
                    finalizer_raw=fout["raw"],
                    options=item["options"],
                    labels=labels,
                    training_echo_detected=echo,
                )

                response = mcq_canonical_response(letter)
                guessed_letter_used = confidence_tier in {
                    "answer_phrase_medium",
                    "option_match_medium",
                }
                malformed = not bool(letter)
                malformed_reason = ""
                if malformed:
                    malformed_reason = "training_echo" if echo else "no_valid_mcq_letter"

                solved[original_idx] = {
                    "response": response,
                    "raw": raw,
                    "meta": self._mcq_build_meta(
                        primary_raw=raw,
                        response=response,
                        n_tokens=primary_outputs[original_idx]["n_tokens"],
                        finalizer_used=True,
                        option_match_used=option_match_used,
                        extractor_path=path,
                        fallback_used=guessed_letter_used,
                        malformed=malformed,
                        malformed_reason=malformed_reason,
                        finalizer_n_tokens=fout["n_tokens"],
                        boxed_in_raw=boxed_in_raw,
                        training_echo_detected=echo,
                        pre_trunc_n_tokens=primary_outputs[original_idx].get("pre_trunc_n_tokens"),
                        generation_hit_max=primary_outputs[original_idx].get("generation_hit_max"),
                        raw_was_post_truncated=bool(
                            primary_outputs[original_idx].get("raw_was_post_truncated", False)
                        ),
                        confidence_tier=confidence_tier,
                        guessed_letter_used=guessed_letter_used,
                        finalizer_extractor_path=path,
                    ),
                }

        return [x for x in solved if x is not None]

    def solve_free_batch(
        self,
        items: list[dict],
        *,
        do_sample: bool = True,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        sampling_seed: int | None = None,
    ) -> list[dict]:
        if not items:
            return []
        print(
            f"[debug] solve_free_batch: batch_size={len(items)} "
            f"free_max_new_tokens={self.free_max_new_tokens}",
            flush=True,
        )

        user_prompts = [build_free_user(item["question"]) for item in items]
        system_prompts = [SYSTEM_PROMPT_FREE] * len(items)

        outputs = self._generate_batch(
            system_prompts,
            user_prompts,
            max_new_tokens=self.free_max_new_tokens,
            temperature=TEMP_FREE if temperature is None else float(temperature),
            top_p=TOP_P_FREE if top_p is None else float(top_p),
            top_k=TOP_K_FREE if top_k is None else int(top_k),
            repetition_penalty=REP_PEN_FREE,
            do_sample=bool(do_sample),
            think_budget=0,
            force_smart_boxed_stop=True,
            mcq_valid_per_row=None,
            post_box_patience_tokens=POST_BOX_PATIENCE_TOKENS_FREE,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE_FREE,
            sampling_seed=sampling_seed,
        )

        solved: list[dict] = []

        for item, out in zip(items, outputs):
            raw = out["raw"]
            response, extract_meta = canonicalize_free_response_with_meta(
                raw,
                question=item.get("question"),
            )
            boxed_in_raw = extract_meta.get("boxed_candidates") or extract_all_boxed(raw)

            solved.append(
                {
                    "response": response,
                    "raw": raw,
                    "meta": {
                        "is_mcq": False,
                        "output_type": "free",
                        "model_id": self.model_id,
                        "lora_adapter_path": self.lora_adapter_path,
                        "raw": raw,
                        "n_tokens": out["n_tokens"],
                        "pre_trunc_n_tokens": out.get("pre_trunc_n_tokens"),
                        "generation_hit_max": out.get("generation_hit_max"),
                        "boxed": extract_boxed(response),
                        "cleaned_response": response,
                        "raw_was_truncated": response.strip() != raw.strip(),
                        "raw_was_post_truncated": out.get("raw_was_post_truncated", False),
                        "boxed_fallback_used": bool(extract_meta.get("fallback_used")),
                        "extractor_path": extract_meta.get("extractor_path", "free"),
                        "boxed_count_in_raw": extract_meta.get("boxed_count_in_raw", len(boxed_in_raw)),
                        "selected_boxed_index": extract_meta.get("selected_boxed_index"),
                        "expected_ans_slots": extract_meta.get("expected_ans_slots", 0),
                        "extracted_values": extract_meta.get("extracted_values", []),
                        "phrase_override": extract_meta.get("phrase_override", False),
                        "cue_matched": extract_meta.get("cue_matched", False),
                        "malformed_output": extract_meta.get("malformed_output", False),
                        "malformed_reason": extract_meta.get("malformed_reason", ""),
                        "training_echo_detected": _has_training_echo(raw),
                        "stop_policy": "smart_boxed_post",
                    },
                }
            )

        return solved