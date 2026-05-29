"""Transformers + PEFT inference fallback when vLLM LoRA is unstable.

Slower than vLLM but useful for verifying adapter weights and local eval.
Enable with: --inference-backend peft --lora-adapter-path <adapter_dir>
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from settings import MODEL_ID


class PeftGenerateEngine:
    """Minimal batched generator using PeftModel (no vLLM)."""

    def __init__(
        self,
        *,
        model_id: str = MODEL_ID,
        lora_adapter_path: str,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.model_id = model_id
        self.lora_adapter_path = lora_adapter_path

        print("Initializing Transformers + PEFT backend:")
        print(f"  model_id={model_id}")
        print(f"  lora_adapter_path={lora_adapter_path}")
        print(f"  dtype={dtype}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=False,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto",
        )
        from peft import PeftModel

        self.model = PeftModel.from_pretrained(
            base_model,
            lora_adapter_path,
            is_trainable=False,
        )
        self.model.eval()
        print("PEFT model loaded.", flush=True)

    @torch.inference_mode()
    def generate_texts(
        self,
        chats: list[str],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        do_sample: bool,
    ) -> list[str]:
        device = self.model.device
        encoded = self.tokenizer(
            chats,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        prompt_lens = attention_mask.sum(dim=1).tolist()

        gen_kwargs: dict[str, Any] = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            repetition_penalty=repetition_penalty,
        )
        if do_sample:
            gen_kwargs.update(
                do_sample=True,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
            )
            if top_k > 0:
                gen_kwargs["top_k"] = top_k
        else:
            gen_kwargs["do_sample"] = False

        print(
            f"[peft] generate_texts: batch={len(chats)} max_new_tokens={max_new_tokens}",
            flush=True,
        )
        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )
        print("[peft] generate_texts: done", flush=True)

        texts: list[str] = []
        for row_idx, seq in enumerate(output_ids):
            new_tokens = seq[int(prompt_lens[row_idx]) :]
            texts.append(
                self.tokenizer.decode(new_tokens, skip_special_tokens=False).strip()
            )
        return texts
