"""Custom LoRA fine-tuning entrypoint for the modular pipeline."""

import json
import os
import random
import re
import subprocess
import sys
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
    build_mcq_reasoning_target,
    build_reasoning_train_user,
)
from settings import (
    ADAPT_DEFAULT_LEARNING_RATE,
    ADAPT_DEFAULT_MAX_STEPS,
    MODEL_ID,
    REASONING_DEFAULT_LEARNING_RATE,
    REASONING_DEFAULT_MAX_STEPS,
    STAGE2_DEFAULT_HOLDOUT_FRACTION,
    STAGE2_MCQ_WITH_REASONING,
    ADAPT_DEFAULT_MAX_SEQ_LEN,
    TRAIN_LOAD_IN_4BIT,
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


def _load_openmath_examples(max_examples: int | None, seed: int) -> list[dict]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(
            "Stage `reasoning` requires `datasets`. Install with `pip install datasets`."
        ) from exc

    ds = load_dataset("unsloth/OpenMathReasoning-mini", split="cot")
    rows = [dict(row) for row in ds]
    rows = _sample_cap(rows, max_examples, seed)
    examples: list[dict] = []
    for row in rows:
        problem = str(row.get("problem", "")).strip()
        solution = str(row.get("generated_solution", "")).strip()
        expected = str(row.get("expected_answer", "")).strip()
        if not problem or not solution:
            continue
        target = _enforce_single_final_boxed(solution, fallback_answer=expected)
        examples.append(
            {
                "prompt": build_reasoning_train_user(problem),
                "target": target,
                "system_prompt": (
                    "You are an expert competition mathematician. "
                    "Provide concise step-by-step reasoning and finish with exactly one final \\boxed{...}."
                ),
                "source": "openmath",
            }
        )
    return examples


def _load_hendrycks_examples(
    configs: list[str],
    max_examples: int | None,
    seed: int,
) -> list[dict]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(
            "Stage `reasoning` requires `datasets`. Install with `pip install datasets`."
        ) from exc

    merged: list[dict] = []
    for cfg in configs:
        ds = load_dataset("EleutherAI/hendrycks_math", cfg, split="train")
        for row in ds:
            merged.append(dict(row))

    merged = _sample_cap(merged, max_examples, seed)
    examples: list[dict] = []
    for row in merged:
        problem = str(row.get("problem", "")).strip()
        solution = str(row.get("solution", "")).strip()
        if not problem or not solution:
            continue
        target = _enforce_single_final_boxed(solution)
        examples.append(
            {
                "prompt": build_reasoning_train_user(problem),
                "target": target,
                "system_prompt": (
                    "You are an expert competition mathematician. "
                    "Provide concise step-by-step reasoning and finish with exactly one final \\boxed{...}."
                ),
                "source": "hendrycks",
            }
        )
    return examples


def _build_adapt_examples(
    input_path: Path,
    *,
    limit_mcq: int | None,
    limit_free: int | None,
    sample_seed: int,
    train_on_full_chat: bool,
    final_answer_only: bool,
    mcq_with_reasoning: bool,
    freeze_reasoning_style: bool,
    holdout_fraction: float,
    holdout_seed: int,
    holdout_output_path: Path | None,
    curated_input_path: Path | None = None,
) -> list[dict]:
    raw_data = _load_jsonl(input_path)
    supervised_full = [item for item in raw_data if item.get("answer") is not None]
    if not supervised_full:
        raise SystemExit("No supervised samples found. Training data must include `answer` fields.")

    holdout_supervised: list[dict] = []
    if holdout_fraction > 0:
        train_supervised, holdout_supervised = split_train_holdout(
            supervised_full,
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
        supervised = train_supervised
    else:
        supervised = supervised_full

    if curated_input_path is not None:
        if not curated_input_path.exists():
            raise SystemExit(f"Curated input not found: {curated_input_path}")
        curated = [item for item in _load_jsonl(curated_input_path) if item.get("answer") is not None]
        holdout_ids = {item.get("id") for item in holdout_supervised}
        curated = [item for item in curated if item.get("id") not in holdout_ids]
        if not curated:
            raise SystemExit(f"No curated training items after excluding holdout ids: {curated_input_path}")
        supervised = curated
        print(f"Using {len(supervised)} curated hard examples from {curated_input_path}")
    else:
        supervised = apply_subset_caps(
            supervised,
            limit_mcq=limit_mcq,
            limit_free=limit_free,
            seed=sample_seed,
        )

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
        is_mcq = bool(item.get("options"))
        if is_mcq:
            letter = _normalize_mcq_answer(item)
            if not letter:
                continue
            use_mcq_reasoning = mcq_with_reasoning
            if use_mcq_reasoning:
                target = build_mcq_reasoning_target(letter)
            else:
                target = f"\\boxed{{{letter}}}"
                if train_on_full_chat and not final_answer_only:
                    target = (
                        "Compute the answer and compare to options carefully.\n"
                        f"Final answer: \\boxed{{{letter}}}"
                    )
            prompt = build_adapt_train_mcq_user(
                item["question"],
                item["options"],
                with_reasoning=use_mcq_reasoning,
            )
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

        # Free-form: optional box-only labels. MCQ with reasoning keeps scaffold text in labels.
        if final_answer_only and not (is_mcq and mcq_with_reasoning):
            target = _enforce_single_final_boxed("", fallback_answer=extract_boxed(target))
        elif final_answer_only and is_mcq and mcq_with_reasoning:
            target = _enforce_single_final_boxed(target, fallback_answer=letter)

        examples.append(
            {
                "prompt": prompt,
                "target": _enforce_single_final_boxed(target),
                "system_prompt": system_prompt,
                "source": "competition_adapt",
            }
        )
    return examples


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


def _enable_gradient_checkpointing(model) -> None:
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled (lower VRAM during training).")
    elif hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()


def _load_base_model(model_id: str, *, load_in_4bit: bool, load_in_8bit: bool = False):
    """Load base model for LoRA training. Default path is bf16 (no bitsandbytes quant)."""
    from peft import prepare_model_for_kbit_training

    if load_in_8bit and not load_in_4bit:
        print("Loading base model in 8-bit (bitsandbytes)...")
        try:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                quantization_config=bnb_config,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            return prepare_model_for_kbit_training(model)
        except Exception as exc:
            print(f"8-bit load failed: {exc}")
            print("Falling back to bfloat16.")

    if load_in_4bit:
        print("Loading base model in 4-bit (bitsandbytes)...")
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                quantization_config=bnb_config,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            return prepare_model_for_kbit_training(model)
        except Exception as exc:
            print(f"4-bit load failed: {exc}")
            print("Falling back to bfloat16 weights (set --no-load-in-4bit to skip 4-bit attempt).")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Loading base model in {dtype} (no 4-bit quantization)...")
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _apply_stage_hparam_defaults(args) -> None:
    if args.stage == "reasoning":
        if args.learning_rate >= 2e-4:
            args.learning_rate = REASONING_DEFAULT_LEARNING_RATE
        if args.max_steps == 500:
            args.max_steps = REASONING_DEFAULT_MAX_STEPS
        if not args.train_on_full_chat:
            print("Stage `reasoning` enables --train-on-full-chat by default.")
            args.train_on_full_chat = True
        return

    if args.stage == "adapt":
        if args.learning_rate >= 2e-4:
            args.learning_rate = ADAPT_DEFAULT_LEARNING_RATE
        if args.max_steps == 500:
            args.max_steps = ADAPT_DEFAULT_MAX_STEPS
        if args.stage2_holdout_fraction is None:
            args.stage2_holdout_fraction = STAGE2_DEFAULT_HOLDOUT_FRACTION
        if args.stage2_mcq_with_reasoning is None:
            args.stage2_mcq_with_reasoning = STAGE2_MCQ_WITH_REASONING
        if args.max_seq_len == MAX_SEQ_LEN:
            args.max_seq_len = ADAPT_DEFAULT_MAX_SEQ_LEN
            print(f"Stage `adapt` using max_seq_len={ADAPT_DEFAULT_MAX_SEQ_LEN} (lower VRAM).")
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
        if getattr(args, "stage2_mcq_with_reasoning", STAGE2_MCQ_WITH_REASONING):
            print(
                "Stage `adapt` MCQ: supervising brief reasoning + \\boxed{letter} "
                "(not bare \\boxed{A} only)."
            )


def main() -> None:
    args = parse_train_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    _set_seed(args.seed)
    if args.load_in_4bit is None:
        args.load_in_4bit = TRAIN_LOAD_IN_4BIT
    if getattr(args, "gradient_checkpointing", None) is None:
        args.gradient_checkpointing = True
    _apply_stage_hparam_defaults(args)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    here = Path(__file__).resolve().parent
    root = _discover_project_root(here)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "reasoning":
        if not (args.include_openmath or args.include_hendrycks):
            raise SystemExit(
                "Stage `reasoning` requires at least one dataset. Pass --include-openmath and/or --include-hendrycks."
            )

        all_examples: list[dict] = []
        if args.include_openmath:
            print("Loading unsloth/OpenMathReasoning-mini (split=cot)")
            openmath_examples = _load_openmath_examples(
                max_examples=args.max_openmath_examples,
                seed=args.sample_seed,
            )
            print(f"OpenMath examples: {len(openmath_examples)}")
            all_examples.extend(openmath_examples)
        if args.include_hendrycks:
            print("Loading EleutherAI/hendrycks_math")
            hendrycks_examples = _load_hendrycks_examples(
                configs=args.hendrycks_configs,
                max_examples=args.max_hendrycks_examples,
                seed=args.sample_seed,
            )
            print(f"Hendrycks examples: {len(hendrycks_examples)}")
            all_examples.extend(hendrycks_examples)
        if not all_examples:
            raise SystemExit(
                "Stage `reasoning` produced no samples. Enable --include-openmath and/or --include-hendrycks."
            )
    else:
        input_path = resolve_input_path(args.input, root)
        if not input_path.exists():
            raise SystemExit(f"Input file not found: {input_path}")
        print(f"Loading competition adaptation data from {input_path}")
        holdout_path = output_dir / "stage2_holdout.jsonl"
        curated_path = None
        if args.curated_input:
            curated_path = Path(args.curated_input)
            if not curated_path.is_absolute():
                curated_path = root / curated_path
        all_examples = _build_adapt_examples(
            input_path,
            limit_mcq=args.limit_mcq,
            limit_free=args.limit_free,
            sample_seed=args.sample_seed,
            train_on_full_chat=args.train_on_full_chat,
            final_answer_only=args.stage2_final_answer_only,
            mcq_with_reasoning=args.stage2_mcq_with_reasoning,
            freeze_reasoning_style=args.stage2_freeze_reasoning_style,
            holdout_fraction=float(args.stage2_holdout_fraction),
            holdout_seed=args.stage2_holdout_seed,
            holdout_output_path=holdout_path,
            curated_input_path=curated_path,
        )
        print(f"Adaptation examples: {len(all_examples)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    from peft import LoraConfig, PeftModel, get_peft_model

    load_in_8bit = bool(getattr(args, "load_in_8bit", False))
    base_model = _load_base_model(
        MODEL_ID,
        load_in_4bit=bool(args.load_in_4bit),
        load_in_8bit=load_in_8bit,
    )
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

    if args.gradient_checkpointing:
        _enable_gradient_checkpointing(model)

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
    use_bnb_optim = bool(getattr(args, "use_bnb_optimizer", False))
    optimizer = None
    if use_bnb_optim:
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
        except Exception as exc:
            print(f"bitsandbytes optimizer failed ({exc}); using torch.optim.AdamW")
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=args.weight_decay,
        )
        print("Using torch.optim.AdamW on trainable LoRA params only")

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

    def _maybe_run_holdout_eval(adapter_dir: Path, step: int) -> None:
        if args.val_eval_every_steps <= 0 or args.stage != "adapt":
            return
        holdout_file = output_dir / "stage2_holdout.jsonl"
        if not holdout_file.is_file():
            return
        eval_script = here / "eval_runner.py"
        report_path = output_dir / f"val_eval_step_{step}.json"
        cmd = [
            sys.executable,
            str(eval_script),
            "--input",
            str(holdout_file),
            "--lora-adapter-path",
            str(adapter_dir),
            "--split-name",
            "val",
            "--gpu-id",
            args.gpu_id,
            "--eval-report",
            str(report_path),
            "--limit-mcq",
            str(min(15, args.val_eval_max_items)),
            "--limit-free",
            str(min(15, args.val_eval_max_items)),
            "--no-save-raw-output",
        ]
        print(f"Running holdout validation eval (step {step}): {' '.join(cmd)}")
        subprocess.run(cmd, check=False)

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
                if args.val_eval_every_steps > 0 and global_step % args.val_eval_every_steps == 0:
                    _maybe_run_holdout_eval(ckpt_dir, global_step)

            if global_step >= args.max_steps:
                break
    pbar.close()

    final_adapter_dir = output_dir / "final_adapter"
    final_adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)
    print(f"Saved final adapter to {final_adapter_dir}")
    if args.val_eval_every_steps > 0 and args.stage == "adapt":
        _maybe_run_holdout_eval(final_adapter_dir, global_step)

    if args.save_final_merged:
        merged_dir = output_dir / "merged_model"
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"Saved merged full model to {merged_dir}")

    config_path = output_dir / "train_config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    print(f"Saved train config to {config_path}")


if __name__ == "__main__":
    main()
