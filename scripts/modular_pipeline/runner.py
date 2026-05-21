"""Top-level orchestration for modular pipeline runs."""

import csv
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

from cli_utils import apply_subset_caps, build_run_stem, parse_args, resolve_input_path
from evaluation import evaluate_with_judger
from model_pipeline import ModularPipeline
from settings import FREE_BATCH_SIZE, MCQ_BATCH_SIZE


def _discover_project_root(start: Path) -> Path:
    """Find the project root containing `data/`, falling back safely."""
    for candidate in [start, *start.parents]:
        if (candidate / "data").exists():
            return candidate
    # Fallback for unexpected layouts: preserve previous behavior.
    return start.parent


def _write_records(
    file_obj,
    chunk: list[dict],
    solved_batch: list[dict],
    *,
    save_raw_output: bool = True,
) -> None:
    for item, solved in zip(chunk, solved_batch):
        rec = {
            "id": item.get("id"),
            "is_mcq": bool(item.get("options")),
            "response": solved["response"],
            "meta": solved["meta"],
        }
        if save_raw_output:
            rec["raw"] = solved.get("raw")
        file_obj.write(json.dumps(rec) + "\n")

    file_obj.flush()
    os.fsync(file_obj.fileno())


def _load_done_ids(output_path: Path) -> set:
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
    return done_ids


def _load_records_by_id(output_path: Path) -> dict:
    records_by_id: dict = {}
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
    return records_by_id


def main() -> None:
    args = parse_args()

    here = Path(__file__).resolve().parent
    root = _discover_project_root(here)

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    input_path = resolve_input_path(args.input, root)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    output_dir = Path(args.output_dir) if args.output_dir else root / "results"
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

    done_ids = _load_done_ids(output_path)
    print(f"Found {len(done_ids)} completed records in {output_path}")

    remaining_data = [item for item in data if item.get("id") not in done_ids]
    print(f"Remaining questions to solve: {len(remaining_data)}")

    if remaining_data:
        pipe = ModularPipeline(
            gpu_id=args.gpu_id,
            lora_adapter_path=args.lora_adapter_path,
            vllm_quantization=args.vllm_quantization,
            vllm_load_format=args.vllm_load_format,
            enforce_eager=True if args.vllm_enforce_eager else None,
            inference_backend=args.inference_backend,
            mcq_max_new_tokens=args.mcq_max_new_tokens,
            mcq_final_max_new_tokens=args.mcq_final_max_new_tokens,
            free_max_new_tokens=args.free_max_new_tokens,
        )
        mcq_items = [item for item in remaining_data if item.get("options")]
        free_items = [item for item in remaining_data if not item.get("options")]

        print(f"Remaining MCQ questions: {len(mcq_items)}")
        print(f"Remaining free-form questions: {len(free_items)}")

        with open(output_path, "a") as f:
            for start in tqdm(range(0, len(mcq_items), MCQ_BATCH_SIZE), desc="Solving MCQ batches"):
                chunk = mcq_items[start : start + MCQ_BATCH_SIZE]
                solved_batch = pipe.solve_mcq_batch(chunk)
                _write_records(
                    f,
                    chunk,
                    solved_batch,
                    save_raw_output=args.save_raw_output,
                )

            for start in tqdm(range(0, len(free_items), FREE_BATCH_SIZE), desc="Solving free-form batches"):
                chunk = free_items[start : start + FREE_BATCH_SIZE]
                solved_batch = pipe.solve_free_batch(chunk)
                _write_records(
                    f,
                    chunk,
                    solved_batch,
                    save_raw_output=args.save_raw_output,
                )

        print(f"Saved incremental outputs to {output_path.resolve()}")
    else:
        print("Nothing left to solve.")

    records_by_id = _load_records_by_id(output_path)
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

    with open(submission_path, "w", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["id", "response"])
        short_trace_count = 0
        for item in data:
            rec = records_by_id[item.get("id")]
            if args.submission_full_trace:
                meta = rec.get("meta") or {}
                trace = rec.get("raw") or meta.get("raw") or rec.get("response", "")
                n_tok = meta.get("n_tokens") or meta.get("total_n_tokens")
                if isinstance(n_tok, (int, float)) and n_tok < 32:
                    short_trace_count += 1
            else:
                trace = rec.get("response", "")
            response = str(trace).replace("\r\n", "\n").replace("\r", "\n")
            writer.writerow([rec["id"], response])
        if args.submission_full_trace:
            print(
                "Submission CSV uses full model traces (--submission-full-trace)."
            )
            if short_trace_count:
                print(
                    f"Warning: {short_trace_count} rows have very short traces (<32 tokens). "
                    "MCQ-only LoRA often emits only \\boxed{{X}}; use base or Stage-1 LoRA "
                    "for competition-style reasoning traces."
                )
    print(f"Saved submission CSV to {submission_path.resolve()}")

    if has_answers and not args.no_eval:
        evaluate_with_judger(data, records_by_id)
