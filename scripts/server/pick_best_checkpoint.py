#!/usr/bin/env python3
"""Print best Stage-2 checkpoint from eval_runner sweep JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--stage2-root",
        default="workspaces/stage2_adapt",
        help="Stage 2 output directory containing holdout_checkpoint_sweep.json",
    )
    args = p.parse_args()
    root = Path(args.stage2_root)
    sweep_path = root / "holdout_checkpoint_sweep.json"
    if not sweep_path.is_file():
        raise SystemExit(f"Missing sweep report: {sweep_path}")

    csv_path = sweep_path.with_suffix(".csv")
    rows: list[dict] = []
    if csv_path.is_file():
        import csv

        with csv_path.open(newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        data = json.loads(sweep_path.read_text())
        for run in data.get("runs") or []:
            metrics = run.get("metrics") or {}
            if metrics.get("error"):
                continue
            rows.append(
                {
                    "checkpoint": run.get("checkpoint_label") or "",
                    "validation_accuracy_pct": metrics.get("validation_accuracy_pct", 0),
                }
            )
    if not rows:
        raise SystemExit(f"No checkpoint rows in {csv_path} or {sweep_path}")

    def acc(row: dict) -> float:
        for key in ("validation_accuracy_pct", "accuracy_pct", "accuracy"):
            if key in row and row[key] not in (None, ""):
                return float(row[key])
        return -1.0

    best = max(rows, key=acc)
    best_acc = acc(best)
    ckpt_name = best.get("checkpoint") or best.get("checkpoint_label") or ""
    adapter = root / ckpt_name if ckpt_name else None
    if adapter and not adapter.is_dir() and ckpt_name == "final_adapter":
        adapter = root / "final_adapter"
    print(f"Best holdout accuracy: {best_acc:.2f}%")
    print(f"Adapter path: {adapter}")
    link_path = root / "best_adapter.txt"
    if adapter:
        link_path.write_text(str(adapter) + "\n")
        print(f"Wrote {link_path}")


if __name__ == "__main__":
    main()
