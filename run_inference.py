#!/usr/bin/env python3
"""Competition entry point: ``run_inference()`` → ``private_submission.csv``.

Hyperparameters are fixed in ``scripts/modular_pipeline/settings.py``.

    from run_inference import run_inference
    submission_csv = run_inference()              # private (default)
    submission_csv = run_inference(input="public")  # optional override
"""

from __future__ import annotations

import sys
from pathlib import Path

_PIPELINE_DIR = Path(__file__).resolve().parent / "scripts" / "modular_pipeline"
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

from run_inference import run_inference  # noqa: E402

__all__ = ["run_inference"]


if __name__ == "__main__":
    from run_inference import main

    main()
