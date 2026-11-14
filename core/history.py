"""Helpers for results/training_history.csv (one file, one row per epoch of every stage)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def reset_stage(path: Path, stage_prefix: str):
    """Drop earlier rows of a stage before it is re-run from scratch."""
    path = Path(path)
    if not path.exists():
        return
    df = pd.read_csv(path)
    df = df[~df["stage"].astype(str).str.startswith(stage_prefix)]
    df.to_csv(path, index=False)
