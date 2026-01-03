#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("make_kfold_splits")


def setup_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), None)
    if not isinstance(numeric, int):
        raise ValueError(f"Invalid log level: {level}")
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _round_robin_stratified_indices(y: np.ndarray, k: int, seed: int) -> List[np.ndarray]:
    """
    Simple stratified splitter (no sklearn):
    - shuffle indices within each class
    - distribute to folds round-robin
    Works best for binary / multiclass single-label.
    """
    rng = np.random.default_rng(seed)
    folds: List[List[int]] = [[] for _ in range(k)]

    classes = np.unique(y)
    for c in classes:
        idx = np.where(y == c)[0].tolist()
        rng.shuffle(idx)
        for j, ii in enumerate(idx):
            folds[j % k].append(ii)

    # shuffle within each fold for randomness
    out: List[np.ndarray] = []
    for f in folds:
        rng.shuffle(f)
        out.append(np.array(f, dtype=np.int64))
    return out


def _plain_kfold_indices(n: int, k: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n, dtype=np.int64)
    rng.shuffle(idx)
    return [arr for arr in np.array_split(idx, k)]


def _write_split(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def make_folds(
    df: pd.DataFrame,
    k: int,
    seed: int,
    stratify_col: Optional[str],
) -> List[pd.DataFrame]:
    if k < 2:
        raise ValueError("k must be >= 2")
    n = len(df)
    if n < k:
        raise ValueError(f"Not enough rows ({n}) for k={k}")

    if stratify_col is not None:
        if stratify_col not in df.columns:
            raise ValueError(f"--stratify_col '{stratify_col}' not found in CSV columns")
        y = df[stratify_col].to_numpy()
        folds_idx = _round_robin_stratified_indices(y=y, k=k, seed=seed)
    else:
        folds_idx = _plain_kfold_indices(n=n, k=k, seed=seed)

    folds = [df.iloc[idx].reset_index(drop=True) for idx in folds_idx]
    return folds


def main() -> None:
    p = argparse.ArgumentParser(description="Create k-fold train/val/test CSVs from patients_breast.csv")
    p.add_argument("--meta_csv", type=str, required=True, help="Path to patients_breast.csv")
    p.add_argument("--out_dir", type=str, required=True, help="Directory to write splits/ folder into")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)

    # what your training script needs
    p.add_argument(
        "--label_cols",
        type=str,
        required=True,
        help="Comma-separated label columns to include in split CSVs (e.g. survival_event or survival_event,survival_time_days)",
    )

    # stratify (recommended for binary classification)
    p.add_argument(
        "--stratify_col",
        type=str,
        default="",
        help="Optional column for stratification (e.g. survival_event). Leave empty to disable.",
    )

    p.add_argument("--log_level", type=str, default="INFO")
    args = p.parse_args()

    setup_logging(args.log_level)

    meta_csv = Path(args.meta_csv)
    out_dir = Path(args.out_dir)
    k = int(args.k)
    seed = int(args.seed)

    label_cols = [c.strip() for c in args.label_cols.split(",") if c.strip()]
    if not label_cols:
        raise ValueError("--label_cols cannot be empty")

    stratify_col = args.stratify_col.strip() or None

    df = pd.read_csv(meta_csv)

    # validate required columns
    required_base = ["patient_id", "h5_path", "text_row"]
    missing = [c for c in required_base + label_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in meta CSV: {missing}")

    # create case_id expected by your training dataset
    df = df.copy()
    df["case_id"] = df["patient_id"].astype(str)

    # keep only what model loader expects (+ optional extras are fine, but keep it clean)
    keep_cols = ["case_id", "h5_path", "text_row"] + label_cols
    df = df[keep_cols].copy()

    # drop rows with missing labels/text_row/h5_path
    df = df.dropna(subset=["h5_path", "text_row"] + label_cols).reset_index(drop=True)

    LOGGER.info("Loaded %d rows after dropna", len(df))
    if stratify_col is not None:
        LOGGER.info("Stratifying by: %s", stratify_col)
        LOGGER.info("Stratify value counts:\n%s", df[stratify_col].value_counts(dropna=False))

    folds = make_folds(df=df, k=k, seed=seed, stratify_col=stratify_col)

    # For each fold i:
    #   test = fold i
    #   val  = fold (i+1) % k
    #   train = remaining folds
    splits_root = out_dir / "splits"
    for i in range(k):
        test_df = folds[i]
        val_df = folds[(i + 1) % k]
        train_df = pd.concat([folds[j] for j in range(k) if j not in (i, (i + 1) % k)], axis=0).reset_index(drop=True)

        fold_dir = splits_root / f"fold_{i:02d}"
        _write_split(train_df, fold_dir / "train.csv")
        _write_split(val_df, fold_dir / "val.csv")
        _write_split(test_df, fold_dir / "test.csv")

        LOGGER.info(
            "fold_%02d: train=%d val=%d test=%d",
            i, len(train_df), len(val_df), len(test_df)
        )

    LOGGER.info("Done. Wrote folds to: %s", splits_root)


if __name__ == "__main__":
    main()








