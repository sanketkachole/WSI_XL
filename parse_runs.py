#!/usr/bin/env python3
'''
You already have:
    runs/fold_XX/final_metrics.json
    runs/fold_XX/config.json
    logs/wsixl_*.err (contains VAL/TEST lines)

So we’ll parse those into:
    Table A1 (overall mean across folds)
    Fold-wise table (for error bars if you want later)
    “Best checkpoint C-index” if present
    Anything else in final_metrics.json

run this script in bigred at location "(WSIXL) (WSIXL) skachole@login1:/N/project/Sanket_Slate_Project/4_modelling/MICCAI_Slide_XL" : parse_runs.py
'''
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


VAL_RE = re.compile(r"\bVAL\b.*?\bacc\b=\s*([0-9.]+)")
TEST_RE = re.compile(r"\bTEST\b.*?\bacc\b=\s*([0-9.]+)")
# If you later switch logging from "acc" to "cindex", update regex accordingly:
CINDEX_RE = re.compile(r"\b(cindex|c-index)\b\s*=\s*([0-9.]+)", re.IGNORECASE)


def safe_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def parse_err_for_metrics(err_path: Path) -> Dict[str, float]:
    """
    Extracts last seen VAL/TEST metrics from your .err log.
    Right now your script prints acc=...; for Cox you should log cindex=...
    This parser supports both patterns.
    """
    text = err_path.read_text(errors="ignore")

    out: Dict[str, float] = {}

    # Prefer explicit cindex if present anywhere
    c_all = CINDEX_RE.findall(text)
    if c_all:
        # keep last one
        out["cindex_last"] = float(c_all[-1][1])

    # Otherwise fall back to your current "acc" tokens
    val = VAL_RE.findall(text)
    if val:
        out["val_metric_last"] = float(val[-1])
    test = TEST_RE.findall(text)
    if test:
        out["test_metric_last"] = float(test[-1])

    return out


def collect_runs(runs_root: Path) -> pd.DataFrame:
    rows = []

    for fold_dir in sorted(runs_root.glob("fold_*")):
        if not fold_dir.is_dir():
            continue

        config = safe_json(fold_dir / "config.json") or {}
        finalm = safe_json(fold_dir / "final_metrics.json") or {}

        row = {"fold": fold_dir.name}
        # Bring config keys you care about into the table
        for k in [
            "task_type",
            "label_cols",
            "d_v",
            "d_t",
            "d_model",
            "n_layers",
            "n_heads",
            "chunk_size",
            "max_report_len",
            "lr",
            "epochs",
            "batch_size",
        ]:
            if k in config:
                row[k] = config[k]

        # Bring final metrics into the table
        for k, v in finalm.items():
            if isinstance(v, (int, float, str)):
                row[f"final_{k}"] = v

        rows.append(row)

    return pd.DataFrame(rows)


def collect_logs(logs_dir: Path) -> pd.DataFrame:
    rows = []
    for err in sorted(logs_dir.glob("*.err")):
        m = re.search(r"_(\d+)_([0-9]+)\.err$", err.name)
        jobid = m.group(1) if m else ""
        taskid = m.group(2) if m else ""

        metrics = parse_err_for_metrics(err)
        rows.append(
            {
                "log_file": err.name,
                "jobid": jobid,
                "taskid": taskid,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=Path, required=True, help="e.g., /.../MICCAI_Slide_XL/runs")
    ap.add_argument("--logs_dir", type=Path, required=True, help="e.g., /.../MICCAI_Slide_XL/logs")
    ap.add_argument("--out_dir", type=Path, default=Path("paper_tables"))
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df_runs = collect_runs(args.runs_root)
    df_logs = collect_logs(args.logs_dir)

    # Save raw extracted tables
    df_runs.to_csv(out_dir / "runs_summary.csv", index=False)
    df_logs.to_csv(out_dir / "logs_summary.csv", index=False)

    # Produce a simple overall table (mean across folds if available)
    overall = {}
    # if your final_metrics.json contains test cindex later, it’ll appear as final_test_cindex etc.
    numeric_cols = [c for c in df_runs.columns if c.startswith("final_")]
    for c in numeric_cols:
        try:
            overall[c] = float(pd.to_numeric(df_runs[c], errors="coerce").mean())
        except Exception:
            pass
    df_overall = pd.DataFrame([overall]) if overall else pd.DataFrame()

    if not df_overall.empty:
        df_overall.to_csv(out_dir / "overall_means.csv", index=False)

    print(f"[OK] Wrote: {out_dir.resolve()}")
    print(" - runs_summary.csv (from runs/*/config.json + final_metrics.json)")
    print(" - logs_summary.csv (from logs/*.err)")
    if not df_overall.empty:
        print(" - overall_means.csv")


if __name__ == "__main__":
    main()
