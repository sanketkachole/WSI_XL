#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


# -------------------------
# Data containers
# -------------------------
@dataclass
class CompressionPoint:
    patches: int
    ratio: float
    cindex: float

    @property
    def retention_pct(self) -> float:
        # will be computed later relative to upper bound
        raise NotImplementedError


# -------------------------
# Utilities
# -------------------------
def safe_read_json(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def save_table(df: pd.DataFrame, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)


# -------------------------
# Figures
# -------------------------
def fig_compression_techniques(
    df: pd.DataFrame, out_path: Path, title: str = "Compression technique comparison (C-index)"
) -> None:
    """
    Expects df columns: method, cindex
    """
    df = df.sort_values("cindex", ascending=False).reset_index(drop=True)

    plt.figure()
    plt.bar(df["method"], df["cindex"])
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("C-index")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def fig_compression_ratio_retention(
    df: pd.DataFrame,
    out_path: Path,
    title: str = "Retention vs Compression Ratio (C-index retention)",
) -> None:
    """
    Expects df columns: ratio, retention_pct, method
    """
    plt.figure()
    for method, sub in df.groupby("method"):
        sub = sub.sort_values("ratio")
        plt.plot(sub["ratio"], sub["retention_pct"], marker="o", label=method)

    plt.xscale("log", base=2)
    plt.xlabel("Compression ratio (log2)")
    plt.ylabel("Retention (%)")
    plt.ylim(0, 105)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def fig_curriculum_bar(
    df: pd.DataFrame, out_path: Path, title: str = "Curriculum compression ablation (C-index)"
) -> None:
    """
    Expects df columns: setting, cindex
    """
    df = df.sort_values("cindex", ascending=False).reset_index(drop=True)

    plt.figure()
    plt.bar(df["setting"], df["cindex"])
    plt.ylim(0.0, 1.0)
    plt.ylabel("C-index")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def fig_cancer_type_grouped(
    df: pd.DataFrame,
    out_path: Path,
    title: str = "Cancer-type C-index (LUAD / BRCA / COAD)",
) -> None:
    """
    Expects df in long format: model, cancer, cindex
    """
    cancers = ["LUAD", "BRCA", "COAD"]
    df = df[df["cancer"].isin(cancers)].copy()

    pivot = df.pivot_table(index="model", columns="cancer", values="cindex", aggfunc="mean")
    pivot = pivot[cancers]  # enforce order

    plt.figure()
    x = range(len(pivot.index))
    width = 0.25

    for i, cancer in enumerate(cancers):
        offsets = [xi + (i - 1) * width for xi in x]
        plt.bar(offsets, pivot[cancer].values, width=width, label=cancer)

    plt.xticks(list(x), list(pivot.index), rotation=25, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("C-index")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


# -------------------------
# Example “dummy” inputs (replace with parsed real ones)
# -------------------------
def dummy_tables() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # B: compression techniques
    df_comp = pd.DataFrame(
        {
            "method": ["Random Pooling", "Grid Sampling", "Attention Top-K", "Fixed Chunking", "Dynamic (Ours)"],
            "cindex": [0.66, 0.67, 0.69, 0.71, 0.74],
        }
    )

    # C: compression ratio variants (retention)
    upper = 0.76
    points = [
        ("Upper bound", 1, 0.76),
        ("Dynamic (Ours)", 2, 0.75),
        ("Dynamic (Ours)", 4, 0.74),
        ("Dynamic (Ours)", 8, 0.72),
        ("Dynamic (Ours)", 16, 0.69),
        ("Pooling", 2, 0.70),
        ("Pooling", 4, 0.68),
        ("Pooling", 8, 0.65),
        ("Pooling", 16, 0.61),
        ("Q-former", 2, 0.71),
        ("Q-former", 4, 0.69),
        ("Q-former", 8, 0.66),
        ("Q-former", 16, 0.63),
    ]
    df_ratio = pd.DataFrame(points, columns=["method", "ratio", "cindex"])
    df_ratio["retention_pct"] = 100.0 * (df_ratio["cindex"] / upper)

    # D: curriculum
    df_curr = pd.DataFrame({"setting": ["No curriculum", "Curriculum"], "cindex": [0.71, 0.74]})

    # A2: cancer-specific
    df_cancer = pd.DataFrame(
        {
            "model": ["CoxPH", "TransMIL", "Late Fusion", "WSI-XL (Ours)"] * 3,
            "cancer": ["LUAD"] * 4 + ["BRCA"] * 4 + ["COAD"] * 4,
            "cindex": [0.62, 0.69, 0.72, 0.77, 0.65, 0.71, 0.74, 0.78, 0.60, 0.68, 0.70, 0.74],
        }
    )
    return df_comp, df_ratio, df_curr, df_cancer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, default=Path("paper_figures"))
    ap.add_argument("--use_dummy", action="store_true", help="Generate figures from dummy tables.")
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.use_dummy:
        df_comp, df_ratio, df_curr, df_cancer = dummy_tables()
    else:
        raise SystemExit(
            "Real log parsing is in a separate script below (parse_runs.py). "
            "Run parse_runs.py first to produce CSV tables, then load them here."
        )

    # Save tables (paper artifacts)
    save_table(df_comp, out_dir / "table_B1_compression_techniques.csv")
    save_table(df_ratio, out_dir / "table_C1_ratio_retention.csv")
    save_table(df_curr, out_dir / "table_D1_curriculum.csv")
    save_table(df_cancer, out_dir / "table_A2_cancer_specific.csv")

    # Figures
    fig_compression_techniques(df_comp, out_dir / "fig_B1_compression_techniques.png")
    fig_compression_ratio_retention(df_ratio, out_dir / "fig_C1_ratio_retention.png")
    fig_curriculum_bar(df_curr, out_dir / "fig_D1_curriculum.png")
    fig_cancer_type_grouped(df_cancer, out_dir / "fig_A2_cancer_specific.png")

    print(f"[OK] Wrote tables + figures to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
