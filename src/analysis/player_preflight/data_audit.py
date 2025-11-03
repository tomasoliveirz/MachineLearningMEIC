#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data quality audit: missingness, duplicates, ranges, outliers.
"""

import sys
from pathlib import Path
from typing import List

# Allow running as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.players import compute_per36


def write_audit_summary(df_stints: pd.DataFrame, df_agg: pd.DataFrame, meta_path: Path) -> None:
    """Write data quality summary: duplicates, zero rows, units, ranges."""
    lines = []
    
    # Key uniqueness after aggregating
    dup_mask = df_agg.duplicated(subset=["bioID", "year", "tmID"], keep=False)
    n_dup = int(dup_mask.sum())
    lines.append(f"(bioID, year, tmID) duplicates after stint-aggregation: {n_dup}")

    # Zero/empty lines
    stat_cols = [c for c in ["minutes", "points", "rebounds", "assists", "steals", "blocks", "turnovers"] if c in df_agg.columns]
    zero_rows = df_agg[stat_cols].fillna(0).sum(axis=1).eq(0).sum()
    lines.append(f"Rows with all-zero core stats: {int(zero_rows)}")

    # Unit consistency
    has_mp = "minutes" in df_stints.columns
    has_g = "g" in df_stints.columns
    lines.append(f"Minutes source present: minutes={has_mp} | fallback g={has_g}")

    # Ranges
    def rng(s: pd.Series) -> str:
        s = pd.to_numeric(s, errors="coerce")
        return f"min={np.nanmin(s):.2f} | median={np.nanmedian(s):.2f} | p95={np.nanpercentile(s,95):.2f} | p99={np.nanpercentile(s,99):.2f}"
    for c in stat_cols:
        lines.append(f"{c}: {rng(df_agg[c])}")

    meta_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_missingness_heatmap(df: pd.DataFrame, columns: List[str], out_path: Path) -> None:
    """Plot year × columns heatmap showing % non-null data."""
    cols = [c for c in columns if c in df.columns]
    z = df[["year"] + cols].copy()
    # Compute non-null percentage per year for each column
    non_null = z.groupby("year")[cols].apply(lambda g: g.notna().mean() * 100.0)
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(non_null.T, cmap="Blues", vmin=0, vmax=100, cbar_kws={"label": "% non-null"})
    plt.title("Data availability by year (% non-null)")
    plt.xlabel("Year")
    plt.ylabel("Columns")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def write_yearly_coverage_table(df: pd.DataFrame, columns: List[str], out_csv: Path) -> None:
    """Write CSV with per-year coverage stats (non-null %, n, median, IQR)."""
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return
    
    def iqr(x: pd.Series) -> float:
        x = pd.to_numeric(x, errors="coerce")
        return float(np.nanpercentile(x, 75) - np.nanpercentile(x, 25))
    
    parts = []
    for y, g in df.groupby("year"):
        row = {"year": y}
        for c in cols:
            s = pd.to_numeric(g[c], errors="coerce")
            row[f"{c}_nonnull_pct"] = float((s.notna().mean()) * 100.0)
            row[f"{c}_n"] = int(s.notna().sum())
            row[f"{c}_median"] = float(np.nanmedian(s))
            row[f"{c}_iqr"] = iqr(s)
        parts.append(row)
    pd.DataFrame(parts).sort_values("year").to_csv(out_csv, index=False)


def plot_correlations(df: pd.DataFrame, out_png: Path, out_txt: Path) -> None:
    """Plot correlation matrix for per36 and key stats (using REAL column names)."""
    per36, minutes = compute_per36(df)
    d = pd.DataFrame({
        "per36": per36,
        "minutes": minutes,
        "points": pd.to_numeric(df.get("points", np.nan), errors="coerce"),
        "rebounds": pd.to_numeric(df.get("rebounds", np.nan), errors="coerce"),
        "assists": pd.to_numeric(df.get("assists", np.nan), errors="coerce"),
        "steals": pd.to_numeric(df.get("steals", np.nan), errors="coerce"),
        "blocks": pd.to_numeric(df.get("blocks", np.nan), errors="coerce"),
        "turnovers": pd.to_numeric(df.get("turnovers", np.nan), errors="coerce"),
    })
    # keep only columns with at least some data
    d = d.dropna(how="all", axis=1)
    c = d.corr(method="pearson")

    plt.figure(figsize=(7.5, 6.5))
    sns.heatmap(c, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation matrix (per-36 and stats)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

    out_txt.write_text(c.round(3).to_string() + "\n", encoding="utf-8")


def write_outliers_top20(df: pd.DataFrame, out_csv: Path) -> None:
    """Identify top-20 abs(z-score) outliers in per36 (by year)."""
    per36, _ = compute_per36(df)
    z = df.assign(per36=per36).copy()
    # z-score within year
    z["z"] = z.groupby("year")["per36"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) not in [0, np.nan] else np.nan
    )
    top = z.loc[
        z["z"].abs().sort_values(ascending=False).head(20).index,
        ["bioID", "year", "tmID", "minutes", "points", "per36", "z"]
    ]
    top.to_csv(out_csv, index=False)

