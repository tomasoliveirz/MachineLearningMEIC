#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Per-36 stability analysis and rookie minutes threshold calibration.
"""

import sys
from pathlib import Path
from typing import List, Tuple

# Allow running as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.players import compute_per36, label_rookies, per36_next_year


def plot_per36_vs_minutes(df: pd.DataFrame, out_path: Path) -> None:
    """Plot per-36 vs minutes with binned mean and bootstrap CI."""
    per36, minutes = compute_per36(df)
    x = minutes.clip(upper=np.nanpercentile(minutes, 99))
    y = per36.clip(upper=np.nanpercentile(per36, 99))
    mask = x.notna() & y.notna()
    xv = x[mask].values
    yv = y[mask].values

    plt.figure(figsize=(10, 6))
    plt.scatter(xv, yv, s=10, alpha=0.25)

    # binned smooth (quantile bins) with bootstrap CI
    if len(xv) >= 50:
        q = np.linspace(0, 1, 21)
        edges = np.unique(np.quantile(xv, q))
        mids = []
        means = []
        los = []
        his = []
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            sel = (xv >= lo) & (xv < hi)
            if sel.sum() >= 10:
                mids.append((lo + hi) / 2)
                ys = yv[sel]
                means.append(float(np.nanmean(ys)))
                # bootstrap 95% CI
                if len(ys) >= 20:
                    rng = np.random.default_rng(42)
                    boots = []
                    for _ in range(500):
                        idx = rng.integers(0, len(ys), size=len(ys))
                        boots.append(float(np.nanmean(ys[idx])))
                    lo_ci, hi_ci = np.percentile(boots, [2.5, 97.5])
                else:
                    lo_ci = np.nan
                    hi_ci = np.nan
                los.append(lo_ci)
                his.append(hi_ci)
        if mids:
            plt.plot(mids, means, color="red", lw=2, label="binned mean")
            try:
                plt.fill_between(mids, los, his, color="red", alpha=0.15, label="95% CI")
            except Exception:
                pass

    plt.axvline(12.0, color="orange", ls="--", lw=1, label="12 min floor")
    plt.xlabel("Minutes played (season)")
    plt.ylabel("Per-36 composite rate")
    plt.title("Per-36 stability vs minutes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def choose_minutes_threshold_by_rmse(df: pd.DataFrame, candidates: List[int]) -> Tuple[int, float]:
    """Choose rookie_min_minutes threshold by minimizing RMSE vs next-year per36 for rookies.
    
    Returns:
        (best_threshold, best_rmse)
    """
    per36, minutes = compute_per36(df)
    z = df.assign(per36=per36, minutes=minutes).copy()
    z["per36_next"] = per36_next_year(z)
    z["rookie"] = label_rookies(z)
    rook = z[z["rookie"]].dropna(subset=["per36", "per36_next"]).copy()
    
    best_m = None
    best_rmse = np.inf
    for m in candidates:
        g = rook[rook["minutes"] >= m]
        if len(g) < 25:
            continue
        err = (g["per36"] - g["per36_next"]).values
        rmse = float(np.sqrt(np.mean(err ** 2)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_m = m
    
    if best_m is None:
        return 150, np.nan
    return best_m, best_rmse

