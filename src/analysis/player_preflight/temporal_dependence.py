#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Temporal dependence analysis: seasons_back and decay parameter calibration.
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

from src.utils.players import compute_per36


def autocorr_year_to_year(df: pd.DataFrame, min_minutes: float = 200.0) -> float:
    """Calculate year-to-year autocorrelation of per36 (for players with sufficient minutes).
    
    Returns:
        Pearson correlation between per36(t) and per36(t+1)
    """
    per36, minutes = compute_per36(df)
    z = df.assign(per36=per36, minutes=minutes).sort_values(["bioID", "year"]).copy()
    z = z[z["minutes"] >= min_minutes]
    z["per36_prev"] = z.groupby("bioID")["per36"].shift(1)
    m = z[["per36", "per36_prev"]].dropna()
    if len(m) < 3:
        return np.nan
    return float(m["per36"].corr(m["per36_prev"]))


def r2_vs_seasons_back(df: pd.DataFrame, decays: List[float], ks: List[int], min_minutes: float, out_png: Path, out_csv: Path) -> pd.DataFrame:
    """Walk-forward validation: test different (decay, k) combinations.
    
    For each combination, predict per36(t+1) using weighted average of up to k past seasons.
    
    Returns:
        DataFrame with columns: decay, k, r2, n
    """
    per36, minutes = compute_per36(df)
    base = df.assign(per36=per36, minutes=minutes).sort_values(["bioID", "year"]).copy()

    rows = []
    for decay in decays:
        for k in ks:
            preds = []
            trues = []
            for pid, g in base.groupby("bioID"):
                vals = g["per36"].values
                mins = g["minutes"].values
                for i in range(1, len(g) - 1):  # predict t+1 using up to k years up to t
                    # build weighted average of up to k seasons ending at i
                    weights = []
                    values = []
                    for back in range(0, min(k, i) + 1):
                        idx = i - back
                        w = (decay ** back) * (mins[idx] if mins[idx] > 0 else 0.0)
                        weights.append(w)
                        values.append(vals[idx])
                    if np.sum(weights) > 0:
                        pred = float(np.average(values, weights=weights))
                        nxt = float(vals[i + 1])
                        if np.isfinite(pred) and np.isfinite(nxt):
                            preds.append(pred)
                            trues.append(nxt)
            if len(preds) >= 5:
                y = np.asarray(trues)
                x = np.asarray(preds)
                y_mean = y.mean()
                sse = float(((y - x) ** 2).sum())
                sst = float(((y - y_mean) ** 2).sum())
                r2 = 1.0 - (sse / sst) if sst > 0 else np.nan
                n = int(len(y))
            else:
                r2 = np.nan
                n = 0
            rows.append({"decay": decay, "k": k, "r2": r2, "n": n})

    res = pd.DataFrame(rows)
    res.to_csv(out_csv, index=False)
    
    # Plot best r2 per k
    best = res.sort_values(["k", "r2"], ascending=[True, False]).groupby("k").head(1)
    plt.figure(figsize=(7, 4.5))
    plt.plot(best["k"], best["r2"], marker="o")
    plt.xlabel("seasons_back (k)")
    plt.ylabel("Best R^2 vs t+1 per36")
    plt.title("Marginal benefit of more seasons")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    return res


def optimize_decay_k(df: pd.DataFrame, out_png: Path, out_csv: Path, out_meta: Path) -> Tuple[int, float, float, int]:
    """Find best (k, decay) combination via walk-forward validation.
    
    Returns:
        (k_best, decay_best, r2_best, n)
    """
    decays = [round(x, 2) for x in np.arange(0.4, 0.91, 0.05)]
    ks = list(range(1, 6))
    res = r2_vs_seasons_back(df, decays=decays, ks=ks, min_minutes=0.0, out_png=out_png, out_csv=out_csv)
    
    if res.empty or res["r2"].isna().all():
        return 3, 0.65, np.nan, 0
    
    row = res.sort_values(["r2", "n"], ascending=[False, False]).head(1).squeeze()
    k_best = int(row["k"]) if "k" in row else 3
    d_best = float(row["decay"]) if "decay" in row else 0.65
    r2_best = float(row["r2"]) if "r2" in row else np.nan
    n = int(row["n"]) if "n" in row else 0
    
    out_meta.write_text(
        f"best_k={k_best}\nbest_decay={d_best}\nbest_r2={r2_best:.3f}\nn={n}\n",
        encoding="utf-8",
    )
    return k_best, d_best, r2_best, n

