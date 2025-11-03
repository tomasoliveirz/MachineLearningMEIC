#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rookie prior calibration: shrinkage strength and team vs global priors.
"""

import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

# Allow running as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.players import compute_per36, label_rookies, per36_next_year


def shrink_rookie(per36_obs: float, minutes_obs: float, prior_mean: float, prior_strength_minutes: float) -> float:
    """Apply Bayesian shrinkage to rookie observed per36 toward a prior.
    
    Args:
        per36_obs: Observed per-36 rate
        minutes_obs: Minutes played (determines weight of observation)
        prior_mean: Prior mean (e.g., league-average rookie per36)
        prior_strength_minutes: Strength of prior in equivalent minutes
        
    Returns:
        Shrunk per36 estimate
    """
    if not np.isfinite(per36_obs):
        return prior_mean
    w_obs = max(minutes_obs, 0.0) / 36.0
    w_prior = max(prior_strength_minutes, 0.0) / 36.0
    denom = w_obs + w_prior
    if denom <= 0:
        return prior_mean
    return float((w_obs * per36_obs + w_prior * prior_mean) / denom)


def calibrate_rookie_prior(df: pd.DataFrame, strengths: Iterable[float], out_png: Path, out_csv: Path) -> pd.DataFrame:
    """Grid-search rookie prior strength using global prior from previous seasons (temporal LOO).

    Metrics: correlation and RMSE vs next-season per36 for rookies.
    Saves detailed grid as CSV and plot as PNG.
    
    Returns:
        DataFrame with columns: prior_strength, corr, rmse, n
    """
    per36, minutes = compute_per36(df)
    z = df.copy()
    z["is_rookie"] = label_rookies(z)
    z["per36"] = per36
    z["minutes"] = minutes
    target_next = per36_next_year(z)

    # Build global rookie prior by year using previous 3 seasons (exclude current season rookies)
    rook = z[z["is_rookie"]].copy()
    prior_by_year: Dict[int, float] = {}
    for y, g in rook.groupby("year"):
        prev_mask = (rook["year"] < y) & (rook["year"] >= y - 3)
        prev_vals = rook.loc[prev_mask, "per36"].astype(float)
        val = float(prev_vals.mean()) if prev_vals.notna().any() else float(rook.loc[rook["year"] < y, "per36"].mean())
        prior_by_year[int(y)] = val

    rows = []
    for s in strengths:
        preds = []
        trues = []
        for idx, r in rook.iterrows():
            y = int(r["year"]) if np.isfinite(r["year"]) else None
            obs = r["per36"]
            mins = r["minutes"]
            if y is None:
                continue
            prior_mean = prior_by_year.get(y, float(rook.loc[rook["year"] < y, "per36"].mean()))
            pred = shrink_rookie(float(obs), float(mins), float(prior_mean), float(s))
            true_ny = target_next.at[idx]
            if np.isfinite(pred) and np.isfinite(true_ny):
                preds.append(pred)
                trues.append(true_ny)
        if len(preds) >= 5:
            y_arr = np.asarray(trues, dtype=float)
            x_arr = np.asarray(preds, dtype=float)
            corr = float(np.corrcoef(x_arr, y_arr)[0, 1])
            rmse = float(np.sqrt(np.mean((x_arr - y_arr) ** 2)))
            n = int(len(y_arr))
        else:
            corr, rmse, n = np.nan, np.nan, 0
        rows.append({"prior_strength": float(s), "corr": corr, "rmse": rmse, "n": n})

    res = pd.DataFrame(rows).sort_values("prior_strength")
    res.to_csv(out_csv, index=False)

    plt.figure(figsize=(7.5, 4.8))
    ax1 = plt.gca()
    ax1.plot(res["prior_strength"], res["corr"], marker="o", color="#1f77b4", label="corr")
    ax1.set_xscale("log")
    ax1.set_ylabel("Correlation (rookies)")
    ax2 = ax1.twinx()
    ax2.plot(res["prior_strength"], res["rmse"], marker="s", color="#d62728", label="rmse")
    ax2.set_ylabel("RMSE vs next-year per36")
    ax1.set_xlabel("Rookie prior strength (equivalent minutes, log scale)")
    plt.title("Rookie prior calibration (global prior, temporal LOO)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    return res


def team_rookie_prior_loo(df: pd.DataFrame, per36: pd.Series) -> Dict[str, Tuple[float, int]]:
    """Return dict of team rookie mean per36 excluding the current rookie row when used.

    Implementation: compute simple team-year rookie mean; LOO will be applied at use-time by
    adjusting the team total and count if the row is itself a rookie.
    
    Returns:
        Dict with keys "year|tmID" -> (sum, count)
    """
    z = df.copy()
    z["is_rookie"] = label_rookies(z)
    rook = z[z["is_rookie"]].copy()
    rook_per36 = per36.loc[rook.index]
    grp = rook.assign(per36=rook_per36).groupby(["year", "tmID"])  # team-year aggregation
    sums = grp["per36"].sum(min_count=1)
    cnts = grp.size()
    return {
        f"{y}|{t}": (float(sums.loc[(y, t)]) if (y, t) in sums.index else np.nan, 
                     int(cnts.loc[(y, t)]) if (y, t) in cnts.index else 0)
        for (y, t) in set(zip(rook["year"], rook["tmID"]))
    }


def compare_team_vs_global_prior(df: pd.DataFrame, strength: float, out_png: Path) -> Tuple[float, float]:
    """Compare team-based vs global rookie priors.
    
    Returns:
        (mse_team, mse_global)
    """
    per36, minutes = compute_per36(df)
    df = df.copy()
    df["is_rookie"] = label_rookies(df)
    target_next = per36_next_year(df)
    team_prior_map = team_rookie_prior_loo(df, per36)
    global_prior = float(per36[df["is_rookie"]].mean())

    errs_team = []
    errs_global = []
    for idx, r in df[df["is_rookie"]].iterrows():
        y = int(r["year"]) if np.isfinite(r["year"]) else None
        t = str(r["tmID"]) if pd.notna(r["tmID"]) else None
        obs = per36.at[idx]
        mins = minutes.at[idx]
        # team LOO
        if y is not None and t is not None and f"{y}|{t}" in team_prior_map:
            sum_rt, cnt_rt = team_prior_map[f"{y}|{t}"]
            if np.isfinite(obs):
                prior_team = (sum_rt - obs) / max(cnt_rt - 1, 1)
            else:
                prior_team = sum_rt / max(cnt_rt, 1)
        else:
            prior_team = global_prior
        pred_team = shrink_rookie(obs, float(mins), float(prior_team), float(strength))
        pred_global = shrink_rookie(obs, float(mins), float(global_prior), float(strength))
        true_ny = target_next.at[idx]
        if np.isfinite(true_ny):
            if np.isfinite(pred_team):
                errs_team.append((pred_team - true_ny) ** 2)
            if np.isfinite(pred_global):
                errs_global.append((pred_global - true_ny) ** 2)

    mse_team = float(np.mean(errs_team)) if errs_team else np.nan
    mse_global = float(np.mean(errs_global)) if errs_global else np.nan

    plt.figure(figsize=(6, 4))
    vals = [mse_team, mse_global]
    plt.bar(["team LOO", "global"], vals)
    plt.ylabel("MSE vs next-season per36 (rookies)")
    plt.title("Team prior vs Global prior")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    return mse_team, mse_global

