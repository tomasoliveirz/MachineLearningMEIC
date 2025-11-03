#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Survival bias analysis and inverse probability weighting (IPW).
"""

import sys
from pathlib import Path

# Allow running as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd


def survival_curves(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate league-average survival probability by years since rookie.
    
    Returns:
        DataFrame with columns: years_since_rookie, survival
    """
    z = df.copy()
    z["rookie_year"] = z.groupby("bioID")["year"].transform("min")
    z["years_since_rookie"] = z["year"] - z["rookie_year"]

    # For each cohort year k, fraction still active (has a row at that k)
    counts = z.groupby(["rookie_year", "years_since_rookie"]).size().rename("n").reset_index()
    # Normalize per-cohort
    cohort_sizes = counts[counts["years_since_rookie"] == 0][["rookie_year", "n"]].rename(columns={"n": "n0"})
    m = counts.merge(cohort_sizes, on="rookie_year", how="left")
    m["survival"] = m["n"] / m["n0"].replace(0, np.nan)

    # Plot average across cohorts (to make a single clean curve)
    avg = m.groupby("years_since_rookie")["survival"].mean().reset_index()
    return avg


def simple_ipw_weights(df: pd.DataFrame) -> pd.Series:
    """Inverse-probability weights based on empirical survival by k.

    Weight for a row at k = 1 / P(active at k), where P is the league-average survival.
    
    Returns:
        Series of IPW weights (one per row), unclamped
    """
    z = df.copy()
    z["rookie_year"] = z.groupby("bioID")["year"].transform("min")
    z["k"] = z["year"] - z["rookie_year"]
    surv = survival_curves(df)
    surv_map = dict(zip(surv["years_since_rookie"], surv["survival"].replace(0, np.nan)))
    w = z["k"].map(lambda kk: 1.0 / surv_map.get(kk, np.nan))
    return w


def get_ipw_weights(df: pd.DataFrame, max_weight: float = 4.0) -> pd.Series:
    """Get clamped IPW weights suitable for modeling.
    
    This is the recommended function to use in models. It applies the raw IPW
    calculation but clamps extreme weights to avoid undue influence from
    small survival cohorts.
    
    Args:
        df: Player data
        max_weight: Maximum allowed weight (default: 4.0 based on preflight analysis)
        
    Returns:
        Series of clamped IPW weights
    """
    raw_weights = simple_ipw_weights(df)
    return raw_weights.clip(upper=max_weight)


def write_survival_weights(df: pd.DataFrame, out_csv: Path, out_meta: Path) -> None:
    """Write survival probability and IPW weights table, with warnings about extreme weights.
    
    Saves:
        - CSV with k, p_survival, ipw_weight
        - Meta file with warnings if weights exceed recommended thresholds
    """
    avg = survival_curves(df)  # naive survival by years since rookie
    avg = avg.rename(columns={"years_since_rookie": "k", "survival": "p_survival"})
    avg["ipw_weight"] = 1.0 / avg["p_survival"].replace(0, np.nan)
    avg.to_csv(out_csv, index=False)
    
    # Generate warnings for extreme weights
    warnings = []
    max_weight = avg["ipw_weight"].max()
    high_weights = avg[avg["ipw_weight"] > 4.0]
    
    if max_weight > 4.0:
        warnings.append(f"⚠️  Maximum IPW weight = {max_weight:.2f}")
        warnings.append(f"   {len(high_weights)} year(s) have weights > 4.0:")
        for _, row in high_weights.iterrows():
            warnings.append(f"   - k={int(row['k'])}: P={row['p_survival']:.3f}, weight={row['ipw_weight']:.2f}")
        warnings.append("")
        warnings.append("RECOMMENDATION:")
        warnings.append("  When applying IPW in models, consider:")
        warnings.append("  1. Clamping weights: w_clamped = min(ipw_weight, 4.0)")
        warnings.append("  2. Restricting analysis to k <= 5 or 6")
        warnings.append("  3. Using robust estimators that downweight extreme values")
    else:
        warnings.append("✓ All IPW weights are <= 4.0 (no extreme values)")
    
    out_meta.write_text("\n".join(warnings) + "\n", encoding="utf-8")

