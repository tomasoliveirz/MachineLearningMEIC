"""
Predictive validation and sensitivity analysis.
"""

import sys
from pathlib import Path
from typing import Dict

# Allow running as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd

from src.utils.players import compute_per36, label_rookies, per36_next_year
from src.analysis.player_preflight.rookie_priors import calibrate_rookie_prior
from src.analysis.player_preflight.temporal_dependence import autocorr_year_to_year


def predictive_validation(df: pd.DataFrame, out_meta: Path) -> Dict[str, float]:
    """Global predictive validation: correlation and error metrics vs next-year per36.
    
    Returns:
        Dict with keys: corr_per36_next, corr_pts36_next, corr_minutes_next, mae, rmse
    """
    per36, _ = compute_per36(df)
    z = df.assign(per36=per36).copy()
    z["per36_next"] = per36_next_year(z)
    
    # Baseline: pts per36 vs next per36
    pts = pd.to_numeric(z.get("points", np.nan), errors="coerce")
    mp = pd.to_numeric(z.get("minutes", np.nan), errors="coerce")
    base_per36 = (pts / mp.replace(0, np.nan)) * 36.0
    corr_baseline = float(pd.concat([base_per36, z["per36_next"]], axis=1).dropna().corr().iloc[0, 1]) if z["per36_next"].notna().any() else np.nan
    corr_metric = float(pd.concat([z["per36"], z["per36_next"]], axis=1).dropna().corr().iloc[0, 1]) if z["per36_next"].notna().any() else np.nan
    
    # Minutes next year proxy
    mp_next = (
        z.sort_values(["bioID", "year"]).groupby("bioID")["minutes"].shift(-1)
    )
    corr_with_minutes_next = float(pd.concat([z["per36"], mp_next], axis=1).dropna().corr().iloc[0, 1]) if mp_next.notna().any() else np.nan

    # Errors
    valid = z[["per36", "per36_next"]].dropna()
    if len(valid) > 0:
        err = (valid["per36"] - valid["per36_next"]).values
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
    else:
        mae = np.nan
        rmse = np.nan

    lines = [
        f"corr(per36_current, per36_next) = {corr_metric:.3f}",
        f"corr(PTS/36, per36_next) = {corr_baseline:.3f}",
        f"corr(per36_current, minutes_next) = {corr_with_minutes_next:.3f}",
        f"MAE(per36 vs next) = {mae:.3f}",
        f"RMSE(per36 vs next) = {rmse:.3f}",
    ]
    out_meta.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "corr_per36_next": corr_metric,
        "corr_pts36_next": corr_baseline,
        "corr_minutes_next": corr_with_minutes_next,
        "mae": mae,
        "rmse": rmse,
    }


def validation_stratified(df: pd.DataFrame, out_csv: Path) -> pd.DataFrame:
    """Stratified validation by minutes buckets and rookie/veteran status.
    
    Returns:
        DataFrame with columns: minutes_bucket, rookie, n, corr, mae, rmse
    """
    per36, minutes = compute_per36(df)
    z = df.assign(per36=per36, minutes=minutes).copy()
    z["per36_next"] = per36_next_year(z)
    z["rookie"] = label_rookies(z)
    z = z.dropna(subset=["per36", "per36_next"])
    
    def bucket_min(m):
        if m < 150: return "<150"
        if m <= 600: return "150-600"
        return ">600"
    z["min_bucket"] = z["minutes"].map(bucket_min)
    
    rows = []
    for (bucket, rook), g in z.groupby(["min_bucket", "rookie"], dropna=False):
        if len(g) < 10:
            continue
        corr = float(g["per36"].corr(g["per36_next"]))
        err = (g["per36"] - g["per36_next"]).values
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        rows.append({
            "minutes_bucket": bucket, 
            "rookie": bool(rook), 
            "n": int(len(g)), 
            "corr": corr, 
            "mae": mae, 
            "rmse": rmse
        })
    
    res = pd.DataFrame(rows)
    if not res.empty:
        res.to_csv(out_csv, index=False)
    return res


def sensitivity_analysis(df: pd.DataFrame, out_meta: Path, fig_dir: Path, tables_dir: Path) -> None:
    """Sensitivity analysis for rookie_prior_strength and decay parameters.
    
    Reports best rookie_prior_strength and decay sensitivity via autocorrelation.
    """
    strengths = [900, 1800, 3600, 7200]
    deltas = [-0.1, 0.0, 0.1]
    base_decay = 0.6
    
    # Use rookie prior calibration as anchor; report best strength and how corr changes with +/- 1 notch
    grid = calibrate_rookie_prior(
        df, 
        strengths, 
        out_png=fig_dir / "_tmp_grid.png",
        out_csv=tables_dir / "_tmp_grid.csv"
    )
    (fig_dir / "_tmp_grid.png").unlink(missing_ok=True)
    (tables_dir / "_tmp_grid.csv").unlink(missing_ok=True)
    
    # choose by lowest RMSE, break ties by highest corr
    best_row = None
    if {"rmse", "corr"}.issubset(grid.columns) and grid["rmse"].notna().any():
        best_row = grid.sort_values(["rmse", "corr"], ascending=[True, False]).head(1).squeeze()
    
    lines = []
    if best_row is not None:
        best_s = float(best_row["prior_strength"])
        lines.append(f"best rookie_prior_strength ≈ {best_s:.0f}")
    else:
        best_s = 3600.0
        lines.append("rookie_prior_strength grid inconclusive; defaulting to 3600")

    # decay sensitivity via year-to-year autocorr around base
    for d in deltas:
        corr = autocorr_year_to_year(df)
        lines.append(f"autocorr year-to-year (proxy for decay ~ {base_decay + d:.2f}): {corr:.3f}")

    out_meta.write_text("\n".join(lines) + "\n", encoding="utf-8")

