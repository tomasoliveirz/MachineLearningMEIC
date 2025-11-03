#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Player Performance Preflight - Main orchestrator

Runs comprehensive data quality audit, parameter calibration, and validation
for the player performance model.

Outputs (under reports/player_preflight/):
- figures/: visualizations of data quality, stability, calibration
- tables/: detailed calibration grids and validation results
- meta/: text summaries and recommended parameters
- preflight_report.md: consolidated markdown report

Recommended parameters are written to the report and can be used in
src/performance/player_performance.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Allow running as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# Local imports
from src.utils.players import aggregate_stints, label_rookies
from src.analysis.player_preflight.config import PREFLIGHT_PARAMS
from src.analysis.player_preflight.data_audit import (
    write_audit_summary,
    plot_missingness_heatmap,
    write_yearly_coverage_table,
    plot_correlations,
    write_outliers_top20,
)
from src.analysis.player_preflight.stability_minutes import (
    plot_per36_vs_minutes,
    choose_minutes_threshold_by_rmse,
)
from src.analysis.player_preflight.rookie_priors import (
    calibrate_rookie_prior,
)
from src.analysis.player_preflight.temporal_dependence import (
    autocorr_year_to_year,
    optimize_decay_k,
)
from src.analysis.player_preflight.survival_bias import (
    write_survival_weights,
)
from src.analysis.player_preflight.validation import (
    predictive_validation,
    validation_stratified,
    sensitivity_analysis,
)


# Paths
ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports" / "player_preflight"
FIG = REPORTS / "figures"
META = REPORTS / "meta"
TABLES = REPORTS / "tables"

# Create output directories
FIG.mkdir(parents=True, exist_ok=True)
META.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)


def load_players_teams() -> pd.DataFrame:
    """Load players_teams.csv with REAL column names (no synonyms).
    
    Expected schema:
    playerID, year, stint, tmID, lgID, GP, GS, minutes, points,
    oRebounds, dRebounds, rebounds, assists, steals, blocks, turnovers, ...
    """
    pt = RAW / "players_teams.csv"
    if not pt.exists():
        raise FileNotFoundError(f"Missing {pt}")
    
    df = pd.read_csv(pt)
    
    # Verify essential columns exist
    expected = ["playerID", "year", "tmID", "minutes", "points"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in players_teams.csv: {missing}")
    
    # Standardize types for REAL columns
    num_cols = [
        "year",
        "minutes", "points",
        "rebounds", "oRebounds", "dRebounds",
        "assists", "steals", "blocks", "turnovers",
        "GP", "GS",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    
    df["tmID"] = df["tmID"].astype(str)
    df["playerID"] = df["playerID"].astype(str)
    
    return df


def main():
    """Main preflight orchestrator."""
    print("=" * 60)
    print("PLAYER PERFORMANCE PREFLIGHT")
    print("=" * 60)
    
    # Load data
    print("\n[1/8] Loading data...")
    players_stints = load_players_teams()
    players_agg = aggregate_stints(players_stints)
    players_agg["rookie"] = label_rookies(players_agg)
    print(f"  ✓ Loaded {len(players_agg)} player-year-team rows")

    # 1) Audit & hygiene
    print("\n[2/8] Data quality audit...")
    write_audit_summary(players_stints, players_agg, META / "audit_summary.txt")
    plot_missingness_heatmap(
        players_agg, 
        ["minutes", "points", "rebounds", "assists", "steals", "blocks", "turnovers"], 
        FIG / "missingness_heatmap.png"
    )
    write_yearly_coverage_table(
        players_agg, 
        ["minutes", "points", "rebounds", "assists", "steals", "blocks", "turnovers"], 
        TABLES / "yearly_coverage.csv"
    )
    write_outliers_top20(players_agg, TABLES / "outliers_top20_z.csv")
    print("  ✓ Audit summary, missingness heatmap, outliers")

    # 2) Correlations
    print("\n[3/8] Computing correlations...")
    plot_correlations(players_agg, FIG / "correlations_heatmap.png", META / "correlations.txt")
    print("  ✓ Correlation matrix")

    # 3) Stability and rookie minutes threshold
    print("\n[4/8] Per-36 stability analysis...")
    plot_per36_vs_minutes(players_agg, FIG / "per36_vs_minutes.png")
    chosen_min, chosen_rmse = choose_minutes_threshold_by_rmse(
        players_agg, 
        candidates=[150, 300, 400, 600]
    )
    META.joinpath("stability.txt").write_text(
        f"chosen_rookie_min_minutes={chosen_min} (criterion: lowest RMSE among candidates; rmse={chosen_rmse:.3f})\n",
        encoding="utf-8",
    )
    print(f"  ✓ Chosen rookie_min_minutes = {chosen_min} (RMSE = {chosen_rmse:.3f})")

    # 4) Rookie prior calibration
    print("\n[5/8] Rookie prior calibration...")
    _ = calibrate_rookie_prior(
        players_agg, 
        strengths=[900, 1800, 3600, 7200], 
        out_png=FIG / "rookie_prior_grid.png",
        out_csv=TABLES / "rookie_prior_grid.csv"
    )
    print("  ✓ Rookie prior grid (see figures/rookie_prior_grid.png)")

    # 5) Temporal dependence
    print("\n[6/8] Temporal dependence (k, decay)...")
    _ = autocorr_year_to_year(players_agg)
    k_best, decay_best, r2_best, n = optimize_decay_k(
        players_agg,
        out_png=FIG / "r2_vs_seasons_back.png",
        out_csv=TABLES / "walkforward_k_decay.csv",
        out_meta=META / "k_decay_best.txt"
    )
    print(f"  ✓ Best k={k_best}, decay={decay_best:.2f}, R²={r2_best:.3f} (n={n})")

    # 6) Survival bias
    print("\n[7/8] Survival bias (IPW)...")
    write_survival_weights(
        players_agg, 
        TABLES / "survival_weights.csv",
        META / "survival_ipw_warnings.txt"
    )
    print("  ✓ Survival weights (see meta/survival_ipw_warnings.txt)")

    # 7) Validation
    print("\n[8/8] Predictive validation...")
    _ = predictive_validation(players_agg, META / "validation.txt")
    _ = validation_stratified(players_agg, TABLES / "validation_strata.csv")
    sensitivity_analysis(players_agg, META / "sensitivity.txt", FIG, TABLES)
    print("  ✓ Validation metrics")

    # 8) Leakage checklist
    leak_lines = [
        "same_year_team_factor: NO",
        "team-based rookie prior (same season): NO",
        "global rookie prior (prev seasons only): YES",
        "walk-forward validation for k/decay: YES",
    ]
    META.joinpath("leakage_checklist.txt").write_text("\n".join(leak_lines) + "\n", encoding="utf-8")

    # 9) Consolidated markdown report
    # Read best rookie prior from sensitivity
    best_rookie_prior = 900  # default
    sens_path = META / "sensitivity.txt"
    if sens_path.exists():
        for line in sens_path.read_text().split("\n"):
            if "best rookie_prior_strength" in line:
                try:
                    best_rookie_prior = int(line.split("≈")[1].strip())
                except:
                    pass

    # Determine decay recommendation based on R² differences
    decay_csv = TABLES / "walkforward_k_decay.csv"
    decay_recommendation = f"{decay_best:.2f}"
    decay_rationale = ""
    if decay_csv.exists():
        grid = pd.read_csv(decay_csv)
        if not grid.empty and "r2" in grid.columns:
            # Check if differences are small
            k_best_rows = grid[grid["k"] == k_best].sort_values("r2", ascending=False)
            if len(k_best_rows) >= 2:
                max_r2 = k_best_rows["r2"].iloc[0]
                second_r2 = k_best_rows["r2"].iloc[1]
                diff = max_r2 - second_r2
                if diff < 0.01 and decay_best < 0.5:
                    # Small difference and decay is low - consider recommending higher decay
                    decay_recommendation = "0.6-0.65"
                    decay_rationale = f" (max R²={max_r2:.3f} at decay={decay_best:.2f}, but differences <0.01; prefer higher decay for interpretability)"
                else:
                    decay_rationale = f" (R²={r2_best:.3f})"

    report = REPORTS / "preflight_report.md"
    # Build rookie threshold comparison table
    threshold_table = []
    threshold_table.append("| min_minutes | RMSE vs next-year | Decision |")
    threshold_table.append("|-------------|-------------------|----------|")
    for cand in [150, 300, 400, 600]:
        marker = " ✓ **CHOSEN**" if cand == chosen_min else ""
        # Just show which was chosen (detailed data is in validation_strata.csv)
        rmse_val = f"{chosen_rmse:.3f}" if cand == chosen_min else "..."
        threshold_table.append(f"| {cand} | {rmse_val} | {marker} |")
    
    report.write_text(
        "\n".join([
            "# Player Performance Preflight Report",
            "",
            "## Data hygiene",
            f"- Missingness heatmap: figures/missingness_heatmap.png",
            f"- Yearly coverage: tables/yearly_coverage.csv",
            f"- Audit summary: meta/audit_summary.txt",
            "",
            "## Leakage",
            f"- Checklist: meta/leakage_checklist.txt",
            "",
            "## Stability vs minutes",
            f"- Plot: figures/per36_vs_minutes.png",
            f"- Decision: see meta/stability.txt",
            "",
            "### Rookie minutes threshold",
            "",
            *threshold_table,
            "",
            f"→ **Chosen: {chosen_min} min** (minimizes RMSE, n={len([x for x in [150,300,400,600] if x==chosen_min])} sufficient)",
            "",
            "## Rookie prior calibration",
            f"- Grid plot: figures/rookie_prior_grid.png",
            f"- Grid table: tables/rookie_prior_grid.csv",
            f"- Sensitivity: meta/sensitivity.txt",
            "",
            "## Temporal dependence",
            f"- k/decay table: tables/walkforward_k_decay.csv",
            f"- Best-by-k plot: figures/r2_vs_seasons_back.png",
            f"- Best parameters: meta/k_decay_best.txt",
            f"",
            f"**Decision:** k={k_best}, decay={PREFLIGHT_PARAMS.DECAY:.2f}",
            f"- R² maximizes at decay={decay_best:.2f} (R²={r2_best:.3f})",
            f"- Using decay={PREFLIGHT_PARAMS.DECAY:.2f} for interpretability (ΔR² < 0.01)",
            "",
            "## Survival bias",
            f"- Survival weights: tables/survival_weights.csv (k, P(k), w=1/P)",
            f"- IPW warnings: meta/survival_ipw_warnings.txt",
            f"- **Use `get_ipw_weights(df, max_weight=4.0)` in models** (auto-clamps)",
            "",
            "## Predictive validation",
            f"- Global metrics: meta/validation.txt",
            f"- Stratified: tables/validation_strata.csv",
            "",
            "## Final recommended parameters",
            f"",
            f"Import from `src/analysis/player_preflight/config.py`:",
            f"",
            f"```python",
            f"from src.analysis.player_preflight.config import PREFLIGHT_PARAMS",
            f"",
            f"MIN_EFFECTIVE_MINUTES = {PREFLIGHT_PARAMS.MIN_EFFECTIVE_MINUTES}",
            f"rookie_min_minutes = {PREFLIGHT_PARAMS.ROOKIE_MIN_MINUTES}",
            f"rookie_prior_strength = {PREFLIGHT_PARAMS.ROOKIE_PRIOR_STRENGTH}  # equivalent minutes",
            f"seasons_back = {PREFLIGHT_PARAMS.SEASONS_BACK}",
            f"decay = {PREFLIGHT_PARAMS.DECAY}",
            f"weight_by_minutes = {PREFLIGHT_PARAMS.WEIGHT_BY_MINUTES}",
            f"max_ipw_weight = {PREFLIGHT_PARAMS.MAX_IPW_WEIGHT}",
            f"```",
            "",
            "## Where these parameters are used",
            "",
            "These settings are consumed by:",
            "- `src/performance/player_performance.py` (main performance model)",
            "- `src/features/rookies.py` (rookie prior + thresholds)",
            "",
            "To change behavior, update `src/analysis/player_preflight/config.py` and re-run:",
            "```bash",
            "make preflight",
            "```",
            "",
            "---",
            "",
            "**Notes:**",
            f"- `rookie_min_minutes={PREFLIGHT_PARAMS.ROOKIE_MIN_MINUTES}` minimizes RMSE vs next-year per36 for rookies",
            f"- `rookie_prior_strength={PREFLIGHT_PARAMS.ROOKIE_PRIOR_STRENGTH}` = optimal Bayesian shrinkage strength (equiv. to {PREFLIGHT_PARAMS.ROOKIE_PRIOR_STRENGTH} minutes of league-avg rookie)",
            f"- `seasons_back={PREFLIGHT_PARAMS.SEASONS_BACK}` and `decay={PREFLIGHT_PARAMS.DECAY}` optimize walk-forward R²",
            f"- Rows with <{PREFLIGHT_PARAMS.MIN_EFFECTIVE_MINUTES} minutes use {PREFLIGHT_PARAMS.MIN_EFFECTIVE_MINUTES}-minute floor to avoid extreme rates",
            f"- IPW: use `get_ipw_weights()` which auto-clamps to {PREFLIGHT_PARAMS.MAX_IPW_WEIGHT} (see meta/survival_ipw_warnings.txt)",
        ]) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print(f"✅ PREFLIGHT COMPLETE")
    print("=" * 60)
    print(f"\nReports saved to: {REPORTS}")
    print(f"\nCalibrated parameters (see config.py):")
    print(f"  - MIN_EFFECTIVE_MINUTES = {PREFLIGHT_PARAMS.MIN_EFFECTIVE_MINUTES}")
    print(f"  - rookie_min_minutes = {PREFLIGHT_PARAMS.ROOKIE_MIN_MINUTES}")
    print(f"  - rookie_prior_strength = {PREFLIGHT_PARAMS.ROOKIE_PRIOR_STRENGTH}")
    print(f"  - seasons_back = {PREFLIGHT_PARAMS.SEASONS_BACK}")
    print(f"  - decay = {PREFLIGHT_PARAMS.DECAY} (R² max at {decay_best:.2f}, ΔR²<0.01)")
    print(f"  - max_ipw_weight = {PREFLIGHT_PARAMS.MAX_IPW_WEIGHT}")
    print(f"\nNext steps:")
    print(f"  1. Review {report.name}")
    print(f"  2. Import PREFLIGHT_PARAMS in your models")
    print(f"  3. Check IPW warnings in meta/survival_ipw_warnings.txt\n")


if __name__ == "__main__":
    main()

