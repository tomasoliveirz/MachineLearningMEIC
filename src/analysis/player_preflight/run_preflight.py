#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Player Performance Preflight - Basic Mode

Runs core data quality audit and temporal optimization for player performance.

FOCUS: Establish per36 metric, understand data, optimize temporal weights.

Outputs (under reports/player_preflight/):
- figures/: data quality visualizations, per36 stability, temporal R²
- tables/: coverage, outliers
- meta/: audit summaries, correlations, best k/decay
- preflight_report.md: consolidated report

Advanced features (rookies priors, survival bias, validation) disabled for now.
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
)
from src.analysis.player_preflight.temporal_dependence import (
    autocorr_year_to_year,
    optimize_decay_k,
)

# Advanced modules (disabled for basic mode):
# from src.analysis.player_preflight.rookie_priors import calibrate_rookie_prior
# from src.analysis.player_preflight.validation import predictive_validation, validation_stratified, sensitivity_analysis


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
    print("PLAYER PERFORMANCE PREFLIGHT (BASIC MODE)")
    print("=" * 60)
    
    # Load data
    print("\n[1/4] Loading data...")
    players_stints = load_players_teams()
    players_agg = aggregate_stints(players_stints)
    players_agg["rookie"] = label_rookies(players_agg)
    print(f"  ✓ Loaded {len(players_agg)} player-year-team rows")

    # 1) Audit & hygiene
    print("\n[2/4] Data quality audit...")
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
    print("\n[3/4] Computing correlations...")
    plot_correlations(players_agg, FIG / "correlations_heatmap.png", META / "correlations.txt")
    print("  ✓ Correlation matrix")
    
    # 3) Per-36 stability (visual only, no auto-threshold)
    plot_per36_vs_minutes(players_agg, FIG / "per36_vs_minutes.png")
    print("  ✓ Per-36 vs minutes plot (visual inspection)")

    # 4) Temporal dependence (k, decay)
    print("\n[4/4] Temporal dependence (k, decay)...")
    _ = autocorr_year_to_year(players_agg)
    k_best, decay_best, r2_best, n = optimize_decay_k(
        players_agg,
        out_png=FIG / "r2_vs_seasons_back.png",
        out_csv=TABLES / "walkforward_k_decay.csv",
        out_meta=META / "k_decay_best.txt"
    )
    print(f"  ✓ Best k={k_best}, decay={decay_best:.2f}, R²={r2_best:.3f} (n={n})")

    # 5) Leakage checklist
    leak_lines = [
        "same_year_team_factor: NO",
        "team-based features (same season): NO",
        "walk-forward validation for k/decay: YES",
    ]
    META.joinpath("leakage_checklist.txt").write_text("\n".join(leak_lines) + "\n", encoding="utf-8")

    # 6) Simple markdown report
    report = REPORTS / "preflight_report.md"
    report.write_text(
        "\n".join([
            "# Player Performance Preflight Report (Basic Mode)",
            "",
            "## Data hygiene",
            f"- Missingness heatmap: figures/missingness_heatmap.png",
            f"- Yearly coverage: tables/yearly_coverage.csv",
            f"- Audit summary: meta/audit_summary.txt",
            f"- Outliers (top 20): tables/outliers_top20_z.csv",
            "",
            "## Correlations",
            f"- Heatmap: figures/correlations_heatmap.png",
            f"- Details: meta/correlations.txt",
            "",
            "## Per-36 stability",
            f"- Visual inspection: figures/per36_vs_minutes.png",
            f"- Interpretation: Low minutes → high variance. Threshold choice TBD.",
            "",
            "## Temporal dependence",
            f"- k/decay optimization: tables/walkforward_k_decay.csv",
            f"- Best-by-k plot: figures/r2_vs_seasons_back.png",
            f"- Best parameters: meta/k_decay_best.txt",
            f"",
            f"**Optimal:** k={k_best}, decay={decay_best:.2f} (R²={r2_best:.3f}, n={n})",
            "",
            "## Leakage checklist",
            f"- See: meta/leakage_checklist.txt",
            "",
            "---",
            "",
            "## Core parameters (config.py)",
            "",
            "```python",
            f"MIN_EFFECTIVE_MINUTES = {PREFLIGHT_PARAMS.MIN_EFFECTIVE_MINUTES}  # floor for per-36 calc",
            f"SEASONS_BACK = {PREFLIGHT_PARAMS.SEASONS_BACK}  # temporal window",
            f"DECAY = {PREFLIGHT_PARAMS.DECAY}  # weight for older seasons",
            f"WEIGHT_BY_MINUTES = {PREFLIGHT_PARAMS.WEIGHT_BY_MINUTES}  # minutes-weighted averages",
            "```",
            "",
            "**Notes:**",
            f"- Temporal optimization uses walk-forward validation (no data leakage)",
            f"- Per-36 floor ({PREFLIGHT_PARAMS.MIN_EFFECTIVE_MINUTES} min) avoids extreme rates",
            f"- Advanced features (rookie priors, survival bias) disabled for basic mode",
        ]) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print(f"✅ PREFLIGHT COMPLETE (BASIC MODE)")
    print("=" * 60)
    print(f"\nReports saved to: {REPORTS}")
    print(f"\nCore parameters (see config.py):")
    print(f"  - MIN_EFFECTIVE_MINUTES = {PREFLIGHT_PARAMS.MIN_EFFECTIVE_MINUTES}")
    print(f"  - seasons_back = {PREFLIGHT_PARAMS.SEASONS_BACK}")
    print(f"  - decay = {PREFLIGHT_PARAMS.DECAY} (optimal: {decay_best:.2f}, R²={r2_best:.3f})")
    print(f"  - weight_by_minutes = {PREFLIGHT_PARAMS.WEIGHT_BY_MINUTES}")
    print(f"\nNext steps:")
    print(f"  1. Review {report.name}")
    print(f"  2. Inspect per36_vs_minutes.png and choose threshold")
    print(f"  3. Fine-tune per36 metric weights if needed\n")


if __name__ == "__main__":
    main()

