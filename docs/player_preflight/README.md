# Player Performance Preflight

Comprehensive data quality audit, parameter calibration, and validation for the player performance model.

## Purpose

This module runs **before** the main player performance model (`src/performance/player_performance.py`) to:

1. **Audit data quality**: check for duplicates, missing values, outliers, unit consistency
2. **Calibrate parameters**: find optimal values for `rookie_min_minutes`, `rookie_prior_strength`, `seasons_back`, `decay`
3. **Validate predictively**: measure correlation and RMSE vs next-year performance
4. **Check for survival bias**: compute IPW weights and warn about extreme values
5. **Prevent data leakage**: explicit checklist of temporal validation procedures

## Structure

```
src/analysis/player_preflight/
├── __init__.py               # Package metadata
├── README.md                 # This file
├── run_preflight.py          # Main orchestrator (run this!)
├── data_audit.py             # Data quality checks
├── stability_minutes.py      # Per-36 stability and rookie minutes threshold
├── rookie_priors.py          # Bayesian shrinkage calibration
├── temporal_dependence.py    # Seasons-back and decay optimization
├── survival_bias.py          # Survival curves and IPW
└── validation.py             # Predictive validation and sensitivity
```

## Usage

### Run the full preflight:

```bash
cd /home/tomio/Documents/UNI/AC
python -m src.analysis.player_preflight.run_preflight
```

### Outputs

All outputs are saved to `reports/player_performance_preflight/`:

```
reports/player_performance_preflight/
├── preflight_report.md         # Consolidated report with recommended parameters
├── figures/                    # Plots
│   ├── missingness_heatmap.png
│   ├── per36_vs_minutes.png
│   ├── correlations_heatmap.png
│   ├── rookie_prior_grid.png
│   └── r2_vs_seasons_back.png
├── tables/                     # Data tables
│   ├── yearly_coverage.csv
│   ├── outliers_top20_z.csv
│   ├── rookie_prior_grid.csv
│   ├── walkforward_k_decay.csv
│   ├── validation_strata.csv
│   └── survival_weights.csv
└── meta/                       # Text summaries
    ├── audit_summary.txt
    ├── leakage_checklist.txt
    ├── stability.txt
    ├── k_decay_best.txt
    ├── sensitivity.txt
    ├── validation.txt
    ├── correlations.txt
    └── survival_ipw_warnings.txt
```

### Recommended Parameters

After running, check `preflight_report.md` for the final recommended parameters. Example:

```python
MIN_EFFECTIVE_MINUTES = 12
rookie_min_minutes = 400
rookie_prior_strength = 900  # equivalent minutes
seasons_back = 3
decay = 0.6  # or 0.4 if R² is significantly better
weight_by_minutes = True
```

## Key Decisions

### 1. `rookie_min_minutes`

Chosen by minimizing RMSE vs next-year per36 for rookies. Typical values: 150-600.

- Lower threshold → more rookies included, but noisier predictions
- Higher threshold → more stable, but fewer rookies in calibration

### 2. `rookie_prior_strength`

Bayesian shrinkage strength in "equivalent minutes". Typical values: 900-3600.

- Lower strength → less shrinkage, trust observations more
- Higher strength → more shrinkage toward league-average rookie

Interpretation: `rookie_prior_strength=900` means a rookie with 900 minutes gets 50% weight from observation and 50% from prior.

### 3. `seasons_back` and `decay`

Walk-forward validation finds optimal (k, decay) for predicting per36(t+1).

- `seasons_back (k)`: how many previous seasons to include (typical: 2-4)
- `decay`: exponential weight decay for older seasons (typical: 0.4-0.7)
  - `decay=1.0` → all seasons weighted equally
  - `decay=0.5` → each season back gets 50% weight of previous

If R² differences are small (<0.01), prefer higher decay (0.6-0.7) for interpretability.

### 4. Survival Bias / IPW

IPW weights can become extreme (>4.0) for players many years after rookie season. The module warns if this happens and recommends:

- Clamping weights: `w_clamped = min(ipw_weight, 4.0)`
- Restricting analysis to k ≤ 5 or 6
- Using robust estimators

## Integration with Main Model

The recommended parameters from `preflight_report.md` should be used in `src/performance/player_performance.py`.

Example workflow:

1. Run preflight: `python -m src.analysis.player_preflight.run_preflight`
2. Review `reports/player_performance_preflight/preflight_report.md`
3. Update parameters in `src/performance/player_performance.py`
4. Run performance model with calibrated parameters

## Shared Utilities

Common functions (e.g., `compute_per36`, `label_rookies`, `aggregate_stints`) are in `src/utils/players.py` and can be reused across modules.

---

**Version:** 1.0.0  
**Last Updated:** 2025-11-03

