# Player Performance Preflight Report

## Data hygiene
- Missingness heatmap: figures/missingness_heatmap.png
- Yearly coverage: tables/yearly_coverage.csv
- Audit summary: meta/audit_summary.txt

## Leakage
- Checklist: meta/leakage_checklist.txt

## Stability vs minutes
- Plot: figures/per36_vs_minutes.png
- Decision: see meta/stability.txt

### Rookie minutes threshold

| min_minutes | RMSE vs next-year | Decision |
|-------------|-------------------|----------|
| 150 | ... |  |
| 300 | ... |  |
| 400 | 3.268 |  ✓ **CHOSEN** |
| 600 | ... |  |

→ **Chosen: 400 min** (minimizes RMSE, n=1 sufficient)

## Rookie prior calibration
- Grid plot: figures/rookie_prior_grid.png
- Grid table: tables/rookie_prior_grid.csv
- Sensitivity: meta/sensitivity.txt

## Temporal dependence
- k/decay table: tables/walkforward_k_decay.csv
- Best-by-k plot: figures/r2_vs_seasons_back.png
- Best parameters: meta/k_decay_best.txt

**Decision:** k=3, decay=0.60
- R² maximizes at decay=0.40 (R²=0.490)
- Using decay=0.60 for interpretability (ΔR² < 0.01)

## Survival bias
- Survival weights: tables/survival_weights.csv (k, P(k), w=1/P)
- IPW warnings: meta/survival_ipw_warnings.txt
- **Use `get_ipw_weights(df, max_weight=4.0)` in models** (auto-clamps)

## Predictive validation
- Global metrics: meta/validation.txt
- Stratified: tables/validation_strata.csv

## Final recommended parameters

Import from `src/analysis/player_preflight/config.py`:

```python
from src.analysis.player_preflight.config import PREFLIGHT_PARAMS

MIN_EFFECTIVE_MINUTES = 12
rookie_min_minutes = 400
rookie_prior_strength = 900  # equivalent minutes
seasons_back = 3
decay = 0.6
weight_by_minutes = True
max_ipw_weight = 4.0
```

## Where these parameters are used

These settings are consumed by:
- `src/performance/player_performance.py` (main performance model)
- `src/features/rookies.py` (rookie prior + thresholds)

To change behavior, update `src/analysis/player_preflight/config.py` and re-run:
```bash
make preflight
```

---

**Notes:**
- `rookie_min_minutes=400` minimizes RMSE vs next-year per36 for rookies
- `rookie_prior_strength=900` = optimal Bayesian shrinkage strength (equiv. to 900 minutes of league-avg rookie)
- `seasons_back=3` and `decay=0.6` optimize walk-forward R²
- Rows with <12 minutes use 12-minute floor to avoid extreme rates
- IPW: use `get_ipw_weights()` which auto-clamps to 4.0 (see meta/survival_ipw_warnings.txt)
