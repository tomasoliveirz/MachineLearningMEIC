# Player Performance Preflight Report (Basic Mode)

## Data hygiene
- Missingness heatmap: figures/missingness_heatmap.png
- Yearly coverage: tables/yearly_coverage.csv
- Audit summary: meta/audit_summary.txt
- Outliers (top 20): tables/outliers_top20_z.csv

## Correlations
- Heatmap: figures/correlations_heatmap.png
- Details: meta/correlations.txt

## Per-36 stability
- Visual inspection: figures/per36_vs_minutes.png
- Interpretation: Low minutes → high variance. Threshold choice TBD.

## Temporal dependence
- k/decay optimization: tables/walkforward_k_decay.csv
- Best-by-k plot: figures/r2_vs_seasons_back.png
- Best parameters: meta/k_decay_best.txt

**Optimal:** k=3, decay=0.40 (R²=0.490, n=954)

## Leakage checklist
- See: meta/leakage_checklist.txt

---

## Core parameters (config.py)

```python
MIN_EFFECTIVE_MINUTES = 12  # floor for per-36 calc
SEASONS_BACK = 3  # temporal window
DECAY = 0.6  # weight for older seasons
WEIGHT_BY_MINUTES = True  # minutes-weighted averages
```

**Notes:**
- Temporal optimization uses walk-forward validation (no data leakage)
- Per-36 floor (12 min) avoids extreme rates
- Advanced features (rookie priors, survival bias) disabled for basic mode
