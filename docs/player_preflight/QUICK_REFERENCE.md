# Quick Reference: Player Performance Preflight

## 🚀 Run

```bash
source venv/bin/activate
python -m src.analysis.player_preflight.run_preflight
```

## 📊 Key Outputs

| File | What it tells you |
|------|-------------------|
| `preflight_report.md` | **Start here!** Consolidated report with recommended parameters |
| `meta/audit_summary.txt` | Data quality: duplicates, zeros, ranges |
| `meta/stability.txt` | Chosen `rookie_min_minutes` and RMSE |
| `meta/k_decay_best.txt` | Best `seasons_back` and `decay` from walk-forward |
| `meta/survival_ipw_warnings.txt` | ⚠️ IPW weight warnings and recommendations |
| `tables/validation_strata.csv` | Prediction quality by minutes buckets |
| `figures/per36_vs_minutes.png` | Stability visualization |
| `figures/r2_vs_seasons_back.png` | Marginal benefit of more seasons |

## 🎯 Recommended Parameters (Typical)

```python
MIN_EFFECTIVE_MINUTES = 12          # Floor to avoid extreme rates
rookie_min_minutes = 400             # RMSE-minimizing threshold
rookie_prior_strength = 900          # Equivalent minutes of prior
seasons_back = 3                     # Look back 3 years
decay = 0.6                          # Weight decay per season
weight_by_minutes = True             # Weight seasons by minutes played
```

## 🔍 Interpretation Guide

### `rookie_min_minutes`
- **Lower (150-300)**: More rookies, noisier
- **Higher (400-600)**: Fewer rookies, more stable
- **Chosen by**: Minimizing RMSE vs next-year per36

### `rookie_prior_strength`
- **900**: Rookie with 900 min → 50% observation, 50% prior
- **1800**: More shrinkage toward league average
- **Chosen by**: Best correlation + RMSE vs next-year

### `seasons_back`
- **1**: Only current season
- **3**: Current + 2 previous (typical optimum)
- **5+**: Marginal gains usually <0.001 R²
- **Chosen by**: Walk-forward R² optimization

### `decay`
- **1.0**: All seasons weighted equally
- **0.6**: Season t-1 gets 60% weight of season t
- **0.4**: More aggressive decay
- **Chosen by**: R² vs next-year (but prefer 0.6+ for interpretability if close)

## ⚠️ Common Issues

### IPW Weights > 4.0
**Symptom:** `survival_ipw_warnings.txt` shows extreme weights

**Fix:**
```python
ipw_clamped = np.minimum(ipw_weights, 4.0)
```

or restrict analysis to k ≤ 5 years from rookie

### Seasons with 0 stats
**Symptom:** `audit_summary.txt` shows "Rows with all-zero core stats: 3"

**Fix:** Filter before calibration:
```python
df = df[df[["minutes", "points"]].sum(axis=1) > 0]
```

### Low minutes seasons dominating
**Symptom:** High RMSE in validation_strata for <150 min bucket

**Already handled:** `rookie_min_minutes` threshold filters these out

## 📖 Full Documentation

- **Package README**: `README.md` (in this directory)
- **Migration Guide**: `../../../MIGRATION_PREFLIGHT.md`
- **Complete Summary**: `../../../REFACTORING_SUMMARY.md`

## 🔄 Modular Usage

Import specific modules:

```python
from src.analysis.player_preflight.rookie_priors import calibrate_rookie_prior
from src.analysis.player_preflight.validation import predictive_validation
from src.utils.players import compute_per36, label_rookies

# Use functions independently
grid = calibrate_rookie_prior(df, strengths=[900, 1800], ...)
metrics = predictive_validation(df, ...)
```

## 🐛 Troubleshooting

### ModuleNotFoundError
```bash
# Make sure you're in venv
source venv/bin/activate
# and running from project root
cd /home/tomio/Documents/UNI/AC
```

### FutureWarning from pandas
Should be fixed in current version. If not:
```bash
pip install --upgrade pandas
```

### matplotlib backend issues
```python
import matplotlib
matplotlib.use('Agg')  # Add at top of script if running headless
```

---

**Version:** 1.0.0  
**Last Updated:** 2025-11-03

