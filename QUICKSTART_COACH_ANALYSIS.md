# Quick Start: Coach Performance Analysis

## 🚀 TL;DR

```bash
cd /home/tomio/Documents/UNI/AC
source venv/bin/activate

# Generate all 3 tables
python3 src/performance/team_performance.py
python3 src/performance/coach_season_performance.py
python3 src/performance/coach_career_performance.py

# Run analyses + generate plots
python3 src/performance/example_analyses.py
```

**Output:**
- 3 CSV files in `data/processed/`
- 5 PNG plots in `reports/plots/coach_performance/`

---

## 📁 What You Get

### Files Generated

1. **`team_performance.csv`** (142 rows)
   - Team-season level
   - Pythag win%, roster strength, overachievement metrics
   - Playoff results integrated

2. **`coach_season_performance.csv`** (162 rows)
   - Coach-season-stint level (handles mid-season changes)
   - EB-smoothed win%, overachievement vs Pythag/Roster
   - First-year flags, COY awards, playoff win%

3. **`coach_career_performance.csv`** (57 rows)
   - Coach career aggregates (GP-weighted)
   - Avg overachievement, consistency, trend
   - Career playoff win%, COY count

### Plots Generated

1. **`top_overachievers.png`** - Top 10 coaches by overachievement
2. **`correlation_matrix.png`** - Heatmap of key metrics
3. **`first_year_impact.png`** - Do first-year coaches improve teams?
4. **`rs_vs_po_performance.png`** - Regular season vs playoffs scatter
5. **`coy_predictors.png`** - What predicts Coach of the Year?

---

## 🔑 Key Metrics Explained

### Pythagorean Win%
```
pythag_win_pct = (PF^10.8) / (PF^10.8 + PA^10.8)
```
**Interpretation:** Expected win% based purely on points scored/allowed  
**WNBA exponent:** 10.80 (vs ~13.9 for NBA)

### Team Strength (Roster Quality)
```
team_strength = Σ(player_performance × minutes) / Σ(minutes)
```
**Interpretation:** Weighted average of player performance  
**Predictive power:** R² = 0.283 (explains 28% of win% variance)

### Overachievement (2 variants)
```
overach_pythag = actual_win% - pythag_win%
overach_roster = actual_win% - expected_win%_from_roster
```
**Interpretation:** How much coach beats expectation  
**Ranges:** Pythag ±13%, Roster ±39%

### Empirical Bayes Smoothing
```
eb_win_pct = (won + 34×0.501) / (games + 34)
```
**Interpretation:** Shrink small samples toward league average (0.501)  
**Alpha = 34:** One WNBA season worth of "prior"

---

## 💡 Quick Analyses (Copy-Paste)

### 1. Who are the best coaches?
```python
import pandas as pd

cc = pd.read_csv('data/processed/coach_career_performance.csv')

# Filter: min 30 games
cc_valid = cc[cc['games'] >= 30]

# Top 10 by overachievement
top10 = cc_valid.nlargest(10, 'avg_overach_pythag')
print(top10[['coachID', 'seasons', 'games', 'avg_overach_pythag', 
             'eb_career_win_pct', 'coy_awards']])
```

### 2. Does Coach of the Year = Overachievement?
```python
cs = pd.read_csv('data/processed/coach_season_performance.csv')

coy = cs[cs['is_coy_winner'] == 1]
not_coy = cs[cs['is_coy_winner'] == 0]

print(f"COY overach:     {coy['coach_overach_pythag'].mean():.3f}")
print(f"Non-COY overach: {not_coy['coach_overach_pythag'].mean():.3f}")
print(f"COY delta:       {coy['delta_vs_prev_team'].mean():.3f}")
print(f"Non-COY delta:   {not_coy['delta_vs_prev_team'].mean():.3f}")
```
**Result:** COY correlates more with *team improvement* (+21.6%) than overachievement (+1.4%)

### 3. Are playoff coaches different?
```python
cc = pd.read_csv('data/processed/coach_career_performance.csv')

valid = cc.dropna(subset=['eb_career_win_pct', 'career_po_win_pct'])
corr = valid[['eb_career_win_pct', 'career_po_win_pct']].corr()

print(f"RS vs PO correlation: {corr.iloc[0,1]:.3f}")

# Best PO overperformers
valid['po_gap'] = valid['career_po_win_pct'] - valid['eb_career_win_pct']
print("\nBest playoff overperformers:")
print(valid.nlargest(3, 'po_gap')[['coachID', 'eb_career_win_pct', 
                                     'career_po_win_pct', 'po_gap']])
```

### 4. Do first-year coaches struggle?
```python
cs = pd.read_csv('data/processed/coach_season_performance.csv')

first = cs[cs['is_first_year_with_team'] == 1]
not_first = cs[cs['is_first_year_with_team'] == 0]

print(f"First-year delta: {first['delta_vs_prev_team'].mean():.3f}")
print(f"Not first delta:  {not_first['delta_vs_prev_team'].mean():.3f}")
print(f"First-year overach: {first['coach_overach_pythag'].mean():.3f}")
print(f"Not first overach:  {not_first['coach_overach_pythag'].mean():.3f}")
```
**Result:** First-year coaches average -1.5% vs prev year (slightly worse)

---

## 📊 Sample Results (WNBA Data)

### Top Overachievers (Career, min 30 games)
| Coach | Games | Overach | EB Win% | COY |
|-------|-------|---------|---------|-----|
| bryanjo01w | 40 | +13.96% | 62.2% | 0 |
| coopecy01w | 42 | +10.65% | 47.4% | 0 |
| rollitr01w | 52 | +9.25% | 47.7% | 0 |
| coopemi01w | 252 | +5.02% | 64.3% | 1 |

### COY Predictors
| Metric | COY | Non-COY | Δ |
|--------|-----|---------|---|
| RS Win% | 65.9% | 47.8% | +18.1% |
| Overach Pythag | +1.4% | -0.3% | +1.7% |
| Overach Roster | +11.6% | -1.7% | +13.4% |
| **Δ vs Prev Year** | **+21.6%** | **-2.2%** | **+23.8%** ← strongest |

### RS vs PO Performance
- **Correlation:** r = 0.369 (moderate)
- **Best PO coach:** westhpa99w (RS 56.9% → PO 77.8%, gap +20.9%)
- **Worst PO coach:** allenso99w (RS 56.4% → PO 0%, gap -56.4%)

---

## 🛠️ Troubleshooting

### "FileNotFoundError: team_performance.csv"
**Fix:** Run scripts in order (team → coach_season → coach_career)

### "KeyError: 'minutes'"
**Fix:** Ensure `player_performance.csv` exists and has `minutes` column (not `mp`)

### Plots don't show
**Fix:** Check `reports/plots/coach_performance/` directory exists  
**Or:** Run `example_analyses.py` which creates it automatically

### Small sample warnings
**Fix:** Filter coaches with `cc[cc['games'] >= 30]` before ranking

---

## 📚 Full Documentation

- **Technical details:** `src/performance/README.md`
- **Architecture overview:** `COACH_ANALYSIS_ARCHITECTURE.md`
- **Implementation summary:** `IMPLEMENTATION_SUMMARY.md`

---

## ✅ Validation Checklist

Before using results, verify:
- [ ] All 3 CSV files generated (check `data/processed/`)
- [ ] Pythag exponent = 10.80 (printed during `team_performance.py`)
- [ ] 10 COY awards found (printed during `coach_season_performance.py`)
- [ ] Ranges plausible: overach_pythag ∈ [-20%, +20%]
- [ ] No unexpected NaNs (except PO data for non-playoff teams)

---

**Need help?** Check the documentation files or inspect the scripts' print outputs for diagnostics.

**Version:** 1.0.0 | **Updated:** 2025-11-05

