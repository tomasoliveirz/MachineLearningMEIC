# Team Ranking Model: Predictive vs Descriptive Modes

## Overview

The team ranking model now supports two distinct modes to address data leakage concerns:

1. **STRICT PREDICTIVE mode** (`strict_predictive=True`) - For honest pre-season forecasting
2. **DESCRIPTIVE mode** (`strict_predictive=False`) - For post-season analysis

## The Leakage Problem

### Original Issue

The model previously had **direct target leakage** through features that contained the actual season results:

```python
# These features algebraically reconstruct rs_win_pct:
overach_pythag = rs_win_pct - pythag_win_pct
overach_roster = rs_win_pct - rs_win_pct_expected_roster

# Therefore:
rs_win_pct = overach_pythag + pythag_win_pct  # Exact reconstruction
```

Since `rank` is essentially the ordering by `rs_win_pct`, the model was effectively:
- **Input**: rs_win_pct (disguised as overach_* + pythag_win_pct)
- **Output**: rank (derived from rs_win_pct)

### Evidence of Leakage

**Before fix (with leakage):**
- MAE_rank: 0.22 positions
- Mean Spearman: 0.96
- Overall accuracy: 81.48% (22/27 teams exact)
- Top-1 accuracy: 100% (always predicts champion correctly)

**After fix (strict predictive mode):**
- MAE_rank: 1.70 positions
- Mean Spearman: 0.32
- Overall accuracy: 22.22% (6/27 teams exact)
- Top-1 accuracy: 25% (realistic for honest forecasting)

## Mode Comparison

### STRICT PREDICTIVE Mode (`strict_predictive=True`)

**Purpose:** Honest pre-season forecasting

**Features used:**
- ✅ Historical performance: `prev_win_pct_*`, `win_pct_change`
- ✅ Roster strength: `team_strength` (can be estimated pre-season)
- ✅ Rolling averages (past seasons): `*_ma3`, `*_ma5`
- ✅ Trends (past seasons): `*_trend3`, `*_trend5`
- ✅ Structural context: `franchise_changed`, `confID`

**Features excluded:**
- ❌ `overach_pythag`, `overach_roster` (contain rs_win_pct)
- ❌ `rs_win_pct_expected_roster` (temporal leakage)
- ❌ `pythag_win_pct` (requires season results)
- ❌ All boxscore stats from current season: `point_diff`, `off_eff`, `def_eff`, etc.
- ❌ Normalized stats: `*_norm`

**Use cases:**
- Pre-season rank forecasting
- Fair model evaluation and comparison
- Academic research on predictive sports analytics
- Building prediction systems for betting/fantasy sports

### DESCRIPTIVE Mode (`strict_predictive=False`)

**Purpose:** Post-season analysis and interpretation

**Features used:**
- All features from STRICT PREDICTIVE mode
- ➕ Current season boxscore: `point_diff`, `off_eff`, `def_eff`, etc.
- ➕ Pythagorean expectation: `pythag_win_pct`
- ➕ Overachievement metrics: `overach_pythag`, `overach_roster`
- ➕ Roster expectations: `rs_win_pct_expected_roster`

**Use cases:**
- Understanding what explains final rankings
- Identifying overachieving/underachieving teams
- Causal inference: "What factors drive success?"
- Post-season retrospective analysis

## Usage

### Python API

```python
from src.model.ranking_model.team_ranking_model import run_team_ranking_model

# Strict predictive mode (no leakage)
run_team_ranking_model(
    max_train_year=8,
    report_name="team_ranking_report_predictive.txt",
    strict_predictive=True
)

# Descriptive mode (includes season results)
run_team_ranking_model(
    max_train_year=8,
    report_name="team_ranking_report_descriptive.txt",
    strict_predictive=False
)
```

### CLI

Edit the configuration in `src/model/ranking_model/team_ranking_model.py`:

```python
if __name__ == "__main__":
    MAX_TRAIN_YEAR = 8
    REPORT_NAME = "team_ranking_report.txt"
    STRICT_PREDICTIVE = True  # Change to False for descriptive mode
    
    run_team_ranking_model(
        max_train_year=MAX_TRAIN_YEAR,
        report_name=REPORT_NAME,
        strict_predictive=STRICT_PREDICTIVE
    )
```

Then run:
```bash
python src/model/ranking_model/team_ranking_model.py
```

## Temporal Leakage Fix in team_performance.py

### Problem

The regression `rs_win_pct ~ team_strength` was fit on **all years** (including test data), then used as a feature:

```python
# BAD: Fits on all data including future/test years
valid = df[df['team_strength'].notna() & df['rs_win_pct'].notna()].copy()
model.fit(X, y)
```

This means the regression coefficients "saw" the test data, creating temporal leakage.

### Solution

Added `max_train_year` parameter to restrict regression training to training years only:

```python
# GOOD: Fits only on training years
if max_train_year is not None:
    valid = df[
        (df['team_strength'].notna()) &
        (df['rs_win_pct'].notna()) &
        (df['year'] <= max_train_year)
    ].copy()
```

### Usage

```bash
# Generate team_performance.csv with temporal split
python src/performance/team_performance.py --max-train-year 8

# Generate without temporal split (descriptive mode)
python src/performance/team_performance.py
```

**Note:** For the ranking model's STRICT PREDICTIVE mode, we don't use `rs_win_pct_expected_roster` anyway (it's excluded from features). But this fix is important for:
1. Consistency and scientific rigor
2. Other analyses that might use `rs_win_pct_expected_roster`
3. The DESCRIPTIVE mode (which includes it)

## Expected Performance

### STRICT PREDICTIVE Mode

For a 10-season dataset with train/test split at year 8:

| Metric | Expected Range | Actual |
|--------|----------------|--------|
| MAE_rank | 1.5 - 3.0 positions | 1.70 |
| Mean Spearman | 0.25 - 0.50 | 0.32 |
| Overall accuracy | 15% - 30% | 22.22% |
| Top-1 accuracy | 20% - 50% | 25% |

These modest numbers are **normal and honest** for real sports forecasting.

### DESCRIPTIVE Mode

| Metric | Expected Range | Actual |
|--------|----------------|--------|
| MAE_rank | 0.2 - 0.5 positions | 0.22 |
| Mean Spearman | 0.90 - 0.98 | 0.96 |
| Overall accuracy | 70% - 90% | 81.48% |
| Top-1 accuracy | 90% - 100% | 100% |

These high numbers reflect that the model has access to season results.

## Recommendations

### For Academic Work

- **Always use STRICT PREDICTIVE mode** for model evaluation
- Report both modes if doing explanatory analysis
- Clearly state which mode was used in papers/reports
- Never compare predictive and descriptive models directly

### For Practical Applications

- **Pre-season forecasting:** STRICT PREDICTIVE only
- **In-season updates:** Neither mode (would need mid-season features)
- **Post-season analysis:** DESCRIPTIVE mode is appropriate
- **Feature importance:** Use STRICT PREDICTIVE for predictive importance, DESCRIPTIVE for explanatory importance

### For Teaching

- Show both modes to demonstrate impact of leakage
- Use as case study for data leakage detection
- Excellent example of "suspiciously good" performance indicating problems

## Files Modified

1. `src/model/ranking_model/team_ranking_model.py`
   - Changed `REPORTS_DIR` to `reports/models/`
   - Added `strict_predictive` parameter to `build_feature_matrix()`
   - Created two feature lists: predictive and descriptive
   - Propagated `strict_predictive` through pipeline

2. `src/performance/team_performance.py`
   - Added `max_train_year` parameter to `compute_overachieves()`
   - Added argparse for CLI usage
   - Added temporal split logging

3. `requirements.txt`
   - Added `scipy>=1.9.0`
   - Added `scikit-learn>=1.0.0`

## Validation

Run the test script to verify both modes:

```bash
python test_both_modes.py
```

This will generate two reports showing the dramatic difference in performance between modes.

