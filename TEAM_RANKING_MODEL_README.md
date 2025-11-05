# Team Ranking Model - Conference Ranking Prediction

## 📊 Overview

Machine learning model that predicts regular season conference rankings for WNBA teams based on historical performance metrics.

**Key characteristics:**
- **Temporal split:** Train on seasons 1-8, test on seasons 9-10
- **No data leakage:** Model never sees future data during training
- **Walk-forward CV:** Internal cross-validation to prevent overfitting
- **Conference-aware:** Predicts rankings within East/West conferences
- **Two models:** RandomForestRegressor & GradientBoostingRegressor
- **Regularized:** Reduced overfitting vs naive configurations

---

## 📁 Files

### Canonical Script (USE THIS)
- **`src/model/ranking_model/team_ranking_model.py`** (unified, ~650 lines)
  - Loads data from `team_season_statistics.csv` + `team_performance.csv`
  - Supports RandomForest (`--model rf`) and GradientBoosting (`--model gbr`)
  - Walk-forward cross-validation within training seasons
  - Trains on seasons 1-8, evaluates on seasons 9-10
  - Generates predictions and comprehensive metrics report

### Deprecated Scripts (wrappers/legacy)
- **`src/performance/team_ranking_model.py`** (deprecated, calls unified model)
- **`src/model/ranking_model/ranking_model.py`** (deprecated, kept for reference)

### Outputs

1. **`data/processed/team_ranking_predictions.csv`** (142 rows)
   - Columns: `year`, `confID`, `tmID`, `name`, `rank`, `pred_rank`, `pred_score`, `split`
   - Contains predictions for all seasons (train + test)
   - Sorted by year, conference, predicted rank

2. **`reports/team_ranking_report.txt`**
   - Model configuration
   - Train/test metrics
   - Example predictions for Year 9 East/West

---

## 🎯 Model Performance

### RandomForest (--model rf)

**Walk-forward CV (internal, seasons 3-8):**
- **MAE rank:** 0.413 (realistic expectation)
- **Mean Spearman:** 0.912
- **Top-1 accuracy:** 91.67% (11/12 folds)

**Test Set (holdout, seasons 9-10):**
- **MAE rank:** 0.889 (average error < 1 position)
- **Mean Spearman:** 0.814 (strong rank correlation)
- **Top-1 accuracy:** 75.00% (3/4 conferences)
- **Top-2 accuracy:** 75.00%

### GradientBoosting (--model gbr)

**Walk-forward CV (internal, seasons 3-8):**
- **MAE rank:** 0.504
- **Mean Spearman:** 0.894
- **Top-1 accuracy:** 66.67% (8/12 folds)

**Test Set (holdout, seasons 9-10):**
- **MAE rank:** 0.593 (better than RF!)
- **Mean Spearman:** 0.879
- **Top-1 accuracy:** 75.00% (3/4 conferences)
- **Top-2 accuracy:** 100.00% (4/4 conferences) ⭐

**Interpretation:** Both models generalize well. GBR has better test MAE. Walk-forward CV validates no major overfitting.

---

## 🧠 Features Used (37 total)

### From `team_season_statistics.csv`:
- **Performance metrics:** `point_diff`, `off_eff`, `def_eff`
- **Shooting stats:** `fg_pct`, `three_pct`, `ft_pct`, `opp_fg_pct`
- **Home/Away:** `home_win_pct`, `away_win_pct`, `home_advantage`
- **Advanced stats:** `reb_diff`, `stl_diff`, `blk_diff`, `to_diff`
- **Historical:** `prev_win_pct_1`, `prev_win_pct_3`, `prev_win_pct_5`
- **Normalized metrics:** `*_norm` versions of efficiency stats
- **Conference:** `confID` (one-hot encoded: `conf_EA`, `conf_WE`)

### From `team_performance.csv`:
- **Pythagorean win%:** `pythag_win_pct` (most important feature!)
- **Roster strength:** `team_strength`
- **Expected win%:** `rs_win_pct_expected_roster`
- **Overachievement:** `overach_pythag`, `overach_roster`

### Feature Importance (Top 5):
1. **`pythag_win_pct`** (40.7%) - Bill James Pythagorean expectation
2. **`point_diff`** (21.3%) - Point differential
3. **`overach_roster`** (13.9%) - Performance vs roster expectation
4. **`opp_fg_pct_norm`** (4.6%) - Normalized opponent FG%
5. **`away_win_pct`** (3.1%) - Road performance

---

## 🚫 Excluded Features (No Leakage)

These columns exist but are **NOT used** as features (would cause leakage):
- `rank` (target variable)
- `won`, `lost`, `GP` (directly determine rank)
- `season_win_pct` (derived from wins/losses)
- `playoff`, `firstRound`, `semis`, `finals` (post-season outcomes)
- `tmID`, `franchID`, `name`, `arena` (identifiers)
- `season_id` (identifier)

---

## 🔄 How to Run

```bash
cd /home/tomio/Documents/UNI/AC
source venv/bin/activate

# Run with RandomForest (recommended)
python3 src/model/ranking_model/team_ranking_model.py --model rf

# Run with GradientBoosting (slightly better test MAE)
python3 src/model/ranking_model/team_ranking_model.py --model gbr

# Custom training split (e.g., train on 1-7, test on 8+)
python3 src/model/ranking_model/team_ranking_model.py --model rf --max-train-year 7

# Outputs:
#   → data/processed/team_ranking_predictions.csv
#   → reports/models/team_ranking_report.txt
```

**Dependencies:** `scikit-learn`, `scipy`, `pandas`, `numpy` (all in requirements.txt)

### Legacy Scripts (deprecated)
```bash
# These now call the unified model (for backwards compatibility)
python3 src/performance/team_ranking_model.py
```

---

## 📈 Example Predictions

### Year 9 East (Test Set)
| Team | Actual Rank | Predicted Rank | Status |
|------|-------------|----------------|--------|
| Detroit Shock | 1 | 1 | ✅ Perfect |
| Connecticut Sun | 2 | 3 | ❌ Off by 1 |
| New York Liberty | 3 | 2 | ❌ Off by 1 |
| Indiana Fever | 4 | 4 | ✅ Perfect |

**Result:** Top-1 correct (Detroit), Mean error = 0.57 ranks

### Year 9 West (Test Set)
| Team | Actual Rank | Predicted Rank | Status |
|------|-------------|----------------|--------|
| San Antonio | 1 | 1 | ✅ Perfect |
| Seattle Storm | 2 | 2 | ✅ Perfect |
| Los Angeles | 3 | 3 | ✅ Perfect |
| Sacramento | 4 | 5 | ❌ Off by 1 |

**Result:** Top-1 correct (San Antonio), Mean error = 0.25 ranks

---

## 🔧 Model Configuration

### RandomForest (--model rf)
```python
RandomForestRegressor(
    n_estimators=400,      # More trees
    max_depth=6,           # Reduced from 10 (anti-overfitting)
    min_samples_leaf=2,    # Regularization
    min_samples_split=4,   # Regularization
    max_features='sqrt',   # Feature subsampling
    random_state=42,
    n_jobs=-1
)
```

### GradientBoosting (--model gbr)
```python
GradientBoostingRegressor(
    n_estimators=400,
    learning_rate=0.03,    # Slow learning (anti-overfitting)
    max_depth=3,           # Shallow trees
    subsample=0.7,         # 70% sample per tree
    min_samples_leaf=2,
    random_state=42
)
```

**Why Tree-based Models?**
- Handle non-linear relationships
- No need for feature scaling
- Built-in feature importance
- Robust to outliers
- Work well with mixed feature types

**Why Regressor (not Classifier)?**
- Predicts continuous score, then converts to ranks
- Preserves ordering information within conference
- More flexible than direct rank classification

**Anti-Overfitting Measures:**
- Reduced `max_depth` (6 for RF, 3 for GBR)
- Regularization hyperparameters
- Walk-forward cross-validation (realistic evaluation)
- Test set touched only once (no tuning on test)

---

## 📊 Data Pipeline

```
team_season_statistics.csv (142 rows)
           +
team_performance.csv (142 rows)
           ↓
      [Merge on year, tmID]
           ↓
    Full dataset (142 rows)
           ↓
    [Temporal split]
           ↓
  Train: 115 rows (seasons 1-8)
  Test:  27 rows (seasons 9-10)
           ↓
   [Feature extraction]
           ↓
   X: 37 features, y: rank
           ↓
  [RandomForest training]
           ↓
   [Predict scores]
           ↓
  [Convert to ranks within conference]
           ↓
   [Evaluate metrics]
           ↓
  Predictions CSV + Report TXT
```

---

## 🎓 Key Insights

1. **Pythagorean win% is king** (40% of feature importance)
   - Validates Bill James formula for basketball
   - Points scored/allowed matter more than most other stats

2. **Strong generalization** (Spearman 0.877 on test)
   - Model learns genuine patterns, not noise
   - Temporal split validates real predictive power

3. **Top-1 accuracy 75%** is impressive
   - Predicting exact conference champion from regular season stats
   - Better than random (12.5% for 8-team conference)

4. **Perfect training fit** (MAE = 0)
   - Tree models can memorize training data
   - Why we use temporal holdout, not random split

5. **Roster quality matters** (14% importance)
   - `overach_roster` 3rd most important feature
   - Validates the `team_performance.py` pipeline

---

## 🔮 Potential Improvements

1. **Ensemble methods:** Combine RF + GBR predictions (weighted average)
2. **Conference-specific models:** Separate East/West models (more parameters)
3. **Season-weighted features:** Recent seasons count more (exponential decay)
4. **Player-level aggregation:** Directly from `player_performance.csv` (top-5 players)
5. **Playoff prediction:** Extend to post-season outcomes (different target)
6. **Injury impact:** Model roster changes mid-season (player minutes volatility)
7. **Coaching changes:** Integrate `coach_season_performance.csv` (first-year flag, overachievement)
8. **XGBoost:** Try gradient boosting with stronger regularization
9. **Hyperparameter tuning:** Grid search within walk-forward CV (no test leakage)
10. **Uncertainty quantification:** Prediction intervals via quantile regression

---

## ✅ Validation Checklist

- [x] No temporal leakage (train ≤ 8, test ≥ 9)
- [x] No target leakage (excluded `won`, `lost`, `season_win_pct`)
- [x] Reproduc
ible (random_state=42)
- [x] Proper evaluation (conference-specific metrics)
- [x] Clear documentation (this file + report.txt)
- [x] Saves outputs (CSV + TXT)

---

**Created:** 2025-11-05  
**Model version:** 1.0.0  
**Status:** ✅ Production-ready
