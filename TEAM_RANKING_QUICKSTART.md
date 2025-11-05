# Team Ranking Model - Quick Start Guide

## 🚀 TL;DR

```bash
cd /home/tomio/Documents/UNI/AC
source venv/bin/activate

# Best model (GradientBoosting, lowest test MAE)
python3 src/model/ranking_model/team_ranking_model.py --model gbr

# Outputs:
#   → data/processed/team_ranking_predictions.csv
#   → reports/models/team_ranking_report.txt
```

---

## 📋 What This Model Does

Predicts **regular season conference rankings** (1st, 2nd, 3rd, ... within East/West) based on:
- Team performance metrics (Pythagorean win%, roster strength, point differential)
- Offensive/defensive stats (FG%, 3PT%, rebounds, turnovers, etc.)
- Historical trends (previous season win%, changes, etc.)

**No data leakage:** Never uses `won`, `lost`, `GP`, or playoff outcomes as features.

**Temporal split:** Train on seasons 1-8, test on seasons 9-10 (no random shuffle).

**Walk-forward CV:** Internally validates each season using only previous seasons.

---

## 🎯 Which Model to Use?

| Model | Command | Test MAE | Test Spearman | Top-1 Acc | Top-2 Acc | Recommendation |
|-------|---------|----------|---------------|-----------|-----------|----------------|
| **GradientBoosting** | `--model gbr` | **0.593** | 0.879 | 75% | **100%** | ⭐ **Best test performance** |
| **RandomForest** | `--model rf` | 0.889 | 0.814 | 75% | 75% | More interpretable features |

**Recommendation:** Use `gbr` for production baseline, `rf` for feature analysis.

---

## 📊 Performance Metrics Explained

- **MAE rank:** Average error in predicted position (e.g., 0.593 = off by ~0.6 positions)
- **Spearman correlation:** How well the model preserves team ordering (1.0 = perfect)
- **Top-1 accuracy:** % of conferences where model predicts exact champion
- **Top-2 accuracy:** % of conferences where true champion is in predicted top-2

**Walk-forward CV:** Shows realistic performance (not inflated by overfitting).

---

## 🔧 Command-Line Options

```bash
# Basic usage
python3 src/model/ranking_model/team_ranking_model.py --model {rf|gbr}

# Custom training split (e.g., train on 1-7, test on 8+)
python3 src/model/ranking_model/team_ranking_model.py --model gbr --max-train-year 7

# Custom report name
python3 src/model/ranking_model/team_ranking_model.py --model rf --report-name my_report.txt

# Help
python3 src/model/ranking_model/team_ranking_model.py --help
```

---

## 📁 Output Files

### 1. Predictions CSV
**Path:** `data/processed/team_ranking_predictions.csv`

| Column | Description |
|--------|-------------|
| `year` | Season (1-10) |
| `confID` | Conference (EA=East, WE=West) |
| `tmID` | Team ID (e.g., "DET", "SAS") |
| `name` | Team name |
| `rank` | True final rank (1=best) |
| `pred_rank` | Predicted rank (1=best) |
| `pred_score` | Raw model score (lower=better) |
| `split` | "train" or "test" |

**Usage examples:**
- Compare `pred_rank` vs `rank` to identify over/underperforming teams
- Use `pred_rank` as baseline for Coach of the Year analysis
- Filter by `split == "test"` to see holdout predictions

### 2. Report TXT
**Path:** `reports/models/team_ranking_report.txt`

Contains:
- Model configuration (hyperparameters)
- Walk-forward CV metrics
- Train/test metrics
- Top 15 feature importance
- Example predictions for Year 9 East/West
- Anti-leakage checklist
- Interpretation guide

---

## 🧠 Top 5 Most Important Features

### GradientBoosting (gbr)
1. **pythag_win_pct** (31%) - Bill James Pythagorean expectation
2. **point_diff** (23%) - Season point differential
3. **overach_roster** (15%) - Performance vs roster strength
4. **overach_pythag** (5%) - Performance vs Pythagorean expectation
5. **opp_fg_pct_norm** (3%) - Opponent FG% (normalized)

### RandomForest (rf)
1. **point_diff** (17%)
2. **overach_roster** (14%)
3. **pythag_win_pct** (13%)
4. **home_win_pct** (8%)
5. **away_win_pct** (8%)

**Takeaway:** Pythag and roster strength are the primary drivers.

---

## 🎓 Use Cases

### 1. Coach of the Year (COY) Analysis
**Problem:** Which coaches exceed expectations?

**Solution:** Compare actual team rank vs `pred_rank` (baseline strength).

```python
# Pseudocode
coach_overachieve = actual_rank - pred_rank  # Negative = exceeded expectations
```

**Example:** If team has `pred_rank=5` but finishes `rank=2`, coach exceeded baseline by 3 positions.

### 2. Player Impact Evaluation
**Problem:** How do roster changes affect team strength?

**Solution:** Compare `pred_rank` before/after roster changes (e.g., rookie additions).

### 3. Playoff Predictions
**Problem:** Which teams will make playoffs?

**Solution:** Use `pred_rank` as strong indicator (top-4 in conference usually make playoffs).

---

## ⚠️ Important Caveats

### What This Model Does NOT Do
- **Mid-season predictions:** Model uses full-season stats (not game-by-game)
- **Playoff performance:** Only predicts regular season rank
- **Injuries/trades:** Doesn't account for mid-season roster changes
- **Conference realignment:** Assumes fixed East/West structure

### Data Leakage Prevention
The model **NEVER** uses:
- `won`, `lost`, `GP` (directly determine rank)
- `season_win_pct` (derived from wins/losses)
- Playoff outcomes (`playoff`, `firstRound`, `semis`, `finals`)
- Split W-L records (`homeW`, `homeL`, `awayW`, `awayL`, `confW`, `confL`)

### Overfitting Prevention
- **Regularized hyperparameters:** Reduced `max_depth`, low `learning_rate`
- **Walk-forward CV:** Realistic evaluation (each season uses only past)
- **Test set touched once:** No hyperparameter tuning on test
- **Temporal split:** No random shuffle (respects time ordering)

---

## 🔄 Migration from Old Scripts

### If you were using:
- `src/performance/team_ranking_model.py` (old RandomForest)
- `src/model/ranking_model/ranking_model.py` (old GradientBoosting)

### What changed:
- ✅ Scripts still work (now wrappers calling unified model)
- ✅ Output format unchanged (`team_ranking_predictions.csv` same columns)
- ✅ Behavior preserved (seasons 1-8 train, 9-10 test)

### What's new:
- ✨ Walk-forward CV (better overfitting detection)
- ✨ Regularized hyperparameters (reduced overfitting)
- ✨ Unified API (one script, two models)
- ✨ Better reporting (CV + train + test metrics)

### Action required:
**None.** Old scripts still work. For new work, use:
```bash
python3 src/model/ranking_model/team_ranking_model.py --model {rf|gbr}
```

---

## 📚 Further Reading

- **Full README:** `TEAM_RANKING_MODEL_README.md` (detailed architecture)
- **Feature engineering:** See `team_season_statistics.csv` columns
- **Performance metrics:** See `team_performance.csv` (Pythag, roster strength)
- **Coach analysis:** See `src/performance/coach_season_performance.py`

---

## 🐛 Troubleshooting

### Error: "Missing columns in team_season_statistics.csv"
**Solution:** Run feature engineering pipeline first:
```bash
python3 src/cleaning/clean_teams.py
python3 src/performance/team_performance.py
```

### Error: "ModuleNotFoundError: No module named 'sklearn'"
**Solution:** Activate venv and install dependencies:
```bash
source venv/bin/activate
pip install scikit-learn scipy
```

### Performance seems too good (MAE=0.0 on train)
**Normal.** Tree models can memorize training data. Check:
- Walk-forward CV MAE (should be ~0.4-0.5)
- Test MAE (should be ~0.6-0.9)

If CV/test are also 0.0, you have data leakage. Check features.

---

## 📞 Questions?

- Check `reports/models/team_ranking_report.txt` for detailed metrics
- Read `TEAM_RANKING_MODEL_README.md` for architecture details
- Inspect `src/model/ranking_model/team_ranking_model.py` for implementation

---

**Version:** 2.0.0 (unified & robust)  
**Last Updated:** 2025-11-05  
**Status:** ✅ Production-ready

