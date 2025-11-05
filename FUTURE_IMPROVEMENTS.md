# Future Improvements - Team Ranking Model

## 🎯 Current Status

**Model:** GradientBoostingRegressor (v2.0.0)
**Performance:** Test MAE = 0.593, Spearman = 0.879, Top-2 Acc = 100%
**Status:** Production-ready baseline

---

## 📊 Structural Limitations (No Silver Bullet)

### Data Volume Constraint
- **Training data:** 115 rows (seasons 1-8)
- **Test data:** 27 rows (seasons 9-10) = only 4 groups (2 years × 2 conferences)
- **Total team-seasons:** ~142

**Reality check:** With 8 training seasons, MAE around 0.5-0.9 positions is **close to the noise floor** of this problem. Injuries, mid-season trades, coaching drama, and other unpredictable factors set a natural performance ceiling.

**Implication:** Can't significantly "generalize better" without more data. Any smarter model will hit the variance limit quickly.

### For the Report
> "The main limitation for further improvements is data volume (≈8 seasons for training). With so few time points, test MAE around 0.6–0.9 positions is already close to the noise floor of the problem. Unpredictable factors (injuries, mid-season trades, coaching changes) set a natural performance ceiling that no amount of model tuning can overcome without additional data."

---

## 🔧 Real Improvement Opportunities

### 1. Player Performance Pipeline (`player_performance.py`)

**Current approach (already sophisticated):**
- Aggregates stints → player-year-team
- Uses per-36 minutes normalization
- Applies position-specific weights from `weights_positions.json`
- Fallback by team/position for low-minutes players

#### Improvement 1.1: Recalibrate Position Weights with Honest Validation

**Current:** `_run_role_regressions` uses all data:
```python
minutes_next ~ pts36 + reb36 + ast36 + stl36 + blk36 + tov36
```

**Better approach:**
1. Split by odd/even seasons
2. Fit weights on even seasons
3. Validate on odd seasons (predict `minutes_next`)
4. If holdout performance is poor, simplify:
   - Drop stats with insignificant weights
   - Add manual regularization (zero out noisy features)
   - Re-fit with reduced feature set

**Expected gain:** More stable `performance` metric, less noise in `team_strength`.

#### Improvement 1.2: Prune Noisy Stats from Weights

**Action plan:**
1. Sort weights by `|value|` per role (guard/wing/big)
2. Identify stats with:
   - Very small absolute weight (< 0.05)
   - High variance across roles
   - Counter-intuitive signs
3. Manually zero out these stats
4. Recalculate `performance` with pruned weights
5. Compare `team_strength` → `rank` correlation before/after

**Expected gain:** Signal-to-noise ratio improvement, fewer spurious correlations.

#### Improvement 1.3: Check `performance` Distribution

**Issue:** If distribution is heavily skewed (most players at 0, few at +15), model sees weak signal.

**Solution:**
```python
# Option A: Standardize by position (z-score per role)
performance_z = (performance - mean_by_role) / std_by_role

# Option B: Quantile-based normalization
performance_quantile = rank(performance) / n_players_in_role
```

**Expected gain:** Prevents centers from dominating aggregate metrics, more balanced signal.

---

### 2. Team Performance Pipeline (`team_performance.py`)

**Current output:**
- `pythag_win_pct` (Bill James formula)
- `team_strength` (weighted average of player performance)
- `rs_win_pct_expected_roster` (linear regression on `team_strength`)
- `overach_pythag`, `overach_roster` (residuals)

#### Improvement 2.1: Check `overach_*` for Near-Leakage

**Concern:** While not direct leakage, `overach_pythag` and `overach_roster` are derived from `rs_win_pct`:
```python
overach_pythag = rs_win_pct - pythag_win_pct
overach_roster = rs_win_pct - rs_win_pct_expected_roster
```

This is a **"second pathway"** to the same information (how well the team actually did).

**Experiment:**
1. Train model **WITHOUT** `overach_pythag` and `overach_roster`
2. Keep only:
   - `pythag_win_pct`
   - `team_strength`
   - Historical features (`prev_win_pct_*`, `prev_point_diff_*`)
3. Measure test MAE

**Expected outcomes:**
- **If MAE increases slightly (0.59 → 0.7):** Acceptable tradeoff for cleaner interpretation and better generalization
- **If MAE stays similar:** Strong evidence that model doesn't need "storytelling" features
- **If MAE increases significantly (>1.0):** Current features are necessary

#### Improvement 2.2: Regularize `team_strength` → `rs_win_pct_expected_roster`

**Current:** Simple `LinearRegression`

**Better approach:**
```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)  # L2 regularization
model.fit(team_strength.reshape(-1, 1), rs_win_pct)
```

**Why:** With only ~142 data points, OLS can produce unstable coefficients. Ridge prevents extreme slopes.

**Additional ideas:**
- Add features to roster expectation model:
  - `prev_win_pct_1` (franchise momentum)
  - `franchise_changed` (new team penalty)
  - Conference dummies (different competitive levels)
- Split by conference (separate East/West regressions)

**Expected gain:** More stable `rs_win_pct_expected_roster`, better `overach_roster` signal.

#### Improvement 2.3: Feature Normalization Audit

**Current:** Some `*_norm` features exist (good!)

**Check:**
1. Range consistency: All numeric features in comparable scales?
2. No "quasi-label" features: Nothing that's a disguised form of rank?
3. Temporal stability: Do normalized features behave consistently across seasons?

**Action:** If any feature has suspicious patterns, investigate or remove.

---

### 3. Model Regularization (Even More Anti-Overfitting)

**Current concern:** MAE train = 0.0 (perfect fit) might make reviewers nervous.

#### Option 3.1: More Aggressive RandomForest

**Current:**
```python
RandomForestRegressor(
    n_estimators=400,
    max_depth=6,
    min_samples_leaf=2,
    min_samples_split=4,
    max_features='sqrt'
)
```

**More conservative:**
```python
RandomForestRegressor(
    n_estimators=200,      # Fewer trees
    max_depth=4,           # Shallower trees
    min_samples_leaf=4,    # More samples per leaf
    min_samples_split=8,   # More samples to split
    max_features=0.5,      # Even more randomness
    random_state=42,
    n_jobs=-1
)
```

**Expected:** Train MAE will rise to 0.2-0.4. If test MAE stays similar or improves, use it. If test MAE worsens significantly, current config is optimal.

#### Option 3.2: More Aggressive GradientBoosting

**Current:**
```python
GradientBoostingRegressor(
    n_estimators=400,
    learning_rate=0.03,
    max_depth=3,
    subsample=0.7,
    min_samples_leaf=2
)
```

**Even more conservative:**
```python
GradientBoostingRegressor(
    n_estimators=300,      # Fewer boosting rounds
    learning_rate=0.02,    # Smaller learning rate
    max_depth=2,           # Very shallow trees
    subsample=0.6,         # More stochastic
    min_samples_leaf=5,    # Lower variance
    random_state=42
)
```

**Expected:** Similar to RF - if test performance is stable, proves current config is well-tuned.

#### Option 3.3: Simple Ensemble (RF + GBR)

**Implementation:**
```python
# Train both models
rf_model = RandomForestRegressor(...)
gbr_model = GradientBoostingRegressor(...)

rf_model.fit(X_train, y_train)
gbr_model.fit(X_train, y_train)

# Predict with both
y_pred_rf = rf_model.predict(X_test)
y_pred_gbr = gbr_model.predict(X_test)

# Average scores before ranking
y_pred_ensemble = 0.5 * y_pred_rf + 0.5 * y_pred_gbr

# Rank within (year, confID)
df_test['pred_score'] = y_pred_ensemble
df_test['pred_rank'] = df_test.groupby(['year', 'confID'])['pred_score'].rank(...)
```

**Why:** With small datasets, ensembles often stabilize outlier predictions from individual models.

**Expected:** Slight improvement in test MAE or Spearman (0.59 → 0.55?), more robust rankings.

---

## 📝 For the Report (Discussion Section)

### English Version
> "Given the small number of seasons available, we validated the model using a strict walk-forward scheme and a temporal holdout (seasons 9–10). Although tree ensembles perfectly fit the training set (MAE=0 on train, as expected for high-capacity models), the cross-validation and test errors remain in the 0.5–0.9 range, with Spearman correlations above 0.8.
>
> The main avenues for further improvement are:
> 1. **Data expansion:** Gathering more seasons or richer play-by-play features
> 2. **Pipeline refinement:** Improving player and team performance metrics (e.g., position-specific weight validation, regularized roster-based expected wins)
> 3. **Model tuning:** Testing stronger regularization (shallower trees, smaller learning rate) and simple ensembles (RF+GBR)
>
> However, the fundamental constraint is data volume. With only 8 training seasons, the model is already near the performance ceiling imposed by the inherent unpredictability of sports outcomes (injuries, trades, coaching dynamics)."

### Portuguese Version
> "O limite principal hoje é a quantidade de épocas disponíveis. Com aproximadamente 8 épocas de treino, o modelo já está perto do máximo que é possível extrair sem adicionar novas fontes de dados (e.g., play-by-play detalhado, informação sobre lesões e trocas).
>
> A pipeline de performance de jogadores e equipas já é sofisticada (normalização per-36 por posição, Pythagorean expectation, regressão de roster strength). Melhorias futuras passariam por:
>
> 1. **Refinar métricas:** Validação temporal dos pesos por posição, regularização das regressões de expectativa
> 2. **Testar regularizações mais agressivas:** Árvores mais rasas, learning rates menores
> 3. **Ensembles simples:** Combinação de RandomForest + GradientBoosting
>
> Contudo, o constrangimento fundamental é o volume de dados. Com apenas 8 épocas de treino, o modelo já se aproxima do limite de performance imposto pela imprevisibilidade inerente ao desporto (lesões, mudanças táticas, dinâmicas de balneário)."

---

## 🎯 Priority Ranking

### High Priority (Significant Expected Gain)
1. **Test without `overach_*` features** (cleaner baseline, potential for better generalization)
2. **Recalibrate position weights with temporal validation** (reduces noise in player performance)
3. **Ridge regularization for roster expectation** (more stable expected wins)

### Medium Priority (Moderate Expected Gain)
4. **Prune noisy stats from position weights** (better signal-to-noise)
5. **Simple RF+GBR ensemble** (potentially lower test MAE by 0.05-0.1)
6. **Check performance distribution by position** (prevents position bias)

### Low Priority (Marginal Gain, Good for Robustness)
7. **More aggressive regularization experiments** (proves current config is optimal)
8. **Feature normalization audit** (catches edge cases)

---

## 🚫 What NOT to Do

1. **Don't add more complex models** (XGBoost, neural nets, etc.) - data volume doesn't support them
2. **Don't tune hyperparameters on test set** - defeats the purpose of holdout validation
3. **Don't add features derived from test seasons** - temporal leakage
4. **Don't over-interpret small differences** - with 4 test groups, variance is high

---

## 💡 One-Liner Summary

**If using for course now:** Already solid, not "shitty" or problematic overfitting.

**If going hardcore nerd mode:** Focus on *performance pipeline* (weights, roster expectations) and *regularization experiments* rather than inventing new magical models. The bottleneck is **data volume and signal quality**, not algorithm sophistication.

---

**Status:** Roadmap for future work  
**Priority:** Low (current model is production-ready)  
**Expected ROI:** Marginal improvements (0.05-0.15 MAE reduction)  
**Time investment:** High (weeks of experimentation)

**Recommendation:** Use current model as-is for course. Revisit if/when more seasons become available or for research publication.

