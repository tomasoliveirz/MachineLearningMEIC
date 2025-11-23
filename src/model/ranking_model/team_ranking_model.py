from pathlib import Path
from typing import Tuple, Dict
from datetime import datetime
import warnings
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error
from model_graphics import generate_all_graphics

warnings.filterwarnings('ignore')

# Paths
ROOT = Path(__file__).resolve().parents[3]
PROC_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports" / "models"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Random seed for reproducibility
RANDOM_STATE = 42

# =============================================================================
# 1. DATA LOADING AND VALIDATION
# =============================================================================

def load_and_merge() -> pd.DataFrame:
    """Load and merge team season statistics with performance data."""
    print("[TeamRanking] Loading data...")
    
    # Load teams data (enriched with features)
    stats_path = PROC_DIR / "team_season_statistics.csv"
    if not stats_path.exists():
        raise FileNotFoundError(f"team_season_statistics.csv not found in {PROC_DIR}")
    
    df_stats = pd.read_csv(stats_path)
    print(f"  ✓ Loaded {len(df_stats)} records from team_season_statistics.csv")
    
    # Load team performance data (Pythag, roster strength, etc.)
    perf_path = PROC_DIR / "team_performance.csv"
    if not perf_path.exists():
        raise FileNotFoundError(f"team_performance.csv not found in {PROC_DIR}")
    
    df_perf = pd.read_csv(perf_path)
    print(f"  ✓ Loaded {len(df_perf)} records from team_performance.csv")
    
    # Merge on year and team
    df_all = df_stats.merge(
        df_perf,
        left_on=['year', 'tmID'],
        right_on=['year', 'team_id'],
        how='left'
    )
    
    print(f"  ✓ Merged dataset: {len(df_all)} rows")
    
    # Validate required columns exist
    required = ['year', 'tmID', 'confID', 'rank', 'name',
                'pythag_win_pct', 'overach_roster', 'overach_pythag',
                'team_strength', 'rs_win_pct_expected_roster']
    missing = set(required) - set(df_all.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    
    # Remove rows without rank or confID
    df_all = df_all.dropna(subset=['rank', 'confID'])
    print(f"  ✓ After removing missing rank/confID: {len(df_all)} rows")
    
    return df_all

# =============================================================================
# 2. TEMPORAL FEATURES (Rolling Averages & Trends)
# =============================================================================

def add_coach_career_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute coach overachievement RESULTS (in-memory) and then create 
    predictive features (rolling averages with shift(1)).
    
    This function:
    1. Calculates overachievement for each season (RESULT, not usable)
    2. Creates lagged rolling averages (PREDICTIVE features)
    """
    print("[TeamRanking] Adding coach career features (predictive, no leakage)...")
    
    # Check if coach data exists
    if 'coachID' not in df.columns:
        print("  ⚠ No coach data found, skipping coach features")
        return df
    
    df = df.sort_values(['coachID', 'year']).copy()
    
    # --- STEP 1: Calculate RESULTS for the season (in-memory) ---
    # These columns CANNOT be used directly in the model
    
    # Find coach win% column
    rs_col = None
    for col in ['rs_win_pct_coach', 'coach_rs_win_pct']:
        if col in df.columns:
            rs_col = col
            break
    
    if rs_col is None:
        print("  ⚠ Coach win% column not found, skipping coach features")
        return df
    
    # Calculate overachievement RESULTS (current season)
    if 'pythag_win_pct' in df.columns:
        df['coach_overach_pythag_RESULT'] = df[rs_col] - df['pythag_win_pct']
        df['coach_overach_pythag_RESULT'] = df['coach_overach_pythag_RESULT'].fillna(0.0)
    
    if 'rs_win_pct_expected_roster' in df.columns:
        df['coach_overach_roster_RESULT'] = df[rs_col] - df['rs_win_pct_expected_roster']
        df['coach_overach_roster_RESULT'] = df['coach_overach_roster_RESULT'].fillna(0.0)
    
    # --- STEP 2: Create PREDICTIVE FEATURES (with shift(1)) ---
    # The model will ONLY use these _ma3 columns (3-year rolling averages)
    
    # Career overachievement vs Pythagorean
    if 'coach_overach_pythag_RESULT' in df.columns:
        df['coach_career_overach_pythag_ma3'] = df.groupby('coachID')['coach_overach_pythag_RESULT'].transform(
            lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
        ).fillna(0.0)
    
    # Career overachievement vs Roster
    if 'coach_overach_roster_RESULT' in df.columns:
        df['coach_career_overach_roster_ma3'] = df.groupby('coachID')['coach_overach_roster_RESULT'].transform(
            lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
        ).fillna(0.0)
    
    # Career win% (smoothed)
    df['coach_career_rs_win_pct_ma3'] = df.groupby('coachID')[rs_col].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    ).fillna(0.0)
    
    # Coach tenure (years with same team)
    if 'team_id' in df.columns or 'tmID' in df.columns:
        team_col = 'team_id' if 'team_id' in df.columns else 'tmID'
        df = df.sort_values([team_col, 'coachID', 'year'])
        df['coach_tenure'] = df.groupby([team_col, 'coachID']).cumcount() + 1
        df['coach_tenure_prev'] = df.groupby([team_col, 'coachID'])['coach_tenure'].shift(1).fillna(0).astype(int)
    else:
        df['coach_tenure_prev'] = 0
    
    # Count new predictive features
    coach_features = [
        'coach_career_overach_pythag_ma3',
        'coach_career_overach_roster_ma3', 
        'coach_career_rs_win_pct_ma3',
        'coach_tenure_prev'
    ]
    existing_features = [f for f in coach_features if f in df.columns]
    
    print(f"  ✓ Added {len(existing_features)} coach career features")
    
    return df

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal rolling and trend features using past seasons (no leakage)."""
    print("[TeamRanking] Adding temporal features (rolling averages & trends)...")
    
    df = df.sort_values(['tmID', 'year']).copy()
    
    # Columns to compute temporal features for
    temporal_cols = ['point_diff', 'off_eff', 'def_eff', 'pythag_win_pct', 'team_strength']
    
    # Ensure columns exist
    for col in temporal_cols:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    
    # Group by team
    for col in temporal_cols:
        # Rolling mean (3 and 5 years) - using only past data
        df[f'{col}_ma3'] = df.groupby('tmID')[col].transform(
            lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
        )
        df[f'{col}_ma5'] = df.groupby('tmID')[col].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
        )
        
        # Trend (slope) over 3 and 5 years
        def calculate_slope(series, window):
            """Calculate slope of linear regression over window"""
            if len(series) < 2:
                return 0.0
            x = np.arange(len(series))
            y = series.values
            if np.std(y) == 0:
                return 0.0
            # Simple linear regression: slope = cov(x,y) / var(x)
            slope = np.cov(x, y)[0, 1] / (np.var(x) + 1e-10)
            return slope
        
        df[f'{col}_trend3'] = df.groupby('tmID')[col].transform(
            lambda x: x.shift(1).rolling(window=3, min_periods=2).apply(
                lambda series: calculate_slope(series, 3), raw=False
            )
        )
        df[f'{col}_trend5'] = df.groupby('tmID')[col].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=2).apply(
                lambda series: calculate_slope(series, 5), raw=False
            )
        )
    
    # Fill NaN values with 0 (for teams with insufficient history)
    temporal_feature_cols = [
        f'{col}_{suffix}' 
        for col in temporal_cols 
        for suffix in ['ma3', 'ma5', 'trend3', 'trend5']
    ]
    df[temporal_feature_cols] = df[temporal_feature_cols].fillna(0.0)
    
    print(f"  ✓ Added {len(temporal_feature_cols)} temporal features")
    
    return df

# =============================================================================
# 3. TEMPORAL SPLIT (Train/Validation/Test) AND WALK-FORWARD VALIDATION
# =============================================================================

def split_train_val_test(
    df_all: pd.DataFrame, 
    max_train_year: int = 8,
    val_years: int = 2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Temporal split with validation set (no shuffle).
    
    Args:
        df_all: Full dataset
        max_train_year: Last year in test set (e.g., 8)
        val_years: Number of years for validation (e.g., 2 → years 7-8)
    
    Returns:
        train_df: Years 1 to (max_train_year - val_years)
        val_df: Years (max_train_year - val_years + 1) to max_train_year
        test_df: Years > max_train_year
    """
    val_start_year = max_train_year - val_years + 1
    
    train_df = df_all[df_all['year'] < val_start_year].copy()
    val_df = df_all[(df_all['year'] >= val_start_year) & (df_all['year'] <= max_train_year)].copy()
    test_df = df_all[df_all['year'] > max_train_year].copy()
    
    print(f"  Train years: 1-{val_start_year-1} ({len(train_df)} samples)")
    print(f"  Val years: {val_start_year}-{max_train_year} ({len(val_df)} samples)")
    print(f"  Test years: {max_train_year+1}+ ({len(test_df)} samples)")
    
    return train_df, val_df, test_df


def generate_walk_forward_splits(
    df_all: pd.DataFrame,
    max_train_year: int = 8,
    val_years: int = 2
) -> list:
    """Generate walk-forward (rolling) validation folds.
    For each validation year in the range (max_train_year - val_years + 1) .. max_train_year
    create a fold: train = years < val_year, val = year == val_year.
    """
    folds = []
    val_start = max_train_year - val_years + 1
    for val_year in range(val_start, max_train_year + 1):
        train = df_all[df_all['year'] < val_year].copy()
        val = df_all[df_all['year'] == val_year].copy()
        folds.append((train, val))
    return folds


# =============================================================================
# 4. FEATURE ENGINEERING WITH NO LEAKAGE AND REDUCED SET TO PREVENT OVERFITTING
# =============================================================================

def build_feature_matrix(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build feature matrix X, target y and meta (predictive-only, no leakage)."""
    
    # OLD features (before reduction to prevent overfitting)
    Old = [
        # Historical performance (from previous seasons only)
        'prev_win_pct_1', 'prev_win_pct_3', 'prev_win_pct_5',
        'prev_point_diff_3', 'prev_point_diff_5',
        'win_pct_change',
        
        # Roster quality (can be estimated pre-season from player metrics)
        'team_strength',
        
        # Rolling averages and trends (computed from past seasons with shift(1))
        'point_diff_ma3', 'point_diff_ma5', 'point_diff_trend3', 'point_diff_trend5',
        'off_eff_ma3', 'off_eff_ma5', 'off_eff_trend3', 'off_eff_trend5',
        'def_eff_ma3', 'def_eff_ma5', 'def_eff_trend3', 'def_eff_trend5',
        'pythag_win_pct_ma3', 'pythag_win_pct_ma5', 'pythag_win_pct_trend3', 'pythag_win_pct_trend5',
        'team_strength_ma3', 'team_strength_ma5', 'team_strength_trend3', 'team_strength_trend5',
        
        # Structural context (not tied to game results)
        'franchise_changed',
        
        # Coach career features (predictive, lagged)
        'coach_career_overach_pythag_ma3',
        'coach_career_overach_roster_ma3',
        'coach_career_rs_win_pct_ma3',
        'coach_tenure_prev',
    ]
    # New features (reduced set to prevent overfitting)
    feature_cols_numeric = [
        # Core historical performance (most stable)
        'prev_win_pct_3', 'prev_win_pct_5',
        'prev_point_diff_5',
        
        # Key rolling averages (removed short-term MA3 to reduce noise)
        'point_diff_ma5', 'point_diff_trend5',
        'off_eff_ma5', 'def_eff_ma5',
        'pythag_win_pct_ma5', 'pythag_win_pct_trend5',
        'team_strength_ma5',
        
        # Coach features (career averages only, most stable)
        'coach_career_rs_win_pct_ma3',
        'coach_tenure_prev',
    ]    

    df_work = df.copy()
    # Ensure numeric features exist and are float (no silent NaNs)
    for col in feature_cols_numeric:
        if col not in df_work.columns:
            # Fill missing feature with 0 (safe default)
            df_work[col] = 0.0
        else:
            df_work[col] = pd.to_numeric(df_work[col], errors='coerce').fillna(0.0)
    
    # One-hot encode confID (East/West)
    conf_dummies = pd.get_dummies(df_work['confID'], prefix='conf', drop_first=False)
    
    # Combine features
    X = pd.concat([
        df_work[feature_cols_numeric],
        conf_dummies
    ], axis=1)
    
    
    # Target (rank within conference)
    y = pd.to_numeric(df_work['rank'], errors='coerce')
    
    # Metadata for analysis and reporting
    meta_df = df_work[['year', 'confID', 'tmID', 'name', 'rank']].copy()
    
    return X, y, meta_df


# =============================================================================
# 5. PAIRWISE TRAINING (Learning-to-Rank)
# =============================================================================

def generate_pairwise_data(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Generate pairwise feature differences and binary labels for ranking."""
    print("[TeamRanking] Generating pairwise training data...")
    
    df_work = df.copy()
    df_work['index_original'] = df_work.index
    
    X_pairs_list = []
    y_pairs_list = []
    
    # Convert X to numpy for numeric operations
    X_np = X.values.astype(float)
    X_index = X.index.tolist()
    
    # Group by (year, confID)
    for (year, conf), group in df_work.groupby(['year', 'confID']):
        indices = group['index_original'].values
        ranks = y.loc[indices].values
        
        # Generate all pairs (i, j) where i != j
        for i, idx_i in enumerate(indices):
            for j, idx_j in enumerate(indices):
                if i >= j:
                    continue
                
                rank_i = ranks[i]
                rank_j = ranks[j]
                
                # Get positions in X
                pos_i = X_index.index(idx_i)
                pos_j = X_index.index(idx_j)
                
                # Always create pair as (i, j) with appropriate label
                X_pair = X_np[pos_i] - X_np[pos_j]
                
                # Label: 1 if team i is better (lower rank), 0 if team j is better
                if rank_i < rank_j:
                    y_pair = 1  # Team i wins
                elif rank_j < rank_i:
                    y_pair = 0  # Team j wins (team i loses)
                else:
                    # Equal ranks - create both labels with 50% probability
                    y_pair = 0.5
                
                X_pairs_list.append(X_pair)
                y_pairs_list.append(y_pair)
    
    X_pairs = pd.DataFrame(X_pairs_list, columns=X.columns)
    y_pairs = np.array(y_pairs_list)
    
    # Remove ties (0.5 labels)
    mask = y_pairs != 0.5
    X_pairs = X_pairs[mask].reset_index(drop=True)
    y_pairs = y_pairs[mask]
    
    print(f"  ✓ Generated {len(X_pairs)} pairwise samples")
    print(f"    - Class 1 (better): {np.sum(y_pairs == 1)}")
    print(f"    - Class 0 (worse): {np.sum(y_pairs == 0)}")
    
    return X_pairs, y_pairs


def predict_ranks_pairwise(
    model,
    df: pd.DataFrame,
    X: pd.DataFrame
) -> np.ndarray:
    """Predict per-team scores by summing pairwise probabilities per group."""
    df_work = df.copy()
    df_work = df_work.reset_index(drop=True)
    
    scores = np.zeros(len(df_work))
    
    # Convert X to numpy for numeric operations (reset index to match df_work)
    X_reset = X.reset_index(drop=True)
    X_np = X_reset.values.astype(float)
    
    # Group by (year, confID)
    for (year, conf), group in df_work.groupby(['year', 'confID']):
        indices = group.index.tolist()
        
        # Compute pairwise probabilities
        for i in indices:
            score_i = 0.0
            
            for j in indices:
                if i == j:
                    continue
                
                # X_pair = X_i - X_j
                X_pair = (X_np[i] - X_np[j]).reshape(1, -1)
                
                # P(team_i > team_j)
                prob_i_better = model.predict_proba(X_pair)[0, 1]
                score_i += prob_i_better
            
            scores[i] = score_i
    
    return scores


# =============================================================================
# 6. MODEL FACTORY WITH EARLY STOPPING & REGULARIZATION
# =============================================================================

def create_pairwise_model() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=200,          # Number of boosting stages (trees). More trees can capture more complexity but increase risk of overfitting and training time.
        learning_rate=0.03,        # Shrinkage applied to each tree's contribution. Smaller values improve generalization but require more trees.
        max_depth=2,               # Max tree depth. Low depth (very shallow trees) constrains model complexity and helps prevent overfitting.
        min_samples_split=30,      # Minimum samples required to split an internal node. Larger values avoid splits on small, noisy subsets.
        min_samples_leaf=15,       # Minimum samples required to be at a leaf node. Ensures leaf predictions are based on enough data for stability.
        subsample=0.7,             # Fraction of samples used per tree (stochastic boosting). Values <1.0 reduce variance and improve robustness.
        max_features='sqrt',       # Number of features to consider when looking for best split ('sqrt' reduces correlation between trees).
        random_state=RANDOM_STATE, # Seed for reproducible results (controls randomness in subsampling and feature selection).
        validation_fraction=0.15,  # Fraction of training data reserved internally for early stopping evaluation.
        n_iter_no_change=15,       # Stop if validation score does not improve for this many iterations (early stopping patience).
        tol=1e-3                   # Minimum relative improvement to qualify as an actual improvement for early stopping.
    )

# =============================================================================
# 7. RANKING CONVERSION (score → rank within conference)
# =============================================================================

def add_predicted_rank(meta_df: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
    """Attach predicted score and compute rank within (year, confID)."""
    df_result = meta_df.copy()
    df_result['pred_score'] = y_pred
    
    # Convert scores to ranks within each (year, confID) group
    # Higher score = better team = lower rank number
    df_result['pred_rank'] = df_result.groupby(['year', 'confID'])['pred_score'].rank(
        method='first', 
        ascending=False  # Higher score gets rank 1
    ).astype(int)
    
    return df_result


# =============================================================================
# 8. EVALUATION METRICS AND NDCG
# =============================================================================

def calculate_ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
    """Calculate Normalized Discounted Cumulative Gain at K.
        Best rank = 1 (highest relevance)
    """
    # Convert ranks to relevance scores (lower rank = higher relevance)
    # relevance = max_rank - rank + 1
    max_rank = max(y_true.max(), y_pred.max())
    true_relevance = max_rank - y_true + 1
    
    # Sort by predicted rank and get top-k
    order = np.argsort(y_pred)[:k]
    true_relevance_sorted = true_relevance[order]
    
    # DCG: sum of (relevance / log2(position + 1))
    positions = np.arange(1, len(true_relevance_sorted) + 1)
    dcg = np.sum(true_relevance_sorted / np.log2(positions + 1))
    
    # IDCG: DCG of perfect ranking
    ideal_order = np.argsort(y_true)[:k]
    ideal_relevance = true_relevance[ideal_order]
    idcg = np.sum(ideal_relevance / np.log2(np.arange(1, len(ideal_relevance) + 1) + 1))
    
    # NDCG
    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate(df_with_ranks: pd.DataFrame, split_name: str) -> Dict:
    """Compute MAE, mean Spearman, NDCG and Top-K accuracies per conference groups."""
    
    # Global MAE
    mae_rank = mean_absolute_error(df_with_ranks['rank'], df_with_ranks['pred_rank'])
    
    # Per-group metrics
    spearman_corrs = []
    ndcg_scores = []
    top_k_correct = {k: 0 for k in range(1, 11)}  # top-1 to top-10
    total_groups = 0
    
    for (year, conf), group in df_with_ranks.groupby(['year', 'confID']):
        # Spearman correlation (rank preservation)
        if len(group) > 1:
            corr, _ = spearmanr(group['rank'], group['pred_rank'])
            if not np.isnan(corr):
                spearman_corrs.append(corr)
            
            # NDCG@10
            ndcg = calculate_ndcg_at_k(
                group['rank'].values, 
                group['pred_rank'].values, 
                k=min(10, len(group))
            )
            if not np.isnan(ndcg):
                ndcg_scores.append(ndcg)
        
        # True champion
        true_top1 = group[group['rank'] == 1]['tmID'].values
        
        # Top-K accuracy for K=1 to 10
        for k in range(1, 11):
            pred_top_k = group[group['pred_rank'] <= k]['tmID'].values
            if len(true_top1) > 0 and len(pred_top_k) > 0:
                if true_top1[0] in pred_top_k:
                    top_k_correct[k] += 1
        
        total_groups += 1
    
    mean_spearman = np.mean(spearman_corrs) if spearman_corrs else 0.0
    mean_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
    
    # Calculate top-K accuracies
    top_k_acc = {k: top_k_correct[k] / total_groups if total_groups > 0 else 0.0 
                 for k in range(1, 11)}
    
    metrics = {
        'mae_rank': mae_rank,
        'mean_spearman': mean_spearman,
        'mean_ndcg': mean_ndcg,
        'n_groups': total_groups
    } 

    return metrics


# =============================================================================
# SAVE OUTPUTS (corrigido) + RELATÓRIO CONSISTENTE LENDO O CSV (fonte da verdade)
# =============================================================================


def save_predictions(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Path:
    """Save predictions to CSV with consistent columns and types."""
    print("\n[TeamRanking] Saving predictions to CSV...")

    # copy pra não poluir objetos externos
    train = train_df.copy()
    test = test_df.copy()

    train["split"] = "train"
    test["split"] = "test"

    df_all = pd.concat([train, test], ignore_index=True)

    # colunas obrigatórias na ordem desejada
    out_cols = ["year", "confID", "tmID", "name", "rank", "pred_rank", "pred_score", "split"]
    for c in out_cols:
        if c not in df_all.columns:
            df_all[c] = np.nan

    # normaliza tipos: ranks inteiros se possível, scores floats
    def safe_to_int(series):
        try:
            return series.astype(float).round().astype(int)
        except Exception:
            # se falhar, retorna NaNs
            return pd.Series([np.nan] * len(series))

    df_all["rank"] = safe_to_int(df_all["rank"])
    # pred_rank pode já existir; mantenha como inteiro quando possível
    df_all["pred_rank"] = safe_to_int(df_all["pred_rank"])
    df_all["pred_score"] = pd.to_numeric(df_all["pred_score"], errors="coerce")

    # ordenar de forma estável: year, confID, pred_rank (nulos no final)
    df_all = df_all[out_cols].sort_values(
        ["year", "confID", "pred_rank"], na_position="last", ignore_index=True
    )

    out_path = PROC_DIR / "team_ranking_predictions.csv"
    df_all.to_csv(out_path, index=False, float_format="%.6f", encoding="utf-8")

    print(f"  ✓ Saved predictions to {out_path}")
    return out_path


def save_report(
    max_train_year: int,
    best_params: Dict,
    best_cv_score: float,
    train_metrics: Dict,
    val_metrics: Dict,
    test_metrics: Dict,
    test_df: pd.DataFrame,
    report_name: str = "team_ranking_report_metrics.txt",
    strict_predictive: bool = True
) -> Path:
    """Read saved CSV and write a comprehensive report with all metrics."""
    report_path = REPORTS_DIR / "team_ranking" / report_name
    csv_path = PROC_DIR / "team_ranking_predictions.csv"

    # Ler CSV salvo (fonte da verdade)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV não encontrado: {csv_path}")
    
    df_all = pd.read_csv(csv_path, encoding="utf-8")
    
    # Validar colunas necessárias
    required_cols = ["year", "confID", "tmID", "name", "rank", "pred_rank", "split"]
    for col in required_cols:
        if col not in df_all.columns:
            raise ValueError(f"Coluna '{col}' não encontrada no CSV")

    # Filtrar apenas dados de teste (split == 'test' e year > max_train_year)
    df_test = df_all.loc[
        (df_all["split"] == "test") &
        (df_all["year"] > max_train_year)
    ].copy()

    if df_test.empty:
        raise ValueError(f"Nenhum dado de teste encontrado para years > {max_train_year}")

    # Normalizar tipos
    df_test["rank"] = pd.to_numeric(df_test["rank"], errors="coerce").astype('Int64')
    df_test["pred_rank"] = pd.to_numeric(df_test["pred_rank"], errors="coerce").astype('Int64')

    # ============================================================================
    # CALCULAR MÉTRICAS
    # ============================================================================
    
    # MAE (Mean Absolute Error)
    valid_mask = df_test["rank"].notna() & df_test["pred_rank"].notna()
    if valid_mask.sum() > 0:
        mae_rank = mean_absolute_error(
            df_test.loc[valid_mask, "rank"].astype(int),
            df_test.loc[valid_mask, "pred_rank"].astype(int)
        )
    else:
        mae_rank = float("nan")

    # Mean Spearman Correlation (por grupo year-confID)
    spearman_corrs = []
    ndcg_scores = []
    for (year, conf), g in df_test.groupby(["year", "confID"]):
        g = g.dropna(subset=["rank", "pred_rank"])
        if len(g) <= 1:
            continue
        try:
            corr, _ = spearmanr(g["rank"].astype(int), g["pred_rank"].astype(int))
            if not np.isnan(corr):
                spearman_corrs.append(corr)
            
            # NDCG@10
            ndcg = calculate_ndcg_at_k(
                g["rank"].values.astype(int),
                g["pred_rank"].values.astype(int),
                k=min(10, len(g))
            )
            if not np.isnan(ndcg):
                ndcg_scores.append(ndcg)
        except Exception:
            continue
    
    mean_spearman = float(np.mean(spearman_corrs)) if spearman_corrs else float("nan")
    mean_ndcg = float(np.mean(ndcg_scores)) if ndcg_scores else float("nan")

    # Top-K Accuracy
    total_groups = 0
    top_k_correct = {k: 0 for k in range(1, 11)}
    
    for (year, conf), g in df_test.groupby(["year", "confID"]):
        # Limpar dados inválidos
        g = g.dropna(subset=["rank", "pred_rank"])
        if len(g) == 0:
            continue
        
        total_groups += 1
        
        # Converter ranks para inteiros
        g = g.copy()
        g["rank"] = g["rank"].astype(int)
        g["pred_rank"] = g["pred_rank"].astype(int)
        
        # Para cada K de 1 a 10
        for k in range(1, 11):
            # Encontrar times verdadeiros nas top-K posições (ranks 1 a K)
            true_top_k = set(g.loc[g["rank"] <= k, "tmID"].astype(str))
            
            # Encontrar times previstos nas top-K posições (pred_ranks 1 a K)
            pred_top_k = set(g.loc[g["pred_rank"] <= k, "tmID"].astype(str))
            
            # Calcular interseção (acertos)
            correct_in_top_k = len(true_top_k & pred_top_k)
            
            # Se todos os K times verdadeiros estão nos K previstos, é um acerto completo
            if correct_in_top_k == k:
                top_k_correct[k] += 1

    # Calcular porcentagens
    if total_groups == 0:
        top_k_acc = {k: 0.0 for k in range(1, 11)}
    else:
        top_k_acc = {k: top_k_correct[k] / total_groups for k in range(1, 11)}

    # ============================================================================
    # SALVAR RELATÓRIO
    # ============================================================================
    
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"GENERATED: {now} UTC\n")
        f.write("MODE: PREDICTIVE WITH REGULARIZATION\n")
        f.write(f"TRAIN_SEASONS: 1-{max_train_year - 2}\n")
        f.write(f"VAL_SEASONS: {max_train_year - 1}-{max_train_year}\n")
        f.write(f"TEST_SEASONS: {max_train_year + 1}+\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("TRAIN METRICS\n")
        f.write("=" * 60 + "\n")
        f.write(f"MAE_rank: {train_metrics['mae_rank']:.4f}\n")
        f.write(f"Mean_Spearman: {train_metrics['mean_spearman']:.4f}\n")
        f.write(f"Mean_NDCG@10: {train_metrics.get('mean_ndcg', 0.0):.4f}\n")
        f.write(f"n_groups: {train_metrics['n_groups']}\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("VALIDATION METRICS\n")
        f.write("=" * 60 + "\n")
        f.write(f"MAE_rank: {val_metrics['mae_rank']:.4f}\n")
        f.write(f"Mean_Spearman: {val_metrics['mean_spearman']:.4f}\n")
        f.write(f"Mean_NDCG@10: {val_metrics.get('mean_ndcg', 0.0):.4f}\n")
        f.write(f"n_groups: {val_metrics['n_groups']}\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("TEST METRICS\n")
        f.write("=" * 60 + "\n")
        f.write(f"MAE_rank: {mae_rank:.4f}\n")
        f.write(f"Mean_Spearman: {mean_spearman:.4f}\n")
        f.write(f"Mean_NDCG@10: {mean_ndcg:.4f}\n")
        f.write(f"n_groups: {total_groups}\n\n")
        
        f.write("Top-K Accuracy (Test):\n")
        for k in range(1, 11):
            f.write(f"  Top-{k:2d}: {top_k_acc[k]:.2%}\n")
        
        # Overall accuracy
        valid_rows = df_test[df_test['rank'].notna() & df_test['pred_rank'].notna()]
        total_rows = len(valid_rows)
        if total_rows > 0:
            correct_rows = int((valid_rows['rank'].astype(int) == valid_rows['pred_rank'].astype(int)).sum())
            overall_acc = correct_rows / total_rows
        else:
            correct_rows = 0
            overall_acc = 0.0

        f.write("\n")
        f.write(f"Overall_accuracy: {overall_acc:.2%} ({correct_rows}/{total_rows})\n\n")
        
        # Overfitting diagnosis
        train_test_gap = train_metrics['mae_rank'] - mae_rank
        val_test_gap = val_metrics['mae_rank'] - mae_rank
        
        f.write("=" * 60 + "\n")
        f.write("OVERFITTING DIAGNOSIS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Train-Test MAE gap: {train_test_gap:.4f}\n")
        f.write(f"Val-Test MAE gap: {val_test_gap:.4f}\n")
    
    return report_path


# =============================================================================
# 10. MAIN PIPELINE WITH VALIDATION SET
# =============================================================================

def run_team_ranking_model(
    max_train_year: int = 8,
    val_years: int = 2,
    report_name: str = "team_ranking_report_enhanced.txt",
    generate_graphics: bool = True
) -> None:
    """Main predictive pipeline with validation set and strong regularization."""
    print("\n" + "=" * 80)
    print("TEAM RANKING MODEL")
    print("=" * 80)
    
    # 1. Load and merge data
    df_all = load_and_merge()
    
    # 2. Add temporal features
    df_all = add_temporal_features(df_all)
    
    # 2B. Add coach career features 
    coach_perf_path = PROC_DIR / "coach_season_facts_performance.csv"
    if coach_perf_path.exists():
        print("\n[TeamRanking] Loading coach performance data...")
        df_coaches = pd.read_csv(coach_perf_path)
                
        # Calculate weights (games per stint)
        df_coaches['gp'] = df_coaches['won'] + df_coaches['lost']
        
        # Group by team_id + year and aggregate
        coach_agg = df_coaches.groupby(['team_id', 'year']).agg({
            'coachID': 'first',  # Use first coach (arbitrary choice, could use last)
            'rs_win_pct_coach': lambda x: np.average(x, weights=df_coaches.loc[x.index, 'gp']),
            'eb_rs_win_pct': lambda x: np.average(x, weights=df_coaches.loc[x.index, 'gp']),
            'is_first_year_with_team': 'max',  # 1 if any coach is in first year
            'gp': 'sum'  # Total games
        }).reset_index()
        
        # Merge coach data (now unique per team-year)
        df_all = df_all.merge(
            coach_agg[['team_id', 'year', 'coachID', 'rs_win_pct_coach', 
                       'eb_rs_win_pct', 'is_first_year_with_team']],
            on=['team_id', 'year'],
            how='left'
        )
        print(f"  ✓ Merged coach data: {df_all['coachID'].notna().sum()} teams with coaches")
        
        # Add coach career features
        df_all = add_coach_career_features(df_all)
    else:
        print("\n⚠ Coach performance data not found, skipping coach features")
    
    # 3. Temporal split with walk-forward validation
    folds = generate_walk_forward_splits(df_all, max_train_year, val_years)

    # Folds to assess validation performance. For each fold, train on train_fold and eval on val_fold.
    val_metrics_list = []
    print("\n[TeamRanking] Running walk-forward validation over folds...")
    for idx, (train_fold, val_fold) in enumerate(folds, start=1):
        print(f"\n[Fold {idx}/{len(folds)}] Train years <= {train_fold['year'].max() if not train_fold.empty else 'N/A'} | Val year = {val_fold['year'].unique()}")

        # Build features for fold
        X_train_fold, y_train_fold, meta_train_fold = build_feature_matrix(train_fold)
        X_val_fold, y_val_fold, meta_val_fold = build_feature_matrix(val_fold)

        # pairwise for fold
        X_pairs_fold, y_pairs_fold = generate_pairwise_data(train_fold, X_train_fold, y_train_fold)

        # train model
        model_fold = create_pairwise_model()
        if len(X_pairs_fold) == 0:
            print("  ⚠ Not enough pairwise samples for fold, skipping")
            continue
        model_fold.fit(X_pairs_fold, y_pairs_fold)

        # predict and evaluate on validation fold
        y_pred_val_fold = predict_ranks_pairwise(model_fold, val_fold, X_val_fold)
        val_with_ranks_fold = add_predicted_rank(meta_val_fold, y_pred_val_fold)
        metrics_fold = evaluate(val_with_ranks_fold, f"FOLD_{idx}")
        val_metrics_list.append(metrics_fold)
        print(f"  Fold {idx} val MAE: {metrics_fold['mae_rank']:.3f} | Spearman: {metrics_fold['mean_spearman']:.3f} | NDCG: {metrics_fold.get('mean_ndcg',0):.3f}")

    # Aggregate validation metrics
    if val_metrics_list:
        agg_val_metrics = {
            'mae_rank': float(np.mean([m['mae_rank'] for m in val_metrics_list])),
            'mean_spearman': float(np.mean([m['mean_spearman'] for m in val_metrics_list])),
            'mean_ndcg': float(np.mean([m.get('mean_ndcg', 0.0) for m in val_metrics_list])),
            'n_groups': int(np.sum([m['n_groups'] for m in val_metrics_list]))
        }
    else:
        agg_val_metrics = {'mae_rank': float('nan'), 'mean_spearman': float('nan'), 'mean_ndcg': float('nan'), 'n_groups': 0}

    print(f"\n[TeamRanking] Aggregated validation MAE: {agg_val_metrics['mae_rank']:.3f} | Spearman: {agg_val_metrics['mean_spearman']:.3f}")

    # 4. Train final model on all data up to max_train_year (include validation years) and evaluate on test
    train_final = df_all[df_all['year'] <= max_train_year].copy()
    test_final = df_all[df_all['year'] > max_train_year].copy()

    X_train_final, y_train_final, meta_train_final = build_feature_matrix(train_final)
    X_test_final, y_test_final, meta_test_final = build_feature_matrix(test_final)

    X_pairs_final, y_pairs_final = generate_pairwise_data(train_final, X_train_final, y_train_final)
    final_model = create_pairwise_model()
    if len(X_pairs_final) == 0:
        raise RuntimeError("Not enough pairwise samples to train final model")
    final_model.fit(X_pairs_final, y_pairs_final)
    best_params = final_model.get_params()
    best_cv_score = 0.0
    print(f"  ✓ Final model trained with {final_model.n_estimators_} estimators (early stopped)")

    # Predict on train_final and test_final
    y_pred_train = predict_ranks_pairwise(final_model, train_final, X_train_final)
    y_pred_test = predict_ranks_pairwise(final_model, test_final, X_test_final)

    # Convert to ranks
    train_with_ranks = add_predicted_rank(meta_train_final, y_pred_train)
    test_with_ranks = add_predicted_rank(meta_test_final, y_pred_test)

    # Evaluate
    print("\n[TeamRanking] Evaluating final model on train_final and test_final...")
    train_metrics = evaluate(train_with_ranks, "TRAIN")
    val_metrics = agg_val_metrics
    test_metrics = evaluate(test_with_ranks, "TEST")

    # Save outputs
    train_with_ranks['split'] = 'train'
    test_with_ranks['split'] = 'test'
    save_predictions(train_with_ranks, test_with_ranks)

    save_report(
        max_train_year, best_params, best_cv_score,
        train_metrics, val_metrics, test_metrics,
        test_with_ranks, report_name,
        strict_predictive=True
    )

    # 11. Generate graphics
    print("\n[TeamRanking] Generating visualizations...")
    if generate_graphics:
        generate_all_graphics()


# =============================================================================
# 12. CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # ======== Configurations ========
    MAX_TRAIN_YEAR = 8                 # Last year for training+validation
    VAL_YEARS = 2                      # Years reserved for validation (e.g., 7-8)
    REPORT_NAME = "team_ranking_report.txt"  # Output report filename
    GRAFICS = True                     # Whether to generate graphics
    
    # This script runs the predictive (pre-season, no-leakage) model
    # with strong regularization to prevent overfitting
    run_team_ranking_model(
        max_train_year=MAX_TRAIN_YEAR,
        val_years=VAL_YEARS,
        report_name=REPORT_NAME,
        generate_graphics=GRAFICS
    )
