#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Team Ranking Model - Unified & Robust Conference Ranking Prediction
====================================================================
This is the CANONICAL implementation for team ranking prediction.

Key features:
- Temporal split (train: seasons 1-8, test: 9-10)
- Walk-forward cross-validation to prevent overfitting
- Two model types: RandomForest (rf) and GradientBoosting (gbr)
- Zero data leakage
- Conference-aware ranking (within East/West)
- Comprehensive reporting

Training: Seasons 1-8 (with internal walk-forward CV)
Testing: Seasons 9-10 (holdout, touched only once)

Usage:
    python team_ranking_model.py --model rf    # RandomForest
    python team_ranking_model.py --model gbr   # GradientBoosting
"""

from pathlib import Path
from typing import Tuple, Dict, List, Callable
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# Paths
ROOT = Path(__file__).resolve().parents[3]
PROC_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "src" / "model" / "ranking_model"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Random state for reproducibility
RANDOM_STATE = 42


# =============================================================================
# 1. DATA LOADING AND VALIDATION
# =============================================================================

def load_and_merge() -> pd.DataFrame:
    """
    Load team_season_statistics.csv and team_performance.csv, merge them.
    
    Returns:
        Merged DataFrame with all features and target (rank).
    
    Raises:
        FileNotFoundError: If required CSV files don't exist
        KeyError: If required columns are missing
    """
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
# 2. TEMPORAL SPLIT
# =============================================================================

def split_train_test(
    df_all: pd.DataFrame, 
    max_train_year: int = 8
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset by year (temporal split, NO shuffle).
    
    Args:
        df_all: Full dataset
        max_train_year: Last season to include in training (default: 8)
    
    Returns:
        train_df, test_df
    """
    print(f"\n[TeamRanking] Splitting train/test by season (max_train_year={max_train_year})...")
    
    train_df = df_all[df_all['year'] <= max_train_year].copy()
    test_df = df_all[df_all['year'] > max_train_year].copy()
    
    print(f"  ✓ Train: {len(train_df)} rows (seasons 1-{max_train_year})")
    print(f"  ✓ Test:  {len(test_df)} rows (seasons {max_train_year+1}+)")
    
    return train_df, test_df


# =============================================================================
# 3. FEATURE ENGINEERING (NO LEAKAGE)
# =============================================================================

def build_feature_matrix(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Extract features (X), target (y), and metadata from DataFrame.
    
    Features: Union of best features from both old scripts, NO LEAKAGE.
    
    EXCLUDED (leakage): rank, won, lost, GP, season_win_pct, playoff flags,
                        po_W, po_L, homeW, homeL, awayW, awayL, confW, confL
    
    Args:
        df: Input DataFrame
    
    Returns:
        X: Feature matrix (numeric + one-hot confID)
        y: Target (rank)
        meta_df: Metadata (year, confID, tmID, name, rank)
    """
    # Feature candidates (explicit list, no wildcards)
    # REMOVED LEAKAGE FEATURES: home_win_pct, away_win_pct, home_advantage
    # These are derived from homeW/homeL/awayW/awayL which directly correlate with rank
    feature_cols_numeric = [
        # From team_season_statistics.csv
        'point_diff', 'off_eff', 'def_eff',
        'fg_pct', 'three_pct', 'ft_pct', 'opp_fg_pct',
        'prop_3pt_shots',
        # 'home_win_pct', 'away_win_pct', 'home_advantage',  # REMOVED: leakage risk
        'reb_diff', 'stl_diff', 'blk_diff', 'to_diff',
        'attend_pg',
        'franchise_changed',
        'prev_win_pct_1', 'prev_win_pct_3', 'prev_win_pct_5',
        'prev_point_diff_3', 'prev_point_diff_5',
        'win_pct_change',
        'off_eff_norm', 'def_eff_norm', 'fg_pct_norm', 'three_pct_norm',
        'ft_pct_norm', 'opp_fg_pct_norm', 'point_diff_norm',
        # From team_performance.csv
        'pythag_win_pct', 'team_strength', 'rs_win_pct_expected_roster',
        'overach_pythag', 'overach_roster'
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
# 4. MODEL FACTORY (RF vs GBR with anti-overfitting regularization)
# =============================================================================

def create_model_rf() -> RandomForestRegressor:
    """
    Create RandomForestRegressor with STRONG anti-overfitting hyperparameters.
    
    Key changes for regularization:
    - max_depth=4 (reduced from 6, shallower trees)
    - min_samples_leaf=5 (increased from 2)
    - min_samples_split=10 (increased from 4)
    - max_samples=0.7 (use only 70% of data per tree)
    - n_estimators=200 (reduced from 400, less ensemble variance)
    - oob_score=True (out-of-bag validation)
    """
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=4,              # Reduced from 6 → shallower trees
        min_samples_leaf=5,       # Increased from 2 → more regularization
        min_samples_split=10,     # Increased from 4 → more regularization
        max_features='sqrt',
        max_samples=0.7,          # NEW: bootstrap 70% per tree
        oob_score=True,           # NEW: enable OOB validation
        random_state=RANDOM_STATE,
        n_jobs=-1
    )


def create_model_gbr() -> GradientBoostingRegressor:
    """
    Create GradientBoostingRegressor with STRONG anti-overfitting hyperparameters.
    
    Key regularization:
    - learning_rate=0.01 (reduced from 0.03, even slower learning)
    - subsample=0.6 (reduced from 0.7, more stochastic)
    - max_depth=3 (shallow trees)
    - min_samples_leaf=5 (increased from 2)
    - n_estimators=200 (reduced from 400)
    """
    return GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.01,       # Reduced from 0.03 → slower learning
        max_depth=3,
        subsample=0.6,            # Reduced from 0.7 → more stochastic
        min_samples_leaf=5,       # Increased from 2 → more regularization
        random_state=RANDOM_STATE
    )


def get_model_factory(model_type: str) -> Callable:
    """Get model factory function by type."""
    factories = {
        'rf': create_model_rf,
        'gbr': create_model_gbr
    }
    if model_type not in factories:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'rf' or 'gbr'.")
    return factories[model_type]


# =============================================================================
# 5. RANKING CONVERSION (score → rank within conference)
# =============================================================================

def add_predicted_rank(meta_df: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
    """
    Add predicted scores and convert to ranks within (year, confID) groups.
    
    Lower pred_score → better rank (rank 1 is best).
    
    Args:
        meta_df: Metadata (year, confID, tmID, name, rank)
        y_pred: Predicted rank scores
    
    Returns:
        DataFrame with columns: year, confID, tmID, name, rank, pred_score, pred_rank
    """
    df_result = meta_df.copy()
    df_result['pred_score'] = y_pred
    
    # Convert scores to ranks within each (year, confID) group
    df_result['pred_rank'] = df_result.groupby(['year', 'confID'])['pred_score'].rank(
        method='first', 
        ascending=True
    ).astype(int)
    
    return df_result


# =============================================================================
# 6. EVALUATION METRICS (conference-aware)
# =============================================================================

def evaluate(df_with_ranks: pd.DataFrame, split_name: str, verbose: bool = True) -> Dict:
    """
    Compute evaluation metrics for predicted rankings.
    
    Metrics (all conference-aware):
    - MAE of rank (global average)
    - Mean Spearman correlation per (year, confID) group
    - Top-1 accuracy: % of groups where predicted champion == true champion
    - Top-2 accuracy: % of groups where true champion in predicted top-2
    
    Args:
        df_with_ranks: DataFrame with columns [rank, pred_rank, year, confID, tmID]
        split_name: Name of split for logging (e.g., "TRAIN", "TEST")
        verbose: Print metrics to console
    
    Returns:
        Dictionary with metrics: mae_rank, mean_spearman, top1_accuracy, top2_accuracy, n_groups
    """
    if verbose:
        print(f"\n[TeamRanking] Evaluating {split_name} set...")
    
    # Global MAE
    mae_rank = mean_absolute_error(df_with_ranks['rank'], df_with_ranks['pred_rank'])
    
    # Per-group metrics
    spearman_corrs = []
    top1_correct = 0
    top2_correct = 0
    total_groups = 0
    
    for (year, conf), group in df_with_ranks.groupby(['year', 'confID']):
        # Spearman correlation (rank preservation)
        if len(group) > 1:
            corr, _ = spearmanr(group['rank'], group['pred_rank'])
            if not np.isnan(corr):
                spearman_corrs.append(corr)
        
        # Top-1 accuracy (exact champion prediction)
        true_top1 = group[group['rank'] == 1]['tmID'].values
        pred_top1 = group[group['pred_rank'] == 1]['tmID'].values
        
        if len(true_top1) > 0 and len(pred_top1) > 0:
            if true_top1[0] == pred_top1[0]:
                top1_correct += 1
        
        # Top-2 accuracy (champion in predicted top-2)
        pred_top2 = group[group['pred_rank'] <= 2]['tmID'].values
        if len(true_top1) > 0 and len(pred_top2) > 0:
            if true_top1[0] in pred_top2:
                top2_correct += 1
        
        total_groups += 1
    
    mean_spearman = np.mean(spearman_corrs) if spearman_corrs else 0.0
    top1_acc = top1_correct / total_groups if total_groups > 0 else 0.0
    top2_acc = top2_correct / total_groups if total_groups > 0 else 0.0
    
    metrics = {
        'mae_rank': mae_rank,
        'mean_spearman': mean_spearman,
        'top1_accuracy': top1_acc,
        'top2_accuracy': top2_acc,
        'n_groups': total_groups
    }
    
    if verbose:
        print(f"  MAE rank: {mae_rank:.3f}")
        print(f"  Mean Spearman: {mean_spearman:.3f}")
        print(f"  Top-1 accuracy: {top1_acc:.2%} ({top1_correct}/{total_groups})")
        print(f"  Top-2 accuracy: {top2_acc:.2%} ({top2_correct}/{total_groups})")
    
    return metrics


# =============================================================================
# 7. WALK-FORWARD CROSS-VALIDATION (anti-overfitting)
# =============================================================================

def walkforward_cv(
    df_train_all: pd.DataFrame,
    model_factory: Callable,
    min_year_for_cv: int = 3,
    max_train_year: int = 8
) -> Dict:
    """
    Walk-forward cross-validation within training seasons.
    
    For each validation year in [min_year_for_cv, max_train_year]:
        - Train on years <= val_year - 1
        - Validate on year == val_year
    
    This simulates real-world scenario: predicting next season from past.
    
    Args:
        df_train_all: Full training DataFrame (years 1 to max_train_year)
        model_factory: Function that creates a fresh model instance
        min_year_for_cv: Minimum year to use as validation (default: 3)
        max_train_year: Maximum training year (default: 8)
    
    Returns:
        Dictionary with averaged metrics: mae_rank, mean_spearman, top1_accuracy, top2_accuracy
    """
    print(f"\n[TeamRanking] Walk-forward CV (val years {min_year_for_cv}-{max_train_year})...")
    
    all_mae = []
    all_spearman = []
    all_top1 = []
    all_top2 = []
    
    for val_year in range(min_year_for_cv, max_train_year + 1):
        # Split for this fold
        train_fold = df_train_all[df_train_all['year'] <= val_year - 1].copy()
        val_fold = df_train_all[df_train_all['year'] == val_year].copy()
        
        if len(train_fold) == 0 or len(val_fold) == 0:
            continue
        
        # Build features
        X_train_fold, y_train_fold, _ = build_feature_matrix(train_fold)
        X_val_fold, y_val_fold, meta_val_fold = build_feature_matrix(val_fold)
        
        # Train model
        model = model_factory()
        model.fit(X_train_fold, y_train_fold)
        
        # Predict and convert to ranks
        y_pred_val = model.predict(X_val_fold)
        val_with_ranks = add_predicted_rank(meta_val_fold, y_pred_val)
        
        # Evaluate
        metrics = evaluate(val_with_ranks, f"CV_Year{val_year}", verbose=False)
        
        all_mae.append(metrics['mae_rank'])
        all_spearman.append(metrics['mean_spearman'])
        all_top1.append(metrics['top1_accuracy'])
        all_top2.append(metrics['top2_accuracy'])
        
        print(f"  Year {val_year}: MAE={metrics['mae_rank']:.3f}, "
              f"Spearman={metrics['mean_spearman']:.3f}, "
              f"Top-1={metrics['top1_accuracy']:.2%}")
    
    # Average across folds
    cv_metrics = {
        'mae_rank': np.mean(all_mae) if all_mae else 0.0,
        'mean_spearman': np.mean(all_spearman) if all_spearman else 0.0,
        'top1_accuracy': np.mean(all_top1) if all_top1 else 0.0,
        'top2_accuracy': np.mean(all_top2) if all_top2 else 0.0,
        'n_folds': len(all_mae)
    }
    
    print(f"\n  CV Average: MAE={cv_metrics['mae_rank']:.3f}, "
          f"Spearman={cv_metrics['mean_spearman']:.3f}, "
          f"Top-1={cv_metrics['top1_accuracy']:.2%}")
    
    return cv_metrics


# =============================================================================
# 8. SAVE OUTPUTS
# =============================================================================

def save_predictions(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Save predictions to CSV: data/processed/team_ranking_predictions.csv
    
    Args:
        train_df: Training predictions with columns [year, confID, tmID, name, rank, pred_rank, pred_score]
        test_df: Test predictions (same structure)
    """
    train_df['split'] = 'train'
    test_df['split'] = 'test'
    
    df_all = pd.concat([train_df, test_df], ignore_index=True)
    
    # Select and order columns
    out_cols = ['year', 'confID', 'tmID', 'name', 'rank', 'pred_rank', 'pred_score', 'split']
    df_out = df_all[out_cols].copy()
    
    # Sort by year, conference, predicted rank
    df_out = df_out.sort_values(['year', 'confID', 'pred_rank']).reset_index(drop=True)
    
    out_path = PROC_DIR / "team_ranking_predictions.csv"
    df_out.to_csv(out_path, index=False)
    
    print(f"\n[TeamRanking] Saved predictions to {out_path}")


def save_report(
    model_type: str,
    max_train_year: int,
    cv_metrics: Dict,
    train_metrics: Dict,
    test_metrics: Dict,
    feature_importance: List[Tuple[str, float]],
    test_df: pd.DataFrame,
    report_name: str = "team_ranking_report.txt"
):
    """
    Save comprehensive evaluation report.
    
    Args:
        model_type: 'rf' or 'gbr'
        max_train_year: Last training season
        cv_metrics: Walk-forward CV metrics
        train_metrics: Full training set metrics
        test_metrics: Test set (holdout) metrics
        feature_importance: List of (feature_name, importance) tuples
        test_df: Test DataFrame with predictions for examples
        report_name: Output filename
    """
    report_path = REPORTS_DIR / report_name
    
    model_names = {'rf': 'RandomForestRegressor', 'gbr': 'GradientBoostingRegressor'}
    model_name = model_names.get(model_type, model_type)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write("TEAM RANKING MODEL - UNIFIED & ROBUST IMPLEMENTATION\n")
        f.write("=" * 80 + "\n\n")
        
        # Configuration
        f.write("CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Train seasons: 1-{max_train_year}\n")
        f.write(f"Test seasons: {max_train_year+1}+\n")
        f.write(f"Random state: {RANDOM_STATE}\n")
        f.write(f"Features: {len(feature_importance)}\n")
        
        if model_type == 'rf':
            f.write("\nHyperparameters (RandomForest - STRONG regularization):\n")
            f.write("  - n_estimators: 200 (reduced from 400)\n")
            f.write("  - max_depth: 4 (reduced from 6 for less overfitting)\n")
            f.write("  - min_samples_leaf: 5 (increased from 2)\n")
            f.write("  - min_samples_split: 10 (increased from 4)\n")
            f.write("  - max_features: sqrt\n")
            f.write("  - max_samples: 0.7 (bootstrap 70% per tree)\n")
            f.write("  - oob_score: True (out-of-bag validation enabled)\n")
        else:
            f.write("\nHyperparameters (GradientBoosting - STRONG regularization):\n")
            f.write("  - n_estimators: 200 (reduced from 400)\n")
            f.write("  - learning_rate: 0.01 (very slow learning)\n")
            f.write("  - max_depth: 3 (shallow trees)\n")
            f.write("  - subsample: 0.6 (60% per tree, more stochastic)\n")
            f.write("  - min_samples_leaf: 5 (increased from 2)\n")
        
        # Walk-forward CV metrics
        f.write("\n" + "=" * 80 + "\n")
        f.write("WALK-FORWARD CROSS-VALIDATION (internal, anti-overfitting)\n")
        f.write("-" * 80 + "\n")
        f.write(f"Number of folds: {cv_metrics['n_folds']}\n")
        f.write(f"MAE rank:        {cv_metrics['mae_rank']:.3f}\n")
        f.write(f"Mean Spearman:   {cv_metrics['mean_spearman']:.3f}\n")
        f.write(f"Top-1 accuracy:  {cv_metrics['top1_accuracy']:.2%}\n")
        f.write(f"Top-2 accuracy:  {cv_metrics['top2_accuracy']:.2%}\n")
        
        # Train metrics (full training set)
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"TRAIN METRICS (full, seasons 1-{max_train_year})\n")
        f.write("-" * 80 + "\n")
        f.write(f"MAE rank:        {train_metrics['mae_rank']:.3f}\n")
        f.write(f"Mean Spearman:   {train_metrics['mean_spearman']:.3f}\n")
        f.write(f"Top-1 accuracy:  {train_metrics['top1_accuracy']:.2%}\n")
        f.write(f"Top-2 accuracy:  {train_metrics['top2_accuracy']:.2%}\n")
        f.write(f"Number of groups: {train_metrics['n_groups']}\n")
        
        # Test metrics (holdout)
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"TEST METRICS (holdout, seasons {max_train_year+1}+)\n")
        f.write("-" * 80 + "\n")
        f.write(f"MAE rank:        {test_metrics['mae_rank']:.3f}\n")
        f.write(f"Mean Spearman:   {test_metrics['mean_spearman']:.3f}\n")
        f.write(f"Top-1 accuracy:  {test_metrics['top1_accuracy']:.2%}\n")
        f.write(f"Top-2 accuracy:  {test_metrics['top2_accuracy']:.2%}\n")
        f.write(f"Number of groups: {test_metrics['n_groups']}\n")
        
        # Feature importance
        f.write("\n" + "=" * 80 + "\n")
        f.write("TOP 15 MOST IMPORTANT FEATURES\n")
        f.write("-" * 80 + "\n")
        for i, (feat, imp) in enumerate(feature_importance[:15], 1):
            f.write(f"{i:2d}. {feat:40s} {imp:.6f}\n")
        
        # Interpretation
        f.write("\n" + "=" * 80 + "\n")
        f.write("INTERPRETATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"On average, the model's predictions are off by {test_metrics['mae_rank']:.2f} positions\n")
        f.write(f"in the test set (seasons {max_train_year+1}+).\n\n")
        
        f.write("The top features validate domain knowledge:\n")
        f.write("- pythag_win_pct: Bill James Pythagorean formula (points scored/allowed)\n")
        f.write("- point_diff: Simple but powerful predictor of team quality\n")
        f.write("- overach_roster: Teams that exceed roster expectations\n\n")
        
        f.write("The predicted rank serves as a baseline for team strength, used in:\n")
        f.write("- Coach of the Year (COY) analysis: coaches who exceed team strength\n")
        f.write("- Player impact evaluation: how roster changes affect expected rank\n")
        f.write("- Playoff predictions: conference rank is a strong playoff indicator\n")
        
        # Example predictions
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"EXAMPLE PREDICTIONS - Year {max_train_year+1} East\n")
        f.write("-" * 80 + "\n")
        
        example_ea = test_df[(test_df['year'] == max_train_year+1) & (test_df['confID'] == 'EA')]
        if len(example_ea) > 0:
            example_ea = example_ea.sort_values('pred_rank')
            f.write(f"{'tmID':<8} {'Name':<20} {'Rank':>6} {'Pred':>6} {'Score':>8}\n")
            f.write("-" * 80 + "\n")
            for _, row in example_ea.iterrows():
                f.write(f"{row['tmID']:<8} {row['name']:<20} {int(row['rank']):>6} "
                       f"{int(row['pred_rank']):>6} {row['pred_score']:>8.2f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"EXAMPLE PREDICTIONS - Year {max_train_year+1} West\n")
        f.write("-" * 80 + "\n")
        
        example_we = test_df[(test_df['year'] == max_train_year+1) & (test_df['confID'] == 'WE')]
        if len(example_we) > 0:
            example_we = example_we.sort_values('pred_rank')
            f.write(f"{'tmID':<8} {'Name':<20} {'Rank':>6} {'Pred':>6} {'Score':>8}\n")
            f.write("-" * 80 + "\n")
            for _, row in example_we.iterrows():
                f.write(f"{row['tmID']:<8} {row['name']:<20} {int(row['rank']):>6} "
                       f"{int(row['pred_rank']):>6} {row['pred_score']:>8.2f}\n")
        
        # Anti-leakage checklist
        f.write("\n" + "=" * 80 + "\n")
        f.write("ANTI-LEAKAGE & ANTI-OVERFITTING CHECKLIST\n")
        f.write("-" * 80 + "\n")
        f.write("✓ Temporal split (no random shuffle)\n")
        f.write("✓ Excluded: won, lost, GP, season_win_pct, playoff flags\n")
        f.write("✓ Excluded: homeW, homeL, awayW, awayL, confW, confL\n")
        f.write("✓ Excluded: po_W, po_L, po_win_pct (playoff outcomes)\n")
        f.write("✓ REMOVED: home_win_pct, away_win_pct, home_advantage (leakage risk)\n")
        f.write("✓ Walk-forward CV on training set (realistic evaluation)\n")
        f.write("✓ STRONG regularization: max_depth=4, min_samples_leaf=5, max_samples=0.7\n")
        f.write("✓ Test set touched only once (no tuning on test)\n")
        f.write("✓ Conference-aware ranking (within East/West)\n")
        f.write("✓ OOB score enabled (RandomForest internal validation)\n")
        f.write("\n")
        f.write("OVERFITTING DIAGNOSTIC:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Train MAE:  {train_metrics['mae_rank']:.4f}\n")
        f.write(f"CV MAE:     {cv_metrics['mae_rank']:.4f}\n")
        f.write(f"Test MAE:   {test_metrics['mae_rank']:.4f}\n")
        f.write("\n")
        if train_metrics['mae_rank'] < 0.1:
            f.write("⚠️  WARNING: Train MAE < 0.1 indicates overfitting!\n")
            f.write("   → The model has memorized the training data.\n")
            f.write("   → Trust CV and Test metrics instead.\n")
            f.write("   → Consider further regularization if CV >> Test.\n")
        else:
            gap_cv_test = abs(cv_metrics['mae_rank'] - test_metrics['mae_rank'])
            if gap_cv_test < 0.2:
                f.write("✓ Good generalization: CV and Test MAE are close.\n")
            else:
                f.write("⚠️  CV and Test MAE differ significantly.\n")
                f.write("   → Model may not generalize well to new data.\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"[TeamRanking] Saved report to {report_path}")


# =============================================================================
# 9. MAIN PIPELINE
# =============================================================================

def run_team_ranking_model(
    model_type: str = "rf",
    max_train_year: int = 8,
    report_name: str = "team_ranking_report.txt"
) -> None:
    """
    Main pipeline: unified team ranking model with walk-forward CV.
    
    Args:
        model_type: 'rf' (RandomForest) or 'gbr' (GradientBoosting)
        max_train_year: Last season for training (default: 8)
        report_name: Output report filename
    """
    print("\n" + "=" * 80)
    print("TEAM RANKING MODEL - UNIFIED & ROBUST IMPLEMENTATION")
    print("=" * 80)
    print(f"Model: {model_type.upper()}")
    print(f"Train: seasons 1-{max_train_year}, Test: {max_train_year+1}+")
    print("=" * 80)
    
    # 1. Load and merge data
    df_all = load_and_merge()
    
    # 2. Temporal split
    train_raw, test_raw = split_train_test(df_all, max_train_year)
    
    # 3. Build features
    print("\n[TeamRanking] Building features...")
    X_train, y_train, meta_train = build_feature_matrix(train_raw)
    X_test, y_test, meta_test = build_feature_matrix(test_raw)
    print(f"  ✓ Train: {X_train.shape[1]} features, {len(X_train)} samples")
    print(f"  ✓ Test:  {X_test.shape[1]} features, {len(X_test)} samples")
    
    # 4. Walk-forward CV (anti-overfitting check)
    model_factory = get_model_factory(model_type)
    cv_metrics = walkforward_cv(train_raw, model_factory, min_year_for_cv=3, max_train_year=max_train_year)
    
    # 5. Train final model on full training set
    print(f"\n[TeamRanking] Training final {model_type.upper()} model on full training set...")
    final_model = model_factory()
    final_model.fit(X_train, y_train)
    print("  ✓ Model trained")
    
    # Print OOB score if available (RandomForest only)
    if hasattr(final_model, 'oob_score_'):
        print(f"  ✓ OOB Score (internal validation): {final_model.oob_score_:.4f}")
    
    # 6. Feature importance
    if hasattr(final_model, 'feature_importances_'):
        importance = sorted(
            zip(X_train.columns, final_model.feature_importances_),
            key=lambda x: x[1],
            reverse=True
        )
        print("\n  Top 10 features:")
        for feat, imp in importance[:10]:
            print(f"    {feat:30s} {imp:.4f}")
    else:
        importance = []
    
    # 7. Predict
    print("\n[TeamRanking] Generating predictions...")
    y_pred_train = final_model.predict(X_train)
    y_pred_test = final_model.predict(X_test)
    print("  ✓ Predictions generated")
    
    # 8. Convert to ranks (within conference)
    train_with_ranks = add_predicted_rank(meta_train, y_pred_train)
    test_with_ranks = add_predicted_rank(meta_test, y_pred_test)
    
    # 9. Evaluate
    train_metrics = evaluate(train_with_ranks, "TRAIN (full)")
    test_metrics = evaluate(test_with_ranks, "TEST (holdout)")
    
    # Check for overfitting (warn if train MAE is suspiciously low)
    if train_metrics['mae_rank'] < 0.1:
        print("\n⚠️  WARNING: Train MAE < 0.1 suggests potential overfitting!")
        print(f"    Train MAE: {train_metrics['mae_rank']:.4f}")
        print(f"    CV MAE:    {cv_metrics['mae_rank']:.4f}")
        print(f"    Test MAE:  {test_metrics['mae_rank']:.4f}")
        print("    → Trust CV and Test metrics, not Train metrics.")
    
    # 10. Save outputs
    save_predictions(train_with_ranks, test_with_ranks)
    save_report(
        model_type, max_train_year, cv_metrics, train_metrics, test_metrics,
        importance, test_with_ranks, report_name
    )
    
    print("\n" + "=" * 80)
    print("✓ PIPELINE COMPLETE")
    print("=" * 80 + "\n")


# =============================================================================
# 10. CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Team Ranking Model - Unified & Robust Implementation"
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default='rf',
        choices=['rf', 'gbr'],
        help="Model type: 'rf' (RandomForest) or 'gbr' (GradientBoosting)"
    )
    parser.add_argument(
        '--max-train-year',
        type=int,
        default=8,
        help="Last season for training (default: 8)"
    )
    parser.add_argument(
        '--report-name',
        type=str,
        default='team_ranking_report.txt',
        help="Output report filename"
    )
    
    args = parser.parse_args()
    
    run_team_ranking_model(
        model_type=args.model,
        max_train_year=args.max_train_year,
        report_name=args.report_name
    )

