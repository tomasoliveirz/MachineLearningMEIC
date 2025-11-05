#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEPRECATED: Team Ranking Model - Conference Ranking Prediction
===============================================================

⚠️  THIS FILE IS DEPRECATED ⚠️

The canonical implementation has been moved to:
    src/model/ranking_model/team_ranking_model.py

This file is kept for backwards compatibility only.
It now acts as a simple wrapper calling the unified implementation.

For new work, use the unified model directly:
    python src/model/ranking_model/team_ranking_model.py --model rf
    python src/model/ranking_model/team_ranking_model.py --model gbr

Old behavior (seasons 1-8 train, 9-10 test) is preserved.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr

# Paths
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Model hyperparameters
RANDOM_STATE = 42
N_ESTIMATORS = 100
MAX_DEPTH = 10


def load_and_merge() -> pd.DataFrame:
    """
    Load team_season_statistics.csv and team_performance.csv, merge them.
    
    Returns merged DataFrame with all features and target (rank).
    """
    print("[TeamRanking] Loading data...")
    
    # Load teams data (use team_season_statistics.csv which has enriched features)
    teams_path = PROC_DIR / "team_season_statistics.csv"
    if not teams_path.exists():
        raise FileNotFoundError(f"team_season_statistics.csv not found in {PROC_DIR}")
    
    df_teams = pd.read_csv(teams_path)
    print(f"  ✓ Loaded {len(df_teams)} team-season records from team_season_statistics.csv")
    
    # Load team performance data
    perf_path = PROC_DIR / "team_performance.csv"
    if not perf_path.exists():
        raise FileNotFoundError(f"team_performance.csv not found in {PROC_DIR}")
    
    df_perf = pd.read_csv(perf_path)
    print(f"  ✓ Loaded {len(df_perf)} records from team_performance.csv")
    
    # Merge on year and team
    df_all = df_teams.merge(
        df_perf,
        left_on=['year', 'tmID'],
        right_on=['year', 'team_id'],
        how='left'
    )
    
    print(f"  ✓ Merged dataset: {len(df_all)} rows")
    
    # Validate required columns exist
    required = ['year', 'tmID', 'confID', 'rank', 'name']
    missing = set(required) - set(df_all.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    
    return df_all


def split_train_test(df_all: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset by year (temporal split).
    
    Train: years 1-8
    Test: years 9-10
    """
    print("\n[TeamRanking] Splitting train/test by season...")
    
    train_df = df_all[df_all['year'] <= 8].copy()
    test_df = df_all[df_all['year'] >= 9].copy()
    
    print(f"  ✓ Train: {len(train_df)} rows (seasons 1-8)")
    print(f"  ✓ Test:  {len(test_df)} rows (seasons 9-10)")
    
    return train_df, test_df


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Extract features (X), target (y), and metadata from DataFrame.
    
    Returns:
        X: Feature matrix
        y: Target (rank)
        meta_df: Metadata columns (year, confID, tmID, name, rank)
    """
    # Feature columns (explicitly defined, no leakage)
    feature_cols_numeric = [
        # From teams_cleaned.csv
        'point_diff', 'off_eff', 'def_eff',
        'fg_pct', 'three_pct', 'ft_pct', 'opp_fg_pct',
        'prop_3pt_shots',
        'home_win_pct', 'away_win_pct', 'home_advantage',
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
    
    # Ensure numeric features exist and are float
    for col in feature_cols_numeric:
        if col not in df_work.columns:
            print(f"  ⚠ Warning: Feature '{col}' not found, filling with 0.0")
            df_work[col] = 0.0
        else:
            df_work[col] = pd.to_numeric(df_work[col], errors='coerce').fillna(0.0)
    
    # One-hot encode confID
    conf_dummies = pd.get_dummies(df_work['confID'], prefix='conf', drop_first=False)
    
    # Combine features
    X = pd.concat([
        df_work[feature_cols_numeric],
        conf_dummies
    ], axis=1)
    
    # Target
    y = pd.to_numeric(df_work['rank'], errors='coerce')
    
    # Metadata for later analysis
    meta_df = df_work[['year', 'confID', 'tmID', 'name', 'rank']].copy()
    
    return X, y, meta_df


def fit_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    """
    Train RandomForestRegressor on training data.
    """
    print(f"\n[TeamRanking] Training RandomForestRegressor...")
    print(f"  - Features: {X_train.shape[1]}")
    print(f"  - Samples: {len(X_train)}")
    print(f"  - n_estimators: {N_ESTIMATORS}")
    print(f"  - max_depth: {MAX_DEPTH}")
    print(f"  - random_state: {RANDOM_STATE}")
    
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    print("  ✓ Model trained")
    
    # Feature importance (top 10)
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n  Top 10 most important features:")
    for idx, row in importance.head(10).iterrows():
        print(f"    {row['feature']:30s} {row['importance']:.4f}")
    
    return model


def add_predicted_rank(meta_df: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
    """
    Add predicted scores and convert to ranks within (year, confID) groups.
    
    Returns DataFrame with: year, confID, tmID, name, rank, pred_score, pred_rank
    """
    df_result = meta_df.copy()
    df_result['pred_score'] = y_pred
    
    # Convert scores to ranks within each (year, confID) group
    # Lower score = better rank (rank 1 is best)
    df_result['pred_rank'] = df_result.groupby(['year', 'confID'])['pred_score'].rank(
        method='first', 
        ascending=True
    ).astype(int)
    
    return df_result


def evaluate(df_with_ranks: pd.DataFrame, split_name: str) -> dict:
    """
    Compute evaluation metrics for predicted rankings.
    
    Metrics:
    - MAE of rank (global)
    - Mean Spearman correlation per (year, confID) group
    - Top-1 accuracy per group
    - Top-2 accuracy per group
    """
    print(f"\n[TeamRanking] Evaluating {split_name} set...")
    
    # Global MAE
    mae_rank = mean_absolute_error(df_with_ranks['rank'], df_with_ranks['pred_rank'])
    
    # Per-group metrics
    spearman_corrs = []
    top1_correct = 0
    top2_correct = 0
    total_groups = 0
    
    for (year, conf), group in df_with_ranks.groupby(['year', 'confID']):
        # Spearman correlation
        if len(group) > 1:
            corr, _ = spearmanr(group['rank'], group['pred_rank'])
            if not np.isnan(corr):
                spearman_corrs.append(corr)
        
        # Top-1 accuracy
        true_top1 = group[group['rank'] == 1]['tmID'].values
        pred_top1 = group[group['pred_rank'] == 1]['tmID'].values
        
        if len(true_top1) > 0 and len(pred_top1) > 0:
            if true_top1[0] == pred_top1[0]:
                top1_correct += 1
        
        # Top-2 accuracy (true champion in predicted top-2)
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
    
    print(f"  MAE rank: {mae_rank:.3f}")
    print(f"  Mean Spearman correlation: {mean_spearman:.3f}")
    print(f"  Top-1 accuracy: {top1_acc:.2%} ({top1_correct}/{total_groups})")
    print(f"  Top-2 accuracy: {top2_acc:.2%} ({top2_correct}/{total_groups})")
    
    return metrics


def save_predictions(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Save predictions to CSV.
    """
    train_df['split'] = 'train'
    test_df['split'] = 'test'
    
    df_all = pd.concat([train_df, test_df], ignore_index=True)
    
    # Select columns
    out_cols = ['year', 'confID', 'tmID', 'name', 'rank', 'pred_rank', 'pred_score', 'split']
    df_out = df_all[out_cols].copy()
    
    # Sort
    df_out = df_out.sort_values(['year', 'confID', 'pred_rank']).reset_index(drop=True)
    
    out_path = PROC_DIR / "team_ranking_predictions.csv"
    df_out.to_csv(out_path, index=False)
    
    print(f"\n[TeamRanking] Saved predictions to {out_path}")


def save_report(
    train_metrics: dict,
    test_metrics: dict,
    feature_names: list,
    examples_df: pd.DataFrame
):
    """
    Save evaluation report to text file.
    """
    report_path = REPORTS_DIR / "team_ranking_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("TEAM RANKING MODEL REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("CONFIGURATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Train seasons: 1-8\n")
        f.write(f"Test seasons:  9-10\n")
        f.write(f"Model: RandomForestRegressor\n")
        f.write(f"  - n_estimators: {N_ESTIMATORS}\n")
        f.write(f"  - max_depth: {MAX_DEPTH}\n")
        f.write(f"  - random_state: {RANDOM_STATE}\n")
        f.write(f"\nFeatures used: {len(feature_names)}\n")
        f.write(f"  {', '.join(feature_names[:10])}...\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("TRAIN METRICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"MAE rank:                    {train_metrics['mae_rank']:.3f}\n")
        f.write(f"Mean Spearman correlation:   {train_metrics['mean_spearman']:.3f}\n")
        f.write(f"Top-1 accuracy:              {train_metrics['top1_accuracy']:.2%}\n")
        f.write(f"Top-2 accuracy:              {train_metrics['top2_accuracy']:.2%}\n")
        f.write(f"Number of groups:            {train_metrics['n_groups']}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("TEST METRICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"MAE rank:                    {test_metrics['mae_rank']:.3f}\n")
        f.write(f"Mean Spearman correlation:   {test_metrics['mean_spearman']:.3f}\n")
        f.write(f"Top-1 accuracy:              {test_metrics['top1_accuracy']:.2%}\n")
        f.write(f"Top-2 accuracy:              {test_metrics['top2_accuracy']:.2%}\n")
        f.write(f"Number of groups:            {test_metrics['n_groups']}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("EXAMPLE PREDICTIONS - Year 9 East\n")
        f.write("-" * 70 + "\n")
        
        example_ea = examples_df[(examples_df['year'] == 9) & (examples_df['confID'] == 'EA')]
        if len(example_ea) > 0:
            example_ea = example_ea.sort_values('pred_rank')
            f.write(f"{'tmID':<8} {'Name':<20} {'Rank':>6} {'Pred':>6} {'Score':>8}\n")
            f.write("-" * 70 + "\n")
            for _, row in example_ea.iterrows():
                f.write(f"{row['tmID']:<8} {row['name']:<20} {row['rank']:>6.0f} "
                       f"{row['pred_rank']:>6.0f} {row['pred_score']:>8.2f}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("EXAMPLE PREDICTIONS - Year 9 West\n")
        f.write("-" * 70 + "\n")
        
        example_we = examples_df[(examples_df['year'] == 9) & (examples_df['confID'] == 'WE')]
        if len(example_we) > 0:
            example_we = example_we.sort_values('pred_rank')
            f.write(f"{'tmID':<8} {'Name':<20} {'Rank':>6} {'Pred':>6} {'Score':>8}\n")
            f.write("-" * 70 + "\n")
            for _, row in example_we.iterrows():
                f.write(f"{row['tmID']:<8} {row['name']:<20} {row['rank']:>6.0f} "
                       f"{row['pred_rank']:>6.0f} {row['pred_score']:>8.2f}\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"[TeamRanking] Saved report to {report_path}")


def main():
    """Main pipeline"""
    print("\n" + "=" * 70)
    print("TEAM RANKING MODEL - CONFERENCE RANKING PREDICTION")
    print("=" * 70)
    
    # 1. Load and merge data
    df_all = load_and_merge()
    
    # 2. Split train/test (temporal)
    train_raw, test_raw = split_train_test(df_all)
    
    # 3. Build features
    print("\n[TeamRanking] Building features...")
    X_train, y_train, meta_train = build_features(train_raw)
    X_test, y_test, meta_test = build_features(test_raw)
    print(f"  ✓ Train: {X_train.shape[1]} features, {len(X_train)} samples")
    print(f"  ✓ Test:  {X_test.shape[1]} features, {len(X_test)} samples")
    
    # 4. Train model
    model = fit_model(X_train, y_train)
    
    # 5. Predict
    print("\n[TeamRanking] Generating predictions...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    print("  ✓ Predictions generated")
    
    # 6. Convert to ranks
    train_with_ranks = add_predicted_rank(meta_train, y_pred_train)
    test_with_ranks = add_predicted_rank(meta_test, y_pred_test)
    
    # 7. Evaluate
    train_metrics = evaluate(train_with_ranks, "TRAIN")
    test_metrics = evaluate(test_with_ranks, "TEST")
    
    # 8. Save outputs
    save_predictions(train_with_ranks, test_with_ranks)
    save_report(train_metrics, test_metrics, X_train.columns.tolist(), test_with_ranks)
    
    print("\n" + "=" * 70)
    print("✓ PIPELINE COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("⚠️  DEPRECATED WRAPPER")
    print("=" * 70)
    print("This script is deprecated. Calling unified model...")
    print("For new work, use: src/model/ranking_model/team_ranking_model.py")
    print("=" * 70 + "\n")
    
    # Call unified model
    import sys
    sys.path.insert(0, str(ROOT / "src" / "model" / "ranking_model"))
    from team_ranking_model import run_team_ranking_model
    
    run_team_ranking_model(model_type="rf", max_train_year=8)

