"""
Team Ranking Prediction Model

This script builds a machine learning model to predict the final conference ranking
of basketball teams based on their offensive, defensive, and aggregate performance
statistics.

The model:
- Merges team_performance.csv and teams_cleaned.csv
- Uses seasons 1-8 for training and seasons 9-10 for testing
- Avoids data leakage by excluding outcome variables (rank, won, lost, playoff)
- Generates a comprehensive report with metrics and predictions
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 1. DATA LOADING AND MERGING
# ============================================================================

def load_and_merge_data(repo_root):
    """
    Load and merge team_performance.csv and teams_cleaned.csv
    
    Args:
        repo_root: Path to repository root
        
    Returns:
        Merged DataFrame
    """
    # Load datasets
    team_perf_path = repo_root / "data" / "processed" / "team_performance.csv"
    teams_cleaned_path = repo_root / "data" / "processed" / "teams_cleaned.csv"
    
    df_perf = pd.read_csv(team_perf_path)
    df_teams = pd.read_csv(teams_cleaned_path)
    
    print(f"✓ Loaded team_performance.csv: {df_perf.shape}")
    print(f"✓ Loaded teams_cleaned.csv: {df_teams.shape}")
    
    # Merge on year and team_id/tmID
    df_merged = pd.merge(
        df_perf,
        df_teams,
        left_on=['year', 'team_id'],
        right_on=['year', 'tmID'],
        how='inner'
    )
    
    print(f"✓ Merged dataset: {df_merged.shape}")
    return df_merged


# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================

def select_features(df):
    """
    Select features for the model, avoiding data leakage.
    
    Excludes: rank, won, lost, playoff, firstRound, semis, finals, 
              rs_win_pct, po_W, po_L, po_win_pct
    
    Args:
        df: Merged DataFrame
        
    Returns:
        Feature names list
    """
    # Columns to exclude (outcome variables and derivatives)
    exclude_cols = [
        'rank', 'won', 'lost', 'playoff', 'firstRound', 'semis', 'finals',
        'rs_win_pct', 'po_W', 'po_L', 'po_win_pct',
        'year', 'team_id', 'tmID', 'franchID', 'confID', 'name', 'arena',
        'is_last_season', 'homeW', 'homeL', 'awayW', 'awayL', 
        'confW', 'confL', 'GP', 'min', 'attend'
    ]
    
    # Select numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove excluded columns and any that contain 'won' or 'lost' (from merge)
    features = [
        col for col in numeric_cols 
        if col not in exclude_cols 
        and not col.endswith('_x')  # Remove duplicates from merge (won_x, lost_x)
        and not col.endswith('_y')  # Remove duplicates from merge (won_y, lost_y)
        and 'won' not in col.lower()
        and 'lost' not in col.lower()
    ]
    
    return features


def prepare_data(df, features, target='rank'):
    """
    Prepare data for modeling: handle missing values and create train/test splits
    
    Args:
        df: Merged DataFrame
        features: List of feature names
        target: Target variable name
        
    Returns:
        X_train, X_test, y_train, y_test, train_df, test_df
    """
    # Remove rows with missing target
    df = df.dropna(subset=[target])
    
    # Split by year
    train_df = df[df['year'].between(1, 8)].copy()
    test_df = df[df['year'].between(9, 10)].copy()
    
    print(f"\n✓ Training samples: {len(train_df)} (years 1-8)")
    print(f"✓ Testing samples: {len(test_df)} (years 9-10)")
    
    # Handle missing values in features
    # Fill with median for each feature
    for feat in features:
        if train_df[feat].isna().any():
            median_val = train_df[feat].median()
            train_df[feat].fillna(median_val, inplace=True)
            test_df[feat].fillna(median_val, inplace=True)
    
    # Extract features and target
    X_train = train_df[features]
    X_test = test_df[features]
    y_train = train_df[target]
    y_test = test_df[target]
    
    return X_train, X_test, y_train, y_test, train_df, test_df


# ============================================================================
# 3. MODEL TRAINING
# ============================================================================

def train_model(X_train, y_train, model_type='gradient_boosting'):
    """
    Train a regression model to predict rank
    
    Args:
        X_train: Training features
        y_train: Training target
        model_type: 'gradient_boosting' or 'random_forest'
        
    Returns:
        Trained model, scaler
    """
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model
    if model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=42,
            verbose=0
        )
    else:  # random_forest
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_split=5,
            random_state=42,
            verbose=0
        )
    
    model.fit(X_train_scaled, y_train)
    print(f"\n✓ Model trained: {model_type}")
    
    return model, scaler


# ============================================================================
# 4. EVALUATION AND REPORTING
# ============================================================================

def evaluate_model(model, scaler, X_test, y_test, features):
    """
    Evaluate model performance
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        X_test: Test features
        y_test: Test target
        features: Feature names
        
    Returns:
        Dictionary with metrics
    """
    # Scale test data
    X_test_scaled = scaler.transform(X_test)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    
    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importance = sorted(
            zip(features, model.feature_importances_),
            key=lambda x: x[1],
            reverse=True
        )
    else:
        importance = []
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'predictions': y_pred,
        'importance': importance
    }
    
    print(f"\n✓ MAE: {mae:.4f}")
    print(f"✓ RMSE: {rmse:.4f}")
    print(f"✓ R²: {r2:.4f}")
    
    return metrics


def generate_ranking_comparison(test_df, predictions):
    """
    Generate ranking comparison between actual and predicted ranks
    
    Args:
        test_df: Test DataFrame
        predictions: Predicted ranks
        
    Returns:
        DataFrame with ranking comparison, Spearman correlations
    """
    test_df = test_df.copy()
    test_df['predicted_rank'] = predictions
    
    # Round predictions to nearest integer for ranking
    test_df['predicted_rank_rounded'] = test_df['predicted_rank'].round()
    
    # Calculate Spearman correlation per year
    spearman_scores = []
    for year in sorted(test_df['year'].unique()):
        year_data = test_df[test_df['year'] == year]
        corr, _ = spearmanr(year_data['rank'], year_data['predicted_rank'])
        spearman_scores.append((year, corr))
    
    mean_spearman = np.mean([s for _, s in spearman_scores])
    
    # Select columns for comparison
    comparison = test_df[[
        'year', 'team_id', 'confID', 'rank', 'predicted_rank', 
        'predicted_rank_rounded'
    ]].sort_values(['year', 'rank'])
    
    return comparison, spearman_scores, mean_spearman


def save_report(metrics, comparison, spearman_scores, mean_spearman, 
                features, model_type, output_path='report.txt'):
    """
    Generate comprehensive report
    
    Args:
        metrics: Dictionary with evaluation metrics
        comparison: DataFrame with ranking comparison
        spearman_scores: List of (year, correlation) tuples
        mean_spearman: Mean Spearman correlation
        features: Feature names
        model_type: Model type string
        output_path: Path to save report
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write("TEAM RANKING PREDICTION MODEL - COMPREHENSIVE REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Model description
        f.write("MODEL DESCRIPTION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Algorithm: {model_type.replace('_', ' ').title()}\n")
        f.write("Training Data: Seasons 1-8\n")
        f.write("Testing Data: Seasons 9-10\n")
        f.write("Target Variable: Conference Rank (position)\n")
        f.write("Number of Features: {}\n\n".format(len(features)))
        
        # Data preprocessing
        f.write("DATA PREPROCESSING\n")
        f.write("-" * 80 + "\n")
        f.write("1. Merged team_performance.csv and teams_cleaned.csv\n")
        f.write("2. Removed data leakage variables: rank, won, lost, playoff, etc.\n")
        f.write("3. Handled missing values using median imputation\n")
        f.write("4. Standardized features using StandardScaler\n\n")
        
        # Performance metrics
        f.write("PERFORMANCE METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Mean Absolute Error (MAE):     {metrics['mae']:.4f} positions\n")
        f.write(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.4f} positions\n")
        f.write(f"R² Score:                       {metrics['r2']:.4f}\n")
        f.write(f"Mean Spearman Correlation:      {mean_spearman:.4f}\n\n")
        
        # Interpretation
        f.write("INTERPRETATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"On average, the model's predictions are off by {metrics['mae']:.2f} positions.\n")
        if metrics['r2'] > 0.7:
            f.write(f"The R² of {metrics['r2']:.4f} indicates strong predictive power.\n")
        elif metrics['r2'] > 0.5:
            f.write(f"The R² of {metrics['r2']:.4f} indicates moderate predictive power.\n")
        else:
            f.write(f"The R² of {metrics['r2']:.4f} indicates room for improvement.\n")
        f.write(f"The Spearman correlation of {mean_spearman:.4f} shows how well the model\n")
        f.write("preserves the relative ordering of teams.\n\n")
        
        # Per-year performance
        f.write("PER-YEAR SPEARMAN CORRELATION\n")
        f.write("-" * 80 + "\n")
        for year, corr in spearman_scores:
            f.write(f"  Season {year:2d}: {corr:7.4f}\n")
        f.write("\n")
        
        # Feature importance
        if metrics['importance']:
            f.write("TOP 15 MOST IMPORTANT FEATURES\n")
            f.write("-" * 80 + "\n")
            for i, (feat, val) in enumerate(metrics['importance'][:15], 1):
                f.write(f"{i:2d}. {feat:40s} {val:.6f}\n")
            f.write("\n")
        
        # Ranking comparison
        f.write("RANKING COMPARISON (ACTUAL vs PREDICTED)\n")
        f.write("=" * 80 + "\n\n")
        
        for year in sorted(comparison['year'].unique()):
            year_data = comparison[comparison['year'] == year].copy()
            f.write(f"SEASON {year}\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Team':<8} {'Conf':<6} {'Actual Rank':<12} {'Predicted Rank':<15} {'Difference':<10}\n")
            f.write("-" * 80 + "\n")
            
            for _, row in year_data.iterrows():
                diff = row['predicted_rank_rounded'] - row['rank']
                f.write(f"{row['team_id']:<8} {row['confID']:<6} "
                       f"{int(row['rank']):<12} {row['predicted_rank_rounded']:<15.0f} "
                       f"{diff:+.0f}\n")
            f.write("\n")
        
        # Model explanation
        f.write("\n" + "=" * 80 + "\n")
        f.write("MODEL EXPLANATION\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("HOW IT WORKS:\n")
        f.write("-" * 80 + "\n")
        f.write("1. The model learns patterns from teams' offensive and defensive statistics\n")
        f.write("2. It combines performance metrics like shooting efficiency, rebounds, assists\n")
        f.write("3. Historical performance indicators (previous season win %) are considered\n")
        f.write("4. The model predicts the conference rank directly as a numeric value\n")
        f.write("5. Predictions are evaluated against actual final standings\n\n")
        
        f.write("DATA LEAKAGE PREVENTION:\n")
        f.write("-" * 80 + "\n")
        f.write("The following variables were EXCLUDED to prevent data leakage:\n")
        f.write("- rank, won, lost (outcome variables)\n")
        f.write("- playoff, firstRound, semis, finals (derived from rank)\n")
        f.write("- rs_win_pct, po_W, po_L, po_win_pct (direct win/loss metrics)\n")
        f.write("- homeW, homeL, awayW, awayL, confW, confL (split win/loss records)\n\n")
        
        f.write("FEATURES USED:\n")
        f.write("-" * 80 + "\n")
        f.write("The model uses:\n")
        f.write("- Offensive statistics (field goals, free throws, 3-pointers, rebounds, etc.)\n")
        f.write("- Defensive statistics (opponent's shooting, turnovers, steals, blocks)\n")
        f.write("- Aggregate metrics (pythag win %, team strength, expected roster performance)\n")
        f.write("- Historical indicators (previous season win %, win % change)\n\n")
        
        f.write("FUTURE IMPROVEMENTS:\n")
        f.write("-" * 80 + "\n")
        f.write("1. Incorporate player performance metrics (scoring, efficiency, experience)\n")
        f.write("2. Add coach performance indicators (years of experience, past success)\n")
        f.write("3. Include rookie development features\n")
        f.write("4. Consider team chemistry and roster stability metrics\n")
        f.write("5. Experiment with ensemble methods or deep learning approaches\n")
        f.write("6. Add cross-validation for more robust evaluation\n")
        f.write("7. Feature engineering: create interaction terms and derived statistics\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"\n✅ Report saved to: {output_path}")


# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function
    """
    print("\n" + "=" * 80)
    print("TEAM RANKING PREDICTION MODEL")
    print("=" * 80 + "\n")
    
    # Setup paths
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent.parent
    
    # 1. Load and merge data
    print("STEP 1: Loading and merging datasets...")
    df = load_and_merge_data(repo_root)
    
    # 2. Feature engineering
    print("\nSTEP 2: Feature engineering...")
    features = select_features(df)
    print(f"✓ Selected {len(features)} features")
    
    # 3. Prepare data
    print("\nSTEP 3: Preparing data...")
    X_train, X_test, y_train, y_test, train_df, test_df = prepare_data(
        df, features, target='rank'
    )
    
    # 4. Train model
    print("\nSTEP 4: Training model...")
    model, scaler = train_model(X_train, y_train, model_type='gradient_boosting')
    
    # 5. Evaluate
    print("\nSTEP 5: Evaluating model...")
    metrics = evaluate_model(model, scaler, X_test, y_test, features)
    
    # 6. Generate ranking comparison
    print("\nSTEP 6: Generating ranking comparison...")
    comparison, spearman_scores, mean_spearman = generate_ranking_comparison(
        test_df, metrics['predictions']
    )
    print(f"✓ Mean Spearman Correlation: {mean_spearman:.4f}")
    
    # 7. Save report
    print("\nSTEP 7: Saving report...")
    report_path = script_dir / "report.txt"
    save_report(
        metrics, comparison, spearman_scores, mean_spearman,
        features, 'gradient_boosting', output_path=report_path
    )
    
    print("\n" + "=" * 80)
    print("✅ MODEL TRAINING AND EVALUATION COMPLETE!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
