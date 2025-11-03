"""
Module to calculate position-specific statistical weights using regression analysis.

This module reads cleaned player data, performs multiple linear regression (with Ridge regularization)
to derive optimal weights for each position category, and saves the results to a JSON file.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def _pos_to_role(pos: str) -> str:
    """
    Map position string to role category.
    
    Args:
        pos: Position string (e.g., 'C', 'F', 'G', 'C-F', etc.)
    
    Returns:
        Role category: 'guard', 'wing', 'forward', 'forward_center', 'center', or 'unknown'
    """
    if pd.isna(pos):
        return 'unknown'
    
    pos_upper = str(pos).upper()
    
    if 'G' in pos_upper and 'C' not in pos_upper and 'F' not in pos_upper:
        return 'guard'
    if 'G' in pos_upper and 'F' in pos_upper:
        return 'wing'
    if 'F' in pos_upper and 'C' in pos_upper:
        return 'forward_center'
    if 'C' in pos_upper:
        return 'center'
    if 'F' in pos_upper:
        return 'forward'
    
    return 'unknown'


def load_and_prepare_data(
    players_teams_path: Path,
    players_cleaned_path: Path
) -> pd.DataFrame:
    """
    Load and merge player statistics with position information.
    
    Args:
        players_teams_path: Path to raw players_teams.csv
        players_cleaned_path: Path to cleaned players.csv with position data
    
    Returns:
        Merged DataFrame with statistics and positions
    """
    print(f"Loading data from {players_teams_path}...")
    stats = pd.read_csv(players_teams_path)
    
    print(f"Loading position data from {players_cleaned_path}...")
    players = pd.read_csv(players_cleaned_path)
    
    # Rename columns to standard format
    rename_map = {
        'playerID': 'bioID',
        'minutes': 'mp',
        'points': 'pts',
        'rebounds': 'trb',
        'assists': 'ast',
        'steals': 'stl',
        'blocks': 'blk',
        'turnovers': 'tov'
    }
    stats = stats.rename(columns={k: v for k, v in rename_map.items() if k in stats.columns})
    
    # Merge with player info to get positions
    merged = stats.merge(players[['bioID', 'pos']], on='bioID', how='left')
    
    # Map positions to roles
    merged['role'] = merged['pos'].apply(_pos_to_role)
    
    return merged


def calculate_performance_target(df: pd.DataFrame) -> pd.Series:
    """
    Calculate a composite performance metric to use as regression target.
    
    Uses a balanced combination of efficiency metrics normalized per 36 minutes.
    
    Args:
        df: DataFrame with player statistics
    
    Returns:
        Series with performance scores
    """
    # Ensure we have the necessary columns
    required = ['mp', 'pts', 'trb', 'ast', 'stl', 'blk', 'tov']
    for col in required:
        if col not in df.columns:
            df[col] = 0.0
    
    # Calculate per-36 statistics
    df = df.copy()
    df['mp'] = df['mp'].replace(0, np.nan)
    
    # Base efficiency metric (simple weighted combination)
    # This serves as our "ground truth" for what good performance looks like
    per36_pts = (df['pts'] / df['mp']) * 36
    per36_reb = (df['trb'] / df['mp']) * 36
    per36_ast = (df['ast'] / df['mp']) * 36
    per36_stl = (df['stl'] / df['mp']) * 36
    per36_blk = (df['blk'] / df['mp']) * 36
    per36_tov = (df['tov'] / df['mp']) * 36
    
    # Composite metric: balanced weights as baseline
    performance = (
        1.0 * per36_pts +
        0.8 * per36_reb +
        0.8 * per36_ast +
        1.2 * per36_stl +
        1.2 * per36_blk -
        0.8 * per36_tov
    )
    
    return performance.fillna(0.0)


def train_position_weights(
    df: pd.DataFrame,
    min_samples: int = 30,
    alpha: float = 1.0
) -> Dict[str, Dict[str, float]]:
    """
    Train Ridge regression models for each position to derive optimal stat weights.
    
    Args:
        df: DataFrame with player statistics and role assignments
        min_samples: Minimum number of samples required to train a position model
        alpha: Ridge regression regularization parameter
    
    Returns:
        Dictionary mapping role -> stat weights
    """
    stat_cols = ['pts', 'trb', 'ast', 'stl', 'blk', 'tov']
    weights_by_role = {}
    
    # Default weights (fallback)
    default_weights = {
        'pts': 1.0, 'reb': 0.7, 'ast': 0.7, 
        'stl': 1.0, 'blk': 1.0, 'tov': -0.7
    }
    
    # Filter to rows with sufficient minutes (avoid noisy data)
    df_filtered = df[df['mp'] >= 100].copy()
    
    # Calculate target performance metric
    df_filtered['target'] = calculate_performance_target(df_filtered)
    
    # Calculate per-36 features
    for col in stat_cols:
        df_filtered[f'{col}_per36'] = (df_filtered[col] / df_filtered['mp']) * 36
    
    roles = ['guard', 'wing', 'forward', 'forward_center', 'center']
    
    for role in roles:
        role_data = df_filtered[df_filtered['role'] == role].copy()
        
        if len(role_data) < min_samples:
            print(f"[WARN] Insufficient data for {role} ({len(role_data)} samples), using defaults")
            weights_by_role[role] = default_weights.copy()
            continue
        
        # Prepare features (per-36 stats)
        feature_cols = [f'{col}_per36' for col in stat_cols]
        X = role_data[feature_cols].fillna(0.0).values
        y = role_data['target'].values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train Ridge regression
        model = Ridge(alpha=alpha, random_state=42)
        model.fit(X_scaled, y)
        
        # Extract coefficients and normalize
        coefs = model.coef_
        
        # Convert back to original scale (approximate)
        # Normalize by points coefficient to make pts weight = 1.0
        pts_idx = stat_cols.index('pts')
        pts_coef = coefs[pts_idx] if coefs[pts_idx] != 0 else 1.0
        
        normalized_weights = {}
        for i, stat in enumerate(stat_cols):
            weight = coefs[i] / pts_coef if pts_coef != 0 else default_weights.get(stat, 1.0)
            # Map 'trb' back to 'reb' for consistency
            key = 'reb' if stat == 'trb' else stat
            normalized_weights[key] = round(float(weight), 2)
        
        weights_by_role[role] = normalized_weights
        
        print(f"[OK] Trained weights for {role}: {normalized_weights} ({len(role_data)} samples)")
    
    # Add 'unknown' role with default weights
    weights_by_role['unknown'] = default_weights.copy()
    
    return weights_by_role


def save_weights(weights: Dict[str, Dict[str, float]], output_path: Path) -> None:
    """
    Save position weights to JSON file.
    
    Args:
        weights: Dictionary of position weights
        output_path: Path to save JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(weights, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] Weights saved to: {output_path}")


def main():
    """Main execution function."""
    base = Path(__file__).resolve().parents[3]
    
    # Input paths
    players_teams_path = base / "data" / "raw" / "players_teams.csv"
    players_cleaned_path = base / "data" / "processed" / "players_cleaned.csv"
    
    # Output path
    output_path = base / "data" / "processed" / "position_weights.json"
    
    # Validate inputs
    if not players_teams_path.exists():
        return 1
    
    if not players_cleaned_path.exists():
        return 1
    
    
    # Load and prepare data
    try:
        df = load_and_prepare_data(players_teams_path, players_cleaned_path)
    except Exception as e:
        return 1
    
    # Train weights
    try:
        weights = train_position_weights(df, min_samples=30, alpha=1.0)
    except Exception as e:
        return 1
    
    # Save results
    try:
        save_weights(weights, output_path)
    except Exception as e:
        return 1
    
    print("\n" + "=" * 60)
    print("Weight calculation completed successfully!")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
