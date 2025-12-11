"""
Team Performance Analysis (Season-level)
=========================================
Produces: data/processed/team_performance.csv

Columns:
- team_id, year, GP, won, lost, rs_win_pct
- pythag_win_pct (with fitted exponent)
- team_strength (roster-based, from player_performance)
- rs_win_pct_expected_roster (from linear regression)
- overach_pythag = rs_win_pct - pythag_win_pct
- overach_roster = rs_win_pct - rs_win_pct_expected_roster
- po_W, po_L, po_win_pct (playoffs)
- rs_win_pct_prev, win_pct_change
"""

from pathlib import Path
import pandas as pd
import numpy as np

# Paths
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)


def fit_pythag_exponent(df_stats: pd.DataFrame) -> float:
    """
    Fit optimal Pythagorean exponent by minimizing SSE between
    actual win% and pythag win%.
    
    Uses per-game stats: PF = o_pts/GP, PA = d_pts/GP
    """
    df = df_stats.copy()
    df = df[df['GP'] > 0].copy()
    df['actual_win_pct'] = df['won'] / df['GP']
    df['PF'] = df['o_pts'] / df['GP']
    df['PA'] = df['d_pts'] / df['GP']
    
    # Remove invalid rows
    df = df[(df['PF'] > 0) & (df['PA'] > 0)].copy()
    
    best_exp = 14.0  # Default for basketball
    best_sse = float('inf')
    
    # Grid search from 5 to 20, step 0.1
    for exp in np.arange(5.0, 20.1, 0.1):
        pythag = (df['PF'] ** exp) / ((df['PF'] ** exp) + (df['PA'] ** exp))
        sse = ((df['actual_win_pct'] - pythag) ** 2).sum()
        if sse < best_sse:
            best_sse = sse
            best_exp = exp
    
    return best_exp


def compute_team_strength(df_players: pd.DataFrame) -> pd.DataFrame:
    """
    Compute roster strength per team-season from player_performance.csv
    
    Expected columns: tmID, year, minutes, performance
    Returns DataFrame with: team_id, year, team_strength
    (team_strength = weighted avg of player performance by minutes)
    """
    # Validate schema
    required = {'tmID', 'year', 'minutes', 'performance'}
    missing = required - set(df_players.columns)
    if missing:
        raise KeyError(f"Missing columns in player_performance.csv: {missing}")
    
    df = df_players.copy()
    df['team_id'] = df['tmID']  # Convert to canonical name
    
    # Ensure numeric
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['minutes'] = pd.to_numeric(df['minutes'], errors='coerce')
    df['performance'] = pd.to_numeric(df['performance'], errors='coerce')
    
    # Filter valid rows
    df = df[(df['performance'].notna()) & (df['minutes'] > 0)].copy()
    
    # Compute weighted average performance per team-season
    df['weighted_perf'] = df['performance'] * df['minutes']
    
    grouped = df.groupby(['team_id', 'year'], dropna=False).agg({
        'weighted_perf': 'sum',
        'minutes': 'sum'
    }).reset_index()
    
    grouped['team_strength'] = grouped['weighted_perf'] / grouped['minutes']
    
    return grouped[['team_id', 'year', 'team_strength']]


def attach_team_results(df_strength: pd.DataFrame, df_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Merge team strength with team season stats.
    Compute pythag_win_pct and rs_win_pct.
    
    Expected columns in df_stats: tmID, year, won, lost, o_pts, d_pts, GP
    """
    # Validate schema
    required = {'tmID', 'year', 'won', 'lost', 'o_pts', 'd_pts'}
    missing = required - set(df_stats.columns)
    if missing:
        raise KeyError(f"Missing columns in team_season_statistics.csv: {missing}")
    
    df = df_stats.copy()
    df['team_id'] = df['tmID']  # Convert to canonical name
    
    # Ensure numeric
    for col in ['year', 'won', 'lost', 'GP', 'o_pts', 'd_pts']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Compute GP if missing
    if 'GP' not in df.columns or df['GP'].isna().all():
        df['GP'] = df['won'] + df['lost']
    
    # rs_win_pct
    df['rs_win_pct'] = df['won'] / df['GP'].replace(0, np.nan)
    
    # Fit Pythag exponent
    pythag_x = fit_pythag_exponent(df)
    print(f"\n[Team Performance] Fitted Pythagorean exponent: {pythag_x:.2f}")
    
    # Compute Pythag win%
    df['PF'] = df['o_pts'] / df['GP'].replace(0, np.nan)
    df['PA'] = df['d_pts'] / df['GP'].replace(0, np.nan)
    df['pythag_win_pct'] = (df['PF'] ** pythag_x) / (
        (df['PF'] ** pythag_x) + (df['PA'] ** pythag_x)
    )
    df['pythag_win_pct'] = df['pythag_win_pct'].clip(0, 1)
    
    # Merge with team strength
    df = df.merge(df_strength, on=['team_id', 'year'], how='left')
    
    # Drop temporary columns
    df = df.drop(columns=['PF', 'PA'], errors='ignore')
    
    return df


def attach_playoffs(df: pd.DataFrame, teams_post: pd.DataFrame) -> pd.DataFrame:
    """
    Attach playoff results from teams_post.csv
    
    Expected columns: tmID, year, W, L
    """
    # Validate schema
    required = {'tmID', 'year', 'W', 'L'}
    missing = required - set(teams_post.columns)
    if missing:
        raise KeyError(f"Missing columns in teams_post.csv: {missing}")
    
    po = teams_post.copy()
    po['team_id'] = po['tmID']  # Convert to canonical name
    
    # Ensure numeric
    po['year'] = pd.to_numeric(po['year'], errors='coerce')
    po['W'] = pd.to_numeric(po['W'], errors='coerce')
    po['L'] = pd.to_numeric(po['L'], errors='coerce')
    
    # Aggregate by team-year (in case of duplicates)
    po_agg = po.groupby(['team_id', 'year'], dropna=False).agg({
        'W': 'sum',
        'L': 'sum'
    }).reset_index()
    
    po_agg = po_agg.rename(columns={'W': 'po_W', 'L': 'po_L'})
    po_agg['po_win_pct'] = po_agg['po_W'] / (po_agg['po_W'] + po_agg['po_L'])
    po_agg['po_win_pct'] = po_agg['po_win_pct'].clip(0, 1)
    
    # Merge
    df = df.merge(po_agg, on=['team_id', 'year'], how='left')
    
    # Fill NaN with 0 for teams that didn't make playoffs
    for col in ['po_W', 'po_L']:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    return df


def compute_overachieves(df: pd.DataFrame, max_train_year: int | None = None) -> pd.DataFrame:
    """
    Compute overachievement metrics and roster-based expectations.
    
    Args:
        df: DataFrame with team performance data
        max_train_year: If provided, fit regression only on years <= max_train_year
                       to avoid temporal leakage. Predictions are made for all years.
                       This is critical for predictive models to ensure test data
                       doesn't influence the regression coefficients.

    """
    # Fit linear regression: rs_win_pct ~ team_strength
    # If max_train_year is specified, only fit on training years to avoid temporal leakage
    if max_train_year is not None:
        valid = df[
            (df['team_strength'].notna()) &
            (df['rs_win_pct'].notna()) &
            (df['year'] <= max_train_year)
        ].copy()
        print(f"[Team Performance] Fitting roster regression on years <= {max_train_year} (temporal split)")
    else:
        valid = df[df['team_strength'].notna() & df['rs_win_pct'].notna()].copy()
        print("[Team Performance] WARNING: Fitting roster regression on ALL years (includes future/test years)")
    
    if len(valid) > 1:
        from sklearn.linear_model import LinearRegression
        X = valid[['team_strength']].values
        y = valid['rs_win_pct'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict for all rows with team_strength
        df['rs_win_pct_expected_roster'] = np.nan
        mask = df['team_strength'].notna()
        df.loc[mask, 'rs_win_pct_expected_roster'] = model.predict(
            df.loc[mask, ['team_strength']].values
        )
        df['rs_win_pct_expected_roster'] = df['rs_win_pct_expected_roster'].clip(0, 1)
        
        r2 = model.score(X, y)
        print(f"[Team Performance] Roster strength R² = {r2:.3f}")
    else:
        df['rs_win_pct_expected_roster'] = np.nan
        print("[Team Performance] Not enough data for roster regression")
    
    # Compute overachieves
    df['overach_pythag'] = df['rs_win_pct'] - df['pythag_win_pct']
    df['overach_roster'] = df['rs_win_pct'] - df['rs_win_pct_expected_roster']
    
    # Previous year win%
    df = df.sort_values(['team_id', 'year'])
    df['rs_win_pct_prev'] = df.groupby('team_id')['rs_win_pct'].shift(1)
    df['win_pct_change'] = df['rs_win_pct'] - df['rs_win_pct_prev']
    
    return df


def main(max_train_year: int | None = None):
    """
    Main pipeline for team performance calculation.
    
    Args:
        max_train_year: If provided, fit roster regression only on years <= max_train_year
                       to avoid temporal leakage. Use this when generating data for
                       predictive models with a train/test split.
    """
    print("\n" + "="*60)
    print("TEAM PERFORMANCE PIPELINE")
    if max_train_year is not None:
        print(f"TEMPORAL SPLIT MODE: Fitting regressions only on years <= {max_train_year}")
    print("="*60)
    
    # 1. Load team season statistics
    stats_path = PROC_DIR / "team_season_statistics.csv"
    if not stats_path.exists():
        stats_path = RAW_DIR / "teams.csv"
    
    print(f"\n[1/5] Loading team stats from {stats_path.name}...")
    df_stats = pd.read_csv(stats_path)
    print(f"      Loaded {len(df_stats)} team-seasons")
    
    # 2. Load player performance for roster strength
    print("\n[2/5] Computing team roster strength...")
    player_perf_path = PROC_DIR / "player_performance.csv"
    if player_perf_path.exists():
        df_players = pd.read_csv(player_perf_path)
        df_strength = compute_team_strength(df_players)
        print(f"      Computed strength for {len(df_strength)} team-seasons")
    else:
        print("      player_performance.csv not found, skipping roster strength")
        df_strength = pd.DataFrame(columns=['team_id', 'year', 'team_strength'])
    
    # 3. Attach team results and compute Pythag
    print("\n[3/5] Computing Pythagorean expectation...")
    df = attach_team_results(df_strength, df_stats)
    
    # 4. Attach playoffs
    print("\n[4/5] Attaching playoff results...")
    po_path = RAW_DIR / "teams_post.csv"
    if po_path.exists():
        df_po = pd.read_csv(po_path)
        df = attach_playoffs(df, df_po)
        print(f"      Added playoff data for {df['po_W'].notna().sum()} team-seasons")
    else:
        print("      teams_post.csv not found, skipping playoffs")
    
    # 5. Compute overachieves
    print("\n[5/5] Computing overachievement metrics...")
    df = compute_overachieves(df, max_train_year=max_train_year)
    
    # Select canonical columns
    canonical_cols = [
        'team_id',                       
        'year',                          
        'GP',                            
        'won',                           
        'lost',                          
        'rs_win_pct',                    
        'pythag_win_pct',                
        'team_strength',                 
        'rs_win_pct_expected_roster',    
        'overach_pythag',                
        'overach_roster',                
        'po_W',                          
        'po_L',                          
        'po_win_pct',                    
        'rs_win_pct_prev',               
        'win_pct_change'                 
    ]
    
    # Keep only columns that exist
    cols_to_keep = [c for c in canonical_cols if c in df.columns]
    df_out = df[cols_to_keep].copy()
    
    # Sort
    df_out = df_out.sort_values(['team_id', 'year']).reset_index(drop=True)
    
    # Save
    out_path = PROC_DIR / "team_performance.csv"
    df_out.to_csv(out_path, index=False)
    
    print(f"\n✓ Saved {len(df_out)} rows to {out_path}")
    print("\n" + "-"*60)
    print("Sample (first 5 rows):")
    print(df_out.head().to_string())
    print("="*60 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compute team performance metrics including Pythagorean expectation and roster strength."
    )
    parser.add_argument(
        "--max-train-year",
        type=int,
        default=None,
        help="If set, fit roster regression only on years <= max_train_year to avoid temporal leakage. "
             "Use this when generating data for predictive models with a train/test split."
    )
    args = parser.parse_args()
    
    main(max_train_year=args.max_train_year)
