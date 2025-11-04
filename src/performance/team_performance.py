#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calculate team/coach-like performance metrics.

This module computes team strength from player performance and compares it
to actual team results to identify coaching impact (overperformance).

Key metrics:
  - team_strength: Minutes-weighted average of player performance
  - rs_win_pct_expected: Expected win% based on team_strength (OLS regression)
  - coach_like_overperf: Actual win% - Expected win% (coaching effect)
  - win_pct_change: Year-over-year change in win%
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np

# Setup project root
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ensure_team_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize team ID column names.
    
    Accepts any of: team_id, tmID, teamID
    Creates/ensures a 'team_id' column as string type.
    
    Args:
        df: DataFrame with a team identifier column
        
    Returns:
        DataFrame with 'team_id' column
    """
    df = df.copy()
    
    if "team_id" in df.columns:
        df["team_id"] = df["team_id"].astype(str)
    elif "tmID" in df.columns:
        df["team_id"] = df["tmID"].astype(str)
    elif "teamID" in df.columns:
        df["team_id"] = df["teamID"].astype(str)
    else:
        raise KeyError("No team identifier column found (expected: team_id, tmID, or teamID)")
    
    return df


def compute_team_strength(df_players: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate team strength as minutes-weighted average of player performance.
    
    Team strength represents the quality of the roster based on individual
    player performance, weighted by playing time.
    
    Args:
        df_players: DataFrame from player_performance.csv with columns:
            - bioID (or playerID): player identifier
            - tmID (or team_id): team identifier  
            - year: season year
            - minutes: minutes played
            - performance: player performance metric
    
    Returns:
        DataFrame with columns:
            - team_id: team identifier (string)
            - year: season year
            - team_strength: weighted average performance
            - total_minutes: sum of player minutes
            - n_players: number of players on team
    
    Example:
        >>> df_players = pd.DataFrame({
        ...     'bioID': ['p1', 'p2', 'p1', 'p2'],
        ...     'tmID': ['TEA', 'TEA', 'TEA', 'TEA'],
        ...     'year': [2020, 2020, 2021, 2021],
        ...     'minutes': [1000, 800, 1100, 750],
        ...     'performance': [50, 45, 52, 48]
        ... })
        >>> compute_team_strength(df_players)
          team_id  year  team_strength  total_minutes  n_players
        0     TEA  2020      47.777778           1800          2
        1     TEA  2021      50.378378           1850          2
    """
    df = df_players.copy()
    
    # Ensure team_id column
    df = _ensure_team_id(df)
    
    # Convert to numeric
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce")
    df["performance"] = pd.to_numeric(df["performance"], errors="coerce")
    
    # Filter out invalid rows (minutes <= 0 or missing performance)
    df = df[(df["minutes"] > 0) & (df["performance"].notna())].copy()
    
    # Calculate weighted performance contribution
    df["perf_x_min"] = df["performance"] * df["minutes"]
    
    # Get player ID column (bioID or playerID)
    if "bioID" in df.columns:
        player_col = "bioID"
    elif "playerID" in df.columns:
        player_col = "playerID"
    else:
        raise KeyError("Expected either 'bioID' or 'playerID' in player_performance.csv")
    
    # Aggregate by team-year
    team_agg = df.groupby(["team_id", "year"], as_index=False).agg({
        "perf_x_min": "sum",
        "minutes": "sum",
        player_col: "nunique"
    })
    
    # Rename columns for clarity
    team_agg = team_agg.rename(columns={
        "minutes": "total_minutes",
        player_col: "n_players"
    })
    
    # Calculate team strength (minutes-weighted average performance)
    team_agg["team_strength"] = team_agg["perf_x_min"] / team_agg["total_minutes"]
    team_agg["team_strength"] = team_agg["team_strength"].fillna(0.0)
    
    # Drop intermediate column
    team_agg = team_agg.drop(columns=["perf_x_min"])
    
    return team_agg[["team_id", "year", "team_strength", "total_minutes", "n_players"]]


def attach_team_results(team_strength_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach actual team results from team_season_statistics.csv.
    
    Merges team strength data with win percentage and other team statistics.
    
    Args:
        team_strength_df: DataFrame with team_id, year, team_strength columns
        
    Returns:
        DataFrame with added columns:
            - season_win_pct (or rs_win_pct): actual regular season win percentage
            - Additional team stats from team_season_statistics.csv
    """
    # Load team season statistics
    team_stats_path = ROOT / "data" / "processed" / "team_season_statistics.csv"
    
    if not team_stats_path.exists():
        raise FileNotFoundError(f"Cannot find {team_stats_path}")
    
    df_stats = pd.read_csv(team_stats_path)
    
    # Ensure team_id column
    df_stats = _ensure_team_id(df_stats)
    
    # Ensure year is numeric
    if "year" in df_stats.columns:
        df_stats["year"] = pd.to_numeric(df_stats["year"], errors="coerce")
    elif "season" in df_stats.columns:
        df_stats["year"] = pd.to_numeric(df_stats["season"], errors="coerce")
    else:
        raise KeyError("Expected 'year' or 'season' column in team_season_statistics.csv")
    
    # Ensure rs_win_pct column exists
    if "season_win_pct" in df_stats.columns:
        df_stats["rs_win_pct"] = pd.to_numeric(df_stats["season_win_pct"], errors="coerce")
    elif "rs_win_pct" in df_stats.columns:
        df_stats["rs_win_pct"] = pd.to_numeric(df_stats["rs_win_pct"], errors="coerce")
    elif "won" in df_stats.columns and "lost" in df_stats.columns:
        won = pd.to_numeric(df_stats["won"], errors="coerce")
        lost = pd.to_numeric(df_stats["lost"], errors="coerce")
        total_games = won + lost
        df_stats["rs_win_pct"] = np.where(total_games > 0, won / total_games, np.nan)
    else:
        raise KeyError("Cannot find win percentage data (need season_win_pct, rs_win_pct, or won/lost)")
    
    # Select relevant columns for merge
    merge_cols = ["team_id", "year", "rs_win_pct"]
    
    # Add other useful columns if they exist
    optional_cols = ["won", "lost", "GP", "name", "playoff"]
    for col in optional_cols:
        if col in df_stats.columns:
            merge_cols.append(col)
    
    df_stats_subset = df_stats[merge_cols].copy()
    
    # Merge with team strength
    result = team_strength_df.merge(
        df_stats_subset,
        on=["team_id", "year"],
        how="left"
    )
    
    return result


def compute_team_overperformance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute team overperformance using OLS regression.
    
    Fits: rs_win_pct ~ team_strength
    Then calculates:
      - rs_win_pct_expected: predicted win% based on team strength
      - coach_like_overperf: actual - expected (coaching effect)
      - win_pct_change: year-over-year change in actual win%
    
    Args:
        df: DataFrame with team_id, year, team_strength, rs_win_pct
        
    Returns:
        DataFrame with added columns:
            - rs_win_pct_expected: predicted win% from regression
            - coach_like_overperf: actual - expected win%
            - rs_win_pct_prev: previous year's win%
            - win_pct_change: change from previous year
    """
    df = df.copy()
    
    # Ensure numeric types
    df["team_strength"] = pd.to_numeric(df["team_strength"], errors="coerce")
    df["rs_win_pct"] = pd.to_numeric(df["rs_win_pct"], errors="coerce")
    
    # Filter to rows with valid data for regression
    reg_data = df[["team_strength", "rs_win_pct"]].dropna()
    
    if len(reg_data) < 10:
        print(f"Warning: Only {len(reg_data)} valid observations for regression. Results may be unreliable.")
    
    # Fit OLS regression: rs_win_pct ~ team_strength
    X = reg_data["team_strength"].values
    y = reg_data["rs_win_pct"].values
    
    # Add intercept to X
    X_design = np.column_stack([np.ones(len(X)), X])
    
    # Solve OLS using least squares
    coeffs, residuals, rank, s = np.linalg.lstsq(X_design, y, rcond=None)
    intercept, slope = coeffs
    
    print(f"\nRegression: rs_win_pct ~ team_strength")
    print(f"  Intercept: {intercept:.4f}")
    print(f"  Slope:     {slope:.4f}")
    print(f"  N obs:     {len(reg_data)}")
    
    # Calculate expected win% for all rows
    df["rs_win_pct_expected"] = intercept + slope * df["team_strength"]
    
    # Clip expected values to valid range [0, 1]
    df["rs_win_pct_expected"] = df["rs_win_pct_expected"].clip(0.0, 1.0)
    
    # Calculate overperformance (coaching effect)
    df["coach_like_overperf"] = df["rs_win_pct"] - df["rs_win_pct_expected"]
    
    # Sort by team and year for year-over-year calculations
    df = df.sort_values(["team_id", "year"]).reset_index(drop=True)
    
    # Calculate previous year's win% and change
    df["rs_win_pct_prev"] = df.groupby("team_id")["rs_win_pct"].shift(1)
    df["win_pct_change"] = df["rs_win_pct"] - df["rs_win_pct_prev"]
    
    return df


def calculate_coach_like_performance():
    """
    Calculate team/coach-like performance metrics.
    
    Pipeline:
      1. Load player performance data
      2. Compute team strength (minutes-weighted player performance)
      3. Attach actual team results (win%)
      4. Run regression to get expected win% and compute overperformance
      5. Save to team_performance.csv
    """
    print("=" * 60)
    print("TEAM/COACH-LIKE PERFORMANCE CALCULATION")
    print("=" * 60)
    
    # 1) Load player performance
    print("\n[1/4] Loading player performance...")
    player_path = ROOT / "data" / "processed" / "player_performance.csv"
    df_players = pd.read_csv(player_path)
    print(f"  ✓ Loaded {len(df_players):,} player-season records")
    
    # 2) Compute team strength
    print("\n[2/4] Computing team strength (minutes-weighted performance)...")
    df_team_strength = compute_team_strength(df_players)
    print(f"  ✓ Computed team strength for {len(df_team_strength)} team-seasons")
    print(f"  Team strength range: [{df_team_strength['team_strength'].min():.2f}, {df_team_strength['team_strength'].max():.2f}]")
    
    # 3) Attach team results
    print("\n[3/4] Attaching team results (win%)...")
    df_with_results = attach_team_results(df_team_strength)
    print(f"  ✓ Merged with team statistics")
    print(f"  Teams with win% data: {df_with_results['rs_win_pct'].notna().sum()}")
    
    # 4) Compute overperformance
    print("\n[4/4] Computing overperformance (coaching effect)...")
    df_final = compute_team_overperformance(df_with_results)
    
    # Save results
    out_path = ROOT / "data" / "processed" / "team_performance.csv"
    df_final.to_csv(out_path, index=False)
    
    print(f"\n✓ Saved team performance to: {out_path}")
    print(f"  Rows: {len(df_final):,}")
    
    # Show summary
    print("\nTeam strength summary:")
    print(df_final["team_strength"].describe().round(2))
    
    print("\nCoach-like overperformance summary:")
    print(df_final["coach_like_overperf"].describe().round(3))
    
    print("\nSample of results:")
    display_cols = [
        "team_id", "year", "team_strength", "rs_win_pct", 
        "rs_win_pct_expected", "coach_like_overperf", "win_pct_change"
    ]
    print(df_final[display_cols].head(15).to_string(index=False))
    
    print("\n" + "=" * 60)
    
    return df_final


if __name__ == "__main__":
    calculate_coach_like_performance()
