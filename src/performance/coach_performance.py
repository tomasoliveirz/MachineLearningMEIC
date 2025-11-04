#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calculate coach performance metrics.

This module aggregates team performance by coach to identify coaching impact
across different teams and seasons.

Key metrics (to be implemented):
  - Career overperformance: Average coach_like_overperf across all teams/years
  - Consistency: Std dev of overperformance
  - Career win%: Weighted average win percentage
  - Improvement rate: Average win_pct_change when joining teams
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np

# Setup project root
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_coaches_data() -> pd.DataFrame:
    """
    Load coaches.csv and prepare for analysis.
    
    Returns:
        DataFrame with coach-team-season assignments
    """
    coaches_path = ROOT / "data" / "raw" / "coaches.csv"
    
    if not coaches_path.exists():
        raise FileNotFoundError(f"Cannot find {coaches_path}")
    
    df = pd.read_csv(coaches_path)
    
    # Normalize column names
    if "tmID" in df.columns:
        df["team_id"] = df["tmID"].astype(str)
    elif "teamID" in df.columns:
        df["team_id"] = df["teamID"].astype(str)
    
    # Ensure year is numeric
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    
    # Ensure coachID is string
    if "coachID" in df.columns:
        df["coachID"] = df["coachID"].astype(str)
    
    return df


def load_team_performance() -> pd.DataFrame:
    """
    Load team_performance.csv with team strength and overperformance metrics.
    
    Returns:
        DataFrame with team-season performance
    """
    team_perf_path = ROOT / "data" / "processed" / "team_performance.csv"
    
    if not team_perf_path.exists():
        raise FileNotFoundError(
            f"Cannot find {team_perf_path}. "
            f"Run team_performance.py first to generate this file."
        )
    
    df = pd.read_csv(team_perf_path)
    
    # Ensure team_id and year are correct types
    df["team_id"] = df["team_id"].astype(str)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    
    return df


def merge_coach_team_performance(
    coaches_df: pd.DataFrame, 
    team_perf_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge coaches with team performance data.
    
    Args:
        coaches_df: DataFrame with coach assignments
        team_perf_df: DataFrame with team performance metrics
        
    Returns:
        DataFrame with coach-team-season rows and performance metrics
    """
    # Merge on team_id and year
    merged = coaches_df.merge(
        team_perf_df,
        on=["team_id", "year"],
        how="inner"
    )
    
    return merged


def aggregate_coach_performance(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate performance metrics by coach.
    
    This is a placeholder that will compute:
      - Total seasons coached
      - Career overperformance (mean coach_like_overperf)
      - Career win% (weighted by games)
      - Consistency (std of overperformance)
      - Average team strength when coaching
    
    Args:
        merged_df: DataFrame with coach-team-season rows
        
    Returns:
        DataFrame with one row per coach
    """
    # TODO: Implement full aggregation logic
    # For now, just compute basic stats
    
    coach_stats = merged_df.groupby("coachID").agg({
        "year": "count",  # seasons
        "team_id": "nunique",  # teams
        "coach_like_overperf": "mean",  # avg overperformance
        "rs_win_pct": "mean",  # avg win%
        "team_strength": "mean",  # avg team strength
    }).reset_index()
    
    coach_stats = coach_stats.rename(columns={
        "year": "seasons",
        "team_id": "teams",
        "coach_like_overperf": "avg_overperf",
        "rs_win_pct": "avg_win_pct",
        "team_strength": "avg_team_strength"
    })
    
    return coach_stats


def calculate_coach_performance():
    """
    Main pipeline for coach performance calculation.
    
    Currently a stub that:
      1. Loads coaches.csv
      2. Loads team_performance.csv
      3. Merges them
      4. Shows sample of merged data
      5. Computes basic aggregation by coach
    """
    print("=" * 60)
    print("COACH PERFORMANCE CALCULATION (STUB)")
    print("=" * 60)
    
    # 1) Load coaches
    print("\n[1/3] Loading coaches data...")
    coaches_df = load_coaches_data()
    print(f"  ✓ Loaded {len(coaches_df)} coach-team-season assignments")
    print(f"  Unique coaches: {coaches_df['coachID'].nunique()}")
    
    # 2) Load team performance
    print("\n[2/3] Loading team performance...")
    team_perf_df = load_team_performance()
    print(f"  ✓ Loaded {len(team_perf_df)} team-season records")
    
    # 3) Merge
    print("\n[3/3] Merging coach assignments with team performance...")
    merged = merge_coach_team_performance(coaches_df, team_perf_df)
    print(f"  ✓ Merged {len(merged)} coach-team-season records")
    
    # Show sample
    print("\nSample of merged data:")
    display_cols = [
        "coachID", "team_id", "year", 
        "team_strength", "rs_win_pct", "coach_like_overperf"
    ]
    display_cols = [c for c in display_cols if c in merged.columns]
    print(merged[display_cols].head(10).to_string(index=False))
    
    # Basic aggregation
    print("\n" + "=" * 60)
    print("BASIC COACH AGGREGATION")
    print("=" * 60)
    
    coach_stats = aggregate_coach_performance(merged)
    
    print("\nCoach performance summary (top 10 by avg overperformance):")
    print(coach_stats.sort_values("avg_overperf", ascending=False).head(10).to_string(index=False))
    
    # Save
    out_path = ROOT / "data" / "processed" / "coach_performance.csv"
    coach_stats.to_csv(out_path, index=False)
    print(f"\n✓ Saved basic coach stats to: {out_path}")
    
    print("\n" + "=" * 60)
    print("NOTE: This is a stub. Full coach aggregation logic")
    print("      (weighted averages, first-year impact, etc.)")
    print("      can be implemented next.")
    print("=" * 60)
    
    return coach_stats


if __name__ == "__main__":
    calculate_coach_performance()

