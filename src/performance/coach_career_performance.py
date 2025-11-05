#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coach Career Performance Analysis (Aggregated)
===============================================
Produces: data/processed/coach_career_performance.csv

Aggregated by coachID with GP-weighted metrics:
- seasons, teams, games
- avg_overach_pythag (GP-weighted)
- avg_overach_roster (GP-weighted)
- eb_career_win_pct
- consistency_sd (std dev of overach_pythag)
- trend (slope of overach_pythag over time)
- career_po_win_pct (GP-weighted playoff win%)
- coy_awards (count)
"""

from pathlib import Path
import pandas as pd
import numpy as np

# Paths
ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)


def aggregate_career(df_season: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate coach_season_performance to career-level metrics
    
    Expected columns: coachID, year, team_id, gp, won, coach_overach_pythag,
                      coach_overach_roster, po_win_pct_coach, is_coy_winner
    """
    # Validate schema
    required = {'coachID', 'year', 'team_id', 'gp', 'won', 'coach_overach_pythag'}
    missing = required - set(df_season.columns)
    if missing:
        raise KeyError(f"Missing columns in coach_season_performance.csv: {missing}")
    
    # Ensure numeric
    for col in ['gp', 'won', 'coach_overach_pythag', 'coach_overach_roster', 
                'po_win_pct_coach', 'is_coy_winner']:
        if col in df_season.columns:
            df_season[col] = pd.to_numeric(df_season[col], errors='coerce')
    
    # Sort by coach and year for trend calculation
    df_season = df_season.sort_values(['coachID', 'year'])
    
    # Compute weighted averages and aggregates per coach
    results = []
    
    for coach_id, group in df_season.groupby('coachID'):
        # Basic counts
        seasons = group['year'].nunique()
        teams = group['team_id'].nunique()
        games = group['gp'].sum()
        total_won = group['won'].sum()
        
        # GP-weighted averages
        total_gp = group['gp'].sum()
        
        if total_gp > 0:
            # Overachievement metrics
            avg_overach_pythag = (
                (group['coach_overach_pythag'] * group['gp']).sum() / total_gp
            )
            avg_overach_roster = (
                (group['coach_overach_roster'] * group['gp']).sum() / total_gp
            )
            
            # Empirical Bayes career win%
            # Use global league mean from season data
            league_mu = df_season['won'].sum() / df_season['gp'].sum()
            alpha = 34  # 1 season
            eb_career_win_pct = (total_won + alpha * league_mu) / (games + alpha)
            
            # Consistency (std dev of seasonal overach)
            consistency_sd = group['coach_overach_pythag'].std()
            if pd.isna(consistency_sd):
                consistency_sd = 0.0
            
            # Trend (slope of overach over seasons)
            # Use season number (not year) for better interpretation
            group_sorted = group.copy()
            group_sorted['season_no'] = range(len(group_sorted))
            
            if len(group_sorted) >= 2:
                valid = group_sorted[['season_no', 'coach_overach_pythag']].dropna()
                if len(valid) >= 2:
                    trend = np.polyfit(valid['season_no'], valid['coach_overach_pythag'], 1)[0]
                else:
                    trend = 0.0
            else:
                trend = 0.0
            
            # Playoff win% (GP-weighted, only for seasons with PO games)
            po_data = group[group['po_win_pct_coach'].notna()].copy()
            if len(po_data) > 0:
                # Reconstruct PO games from win%
                # po_win_pct_coach was computed from post_wins / (post_wins + post_losses)
                # We need to weight by actual PO games, but we don't have that directly
                # Use simple average of po_win_pct for now
                career_po_win_pct = po_data['po_win_pct_coach'].mean()
            else:
                career_po_win_pct = np.nan
            
            # COY awards
            coy_awards = group['is_coy_winner'].sum()
            
        else:
            avg_overach_pythag = np.nan
            avg_overach_roster = np.nan
            eb_career_win_pct = np.nan
            consistency_sd = 0.0
            trend = 0.0
            career_po_win_pct = np.nan
            coy_awards = 0
        
        results.append({
            'coachID': coach_id,
            'seasons': seasons,
            'teams': teams,
            'games': games,
            'avg_overach_pythag': avg_overach_pythag,
            'avg_overach_roster': avg_overach_roster,
            'eb_career_win_pct': eb_career_win_pct,
            'consistency_sd': consistency_sd,
            'trend': trend,
            'career_po_win_pct': career_po_win_pct,
            'coy_awards': int(coy_awards)
        })
    
    df_career = pd.DataFrame(results)
    
    return df_career


def main():
    """Main pipeline"""
    print("\n" + "="*60)
    print("COACH CAREER PERFORMANCE PIPELINE")
    print("="*60)
    
    # Load coach season performance
    print("\n[1/2] Loading coach season performance...")
    season_path = PROC_DIR / "coach_season_performance.csv"
    if not season_path.exists():
        raise FileNotFoundError(
            f"coach_season_performance.csv not found. Run coach_season_performance.py first."
        )
    
    df_season = pd.read_csv(season_path)
    print(f"      Loaded {len(df_season)} coach-season records")
    
    # Aggregate to career
    print("\n[2/2] Aggregating to career metrics...")
    df_career = aggregate_career(df_season)
    print(f"      Aggregated {len(df_career)} coaches")
    
    # Sort by avg overachievement
    df_career = df_career.sort_values('avg_overach_pythag', ascending=False).reset_index(drop=True)
    
    # Save
    out_path = PROC_DIR / "coach_career_performance.csv"
    df_career.to_csv(out_path, index=False)
    
    print(f"\n✓ Saved {len(df_career)} rows to {out_path}")
    print("\n" + "-"*60)
    print("Top 10 coaches by avg_overach_pythag (GP-weighted):")
    top10 = df_career.head(10)[
        ['coachID', 'seasons', 'games', 'avg_overach_pythag', 
         'eb_career_win_pct', 'coy_awards']
    ]
    print(top10.to_string())
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

