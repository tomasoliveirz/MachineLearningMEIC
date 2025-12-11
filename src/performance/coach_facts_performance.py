"""
Coach Season Performance Analysis (Stint-aware)
================================================
Produces: data/processed/coach_season_facts_performance.csv

Columns per coach-team-season (respecting stint):
- coachID, team_id, year, stint, gp, won, lost, rs_win_pct_coach
- eb_rs_win_pct (Empirical Bayes smoothing)
- is_first_year_with_team
- po_win_pct_coach (if post_wins/post_losses exist)
"""

from pathlib import Path
import pandas as pd
import numpy as np

# Paths
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)


def load_coaches() -> pd.DataFrame:
    """
    Load and process coaches data
    
    Expected columns: coachID, tmID, year, stint, won, lost, post_wins, post_losses
    """
    path = RAW_DIR / "coaches.csv"
    df = pd.read_csv(path)
    
    # Validate schema
    required = {'coachID', 'tmID', 'year', 'stint', 'won', 'lost'}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in coaches.csv: {missing}")
    
    df['team_id'] = df['tmID']  # Convert to canonical name
    
    # Ensure numeric
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['stint'] = pd.to_numeric(df['stint'], errors='coerce')
    df['won'] = pd.to_numeric(df['won'], errors='coerce')
    df['lost'] = pd.to_numeric(df['lost'], errors='coerce')
    
    # Compute gp and rs_win_pct_coach
    df['gp'] = df['won'] + df['lost']
    df['rs_win_pct_coach'] = df['won'] / df['gp'].replace(0, np.nan)
    
    # Playoff win% (optional columns)
    if 'post_wins' in df.columns and 'post_losses' in df.columns:
        df['post_wins'] = pd.to_numeric(df['post_wins'], errors='coerce')
        df['post_losses'] = pd.to_numeric(df['post_losses'], errors='coerce')
        po_total = df['post_wins'] + df['post_losses']
        df['po_win_pct_coach'] = df['post_wins'] / po_total.replace(0, np.nan)
        df['po_win_pct_coach'] = df['po_win_pct_coach'].clip(0, 1)
    else:
        df['po_win_pct_coach'] = np.nan
    
    return df


def merge_team_baselines(df_coaches: pd.DataFrame, df_team_perf: pd.DataFrame) -> pd.DataFrame:
    """
    Merge coach data with team performance baselines
    (pythag_win_pct, rs_win_pct_expected_roster, rs_win_pct_prev)
    
    These are FACTS about the team's season (not predictive features).
    """
    df = df_coaches.merge(
        df_team_perf[['team_id', 'year', 'pythag_win_pct', 
                      'rs_win_pct_expected_roster', 'rs_win_pct_prev', 'team_strength']],
        on=['team_id', 'year'],
        how='left'
    )
    
    return df


def compute_coach_season_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute coach-specific metrics:
    - Empirical Bayes smoothing
    - First year flags (pre-season fact)
    
    REMOVED: overachievement metrics (will be computed in model)
    REMOVED: delta_vs_prev_team (leakage)
    """
    # Empirical Bayes smoothing
    # alpha = 34 games (1 WNBA season)
    # league_mu = overall mean win%
    total_won = df['won'].sum()
    total_gp = df['gp'].sum()
    league_mu = total_won / total_gp if total_gp > 0 else 0.5
    alpha = 34
    
    df['eb_rs_win_pct'] = (df['won'] + alpha * league_mu) / (df['gp'] + alpha)
    df['eb_rs_win_pct'] = df['eb_rs_win_pct'].clip(0, 1)
    
    print(f"[Coach Season] League mean win% = {league_mu:.3f}, EB alpha = {alpha}")
    
    # First year with team flag (pre-season fact)
    df = df.sort_values(['coachID', 'team_id', 'year'])
    
    # Check if this coach-team combo exists in previous year
    df['coach_team'] = df['coachID'].astype(str) + "_" + df['team_id'].astype(str)
    df['prev_year'] = df['year'] - 1
    
    prev_combos = df[['coachID', 'team_id', 'year']].copy()
    prev_combos = prev_combos.rename(columns={'year': 'prev_year'})
    prev_combos['had_prev_year'] = 1
    
    df = df.merge(
        prev_combos[['coachID', 'team_id', 'prev_year', 'had_prev_year']],
        on=['coachID', 'team_id', 'prev_year'],
        how='left'
    )
    
    df['is_first_year_with_team'] = df['had_prev_year'].isna().astype(int)
    
    # Clean up
    df = df.drop(columns=['coach_team', 'prev_year', 'had_prev_year'], errors='ignore')
    
    return df


def main():
    """Main pipeline"""
    print("\n" + "="*60)
    print("COACH SEASON PERFORMANCE PIPELINE")
    print("MODE: RAW FACTS ONLY (no predictive features)")
    print("="*60)
    
    # 1. Load coaches
    print("\n[1/3] Loading coaches data...")
    df = load_coaches()
    print(f"      Loaded {len(df)} coach-season stints")
    
    # 2. Load team performance baselines
    print("\n[2/3] Merging with team performance baselines...")
    team_perf_path = PROC_DIR / "team_performance.csv"
    if not team_perf_path.exists():
        raise FileNotFoundError(
            f"team_performance.csv not found. Run team_performance.py first."
        )
    
    df_team = pd.read_csv(team_perf_path)
    df = merge_team_baselines(df, df_team)
    
    # 3. Compute coach metrics (RAW FACTS ONLY)
    print("\n[3/3] Computing coach season metrics...")
    df = compute_coach_season_metrics(df)
    
    # Select canonical columns (RAW FACTS ONLY)
    canonical_cols = [
        'coachID', 'team_id', 'year', 'stint', 'gp', 'won', 'lost',
        'rs_win_pct_coach', 'eb_rs_win_pct',
        'is_first_year_with_team',
        'po_win_pct_coach',
        # Keep baseline references for model to use
        'pythag_win_pct', 'rs_win_pct_expected_roster', 'rs_win_pct_prev', 'team_strength'
    ]
    
    cols_to_keep = [c for c in canonical_cols if c in df.columns]
    df_out = df[cols_to_keep].copy()
    
    # Sort
    df_out = df_out.sort_values(['coachID', 'year', 'team_id', 'stint']).reset_index(drop=True)
    
    # Save
    out_path = PROC_DIR / "coach_season_facts_performance.csv"
    df_out.to_csv(out_path, index=False)
    
    print(f"\n✓ Saved {len(df_out)} rows to {out_path}")
    print("\n" + "-"*60)
    print("Sample (top coaches by win%):")
    sample = df_out.nlargest(5, 'rs_win_pct_coach')[
        ['coachID', 'team_id', 'year', 'gp', 'rs_win_pct_coach', 'is_first_year_with_team']
    ]
    print(sample.to_string())
    print("="*60 + "\n")


if __name__ == "__main__":
    main()