#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

def load_teams_dataframe():
    cleaned = PROC_DIR / 'teams_cleaned.csv'
    raw = RAW_DIR / 'teams.csv'
    if cleaned.exists():
        return pd.read_csv(cleaned)
    elif raw.exists():
        return pd.read_csv(raw)
    else:
        raise FileNotFoundError('No teams CSV found in data/processed or data/raw')

def safe_div(a, b):
    b = b.copy()
    return np.where((b == 0) | (pd.isna(b)), np.nan, a / b)

def aggregate_team_season(df):
    df = df.copy()
    team_col = 'tmID' if 'tmID' in df.columns else ('franchID' if 'franchID' in df.columns else None)
    if team_col is None:
        raise KeyError('Could not find team id column (tmID or franchID)')

    for col in ['won','lost','GP','o_pts','d_pts','homeW','homeL','awayW','awayL']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['point_diff'] = df.get('o_pts', np.nan) - df.get('d_pts', np.nan)
    df['season_win_pct'] = safe_div(df.get('won', np.nan), df.get('GP', np.nan))
    df['home_win_pct'] = safe_div(df.get('homeW', np.nan), df.get('homeW', 0) + df.get('homeL', 0))
    df['away_win_pct'] = safe_div(df.get('awayW', np.nan), df.get('awayW', 0) + df.get('awayL', 0))

    df = df.sort_values([team_col, 'year'])
    def roll_mean(s, w): 
        return s.shift(1).rolling(w, min_periods=1).mean().values

    grp = df.groupby(team_col, sort=False)
    df['prev_season_win_pct_1'] = grp['season_win_pct'].shift(1)
    df['prev_season_win_pct_3'] = grp['season_win_pct'].apply(lambda s: roll_mean(s,3))
    df['prev_season_win_pct_5'] = grp['season_win_pct'].apply(lambda s: roll_mean(s,5))
    df['prev_point_diff_3'] = grp['point_diff'].apply(lambda s: roll_mean(s,3))
    df['prev_point_diff_5'] = grp['point_diff'].apply(lambda s: roll_mean(s,5))
    if 'won' in df.columns:
        df['prev_wins_1'] = grp['won'].shift(1)

    df['win_pct_change_from_prev'] = df['season_win_pct'] - df['prev_season_win_pct_1']

    cols_keep = [
        'year', team_col, 'name', 'season_win_pct', 'won', 'lost', 'GP', 'o_pts', 'd_pts', 'point_diff',
        'home_win_pct', 'away_win_pct', 'prev_season_win_pct_1', 'prev_season_win_pct_3', 'prev_season_win_pct_5',
        'prev_point_diff_3', 'prev_point_diff_5', 'prev_wins_1', 'win_pct_change_from_prev'
    ]
    existing = [c for c in cols_keep if c in df.columns]
    num_cols = [c for c in existing if c not in [team_col,'name','year']]
    df[num_cols] = df[num_cols].astype(float).round(2)
    return df.loc[:, existing]

def main():
    df_teams = load_teams_dataframe()
    team_season = aggregate_team_season(df_teams)
    out_path = PROC_DIR / 'team_season.csv'
    team_season.to_csv(out_path, index=False)

if __name__ == '__main__':
    main()
