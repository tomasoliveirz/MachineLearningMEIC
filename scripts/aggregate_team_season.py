import os
from pathlib import Path
import pandas as pd
import numpy as np

# This script processes and aggregates team-season statistics from the teams_cleaned.csv dataset.
# It computes features such as win percentage, point differential, home/away win rates, and rolling
# averages over previous seasons for each team. The script also calculates season-over-season changes
# and exports the enriched dataset to data_cleaning_output/team_season.csv for further analysis or modeling.

def load_teams_dataframe():
    # prefer cleaned output if available
    cleaned = Path(__file__).resolve().parents[1] / 'data_cleaning_output' / 'teams_cleaned.csv'
    raw = Path(__file__).resolve().parents[1] / 'data' / 'teams.csv'
    if cleaned.exists():
        print(f"Loading cleaned teams from {cleaned}")
        return pd.read_csv(cleaned)
    elif raw.exists():
        print(f"Loading raw teams from {raw}")
        return pd.read_csv(raw)
    else:
        raise FileNotFoundError('No teams CSV found in data_cleaning_output or data/')


def safe_div(a, b):
    result = np.where((b == 0) | (pd.isna(b)), np.nan, a / b)
    return np.round(result, 2)


def aggregate_team_season(df):
    df = df.copy()
    # normalize common columns
    # team identifier: prefer tmID then franchID
    team_col = 'tmID' if 'tmID' in df.columns else ('franchID' if 'franchID' in df.columns else None)
    if team_col is None:
        raise KeyError('Could not find team id column (tmID or franchID)')

    # ensure numeric columns exist
    for col in ['won', 'lost', 'GP', 'o_pts', 'd_pts', 'homeW', 'homeL', 'awayW', 'awayL']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # basic season features
    df['point_diff'] = df.get('o_pts', np.nan) - df.get('d_pts', np.nan)
    df['season_win_pct'] = safe_div(df.get('won', np.nan), df.get('GP', np.nan))
    df['home_win_pct'] = safe_div(df.get('homeW', np.nan), df.get('homeW', 0) + df.get('homeL', 0))
    df['away_win_pct'] = safe_div(df.get('awayW', np.nan), df.get('awayW', 0) + df.get('awayL', 0))

    # sort for rolling calculations
    df = df.sort_values([team_col, 'year'])

    # rolling features across previous seasons (not per-match; dataset lacks per-game rows)
    # compute previous-season and rolling historical features per team in a way
    # that guarantees the resulting Series align with df.index (avoid groupby.apply
    # returning unexpected MultiIndex shapes). We'll loop over groups and fill
    # preallocated Series.
    groups = list(df.groupby(team_col, sort=False).groups.items())

    prev_season_win_pct_1 = pd.Series(index=df.index, dtype=float)
    prev_season_win_pct_3 = pd.Series(index=df.index, dtype=float)
    prev_season_win_pct_5 = pd.Series(index=df.index, dtype=float)

    prev_point_diff_3 = pd.Series(index=df.index, dtype=float)
    prev_point_diff_5 = pd.Series(index=df.index, dtype=float)

    prev_wins_1 = pd.Series(index=df.index, dtype=float) if 'won' in df.columns else None

    for team_val, idx in groups:
        idx = list(idx)
        s_win = df.loc[idx, 'season_win_pct']
        s_point = df.loc[idx, 'point_diff']

        prev_season_win_pct_1.loc[idx] = s_win.shift(1).values
        prev_season_win_pct_3.loc[idx] = s_win.shift(1).rolling(window=3, min_periods=1).mean().values
        prev_season_win_pct_5.loc[idx] = s_win.shift(1).rolling(window=5, min_periods=1).mean().values

        prev_point_diff_3.loc[idx] = s_point.shift(1).rolling(window=3, min_periods=1).mean().values
        prev_point_diff_5.loc[idx] = s_point.shift(1).rolling(window=5, min_periods=1).mean().values

        if prev_wins_1 is not None:
            prev_wins_1.loc[idx] = df.loc[idx, 'won'].shift(1).values

    df['prev_season_win_pct_1'] = prev_season_win_pct_1
    df['prev_season_win_pct_3'] = prev_season_win_pct_3
    df['prev_season_win_pct_5'] = prev_season_win_pct_5

    df['prev_point_diff_3'] = prev_point_diff_3
    df['prev_point_diff_5'] = prev_point_diff_5

    if prev_wins_1 is not None:
        df['prev_wins_1'] = prev_wins_1
    
    # season-over-season changes
    df['win_pct_change_from_prev'] = df['season_win_pct'] - df['prev_season_win_pct_1']

    # select final columns to export
    cols_keep = [
        'year', team_col, 'name', 'season_win_pct', 'won', 'lost', 'GP', 'o_pts', 'd_pts', 'point_diff',
        'home_win_pct', 'away_win_pct', 'prev_season_win_pct_1', 'prev_season_win_pct_3', 'prev_season_win_pct_5',
        'prev_point_diff_3', 'prev_point_diff_5', 'prev_wins_1', 'win_pct_change_from_prev'
    ]

    # columns to round to 2 decimal places
    cols_to_round = [
        'point_diff', 'season_win_pct', 'home_win_pct', 'away_win_pct',
        'prev_season_win_pct_1', 'prev_season_win_pct_3', 'prev_season_win_pct_5',
        'prev_point_diff_3', 'prev_point_diff_5', 'prev_wins_1', 'win_pct_change_from_prev'
    ]
    
    for c in cols_to_round:
        if c in df.columns:
            df[c] = df[c].round(2)

    existing = [c for c in cols_keep if c in df.columns]
    return df.loc[:, existing]


def main():
    out_dir = Path(__file__).resolve().parents[1] / 'data_cleaning_output'
    out_dir.mkdir(parents=True, exist_ok=True)

    df_teams = load_teams_dataframe()
    team_season = aggregate_team_season(df_teams)

    out_path = out_dir / 'team_season.csv'
    team_season.to_csv(out_path, index=False)
    print(f'Wrote aggregated team-season dataset to {out_path} (rows: {len(team_season)})')


if __name__ == '__main__':
    main()
