#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
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

    # Garantir conversão numérica das colunas relevantes
    numeric_cols = ['won','lost','GP','o_pts','d_pts','homeW','homeL','awayW','awayL',
                    'o_fgm','o_fga','o_3pm','o_3pa','o_ftm','o_fta',
                    'd_fgm','d_fga','o_to','o_stl','o_blk','o_pf','attend']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['point_diff'] = df.get('o_pts', np.nan) - df.get('d_pts', np.nan)
    df['season_win_pct'] = safe_div(df.get('won', np.nan), df.get('GP', np.nan))
    df['home_win_pct'] = safe_div(df.get('homeW', np.nan), df.get('homeW', 0) + df.get('homeL', 0))
    df['away_win_pct'] = safe_div(df.get('awayW', np.nan), df.get('awayW', 0) + df.get('awayL', 0))

    # Eficiências
    df['off_eff'] = safe_div(df.get('o_pts', np.nan), df.get('GP', np.nan))
    df['def_eff'] = safe_div(df.get('d_pts', np.nan), df.get('GP', np.nan))

    # Percentuais de arremesso
    df['fg_pct'] = safe_div(df.get('o_fgm', np.nan), df.get('o_fga', np.nan))
    df['three_pct'] = safe_div(df.get('o_3pm', np.nan), df.get('o_3pa', np.nan))
    df['ft_pct'] = safe_div(df.get('o_ftm', np.nan), df.get('o_fta', np.nan))
    df['opp_fg_pct'] = safe_div(df.get('d_fgm', np.nan), df.get('d_fga', np.nan))

    # Estilo de jogo
    df['prop_3pt_shots'] = safe_div(df.get('o_3pa', np.nan), df.get('o_fga', np.nan))
    df['o_to_per_game'] = safe_div(df.get('o_to', np.nan), df.get('GP', np.nan))
    df['o_stl_per_game'] = safe_div(df.get('o_stl', np.nan), df.get('GP', np.nan))
    df['o_blk_per_game'] = safe_div(df.get('o_blk', np.nan), df.get('GP', np.nan))
    df['o_pf_per_game'] = safe_div(df.get('o_pf', np.nan), df.get('GP', np.nan))

    # Fatores contextuais
    df['attend_per_game'] = safe_div(df.get('attend', np.nan), df.get('GP', np.nan))
    df['home_advantage'] = df['home_win_pct'] - df['away_win_pct']
    # manter confID se existir
    if 'confID' in df.columns:
        df['confID'] = df['confID']
    # flag de mudança de franquia (quando franchID existe e é diferente de tmID)
    if 'franchID' in df.columns and 'tmID' in df.columns:
        df['franchise_changed'] = df['franchID'] != df['tmID']
    else:
        df['franchise_changed'] = False

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

    # Calcular médias anuais da liga para normalizar métricas por época
    league_avg_cols = ['off_eff', 'def_eff', 'fg_pct', 'three_pct', 'ft_pct', 'opp_fg_pct', 'point_diff']
    present_avg_cols = [c for c in league_avg_cols if c in df.columns]
    if present_avg_cols:
        # média por ano
        year_means = df.groupby('year')[present_avg_cols].transform('mean')
        for c in present_avg_cols:
            norm_col = f"{c}_norm"
            # normalização simples: divisão pela média da liga no mesmo ano
            df[norm_col] = safe_div(df[c], year_means[c])

    cols_keep = [
        'year', team_col, 'name', 'season_win_pct', 'won', 'lost', 'GP', 'o_pts', 'd_pts', 'point_diff',
        'off_eff', 'def_eff', 'fg_pct', 'three_pct', 'ft_pct', 'opp_fg_pct', 'prop_3pt_shots',
        'home_win_pct', 'away_win_pct', 'home_advantage', 'attend_per_game', 'confID', 'franchise_changed',
        'prev_season_win_pct_1', 'prev_season_win_pct_3', 'prev_season_win_pct_5',
        'prev_point_diff_3', 'prev_point_diff_5', 'prev_wins_1', 'win_pct_change_from_prev'
    ]
    # adicionar as colunas normalizadas se existirem
    norm_cols = [c + '_norm' for c in ['off_eff', 'def_eff', 'fg_pct', 'three_pct', 'ft_pct', 'opp_fg_pct', 'point_diff']]
    for nc in norm_cols:
        cols_keep.append(nc)
    existing = [c for c in cols_keep if c in df.columns]
    # aplicar formatação numérica nas colunas relevantes
    num_cols = [c for c in existing if c not in [team_col,'name','year','confID']]
    df[num_cols] = df[num_cols].astype(float).round(4)
    # manter uma ordem de colunas estável
    return df.loc[:, existing]

def main():
    df_teams = load_teams_dataframe()
    team_season = aggregate_team_season(df_teams)
    out_path = PROC_DIR / 'team_season_statistics.csv'
    team_season.to_csv(out_path, index=False)

if __name__ == '__main__':
    main()
