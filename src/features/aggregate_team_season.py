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
    """Safely divide a by b.

    Works for scalars and array-like inputs. Returns np.nan when denominator is 0 or either value is NA.
    """
    # Scalars: avoid performing division when denominator is zero
    if np.isscalar(a) and np.isscalar(b):
        if pd.isna(a) or pd.isna(b) or b == 0:
            return np.nan
        try:
            return a / b
        except Exception:
            return np.nan

    # Array-like: use numpy operations without evaluating invalid divisions
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    out = np.full(a_arr.shape, np.nan, dtype=float)
    # mask where division is valid
    valid = (~pd.isna(a_arr)) & (~pd.isna(b_arr)) & (b_arr != 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        out[valid] = a_arr[valid] / b_arr[valid]
    # If inputs were scalars originally, return scalar
    if out.shape == ():
        return float(out)
    return out

def aggregate_team_season(df):
    df = df.copy()
    team_col = 'tmID' if 'tmID' in df.columns else ('franchID' if 'franchID' in df.columns else None)
    if team_col is None:
        raise KeyError('Could not find team id column (tmID or franchID)')

    numeric_cols = [
        'won','lost','GP','o_pts','d_pts','homeW','homeL','awayW','awayL',
        'o_fgm','o_fga','o_3pm','o_3pa','o_ftm','o_fta','o_to','o_stl','o_blk','o_pf',
        'd_fgm','d_fga','d_oreb','d_dreb','o_oreb','o_dreb','attend'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Eficiências e percentuais
    df['point_diff'] = df['o_pts'] - df['d_pts']
    df['season_win_pct'] = safe_div(df['won'], df['GP'])
    df['home_win_pct'] = safe_div(df['homeW'], df['homeW'] + df['homeL'])
    df['away_win_pct'] = safe_div(df['awayW'], df['awayW'] + df['awayL'])
    df['off_eff'] = safe_div(df['o_pts'], df['GP'])
    df['def_eff'] = safe_div(df['d_pts'], df['GP'])
    df['fg_pct'] = safe_div(df['o_fgm'], df['o_fga'])
    df['three_pct'] = safe_div(df['o_3pm'], df['o_3pa'])
    df['ft_pct'] = safe_div(df['o_ftm'], df['o_fta'])
    df['opp_fg_pct'] = safe_div(df['d_fgm'], df['d_fga'])

    # Flag rows that are the team's last season (boolean)
    # If the input already contains `is_last_season`, respect it; otherwise compute it.
    if 'is_last_season' not in df.columns:
        last_years = df.groupby(team_col)['year'].max()
        df['is_last_season'] = df['year'] == df[team_col].map(last_years.to_dict())
        df['is_last_season'] = df['is_last_season'].astype(bool)

    # Estilo e disciplina
    df['prop_3pt_shots'] = safe_div(df['o_3pa'], df['o_fga'])
    df['o_to_pg'] = safe_div(df['o_to'], df['GP'])
    df['o_stl_pg'] = safe_div(df['o_stl'], df['GP'])
    df['o_blk_pg'] = safe_div(df['o_blk'], df['GP'])
    df['o_pf_pg'] = safe_div(df['o_pf'], df['GP'])

    # Rebounds e controle de posse
    df['reb_diff'] = (df.get('o_oreb',0) + df.get('o_dreb',0)) - (df.get('d_oreb',0) + df.get('d_dreb',0))
    df['stl_diff'] = df.get('o_stl',0) - df.get('d_stl',0) if 'd_stl' in df else np.nan
    df['blk_diff'] = df.get('o_blk',0) - df.get('d_blk',0) if 'd_blk' in df else np.nan
    df['to_diff'] = df.get('o_to',0) - df.get('d_to',0) if 'd_to' in df else np.nan

    # Contexto
    df['attend_pg'] = safe_div(df['attend'], df['GP'])
    df['home_advantage'] = df['home_win_pct'] - df['away_win_pct']
    if 'franchID' in df.columns and 'tmID' in df.columns:
        df['franchise_changed'] = df['franchID'] != df['tmID']
    else:
        df['franchise_changed'] = False

    df['season_id'] = df['year'].astype(str) + "_" + df[team_col]

    # Tendências históricas
    df = df.sort_values([team_col, 'year'])
    grp = df.groupby(team_col, sort=False)
    roll = lambda s, w: s.shift(1).rolling(w, min_periods=1).mean().values

    df['prev_win_pct_1'] = grp['season_win_pct'].shift(1)
    df['prev_win_pct_3'] = grp['season_win_pct'].apply(lambda s: roll(s,3))
    df['prev_win_pct_5'] = grp['season_win_pct'].apply(lambda s: roll(s,5))
    df['prev_point_diff_3'] = grp['point_diff'].apply(lambda s: roll(s,3))
    df['prev_point_diff_5'] = grp['point_diff'].apply(lambda s: roll(s,5))
    df['win_pct_change'] = df['season_win_pct'] - df['prev_win_pct_1']

    # Normalização por média da liga
    league_cols = ['off_eff','def_eff','fg_pct','three_pct','ft_pct','opp_fg_pct','point_diff']
    year_means = df.groupby('year')[league_cols].mean().to_dict(orient='dict')
    for c in league_cols:
        df[f"{c}_norm"] = df.apply(lambda r: safe_div(r[c], year_means[c].get(r['year'], np.nan)), axis=1)

    # Seleção final: incluir todas as colunas originais + as colunas derivadas
    original_cols = [
        'year','tmID','franchID','confID','rank','playoff','firstRound','semis','finals','name',
        'o_fgm','o_fga','o_ftm','o_fta','o_3pm','o_3pa','o_oreb','o_dreb','o_reb','o_asts','o_pf','o_stl','o_to','o_blk','o_pts',
        'd_fgm','d_fga','d_ftm','d_fta','d_3pm','d_3pa','d_oreb','d_dreb','d_reb','d_asts','d_pf','d_stl','d_to','d_blk','d_pts',
        'won','lost','GP','homeW','homeL','awayW','awayL','confW','confL','min','attend','arena'
    ]
    # keep is_last_season if present (computed above)
    if 'is_last_season' in df.columns and 'is_last_season' not in original_cols:
        original_cols.append('is_last_season')

    derived_cols = [
        'season_id','season_win_pct','point_diff','off_eff','def_eff',
        'fg_pct','three_pct','ft_pct','opp_fg_pct','prop_3pt_shots',
        'home_win_pct','away_win_pct','home_advantage','reb_diff','stl_diff','blk_diff','to_diff',
        'attend_pg','franchise_changed',
        'prev_win_pct_1','prev_win_pct_3','prev_win_pct_5','prev_point_diff_3','prev_point_diff_5','win_pct_change'
    ] + [f"{c}_norm" for c in league_cols]

    # Compose final column list: original existing columns first, then derived
    cols_keep = [c for c in original_cols if c in df.columns] + [c for c in derived_cols if c in df.columns]

    df = df.loc[:, cols_keep]

    # Round numeric columns only (detect numeric dtypes)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude any integer identifier-like columns from aggressive rounding if desired (keep as-is)
    df[numeric_cols] = df[numeric_cols].round(4)

    return df

def main():
    df_teams = load_teams_dataframe()
    df_stats = aggregate_team_season(df_teams)
    out_path = PROC_DIR / "team_season_statistics.csv"
    df_stats.to_csv(out_path, index=False)

if __name__ == "__main__":
    main()
