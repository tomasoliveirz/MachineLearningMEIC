from pathlib import Path
import pandas as pd
import numpy as np

# define base folders for input and output
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)



def normalize_placeholders(df):
    # replace weird placeholder values with nan or 'unknown'
    replace_map = {
        'None': 'Unknown', '': 'Unknown', 'nan': np.nan, 'NaT': pd.NaT,
        '0-00-0000': pd.NaT, '0000-00-00': pd.NaT,
        '0/00/0000': pd.NaT, '00/00/0000': pd.NaT
    }
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip().replace(replace_map)
        mask = df[col].isin(['None', 'nan', 'NaN', 'NaT'])
        df.loc[mask, col] = None
    return df

def main():
    # load raw teams data and remove duplicates
    df_teams = pd.read_csv(RAW_DIR / 'teams.csv').drop_duplicates()
    
    # clean text placeholders in the dataframe
    df_teams = normalize_placeholders(df_teams)
    
    # list of numeric columns to convert to float
    numeric_cols = [
        'year','o_fgm','o_fga','o_ftm','o_fta','o_3pm','o_3pa','o_oreb','o_dreb','o_reb',
        'o_asts','o_pf','o_stl','o_to','o_blk','o_pts','d_fgm','d_fga','d_ftm','d_fta',
        'd_3pm','d_3pa','d_oreb','d_dreb','d_reb','d_asts','d_pf','d_stl','d_to','d_blk',
        'd_pts','tmORB','tmDRB','tmTRB','opptmORB','opptmDRB','opptmTRB','won','lost',
        'GP','homeW','homeL','awayW','awayL','confW','confL','min','attend'
    ]
    
    # make sure all numeric data is properly typed
    for c in numeric_cols:
        if c in df_teams.columns:
            df_teams[c] = pd.to_numeric(df_teams[c], errors='coerce')
    
    # quick sanity check: won + lost should equal games played
    if all(col in df_teams.columns for col in ['won', 'lost', 'GP']):
        mask = df_teams[['won', 'lost', 'GP']].notna().all(axis=1)
        bad = mask & ((df_teams['won'] + df_teams['lost']) != df_teams['GP'])
        _discrepancies = df_teams.loc[bad, ['year', 'won', 'lost', 'GP']]
    
    # fill some missing categorical fields
    for c in ['divID', 'confID', 'arena']:
        if c in df_teams.columns:
            df_teams[c] = df_teams[c].fillna('Unknown')
    
    # if team name is missing, use franchise id instead
    if 'name' in df_teams.columns:
        df_teams['name'] = df_teams['name'].fillna(df_teams.get('franchID'))
    
    # clean leftover 'None' strings
    for col in df_teams.select_dtypes(include=['object']).columns:
        df_teams[col] = df_teams[col].astype(str).str.strip().replace({'None': None})
    
    # drop numeric columns that are basically all zeros (≥ 98%)
    num_present = [c for c in numeric_cols if c in df_teams.columns]
    to_drop = []
    for c in num_present:
        total = len(df_teams[c])
        if total == 0:
            continue
        zeros = (df_teams[c].fillna(0) == 0).sum()
        if zeros / total >= 0.98:
            to_drop.append(c)
    if to_drop:
        df_teams = df_teams.drop(columns=to_drop)
    
    # drop constant columns (only one unique value)
    const_cols = [c for c in df_teams.columns if df_teams[c].dropna().nunique() <= 1]
    if const_cols:
        df_teams = df_teams.drop(columns=const_cols)
    
    # save cleaned dataset
    df_teams.to_csv(PROC_DIR / 'teams_cleaned.csv', index=False)

if __name__ == '__main__':
    main()
