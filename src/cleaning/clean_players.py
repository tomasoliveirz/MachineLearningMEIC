#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np
import warnings

# define paths for raw and processed data
ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# load the raw players file
df_players = pd.read_csv(RAW_DIR / "players.csv")

# remove duplicate rows just in case
df_players = df_players.drop_duplicates()

# replace blank strings with nan values
df_players = df_players.replace(r'^\s*$', np.nan, regex=True)

# columns that should be numeric
numeric_cols = ['firstseason', 'lastseason', 'height', 'weight']

# force numeric conversion (invalid values become nan)
for c in numeric_cols:
    if c in df_players.columns:
        df_players[c] = pd.to_numeric(df_players[c], errors='coerce')

# replace zeros in numeric columns with nan (zero usually means missing)
df_players[numeric_cols] = df_players[numeric_cols].replace(0, np.nan)

# drop players that have no valid data in the key columns
df_players = df_players.dropna(
    how='all',
    subset=['pos','firstseason','lastseason','height','weight','college','birthDate']
)

# fill missing positions with a generic label
df_players['pos'] = df_players['pos'].fillna('Unknown')

# clean up bad or zero season values
df_players['firstseason'] = df_players['firstseason'].replace(0, np.nan)
df_players['lastseason'] = df_players['lastseason'].replace(0, np.nan)

# remove unrealistic height or weight values
df_players['height'] = df_players['height'].replace(0, np.nan)
df_players['weight'] = df_players['weight'].replace(0, np.nan)
df_players['weight'] = df_players['weight'].where(df_players['weight'] > 60, np.nan)
df_players['height'] = df_players['height'].where(df_players['height'] > 24, np.nan)

# merge college and collegeOther fields
df_players['college'] = df_players['college'].fillna(df_players.get('collegeOther'))
df_players['college'] = df_players['college'].fillna('Unknown')
if 'collegeOther' in df_players.columns:
    df_players['collegeOther'] = df_players['collegeOther'].fillna('Unknown')

# clean up date columns and replace bad placeholders
for col in ['birthDate', 'deathDate']:
    df_players[col] = df_players[col].where(df_players[col].notna(), None)
    df_players[col] = (
        df_players[col]
        .astype(str)
        .str.strip()
        .replace({
            'None': 'Unknown', '': 'Unknown', 'nan': np.nan, 'NaT': pd.NaT,
            '0-00-0000': pd.NaT, '0000-00-00': pd.NaT,
            '0/00/0000': pd.NaT, '00/00/0000': pd.NaT
        })
    )

# helper to safely parse multiple date formats
def _parse_dates(series):
    s = series.copy()
    parsed = pd.to_datetime(s, format='%Y-%m-%d', errors='coerce')
    parsed = parsed.fillna(pd.to_datetime(s, format='%d/%m/%Y', errors='coerce'))
    parsed = parsed.fillna(pd.to_datetime(s, format='%m/%d/%Y', errors='coerce'))
    if parsed.isna().any():
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Could not infer format')
            parsed = parsed.fillna(pd.to_datetime(s, errors='coerce'))
    return parsed

# parse both birth and death dates properly
df_players['birthDate'] = _parse_dates(df_players['birthDate'])
df_players['deathDate'] = _parse_dates(df_players['deathDate'])

# export cleaned dataset
df_players.to_csv(PROC_DIR / 'players_cleaned.csv', index=False, na_rep='NaN')
