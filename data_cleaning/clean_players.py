import os
import pandas as pd
import numpy as np
import warnings

# load players.csv
df_players = pd.read_csv('../data/players.csv')

# clean duplicates
df_players = df_players.drop_duplicates()

df_players = df_players.replace(r'^\s*$', np.nan, regex=True)

# treat 0 as NaN for numeric columns before dropna 
numeric_cols = ['firstseason', 'lastseason', 'height', 'weight']
df_players[numeric_cols] = df_players[numeric_cols].replace(0, np.nan)

# remove if player only has null values (0 or 0-00-0000 date)
df_players = df_players.dropna(how='all', subset=['pos','firstseason','lastseason','height','weight','college','birthDate'])

# unknown positions - replace empty string with 'Unknown'
df_players['pos'] = df_players['pos'].fillna('Unknown')

# firstseason and lastseason - replace 0 with NaN
df_players['firstseason'] = df_players['firstseason'].replace(0, np.nan)
df_players['lastseason'] = df_players['lastseason'].replace(0, np.nan)

# replace value 0 cases in height and weight with NaN
df_players['height'] = df_players['height'].replace(0, np.nan)
df_players['weight'] = df_players['weight'].replace(0, np.nan)

# set weights below or equal to 50 pounds to NaN as unrealistic
df_players['weight'] = df_players['weight'].where(df_players['weight'] > 60, np.nan)

# set heights below or equal to 24 inches to NaN as unrealistic
df_players['height'] = df_players['height'].where(df_players['height'] > 24, np.nan)

# if college is missing but collegeOther is present, use collegeOther value for college
df_players['college'] = df_players['college'].fillna(df_players['collegeOther'])
# college - replace empty string with 'Unknown'
df_players['college'] = df_players['college'].fillna('Unknown')
# collegeOther - empty string with 'Unknown'
df_players['collegeOther'] = df_players['collegeOther'].fillna('Unknown')

# birthDate and deathDate - convert to datetime, replace invalid dates with NaT
# normalize common invalid placeholders and strip whitespace
for col in ['birthDate', 'deathDate']:
	# keep NaNs as-is, convert others to string for normalization
	df_players[col] = df_players[col].where(df_players[col].notna(), None)
	df_players[col] = df_players[col].astype(object).astype(str).str.strip().replace({
        'None': "Unknown",
        '': "Unknown",
        'nan': np.nan,
        'NaT': np.nan,
        '0-00-0000': np.nan,
        '0000-00-00': np.nan,
        '0/00/0000': np.nan,
        '00/00/0000': np.nan
	})

def _parse_dates(series):
	s = series.copy()
	# try explicit common formats first to avoid inference warnings
	parsed = pd.to_datetime(s, format='%Y-%m-%d', errors='coerce')
	parsed = parsed.fillna(pd.to_datetime(s, format='%d/%m/%Y', errors='coerce'))
	parsed = parsed.fillna(pd.to_datetime(s, format='%m/%d/%Y', errors='coerce'))
	# final fallback using dateutil; suppress the specific infer-format warning
	if parsed.isna().any():
		with warnings.catch_warnings():
			# suppress pandas warning about parsing with dayfirst when formats were tried
			warnings.filterwarnings('ignore', message='Could not infer format')
			warnings.filterwarnings('ignore', message="Parsing dates in %Y-%m-%d format when dayfirst=True was specified")
			parsed = parsed.fillna(pd.to_datetime(s, errors='coerce'))
	return parsed

df_players['birthDate'] = _parse_dates(df_players['birthDate'])
df_players['deathDate'] = _parse_dates(df_players['deathDate'])

os.makedirs('../data_cleaning/data_cleaning_output', exist_ok=True)

# save cleaned players dataset, write NaNs explicitly as 'NaN'
df_players.to_csv('../data_cleaning/data_cleaning_output/players_cleaned.csv', index=False, na_rep='NaN')