import os
import pandas as pd
import numpy as np
import warnings

# Carregar o dataset de times
df_teams = pd.read_csv('../data/teams.csv')

# drop exact duplicates
df_teams = df_teams.drop_duplicates()

# normalize common invalid placeholders in all object/string columns
def normalize_placeholders(df):
	replace_map = {
		'': None,
		'nan': None,
		'NaN': None,
		'NaT': None,
		'None': None
	}
	for col in df.select_dtypes(include=['object']).columns:
		# strip whitespace and normalize common placeholder strings to None
		df[col] = df[col].astype(object).astype(str).str.strip().replace(replace_map)
		# convert remaining literal tokens to None
		mask = df[col].isin(['None', 'nan', 'NaN', 'NaT'])
		df.loc[mask, col] = None
	return df

df_teams = normalize_placeholders(df_teams)

# explicit numeric columns based on the dataset header
numeric_cols = [
	'year','o_fgm','o_fga','o_ftm','o_fta','o_3pm','o_3pa','o_oreb','o_dreb','o_reb',
	'o_asts','o_pf','o_stl','o_to','o_blk','o_pts','d_fgm','d_fga','d_ftm','d_fta',
	'd_3pm','d_3pa','d_oreb','d_dreb','d_reb','d_asts','d_pf','d_stl','d_to','d_blk',
	'd_pts','tmORB','tmDRB','tmTRB','opptmORB','opptmDRB','opptmTRB','won','lost',
	'GP','homeW','homeL','awayW','awayL','confW','confL','min','attend'
]

# coerce numeric columns to numeric, invalid parsing -> NaN
for c in numeric_cols:
	if c in df_teams.columns:
		df_teams[c] = pd.to_numeric(df_teams[c], errors='coerce')

# fill sensible defaults for categorical/text columns
if 'divID' in df_teams.columns:
	df_teams['divID'] = df_teams['divID'].fillna('Unknown')
if 'confID' in df_teams.columns:
	df_teams['confID'] = df_teams['confID'].fillna('Unknown')
if 'name' in df_teams.columns:
	df_teams['name'] = df_teams['name'].fillna(df_teams.get('franchID'))
if 'arena' in df_teams.columns:
	df_teams['arena'] = df_teams['arena'].fillna('Unknown')

# trim whitespace again for object columns
for col in df_teams.select_dtypes(include=['object']).columns:
	df_teams[col] = df_teams[col].astype(object).astype(str).str.strip().replace({'None': None})

# --- detect and handle constant / near-constant zero columns ---
# find numeric columns present in the dataframe
numeric_present = [c for c in numeric_cols if c in df_teams.columns]
const_zero = []
near_const = []
for c in numeric_present:
	# proportion of zeros (treat NaN as non-zero here, since NaN != 0)
	total = len(df_teams[c])
	if total == 0:
		continue
	zeros = (df_teams[c].fillna(0) == 0).sum()
	prop_zero = zeros / total
	if prop_zero == 1.0:
		const_zero.append(c)
	elif prop_zero >= 0.98:
		near_const.append(c)

# Decide what to do: drop columns that are always zero. For near-constant columns
# dropping is often OK for modeling, but we keep them if you prefer to convert
# to NaN instead. Here we drop both categories but print what was removed.
to_drop = const_zero + near_const
if to_drop:
	print('Dropping constant/near-constant zero columns:', to_drop)
	df_teams = df_teams.drop(columns=to_drop)

# Drop any column that has no variation (<=1 unique non-null value)
const_cols = []
for col in df_teams.columns:
	unique_nonnull = df_teams[col].dropna().unique()
	if len(unique_nonnull) <= 1:
		const_cols.append(col)

if const_cols:
	print('Dropping columns with no variation (<=1 unique non-null):', const_cols)
	df_teams = df_teams.drop(columns=const_cols)

# ensure output directory exists and save cleaned teams dataset
os.makedirs('../data_cleaning_output', exist_ok=True)
df_teams.to_csv('../data_cleaning_output/teams_cleaned.csv', index=False)

print('Saved cleaned teams to ../data_cleaning_output/teams_cleaned.csv (rows: {})'.format(len(df_teams)))
