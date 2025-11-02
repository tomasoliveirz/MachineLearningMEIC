from pathlib import Path
import pandas as pd
import shutil

DATA_DIR = Path(__file__).parent.parent.parent / 'data' / 'processed'
CSV_PATH = DATA_DIR / 'teams_cleaned.csv'

if not CSV_PATH.exists():
    raise FileNotFoundError(f"Could not find {CSV_PATH}")

# Load
df = pd.read_csv(CSV_PATH)

# Determine grouping column (prefer tmID)
team_col = 'tmID' if 'tmID' in df.columns else ('franchID' if 'franchID' in df.columns else None)
if team_col is None:
    raise KeyError('No tmID or franchID column found')

# Compute last year per team
last_years = df.groupby(team_col)['year'].max().rename('last_year')

# Create a boolean Series `is_last_season` which is True when the row's year
# equals the team's last year. Keep it as a separate variable for ease of use
# and also store it back into the dataframe as `df['is_last_season']`.
is_last_season = df['year'] == df[team_col].map(last_years.to_dict())
df['is_last_season'] = is_last_season

# Ensure dtype is boolean and write back to the same CSV path
df['is_last_season'] = df['is_last_season'].astype(bool)

# Write updated dataframe back to CSV (overwrite)
df.to_csv(CSV_PATH, index=False)

