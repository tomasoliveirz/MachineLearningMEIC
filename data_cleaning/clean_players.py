import pandas as pd
import numpy as np

# load players.csv
df_players = pd.read_csv('../data/players.csv')

# clean duplicates
df_players = df_players.drop_duplicates()

# remove if player only has null values (0 or 0-00-0000 date)
df_players = df_players.dropna(how='all', subset=['pos','firstSeason','lastSeason','height','weight','college','birthDate'])

# unknown positions - replace empty string with 'Unknown'
df_players['pos'] = df_players['pos'].fillna('Unknown')

# firstseason and lastseason - replace 0 with NaN
df_players['firstSeason'] = df_players['firstSeason'].replace(0, np.nan)
df_players['lastSeason'] = df_players['lastSeason'].replace(0, np.nan)

# replace value 0 cases in height and weight with NaN
df_players['height'] = df_players['height'].replace(0, np.nan)
df_players['weight'] = df_players['weight'].replace(0, np.nan)

# if college is missing but collegeOther is present, use collegeOther value for college
df_players['college'] = df_players['college'].fillna(df_players['collegeOther'])
# college - replace empty string with 'Unknown'
df_players['college'] = df_players['college'].fillna('Unknown')
# collegeOther - empty string with 'Unknown'
df_players['collegeOther'] = df_players['collegeOther'].fillna('Unknown')

# birthDate and deathDate - convert to datetime, replace invalid dates with NaT
df_players['birthDate'] = pd.to_datetime(df_players['birthDate'], errors='coerce')
df_players['deathDate'] = pd.to_datetime(df_players['deathDate'], errors='coerce')

# save cleaned players dataset
df_players.to_csv('../data_cleaning_output/players_cleaned.csv', index=False)