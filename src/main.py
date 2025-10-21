import time
import pandas as pd
from pathlib import Path
from contextlib import redirect_stdout
from datetime import datetime
from player_performance import calculate_player_performance


def main():
    # define base paths
    base = Path(__file__).resolve().parent.parent
    raw_path = base / 'data' / 'raw' / 'players_teams.csv'
    raw = pd.read_csv(raw_path)

    # rename columns to match what calculate_player_performance expects
    raw.rename(columns={
        'playerID': 'bioID',
        'minutes': 'mp',
        'points': 'pts',
        'rebounds': 'trb',
        'assists': 'ast',
        'steals': 'stl',
        'blocks': 'blk',
        'turnovers': 'tov',
        'year': 'year',
        'tmID': 'tmID'
    }, inplace=True)

    # keep only the relevant columns
    sample = raw[['bioID', 'year', 'tmID', 'mp', 'pts', 'trb', 'ast', 'stl', 'blk', 'tov']].copy()

    # compute player performance using historical data
    res = calculate_player_performance(
        sample,
        seasons_back=3,          # how many past seasons to include
        decay=0.7,               # how fast old seasons lose importance
        rookie_min_minutes=100.0,    # rookies below this playtime get shrinkage
        rookie_prior_strength=3600.0 # prior strength (like 3600 “virtual” minutes of league average)
    )

    # set up output folder and file path
    out_dir = base / 'data' / 'processed'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'player_perfomance.csv'

    # save the result to csv
    try:
        res.to_csv(out_file, index=False, encoding='utf-8')
        print(f"\n✅ csv saved to: {out_file}")
    except Exception as e:
        print(f"error saving csv to {out_file}: {e}")


if __name__ == '__main__':
    main()
