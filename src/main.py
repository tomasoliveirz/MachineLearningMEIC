import time
import pandas as pd
from pathlib import Path
from contextlib import redirect_stdout
from datetime import datetime
from player_performance import calculate_player_performance


def main():
    # Define base paths
    base = Path(__file__).resolve().parent.parent
    raw_path = base / 'data' / 'raw' / 'players_teams.csv'
    raw = pd.read_csv(raw_path)

    # Rename columns to match expected format
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

    # Keep only relevant columns
    sample = raw[['bioID', 'year', 'tmID', 'mp', 'pts', 'trb', 'ast', 'stl', 'blk', 'tov']].copy()

    # Calculate player performance using historical data
    res = calculate_player_performance(
        sample,
        seasons_back=3,          # How many past seasons to consider
        decay=0.7,               # How quickly past seasons lose weight
        rookie_min_minutes=100.0,    # Minimum minutes for rookies to avoid heavy shrinkage
        rookie_prior_strength=3600.0 # Equivalent to 3600 minutes of league average performance
    )

    # Set up output folder and file path
    out_dir = base / 'data' / 'processed'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'player_performance.csv'

    # Save the result to a CSV file
    try:
        res.to_csv(out_file, index=False, encoding='utf-8')
        print(f"CSV saved to: {out_file}")
    except Exception as e:
        print(f"Error saving CSV to {out_file}: {e}")


if __name__ == '__main__':
    main()
