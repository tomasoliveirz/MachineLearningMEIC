import time
import pandas as pd
from pathlib import Path
from contextlib import redirect_stdout
from datetime import datetime
from player_performance import calculate_player_performance


def main():
    
    base = Path(__file__).resolve().parent.parent
    raw_path = base / 'data' / 'raw' / 'players_teams.csv'
    raw = pd.read_csv(raw_path)

    # normalize columns to match our function expectations
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

    sample = raw[['bioID', 'year', 'tmID', 'mp', 'pts', 'trb', 'ast', 'stl', 'blk', 'tov']].copy()


    # compute performance (uses seasons in the DataFrame)
    res = calculate_player_performance(
        sample, 
        seasons_back=3, 
        decay=0.7,
        rookie_min_minutes=100.0,  # Threshold: rookies need at least 100 minutes to avoid heavy shrinkage
        rookie_prior_strength=3600.0  # Prior strength: equivalent to 3600 minutes of average performance
    )


    # --- Save enhanced player performance CSV to requested folder ---
    out_dir = base / 'data' / 'processed' 
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'player_perfomance.csv'

    # Save full result DataFrame (includes performance and rookie)
    try:
        res.to_csv(out_file, index=False, encoding='utf-8')
        print(f"\n✅ CSV salvo em: {out_file}")
    except Exception as e:
        print(f"Erro ao salvar CSV em {out_file}: {e}")

if __name__ == '__main__':
    main()
