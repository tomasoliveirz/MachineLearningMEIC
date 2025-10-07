import time
import pandas as pd
from pathlib import Path
from player_perfomance import (
    calculate_player_performance,
    calculate_rookies_perfomance_per_team_previous_seasons,
    is_rookie,
)


def main():
    base = Path(__file__).resolve().parent.parent.parent
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


    # detect rookies before performance calculation
    rookie_mask_before = is_rookie(sample)
    n_rookies_before = int(rookie_mask_before.sum())

    # compute performance (uses seasons in the DataFrame)
    start = time.time()
    res = calculate_player_performance(sample, seasons_back=3, decay=0.7)
    duration = time.time() - start

    perf = res['performance']

    # Basic prints
    print("=== Player performance (amostra) ===")
    print(f"Origem: {raw_path}")
    print(f"Sample size: {len(sample)}  |  Execution time: {duration:.3f}s")
    print()

    # Overall performance summary
    print("Performance summary (all rows):")
    print(perf.describe(percentiles=[0.25, 0.5, 0.75]).to_string())
    print()

    # Rookies info
    print("Rookies (identificados antes do cálculo):")
    print(f"Rookies identificados: {n_rookies_before}")
    if n_rookies_before > 0:
        rookies_df = res[rookie_mask_before]
        print(f"Rookies - performance mean: {rookies_df['performance'].mean():.3f}  |  std: {rookies_df['performance'].std():.3f}")
        print("Top 5 rookies (por performance):")
        cols_rookie = ['bioID', 'year', 'tmID', 'mp', 'pts', 'performance', 'rookie']
        existing_cols = [c for c in cols_rookie if c in rookies_df.columns]
        print(rookies_df[existing_cols].sort_values('performance', ascending=False).head(5).to_string(index=False))
    print()

    # Top / bottom performers
    print("Top 10 players (por performance):")
    cols_top = ['bioID', 'year', 'tmID', 'mp', 'pts', 'performance', 'rookie']
    existing_top = [c for c in cols_top if c in res.columns]
    print(res[existing_top].sort_values('performance', ascending=False).head(10).to_string(index=False))
    print()
    print("Bottom 10 players (por performance):")
    cols_bot = ['bioID', 'year', 'tmID', 'mp', 'pts', 'performance', 'rookie']
    existing_bot = [c for c in cols_bot if c in res.columns]
    print(res[existing_bot].sort_values('performance', ascending=True).head(10).to_string(index=False))
    print()

    # By year summary
    if 'year' in res.columns:
        print("Performance por year (mean, count):")
        by_year = res.groupby('year')['performance'].agg(['mean', 'count']).sort_index()
        print(by_year.to_string())
        print()

    # By team summary (top teams by avg performance)
    if 'tmID' in res.columns:
        print("Resumo por time (top 10 por avg performance):")
        by_team = res.groupby('tmID')['performance'].agg(['mean', 'count']).sort_values('mean', ascending=False)
        print(by_team.head(10).to_string())
        print()


    # Final short table
    print("Resumo final (primeiras 10 linhas):")
    cols_final = ['bioID', 'year', 'tmID', 'mp', 'pts', 'performance', 'rookie']
    existing_final = [c for c in cols_final if c in res.columns]
    print(res[existing_final].head(10).to_string(index=False))
    print(f"\nTotal de linhas processadas: {len(res)}\n")

if __name__ == '__main__':
    main()
