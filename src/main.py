import time
import pandas as pd
from pathlib import Path
from contextlib import redirect_stdout
from datetime import datetime
from player_performance import calculate_player_performance


def main():
    # Create timestamped output file
    timestamp = datetime.now().strftime("mes_%m_dia_%d_hora_%H_min_%M")
    output_dir = Path(__file__).resolve().parent.parent / 'reports'
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f'Analysis_{timestamp}.txt'
    
    print(f"Saving complete analysis to: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        with redirect_stdout(f):
            _run_complete_analysis()
    
    print(f"\n✅ Complete analysis saved to: {output_file}")


def _run_complete_analysis():
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


    # Compute player performance (produces a DataFrame named `res`)
    res = calculate_player_performance(sample)


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
