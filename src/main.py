import time
import pandas as pd
from pathlib import Path
from contextlib import redirect_stdout
from datetime import datetime
import sys

# Add src to python path to look for modules
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import pipeline stages
from src.cleaning import clean_players, clean_teams
from src.features import aggregate_team_season
from src.analysis import analyzer
from src.performance import player_performance, team_performance, coach_season_performance
from src.model.ranking_model import team_ranking_model

def run_step(step_name, func, *args, **kwargs):
    print(f"\n{'='*60}")
    print(f"STARTING STEP: {step_name}")
    print(f"{'='*60}\n")
    start_time = time.time()
    try:
        func(*args, **kwargs)
        elapsed = time.time() - start_time
        print(f"\n{'-'*60}")
        print(f"COMPLETED STEP: {step_name} in {elapsed:.2f} seconds")
        print(f"{'-'*60}\n")
    except Exception as e:
        print(f"\n{'!'*60}")
        print(f"FAILED STEP: {step_name}")
        print(f"Error: {e}")
        print(f"{'!'*60}\n")
        raise e

def main():
    print("Starting WNBA Data Pipeline...")
    total_start = time.time()

    # 1. Cleaning
    run_step("Clean Players", clean_players.main)
    run_step("Clean Teams", clean_teams.main)

    # 2. Features (Team Statistics)
    run_step("Aggregate Team Season Stats", aggregate_team_season.main)

    # 3. Analysis (Descriptive)
    # analyzer.main() parses args, so we call the plotting logic directly if possible,
    # or simulate arg parsing. Analyzer.main parses args. Let's wrap it nicely?
    # Actually analyzer main just calls load_data and plots. We can modify it slightly 
    # or just call main() but verify it defaults to 'raw' if we don't pass anything.
    # The requirement says "clean then analysis", so maybe we want mode='cleaned'.
    # Analyzer uses argparse. We can bypass main and call logic, or monkeypatch sys.argv.
    # Let's bypass main and call functions for better control.
    print(f"\n{'='*60}")
    print(f"STARTING STEP: Analysis (Cleaned Data)")
    print(f"{'='*60}\n")
    try:
        ds = analyzer.load_data(mode="cleaned")
        fig_dir, tab_dir = analyzer.get_out_dirs(mode="cleaned")
        analyzer.plot_players(ds, fig_dir)
        analyzer.plot_player_points(ds, fig_dir)
        analyzer.plot_teams(ds, fig_dir)
        analyzer.plot_awards(ds, fig_dir)
        analyzer.write_report(ds, tab_dir)
        print("Analysis completed.")
    except Exception as e:
        print(f"Analysis failed: {e}")

    # 4. Performance Metrics
    run_step("Player Performance", player_performance.main)
    run_step("Team Performance", team_performance.main)
    run_step("Coach Performance", coach_season_performance.main)

    # 5. Model Training & Ranking
    # Assuming standard configuration for the full pipeline run
    run_step(
        "Team Ranking Model", 
        team_ranking_model.run_team_ranking_model,
        max_train_year=8,
        val_years=2,
        report_name="team_ranking_report_pipeline.txt",
        generate_graphics=True
    )

    total_elapsed = time.time() - total_start
    print(f"\n{'='*80}")
    print(f"PIPELINE COMPLETED SUCCESSFULLY in {total_elapsed:.2f} seconds")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
