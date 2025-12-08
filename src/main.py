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
    # A. Standard Validation Run
    run_step(
        "Team Ranking Model (Validation)", 
        team_ranking_model.run_team_ranking_model,
        max_train_year=8,
        val_years=2,
        report_name="team_ranking_report_pipeline.txt",
        generate_graphics=True
    )

    # B. Target Season 11 Prediction & Printing
    def predict_and_print_s11():
        # Train on 1-10, Predict 11
        team_ranking_model.run_team_ranking_model(
            max_train_year=10,
            val_years=2,
            report_name="team_ranking_report_s11.txt",
            generate_graphics=False,
            target_season=11
        )
        
        # Read and print
        pred_path = ROOT / "data" / "processed" / "team_ranking_predictions.csv"
        if pred_path.exists():
            df = pd.read_csv(pred_path)
            df_s11 = df[df['year'] == 11].sort_values(['confID', 'pred_rank'])
            
            print("\n" + "="*60)
            print("SEASON 11 PREDICTED RANKINGS")
            print("="*60)
            
            for conf, group in df_s11.groupby('confID'):
                print(f"\nCONFERENCE: {conf}")
                print(f"{'Rank':<5} {'Team':<30} {'Score':<10}")
                print("-" * 50)
                for _, row in group.iterrows():
                    print(f"{int(row['pred_rank']):<5} {row['name']:<30} {row['pred_score']:.4f}")
            print("\n" + "="*60 + "\n")
        else:
            print("Error: Prediction file not found.")

    run_step("Predict Season 11", predict_and_print_s11)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*80}")
    print(f"PIPELINE COMPLETED SUCCESSFULLY in {total_elapsed:.2f} seconds")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
