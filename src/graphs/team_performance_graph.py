import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Get the project root directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load the team performance data
file_path = BASE_DIR / "data" / "processed" / "team_performance.csv"
df = pd.read_csv(file_path)

# Set Seaborn style
sns.set(style="whitegrid")

# Create output directory for graphs
output_dir = BASE_DIR / "reports" / "performance_graphs" / "team_performance"
output_dir.mkdir(parents=True, exist_ok=True)

# Function to plot win percentage over years for a specific team
def plot_team_win_pct(team_id):
    team_data = df[df['team_id'] == team_id]
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=team_data, x='year', y='rs_win_pct', marker='o', label='Regular Season Win%')
    sns.lineplot(data=team_data, x='year', y='pythag_win_pct', marker='o', label='Pythagorean Win%')
    plt.title(f"Win Percentage Over Years for Team {team_id}")
    plt.xlabel("Year")
    plt.ylabel("Win Percentage")
    plt.legend()
    output_path = output_dir / f"{team_id}_win_percentage.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")

# Function to plot overachievement metrics for all teams
def plot_overachievement():
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='team_id', y='overach_pythag', palette="coolwarm")
    plt.title("Overachievement (Pythagorean) by Team")
    plt.xlabel("Team ID")
    plt.ylabel("Overachievement (Pythagorean)")
    plt.xticks(rotation=90)
    output_path = output_dir / "overachievement_pythagorean.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")

# Function to plot team strength vs. win percentage
def plot_strength_vs_win_pct():
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='team_strength', y='rs_win_pct', hue='team_id', palette="tab10")
    plt.title("Team Strength vs. Regular Season Win Percentage")
    plt.xlabel("Team Strength")
    plt.ylabel("Win Percentage")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Team ID")
    output_path = output_dir / "strength_vs_win_percentage.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")

# Example usage
plot_team_win_pct("ATL")
plot_overachievement()
plot_strength_vs_win_pct()