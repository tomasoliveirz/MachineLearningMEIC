#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WNBA Data Analysis
Exploratory data analysis of WNBA datasets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

def load_datasets():
    """Load all CSV files from the data directory"""
    data_path = Path('../data')
    files = {
        'awards': 'awards_players.csv',
        'coaches': 'coaches.csv',
        'players': 'players.csv',
        'player_stats': 'players_teams.csv',
        'playoffs_series': 'series_post.csv',
        'playoffs_teams': 'teams_post.csv',
        'teams': 'teams.csv'
    }
    
    datasets = {}
    print("Loading datasets...")
    for name, filename in files.items():
        try:
            datasets[name] = pd.read_csv(data_path / filename)
            print(f"{name}: {datasets[name].shape[0]:,} rows")
        except Exception as e:
            print(f"Error loading {name}: {e}")
    
    return datasets

def basic_summary(datasets):
    """Display a basic summary of each dataset"""
    print("\n" + "="*60)
    print("DATASET OVERVIEW")
    print("="*60)
    
    for name, df in datasets.items():
        print(f"\n{name.upper()}")
        print(f"   Rows: {df.shape[0]:,}")
        print(f"   Columns: {df.shape[1]}")
        print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        missing = df.isnull().sum().sum()
        if missing > 0:
            print(f"   Missing values: {missing:,}")
        else:
            print("   No missing values")

def unique_counts(datasets):
    """Count unique values in each dataset"""
    print("\n" + "="*60)
    print("UNIQUE VALUE COUNTS")
    print("="*60)
    
    if 'awards' in datasets:
        df = datasets['awards']
        print("\nAWARDS")
        print(f"   Award types: {df['award'].nunique()}")
        print(f"   Unique players awarded: {df['playerID'].nunique()}")
        print(f"   Years covered: {df['year'].nunique()}")
        print("   Most frequent awards:")
        for award, count in df['award'].value_counts().head(3).items():
            print(f"      {award}: {count}")
    
    if 'players' in datasets:
        df = datasets['players']
        print("\nPLAYERS")
        print(f"   Total players: {len(df):,}")
        positions = df['pos'].dropna()
        if len(positions) > 0:
            print(f"   Positions: {positions.nunique()}")
            print(f"   Colleges: {df['college'].nunique()}")
        print("   Most common positions:")
        for pos, count in positions.value_counts().head(3).items():
            print(f"      {pos}: {count}")
    
    if 'teams' in datasets:
        df = datasets['teams']
        print("\nTEAMS")
        print(f"   Teams: {df['tmID'].nunique()}")
        print(f"   Arenas: {df['arena'].nunique()}")
        print(f"   Years covered: {df['year'].nunique()}")
        print("   Teams with most seasons:")
        for team, count in df['tmID'].value_counts().head(3).items():
            print(f"      {team}: {count} seasons")
    
    if 'player_stats' in datasets:
        df = datasets['player_stats']
        print("\nPLAYER STATISTICS")
        print(f"   Players with statistics: {df['playerID'].nunique():,}")
        print(f"   Teams with data: {df['tmID'].nunique()}")
        print(f"   Records: {len(df):,}")

def extremes_analysis(datasets):
    """Identify maximum and minimum values of interest"""
    print("\n" + "="*60)
    print("EXTREMES ANALYSIS")
    print("="*60)
    
    if 'players' in datasets:
        df = datasets['players']
        height = df['height'].dropna()
        weight = df['weight'].dropna()
        if len(height) > 0:
            print("\nPLAYER PHYSICAL ATTRIBUTES")
            print(f"   Tallest: {height.max():.0f} inches ({height.max()*2.54:.0f} cm)")
            print(f"   Shortest: {height.min():.0f} inches ({height.min()*2.54:.0f} cm)")
            print(f"   Average height: {height.mean():.1f} inches ({height.mean()*2.54:.0f} cm)")
            print(f"   Heaviest: {weight.max():.0f} lbs ({weight.max()*0.45:.0f} kg)")
            print(f"   Lightest: {weight.min():.0f} lbs ({weight.min()*0.45:.0f} kg)")
            print(f"   Average weight: {weight.mean():.1f} lbs ({weight.mean()*0.45:.0f} kg)")
    
    if 'player_stats' in datasets:
        df = datasets['player_stats']
        print("\nGAME RECORDS")
        idx_max_points = df['points'].idxmax()
        print(f"   Most points in a season: {df.loc[idx_max_points, 'points']} ({df.loc[idx_max_points, 'playerID']})")
        idx_max_minutes = df['minutes'].idxmax()
        print(f"   Most minutes in a season: {df.loc[idx_max_minutes, 'minutes']} ({df.loc[idx_max_minutes, 'playerID']})")
        idx_max_assists = df['assists'].idxmax()
        print(f"   Most assists: {df.loc[idx_max_assists, 'assists']} ({df.loc[idx_max_assists, 'playerID']})")
        idx_max_rebounds = df['rebounds'].idxmax()
        print(f"   Most rebounds: {df.loc[idx_max_rebounds, 'rebounds']} ({df.loc[idx_max_rebounds, 'playerID']})")
    
    if 'teams' in datasets:
        df = datasets['teams']
        print("\nTEAM RECORDS")
        idx_max_wins = df['won'].idxmax()
        print(f"   Most wins: {df.loc[idx_max_wins, 'won']} ({df.loc[idx_max_wins, 'name']} - {df.loc[idx_max_wins, 'year']})")
        idx_max_attend = df['attend'].idxmax()
        print(f"   Highest attendance: {df.loc[idx_max_attend, 'attend']:,} ({df.loc[idx_max_attend, 'name']} - {df.loc[idx_max_attend, 'year']})")
        idx_max_pts = df['o_pts'].idxmax()
        print(f"   Most points scored: {df.loc[idx_max_pts, 'o_pts']} ({df.loc[idx_max_pts, 'name']} - {df.loc[idx_max_pts, 'year']})")

def create_histograms(datasets):
    """Generate histograms for key variables"""
    print("\n" + "="*60)
    print("CREATING HISTOGRAMS")
    print("="*60)
    
    plt.style.use('default')
    output_dir = Path('output')
    
    if 'players' in datasets:
        height = datasets['players']['height'].dropna()
        if len(height) > 0:
            plt.figure(figsize=(10, 6))
            plt.hist(height, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title('Player Height Distribution')
            plt.xlabel('Height (inches)')
            plt.ylabel('Number of Players')
            plt.grid(True, alpha=0.3)
            plt.savefig(output_dir / 'height_distribution.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("Height distribution histogram saved")
    
    if 'player_stats' in datasets:
        points = datasets['player_stats']['points'].dropna()
        if len(points) > 0:
            plt.figure(figsize=(10, 6))
            plt.hist(points, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
            plt.title('Points per Season Distribution')
            plt.xlabel('Points per Season')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.savefig(output_dir / 'points_distribution.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("Points distribution histogram saved")
    
    if 'teams' in datasets:
        wins = datasets['teams']['won'].dropna()
        if len(wins) > 0:
            plt.figure(figsize=(10, 6))
            plt.hist(wins, bins=15, alpha=0.7, color='gold', edgecolor='black')
            plt.title('Team Wins Distribution')
            plt.xlabel('Number of Wins')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.savefig(output_dir / 'wins_distribution.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("Wins distribution histogram saved")

def create_analysis_charts(datasets):
    """Create charts and visualizations for analysis"""
    print("\n" + "="*60)
    print("CREATING ANALYSIS CHARTS")
    print("="*60)
    
    output_dir = Path('output')
    
    if 'player_stats' in datasets:
        top_scorers = datasets['player_stats'].groupby('playerID')['points'].sum().sort_values(ascending=False).head(10)
        plt.figure(figsize=(12, 6))
        top_scorers.plot(kind='bar', color='coral')
        plt.title('Top 10 Career Scorers')
        plt.xlabel('Player')
        plt.ylabel('Total Career Points')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'top_career_scorers.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Top career scorers chart saved")
    
    if 'awards' in datasets:
        awards_by_year = datasets['awards']['year'].value_counts().sort_index()
        plt.figure(figsize=(10, 6))
        awards_by_year.plot(kind='line', marker='o', color='purple', linewidth=2)
        plt.title('Awards Distribution Over Years')
        plt.xlabel('Year')
        plt.ylabel('Number of Awards')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'awards_over_years.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Awards over years chart saved")
    
    if 'players' in datasets:
        physical_data = datasets['players'].dropna(subset=['height', 'weight'])
        if len(physical_data) > 50:
            plt.figure(figsize=(10, 6))
            plt.scatter(physical_data['height'], physical_data['weight'], alpha=0.6, color='teal', s=50)
            plt.title('Player Height vs Weight Correlation')
            plt.xlabel('Height (inches)')
            plt.ylabel('Weight (lbs)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / 'height_weight_correlation.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("Height vs weight correlation chart saved")

def generate_report(datasets):
    """Generate a text report with summary statistics"""
    print("\n" + "="*60)
    print("GENERATING ANALYSIS REPORT")
    print("="*60)
    
    output_dir = Path('output')
    
    with open(output_dir / 'analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write("WNBA DATA ANALYSIS REPORT\n")
        f.write("=" * 40 + "\n\n")
        
        f.write("DATASET OVERVIEW\n")
        f.write("-" * 20 + "\n")
        for name, df in datasets.items():
            f.write(f"{name.capitalize()}: {df.shape[0]:,} records\n")
        
        if 'players' in datasets:
            df = datasets['players']
            f.write("\nPLAYERS ANALYSIS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total players: {len(df):,}\n")
            height = df['height'].dropna()
            if len(height) > 0:
                f.write(f"Average height: {height.mean():.1f} inches\n")
                f.write(f"Tallest player: {height.max():.0f} inches\n")
                f.write(f"Shortest player: {height.min():.0f} inches\n")
        
        if 'player_stats' in datasets:
            df = datasets['player_stats']
            f.write("\nPERFORMANCE STATISTICS\n")
            f.write("-" * 25 + "\n")
            f.write(f"Total player-season records: {len(df):,}\n")
            f.write(f"Unique players: {df['playerID'].nunique():,}\n")
            f.write(f"Highest single-season points: {df['points'].max()}\n")
            f.write(f"Average points per season: {df['points'].mean():.1f}\n")
        
        if 'teams' in datasets:
            df = datasets['teams']
            f.write("\nTEAM ANALYSIS\n")
            f.write("-" * 15 + "\n")
            f.write(f"Total team seasons: {len(df)}\n")
            f.write(f"Different teams: {df['tmID'].nunique()}\n")
            f.write(f"Maximum wins in a season: {df['won'].max()}\n")
            f.write(f"Average wins per season: {df['won'].mean():.1f}\n")
    
    print("Analysis report saved as 'analysis_report.txt'")

def main():
    """Main function to run the full analysis"""
    print("WNBA DATA ANALYSIS")
    print("=" * 50)
    
    os.chdir('/home/tomio/Documents/UNI/AC/wnba_analysis')
    
    datasets = load_datasets()
    if not datasets:
        print("No datasets loaded.")
        return
    
    basic_summary(datasets)
    unique_counts(datasets)
    extremes_analysis(datasets)
    create_histograms(datasets)
    create_analysis_charts(datasets)
    generate_report(datasets)
    
    print("\nANALYSIS COMPLETE")
    print("Generated files in 'output' directory:")
    print(" - height_distribution.png")
    print(" - points_distribution.png") 
    print(" - wins_distribution.png")
    print(" - top_career_scorers.png")
    print(" - awards_over_years.png")
    print(" - height_weight_correlation.png")
    print(" - analysis_report.txt")

if __name__ == "__main__":
    main()
