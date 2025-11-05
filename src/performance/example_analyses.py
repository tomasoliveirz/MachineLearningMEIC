#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example Analyses Using Coach Performance Data
==============================================

Demonstra os casos de uso principais da arquitetura de performance.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = ROOT / "data" / "processed"
PLOTS_DIR = ROOT / "reports" / "plots" / "coach_performance"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Load data
tp = pd.read_csv(PROC_DIR / "team_performance.csv")
cs = pd.read_csv(PROC_DIR / "coach_season_performance.csv")
cc = pd.read_csv(PROC_DIR / "coach_career_performance.csv")


def analysis_1_top_overachievers():
    """Who beats expectation the most?"""
    print("\n" + "="*60)
    print("ANALYSIS 1: Top Overachievers (Career)")
    print("="*60)
    
    # Filter coaches with at least 30 games
    cc_filtered = cc[cc['games'] >= 30].copy()
    
    top10 = cc_filtered.nlargest(10, 'avg_overach_pythag')
    
    print("\nTop 10 coaches by avg_overach_pythag (min 30 games):")
    print(top10[['coachID', 'seasons', 'games', 'avg_overach_pythag', 
                  'eb_career_win_pct', 'coy_awards']].to_string())
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    top10_sorted = top10.sort_values('avg_overach_pythag')
    bars = ax.barh(range(len(top10_sorted)), top10_sorted['avg_overach_pythag'])
    
    # Color by COY
    colors = ['gold' if x > 0 else 'steelblue' for x in top10_sorted['coy_awards']]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_yticks(range(len(top10_sorted)))
    ax.set_yticklabels(top10_sorted['coachID'])
    ax.set_xlabel('Avg Overachievement vs Pythag')
    ax.set_title('Top 10 Coaches by Overachievement (min 30 games)\nGold = COY Winner')
    ax.axvline(0, color='red', linestyle='--', alpha=0.3)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "top_overachievers.png", dpi=150)
    print(f"\n✓ Saved plot: {PLOTS_DIR / 'top_overachievers.png'}")


def analysis_2_correlation_matrix():
    """Correlation between key metrics"""
    print("\n" + "="*60)
    print("ANALYSIS 2: Correlation Matrix")
    print("="*60)
    
    vars_of_interest = [
        'eb_rs_win_pct', 'rs_win_pct_coach', 'po_win_pct_coach',
        'coach_overach_pythag', 'coach_overach_roster',
        'gp', 'is_first_year_with_team', 'is_coy_winner'
    ]
    
    corr = cs[vars_of_interest].corr()
    
    print("\nCorrelation matrix:")
    print(corr.round(3).to_string())
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix: Coach Season Metrics')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "correlation_matrix.png", dpi=150)
    print(f"\n✓ Saved plot: {PLOTS_DIR / 'correlation_matrix.png'}")
    
    # Key insights
    print("\n📊 Key Insights:")
    print(f"  - RS win% vs PO win%: r = {corr.loc['rs_win_pct_coach', 'po_win_pct_coach']:.3f}")
    print(f"  - Overach vs COY: r = {corr.loc['coach_overach_pythag', 'is_coy_winner']:.3f}")
    print(f"  - First-year vs Overach: r = {corr.loc['is_first_year_with_team', 'coach_overach_pythag']:.3f}")


def analysis_3_first_year_impact():
    """Do first-year coaches improve teams?"""
    print("\n" + "="*60)
    print("ANALYSIS 3: First-Year Coach Impact")
    print("="*60)
    
    first_year = cs[cs['is_first_year_with_team'] == 1].copy()
    not_first = cs[cs['is_first_year_with_team'] == 0].copy()
    
    # Stats
    print(f"\nFirst-year coaches: {len(first_year)}")
    print(f"Non-first-year: {len(not_first)}")
    
    print(f"\nMean delta vs prev team:")
    print(f"  First-year: {first_year['delta_vs_prev_team'].mean():.3f}")
    print(f"  Not first-year: {not_first['delta_vs_prev_team'].mean():.3f}")
    
    print(f"\nMean overach_pythag:")
    print(f"  First-year: {first_year['coach_overach_pythag'].mean():.3f}")
    print(f"  Not first-year: {not_first['coach_overach_pythag'].mean():.3f}")
    
    # Plot histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Delta vs prev
    first_year['delta_vs_prev_team'].dropna().hist(bins=15, ax=ax1, 
                                                     edgecolor='black', alpha=0.7)
    ax1.axvline(0, color='red', linestyle='--', linewidth=2)
    ax1.axvline(first_year['delta_vs_prev_team'].mean(), color='blue', 
                linestyle='-', linewidth=2, label='Mean')
    ax1.set_xlabel('Win% Change vs Previous Year')
    ax1.set_ylabel('Count')
    ax1.set_title('First-Year Coach Impact on Team Win%')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Overach comparison
    data_to_plot = [
        first_year['coach_overach_pythag'].dropna(),
        not_first['coach_overach_pythag'].dropna()
    ]
    ax2.boxplot(data_to_plot, labels=['First Year', 'Not First Year'])
    ax2.axhline(0, color='red', linestyle='--', alpha=0.3)
    ax2.set_ylabel('Overachievement vs Pythag')
    ax2.set_title('Overachievement by First-Year Status')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "first_year_impact.png", dpi=150)
    print(f"\n✓ Saved plot: {PLOTS_DIR / 'first_year_impact.png'}")


def analysis_4_rs_vs_po_performance():
    """Regular season vs playoff performance"""
    print("\n" + "="*60)
    print("ANALYSIS 4: RS vs Playoff Performance")
    print("="*60)
    
    cc_valid = cc.dropna(subset=['eb_career_win_pct', 'career_po_win_pct']).copy()
    
    print(f"\nCoaches with both RS and PO data: {len(cc_valid)}")
    
    corr = cc_valid[['eb_career_win_pct', 'career_po_win_pct']].corr()
    print(f"Correlation: r = {corr.iloc[0, 1]:.3f}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    scatter = ax.scatter(
        cc_valid['eb_career_win_pct'],
        cc_valid['career_po_win_pct'],
        s=cc_valid['games'],  # Size by sample size
        c=cc_valid['coy_awards'],
        cmap='viridis',
        alpha=0.6,
        edgecolors='black',
        linewidths=0.5
    )
    
    # y=x line
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, linewidth=2, label='Perfect RS=PO')
    
    # Trend line
    z = np.polyfit(cc_valid['eb_career_win_pct'], cc_valid['career_po_win_pct'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(0.3, 0.7, 100)
    ax.plot(x_trend, p(x_trend), 'b-', alpha=0.8, linewidth=2, 
            label=f'Trend (slope={z[0]:.2f})')
    
    ax.set_xlabel('Career RS Win% (EB-adjusted)')
    ax.set_ylabel('Career Playoff Win%')
    ax.set_title('Regular Season vs Playoff Performance\n(Bubble size = games coached)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('COY Awards')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "rs_vs_po_performance.png", dpi=150)
    print(f"\n✓ Saved plot: {PLOTS_DIR / 'rs_vs_po_performance.png'}")
    
    # Identify outliers
    cc_valid['rs_po_gap'] = cc_valid['career_po_win_pct'] - cc_valid['eb_career_win_pct']
    
    print("\n🏆 Best playoff overperformers (PO > RS):")
    top_po = cc_valid.nlargest(3, 'rs_po_gap')
    print(top_po[['coachID', 'eb_career_win_pct', 'career_po_win_pct', 
                   'rs_po_gap', 'games']].to_string())
    
    print("\n📉 Worst playoff underperformers (PO < RS):")
    worst_po = cc_valid.nsmallest(3, 'rs_po_gap')
    print(worst_po[['coachID', 'eb_career_win_pct', 'career_po_win_pct', 
                     'rs_po_gap', 'games']].to_string())


def analysis_5_coy_predictors():
    """What predicts Coach of the Year?"""
    print("\n" + "="*60)
    print("ANALYSIS 5: Coach of the Year Predictors")
    print("="*60)
    
    cs_coy = cs[cs['is_coy_winner'] == 1].copy()
    cs_not = cs[cs['is_coy_winner'] == 0].copy()
    
    print(f"\nCOY winners: {len(cs_coy)}")
    print("\nCOY Winners Detail:")
    print(cs_coy[['coachID', 'year', 'team_id', 'rs_win_pct_coach', 
                   'coach_overach_pythag', 'delta_vs_prev_team']].to_string())
    
    print("\n📊 Mean Comparison (COY vs Non-COY):")
    metrics = ['rs_win_pct_coach', 'coach_overach_pythag', 'coach_overach_roster',
               'delta_vs_prev_team', 'is_first_year_with_team']
    
    for metric in metrics:
        coy_mean = cs_coy[metric].mean()
        not_mean = cs_not[metric].mean()
        diff = coy_mean - not_mean
        print(f"  {metric:25s}: COY={coy_mean:6.3f}, Non={not_mean:6.3f}, Δ={diff:+6.3f}")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    metrics_to_plot = [
        ('rs_win_pct_coach', 'RS Win%'),
        ('coach_overach_pythag', 'Overach vs Pythag'),
        ('delta_vs_prev_team', 'Δ vs Prev Year'),
        ('coach_overach_roster', 'Overach vs Roster')
    ]
    
    for ax, (metric, label) in zip(axes.flat, metrics_to_plot):
        data = [cs_coy[metric].dropna(), cs_not[metric].dropna()]
        bp = ax.boxplot(data, labels=['COY', 'Non-COY'], patch_artist=True)
        
        # Color boxes
        bp['boxes'][0].set_facecolor('gold')
        bp['boxes'][1].set_facecolor('lightblue')
        
        if 'win' in metric.lower() or 'delta' in metric.lower():
            ax.axhline(0.5 if 'win' in metric else 0, color='red', 
                      linestyle='--', alpha=0.3)
        
        ax.set_ylabel(label)
        ax.set_title(f'{label} by COY Status')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "coy_predictors.png", dpi=150)
    print(f"\n✓ Saved plot: {PLOTS_DIR / 'coy_predictors.png'}")


def main():
    """Run all analyses"""
    print("\n" + "="*70)
    print(" "*15 + "COACH PERFORMANCE ANALYSES")
    print("="*70)
    
    analysis_1_top_overachievers()
    analysis_2_correlation_matrix()
    analysis_3_first_year_impact()
    analysis_4_rs_vs_po_performance()
    analysis_5_coy_predictors()
    
    print("\n" + "="*70)
    print("✓ ALL ANALYSES COMPLETE")
    print(f"✓ Plots saved to: {PLOTS_DIR}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

