"""
Model Graphics Module for Team Ranking Model
=============================================

This module generates comparative visualizations for the team ranking model,
including performance metrics across years, train/test comparisons, and
conference-level analysis.

Author: Generated for MachineLearningMEIC Project
Date: 2025-11-11
"""

from pathlib import Path
from typing import Dict, Optional
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

# Set style for all plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Paths
ROOT = Path(__file__).resolve().parents[3]
PROC_DIR = ROOT / "data" / "processed"
GRAPHICS_DIR = ROOT / "reports" / "models" / "team_ranking" / "graphics"
GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)


def load_predictions() -> pd.DataFrame:
    """Load the saved predictions CSV file."""
    csv_path = PROC_DIR / "team_ranking_predictions.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {csv_path}")
    
    df = pd.read_csv(csv_path, encoding="utf-8")
    
    # Validate required columns
    required_cols = ["year", "confID", "tmID", "name", "rank", "pred_rank", "split"]
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Ensure proper types
    df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
    df['pred_rank'] = pd.to_numeric(df['pred_rank'], errors='coerce')
    
    return df

def calculate_metrics_by_group(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """
    Calculate MAE and Spearman correlation for each group.
    
    Args:
        df: DataFrame with predictions
        group_cols: Columns to group by (e.g., ['year'], ['confID'], ['year', 'confID'])
    
    Returns:
        DataFrame with metrics per group
    """
    metrics_list = []
    
    for group_key, group in df.groupby(group_cols):
        # Filter valid predictions
        valid = group.dropna(subset=['rank', 'pred_rank'])
        
        if len(valid) < 2:
            continue
        
        # Calculate MAE
        mae = np.mean(np.abs(valid['rank'] - valid['pred_rank']))
        
        # Calculate Spearman correlation
        try:
            spearman, _ = spearmanr(valid['rank'], valid['pred_rank'])
        except:
            spearman = np.nan
        
        # Create result dict
        result = {}
        if isinstance(group_key, tuple):
            for i, col in enumerate(group_cols):
                result[col] = group_key[i]
        else:
            result[group_cols[0]] = group_key
        
        result['mae'] = mae
        result['spearman'] = spearman
        result['n_teams'] = len(valid)
        
        metrics_list.append(result)
    
    return pd.DataFrame(metrics_list)

def plot_metrics_by_year(df: pd.DataFrame, save_path: Optional[Path] = None) -> None:
    """
    Plot MAE and Spearman correlation evolution over years.
    Separate lines for train and test sets.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Calculate metrics by year and split
    metrics_by_year_split = calculate_metrics_by_group(df, ['year', 'split'])
    
    # Separate train and test
    train_metrics = metrics_by_year_split[metrics_by_year_split['split'] == 'train'].sort_values('year')
    test_metrics = metrics_by_year_split[metrics_by_year_split['split'] == 'test'].sort_values('year')
    
    # Plot MAE
    ax1 = axes[0]
    if not train_metrics.empty:
        ax1.plot(train_metrics['year'], train_metrics['mae'], 
                marker='o', linewidth=2, label='Train', color='#2E86AB')
    if not test_metrics.empty:
        ax1.plot(test_metrics['year'], test_metrics['mae'], 
                marker='s', linewidth=2, label='Test', color='#A23B72')
    
    ax1.set_xlabel('Year')
    ax1.set_ylabel('MAE (Mean Absolute Error)')
    ax1.set_title('Ranking Error Evolution by Year', fontweight='bold', pad=20)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot Spearman Correlation
    ax2 = axes[1]
    if not train_metrics.empty:
        ax2.plot(train_metrics['year'], train_metrics['spearman'], 
                marker='o', linewidth=2, label='Train', color='#2E86AB')
    if not test_metrics.empty:
        ax2.plot(test_metrics['year'], test_metrics['spearman'], 
                marker='s', linewidth=2, label='Test', color='#A23B72')
    
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Spearman Correlation')
    ax2.set_title('Ranking Correlation Evolution by Year', fontweight='bold', pad=20)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = GRAPHICS_DIR / "metrics_by_year.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {save_path}")

def plot_train_vs_test_comparison(df: pd.DataFrame, save_path: Optional[Path] = None) -> None:
    """
    Create comparative box plots for train vs test performance.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Calculate absolute errors
    df_valid = df.dropna(subset=['rank', 'pred_rank']).copy()
    df_valid['abs_error'] = np.abs(df_valid['rank'] - df_valid['pred_rank'])
    
    # Box plot for absolute errors
    ax1 = axes[0]
    train_data = df_valid[df_valid['split'] == 'train']['abs_error']
    test_data = df_valid[df_valid['split'] == 'test']['abs_error']
    
    bp = ax1.boxplot([train_data, test_data], 
                      labels=['Train', 'Test'],
                      patch_artist=True,
                      widths=0.6)
    
    # Color the boxes
    colors = ['#2E86AB', '#A23B72']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_ylabel('Absolute Ranking Error')
    ax1.set_title('Ranking Error Distribution: Train vs Test', fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Bar plot for mean MAE
    ax2 = axes[1]
    train_mae = train_data.mean() if len(train_data) > 0 else 0
    test_mae = test_data.mean() if len(test_data) > 0 else 0
    
    bars = ax2.bar(['Train', 'Test'], [train_mae, test_mae], 
                   color=colors, alpha=0.7, width=0.6)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontweight='bold')
    
    ax2.set_ylabel('Mean Absolute Error (MAE)')
    ax2.set_title('Average Ranking Error: Train vs Test', fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = GRAPHICS_DIR / "train_vs_test_comparison.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {save_path}")

def plot_conference_comparison(df: pd.DataFrame, save_path: Optional[Path] = None) -> None:
    """
    Compare model performance across different conferences.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Calculate metrics by conference
    metrics_by_conf = calculate_metrics_by_group(df, ['confID'])
    
    if metrics_by_conf.empty:
        print("  ⚠ No conference data to plot")
        return
    
    conferences = metrics_by_conf['confID'].tolist()
    mae_values = metrics_by_conf['mae'].tolist()
    spearman_values = metrics_by_conf['spearman'].tolist()
    
    # Colors for conferences
    colors = sns.color_palette("husl", len(conferences))
    
    # Plot MAE by conference
    ax1 = axes[0]
    bars1 = ax1.bar(conferences, mae_values, color=colors, alpha=0.7, width=0.6)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontweight='bold')
    
    ax1.set_xlabel('Conference')
    ax1.set_ylabel('Mean Absolute Error (MAE)')
    ax1.set_title('Ranking Error by Conference', fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot Spearman by conference
    ax2 = axes[1]
    bars2 = ax2.bar(conferences, spearman_values, color=colors, alpha=0.7, width=0.6)
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontweight='bold')
    
    ax2.set_xlabel('Conference')
    ax2.set_ylabel('Spearman Correlation')
    ax2.set_title('Ranking Correlation by Conference', fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = GRAPHICS_DIR / "conference_comparison.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {save_path}")

def plot_year_conference_heatmap(df: pd.DataFrame, save_path: Optional[Path] = None, test_only: bool = True) -> None:
    """
    Create a heatmap showing MAE and Spearman across years and conferences.

    By default (test_only=True) this function uses only the 'test' split so the
    heatmaps reflect predictive performance. Set test_only=False to include
    train+test (useful only for diagnostics).
    """
    # Use only test split by default (avoid optimistic train metrics)
    df_plot = df.copy()
    if test_only:
        if 'split' not in df_plot.columns:
            print("  ⚠ 'split' column not found — plotting using all data")
        else:
            df_plot = df_plot[df_plot['split'] == 'test']
            if df_plot.empty:
                print("  ⚠ No test rows available for heatmap")
                return

    # Calculate metrics by year and conference
    metrics = calculate_metrics_by_group(df_plot, ['year', 'confID'])
    
    if metrics.empty:
        print("  ⚠ No data for year-conference heatmap")
        return
    
    # Pivot for heatmap (MAE and Spearman)
    mae_pivot = metrics.pivot(index='confID', columns='year', values='mae')
    spearman_pivot = metrics.pivot(index='confID', columns='year', values='spearman')
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # MAE Heatmap
    ax1 = axes[0]
    sns.heatmap(mae_pivot, annot=True, fmt='.2f', cmap='YlOrRd', 
                cbar_kws={'label': 'MAE'}, ax=ax1, linewidths=0.5)
    split_label = " (Test only)" if test_only else " (All splits)"
    ax1.set_title(f'Mean Absolute Error (MAE) by Year and Conference{split_label}', 
                  fontweight='bold', pad=20)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Conference')
    
    # Spearman Heatmap
    ax2 = axes[1]
    sns.heatmap(spearman_pivot, annot=True, fmt='.2f', cmap='RdYlGn', 
                center=0, cbar_kws={'label': 'Spearman Correlation'}, 
                ax=ax2, linewidths=0.5, vmin=-1, vmax=1)
    ax2.set_title(f'Spearman Correlation by Year and Conference{split_label}', 
                  fontweight='bold', pad=20)
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Conference')
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = GRAPHICS_DIR / ("year_conference_heatmap_test.png" if test_only else "year_conference_heatmap_all.png")
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {save_path}")

def plot_prediction_scatter(df: pd.DataFrame, save_path: Optional[Path] = None) -> None:
    """
    Create scatter plots showing predicted vs actual rankings.
    Separate plots for train and test.
    """
    df_valid = df.dropna(subset=['rank', 'pred_rank']).copy()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Train scatter
    ax1 = axes[0]
    train_df = df_valid[df_valid['split'] == 'train']
    if not train_df.empty:
        ax1.scatter(train_df['rank'], train_df['pred_rank'], 
                   alpha=0.5, s=50, color='#2E86AB', edgecolors='black', linewidth=0.5)
        
        # Add diagonal line (perfect prediction)
        min_val = min(train_df['rank'].min(), train_df['pred_rank'].min())
        max_val = max(train_df['rank'].max(), train_df['pred_rank'].max())
        ax1.plot([min_val, max_val], [min_val, max_val], 
                'r--', linewidth=2, label='Perfect Prediction')
        
        # Calculate and display correlation
        corr, _ = spearmanr(train_df['rank'], train_df['pred_rank'])
        ax1.text(0.05, 0.95, f'Spearman: {corr:.3f}', 
                transform=ax1.transAxes, fontsize=12, 
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='white', alpha=0.8))
    
    ax1.set_xlabel('Actual Rank')
    ax1.set_ylabel('Predicted Rank')
    ax1.set_title('Train Set: Predicted vs Actual Ranking', fontweight='bold', pad=20)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Test scatter
    ax2 = axes[1]
    test_df = df_valid[df_valid['split'] == 'test']
    if not test_df.empty:
        ax2.scatter(test_df['rank'], test_df['pred_rank'], 
                   alpha=0.5, s=50, color='#A23B72', edgecolors='black', linewidth=0.5)
        
        # Add diagonal line
        min_val = min(test_df['rank'].min(), test_df['pred_rank'].min())
        max_val = max(test_df['rank'].max(), test_df['pred_rank'].max())
        ax2.plot([min_val, max_val], [min_val, max_val], 
                'r--', linewidth=2, label='Perfect Prediction')
        
        # Calculate and display correlation
        corr, _ = spearmanr(test_df['rank'], test_df['pred_rank'])
        ax2.text(0.05, 0.95, f'Spearman: {corr:.3f}', 
                transform=ax2.transAxes, fontsize=12, 
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='white', alpha=0.8))
    
    ax2.set_xlabel('Actual Rank')
    ax2.set_ylabel('Predicted Rank')
    ax2.set_title('Test Set: Predicted vs Actual Ranking', fontweight='bold', pad=20)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = GRAPHICS_DIR / "prediction_scatter.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {save_path}")

def plot_top_k_accuracy(df: pd.DataFrame, save_path: Optional[Path] = None) -> None:
    """
    Plot Top-K accuracy for different values of K.
    Shows how often the model predicts the correct rank within top-K positions.
    """
    # Calculate Top-K accuracy
    k_values = list(range(1, 11))
    train_acc = []
    test_acc = []
    
    for k in k_values:
        # Train accuracy
        train_df = df[df['split'] == 'train'].dropna(subset=['rank', 'pred_rank'])
        if not train_df.empty:
            train_correct = sum(abs(train_df['rank'] - train_df['pred_rank']) < k)
            train_acc.append(100 * train_correct / len(train_df))
        else:
            train_acc.append(0)
        
        # Test accuracy
        test_df = df[df['split'] == 'test'].dropna(subset=['rank', 'pred_rank'])
        if not test_df.empty:
            test_correct = sum(abs(test_df['rank'] - test_df['pred_rank']) < k)
            test_acc.append(100 * test_correct / len(test_df))
        else:
            test_acc.append(0)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(k_values, train_acc, marker='o', linewidth=2.5, 
            label='Train', color='#2E86AB', markersize=8)
    ax.plot(k_values, test_acc, marker='s', linewidth=2.5, 
            label='Test', color='#A23B72', markersize=8)
    
    ax.set_xlabel('K (Rank Tolerance)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Top-K Accuracy: Predictions Within K Positions of True Rank', 
                 fontweight='bold', pad=20)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_values)
    ax.set_ylim([0, 105])
    
    # Add reference line at 50%
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = GRAPHICS_DIR / "top_k_accuracy.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {save_path}")

def generate_all_graphics() -> None:
    """
    Main function to generate all graphics.
    This function is called from team_ranking_model.py after training.
    """
    print("\n" + "=" * 80)
    print("GENERATING MODEL GRAPHICS")
    print("=" * 80)
    
    try:
        # Load predictions
        print("\n[Graphics] Loading predictions...")
        df = load_predictions()
        print(f"  ✓ Loaded {len(df)} predictions")
        
        # Generate all plots
        print("\n[Graphics] Creating visualizations...")
        
        plot_metrics_by_year(df)
        plot_train_vs_test_comparison(df)
        plot_conference_comparison(df)
        plot_year_conference_heatmap(df)
        plot_prediction_scatter(df)
        plot_top_k_accuracy(df)
        
        print("\n" + "=" * 80)
        print(f"✓ All graphics saved to: {GRAPHICS_DIR}")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error generating graphics: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Allow running this module standalone for testing
    generate_all_graphics()
