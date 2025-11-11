# Model Graphics Module

## Overview

The `model_grafics.py` module provides automated visualization generation for the Team Ranking Model. It creates comprehensive comparative graphics to analyze model performance across different dimensions.

## Generated Graphics

All graphics are automatically saved to: `reports/models/graphics/`

### 1. **metrics_by_year.png**
- **Description**: Evolution of MAE and Spearman correlation over years
- **Features**: 
  - Separate lines for train and test sets
  - Two subplots: MAE (top) and Spearman correlation (bottom)
  - Shows model performance trends over time

### 2. **train_vs_test_comparison.png**
- **Description**: Comparative analysis of train vs test performance
- **Features**:
  - Box plots showing error distribution
  - Bar chart comparing mean MAE
  - Helps identify overfitting or underfitting

### 3. **conference_comparison.png**
- **Description**: Performance metrics across different conferences
- **Features**:
  - MAE and Spearman correlation by conference
  - Color-coded bars for easy comparison
  - Identifies conference-specific model strengths/weaknesses

### 4. **year_conference_heatmap.png**
- **Description**: Heatmap visualization of performance across years and conferences
- **Features**:
  - Two heatmaps: MAE and Spearman correlation
  - Color-coded cells showing metric values
  - Reveals temporal and conference-specific patterns

### 5. **prediction_scatter.png**
- **Description**: Scatter plots of predicted vs actual rankings
- **Features**:
  - Separate plots for train and test sets
  - Diagonal line showing perfect prediction
  - Spearman correlation displayed on each plot

### 6. **top_k_accuracy.png**
- **Description**: Top-K accuracy showing prediction tolerance
- **Features**:
  - Line plot for K values from 1 to 10
  - Shows percentage of predictions within K positions of true rank
  - Separate curves for train and test sets

## Usage

### Automatic Integration
The module is automatically called at the end of the `team_ranking_model.py` pipeline:

```python
# In team_ranking_model.py, at the end of run_team_ranking_model():
generate_all_graphics()
```

### Standalone Execution
You can also run the graphics module independently:

```bash
python model_grafics.py
```

This will load the saved predictions from `data/processed/team_ranking_predictions.csv` and generate all graphics.

## Requirements

The module requires the following Python packages:
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `matplotlib` - Base plotting library
- `seaborn` - Statistical visualization
- `scipy` - Statistical functions (Spearman correlation)

## Customization

### Changing Save Location
Edit the `GRAPHICS_DIR` variable in `model_grafics.py`:

```python
GRAPHICS_DIR = ROOT / "reports" / "models" / "graphics"
```

### Adding New Visualizations
To add a new graphic:

1. Create a new function following the pattern:
```python
def plot_new_visualization(df: pd.DataFrame, save_path: Optional[Path] = None) -> None:
    # Your plotting code here
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
```

2. Add the function call to `generate_all_graphics()`:
```python
def generate_all_graphics() -> None:
    # ... existing code ...
    plot_new_visualization(df)
```

### Customizing Plot Style
The module uses seaborn's "whitegrid" style by default. You can change this at the top of `model_grafics.py`:

```python
sns.set_style("whitegrid")  # Options: whitegrid, darkgrid, white, dark, ticks
```

## Data Sources

The module reads from:
- **Input**: `data/processed/team_ranking_predictions.csv`
  - Required columns: `year`, `confID`, `tmID`, `name`, `rank`, `pred_rank`, `split`

## Error Handling

The module includes comprehensive error handling:
- Validates existence of required files
- Checks for required columns in the data
- Gracefully handles missing or invalid data
- Prints informative error messages

If graphics generation fails, the main pipeline will continue and display a warning message.

## Output Format

All graphics are saved as:
- **Format**: PNG
- **Resolution**: 300 DPI (publication quality)
- **Size**: Variable (optimized for readability)
- **Encoding**: RGB

## Performance Considerations

- The module is designed to be efficient even with large datasets
- Graphics generation typically takes < 10 seconds
- All plots are created in memory and saved directly to disk
- Memory is properly cleaned up after each plot (using `plt.close()`)

## Troubleshooting

### Graphics Not Generated
1. Check if predictions file exists: `data/processed/team_ranking_predictions.csv`
2. Verify required columns are present
3. Check console output for specific error messages

### Graphics Folder Not Found
The module automatically creates the graphics folder if it doesn't exist:
```python
GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)
```

### Empty Plots
- Ensure the predictions file contains valid data
- Check that train/test splits are properly labeled
- Verify that rank values are not all NaN

## Future Enhancements

Potential improvements for future versions:
- Interactive plots using Plotly
- PDF report generation with all graphics
- Configurable color schemes
- Additional metrics (precision@K, NDCG)
- Per-team performance tracking
- Feature importance visualization

## Author

Generated for MachineLearningMEIC Project  
Date: 2025-11-11

## License

This module is part of the MachineLearningMEIC project.
