#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calculate player performance using position-specific weights applied to per36 stats.

This module computes a composite performance metric for each player-season by:
  1. Aggregating multi-stint rows to (bioID, year, tmID) level
  2. Converting raw stats to per36-minute rates
  3. Applying position-specific weights from weights_positions.json
  4. Using team/position means as fallback for low-minute players
"""

from pathlib import Path
import sys
import pandas as pd
import json
import numpy as np

# Setup project root and imports
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import shared utilities
from src.utils.io import load_players_teams, load_players_cleaned
from src.utils.players import aggregate_stints

def _load_weights(weights_path: str):
    """Load weights file and normalize to a mapping pos -> {stat: weight}.
    Preserves original supported formats.
    """
    with open(weights_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "positions" in raw:
        pos_block = raw.get("positions", {})
        weights = {}
        for pos_key, pos_info in pos_block.items():
            if isinstance(pos_info, dict) and "weights" in pos_info:
                weights[pos_key] = pos_info.get("weights", {})
            elif isinstance(pos_info, dict):
                weights[pos_key] = pos_info
            else:
                weights[pos_key] = {}
    else:
        weights = raw
    return weights


def _collect_stats_from_weights(weights: dict):
    stats = set()
    for pos_w in weights.values():
        stats.update(pos_w.keys())
    return sorted(stats)


def _compute_pe36_columns(df: pd.DataFrame, stats: list, normalize_pe36: bool):
    """Ensure stat columns exist and optionally compute pe36 temporary cols.
    Returns mapping stat -> column to use for weighting.
    """
    if normalize_pe36:
        mins = df["minutes"].astype(float)
        for s in stats:
            pe36_col = f"__{s}_pe36"
            if s in df.columns:
                df[s] = pd.to_numeric(df[s], errors="coerce").fillna(0.0)
                df[pe36_col] = 0.0
                positive_mask = mins > 0
                df.loc[positive_mask, pe36_col] = df.loc[positive_mask, s] *36.0 / mins[positive_mask]
            else:
                df[pe36_col] = 0.0
        stat_cols = {s: f"__{s}_pe36" for s in stats}
    else:
        for s in stats:
            if s not in df.columns:
                df[s] = 0.0
        stat_cols = {s: s for s in stats}
    return stat_cols


def _apply_weights(df: pd.DataFrame, weights: dict, stat_cols: dict, perf_col: str):
    """Apply positional weights to compute perf_col in-place."""
    for pos, wdict in weights.items():
        mask = df["position"] == pos
        if not mask.any():
            continue
        for stat, w in wdict.items():
            col = stat_cols.get(stat)
            if col is None:
                continue
            df.loc[mask, perf_col] = df.loc[mask, perf_col] + df.loc[mask, col].fillna(0) * float(w)


def _handle_unknown_position(df: pd.DataFrame, weights: dict, perf_col: str):
    """If 'unknown' in weights and there are players with position 'unknown',
    assign them the global mean of known positions (or 0 if na).
    """
    if "unknown" in weights:
        unk_mask = df["position"] == "unknown"
        if unk_mask.any():
            known_mask = ~unk_mask
            if known_mask.any():
                global_mean = df.loc[known_mask, perf_col].mean()
            else:
                global_mean = df[perf_col].mean()
            if pd.isna(global_mean):
                global_mean = 0.0
            df.loc[unk_mask, perf_col] = global_mean


def _apply_min_minutes_fallback(df: pd.DataFrame, min_minutes: int, perf_col: str, fallback: str):
    """Apply fallback logic for players below min_minutes.
    
    For players with minutes < min_minutes, replaces their performance value
    with a hierarchical fallback:
      1. Team-position mean (e.g., team X's guards in year Y)
      2. Team mean (if team-position has no data)
      3. Global mean (if team has no data)
    
    Args:
        df: DataFrame with 'minutes', 'tmID', 'year', 'position' columns
        min_minutes: Minimum minutes threshold
        perf_col: Name of performance column to adjust
        fallback: Must be "team_position_mean" (only supported option)
    
    Note: Only 'team_position_mean' fallback is currently supported.
    """
    if min_minutes is not None and min_minutes > 0:
        low_mask = df["minutes"] < min_minutes
        if low_mask.any():
            df["team_pos_mean"] = df.groupby(["tmID", "year", "position"])[perf_col].transform("mean")
            df["team_mean"] = df.groupby(["tmID", "year"])[perf_col].transform("mean")

            if fallback == "team_position_mean":
                global_mean = df[perf_col].mean()
                replacements = df.loc[low_mask, "team_pos_mean"].fillna(df.loc[low_mask, "team_mean"]).fillna(global_mean)
                df.loc[low_mask, perf_col] = replacements.values
            else:
                raise ValueError(f"Unsupported fallback '{fallback}'. Only 'team_position_mean' is supported.")

            df.drop(columns=["team_pos_mean", "team_mean"], inplace=True, errors="ignore")


def _remove_pe36_columns(df: pd.DataFrame, stats: list):
    for s in stats:
        tmp = f"__{s}_pe36"
        if tmp in df.columns:
            df.drop(columns=[tmp], inplace=True)

def calculate_player_performance(
    df: pd.DataFrame,
    weights_path: str = "src/performance/weights_positions.json",
    min_minutes: int = 400,
    fallback: str = "team_position_mean", 
    normalize_pe36: bool = True,
    perf_col: str = "performance",
) -> pd.DataFrame:
    """Calculate position-weighted performance metric for players.
    
    This function computes a composite performance score by:
      1. Loading position-specific weights from a JSON file
      2. (If normalize_pe36=True) Converting raw stats to per36-minute rates
      3. Applying weights to stats: performance = Σ(weight_i × stat_i)
      4. For low-minute players (< min_minutes), replacing with team/position means
    
    Args:
        df: DataFrame with required columns:
            - tmID: team identifier
            - year: season year
            - position: player position (must match keys in weights JSON)
            - minutes: minutes played
            - stat columns referenced in weights JSON (e.g., points, rebounds, etc.)
        weights_path: Path to JSON file with position-specific stat weights
        min_minutes: Minimum minutes threshold; players below this get fallback values
        fallback: Fallback method (only "team_position_mean" supported)
        normalize_pe36: If True, convert stats to per36 rates before applying weights.
                        If False, apply weights directly to raw totals.
        perf_col: Name of output performance column
    
    Returns:
        DataFrame with added performance column
        
    Example:
        >>> df = pd.DataFrame({
        ...     'bioID': ['p1', 'p2'],
        ...     'tmID': ['TEA', 'TEA'],
        ...     'year': [2020, 2020],
        ...     'position': ['G', 'C'],
        ...     'minutes': [1000, 500],
        ...     'points': [400, 200],
        ...     'rebounds': [100, 150],
        ...     'assists': [200, 50],
        ...     'steals': [50, 20],
        ...     'blocks': [10, 30],
        ...     'turnovers': [80, 40]
        ... })
        >>> result = calculate_player_performance(df)
        >>> 'performance' in result.columns
        True
    """
    # Load position-specific weights
    weights = _load_weights(weights_path)

    # Verify required columns
    required_cols = ["tmID", "year", "position", "minutes"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Required column missing from DataFrame: '{c}'")

    df = df.copy()
    
    # Normalize position names to match weights JSON
    # Convert any "UNKNOWN", "Unknown", NaN, or empty strings to lowercase "unknown"
    df["position"] = df["position"].astype(str).str.strip()
    df["position"] = df["position"].replace({
        "nan": "unknown",
        "NaN": "unknown", 
        "None": "unknown",
        "UNKNOWN": "unknown",
        "Unknown": "unknown",
        "": "unknown"
    })
    # Additional safety: any remaining NaN becomes "unknown"
    df["position"] = df["position"].fillna("unknown")
    
    # Ensure numeric minutes
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0.0)

    # Collect stats referenced in weights (this is the source of truth)
    stats = _collect_stats_from_weights(weights)

    # Compute per36 columns if requested
    stat_cols = _compute_pe36_columns(df, stats, normalize_pe36)

    # Initialize performance column
    df[perf_col] = 0.0

    # Apply position-specific weights
    _apply_weights(df, weights, stat_cols, perf_col)

    # Handle unknown positions (if any)
    _handle_unknown_position(df, weights, perf_col)

    # Apply fallback for low-minute players
    _apply_min_minutes_fallback(df, min_minutes, perf_col, fallback)

    # Clean up temporary per36 columns
    if normalize_pe36:
        _remove_pe36_columns(df, stats)

    return df

def main():
    """Main entry point: compute and save player performance metrics."""
    weights_path = ROOT / "src" / "performance" / "weights_positions.json"
    
    print("=" * 60)
    print("PLAYER PERFORMANCE CALCULATION")
    print("=" * 60)
    
    # 1) Load data using shared utilities
    print("\n[1/4] Loading data...")
    df_players_teams = load_players_teams(root=ROOT)
    df_players = load_players_cleaned(root=ROOT)
    print(f"  ✓ Loaded {len(df_players_teams):,} player-team-season rows")
    print(f"  ✓ Loaded {len(df_players):,} player metadata records")
    
    # 2) Aggregate stints to (bioID, year, tmID) level
    print("\n[2/4] Aggregating multi-stint rows...")
    df = aggregate_stints(df_players_teams)
    print(f"  ✓ Aggregated to {len(df):,} player-year-team records")
    
    # 3) Merge with player metadata to get positions
    print("\n[3/4] Merging positions and preparing data...")
    
    # Create position mapping from players (bioID -> position)
    pos_map = dict(zip(
        df_players["bioID"].astype(str), 
        df_players["pos"].astype(str)
    ))
    
    # Map positions (will be further normalized in calculate_player_performance)
    df["position"] = df["bioID"].astype(str).map(pos_map).fillna("unknown")
    
    # Ensure year is numeric
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    
    # Load weights to determine which stats are needed
    weights = _load_weights(str(weights_path))
    stats_needed = _collect_stats_from_weights(weights)
    
    # Ensure all required stat columns exist and are numeric
    for stat in stats_needed:
        if stat not in df.columns:
            df[stat] = 0.0
        else:
            df[stat] = pd.to_numeric(df[stat], errors="coerce").fillna(0.0)
    
    print(f"  ✓ Position mapping complete")
    print(f"  ✓ Stats required by weights: {', '.join(sorted(stats_needed))}")
    
    # 4) Calculate performance using position-specific per36 weights
    print("\n[4/4] Calculating performance metrics...")
    out = calculate_player_performance(
        df,
        weights_path=str(weights_path),
        min_minutes=500,  # Players below 500 minutes get team/position means
        fallback="team_position_mean",
        normalize_pe36=True,
        perf_col="performance"
    )
    
    # Prepare columns to save
    # Core identifiers + stats + performance metric
    cols_to_save = ["bioID", "year", "tmID", "position", "minutes"]
    
    # Add all stats that were used in the calculation
    for stat in sorted(stats_needed):
        if stat in out.columns:
            cols_to_save.append(stat)
    
    # Add performance metric
    cols_to_save.append("performance")
    
    # Save to processed directory
    out_path = ROOT / "data" / "processed" / "player_performance.csv"
    out[cols_to_save].to_csv(out_path, index=False)
    
    print(f"\n✓ Saved player performance to: {out_path}")
    print(f"  Rows: {len(out):,}")
    print(f"  Columns: {len(cols_to_save)}")
    
    # Show summary statistics
    print("\nPerformance metric summary:")
    print(out["performance"].describe().round(2))
    
    print("\nSample of output:")
    print(out[cols_to_save].head(10).to_string(index=False))
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()