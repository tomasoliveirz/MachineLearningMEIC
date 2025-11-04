#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Centralized CSV loading utilities for consistent data access across the project.

All functions auto-detect the project root and handle common fallback patterns
(e.g., prefer cleaned versions over raw).
"""

from pathlib import Path
from typing import Optional
import pandas as pd


def _infer_root(root: Optional[Path] = None) -> Path:
    """Infer project root directory.
    
    If root is None, searches up from this file until finding a 'data' directory.
    """
    if root is not None:
        return Path(root)
    
    here = Path(__file__).resolve()
    # Go up from src/utils/io.py to find project root
    for parent in [here.parent.parent.parent, here.parents[2]]:
        if (parent / "data").exists():
            return parent
    
    # Fallback: assume 2 levels up from this file
    return here.parents[2]


def load_players_cleaned(root: Optional[Path] = None) -> pd.DataFrame:
    """Load cleaned players data.
    
    Args:
        root: Project root directory (auto-detected if None)
        
    Returns:
        DataFrame with player metadata (pos, height, weight, etc.)
        
    Raises:
        FileNotFoundError: If players_cleaned.csv not found
    """
    root = _infer_root(root)
    path = root / "data" / "processed" / "players_cleaned.csv"
    
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}")
    
    return pd.read_csv(path)


def load_players_raw(root: Optional[Path] = None) -> pd.DataFrame:
    """Load raw players data.
    
    Args:
        root: Project root directory (auto-detected if None)
        
    Returns:
        DataFrame with raw player metadata
        
    Raises:
        FileNotFoundError: If players.csv not found
    """
    root = _infer_root(root)
    path = root / "data" / "raw" / "players.csv"
    
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}")
    
    return pd.read_csv(path)


def load_players(prefer_cleaned: bool = True, root: Optional[Path] = None) -> pd.DataFrame:
    """Load players data with automatic fallback.
    
    Args:
        prefer_cleaned: If True, try cleaned version first, then raw
        root: Project root directory (auto-detected if None)
        
    Returns:
        DataFrame with player metadata
        
    Raises:
        FileNotFoundError: If no players file found
    """
    root = _infer_root(root)
    
    if prefer_cleaned:
        try:
            return load_players_cleaned(root)
        except FileNotFoundError:
            return load_players_raw(root)
    else:
        try:
            return load_players_raw(root)
        except FileNotFoundError:
            return load_players_cleaned(root)


def load_players_teams(root: Optional[Path] = None) -> pd.DataFrame:
    """Load player-team-season statistics.
    
    Args:
        root: Project root directory (auto-detected if None)
        
    Returns:
        DataFrame with player stats by season and team (stints)
        
    Raises:
        FileNotFoundError: If players_teams.csv not found
    """
    root = _infer_root(root)
    path = root / "data" / "raw" / "players_teams.csv"
    
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}")
    
    return pd.read_csv(path)


def load_teams_cleaned(root: Optional[Path] = None) -> pd.DataFrame:
    """Load cleaned teams data.
    
    Args:
        root: Project root directory (auto-detected if None)
        
    Returns:
        DataFrame with team metadata and season statistics
        
    Raises:
        FileNotFoundError: If teams_cleaned.csv not found
    """
    root = _infer_root(root)
    path = root / "data" / "processed" / "teams_cleaned.csv"
    
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}")
    
    return pd.read_csv(path)


def load_teams_raw(root: Optional[Path] = None) -> pd.DataFrame:
    """Load raw teams data.
    
    Args:
        root: Project root directory (auto-detected if None)
        
    Returns:
        DataFrame with raw team data
        
    Raises:
        FileNotFoundError: If teams.csv not found
    """
    root = _infer_root(root)
    path = root / "data" / "raw" / "teams.csv"
    
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}")
    
    return pd.read_csv(path)


def load_teams(prefer_cleaned: bool = True, root: Optional[Path] = None) -> pd.DataFrame:
    """Load teams data with automatic fallback.
    
    Args:
        prefer_cleaned: If True, try cleaned version first, then raw
        root: Project root directory (auto-detected if None)
        
    Returns:
        DataFrame with team data
        
    Raises:
        FileNotFoundError: If no teams file found
    """
    root = _infer_root(root)
    
    if prefer_cleaned:
        try:
            return load_teams_cleaned(root)
        except FileNotFoundError:
            return load_teams_raw(root)
    else:
        try:
            return load_teams_raw(root)
        except FileNotFoundError:
            return load_teams_cleaned(root)


def normalize_player_ids(df: pd.DataFrame, target_col: str = "bioID") -> pd.DataFrame:
    """Ensure consistent player ID column naming.
    
    Many datasets use either 'bioID' or 'playerID'. This function ensures
    the target column exists, creating it from the other if needed.
    
    Args:
        df: DataFrame to normalize
        target_col: Desired ID column name (default: "bioID")
        
    Returns:
        DataFrame with target_col guaranteed to exist (modified in-place and returned)
    """
    if target_col in df.columns:
        df[target_col] = df[target_col].astype(str)
        return df
    
    # Try to find alternative column
    alt_col = "playerID" if target_col == "bioID" else "bioID"
    
    if alt_col in df.columns:
        df[target_col] = df[alt_col].astype(str)
    else:
        raise KeyError(f"Cannot normalize player IDs: neither '{target_col}' nor '{alt_col}' found in DataFrame")
    
    return df


def normalize_team_ids(df: pd.DataFrame, target_col: str = "tmID") -> pd.DataFrame:
    """Ensure consistent team ID column naming.
    
    Args:
        df: DataFrame to normalize
        target_col: Desired ID column name (default: "tmID")
        
    Returns:
        DataFrame with target_col guaranteed to exist (modified in-place and returned)
    """
    if target_col in df.columns:
        df[target_col] = df[target_col].astype(str)
        return df
    
    # Try to find alternative columns
    alt_cols = ["teamID", "team_id", "franchID"]
    
    for alt_col in alt_cols:
        if alt_col in df.columns:
            df[target_col] = df[alt_col].astype(str)
            return df
    
    raise KeyError(f"Cannot normalize team IDs: '{target_col}' not found and no alternatives in DataFrame")

