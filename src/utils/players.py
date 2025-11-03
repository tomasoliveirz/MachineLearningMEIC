#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared player data utilities for aggregation, transformation, and metrics.

Uses the actual column names from players_teams.csv:
- minutes, points, rebounds, oRebounds, dRebounds
- assists, steals, blocks, turnovers
- GP, GS
"""

from typing import Tuple

import numpy as np
import pandas as pd


def aggregate_stints(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate multi-stint rows to player-year-team level.
    
    Uses REAL column names from players_teams.csv:
    - playerID, year, tmID
    - minutes, points, rebounds, oRebounds, dRebounds
    - assists, steals, blocks, turnovers
    - GP, GS
    """
    df = df.copy()
    
    # Ensure we have bioID for internal consistency
    if "bioID" not in df.columns:
        if "playerID" not in df.columns:
            raise KeyError("Expected 'playerID' in players_teams.csv")
        df["bioID"] = df["playerID"].astype(str)
    else:
        df["bioID"] = df["bioID"].astype(str)
    
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["tmID"] = df["tmID"].astype(str)
    
    # REAL columns from the CSV (no synonyms!)
    numeric_cols = [
        "minutes",
        "points",
        "rebounds", "oRebounds", "dRebounds",
        "assists", "steals", "blocks", "turnovers",
        "GP", "GS",
    ]
    
    stat_cols = [c for c in numeric_cols if c in df.columns]
    for c in stat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    
    agg = (
        df.groupby(["bioID", "year", "tmID"], dropna=False)[stat_cols]
          .sum(min_count=1)
          .reset_index()
    )
    return agg


def label_rookies(df: pd.DataFrame) -> pd.Series:
    """Return boolean Series marking rookie seasons (first year > min year in dataset)."""
    years = df.groupby("bioID", dropna=False)["year"].transform("min")
    min_year_dataset = df["year"].min()
    return (df["year"].eq(years)) & (df["year"] > min_year_dataset)


def compute_per36(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Return per36 metric and minutes used for rate denominator.
    
    Uses REAL column names from players_teams.csv:
    per36 = points + 0.7*rebounds + 0.7*assists + 1.2*steals + 1.2*blocks - 0.7*turnovers
    (basic, position-agnostic composite)
    
    Returns:
        per36: Series of per-36 composite rates
        minutes: Series of minutes played
    """
    # Helper: get column or zeros if doesn't exist
    def s(col: str) -> pd.Series:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
        return pd.Series(0.0, index=df.index)
    
    pts = s("points")
    reb = s("rebounds")
    ast = s("assists")
    stl = s("steals")
    blk = s("blocks")
    tov = s("turnovers")
    
    raw = (
        pts + 0.7 * reb + 0.7 * ast + 1.2 * stl + 1.2 * blk - 0.7 * tov
    ).astype(float)
    
    if "minutes" not in df.columns:
        raise KeyError("players_teams.csv must have 'minutes' column")
    
    minutes = pd.to_numeric(df["minutes"], errors="coerce")
    minutes = minutes.replace({0: np.nan})
    
    # Floor at 12 minutes to avoid extreme rates
    minutes_floor = minutes.copy()
    minutes_floor.loc[(minutes_floor.notna()) & (minutes_floor < 12.0)] = 12.0
    per36 = (raw / minutes_floor) * 36.0
    return per36, minutes.fillna(0.0)


def per36_next_year(df: pd.DataFrame) -> pd.Series:
    """Map each row's next-season per36 for the same player (any team)."""
    per36, _ = compute_per36(df)
    nxt = (
        df.assign(per36=per36)
          .sort_values(["bioID", "year"]) 
          .groupby("bioID")["per36"]
          .shift(-1)
    )
    return nxt

