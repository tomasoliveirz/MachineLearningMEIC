#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import sys
import pandas as pd
import numpy as np

# setup basic folder structure
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# Add project root to path for imports
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import shared utilities
from src.utils.players import infer_rookie_origin, aggregate_stints, label_rookies
from src.utils.io import load_players, load_players_teams, load_teams

# Wrapper functions for backward compatibility (now use utils.io)
def _load_players():
    """Load players data (prefer cleaned version)."""
    return load_players(prefer_cleaned=True, root=ROOT)

def _load_teams():
    """Load teams data (prefer cleaned version)."""
    try:
        return load_teams(prefer_cleaned=True, root=ROOT)
    except FileNotFoundError:
        return None

def _load_players_teams():
    """Load player-team-season links."""
    return load_players_teams(root=ROOT)

# load team season metrics (used to add form variables later)
def _load_team_season():
    ts = PROC_DIR / "team_season_statistics.csv"
    return pd.read_csv(ts) if ts.exists() else None

# label which rows correspond to rookie seasons
def _label_rookies(players_teams: pd.DataFrame) -> pd.DataFrame:
    """
    Label rookie seasons and aggregate stints.
    
    Uses utilities from src.utils.players for aggregation and labeling.
    """
    # First aggregate stints to player-year-team level
    df = aggregate_stints(players_teams)
    
    # Normalize column names (aggregate_stints uses bioID)
    if "bioID" in df.columns and "playerID" not in df.columns:
        df["playerID"] = df["bioID"]
    
    # Find first year per player for rookie detection
    rookie_year = df.groupby("playerID", dropna=False)["year"].min().rename("rookie_year")
    df = df.merge(rookie_year, on="playerID", how="left")
    df["is_rookie"] = (df["year"] == df["rookie_year"]).astype(int)
    
    return df, rookie_year.reset_index()

# detect player origin (ncaa vs non-ncaa) based on college info
# (Now uses imported function from utils.players)
_rookie_origin = infer_rookie_origin

# build team-level rookie metrics (minutes, points, shares)
def _team_rookie_features(pst: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    players = players.copy()
    players["playerID"] = players.get("bioID", players.get("playerID"))
    origin_map = dict(zip(players["playerID"], _rookie_origin(players)))
    pst["rookie_origin"] = pst["playerID"].map(origin_map)

    # total team minutes and points
    g_all = pst.groupby(["tmID", "year"], dropna=False)
    team_tot = g_all[["minutes", "points"]].sum(min_count=1).rename(
        columns={"minutes": "team_minutes", "points": "team_points"}
    )

    # total rookie minutes and points
    r_all = pst[pst["is_rookie"] == 1]
    g_r = r_all.groupby(["tmID", "year"], dropna=False)
    rook_tot = g_r[["minutes", "points"]].sum(min_count=1).rename(
        columns={"minutes": "rookie_minutes", "points": "rookie_points"}
    )
    rook_cnt = g_r.size().rename("rookie_count")

    # rookies grouped by origin (ncaa vs non-ncaa)
    def bucket(origin):
        r = r_all[r_all["rookie_origin"] == origin]
        g = r.groupby(["tmID", "year"], dropna=False)
        tot = g[["minutes", "points"]].sum(min_count=1)
        cnt = g.size()
        return (
            tot.rename(columns={
                "minutes": f"{origin}_rookie_minutes",
                "points": f"{origin}_rookie_points"
            }),
            cnt.rename(f"{origin}_rookie_count")
        )

    ncaa_tot, ncaa_cnt = bucket("ncaa")
    non_tot, non_cnt = bucket("non_ncaa")

    # merge everything into one team-level dataframe
    feats = (
        team_tot.join(rook_tot, how="left")
                .join(rook_cnt, how="left")
                .join(ncaa_tot, how="left")
                .join(ncaa_cnt, how="left")
                .join(non_tot, how="left")
                .join(non_cnt, how="left")
                .reset_index()
    )

    # fill missing values with zeros
    for c in [
        "rookie_minutes", "rookie_points", "rookie_count",
        "ncaa_rookie_minutes", "ncaa_rookie_points", "ncaa_rookie_count",
        "non_ncaa_rookie_minutes", "non_ncaa_rookie_points", "non_ncaa_rookie_count"
    ]:
        if c in feats.columns:
            feats[c] = feats[c].fillna(0)

    # calculate rookie contribution ratios
    feats["rookie_min_share"] = feats["rookie_minutes"] / feats["team_minutes"].replace(0, np.nan)
    feats["rookie_pts_share"] = feats["rookie_points"] / feats["team_points"].replace(0, np.nan)
    feats["ncaa_rookie_min_share"] = feats["ncaa_rookie_minutes"] / feats["team_minutes"].replace(0, np.nan)
    feats["non_ncaa_rookie_min_share"] = feats["non_ncaa_rookie_minutes"] / feats["team_minutes"].replace(0, np.nan)

    return feats

# attach readable team names
def _attach_team_names(feats: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    if teams is None:
        return feats
    cols = [c for c in ["tmID", "year", "name"] if c in teams.columns]
    if not {"tmID", "year"}.issubset(cols):
        return feats
    names = teams[cols].drop_duplicates()
    return feats.merge(names, on=["tmID", "year"], how="left")

# attach team performance trends (win rates, etc.)
def _attach_team_form(feats: pd.DataFrame) -> pd.DataFrame:
    ts = _load_team_season()
    if ts is None:
        return feats
    keep = [c for c in [
        "tmID", "year", "season_win_pct", "prev_season_win_pct_1",
        "prev_season_win_pct_3", "prev_season_win_pct_5",
        "win_pct_change_from_prev"
    ] if c in ts.columns]
    return feats.merge(ts[keep], on=["tmID", "year"], how="left")

# main script entry point
def main():
    players = _load_players()
    teams = _load_teams()
    pt = _load_players_teams()

    # label rookies and build team-level rookie features
    pst, player_rookie = _label_rookies(pt)
    feats = _team_rookie_features(pst, players)
    feats = _attach_team_names(feats, teams)
    feats = _attach_team_form(feats)

    # export both team-level and player-level outputs
    feats.to_csv(PROC_DIR / "team_rookie_features.csv", index=False)
    player_rookie.rename(columns={"rookie_year": "year"}).to_csv(
        PROC_DIR / "player_rookie_year.csv", index=False
    )

if __name__ == "__main__":
    main()
