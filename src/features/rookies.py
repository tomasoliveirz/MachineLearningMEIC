#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np

# setup basic folder structure
ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# load players data (prefer the cleaned version if available)
def _load_players():
    p_clean = PROC_DIR / "players_cleaned.csv"
    if p_clean.exists():
        return pd.read_csv(p_clean)
    elif p_raw.exists():
        return pd.read_csv(p_raw)
    else:
        raise FileNotFoundError("players.csv not found in data/processed or data/raw")

# load teams data (used to attach team names later)
def _load_teams():
    t_clean = PROC_DIR / "teams_cleaned.csv"
    t_raw = RAW_DIR / "teams.csv"
    if t_clean.exists():
        return pd.read_csv(t_clean)
    elif t_raw.exists():
        return pd.read_csv(t_raw)
    else:
        return None

# load player-team-season links
def _load_players_teams():
    pt = RAW_DIR / "players_teams.csv"
    if not pt.exists():
        raise FileNotFoundError("players_teams.csv not found in data/raw")
    return pd.read_csv(pt)

# load team season metrics (used to add form variables later)
def _load_team_season():
    ts = PROC_DIR / "team_season.csv"
    return pd.read_csv(ts) if ts.exists() else None

# label which rows correspond to rookie seasons
def _label_rookies(players_teams: pd.DataFrame) -> pd.DataFrame:
    df = players_teams.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # find the first season for each player
    rookie_year = df.groupby("playerID", dropna=False)["year"].min().rename("rookie_year")
    df = df.merge(rookie_year, on="playerID", how="left")
    df["is_rookie"] = (df["year"] == df["rookie_year"]).astype(int)

    # merge multiple stints within the same season/team
    for c in ["minutes", "points"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    agg = (
        df.groupby(["playerID", "year", "tmID"], dropna=False)[["minutes", "points", "is_rookie"]]
          .sum(min_count=1)
          .reset_index()
    )
    agg["is_rookie"] = (agg["is_rookie"] > 0).astype(int)
    return agg, rookie_year.reset_index()

# detect player origin (ncaa vs non-ncaa) based on college info
def _rookie_origin(players: pd.DataFrame) -> pd.Series:
    col = "college"
    if col not in players.columns:
        return pd.Series(index=players.index, dtype="object")
    s = players[col].astype(str).str.strip().str.lower()
    is_unknown = s.isna() | s.eq("nan") | s.eq("unknown") | s.eq("")
    return np.where(is_unknown, "non_ncaa", "ncaa")

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
