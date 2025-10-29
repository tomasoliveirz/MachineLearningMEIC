"""
Coach performance calculation

This module computes a per-coach "performance" score by combining three signals:
	- Actual team success (win_pct), standardized by year -> z_win
	- Over/under performance relative to an expected win rate (pred_win) -> over -> z_over
		* pred_win is taken from team talent (average player performance) when available, otherwise from a Pythagorean expectation using points for/against.
	- Rookie development (average rookie performance), standardized by year -> z_rock

Final score:
	performance = z_win + z_over + 1.2 * z_rock

Inputs (expected files):
	- data/raw/coaches.csv                (required)  : coach-season records with won/lost
	- data/processed/teams_cleaned.csv    (optional)  : team-season PF/PA and confID
	- data/processed/player_performance.csv (optional): player-season performance, rookie flag

Outputs:
	- data/processed/coach_performance.csv : per-coach-season performance and basic stats
		(coachID, year, tmID, confID, won, lost, games, performance)

Notes:
	- Missing team or player data is handled permissively (falls back to Pythagorean or NaN).
	- Z-scores are computed separately for each year to control for season-level variance.
	- The module is deterministic and writes the resulting CSV to OUT_PATH.

"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

# Input paths
COACHES_PATH = RAW / "coaches.csv"
TEAMS_CLEANED_PATH = PROC / "teams_cleaned.csv"
TEAM_SEASON_PATH = PROC / "team_season.csv"
PLAYER_PERF_PATH = PROC / "player_performance.csv"

OUT_PATH = PROC / "coach_performance.csv"


def _safe_div(n: float, d: float) -> float:
    return float(n / d) if d and not np.isnan(d) else np.nan


def load_coaches():
    c = pd.read_csv(COACHES_PATH)
    c["year"] = pd.to_numeric(c["year"], errors="coerce")
    c["tmID"] = c["tmID"].astype(str)
    c["won"] = pd.to_numeric(c["won"], errors="coerce").fillna(0).astype(int)
    c["lost"] = pd.to_numeric(c["lost"], errors="coerce").fillna(0).astype(int)
    c["games"] = (c["won"] + c["lost"]).astype(float)
    c["win_pct"] = c.apply(lambda r: _safe_div(r["won"], r["games"]), axis=1)
    return c


def load_teams():
    if not TEAMS_CLEANED_PATH.exists():
        return None
    t = pd.read_csv(TEAMS_CLEANED_PATH)
    t["year"] = pd.to_numeric(t["year"], errors="coerce")
    t["tmID"] = t["tmID"].astype(str)

    # PF = pontos feitos, PA = pontos sofridos
    if {"o_pts", "d_pts"}.issubset(t.columns):
        t["PF"] = t["o_pts"]
        t["PA"] = t["d_pts"]
    else:
        t["PF"] = np.nan
        t["PA"] = np.nan

    keep = ["year", "tmID", "confID", "PF", "PA"]
    return t[keep]


def load_players():
    if not PLAYER_PERF_PATH.exists():
        return None
    p = pd.read_csv(PLAYER_PERF_PATH)
    p["year"] = pd.to_numeric(p["year"], errors="coerce")
    p["tmID"] = p["tmID"].astype(str)

    p["perf_val"] = pd.to_numeric(p["performance"], errors="coerce")

    p["is_rookie"] = p["rookie"].astype(bool)
    return p


def team_from_players(players):
    if players is None:
        return None

    perf = players.groupby(["year", "tmID"])["perf_val"].mean().reset_index()
    perf = perf.rename(columns={"perf_val": "talent_avg"})

    rook = players[players["is_rookie"]].groupby(["year", "tmID"])["perf_val"].mean().reset_index()
    rook = rook.rename(columns={"perf_val": "rookie_dev"})

    return perf.merge(rook, how="left", on=["year", "tmID"])


def pythagorean(PF, PA):
    exp = 13.91
    PF = pd.to_numeric(PF)
    PA = pd.to_numeric(PA)
    with np.errstate(divide="ignore", invalid="ignore"):
        pe = PF ** exp / (PF ** exp + PA ** exp)
    return pe


def compute_performance() -> pd.DataFrame:
    coaches = load_coaches()
    teams = load_teams()
    players = load_players()

    team_players = team_from_players(players)

    # win_pct_team por soma dos coaches (fallback)
    agg = coaches.groupby(["year", "tmID"], as_index=False)[["won", "lost"]].sum()
    agg["games"] = agg["won"] + agg["lost"]
    agg["win_pct_team"] = agg.apply(lambda r: _safe_div(r["won"], r["games"]), axis=1)

    if teams is not None:
        agg = agg.merge(teams, on=["year", "tmID"], how="left")
        agg["pyth_exp"] = pythagorean(agg["PF"], agg["PA"])
    else:
        agg["confID"] = np.nan
        agg["pyth_exp"] = np.nan

    base = coaches.merge(agg[["year", "tmID", "confID", "win_pct_team", "pyth_exp"]], on=["year", "tmID"], how="left")

    if team_players is not None:
        base = base.merge(team_players, on=["year", "tmID"], how="left")

    # Previsão de win rate:
    # - Usar Pythagorean expectation como baseline (baseado em PF/PA)
    # - Se não houver Pythagorean, usar win_pct_team como fallback
    base["pred_win"] = base["pyth_exp"] if "pyth_exp" in base.columns else np.nan
    base["pred_win"] = base["pred_win"].fillna(base["win_pct_team"])

    base["over"] = base["win_pct"] - base["pred_win"]

    def z(g):
        """Calcula z-score. Se std=0 ou todos NaN, retorna 0 para todos."""
        if g.isna().all():
            return pd.Series([0.0] * len(g), index=g.index)
        s = g.std(ddof=0)
        if s == 0 or pd.isna(s):
            return pd.Series([0.0] * len(g), index=g.index)
        return (g - g.mean()) / s

    base["z_win"] = base.groupby("year")["win_pct"].transform(z)
    base["z_over"] = base.groupby("year")["over"].transform(z)
    
    # Calcular z_rock apenas se a coluna rookie_dev existir e tiver valores
    if "rookie_dev" in base.columns:
        base["z_rock"] = base.groupby("year")["rookie_dev"].transform(z)
    else:
        base["z_rock"] = 0.0

    # Garantir que NaN seja substituído por 0 nos z-scores
    base["z_win"] = base["z_win"].fillna(0)
    base["z_over"] = base["z_over"].fillna(0)
    base["z_rock"] = base["z_rock"].fillna(0)

    base["performance"] = base["z_win"] + base["z_over"] + 1.2 * base["z_rock"]

    out = base[["coachID", "year", "tmID", "confID", "won", "lost", "games", "performance"]].copy()
    out.sort_values(["year", "coachID"], inplace=True)
    out.to_csv(OUT_PATH, index=False)
    return out


if __name__ == "__main__":
    df = compute_performance()
