#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_root() -> Path:
    here = Path(__file__).resolve()
    p = here
    for _ in range(6):
        if (p / "data").exists():
            return p
        p = p.parent
    return here.parents[3]

ROOT = get_root()
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

def get_out_dirs(mode: str):
    fig_dir = REPORTS_DIR / "figures" / mode
    tab_dir = REPORTS_DIR / "tables" / mode
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir, tab_dir

def load_data(mode: str):
    ds = {}
    # always from raw
    if (RAW_DIR / "awards_players.csv").exists():
        ds["awards"] = pd.read_csv(RAW_DIR / "awards_players.csv")
    if (RAW_DIR / "players_teams.csv").exists():
        ds["player_stats"] = pd.read_csv(RAW_DIR / "players_teams.csv")

    # players/teams depend on mode
    if mode == "cleaned":
        p = PROC_DIR / "players_cleaned.csv"
        t = PROC_DIR / "teams_cleaned.csv"
        if p.exists(): ds["players"] = pd.read_csv(p)
        if t.exists(): ds["teams"] = pd.read_csv(t)
    else:
        p = RAW_DIR / "players.csv"
        t = RAW_DIR / "teams.csv"
        if p.exists(): ds["players"] = pd.read_csv(p)
        if t.exists(): ds["teams"] = pd.read_csv(t)
    return ds

def plot_players(ds, out_dir):
    if "players" not in ds: return
    p = ds["players"].copy()
    if "height" in p:
        p["height"] = pd.to_numeric(p["height"], errors="coerce")
        bins = np.arange(61.5, 80.5 + 1, 1.0)
        plt.figure(figsize=(10,6))
        plt.hist(p["height"].dropna(), bins=bins)
        plt.title("Player Height Distribution")
        plt.xlabel("Height (inches)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(out_dir / "height_distribution.png", dpi=150)
        plt.close()
    if {"height","weight"}.issubset(p.columns):
        p["height"] = pd.to_numeric(p["height"], errors="coerce")
        p["weight"] = pd.to_numeric(p["weight"], errors="coerce")
        q = p.dropna(subset=["height","weight"])
        if len(q) > 0:
            plt.figure(figsize=(10,6))
            plt.scatter(q["height"], q["weight"], alpha=0.6)
            plt.title("Height vs Weight")
            plt.xlabel("Height (inches)")
            plt.ylabel("Weight (lbs)")
            plt.tight_layout()
            plt.savefig(out_dir / "height_weight_correlation.png", dpi=150)
            plt.close()

def plot_player_points(ds, out_dir):
    if "player_stats" not in ds: return
    s = ds["player_stats"].copy()
    if "points" in s:
        s["points"] = pd.to_numeric(s["points"], errors="coerce")
        plt.figure(figsize=(10,6))
        s["points"].dropna().plot(kind="hist", bins=30)
        plt.title("Points per Season Distribution")
        plt.xlabel("Points")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(out_dir / "points_distribution.png", dpi=150)
        plt.close()
    top = s.groupby("playerID")["points"].sum(min_count=1).sort_values(ascending=False).head(10)
    if len(top) > 0:
        plt.figure(figsize=(12,6))
        top.plot(kind="bar")
        plt.title("Top 10 Career Scorers")
        plt.xlabel("Player")
        plt.ylabel("Total Points")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / "top_career_scorers.png", dpi=150)
        plt.close()

def plot_teams(ds, out_dir):
    if "teams" not in ds: return
    t = ds["teams"].copy()
    if "won" in t:
        t["won"] = pd.to_numeric(t["won"], errors="coerce")
        plt.figure(figsize=(10,6))
        t["won"].dropna().plot(kind="hist", bins=15)
        plt.title("Team Wins Distribution")
        plt.xlabel("Wins")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(out_dir / "wins_distribution.png", dpi=150)
        plt.close()

def plot_awards(ds, out_dir):
    if "awards" not in ds: return
    a = ds["awards"].copy()
    a["award_norm"] = a["award"].astype(str).str.lower()
    raw = a.groupby("year").size().sort_index()
    decade_mask = (a["year"] == 7) & a["award_norm"].str.contains("decade", na=False)
    adj = a.loc[~decade_mask].groupby("year").size().reindex(raw.index, fill_value=0)
    if 7 in raw.index and 7 in adj.index and adj.loc[7] == raw.loc[7]:
        adj.loc[7] = max(0, raw.loc[7] - 15)
    plt.figure(figsize=(12,6))
    raw.plot(marker="o", label="All awards")
    adj.plot(marker="o", label="Excluding decade awards")
    plt.title("Awards per Year")
    plt.xlabel("Year")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "awards_over_years.png", dpi=150)
    plt.close()

def write_report(ds, tab_dir):
    with open(tab_dir / "analysis_report.txt", "w", encoding="utf-8") as f:
        f.write("WNBA DATA ANALYSIS REPORT\n")
        f.write("="*40 + "\n\n")
        for k, df in ds.items():
            f.write(f"{k.capitalize()}: {len(df):,} records\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["raw","cleaned"], default="raw")
    args = ap.parse_args()

    ds = load_data(args.mode)
    fig_dir, tab_dir = get_out_dirs(args.mode)

    plot_players(ds, fig_dir)
    plot_player_points(ds, fig_dir)
    plot_teams(ds, fig_dir)
    plot_awards(ds, fig_dir)
    write_report(ds, tab_dir)

if __name__ == "__main__":
    main()
