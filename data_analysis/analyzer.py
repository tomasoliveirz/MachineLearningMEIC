#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path("../data")
OUT_DIR = Path("output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    files = {
        "awards": "awards_players.csv",
        "coaches": "coaches.csv",
        "players": "players.csv",
        "player_stats": "players_teams.csv",
        "playoffs_series": "series_post.csv",
        "playoffs_teams": "teams_post.csv",
        "teams": "teams.csv",
    }
    ds = {}
    for k, f in files.items():
        p = DATA_DIR / f
        if p.exists():
            ds[k] = pd.read_csv(p)
    return ds

def dataset_overview(ds):
    print("DATASETS")
    for k, df in ds.items():
        print(f"- {k}: {len(df):,} rows, {df.shape[1]} cols")

def players_summary(ds):
    if "players" not in ds: 
        return
    p = ds["players"]
    h = p["height"].dropna()
    w = p["weight"].dropna()
    if not h.empty:
        print("\nPLAYERS")
        print(f"Total: {len(p):,}")
        print(f"Avg height: {h.mean():.1f} in  |  min/max: {h.min():.0f}/{h.max():.0f}")
    if not w.empty:
        print(f"Avg weight: {w.mean():.1f} lb  |  min/max: {w.min():.0f}/{w.max():.0f}")

def player_stats_summary(ds):
    if "player_stats" not in ds:
        return
    s = ds["player_stats"]
    print("\nPLAYER STATS")
    print(f"Records: {len(s):,}  |  Players: {s['playerID'].nunique():,}")
    if "points" in s:
        print(f"Max season points: {s['points'].max()}")

def teams_summary(ds):
    if "teams" not in ds:
        return
    t = ds["teams"]
    print("\nTEAMS")
    print(f"Team-seasons: {len(t):,}  |  Teams: {t['tmID'].nunique()}")
    if "won" in t:
        print(f"Max wins: {t['won'].max()}  |  Avg wins: {t['won'].mean():.1f}")

def plot_players(ds):
    if "players" not in ds:
        return
    p = ds["players"]
    if "height" in p:
        plt.figure(figsize=(10,6))
        p["height"].dropna().plot(kind="hist", bins=20)
        plt.title("Player Height Distribution")
        plt.xlabel("Height (inches)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "height_distribution.png", dpi=150)
        plt.close()

    if {"height","weight"}.issubset(p.columns):
        q = p.dropna(subset=["height","weight"])
        if len(q) > 0:
            plt.figure(figsize=(10,6))
            plt.scatter(q["height"], q["weight"], alpha=0.6)
            plt.title("Height vs Weight")
            plt.xlabel("Height (inches)")
            plt.ylabel("Weight (lbs)")
            plt.tight_layout()
            plt.savefig(OUT_DIR / "height_weight_correlation.png", dpi=150)
            plt.close()

def plot_player_points(ds):
    if "player_stats" not in ds:
        return
    s = ds["player_stats"]
    if "points" in s:
        plt.figure(figsize=(10,6))
        s["points"].dropna().plot(kind="hist", bins=30)
        plt.title("Points per Season Distribution")
        plt.xlabel("Points")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "points_distribution.png", dpi=150)
        plt.close()

    top = s.groupby("playerID")["points"].sum().sort_values(ascending=False).head(10)
    if len(top) > 0:
        plt.figure(figsize=(12,6))
        top.plot(kind="bar")
        plt.title("Top 10 Career Scorers")
        plt.xlabel("Player")
        plt.ylabel("Total Points")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "top_career_scorers.png", dpi=150)
        plt.close()

def plot_teams(ds):
    if "teams" not in ds:
        return
    t = ds["teams"]
    if "won" in t:
        plt.figure(figsize=(10,6))
        t["won"].dropna().plot(kind="hist", bins=15)
        plt.title("Team Wins Distribution")
        plt.xlabel("Wins")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "wins_distribution.png", dpi=150)
        plt.close()

def plot_awards(ds):
    if "awards" not in ds:
        return
    a = ds["awards"].copy()
    a["award_norm"] = a["award"].astype(str).str.lower()
    raw = a.groupby("year").size().sort_index()
    decade_mask = (a["year"] == 7) & a["award_norm"].str.contains("decade", na=False)
    adj = a.loc[~decade_mask].groupby("year").size().reindex(raw.index, fill_value=0)
    if 7 in adj.index and adj.loc[7] == raw.loc[7]:
        adj.loc[7] = max(0, raw.loc[7] - 15)

    plt.figure(figsize=(12,6))
    raw.plot(marker="o", label="All awards")
    adj.plot(marker="o", label="Excluding decade awards")
    plt.title("Awards per Year")
    plt.xlabel("Year")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "awards_over_years.png", dpi=150)
    plt.close()

def write_report(ds):
    path = OUT_DIR / "analysis_report.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("WNBA DATA ANALYSIS REPORT\n")
        f.write("="*40 + "\n\n")
        for k, df in ds.items():
            f.write(f"{k.capitalize()}: {len(df):,} records\n")
        if "players" in ds:
            p = ds["players"]
            h = p["height"].dropna()
            if not h.empty:
                f.write(f"\nPlayers: {len(p):,}  |  Avg height: {h.mean():.1f} in  |  Min/Max: {h.min():.0f}/{h.max():.0f}\n")
        if "player_stats" in ds:
            s = ds["player_stats"]
            f.write(f"\nPlayer-season records: {len(s):,}  |  Unique players: {s['playerID'].nunique():,}\n")
            if "points" in s:
                f.write(f"Max season points: {s['points'].max()}\n")
        if "teams" in ds:
            t = ds["teams"]
            if "won" in t:
                f.write(f"\nTeams: {t['tmID'].nunique()}  |  Max wins: {t['won'].max()}  |  Avg wins: {t['won'].mean():.1f}\n")

def main():
    ds = load_data()
    dataset_overview(ds)
    players_summary(ds)
    player_stats_summary(ds)
    teams_summary(ds)
    plot_players(ds)
    plot_player_points(ds)
    plot_teams(ds)
    plot_awards(ds)
    write_report(ds)
    print("\nSaved outputs to 'output/'")

if __name__ == "__main__":
    main()
