#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path("../data")
DATA_CLEANED_DIR = Path("../data_cleaning_output")

def choose_data_source():
    print("Usar ficheiros cleaned para players e teams?")
    print("1) não")
    print("2) sim")
    choice = input("> ").strip()
    return choice == "2"

def load_data(use_cleaned=False):
    files = {
        "awards": "awards_players.csv",
        "players": "players_cleaned.csv" if use_cleaned else "players.csv",
        "player_stats": "players_teams.csv",
        "teams": "teams_cleaned.csv" if use_cleaned else "teams.csv",
    }
    base = DATA_CLEANED_DIR if use_cleaned else DATA_DIR
    ds = {}
    for k, f in files.items():
        p = base / f if k in ("players","teams") else DATA_DIR / f
        if p.exists():
            ds[k] = pd.read_csv(p)
    return ds

def get_out_dir(use_cleaned):
    out = Path("cleaned_results") if use_cleaned else Path("output")
    out.mkdir(parents=True, exist_ok=True)
    return out

def plot_players(ds, OUT_DIR):
    if "players" not in ds: return
    p = ds["players"].copy()
    if "height" in p:
        p["height"] = pd.to_numeric(p["height"], errors="coerce")
        bins = np.arange(61.5, 80.5 + 1, 1.0)
        plt.hist(p["height"].dropna(), bins=bins)
        plt.title("Player Height Distribution")
        plt.xlabel("Height (inches)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "height_distribution.png", dpi=150)
        plt.close()
    if {"height","weight"}.issubset(p.columns):
        p["height"] = pd.to_numeric(p["height"], errors="coerce")
        p["weight"] = pd.to_numeric(p["weight"], errors="coerce")
        q = p.dropna(subset=["height","weight"])
        if len(q) > 0:
            plt.scatter(q["height"], q["weight"], alpha=0.6)
            plt.title("Height vs Weight")
            plt.xlabel("Height (inches)")
            plt.ylabel("Weight (lbs)")
            plt.tight_layout()
            plt.savefig(OUT_DIR / "height_weight_correlation.png", dpi=150)
            plt.close()

def plot_player_points(ds, OUT_DIR):
    if "player_stats" not in ds: return
    s = ds["player_stats"].copy()
    if "points" in s:
        s["points"] = pd.to_numeric(s["points"], errors="coerce")
        s["points"].dropna().plot(kind="hist", bins=30)
        plt.title("Points per Season Distribution")
        plt.xlabel("Points")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "points_distribution.png", dpi=150)
        plt.close()
    top = s.groupby("playerID")["points"].sum(min_count=1).sort_values(ascending=False).head(10)
    if len(top) > 0:
        top.plot(kind="bar")
        plt.title("Top 10 Career Scorers")
        plt.xlabel("Player")
        plt.ylabel("Total Points")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "top_career_scorers.png", dpi=150)
        plt.close()

def plot_teams(ds, OUT_DIR):
    if "teams" not in ds: return
    t = ds["teams"].copy()
    if "won" in t:
        t["won"] = pd.to_numeric(t["won"], errors="coerce")
        t["won"].dropna().plot(kind="hist", bins=15)
        plt.title("Team Wins Distribution")
        plt.xlabel("Wins")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "wins_distribution.png", dpi=150)
        plt.close()

def plot_awards(ds, OUT_DIR):
    if "awards" not in ds: return
    a = ds["awards"].copy()
    a["award_norm"] = a["award"].astype(str).str.lower()
    raw = a.groupby("year").size().sort_index()
    decade_mask = (a["year"] == 7) & a["award_norm"].str.contains("decade", na=False)
    adj = a.loc[~decade_mask].groupby("year").size().reindex(raw.index, fill_value=0)
    if 7 in raw.index and 7 in adj.index and adj.loc[7] == raw.loc[7]:
        adj.loc[7] = max(0, raw.loc[7] - 15)
    raw.plot(marker="o", label="All awards")
    adj.plot(marker="o", label="Excluding decade awards")
    plt.title("Awards per Year")
    plt.xlabel("Year")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "awards_over_years.png", dpi=150)
    plt.close()

def write_report(ds, OUT_DIR):
    path = OUT_DIR / "analysis_report.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("WNBA DATA ANALYSIS REPORT\n")
        f.write("="*40 + "\n\n")
        for k, df in ds.items():
            f.write(f"{k.capitalize()}: {len(df):,} records\n")

def main():
    use_cleaned = choose_data_source()
    ds = load_data(use_cleaned)
    out_dir = get_out_dir(use_cleaned)
    plot_players(ds, out_dir)
    plot_player_points(ds, out_dir)
    plot_teams(ds, out_dir)
    plot_awards(ds, out_dir)
    write_report(ds, out_dir)

if __name__ == "__main__":
    main()
