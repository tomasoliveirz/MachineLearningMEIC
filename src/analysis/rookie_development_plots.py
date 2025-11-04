# src/analysis/rookie_development_plots.py
# -*- coding: utf-8 -*-
"""
plots-first rookie development (experience-based, no ages)

what this does
--------------
- loads player_performance.csv and players*.csv (to tag ncaa vs non_ncaa)
- builds per-player experience index: exp_year = 0 for rookie, 1, 2, ...
- aggregates clean, legible visuals:
  (1) mean ± se performance by experience (ncaa vs non_ncaa)
  (2) delta vs rookie baseline by experience (ncaa vs non_ncaa)
  (3) survival by experience (share of players still present)
  (4) rookie performance distribution (boxplot, ncaa vs non_ncaa)
- writes meta/rookies_correlations.txt with corr(perf, exp) overall and by origin
- no csv tables are written

notes
-----
- we never use birth years or ages; only experience inferred from rows order/rookie flag
- code/comments lowercase; plot texts sentence case
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path for imports
ROOT_PROJECT = Path(__file__).resolve().parents[2]
if str(ROOT_PROJECT) not in sys.path:
    sys.path.insert(0, str(ROOT_PROJECT))

# Import shared utilities
from src.utils.plots import ensure_dir, sentence_case, setup_plot_rcparams
from src.utils.players import infer_rookie_origin

# Setup consistent plot style
setup_plot_rcparams(dpi=140, title_size=14, label_size=12, tick_size=10)

# ---------- helpers ----------
# (ensure_dir, sentence_case, and infer_rookie_origin now imported from utils)

# Aliases for backward compatibility in this file
sc = sentence_case
origin_from_college = infer_rookie_origin

def load_perf(perf_path: Path) -> pd.DataFrame:
    """load player_performance with a couple of robust fallbacks."""
    candidates = [perf_path]
    if perf_path.name == "player_performance.csv":
        candidates.append(perf_path.with_name("player_perfomance.csv"))  # legacy typo
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(f"could not find {perf_path} (or legacy typo).")
    pp = pd.read_csv(path)

    # normalize ids/columns
    if "bioID" not in pp.columns and "playerID" in pp.columns:
        pp = pp.rename(columns={"playerID": "bioID"})
    if "performance" not in pp.columns:
        raise KeyError("player_performance needs a 'performance' column.")
    if "year" not in pp.columns:
        raise KeyError("player_performance needs a 'year' column (season ordering key).")
    if "rookie" not in pp.columns:
        # if rookie flag missing, we will infer rookie as the earliest row per player after sorting by 'year'
        pp["rookie"] = False

    # coerce basic types
    pp["year"] = pd.to_numeric(pp["year"], errors="coerce")
    pp["rookie"] = pp["rookie"].astype(bool)
    return pp

def load_players(players_path: Path) -> pd.DataFrame:
    """prefer cleaned; fall back to raw if needed."""
    # allow user to pass either cleaned or raw; if the path doesn't exist, try common fallbacks
    if players_path.exists():
        players = pd.read_csv(players_path)
    else:
        raw = Path("data/raw/players.csv")
        cleaned = Path("data/processed/players_cleaned.csv")
        if cleaned.exists():
            players = pd.read_csv(cleaned)
        elif raw.exists():
            players = pd.read_csv(raw)
        else:
            # create a minimal frame so downstream code still works (all non_ncaa)
            players = pd.DataFrame({"playerID": [], "college": []})

    if "playerID" not in players.columns and "bioID" in players.columns:
        players["playerID"] = players["bioID"]
    if "playerID" not in players.columns:
        # fabricate from any id-like column; last resort keeps merge from failing silently
        any_id = next((c for c in players.columns if "id" in str(c).lower()), None)
        if any_id:
            players["playerID"] = players[any_id].astype(str)
        else:
            players["playerID"] = np.nan
    return players

def build_experience(pp: pd.DataFrame) -> pd.DataFrame:
    """
    build exp_year for each player:
    - sort rows by 'year' within player
    - base exp on cumulative order
    - if a 'rookie' row exists, shift so rookie has exp=0
    - clamp to >=0 and keep as integer
    """
    pp = pp.sort_values(["bioID", "year"], kind="mergesort").copy()
    pp["exp_rank"] = pp.groupby("bioID").cumcount()  # 0,1,2,...
    # if rookie present, align that row to exp 0
    def _align(g):
        if g["rookie"].any():
            idx0 = g.index[g["rookie"]].min()
            shift = int(g.loc[idx0, "exp_rank"])
            g["exp_year"] = g["exp_rank"] - shift
        else:
            g["exp_year"] = g["exp_rank"]
        g["exp_year"] = g["exp_year"].clip(lower=0).astype(int)
        return g
    pp = pp.groupby("bioID", group_keys=False).apply(_align)
    return pp.drop(columns=["exp_rank"])

def restrict_experience(pp: pd.DataFrame, max_exp: int) -> pd.DataFrame:
    """keep only rows with exp_year in [0, max_exp]."""
    return pp.loc[(pp["exp_year"] >= 0) & (pp["exp_year"] <= max_exp)].copy()

def corr(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0, 1])

def mean_se_by_exp(df: pd.DataFrame, value_col: str, min_n: int) -> pd.DataFrame:
    g = df.groupby("exp_year")[value_col]
    out = g.agg(["count", "mean", "std"]).rename(columns={"count": "n"})
    out["se"] = out["std"] / np.sqrt(out["n"])
    out = out[out["n"] >= min_n].reset_index()
    return out

# ---------- plots ----------

def plot_mean_se_by_exp(pp, outdir, min_n):
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for origin in ["ncaa", "non_ncaa"]:
        sub = pp[pp["rookie_origin"] == origin]
        stats = mean_se_by_exp(sub, "performance", min_n=min_n)
        if len(stats) == 0:
            continue
        ax.errorbar(stats["exp_year"], stats["mean"], yerr=stats["se"],
                    fmt="-o", capsize=4, label=f"{origin} (n≥{min_n})")
    ax.set_xlabel(sc("experience (years since rookie)"))
    ax.set_ylabel(sc("mean performance ± se"))
    ax.set_title(sc("performance by experience — ncaa vs non_ncaa"))
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(outdir / "perf_by_experience_mean_se.png", bbox_inches="tight")
    plt.close(fig)

def plot_delta_vs_rookie(pp, outdir, min_n):
    # compute baseline per origin at exp 0
    base = (pp[pp["exp_year"] == 0]
            .groupby("rookie_origin")["performance"]
            .mean().rename("base")).to_dict()
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for origin in ["ncaa", "non_ncaa"]:
        sub = pp[pp["rookie_origin"] == origin].copy()
        if origin in base:
            sub["delta"] = sub["performance"] - base[origin]
        else:
            continue
        stats = mean_se_by_exp(sub, "delta", min_n=min_n)
        if len(stats) == 0:
            continue
        ax.errorbar(stats["exp_year"], stats["mean"], yerr=stats["se"],
                    fmt="-o", capsize=4, label=f"{origin} (n≥{min_n})")
    ax.set_xlabel(sc("experience (years since rookie)"))
    ax.set_ylabel(sc("delta vs rookie (performance)"))
    ax.set_title(sc("growth relative to rookie — ncaa vs non_ncaa"))
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(outdir / "delta_vs_rookie_by_experience.png", bbox_inches="tight")
    plt.close(fig)

def plot_survival(pp, outdir):
    # survival = players present at exp k / players present at exp 0 (per origin)
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for origin in ["ncaa", "non_ncaa"]:
        sub = pp[pp["rookie_origin"] == origin]
        base_n = sub[sub["exp_year"] == 0]["bioID"].nunique()
        if base_n == 0:
            continue
        counts = sub.groupby("exp_year")["bioID"].nunique().sort_index()
        surv = (counts / base_n).reset_index(name="survival")
        ax.plot(surv["exp_year"], surv["survival"], "-o", label=f"{origin} (base={base_n})")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(sc("experience (years since rookie)"))
    ax.set_ylabel(sc("share of players still active"))
    ax.set_title(sc("survival by experience — ncaa vs non_ncaa"))
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(outdir / "survival_by_experience.png", bbox_inches="tight")
    plt.close(fig)

def plot_rookie_distribution(pp, outdir):
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    data = [pp[(pp["exp_year"] == 0) & (pp["rookie_origin"] == o)]["performance"].dropna().values
            for o in ["ncaa", "non_ncaa"]]
    labels = [sc("ncaa"), sc("non_ncaa")]
    ax.boxplot(data, labels=labels, showfliers=True)
    ax.set_ylabel(sc("rookie performance"))
    ax.set_title(sc("rookie performance distribution"))
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(outdir / "rookie_performance_boxplot.png", bbox_inches="tight")
    plt.close(fig)

def write_correlations(pp, metadir):
    lines = []
    overall = corr(pp["exp_year"], pp["performance"])
    lines.append(f"overall corr(perf, exp): {overall:.3f}" if np.isfinite(overall) else "overall corr(perf, exp): nan")
    for origin in ["ncaa", "non_ncaa"]:
        sub = pp[pp["rookie_origin"] == origin]
        r = corr(sub["exp_year"], sub["performance"])
        lines.append(f"{origin} corr(perf, exp): {r:.3f}" if np.isfinite(r) else f"{origin} corr(perf, exp): nan")
    ensure_dir(metadir).joinpath("rookies_correlations.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="plots-first rookie development (experience-based, no ages).")
    ap.add_argument("--perf", default="data/processed/player_performance.csv", help="path to player_performance.csv")
    ap.add_argument("--players", default="data/processed/players_cleaned.csv", help="path to players file (for college)")
    ap.add_argument("--outdir", default="reports/rookies", help="output directory for figures/meta")
    ap.add_argument("--max_exp", type=int, default=10, help="keep exp_year in [0, max_exp]")
    ap.add_argument("--min_n", type=int, default=5, help="min group size to draw mean±se points")
    args = ap.parse_args()

    outdir = ensure_dir(Path(args.outdir))
    figdir = ensure_dir(outdir / "figures")
    metadir = ensure_dir(outdir / "meta")

    pp = load_perf(Path(args.perf))
    players = load_players(Path(args.players))

    # add origin and unify ids for merge
    players["rookie_origin"] = origin_from_college(players)
    if "playerID" not in players.columns and "bioID" in players.columns:
        players["playerID"] = players["bioID"]

    # experience index
    pp = build_experience(pp)
    pp = pp.rename(columns={"bioID": "playerID"})  # align id for merge
    pp = pp.merge(players[["playerID", "rookie_origin"]], on="playerID", how="left")
    pp["rookie_origin"] = pp["rookie_origin"].fillna("non_ncaa")
    pp = pp.rename(columns={"playerID": "bioID"})  # restore name for consistency
    pp = restrict_experience(pp, args.max_exp)

    # plots
    plot_mean_se_by_exp(pp, figdir, min_n=args.min_n)
    plot_delta_vs_rookie(pp, figdir, min_n=args.min_n)
    plot_survival(pp, figdir)
    plot_rookie_distribution(pp, figdir)

    # correlations meta
    write_correlations(pp, metadir)

if __name__ == "__main__":
    main()
