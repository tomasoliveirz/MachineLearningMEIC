"""
plots-first analysis of wnba coaches

what this does (plots only)
---------------------------
- loads season-level rows per coach and merges team-season context
- computes per season/career:
  regular/postseason win%, empirical-bayes (eb) career win% (+~95% ci),
  pythagorean expectation and overachieve (rs% - pythag),
  first-year impact vs the same team's previous season,
  consistency (sd of rs%) and trend (slope rs% ~ season number),
  rs->po career gap (po% - rs%)
- renders only legible plots (no csvs):
  top eb bar, rs->po slopegraph, small-multiples by longevity,
  overachievers bar, first-year histogram, stint averages,
  correlations (rs vs po scatter + correlation matrix)
- saves text meta: correlations.txt

notes
-----
- stint: 0=full season, 1=hired mid-season, 2=left mid-season
- only matplotlib; no seaborn
- all code/comments are lowercase; all plot texts are sentence case
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path for imports
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import shared utilities
from src.utils.plots import ensure_dir, sentence_case, setup_plot_rcparams

# Setup consistent plot style
setup_plot_rcparams(dpi=140, title_size=14, label_size=12, tick_size=10)

# ---------- helpers ----------
# (ensure_dir and sentence_case now imported from utils.plots)

# Alias for backward compatibility in this file
sc = sentence_case

def pythag_win_pct(points_for_pg, points_against_pg, exponent=13.91):
    """pythagorean expectation using per-game points."""
    pf = np.asarray(points_for_pg, dtype=float)
    pa = np.asarray(points_against_pg, dtype=float)
    out = np.full_like(pf, np.nan, dtype=float)
    mask = (pf > 0) & (pa > 0)
    if mask.any():
        out[mask] = (pf[mask] ** exponent) / ((pf[mask] ** exponent) + (pa[mask] ** exponent))
    return out

def beta_post_mean_ci(wins, games, prior_mean=0.5, prior_games=20):
    """conjugate beta posterior for a binomial proportion with normal ci (visual-grade)."""
    wins = float(wins)
    games = float(games)
    a0 = prior_mean * prior_games
    b0 = (1 - prior_mean) * prior_games
    a = a0 + wins
    b = b0 + (games - wins)
    mean = a / (a + b)
    var = (a * b) / (((a + b) ** 2) * (a + b + 1))
    se = np.sqrt(var)
    z = 1.959963984540054
    lo = max(0.0, mean - z * se)
    hi = min(1.0, mean + z * se)
    return mean, lo, hi

def detect_points_columns(df: pd.DataFrame):
    """try to discover points-for/against columns in teams.csv; return (pf_total, pa_total) or (none, none)."""
    candidates_pf = ["o_pts", "pts", "pf", "points_for", "tm_pts", "tmpts"]
    candidates_pa = ["d_pts", "opp_pts", "pa", "points_against", "opp_pts_total", "opppts"]
    pf_col = next((c for c in candidates_pf if c in df.columns), None)
    pa_col = next((c for c in candidates_pa if c in df.columns), None)
    return pf_col, pa_col

def fig_save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

# ---------- etl ----------

def load_coach_seasons(coaches_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(coaches_csv)
    df["games"] = df["won"].fillna(0) + df["lost"].fillna(0)
    df["reg_win_pct"] = np.where(df["games"] > 0, df["won"] / df["games"], np.nan)
    df["post_games"] = df["post_wins"].fillna(0) + df["post_losses"].fillna(0)
    df["post_win_pct"] = np.where(df["post_games"] > 0, df["post_wins"] / df["post_games"], np.nan)
    df = df.sort_values(["coachID", "year", "stint"]).reset_index(drop=True)
    df["season_no"] = df.groupby("coachID").cumcount() + 1
    return df

def load_team_seasons(teams_csv: Path) -> pd.DataFrame:
    t = pd.read_csv(teams_csv)
    if not {"year", "tmID"}.issubset(set(t.columns)):
        raise ValueError("teams.csv must contain 'year' and 'tmID'")
    if {"won", "lost"}.issubset(set(t.columns)):
        t["gp"] = t["won"] + t["lost"]
        t["reg_win_pct_team"] = np.where(t["gp"] > 0, t["won"] / t["gp"], np.nan)
    else:
        t["gp"] = np.nan
        t["reg_win_pct_team"] = np.nan
    pf_col, pa_col = detect_points_columns(t)
    if pf_col and pa_col:
        if "gp" in t.columns and t["gp"].notna().any():
            t["pf_pg"] = t[pf_col] / t["gp"]
            t["pa_pg"] = t[pa_col] / t["gp"]
        else:
            t["pf_pg"] = t[pf_col]
            t["pa_pg"] = t[pa_col]
        t["pyth_win_pct"] = pythag_win_pct(t["pf_pg"], t["pa_pg"])
        t["overachieve_team"] = t["reg_win_pct_team"] - t["pyth_win_pct"]
    else:
        t["pyth_win_pct"] = np.nan
        t["overachieve_team"] = np.nan
    keep = ["year", "tmID", "gp", "reg_win_pct_team", "pyth_win_pct", "overachieve_team"]
    return t[keep]

def merge_context(coach_seasons: pd.DataFrame, team_seasons: pd.DataFrame) -> pd.DataFrame:
    return coach_seasons.merge(team_seasons, on=["year", "tmID"], how="left")

# ---------- features ----------

def compute_first_year_deltas(df: pd.DataFrame) -> pd.DataFrame:
    prev = df[["tmID", "year", "reg_win_pct_team"]].dropna().copy()
    prev["year"] = prev["year"] + 1
    prev = prev.rename(columns={"reg_win_pct_team": "team_prev_win_pct"})
    df = df.merge(prev, on=["tmID", "year"], how="left")
    df["team_season_rank"] = df.sort_values(["year", "stint"]).groupby(["coachID", "tmID"]).cumcount() + 1
    is_first = df["team_season_rank"] == 1
    df["first_year_delta"] = np.where(is_first, df["reg_win_pct"] - df["team_prev_win_pct"], np.nan)
    return df

def summarize_careers(df: pd.DataFrame) -> pd.DataFrame:
    global_prior = df.loc[df["games"] > 0, "reg_win_pct"].mean()
    if not np.isfinite(global_prior):
        global_prior = 0.5
    rows = []
    for coach, g in df.groupby("coachID"):
        tw = float(g["won"].sum())
        tg = float(g["games"].sum())
        tpw = float(g["post_wins"].sum())
        tpg = float(g["post_games"].sum())
        reg = (tw / tg) if tg > 0 else np.nan
        post = (tpw / tpg) if tpg > 0 else np.nan
        eb_mean, eb_lo, eb_hi = beta_post_mean_ci(tw, tg, prior_mean=global_prior, prior_games=20)
        if g["overachieve_team"].notna().any():
            oa = np.average(g["overachieve_team"].fillna(np.nan), weights=g["games"].fillna(0))
        else:
            oa = np.nan
        reg_series = g.loc[g["reg_win_pct"].notna(), ["season_no", "reg_win_pct"]]
        if len(reg_series) >= 2:
            slope = np.polyfit(reg_series["season_no"], reg_series["reg_win_pct"], 1)[0]
            sd = reg_series["reg_win_pct"].std(ddof=0)
        elif len(reg_series) == 1:
            slope = np.nan
            sd = 0.0
        else:
            slope = np.nan
            sd = np.nan
        fy = g.loc[g["first_year_delta"].notna(), "first_year_delta"]
        fy_mean = float(fy.mean()) if len(fy) else np.nan
        rows.append({
            "coachID": coach,
            "seasons": int(g["year"].nunique()),
            "teams": int(g["tmID"].nunique()),
            "total_wins": int(tw),
            "total_losses": int(g["lost"].sum()),
            "total_games": int(tg),
            "career_reg_win_pct": reg,
            "playoff_seasons": int((g["post_games"] > 0).sum()),
            "career_post_games": int(tpg),
            "career_post_win_pct": post,
            "career_eb_win_pct": eb_mean,
            "career_eb_lo": eb_lo,
            "career_eb_hi": eb_hi,
            "career_overachieve": oa,
            "consistency_sd": sd,
            "trend_slope": slope,
            "first_year_delta_mean": fy_mean,
            "avg_reg_win_pct": float(g["reg_win_pct"].mean()),
            "rs_to_po_gap": (post - reg) if np.isfinite(post) and np.isfinite(reg) else np.nan,
        })
    return pd.DataFrame(rows).sort_values(
        ["career_eb_win_pct", "total_games"], ascending=[False, False]
    ).reset_index(drop=True)

# ---------- plots ----------

def plot_topk_bar_eb(wg: pd.DataFrame, figdir: Path, topk=15, min_games=68):
    x = wg.loc[wg["total_games"] >= min_games].copy()
    x = x.sort_values(["career_eb_win_pct", "total_games"], ascending=[False, False]).head(topk)
    if x.empty:
        return
    idx = np.arange(len(x))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(idx, x["career_eb_win_pct"])
    err = np.vstack([x["career_eb_win_pct"] - x["career_eb_lo"], x["career_eb_hi"] - x["career_eb_win_pct"]])
    ax.errorbar(idx, x["career_eb_win_pct"], yerr=err, fmt="none", capsize=4, lw=1)
    ax.set_xticks(idx)
    # keep coach ids as-is (codes), not sentence-cased
    ax.set_xticklabels(x["coachID"], rotation=45, ha="right")
    ax.set_ylabel(sc("career eb win% (posterior)"))
    ax.set_title(sc(f"top {len(x)} coaches — empirical-bayes career win% (min {min_games} games)"))
    fig_save(fig, figdir / "topK_bar_eb_career.png")

def plot_slopegraph_rs_po(wg: pd.DataFrame, figdir: Path, topk=15, min_games=68):
    x = wg.loc[(wg["total_games"] >= min_games) & (wg["career_post_win_pct"].notna())].copy()
    if x.empty:
        return
    x = x.sort_values(["total_games", "career_reg_win_pct"], ascending=[False, False]).head(topk)
    fig, ax = plt.subplots(figsize=(8, 7))
    for _, r in x.reset_index(drop=True).iterrows():
        ax.plot([0, 1], [r["career_reg_win_pct"], r["career_post_win_pct"]], marker="o", lw=1.8)
        ax.text(-0.02, r["career_reg_win_pct"], r["coachID"], ha="right", va="center", fontsize=9)
        ax.text(1.02, r["career_post_win_pct"], f"{r['career_post_win_pct']:.2f}", ha="left", va="center", fontsize=9)
    ax.set_xlim(-0.2, 1.2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([sc("regular-season"), sc("playoffs")])
    ax.set_ylabel(sc("career win%"))
    ax.set_title(sc(f"does regular-season success translate to playoffs? (top {len(x)} by games)"))
    ax.grid(axis="y", alpha=0.2)
    fig_save(fig, figdir / "slopegraph_rs_po_topK.png")

def plot_small_multiples(df: pd.DataFrame, figdir: Path, topn=12, min_games=68):
    agg = df.groupby("coachID").agg(total_games=("games", "sum"), seasons=("year", "nunique"))
    keep = (agg.query("total_games >= @min_games")
                .sort_values(["seasons", "total_games"], ascending=[False, False])
                .head(topn)
                .index.tolist())
    z = df[df["coachID"].isin(keep)].copy()
    if z.empty:
        return
    n = len(keep)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.6, rows * 2.6), sharey=True)
    axes = axes.ravel()
    for i, cid in enumerate(keep):
        ax = axes[i]
        g = z.loc[z["coachID"] == cid].sort_values("season_no")
        ax.plot(g["season_no"], g["reg_win_pct"], marker="o")
        # keep id literal; axes labels/titles sentence-cased
        ax.set_title(cid, fontsize=10)
        ax.set_xlabel(sc("season #"))
        ax.set_ylabel(sc("rs win%"))
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.2)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle(sc("regular-season win% across seasons (top-12 by longevity)"), y=1.02, fontsize=14)
    fig.tight_layout()
    fig_save(fig, figdir / "small_multiples_top12.png")

def plot_overachievers(wg: pd.DataFrame, figdir: Path, topk=15, min_games=68):
    x = (wg.loc[(wg["total_games"] >= min_games) & (wg["career_overachieve"].notna())]
           .sort_values("career_overachieve", ascending=False)
           .head(topk))
    if x.empty:
        return
    idx = np.arange(len(x))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(idx, x["career_overachieve"])
    ax.set_xticks(idx)
    ax.set_xticklabels(x["coachID"], rotation=45, ha="right")
    ax.axhline(0, ls="--", lw=1)
    ax.set_ylabel(sc("average overachieve (rs win% − pythag)"))
    ax.set_title(sc(f"who beats expectation? (top {len(x)}, min {min_games} games)"))
    fig_save(fig, figdir / "bar_overachievers.png")

def plot_hist_first_year(df: pd.DataFrame, figdir: Path):
    x = df.loc[df["first_year_delta"].notna(), "first_year_delta"]
    if x.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(x, bins=14)
    ax.axvline(0, ls="--", lw=1)
    ax.set_xlabel(sc("Δ rs win% in year 1 vs the team's previous season"))
    ax.set_ylabel(sc("frequency"))
    ax.set_title(sc("immediate impact when a coach joins a team (coach-team first seasons)"))
    fig_save(fig, figdir / "hist_first_year_delta.png")

def plot_stint_averages(df: pd.DataFrame, figdir: Path):
    g = df.loc[df["reg_win_pct"].notna()].groupby("stint")["reg_win_pct"]
    means = g.mean()
    counts = g.count()
    stds = g.std(ddof=0)
    sem = stds / np.sqrt(counts.replace(0, np.nan))
    order = sorted(means.index)
    y = [means[s] for s in order]
    yerr = [sem[s] if np.isfinite(sem.get(s, np.nan)) else 0.0 for s in order]
    labels = {0: "0 • full season", 1: "1 • hired mid", 2: "2 • left mid"}
    xticks = [labels.get(s, str(s)) for s in order]
    fig, ax = plt.subplots(figsize=(7, 5))
    idx = np.arange(len(order))
    ax.bar(idx, y, yerr=yerr, capsize=4)
    for i, s in enumerate(order):
        ax.text(i, y[i] + (0.01 if np.isfinite(y[i]) else 0.0), f"n={int(counts[s])}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(idx)
    ax.set_xticklabels([sc(t) for t in xticks], rotation=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel(sc("mean rs win%"))
    ax.set_title(sc("regular-season win% by stint"))
    ax.axhline(0.5, ls="--", lw=1, alpha=0.4)
    fig_save(fig, figdir / "stint_avg_winpct.png")

def plot_correlations(wg: pd.DataFrame, figdir: Path, metadir: Path):
    z = wg.loc[wg["career_post_win_pct"].notna(), ["career_reg_win_pct", "career_post_win_pct", "career_post_games"]].dropna()
    meta_lines = []
    if len(z) >= 3:
        pear = float(z.corr(method="pearson").iloc[0, 1])
        spear = float(z.corr(method="spearman").iloc[0, 1])
        meta_lines.append(f"pearson rs vs po (career): {pear:.3f}")
        meta_lines.append(f"spearman rs vs po (career): {spear:.3f}")
        ensure_dir(metadir).joinpath("correlations.txt").write_text("\n".join(meta_lines) + "\n", encoding="utf-8")
        fig, ax = plt.subplots(figsize=(7.5, 6))
        sizes = 20 + (z["career_post_games"] / (z["career_post_games"].max() or 1.0)) * 120
        ax.scatter(z["career_reg_win_pct"], z["career_post_win_pct"], s=sizes, alpha=0.7)
        m, b = np.polyfit(z["career_reg_win_pct"], z["career_post_win_pct"], 1)
        xs = np.linspace(z["career_reg_win_pct"].min(), z["career_reg_win_pct"].max(), 100)
        ax.plot(xs, m * xs + b)
        ax.set_xlabel(sc("career regular-season win%"))
        ax.set_ylabel(sc("career postseason win%"))
        ax.set_title(sc("rs vs po (career) — correlation"))
        ax.text(0.02, 0.98, sc(f"pearson={pear:.3f} / spearman={spear:.3f}"),
                transform=ax.transAxes, ha="left", va="top", fontsize=10,
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))
        ax.grid(alpha=0.2)
        fig_save(fig, figdir / "corr_rs_vs_po_scatter.png")
    else:
        ensure_dir(metadir).joinpath("correlations.txt").write_text("not enough data with playoffs to compute correlations.\n", encoding="utf-8")

    m = wg[[
        "career_eb_win_pct",
        "career_reg_win_pct",
        "career_post_win_pct",
        "career_overachieve",
        "rs_to_po_gap",
        "consistency_sd",
        "trend_slope",
        "total_games"
    ]].copy()
    m = m.dropna(how="all")
    if len(m) >= 3:
        c = m.corr(method="pearson").values
        labels = ["eb_win%", "rs_win%", "po_win%", "overach", "po-rs_gap", "consistency_sd", "trend", "games"]
        fig, ax = plt.subplots(figsize=(6.8, 5.6))
        im = ax.imshow(c, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(np.arange(len(labels))); ax.set_xticklabels([sc(t) for t in labels], rotation=45, ha="right")
        ax.set_yticks(np.arange(len(labels))); ax.set_yticklabels([sc(t) for t in labels])
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{c[i, j]:.2f}", ha="center", va="center", fontsize=8)
        ax.set_title(sc("correlation matrix (pearson)"))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig_save(fig, figdir / "corr_matrix.png")

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="plots-first analysis of wnba coaches (code/comments lowercase; plot texts sentence case).")
    ap.add_argument("--coaches", default="data/raw/coaches.csv", help="path to coaches csv")
    ap.add_argument("--teams",   default="data/raw/teams.csv", help="path to teams csv")
    ap.add_argument("--outdir",  default="reports/coaches", help="output directory")
    ap.add_argument("--topk", type=int, default=15, help="top-k to display in charts")
    ap.add_argument("--min_games", type=int, default=34, help="min career games for ranks (robustness)")
    args = ap.parse_args()

    outdir = ensure_dir(Path(args.outdir))
    figdir = ensure_dir(outdir / "figures")
    metadir = ensure_dir(outdir / "meta")

    df_c = load_coach_seasons(Path(args.coaches))
    df_t = load_team_seasons(Path(args.teams))
    df = merge_context(df_c, df_t)
    df = compute_first_year_deltas(df)
    wg = summarize_careers(df)

    plot_topk_bar_eb(wg, figdir, topk=args.topk, min_games=args.min_games)
    plot_slopegraph_rs_po(wg, figdir, topk=args.topk, min_games=args.min_games)
    plot_small_multiples(df, figdir, topn=min(12, args.topk), min_games=args.min_games)
    plot_overachievers(wg, figdir, topk=args.topk, min_games=args.min_games)
    plot_hist_first_year(df, figdir)
    plot_stint_averages(df, figdir)
    plot_correlations(wg, figdir, metadir)

if __name__ == "__main__":
    main()
