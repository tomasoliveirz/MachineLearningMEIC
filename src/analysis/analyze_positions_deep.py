#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Deep analysis of positions, stats, and physical profile.

Usa:
  - data/raw/players_teams.csv          → stats por época/equipa (player-year-team)
  - data/processed/players_cleaned.csv  → meta info (pos, height, weight, ...)

Faz:

1) Limpeza / features:
   - agrega stints por (bioID, year, tmID) se necessário
   - calcula per-36 PUROS:
        pts36, reb36, ast36, stl36, blk36, tov36
   - calcula perf36 (tua métrica de performance) via src.utils.players.compute_per36
   - junta altura, peso, career_length
   - cria:
        - role simplificado: guard/wing/forward/forward_center/center/unknown
        - pos_raw: posição original (C, PF, SF-G, etc)
        - pos_bucketed: top-N posições frequentes + "OTHER"

2) Análises:

   A) Contagens:
      - nº de player-seasons por role
      - nº de jogadores por pos_raw (todas) e por pos_bucketed (top-N)

   B) Médias e z-scores por ROLE:
      - mean de pts36, reb36, ast36, stl36, blk36, tov36, perf36 por role
      - z-score (comparado com média global) → heatmap legível

   C) Correlações:
      - numeric (height, weight, career_length, per-36 stats, perf36)
        vs
      - roles (one-hot)
      - pos_bucketed (one-hot, só top-N + OTHER)

   D) Height buckets:
      - divide altura em 4 quantis (Q1_short, Q2, Q3, Q4_tall)
      - calcula médias de per-36 + perf36 por bucket → heatmap

   E) Regressões por ROLE (pesos específicos por posição):
      - para cada role ∈ {guard, wing, forward, forward_center, center}:
          regressão OLS:
            minutes_next ~ pts36 + reb36 + ast36 + stl36 + blk36 + tov36
          (apenas jogadores com minutos_t >= MIN_MINUTES_REG_CURRENT
           e minutos_t+1 >= MIN_MINUTES_REG_NEXT)
      - guarda:
          - raw weights por role
          - normalized weights (pts36 = 1.0) por role
      → isto é a base dos teus weights específicos por posição.

Saídas:

  reports/positions_deep/
    ├─ tables/
    │    roles_counts_player_seasons.csv
    │    pos_raw_counts_players.csv
    │    pos_bucket_counts_player_seasons.csv
    │    per36_means_by_role.csv
    │    per36_means_z_by_role.csv
    │    corr_numeric_vs_roles.csv
    │    corr_numeric_vs_posbucket.csv
    │    height_bucket_means.csv
    │    role_regression_weights_raw.csv
    │    role_regression_weights_normalized.csv
    └─ figures/
         roles_counts_bar.png
         pos_bucket_counts_bar.png
         per36_means_by_role_z.png
         corr_numeric_vs_roles.png
         corr_numeric_vs_posbucket.png
         height_bucket_means_z.png
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# paths / imports internos
# --------------------------------------------------

# Root do projeto AC
ROOT = Path(__file__).resolve().parents[2]  # .../AC
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.players import (
    aggregate_stints, 
    compute_per36,
    pos_to_role,
    compute_boxscore_per36,
    build_height_buckets,
)
from src.utils.plots import ensure_dir


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

# threshold para análises descritivas (stats por posição)
MIN_MINUTES_ANALYSIS: float = 300.0

# thresholds para regressões por role (pesos específicos)
MIN_MINUTES_REG_CURRENT: float = 400.0
MIN_MINUTES_REG_NEXT: float = 200.0

# piso para per-36 PUROS (box-score / 36)
MIN_EFFECTIVE_MINUTES: float = 12.0

# quantas posições distintas queremos tratar como "top"
N_TOP_POS: int = 12


# --------------------------------------------------
# HELPERS
# --------------------------------------------------
# (Most helpers now imported from src.utils)


def _run_role_regressions(df: pd.DataFrame,
                          roles_for_reg: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Para cada role, corre uma regressão OLS:
        minutes_next ~ pts36 + reb36 + ast36 + stl36 + blk36 + tov36

    Apenas linhas com:
      minutes >= MIN_MINUTES_REG_CURRENT
      minutes_next >= MIN_MINUTES_REG_NEXT

    Returns:
        dict role -> {
            'n': n_obs,
            'intercept': ...,
            'w_pts36': ...,
            'w_reb36': ...,
            ...,
            'norm_pts36': 1.0,
            'norm_reb36': ...,
            ...
        }
    """
    per36_cols = ["pts36", "reb36", "ast36", "stl36", "blk36", "tov36"]
    results: Dict[str, Dict[str, float]] = {}

    for role in roles_for_reg:
        g = df[df["role"] == role].copy()

        mask = (
            g["minutes"].ge(MIN_MINUTES_REG_CURRENT)
            & g["minutes_next"].ge(MIN_MINUTES_REG_NEXT)
        )
        mask &= g[per36_cols].notna().all(axis=1)

        reg = g.loc[mask, per36_cols + ["minutes_next"]].copy()
        reg = reg.dropna()
        n_obs = len(reg)

        if n_obs < 40:
            # muito poucos, mas ainda assim guardamos info básica
            results[role] = {"n": float(n_obs)}
            continue

        X = reg[per36_cols].values
        y = reg["minutes_next"].values

        X_design = np.column_stack([np.ones(len(X)), X])
        coef, *_ = np.linalg.lstsq(X_design, y, rcond=None)
        intercept = float(coef[0])
        weights = coef[1:].astype(float)

        res = {
            "n": float(n_obs),
            "intercept": intercept,
        }
        for stat_name, w in zip(per36_cols, weights):
            res[f"w_{stat_name}"] = float(w)

        # normalizar por pts36
        w_pts = weights[0]
        if abs(w_pts) > 1e-6:
            norm = weights / w_pts
            for stat_name, w in zip(per36_cols, norm):
                res[f"norm_{stat_name}"] = float(w)
        else:
            # não conseguimos normalizar
            for stat_name in per36_cols:
                res[f"norm_{stat_name}"] = np.nan

        results[role] = res

    return results


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main() -> None:
    sns.set(style="whitegrid")

    players_teams_path = ROOT / "data" / "raw" / "players_teams.csv"
    players_meta_path = ROOT / "data" / "processed" / "players_cleaned.csv"

    if not players_teams_path.exists():
        print(f"Missing file: {players_teams_path}", file=sys.stderr)
        return
    if not players_meta_path.exists():
        print(f"Missing file: {players_meta_path}", file=sys.stderr)
        return

    reports_dir = ROOT / "reports" / "positions_deep"
    fig_dir = reports_dir / "figures"
    tables_dir = reports_dir / "tables"
    ensure_dir(fig_dir)
    ensure_dir(tables_dir)

    print("=" * 60)
    print("DEEP POSITION ANALYSIS (stats + roles + pos + height/weight)")
    print("=" * 60)

    # -----------------------------------------
    # 1) Carregar stats e agregar stints
    # -----------------------------------------
    print("\n[1/7] Loading stats and aggregating stints...")

    df_stats_raw = pd.read_csv(players_teams_path)

    # Agregar para player-year-team
    df_stats = aggregate_stints(df_stats_raw)

    # compute per36 de performance (compósito) e redes
    perf36, minutes_full = compute_per36(df_stats)
    df_stats["perf36"] = perf36
    df_stats["minutes"] = minutes_full  # garantir coluna consistente

    # per-36 PUROS de box-score
    df_stats = compute_boxscore_per36(df_stats, min_effective_minutes=MIN_EFFECTIVE_MINUTES)

    # ordenar para construir minutes_next
    df_stats = df_stats.sort_values(["bioID", "year"]).copy()
    df_stats["minutes_next"] = df_stats.groupby("bioID")["minutes"].shift(-1)

    # filtro para análises descritivas
    df_stats_analysis = df_stats[df_stats["minutes"] >= MIN_MINUTES_ANALYSIS].copy()
    print(f"  ✓ Player-year-team rows after minutes >= {MIN_MINUTES_ANALYSIS}: {len(df_stats_analysis)}")

    # -----------------------------------------
    # 2) Carregar meta players
    # -----------------------------------------
    print("\n[2/7] Loading player meta (positions, height, weight)...")

    df_meta = pd.read_csv(players_meta_path)
    df_meta["bioID"] = df_meta["bioID"].astype(str)

    keep_meta = ["bioID", "pos", "height", "weight", "firstseason", "lastseason"]
    keep_meta = [c for c in keep_meta if c in df_meta.columns]
    df_meta = df_meta[keep_meta].copy()

    # merge stats + meta
    df = df_stats_analysis.merge(df_meta, on="bioID", how="left")

    # role simplificado
    df["role"] = pos_to_role(df.get("pos", pd.Series(index=df.index)))

    # alturas / pesos / career length
    df["height"] = pd.to_numeric(df.get("height"), errors="coerce")
    df["weight"] = pd.to_numeric(df.get("weight"), errors="coerce")

    if {"firstseason", "lastseason"}.issubset(df.columns):
        first = pd.to_numeric(df["firstseason"], errors="coerce")
        last = pd.to_numeric(df["lastseason"], errors="coerce")
        df["career_length"] = (last - first + 1).where(last.notna() & first.notna(), np.nan)
    else:
        df["career_length"] = np.nan

    # height buckets
    df["height_bucket"] = build_height_buckets(df, col="height", n_quantiles=4)

    # preparar pos_raw e pos_bucketed
    df["pos_raw"] = df.get("pos", pd.Series(index=df.index)).fillna("UNKNOWN").astype(str).str.upper()

    pos_counts = df["pos_raw"].value_counts()
    top_pos = list(pos_counts.head(N_TOP_POS).index)
    df["pos_bucket"] = df["pos_raw"].where(df["pos_raw"].isin(top_pos), "OTHER")

    # ordem legível para roles
    role_order: List[str] = [r for r in ["guard", "wing", "forward", "forward_center", "center", "unknown"]
                             if r in df["role"].unique()]

    print("  ✓ Merged stats + meta. Unique players:", df["bioID"].nunique())

    # -----------------------------------------
    # 3) Contagens básicas
    # -----------------------------------------
    print("\n[3/7] Counts by role and pos...")

    # counts por role (player-seasons)
    role_counts = df["role"].value_counts().reindex(role_order).fillna(0).astype(int)
    role_counts.to_csv(tables_dir / "roles_counts_player_seasons.csv", header=["count"])

    print("Player-seasons by role:")
    print(role_counts)

    plt.figure(figsize=(7, 5))
    plt.bar(role_counts.index, role_counts.values)
    plt.ylabel("Number of player-seasons")
    plt.xlabel("Role")
    plt.title("Player-seasons by role")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "roles_counts_bar.png", dpi=160)
    plt.close()

    # counts por pos_raw (por jogador, não por season) – para ter ideia do universo
    pos_counts_players = (
        df_meta.assign(pos_raw=lambda x: x.get("pos", pd.Series(index=x.index)).fillna("UNKNOWN").astype(str).str.upper())
              .groupby("pos_raw")["bioID"]
              .nunique()
              .sort_values(ascending=False)
    )
    pos_counts_players.to_csv(tables_dir / "pos_raw_counts_players.csv", header=["n_players"])

    # counts por pos_bucket (player-seasons)
    pos_bucket_counts = df["pos_bucket"].value_counts().sort_values(ascending=False)
    pos_bucket_counts.to_csv(tables_dir / "pos_bucket_counts_player_seasons.csv", header=["count"])

    plt.figure(figsize=(9, 5))
    plt.bar(pos_bucket_counts.index, pos_bucket_counts.values)
    plt.ylabel("Number of player-seasons")
    plt.xlabel("Position bucket (top-N + OTHER)")
    plt.title("Player-seasons by pos bucket")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "pos_bucket_counts_bar.png", dpi=160)
    plt.close()

    # -----------------------------------------
    # 4) Médias per-36 por ROLE (incluindo perf36)
    # -----------------------------------------
    print("\n[4/7] Per-36 means by role (incl. perf36)...")

    per36_cols: List[str] = ["pts36", "reb36", "ast36", "stl36", "blk36", "tov36", "perf36"]
    df[per36_cols] = df[per36_cols].apply(pd.to_numeric, errors="coerce")

    means_role = df.groupby("role")[per36_cols].mean().reindex(role_order)
    means_role.to_csv(tables_dir / "per36_means_by_role.csv")

    print("Mean per-36 by role (first rows):")
    print(means_role.round(2).head())

    global_mean = df[per36_cols].mean()
    global_std = df[per36_cols].std()
    z_means_role = (means_role - global_mean) / global_std
    z_means_role.to_csv(tables_dir / "per36_means_z_by_role.csv")

    plt.figure(figsize=(9, 6))
    sns.heatmap(
        z_means_role,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0.0,
        linewidths=0.5,
        cbar_kws={"label": "z-score of mean per-36"},
        annot_kws={"size": 9},
    )
    plt.xlabel("Role")
    plt.ylabel("Per-36 stat (incl. perf36)")
    plt.title("Per-36 stats by role (z-score of means)")
    plt.xticks(rotation=25, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(fig_dir / "per36_means_by_role_z.png", dpi=160)
    plt.close()

    # -----------------------------------------
    # 5) Correlações: numeric vs roles e vs pos_bucket
    # -----------------------------------------
    print("\n[5/7] Correlations: numeric features vs roles & pos buckets...")

    numeric_cols = [
        "height", "weight", "career_length",
        "pts36", "reb36", "ast36", "stl36", "blk36", "tov36", "perf36",
    ]

    d = df[numeric_cols + ["role", "pos_bucket"]].dropna(subset=["pts36", "perf36"])

    # --- vs roles ---
    role_dummies = pd.get_dummies(d["role"], prefix="role")
    corr_role = (
        pd.concat([d[numeric_cols], role_dummies], axis=1)
        .corr(method="pearson")
        .loc[numeric_cols, role_dummies.columns]
    )
    corr_role = corr_role.loc[:, corr_role.abs().sum() > 0]
    corr_role.to_csv(tables_dir / "corr_numeric_vs_roles.csv")

    plt.figure(figsize=(11, 6))
    sns.heatmap(
        corr_role,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        cbar_kws={"label": "Pearson corr"},
        annot_kws={"size": 8},
    )
    plt.xlabel("Role (one-hot)")
    plt.ylabel("Numeric feature")
    plt.title("Correlation: numeric features vs roles")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(fig_dir / "corr_numeric_vs_roles.png", dpi=160)
    plt.close()

    # --- vs pos_bucket ---
    pos_dummies = pd.get_dummies(d["pos_bucket"], prefix="pos")
    corr_pos = (
        pd.concat([d[numeric_cols], pos_dummies], axis=1)
        .corr(method="pearson")
        .loc[numeric_cols, pos_dummies.columns]
    )
    corr_pos = corr_pos.loc[:, corr_pos.abs().sum() > 0]
    corr_pos.to_csv(tables_dir / "corr_numeric_vs_posbucket.csv")

    plt.figure(figsize=(11, 6))
    sns.heatmap(
        corr_pos,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        cbar_kws={"label": "Pearson corr"},
        annot_kws={"size": 8},
    )
    plt.xlabel("Pos bucket (one-hot)")
    plt.ylabel("Numeric feature")
    plt.title("Correlation: numeric features vs top-N positions (+OTHER)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(fig_dir / "corr_numeric_vs_posbucket.png", dpi=160)
    plt.close()

    # -----------------------------------------
    # 6) Height buckets: stats por faixa de altura
    # -----------------------------------------
    print("\n[6/7] Height buckets analysis...")

    hb = df[df["height_bucket"].notna()].copy()
    if hb["height_bucket"].nunique() > 1:
        hb_order = ["Q1_short", "Q2", "Q3", "Q4_tall"]
        hb_order = [x for x in hb_order if x in hb["height_bucket"].unique().tolist()]

        means_hb = hb.groupby("height_bucket")[per36_cols].mean().reindex(hb_order)
        means_hb.to_csv(tables_dir / "height_bucket_means.csv")

        gm_hb = hb[per36_cols].mean()
        gs_hb = hb[per36_cols].std()
        z_means_hb = (means_hb - gm_hb) / gs_hb

        plt.figure(figsize=(9, 5))
        sns.heatmap(
            z_means_hb,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0.0,
            linewidths=0.5,
            cbar_kws={"label": "z-score of mean per-36"},
            annot_kws={"size": 8},
        )
        plt.xlabel("Height bucket (quantiles)")
        plt.ylabel("Per-36 stat (incl. perf36)")
        plt.title("Per-36 stats by height bucket (z-score)")
        plt.xticks(rotation=25, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(fig_dir / "height_bucket_means_z.png", dpi=160)
        plt.close()
    else:
        print("  [WARN] Not enough height variance for height buckets.")

    # -----------------------------------------
    # 7) Regressões por ROLE: pesos específicos por posição
    # -----------------------------------------
    print("\n[7/7] Role-specific regressions for per-36 weights...")

    # Usar df_stats (não filtrado por MIN_MINUTES_ANALYSIS) para regressões mais robustas
    df_reg = df_stats.merge(df_meta, on="bioID", how="left")
    df_reg["role"] = pos_to_role(df_reg.get("pos", pd.Series(index=df_reg.index)))

    # per-36 puros no df_reg
    df_reg = compute_boxscore_per36(df_reg, min_effective_minutes=MIN_EFFECTIVE_MINUTES)

    roles_for_reg = [r for r in ["guard", "wing", "forward", "forward_center", "center"]
                     if r in df_reg["role"].unique()]

    reg_results = _run_role_regressions(df_reg, roles_for_reg)

    # guardar resultados em tabelas
    rows_raw = []
    rows_norm = []
    for role, res in reg_results.items():
        n_obs = res.get("n", np.nan)
        row_raw = {"role": role, "n": n_obs}
        row_norm = {"role": role, "n": n_obs}

        for stat in ["pts36", "reb36", "ast36", "stl36", "blk36", "tov36"]:
            row_raw[f"w_{stat}"] = res.get(f"w_{stat}", np.nan)
            row_norm[f"norm_{stat}"] = res.get(f"norm_{stat}", np.nan)

        rows_raw.append(row_raw)
        rows_norm.append(row_norm)

    df_raw = pd.DataFrame(rows_raw)
    df_norm = pd.DataFrame(rows_norm)

    df_raw.to_csv(tables_dir / "role_regression_weights_raw.csv", index=False)
    df_norm.to_csv(tables_dir / "role_regression_weights_normalized.csv", index=False)

    print("Role-specific normalized weights (pts36 = 1.0):")
    print(df_norm.set_index("role").round(2))

    print("\nDone.")
    print(f"Figures saved in: {fig_dir}")
    print(f"Tables saved in:  {tables_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
