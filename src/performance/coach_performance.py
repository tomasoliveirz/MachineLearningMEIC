#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera métricas de performance de treinadores a partir de data/raw/coaches.csv
e salva em data/processed/coach_perfomance.csv.

Métricas por (coachID, year, tmID):
- win_pct_season                  — win rate da época (won / (won+lost))
- last1_win_pct, last3_win_pct,
  last5_win_pct                  — médias das últimas N épocas ANTES da atual
- career_wins_to_date, career_losses_to_date,
  career_win_pct_to_date         — carreira até a época anterior
- tenure_with_team               — nº de épocas com o mesmo tmID até a atual (contagem cumulativa)
- playoff_win_pct_season         — pós-temporada na época (se houver jogos)
- confID, name                   — juntados de teams_cleaned quando disponíveis

Saída: data/processed/coach_perfomance.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def get_root() -> Path:
	# src/ -> raiz do projeto
	here = Path(__file__).resolve()
	for p in [here] + list(here.parents):
		if (p / "requirements.txt").exists() and (p / "Makefile").exists():
			return p
	# fallback: dois níveis acima de src/
	return Path(__file__).resolve().parents[2]


ROOT = get_root()
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

COACHES_PATH = RAW / "coaches.csv"
TEAMS_CLEANED_PATH = PROC / "teams_cleaned.csv"
OUT_PATH = PROC / "coach_perfomance.csv"  # nota: manter grafia pedida


def _safe_div(n: float, d: float) -> float:
	return float(n / d) if d and not np.isnan(d) else np.nan


def _rolling_mean_prev(values: pd.Series, window: int) -> pd.Series:
	"""Média das últimas N entradas ANTES da atual (shift + rolling)."""
	return (
		values.shift(1)
		.rolling(window=window, min_periods=1)
		.mean()
	)


def _cumsum_prev(values: pd.Series) -> pd.Series:
	"""Cumulativo até a época anterior (shifted cumsum)."""
	return values.shift(1).fillna(0).cumsum()


def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame | None]:
	coaches = pd.read_csv(COACHES_PATH)
	# normalizar tipos
	coaches["year"] = pd.to_numeric(coaches["year"], errors="coerce")
	coaches["tmID"] = coaches["tmID"].astype(str)
	coaches["won"] = pd.to_numeric(coaches["won"], errors="coerce").fillna(0).astype(int)
	coaches["lost"] = pd.to_numeric(coaches["lost"], errors="coerce").fillna(0).astype(int)
	coaches["post_wins"] = pd.to_numeric(coaches.get("post_wins", 0), errors="coerce").fillna(0).astype(int)
	coaches["post_losses"] = pd.to_numeric(coaches.get("post_losses", 0), errors="coerce").fillna(0).astype(int)

	teams = None
	if TEAMS_CLEANED_PATH.exists():
		teams = pd.read_csv(TEAMS_CLEANED_PATH)
		teams["year"] = pd.to_numeric(teams["year"], errors="coerce")
		teams["tmID"] = teams["tmID"].astype(str)
		# manter colunas úteis para contexto
		keep = [
			"year",
			"tmID",
			"confID",
			"name",
			"won",
			"lost",
			"GP",
		]
		miss = [c for c in keep if c not in teams.columns]
		if miss:
			# selecionar interseção se faltar algumas
			keep = [c for c in keep if c in teams.columns]
		teams = teams[keep].copy()
	return coaches, teams


def compute_metrics(coaches: pd.DataFrame, teams: pd.DataFrame | None) -> pd.DataFrame:
	df = coaches.copy()
	# win pct na época
	df["games"] = (df["won"] + df["lost"]).astype(float)
	df["win_pct_season"] = df.apply(lambda r: _safe_div(r["won"], r["games"]), axis=1)
	df["playoff_games"] = (df["post_wins"] + df["post_losses"]).astype(float)
	df["playoff_win_pct_season"] = df.apply(lambda r: _safe_div(r["post_wins"], r["playoff_games"]), axis=1)

	# ordenar por coachID e year
	df = df.sort_values(["coachID", "year", "tmID"]).reset_index(drop=True)

	# rolling por coachID
	def _per_coach(g: pd.DataFrame) -> pd.DataFrame:
		g = g.sort_values(["year"]).reset_index(drop=True)
		# cumulativos até época anterior
		g["career_wins_to_date"] = _cumsum_prev(g["won"].astype(float))
		g["career_losses_to_date"] = _cumsum_prev(g["lost"].astype(float))
		g["career_games_to_date"] = g["career_wins_to_date"] + g["career_losses_to_date"]
		g["career_win_pct_to_date"] = g.apply(
			lambda r: _safe_div(r["career_wins_to_date"], r["career_games_to_date"]), axis=1
		)

		# médias das últimas N épocas (antes da atual)
		for n in (1, 3, 5):
			g[f"last{n}_win_pct"] = _rolling_mean_prev(g["win_pct_season"], n)

		# tenure com o mesmo tmID (contagem cumulativa dentro de cada tmID)
		g["tenure_with_team"] = (
			g.groupby("tmID").cumcount() + 1  # conta incluindo a atual
		)
		return g

	df = df.groupby("coachID", group_keys=False).apply(_per_coach)

	# juntar contexto do teams_cleaned (confID, name) se disponível
	if teams is not None and not teams.empty:
		df = df.merge(teams, on=["year", "tmID"], how="left", suffixes=("", "_team"))

	# colunas de saída ordenadas
	out_cols = [
		"coachID",
		"year",
		"tmID",
		"confID",
		"name",
		"won",
		"lost",
		"games",
		"win_pct_season",
		"last1_win_pct",
		"last3_win_pct",
		"last5_win_pct",
		"career_wins_to_date",
		"career_losses_to_date",
		"career_games_to_date",
		"career_win_pct_to_date",
		"tenure_with_team",
		"post_wins",
		"post_losses",
		"playoff_games",
		"playoff_win_pct_season",
	]
	# manter apenas as que existem
	out_cols = [c for c in out_cols if c in df.columns]
	df_out = df[out_cols].copy()

	# tipos finais
	numeric_cols = [
		"won",
		"lost",
		"games",
		"win_pct_season",
		"last1_win_pct",
		"last3_win_pct",
		"last5_win_pct",
		"career_wins_to_date",
		"career_losses_to_date",
		"career_games_to_date",
		"career_win_pct_to_date",
		"tenure_with_team",
		"post_wins",
		"post_losses",
		"playoff_games",
		"playoff_win_pct_season",
	]
	for c in numeric_cols:
		if c in df_out.columns:
			df_out[c] = pd.to_numeric(df_out[c], errors="coerce")

	return df_out


def main() -> None:
	coaches, teams = load_inputs()
	df = compute_metrics(coaches, teams)
	df.sort_values(["year", "coachID", "tmID"], inplace=True)
	df.to_csv(OUT_PATH, index=False, encoding="utf-8")

if __name__ == "__main__":
	main()

