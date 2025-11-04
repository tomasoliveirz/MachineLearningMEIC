#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared player data utilities for aggregation, transformation, and metrics.

Trabalha sempre com os nomes REAIS do players_teams.csv:
- playerID / bioID, year, tmID
- minutes, points, rebounds, oRebounds, dRebounds
- assists, steals, blocks, turnovers
- GP, GS

Métrica principal:
    perf_per36 = combinação linear de stats per-36

    1) Calculamos:
        PTS36, REB36, AST36, STL36, BLK36, TOV36
    2) Aplicamos pesos:
        - Se USE_ROLE_WEIGHTS = True → pesos dependem do role (guard/wing/…)
        - Caso contrário → pesos globais iguais para todos

Os pesos por role foram escolhidos assim:
    - Base: regressão global minutes_next ~ stats_per36
    - Ajustes: análise profunda por posição (analyze_positions_deep.py)
    - Depois suavizados para ficarem interpretáveis (sem coeficientes malucos tipo 6.49…)
"""

from typing import Tuple, Dict

import numpy as np
import pandas as pd


# --------------------------------------------------------------------
# CONFIG GERAL
# --------------------------------------------------------------------

# piso mínimo de minutos para evitar per-36 absurdos
MIN_EFFECTIVE_MINUTES: float = 12.0

# usar pesos específicos por role? se False → usa GLOBAL_WEIGHTS para toda a gente
USE_ROLE_WEIGHTS: bool = True

# Pesos globais (para todos os jogadores) em cima das stats per-36:
#   perf_per36 ≈ pts + 0.7*reb + 1.4*ast + 0.8*stl + 1.0*blk - 1.6*tov
# Baseado na regressão (learn_per36_weights.py), mas:
#   - reb foi ajustado de ligeiramente negativo para ~0.7 (faz sentido basquetebolisticamente)
GLOBAL_WEIGHTS: Dict[str, float] = {
    "pts": 1.0,
    "reb": 0.7,
    "ast": 1.4,
    "stl": 0.8,
    "blk": 1.0,
    "tov": -1.6,
}

# Pesos por role (guard, wing, forward, forward_center, center, unknown)
# Estes foram construídos assim:
#   - Olhar para:
#        * médias per-36 por role
#        * z-scores por role (quem é forte em quê)
#        * regressões específicas por role (minutes_next ~ stats_per36)
#   - Depois suavizar para algo estável e intuitivo.
#
# Interpretação (regra geral):
#   - guards  -> AST/TOV muito importantes, steals relevantes, reb/blk pouco
#   - wings   -> PTS + AST + STL fortes
#   - forwards-> REB + BLK importantes, PTS razoável
#   - F/C     -> REB + BLK bem fortes, algum AST/STL
#   - centers -> REB + BLK muito fortes, AST baixo, TOV bem penalizado
POSITION_WEIGHTS: Dict[str, Dict[str, float]] = {
    "guard": {
        "pts": 1.0,
        "reb": 0.4,
        "ast": 1.8,
        "stl": 1.2,
        "blk": 0.3,
        "tov": -1.6,
    },
    "wing": {
        "pts": 1.1,
        "reb": 0.7,
        "ast": 1.4,
        "stl": 1.4,
        "blk": 0.6,
        "tov": -1.5,
    },
    "forward": {
        "pts": 1.0,
        "reb": 1.1,
        "ast": 0.6,
        "stl": 0.8,
        "blk": 1.2,
        "tov": -1.2,
    },
    "forward_center": {
        "pts": 1.0,
        "reb": 1.2,
        "ast": 0.8,
        "stl": 0.9,
        "blk": 1.5,
        "tov": -1.3,
    },
    "center": {
        "pts": 0.9,
        "reb": 1.3,
        "ast": 0.4,
        "stl": 0.7,
        "blk": 1.8,
        "tov": -1.4,
    },
    # fallback quando não sabemos a posição (ou algo estranho)
    "unknown": {
        "pts": 1.0,
        "reb": 0.9,
        "ast": 1.0,
        "stl": 1.0,
        "blk": 1.0,
        "tov": -1.3,
    },
}


# --------------------------------------------------------------------
# HELPERS PARA ROLE / POS
# --------------------------------------------------------------------

def _pos_to_role(pos: pd.Series) -> pd.Series:
    """
    Mapear coluna 'pos' (raw, tipo 'G', 'F-C', 'G-F', 'C', etc.)
    para roles simplificados: guard, wing, forward, forward_center, center, unknown.
    """
    if pos is None:
        return pd.Series(["unknown"] * 0, index=pd.Index([], name="idx"))

    p = pos.fillna("unknown").astype(str).str.upper()

    def _map(s: str) -> str:
        s = s.strip()
        if not s or s == "UNKNOWN":
            return "unknown"

        # combos comuns
        if "G" in s and "F" not in s and "C" not in s:
            return "guard"
        if "G" in s and "F" in s:
            # SG/SF, G-F -> wing
            return "wing"
        if "F" in s and "C" in s:
            return "forward_center"
        if "C" in s:
            return "center"
        if "F" in s:
            return "forward"
        return "unknown"

    return p.map(_map)


def _infer_role(df: pd.DataFrame) -> pd.Series:
    """
    Inferir role para cada linha do df.

    Prioridade:
      1) Se existir coluna 'role' já com valores {guard, wing, ...}, usa isso.
      2) Caso contrário, tenta coluna 'pos' (e mapeia com _pos_to_role).
      3) Se nada existir, tudo 'unknown'.
    """
    idx = df.index

    # 1) Se já existir 'role' explícito e bater com o nosso dicionário
    if "role" in df.columns:
        r = df["role"].astype(str).str.lower().str.strip()
        # mantém só roles conhecidos
        valid_roles = set(POSITION_WEIGHTS.keys())
        r = r.where(r.isin(valid_roles), other=np.nan)
    else:
        r = pd.Series(np.nan, index=idx, dtype=object)

    # 2) Fallback: usar 'pos' se ainda houver NaNs
    if r.isna().any():
        if "pos" in df.columns:
            r_pos = _pos_to_role(df["pos"])
            r_pos.index = idx  # garantir alinhamento
        else:
            r_pos = pd.Series("unknown", index=idx, dtype=object)
        r = r.fillna(r_pos)

    # 3) Fallback final: qualquer coisa estranha -> 'unknown'
    valid_roles = set(POSITION_WEIGHTS.keys())
    r = r.where(r.isin(valid_roles), other="unknown")

    return r


# --------------------------------------------------------------------
# AGGREGATION & ROOKIES
# --------------------------------------------------------------------

def aggregate_stints(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate multi-stint rows to player-year-team level.

    Usa as colunas reais do players_teams.csv:
      - playerID / bioID, year, tmID
      - minutes, points, rebounds, oRebounds, dRebounds
      - assists, steals, blocks, turnovers
      - GP, GS
    """
    df = df.copy()

    # garantir bioID
    if "bioID" not in df.columns:
        if "playerID" not in df.columns:
            raise KeyError("Expected 'playerID' or 'bioID' in players_teams.csv")
        df["bioID"] = df["playerID"].astype(str)
    else:
        df["bioID"] = df["bioID"].astype(str)

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    if "tmID" in df.columns:
        df["tmID"] = df["tmID"].astype(str)
    else:
        df["tmID"] = "UNK"

    numeric_cols = [
        "minutes",
        "points",
        "rebounds", "oRebounds", "dRebounds",
        "assists", "steals", "blocks", "turnovers",
        "GP", "GS",
    ]
    stat_cols = [c for c in numeric_cols if c in df.columns]
    for c in stat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    agg = (
        df.groupby(["bioID", "year", "tmID"], dropna=False)[stat_cols]
          .sum(min_count=1)
          .reset_index()
    )
    return agg


def label_rookies(df: pd.DataFrame) -> pd.Series:
    """Return boolean Series marking rookie seasons (first year > min year in dataset)."""
    if "bioID" not in df.columns or "year" not in df.columns:
        raise KeyError("label_rookies precisa de colunas 'bioID' e 'year'")

    years_first = df.groupby("bioID", dropna=False)["year"].transform("min")
    min_year_dataset = df["year"].min()
    return (df["year"].eq(years_first)) & (df["year"] > min_year_dataset)


# --------------------------------------------------------------------
# MÉTRICA PRINCIPAL: compute_per36
# --------------------------------------------------------------------

def compute_per36(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Calcular performance per-36 e devolver também os minutos.

    Passos:
      1) Converte colunas base para numérico:
            minutes, points, rebounds, assists, steals, blocks, turnovers
      2) Calcula stats per-36 com piso de minutos:
            minutes_floor = max(minutes, MIN_EFFECTIVE_MINUTES) (se >0)
            STAT36 = (STAT_TOTAL / minutes_floor) * 36
      3) Se USE_ROLE_WEIGHTS = True:
            - infere 'role' (guard/wing/forward/forward_center/center/unknown)
            - aplica POSITION_WEIGHTS[role] linha a linha
         Senão:
            - aplica GLOBAL_WEIGHTS para toda a gente

      perf_per36 = w_pts*PTS36 + w_reb*REB36 + ... + w_tov*TOV36

    Returns:
        per36  : Series com a métrica de performance per-36
        minutes: Series com minutos reais (sem piso, NaN -> 0.0)
    """
    df = df.copy()

    def col(name: str) -> pd.Series:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
        return pd.Series(np.nan, index=df.index, dtype=float)

    minutes = col("minutes")
    minutes = minutes.replace({0: np.nan})

    minutes_floor = minutes.copy()
    mask_small = minutes_floor.notna() & (minutes_floor < MIN_EFFECTIVE_MINUTES)
    minutes_floor.loc[mask_small] = MIN_EFFECTIVE_MINUTES

    # box-score totals
    pts = col("points")
    reb = col("rebounds")
    ast = col("assists")
    stl = col("steals")
    blk = col("blocks")
    tov = col("turnovers")

    # stats per-36 (com piso de minutos)
    pts36 = (pts / minutes_floor) * 36.0
    reb36 = (reb / minutes_floor) * 36.0
    ast36 = (ast / minutes_floor) * 36.0
    stl36 = (stl / minutes_floor) * 36.0
    blk36 = (blk / minutes_floor) * 36.0
    tov36 = (tov / minutes_floor) * 36.0

    # onde não há minutos válidos, tudo vira NaN
    mask_no_min = minutes_floor.isna()
    pts36[mask_no_min] = np.nan
    reb36[mask_no_min] = np.nan
    ast36[mask_no_min] = np.nan
    stl36[mask_no_min] = np.nan
    blk36[mask_no_min] = np.nan
    tov36[mask_no_min] = np.nan

    # escolher pesos linha a linha
    # construímos um DataFrame com weights por row: w_pts, w_reb, ...
    weights = pd.DataFrame(
        index=df.index,
        columns=["pts", "reb", "ast", "stl", "blk", "tov"],
        dtype=float,
    )

    if USE_ROLE_WEIGHTS:
        roles = _infer_role(df)
        for role_name, w in POSITION_WEIGHTS.items():
            mask = roles == role_name
            if not mask.any():
                continue
            weights.loc[mask, "pts"] = w["pts"]
            weights.loc[mask, "reb"] = w["reb"]
            weights.loc[mask, "ast"] = w["ast"]
            weights.loc[mask, "stl"] = w["stl"]
            weights.loc[mask, "blk"] = w["blk"]
            weights.loc[mask, "tov"] = w["tov"]

        # qualquer linha ainda sem peso recebe 'unknown'
        mask_nan = weights["pts"].isna()
        if mask_nan.any():
            w = POSITION_WEIGHTS["unknown"]
            weights.loc[mask_nan, "pts"] = w["pts"]
            weights.loc[mask_nan, "reb"] = w["reb"]
            weights.loc[mask_nan, "ast"] = w["ast"]
            weights.loc[mask_nan, "stl"] = w["stl"]
            weights.loc[mask_nan, "blk"] = w["blk"]
            weights.loc[mask_nan, "tov"] = w["tov"]

    else:
        # aplica os globais a toda a gente
        for k in ["pts", "reb", "ast", "stl", "blk", "tov"]:
            weights[k] = GLOBAL_WEIGHTS[k]

    # finalmente, perf_per36 linha a linha
    perf36 = (
        weights["pts"] * pts36
        + weights["reb"] * reb36
        + weights["ast"] * ast36
        + weights["stl"] * stl36
        + weights["blk"] * blk36
        + weights["tov"] * tov36
    )

    # devolve perf_per36 + minutos (0.0 onde não há valor)
    return perf36.astype(float), minutes.fillna(0.0)


# --------------------------------------------------------------------
# NEXT-YEAR UTILITY
# --------------------------------------------------------------------

def per36_next_year(df: pd.DataFrame) -> pd.Series:
    """Map each row to the same player's per36 in the next season (t+1).

    Usa o compute_per36 atual (com pesos globais/por role).
    Importante para validação (correlação atual vs próximo ano, etc).
    """
    per36, _ = compute_per36(df)
    nxt = (
        df.assign(per36=per36)
          .sort_values(["bioID", "year"])
          .groupby("bioID")["per36"]
          .shift(-1)
    )
    return nxt
