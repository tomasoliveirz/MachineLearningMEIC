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

def pos_to_role(pos: pd.Series) -> pd.Series:
    """
    Map position strings to simplified roles.
    
    Converts raw position codes (e.g., 'G', 'F-C', 'G-F', 'C', 'PF', 'SG')
    to standardized role categories:
      - 'guard': positions with 'G' but no 'F' or 'C'
      - 'wing': positions with both 'G' and 'F' (e.g., 'G-F', 'SG-SF')
      - 'forward': positions with 'F' but no 'C' or 'G'
      - 'forward_center': positions with both 'F' and 'C' (e.g., 'F-C', 'PF-C')
      - 'center': positions with 'C' but no 'F' or 'G'
      - 'unknown': anything else or missing values
    
    Args:
        pos: Series with position strings
        
    Returns:
        Series with role categories
        
    Example:
        >>> pos = pd.Series(['G', 'F-C', 'G-F', 'C', None])
        >>> pos_to_role(pos)
        0         guard
        1    forward_center
        2          wing
        3        center
        4       unknown
        dtype: object
    """
    if pos is None or len(pos) == 0:
        return pd.Series([], dtype=str)

    p = pos.fillna("unknown").astype(str).str.upper()

    def _map(s: str) -> str:
        s = s.strip()
        if not s or s == "UNKNOWN" or s == "NAN":
            return "unknown"

        # Guard: has G but no F or C
        if "G" in s and "F" not in s and "C" not in s:
            return "guard"
        
        # Wing: has both G and F (e.g., SG-SF, G-F)
        if "G" in s and "F" in s:
            return "wing"
        
        # Forward-Center: has both F and C
        if "F" in s and "C" in s:
            return "forward_center"
        
        # Center: has C but no F
        if "C" in s:
            return "center"
        
        # Forward: has F but no C or G
        if "F" in s:
            return "forward"
        
        return "unknown"

    return p.map(_map)


# Keep internal version for backward compatibility
def _pos_to_role(pos: pd.Series) -> pd.Series:
    """Internal alias for backward compatibility. Use pos_to_role() instead."""
    return pos_to_role(pos)


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
# PURE PER-36 CALCULATIONS (BOX-SCORE)
# --------------------------------------------------------------------

def compute_boxscore_per36(df: pd.DataFrame, min_effective_minutes: float = 12.0) -> pd.DataFrame:
    """
    Compute pure per-36 box-score statistics without any weighting.
    
    Takes raw counting stats (points, rebounds, assists, etc.) and converts
    them to per-36-minute rates. This is different from compute_per36() which
    applies position-specific weights to create a composite performance metric.
    
    Args:
        df: DataFrame with columns (if present):
            - minutes, points, rebounds, assists, steals, blocks, turnovers
        min_effective_minutes: Minimum minutes floor to avoid extreme per-36 values
            (default: 12.0)
            
    Returns:
        Copy of df with added columns: pts36, reb36, ast36, stl36, blk36, tov36
        
    Example:
        >>> df = pd.DataFrame({
        ...     'minutes': [100, 200, 5],
        ...     'points': [50, 120, 10]
        ... })
        >>> result = compute_boxscore_per36(df)
        >>> result[['minutes', 'points', 'pts36']]
           minutes  points  pts36
        0      100      50   18.0
        1      200     120   21.6
        2        5      10   30.0  # floored to min_effective_minutes
    """
    out = df.copy()
    
    # Column mapping: raw stat name -> per-36 column name
    stat_cols = {
        "points": "pts36",
        "rebounds": "reb36",
        "assists": "ast36",
        "steals": "stl36",
        "blocks": "blk36",
        "turnovers": "tov36",
    }
    
    # Ensure all stat columns exist as numeric
    for raw_col in stat_cols.keys():
        if raw_col in out.columns:
            out[raw_col] = pd.to_numeric(out[raw_col], errors="coerce")
        else:
            out[raw_col] = np.nan
    
    # Handle minutes with floor
    if "minutes" in out.columns:
        out["minutes"] = pd.to_numeric(out["minutes"], errors="coerce")
    else:
        out["minutes"] = np.nan
    
    minutes = out["minutes"].replace({0: np.nan})
    minutes_floor = minutes.copy()
    
    # Apply floor: any value below min_effective_minutes gets raised to it
    small_mask = minutes_floor.notna() & (minutes_floor < min_effective_minutes)
    minutes_floor.loc[small_mask] = min_effective_minutes
    
    # Calculate per-36 for each stat
    for raw_col, per36_col in stat_cols.items():
        out[per36_col] = (out[raw_col] / minutes_floor) * 36.0
        # Where minutes are invalid, per36 should be NaN
        out.loc[minutes_floor.isna(), per36_col] = np.nan
    
    return out


# --------------------------------------------------------------------
# HEIGHT BUCKETS
# --------------------------------------------------------------------

def build_height_buckets(df: pd.DataFrame, col: str = "height", 
                        n_quantiles: int = 4) -> pd.Series:
    """
    Create height buckets based on quantiles.
    
    Divides players into height groups (e.g., Q1_short, Q2, Q3, Q4_tall)
    based on quantile cuts. Useful for position-agnostic height analysis.
    
    Args:
        df: DataFrame containing height column
        col: Name of height column (default: "height")
        n_quantiles: Number of quantile groups (default: 4)
        
    Returns:
        Series with bucket labels. If not enough data, returns "unknown" for all.
        
    Example:
        >>> df = pd.DataFrame({'height': [65, 68, 71, 74, 77, 80]})
        >>> build_height_buckets(df)
        0    Q1_short
        1    Q1_short
        2         Q2
        3         Q3
        4         Q3
        5    Q4_tall
        dtype: category
    """
    if col not in df.columns:
        return pd.Series(["unknown"] * len(df), index=df.index, dtype=str)
    
    h = pd.to_numeric(df[col], errors="coerce")
    
    # Need at least some valid values
    if h.notna().sum() < 10:
        return pd.Series(["unknown"] * len(df), index=df.index, dtype=str)
    
    # Create labels based on n_quantiles
    if n_quantiles == 4:
        labels = ["Q1_short", "Q2", "Q3", "Q4_tall"]
    elif n_quantiles == 3:
        labels = ["Q1_short", "Q2", "Q3_tall"]
    elif n_quantiles == 5:
        labels = ["Q1_short", "Q2", "Q3", "Q4", "Q5_tall"]
    else:
        labels = [f"Q{i+1}" for i in range(n_quantiles)]
    
    try:
        buckets = pd.qcut(h, q=n_quantiles, labels=labels, duplicates='drop')
    except (ValueError, TypeError):
        # If qcut fails (e.g., too many duplicate values), try pd.cut
        try:
            buckets = pd.cut(h, bins=n_quantiles, labels=labels[:n_quantiles])
        except (ValueError, TypeError):
            # Last resort: return unknown
            return pd.Series(["unknown"] * len(df), index=df.index, dtype=str)
    
    # Fill NaN values with "unknown"
    buckets = buckets.astype(str).replace("nan", "unknown")
    
    return buckets


# --------------------------------------------------------------------
# ROOKIE ORIGIN DETECTION
# --------------------------------------------------------------------

def infer_rookie_origin(players: pd.DataFrame) -> pd.Series:
    """
    Detect player origin based on college attendance.
    
    Classifies players as:
      - 'ncaa': went to college (college field is non-empty and not unknown)
      - 'non_ncaa': no college or unknown college (international, high school, etc.)
    
    Args:
        players: DataFrame with a 'college' column
        
    Returns:
        Series with 'ncaa' or 'non_ncaa' labels
        
    Example:
        >>> df = pd.DataFrame({
        ...     'playerID': ['p1', 'p2', 'p3'],
        ...     'college': ['Stanford', None, 'Unknown']
        ... })
        >>> infer_rookie_origin(df)
        0        ncaa
        1    non_ncaa
        2    non_ncaa
        dtype: object
    """
    if "college" not in players.columns:
        # No college column: assume all are non-NCAA
        return pd.Series("non_ncaa", index=players.index, dtype=str)
    
    college = players["college"].astype(str).str.strip().str.lower()
    
    # Consider empty, "nan", "none", "unknown" as non-NCAA
    is_unknown = (
        college.isna() 
        | college.eq("nan") 
        | college.eq("none")
        | college.eq("unknown") 
        | college.eq("")
    )
    
    return pd.Series(
        np.where(is_unknown, "non_ncaa", "ncaa"),
        index=players.index,
        dtype=str
    )


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
