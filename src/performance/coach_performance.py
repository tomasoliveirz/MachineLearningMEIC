"""
Coach performance calculation

Este módulo calcula uma métrica de "performance" por coach-season que combina três sinais:

1. **z_win**: Sucesso absoluto da equipe (win_pct), padronizado por ano
   - Mede o quão bem o coach performou em termos de vitórias

2. **z_over**: Over/under performance relativo à expectativa (pred_win)
   - Mede se o coach superou ou ficou abaixo das expectativas
   - over = win_pct - pred_win, depois padronizado por ano
   
3. **z_rock**: Desenvolvimento de rookies (performance média de rookies na equipe)
   - Mede a capacidade do coach de desenvolver jogadores novatos
   - Padronizado por ano e multiplicado por 1.2 (peso aumentado)

**Cálculo de pred_win (expectativa de vitórias):**
A expectativa é calculada com a seguinte cascata de prioridades:
1. Talento da equipe (talent_avg) mapeado via regressão linear para win_pct
   - Para cada ano, ajusta-se: win_pct ~ talent_avg
   - Isso calibra a escala de performance dos jogadores para probabilidade de vitória
2. Pythagorean expectation (baseado em pontos feitos vs. sofridos)
   - Fórmula: PF^13.91 / (PF^13.91 + PA^13.91)
3. win_pct_team agregado (soma de vitórias/derrotas de todos coaches da equipe)

**Fórmula final:**
```
performance = z_win + z_over + 1.2 * z_rock
```

**Inputs (arquivos esperados):**
- data/raw/coaches.csv                    (obrigatório): registros de coach-season com won/lost
- data/processed/teams_cleaned.csv        (opcional):    PF/PA e confID por equipe-season
- data/processed/player_performance.csv   (opcional):    performance de jogadores, flag de rookie

**Outputs:**
- data/processed/coach_performance.csv : performance por coach-season
  Colunas: coachID, year, tmID, confID, won, lost, games, performance

**Notas:**
- Dados faltantes são tratados permissivamente (fallbacks para Pythagorean ou NaN)
- Z-scores são calculados separadamente por ano para controlar variância sazonal
- O módulo é determinístico e grava o CSV resultante em OUT_PATH
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


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
    """
    Calcula a expectativa Pitagórica de vitórias baseada em pontos feitos (PF) e sofridos (PA).
    Fórmula: win_pct_esperado = PF^exp / (PF^exp + PA^exp)
    onde exp ≈ 13.91 para WNBA (calibrado empiricamente).
    """
    exp = 13.91
    PF = pd.to_numeric(PF, errors='coerce')
    PA = pd.to_numeric(PA, errors='coerce')
    with np.errstate(divide="ignore", invalid="ignore"):
        pe = PF ** exp / (PF ** exp + PA ** exp)
    return pe


def _fit_predict_from_talent(base: pd.DataFrame) -> pd.Series:
    """
    Para cada ano, mapeia talent_avg -> win_pct_team via regressão linear.
    
    O talento médio da equipe (performance média dos jogadores) não está em escala 0-1,
    então precisamos calibrar a relação entre talento e win%. Este método:
    1. Para cada ano, ajusta uma regressão linear: win_pct_team ~ talent_avg
    2. Usa esse modelo para prever win_pct a partir de talent_avg
    3. Retorna as previsões (NaN quando não há dados suficientes)
    
    Retorna:
        Series com previsões de win_pct baseadas em talent_avg
    """
    pred = pd.Series(index=base.index, dtype=float)

    for year, gidx in base.groupby("year").groups.items():
        g = base.loc[gidx]
        # Precisamos de dados válidos para treinar
        mask = g["talent_avg"].notna() & g["win_pct_team"].notna()
        if mask.sum() >= 2:  # Mínimo de 2 pontos para ajustar uma linha
            X = g.loc[mask, "talent_avg"].values.reshape(-1, 1)
            y = g.loc[mask, "win_pct_team"].values
            try:
                lr = LinearRegression().fit(X, y)
                # Prever para todas as linhas daquele ano que têm talent_avg
                have_talent = g["talent_avg"].notna()
                if have_talent.sum() > 0:
                    pred.loc[gidx[have_talent]] = lr.predict(
                        g.loc[have_talent, "talent_avg"].values.reshape(-1, 1)
                    )
            except Exception:
                # Se a regressão falhar (dados colineares, etc), deixar como NaN
                pred.loc[gidx] = np.nan
        else:
            # Não há dados suficientes para esse ano
            pred.loc[gidx] = np.nan

    return pred


def compute_performance() -> pd.DataFrame:
    """
    Calcula a performance de cada coach por temporada.
    
    A métrica de performance combina três componentes (todos z-scores por ano):
    1. z_win: Sucesso absoluto (win%)
    2. z_over: Superação das expectativas (win% - pred_win%)
    3. z_rock: Desenvolvimento de rookies (1.2x peso)
    
    pred_win é calculado com prioridade:
    - talent_avg mapeado via regressão (se disponível)
    - pythagorean expectation (se PF/PA disponíveis)
    - win_pct_team agregado (fallback final)
    
    Retorna:
        DataFrame com colunas: coachID, year, tmID, confID, won, lost, games, performance
    """
    coaches = load_coaches()
    teams = load_teams()
    players = load_players()

    team_players = team_from_players(players)

    # Agregação de win_pct por equipe (soma de todos os coaches da equipe)
    agg = coaches.groupby(["year", "tmID"], as_index=False)[["won", "lost"]].sum()
    agg["games"] = agg["won"] + agg["lost"]
    agg["win_pct_team"] = agg.apply(lambda r: _safe_div(r["won"], r["games"]), axis=1)

    # Merge com dados das equipes (para obter PF/PA e confID)
    if teams is not None:
        agg = agg.merge(teams, on=["year", "tmID"], how="left")
        agg["pyth_exp"] = pythagorean(agg["PF"], agg["PA"])
    else:
        agg["confID"] = np.nan
        agg["PF"] = np.nan
        agg["PA"] = np.nan
        agg["pyth_exp"] = np.nan

    # Base de dados por coach
    base = coaches.merge(
        agg[["year", "tmID", "confID", "win_pct_team", "pyth_exp"]], 
        on=["year", "tmID"], 
        how="left"
    )

    # Merge com dados agregados de jogadores (talento da equipe, desenvolvimento de rookies)
    if team_players is not None:
        base = base.merge(team_players, on=["year", "tmID"], how="left")

    # ===== CÁLCULO DE pred_win =====
    # 1) Tentar usar talento da equipe (talent_avg) mapeado para win_pct via regressão
    if "talent_avg" in base.columns:
        base["pred_from_talent"] = _fit_predict_from_talent(base)
    else:
        base["pred_from_talent"] = np.nan

    # 2) Construir pred_win com cascata de fallbacks:
    #    pred_from_talent -> pyth_exp -> win_pct_team
    base["pred_win"] = base["pred_from_talent"]
    base["pred_win"] = base["pred_win"].fillna(base["pyth_exp"])
    base["pred_win"] = base["pred_win"].fillna(base["win_pct_team"])

    # Garantir que pred_win esteja entre 0 e 1
    base["pred_win"] = base["pred_win"].clip(lower=0.0, upper=1.0)

    # Calcular over/under performance
    base["over"] = base["win_pct"] - base["pred_win"]

    # ===== CÁLCULO DE Z-SCORES =====
    def z(g):
        """
        Calcula z-score (padronização).
        Se todos os valores são NaN ou std=0, retorna 0 para todos.
        """
        if g.isna().all():
            return pd.Series([0.0] * len(g), index=g.index)
        s = g.std(ddof=0)
        if s == 0 or pd.isna(s):
            return pd.Series([0.0] * len(g), index=g.index)
        return (g - g.mean()) / s

    # Z-scores são calculados por ano para controlar variância sazonal
    base["z_win"] = base.groupby("year")["win_pct"].transform(z)
    base["z_over"] = base.groupby("year")["over"].transform(z)
    
    # Z-score de desenvolvimento de rookies (se disponível)
    if "rookie_dev" in base.columns:
        base["z_rock"] = base.groupby("year")["rookie_dev"].transform(z)
    else:
        base["z_rock"] = 0.0

    # Garantir que NaN seja substituído por 0 nos z-scores
    base["z_win"] = base["z_win"].fillna(0)
    base["z_over"] = base["z_over"].fillna(0)
    base["z_rock"] = base["z_rock"].fillna(0)

    # ===== MÉTRICA FINAL DE PERFORMANCE =====
    # Combinação linear com peso aumentado para desenvolvimento de rookies
    base["performance"] = base["z_win"] + base["z_over"] + 1.2 * base["z_rock"]

    # Preparar output
    out = base[["coachID", "year", "tmID", "confID", "won", "lost", "games", "performance"]].copy()
    out.sort_values(["year", "coachID"], inplace=True)
    out.to_csv(OUT_PATH, index=False)
    
    print(f"✓ Coach performance calculado para {len(out)} linhas (coach-seasons)")
    print(f"✓ Arquivo salvo em: {OUT_PATH}")
    
    return out


if __name__ == "__main__":
    df = compute_performance()
