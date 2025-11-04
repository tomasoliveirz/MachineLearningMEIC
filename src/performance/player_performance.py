from pathlib import Path
import sys
import pandas as pd
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

def _load_weights(weights_path: str):
    """Load weights file and normalize to a mapping pos -> {stat: weight}.
    Preserves original supported formats.
    """
    with open(weights_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "positions" in raw:
        pos_block = raw.get("positions", {})
        weights = {}
        for pos_key, pos_info in pos_block.items():
            if isinstance(pos_info, dict) and "weights" in pos_info:
                weights[pos_key] = pos_info.get("weights", {})
            elif isinstance(pos_info, dict):
                weights[pos_key] = pos_info
            else:
                weights[pos_key] = {}
    else:
        weights = raw
    return weights


def _collect_stats_from_weights(weights: dict):
    stats = set()
    for pos_w in weights.values():
        stats.update(pos_w.keys())
    return sorted(stats)


def _compute_per90_columns(df: pd.DataFrame, stats: list, normalize_per90: bool):
    """Ensure stat columns exist and optionally compute per90 temporary cols.
    Returns mapping stat -> column to use for weighting.
    """
    if normalize_per90:
        mins = df["minutes"].astype(float)
        for s in stats:
            per90_col = f"__{s}_per90"
            if s in df.columns:
                df[s] = pd.to_numeric(df[s], errors="coerce").fillna(0.0)
                df[per90_col] = 0.0
                positive_mask = mins > 0
                df.loc[positive_mask, per90_col] = df.loc[positive_mask, s] * 90.0 / mins[positive_mask]
            else:
                df[per90_col] = 0.0
        stat_cols = {s: f"__{s}_per90" for s in stats}
    else:
        for s in stats:
            if s not in df.columns:
                df[s] = 0.0
        stat_cols = {s: s for s in stats}
    return stat_cols


def _apply_weights(df: pd.DataFrame, weights: dict, stat_cols: dict, perf_col: str):
    """Apply positional weights to compute perf_col in-place."""
    for pos, wdict in weights.items():
        mask = df["position"] == pos
        if not mask.any():
            continue
        for stat, w in wdict.items():
            col = stat_cols.get(stat)
            if col is None:
                continue
            df.loc[mask, perf_col] = df.loc[mask, perf_col] + df.loc[mask, col].fillna(0) * float(w)


def _handle_unknown_position(df: pd.DataFrame, weights: dict, perf_col: str):
    """If 'unknown' in weights and there are players with position 'unknown',
    assign them the global mean of known positions (or 0 if na).
    """
    if "unknown" in weights:
        unk_mask = df["position"] == "unknown"
        if unk_mask.any():
            known_mask = ~unk_mask
            if known_mask.any():
                global_mean = df.loc[known_mask, perf_col].mean()
            else:
                global_mean = df[perf_col].mean()
            if pd.isna(global_mean):
                global_mean = 0.0
            df.loc[unk_mask, perf_col] = global_mean


def _apply_min_minutes_fallback(df: pd.DataFrame, min_minutes: int, perf_col: str, fallback: str):
    """Apply fallback logic for players below min_minutes.
    Modifies df in-place and returns.
    """
    if min_minutes is not None and min_minutes > 0:
        low_mask = df["minutes"] < min_minutes
        if low_mask.any():
            df["team_pos_mean"] = df.groupby(["team_id", "year", "position"])[perf_col].transform("mean")
            df["team_mean"] = df.groupby(["team_id", "year"])[perf_col].transform("mean")

            if fallback == "team_position_mean":
                global_mean = df[perf_col].mean()
                replacements = df.loc[low_mask, "team_pos_mean"].fillna(df.loc[low_mask, "team_mean"]).fillna(global_mean)
                df.loc[low_mask, perf_col] = replacements.values
            else:
                raise ValueError("fallback inválido; use 'team_position_mean' ou 'team_mean'")

            df.drop(columns=["team_pos_mean", "team_mean"], inplace=True, errors="ignore")


def _remove_per90_columns(df: pd.DataFrame, stats: list):
    for s in stats:
        tmp = f"__{s}_per90"
        if tmp in df.columns:
            df.drop(columns=[tmp], inplace=True)

def calculate_player_performance(
    df: pd.DataFrame,
    weights_path: str = "src/performance/weights_positions.json",
    min_minutes: int = 400,
    fallback: str = "team_position_mean", 
    normalize_per90: bool = True,
    perf_col: str = "performance",
) -> pd.DataFrame:

    # carregar pesos
    weights = _load_weights(weights_path)

    # verificar colunas mínimas
    for c in ("team_id", "year", "position", "minutes"):
        if c not in df.columns:
            raise ValueError(f"Coluna obrigatória ausente no DataFrame: {c}")

    df = df.copy()
    # garantir tipos numéricos básicos
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0.0)

    # Construir lista de estatísticas usadas nos pesos
    stats = _collect_stats_from_weights(weights)

    # garantir e (se pedido) computar colunas per90
    stat_cols = _compute_per90_columns(df, stats, normalize_per90)

    # inicializar coluna de performance
    df[perf_col] = 0.0

    # aplicar pesos por posição
    _apply_weights(df, weights, stat_cols, perf_col)

    # tratar posição unknown
    _handle_unknown_position(df, weights, perf_col)

    # aplicar fallback para minutos baixos
    _apply_min_minutes_fallback(df, min_minutes, perf_col, fallback)

    # remover colunas temporárias per90
    if normalize_per90:
        _remove_per90_columns(df, stats)

    return df

def main():
    pt_path = RAW / "players_teams.csv"
    players_path = PROC / "players_cleaned.csv"
    weights_path = ROOT / "src" / "performance" / "weights_positions.json"

    df_pt = pd.read_csv(pt_path)
    df_players = pd.read_csv(players_path)

    # normalizar colunas esperadas pela função
    df = df_pt.copy()
    df["team_id"] = df["tmID"]
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce")

    # mapear posição a partir de players.csv (bioID == playerID)
    pos_map = dict(zip(df_players["bioID"].astype(str), df_players["pos"].astype(str)))
    df["position"] = df["playerID"].astype(str).map(pos_map).fillna("unknown")

    # garantir existência e tipo numérico das estatísticas usadas nos pesos
    for s in ["points", "rebounds", "assists", "steals", "blocks", "turnovers", "PF"]:
        if s not in df.columns:
            df[s] = 0.0
        else:
            df[s] = pd.to_numeric(df[s], errors="coerce").fillna(0.0)

    # calcular performance
    out = calculate_player_performance(
        df,
        weights_path=str(weights_path),
        min_minutes=500,
        fallback="team_position_mean",
        normalize_per90=True,
        perf_col="performance"
    )

    # salvar apenas colunas relevantes
    cols_to_save = ["playerID","year","team_id","position","minutes","points","rebounds","assists","steals","blocks","turnovers","PF","performance",]

    out_path = PROC / "player_performance.csv"
    out[cols_to_save].to_csv(out_path, index=False)

    print(f"Saved player performance to: {out_path}")
    print(out[cols_to_save].head())

if __name__ == "__main__":
    main()