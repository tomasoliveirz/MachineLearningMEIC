from __future__ import annotations
from typing import Tuple, Dict

import numpy as np
import pandas as pd


def is_rookie(player_stats: pd.DataFrame) -> pd.Series:
    """
    Retorna uma Série booleana indicando se a linha representa um rookie.

    Um jogador é considerado rookie se aquela linha representa a primeira aparição dele no dataset,
    ou seja, se não há nenhuma temporada anterior para aquele bioID.
    Não considera rookies jogadores cuja primeira temporada é a mais antiga do dataset,
    pois não há dados anteriores para comparação.

    Requisitos mínimos das colunas no DataFrame de entrada: 'bioID', 'year'.

    Retorno: pd.Series index-alinhada a `player_stats` com valores True/False.
    """

    if 'bioID' not in player_stats.columns or 'year' not in player_stats.columns:
        raise ValueError("`player_stats` deve conter colunas 'bioID' e 'year' para determinar rookies")

    df = player_stats[['bioID', 'year']].copy()
    min_year = df['year'].min()
    # Para cada jogador, marca True apenas na primeira ocorrência (menor ano) e se não for o menor ano do dataset
    rookie_mask = (df.groupby('bioID')['year'].transform('min') == df['year']) & (df['year'] > min_year)
    return rookie_mask.reindex(player_stats.index, fill_value=False)

def calculate_rookies_perfomance_per_team_previous_seasons(
    player_stats: pd.DataFrame,
    performance_col: str = 'performance',
    seasons_back: int = 1,
    decay: float = 0.6,
    fillna: str = 'global_mean',
) -> pd.Series:
    """
    Calcula, para cada linha (team-season), a média ponderada (com decaimento temporal) da performance dos rookies daquele time
    nas `seasons_back` temporadas anteriores.

    - Para cada (tmID, year), busca rookies do time nas seasons anteriores (até seasons_back), calcula a média da performance deles,
      ponderando por decaimento exponencial (mais peso para temporadas mais recentes).
    - Retorna uma pd.Series alinhada a player_stats com a média histórica de rookies do time para cada linha.

    Parâmetros:
    - player_stats: DataFrame com colunas 'tmID', 'year', 'bioID' e a coluna de performance já calculada.
    - performance_col: nome da coluna com a performance por jogador/temporada (string).
    - seasons_back: número de temporadas anteriores a considerar (padrão 1).
    - decay: fator de decaimento exponencial para pesos temporais (0 < decay < 1).
    - fillna: estratégia quando não existem dados históricos. 'global_mean' (padrão) usa a média global
      dos rookies; 'zero' preenche com 0; ou um float específico pode ser passado (como string convertível).

    Retorna:
    - pd.Series alinhada a player_stats com o valor médio a ser usado para rookies daquela equipe/temporada.

    Explicação:
    - As temporadas mais recentes têm mais peso na média, porque são mais indicativas do desempenho atual dos rookies.
    - Os rookies da season teste não possuem dados anteriores, então a média histórica dos rookies do time é usada como referência.
    """

    required = {'tmID', 'year', 'bioID'}
    if not required.issubset(player_stats.columns):
        raise ValueError(f"player_stats precisa conter as colunas: {required}")
    if performance_col not in player_stats.columns:
        raise ValueError(f"Coluna de performance '{performance_col}' não encontrada em player_stats")

    df = player_stats.copy()
    df['is_rookie'] = is_rookie(df)

    # Média global dos rookies (para fallback)
    rookies_global_mean = df.loc[df['is_rookie'], performance_col].mean(skipna=True)

    # If performance_col contains mostly NaNs for rookies (or is absent), compute a per36-like metric from raw stats
    # to derive rookie means. Use same stat weights as in calculate_player_performance if needed.
    if df.loc[df['is_rookie'], performance_col].isna().all() or df.loc[df['is_rookie'], performance_col].eq(0).all():
        # compute per36 from available stats (defensive: missing columns -> treated as 0)
        stat_weights = {
            'pts': 1.0,
            'reb': 0.7,
            'ast': 0.7,
            'stl': 1.2,
            'blk': 1.2,
            'tov': -0.7,
        }

        def _safe_col(df_local, name):
            return df_local[name] if name in df_local.columns else pd.Series(0, index=df_local.index, dtype=float)

        # prefer 'trb' total rebounds, else sum of 'orb' and 'drb'
        reb_col = _safe_col(df, 'trb') if 'trb' in df.columns else (_safe_col(df, 'orb') + _safe_col(df, 'drb'))
        raw_score = (
            stat_weights['pts'] * _safe_col(df, 'pts') +
            stat_weights['reb'] * reb_col +
            stat_weights['ast'] * _safe_col(df, 'ast') +
            stat_weights['stl'] * _safe_col(df, 'stl') +
            stat_weights['blk'] * _safe_col(df, 'blk') +
            stat_weights['tov'] * _safe_col(df, 'tov')
        )

        minutes = _safe_col(df, 'mp').where(_safe_col(df, 'mp') > 0, _safe_col(df, 'g'))
        minutes = minutes.replace({0: np.nan})
        per36 = raw_score.divide(minutes).multiply(36)

        # Use per36 for rookies where performance_col is NaN or zero
        rookies_per36_mean = per36[df['is_rookie']].mean(skipna=True)
        if not pd.isna(rookies_per36_mean):
            rookies_global_mean = rookies_per36_mean

    # Para lookup rápido: rookies por (tmID, year)
    # compute rookie means per (tmID, year). Prefer provided performance_col; if it's zero/NaN for rookies,
    # fallback to per36 values computed above.
    rookies_subset = df.loc[df['is_rookie']].copy()
    # if performance values for rookies are missing/zero, try to replace with per36
    if 'per36' in locals():
        rookies_subset[performance_col] = rookies_subset[performance_col].where(~rookies_subset[performance_col].isna() & (rookies_subset[performance_col] != 0), per36[rookies_subset.index])

    rookies_by_team_year = (
        rookies_subset
        .groupby(['tmID', 'year'])[performance_col]
        .mean()
        .rename('rookie_mean')
    )
    rookies_map = rookies_by_team_year.to_dict()

    # Precompute para cada (tmID, year) a média ponderada dos rookies das seasons anteriores
    team_years = df[['tmID', 'year']].drop_duplicates().values
    prev_rookie_mean_cache: Dict[Tuple[str, int], float] = {}

    for tm, year in team_years:
        vals = []
        weights = []
        for idx, y in enumerate(range(year - seasons_back, year)[::-1]):  # mais recente primeiro
            key = (tm, y)
            if key in rookies_map and not pd.isna(rookies_map[key]):
                vals.append(rookies_map[key])
                weights.append(decay ** idx)
        if vals and sum(weights) > 0:
            prev_rookie_mean_cache[(tm, year)] = float(np.average(vals, weights=weights))
        else:
            # fallback
            if fillna == 'global_mean':
                prev_rookie_mean_cache[(tm, year)] = float(rookies_global_mean) if not pd.isna(rookies_global_mean) else 0.0
            elif fillna == 'zero':
                prev_rookie_mean_cache[(tm, year)] = 0.0
            else:
                try:
                    prev_rookie_mean_cache[(tm, year)] = float(fillna)
                except Exception:
                    prev_rookie_mean_cache[(tm, year)] = float(rookies_global_mean) if not pd.isna(rookies_global_mean) else 0.0

    # Assign para cada linha
    result = pd.Series(index=df.index, dtype=float)
    for idx, row in df[['tmID', 'year']].iterrows():
        key = (row['tmID'], int(row['year']))
        result.at[idx] = prev_rookie_mean_cache.get(key, float(rookies_global_mean) if not pd.isna(rookies_global_mean) else 0.0)

    df.drop(columns=['is_rookie'], inplace=True)
    return result

def calculate_player_performance(
    player_stats: pd.DataFrame,
    seasons_back: int = 9,
    decay: float = 0.6,
    weight_by_minutes: bool = True,
    rookie_seasons_back: int = 3,
    rookie_fillna: str = 'global_mean',
) -> pd.DataFrame:

    """Calcule uma pontuação de performance por jogador utilizando o histórico de temporadas anteriores.

    Contrato:
    - Entrada: `player_stats` DataFrame com, pelo menos, as colunas:
        - 'bioID' (identificador do jogador) e 'year' (temporada/ano) para ordenar e agrupar
        - 'tmID' (identificador do time) para calcular histórico de rookies por time
        - estatísticas por-temporada: 'pts', 'trb' ou ('orb' e 'drb'), 'ast', 'stl', 'blk', 'tov', 'mp' (minutos), 'g' (games)
        - opcionalmente colunas de time como 'team_pts' ou 'season_team_pts' se quiser incorporar força ofensiva do time
    - Saída: o mesmo DataFrame com uma nova coluna 'performance' (float) que representa a performance estimada

    Regras e escolhas de implementação:
    - Para cada jogador e temporada atual (linha com ano Y), a performance é estimada a partir das até
      `seasons_back` temporadas anteriores (anos < Y) do mesmo jogador.
    - Temporadas anteriores são combinadas usando um decaimento exponencial: peso = decay ** idx,
      onde idx=0 para a temporada imediatamente anterior, idx=1 para a anterior a essa, etc.
      Assim, temporadas mais recentes têm peso maior.
    - Cada temporada anterior contribui com uma métrica per-36 minutos (per36) calculada a partir de uma
      combinação linear de estatísticas (pts, reb, ast, stl, blk, tov). Se `weight_by_minutes=True`, o peso
      da temporada também é multiplicado pelo total de minutos daquela temporada para reduzir a influência
      de temporadas com muito poucos minutos.
    - Se não existirem temporadas anteriores (rookies), a função utiliza o histórico médio de performance
      dos rookies do mesmo time em temporadas anteriores, calculado via calculate_rookies_perfomance_per_team_previous_seasons.
    - A função é defensiva: colunas ausentes são tratadas como zeros e divisões por zero são evitadas.

    Parâmetros:
    - seasons_back: quantas temporadas anteriores considerar por jogador (default 9)
    - decay: fator de decaimento exponencial para pesos temporais (0 < decay < 1). Valores mais próximos de 1
      dão mais peso para temporadas mais antigas; valores pequenos concentram peso nas temporadas muito recentes.
    - weight_by_minutes: se True, multiplica o peso de cada temporada pelo total de minutos jogados nessa temporada.
    - rookie_seasons_back: quantas temporadas anteriores considerar para calcular a média de rookies do time (default 3).
    - rookie_fillna: estratégia de fallback quando não há histórico de rookies do time ('global_mean', 'zero', ou valor numérico).
    """

    df = player_stats.copy()

    # default stat weights para construir um raw score por temporada
    stat_weights: Dict[str, float] = {
        'pts': 1.0,
        'reb': 0.7,
        'ast': 0.7,
        'stl': 1.2,
        'blk': 1.2,
        'tov': -0.7,
    }

    def _safe_col(df_local: pd.DataFrame, name: str) -> pd.Series:
        """Return column if exists, else zeros series aligned to df_local."""
        return df_local[name] if name in df_local.columns else pd.Series(0, index=df_local.index, dtype=float)

    def _compute_raw_score(df_local: pd.DataFrame) -> pd.Series:
        """Compute linear combination of stats using stat_weights."""
        return (
            stat_weights['pts'] * _safe_col(df_local, 'pts') +
            stat_weights['reb'] * (_safe_col(df_local, 'trb') if 'trb' in df_local.columns else (_safe_col(df_local, 'orb') + _safe_col(df_local, 'drb'))) +
            stat_weights['ast'] * _safe_col(df_local, 'ast') +
            stat_weights['stl'] * _safe_col(df_local, 'stl') +
            stat_weights['blk'] * _safe_col(df_local, 'blk') +
            stat_weights['tov'] * _safe_col(df_local, 'tov')
        )

    def _compute_per36_and_minutes(df_local: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Return (per36, minutes) computed per-row."""
        raw = _compute_raw_score(df_local)
        minutes = _safe_col(df_local, 'mp').where(_safe_col(df_local, 'mp') > 0, _safe_col(df_local, 'g'))
        minutes = minutes.replace({0: np.nan})
        per36_local = raw.divide(minutes).multiply(36)
        minutes_total = _safe_col(df_local, 'mp').fillna(0)
        return per36_local, minutes_total

    def _compute_weighted_history(
        df_local: pd.DataFrame,
        per36_col: str = '_per36',
        minutes_col: str = '_minutes',
    ) -> pd.Series:
        """Compute for each row the weighted historical per36 using player history."""
        # Precompute global mean for rookie fallback
        global_mean = df_local[per36_col].mean(skipna=True)
        out = pd.Series(index=df_local.index, dtype=float)

        grouped_local = df_local.sort_values(['bioID', 'year']).groupby('bioID', sort=False)

        for _player, g in grouped_local:
            years = g['year'].values
            per36_vals = g[per36_col].values
            minutes_vals = g[minutes_col].values

            for i, idx in enumerate(g.index):
                year_i = years[i]
                prev_mask = years < year_i
                prev_indices = [j for j, m in enumerate(prev_mask) if m]

                if not prev_indices:
                    # rookie (no previous seasons): leave as NaN so rookies can be handled separately
                    out.at[idx] = np.nan
                    continue

                recent_prev = prev_indices[-seasons_back:][::-1]

                weights_list = []
                values_list = []
                for age, pi in enumerate(recent_prev):
                    val = per36_vals[pi]
                    mins = minutes_vals[pi]
                    if np.isnan(val):
                        continue
                    w = (decay ** age)
                    if weight_by_minutes:
                        w *= (mins if mins > 0 else 0.0)
                    weights_list.append(w)
                    values_list.append(val)

                if not weights_list:
                    # no valid previous-season per36 values: fallback to global mean
                    out.at[idx] = global_mean if not np.isnan(global_mean) else 0.0
                else:
                    weights_arr = np.array(weights_list, dtype=float)
                    vals_arr = np.array(values_list, dtype=float)
                    if weights_arr.sum() > 0:
                        out.at[idx] = float((weights_arr * vals_arr).sum() / weights_arr.sum())
                    else:
                        out.at[idx] = float(vals_arr.mean())

        return out

    def _apply_team_factor(df_local: pd.DataFrame, perf_series: pd.Series) -> pd.Series:
        tf = pd.Series(1.0, index=df_local.index)
        if 'team_pts' in df_local.columns:
            med = df_local['team_pts'].median(skipna=True)
            if med and med > 0:
                tf = df_local['team_pts'] / med
        elif 'season_team_pts' in df_local.columns:
            med = df_local['season_team_pts'].median(skipna=True)
            if med and med > 0:
                tf = df_local['season_team_pts'] / med
        final = perf_series * tf
        return final


    # Ensure year and bioID exist for grouping/sorting
    if 'year' not in df.columns or 'bioID' not in df.columns:
        # fallback: compute per-row per36 and return
        per36, _mins = _compute_per36_and_minutes(df)
        df['performance'] = per36.fillna(per36.mean()).astype(float).round(3)
        return df

    # Compute per36 and minutes columns used by history
    df['_per36'], df['_minutes'] = _compute_per36_and_minutes(df)

    # Compute player-history based performance
    perf_series = _compute_weighted_history(df, per36_col='_per36', minutes_col='_minutes')

    # Apply team-level factor if present
    final_perf = _apply_team_factor(df, perf_series)

    # Now handle rookies: fill NaN values with team-specific rookie history
    # First, we need to compute performance for non-rookies to have a baseline
    df['performance'] = final_perf.astype(float)
    
    # Identify rookies (rows with NaN performance)
    # also expose an explicit boolean 'rookie' column for the caller
    df['rookie'] = is_rookie(df)

    rookie_mask = df['performance'].isna()
    
    if rookie_mask.any():
        # Calculate team-specific rookie performance using historical data
        rookie_perf_series = calculate_rookies_perfomance_per_team_previous_seasons(
            player_stats=df,
            performance_col='performance',
            seasons_back=rookie_seasons_back,
            decay=decay,
            fillna=rookie_fillna,
        )
        
        # Fill rookie performances with team-specific historical averages
        df.loc[rookie_mask, 'performance'] = rookie_perf_series[rookie_mask]
    
    # Round to 3 decimals
    df['performance'] = df['performance'].round(3)

    # Ensure 'rookie' column is boolean and place it right after 'performance'
    df['rookie'] = df['rookie'].astype(bool)
    # Reorder to put 'rookie' immediately after 'performance' if both exist
    cols = list(df.columns)
    if 'performance' in cols and 'rookie' in cols:
        cols.remove('rookie')
        # insert rookie after performance
        perf_idx = cols.index('performance')
        cols.insert(perf_idx + 1, 'rookie')
        df = df[cols]

    # cleanup temporary columns
    df.drop(columns=[c for c in ['_per36', '_minutes'] if c in df.columns], inplace=True)

    return df
