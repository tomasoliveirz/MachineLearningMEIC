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

    Retorno: pd.Series index-alinhada a `player_stats` com valores True/False.
    """

    if 'bioID' not in player_stats.columns or 'year' not in player_stats.columns:
        raise ValueError("`player_stats` deve conter colunas 'bioID' e 'year' para determinar rookies")

    df = player_stats[['bioID', 'year']].copy()
    min_year = df['year'].min()
    rookie_mask = (df.groupby('bioID')['year'].transform('min') == df['year']) & (df['year'] > min_year)
    return rookie_mask.reindex(player_stats.index, fill_value=False)

def calculate_player_performance(
    player_stats: pd.DataFrame,
    seasons_back: int = 9,
    decay: float = 0.6,
    weight_by_minutes: bool = True,
    rookie_min_minutes: float = 100.0,
    rookie_prior_strength: float = 3600.0) -> pd.DataFrame:
    """Calcula a performance por jogador usando histórico de temporadas e regras explícitas para rookies.
    Comportamento detalhado:
    - Para cada jogador/temporada (linha com year = Y), a performance é calculada a partir da temporada atual (Y) e das até
        `seasons_back` temporadas anteriores (anos < Y) do mesmo jogador, exceto para rookies que usam apenas a temporada atual.
    - Temporadas são agregadas com pesos temporais exponenciais: peso = 1.0 para a temporada atual, e decay ** (age + 1) para anteriores,
        onde age=0 é a temporada imediatamente anterior. Se `weight_by_minutes=True`, o peso de cada temporada
        também é multiplicado pelos minutos jogados nessa temporada (minutos maiores aumentam influência).
    - Cada temporada contribui com um valor per-36 (per36) gerado a partir de uma combinação linear das estatísticas
        (pesos internos: pts=1.0, reb=0.7, ast=0.7, stl=1.2, blk=1.2, tov=-0.7). O per36 é calculado como
        raw_score / minutes * 36; minutos faltantes são tratados com as colunas 'mp' ou 'g' quando possível.
    - Aplica-se um fator de time (team factor) multiplicativo se 'team_pts' ou 'season_team_pts' estiver presente
        (team_pts mediana usada como base).

    Regras para rookies (casos específicos):
    - Um jogador é considerado rookie quando a linha representa sua primeira aparição no DataFrame e essa
        temporada não é a mais antiga do dataset (veja `is_rookie`).
    - Se rookie TEM >= rookie_min_minutes (padrão: 100), usa per36 observado (confiável).
    - Se rookie TEM < rookie_min_minutes, aplica SHRINKAGE (Empirical Bayes):
        * performance = (w_obs * per36_observado + w_prior * prior_time) / (w_obs + w_prior)
        * w_obs = minutos / 36 (quanto mais minutos, mais confiança na observação)
        * w_prior = rookie_prior_strength / 36 (força do prior, padrão: 100 "games" de 36min)
        * prior_time = média histórica dos rookies do mesmo time
    - Se rookie NÃO TEM minutos (0), usa 100% o prior do time.

    Parâmetros:
    - seasons_back (int): quantas temporadas anteriores considerar por jogador (default 9)
    - decay (float): fator de decaimento temporal (0 < decay < 1)
    - weight_by_minutes (bool): multiplicar pesos por minutos da temporada
    - rookie_min_minutes (float): threshold de minutos para confiar totalmente no per36 do rookie (padrão: 100)
    - rookie_prior_strength (float): força do prior em "minutos equivalentes" para shrinkage (padrão: 3600 = 100 jogos)
    
    Retorno:
    - DataFrame modificado com as colunas 'performance' (float, 3 casas decimais) e 'rookie' (bool).
    """
    
    # ========== STEP 1: Initialize and validate ==========
    df = player_stats.copy()
    
    if 'year' not in df.columns or 'bioID' not in df.columns:
        # Fallback: compute simple per36 if required columns missing
        per36, _ = _compute_per36_metrics(df)
        df['performance'] = per36.fillna(per36.mean()).astype(float).round(3)
        return df
    
    # ========== STEP 2: Compute base metrics ==========
    df['_per36'], df['_minutes'] = _compute_per36_metrics(df)
    df['rookie'] = is_rookie(df)
    
    # ========== STEP 3: Build rookie priors (vectorized) ==========
    global_per36_mean, global_rookie_mean, rookie_team_prior = _build_rookie_priors(df)
    
    # ========== STEP 4: Calculate historical performance ==========
    perf_series = _compute_weighted_history(
        df, 
        per36_col='_per36', 
        minutes_col='_minutes',
        seasons_back=seasons_back,
        decay=decay,
        weight_by_minutes=weight_by_minutes
    )
    
    # ========== STEP 5: Apply team adjustment factor ==========
    final_perf = _apply_team_factor(df, perf_series)
    df['performance'] = final_perf.astype(float)
    
    # ========== STEP 6: Apply shrinkage for low-minute players ==========
    df = _apply_shrinkage_corrections(
        df,
        global_per36_mean=global_per36_mean,
        global_rookie_mean=global_rookie_mean,
        rookie_team_prior=rookie_team_prior,
        rookie_min_minutes=rookie_min_minutes,
        rookie_prior_strength=rookie_prior_strength
    )
    
    # ========== STEP 7: Finalize and cleanup ==========
    df = _finalize_performance_dataframe(df)
    
    return df


def _compute_per36_metrics(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Calcula per36 e minutos totais para cada linha do DataFrame.
    
    Returns:
        Tuple[pd.Series, pd.Series]: (per36, minutes_total)
    """
    stat_weights = {
        'pts': 1.0, 'reb': 0.7, 'ast': 0.7,
        'stl': 1.2, 'blk': 1.2, 'tov': -0.7,
    }
    
    def _safe_col(name: str) -> pd.Series:
        return df[name] if name in df.columns else pd.Series(0, index=df.index, dtype=float)
    
    # Compute raw score from weighted stats
    reb_col = _safe_col('trb') if 'trb' in df.columns else (_safe_col('orb') + _safe_col('drb'))
    raw_score = (
        stat_weights['pts'] * _safe_col('pts') +
        stat_weights['reb'] * reb_col +
        stat_weights['ast'] * _safe_col('ast') +
        stat_weights['stl'] * _safe_col('stl') +
        stat_weights['blk'] * _safe_col('blk') +
        stat_weights['tov'] * _safe_col('tov')
    )
    
    # Compute minutes (prefer 'mp', fallback to 'g')
    minutes = _safe_col('mp').where(_safe_col('mp') > 0, _safe_col('g'))
    minutes = minutes.replace({0: np.nan})
    
    # Apply minimum effective minutes for per36 calculation
    MIN_EFFECTIVE_MINUTES = 12.0
    minutes_for_per36 = minutes.copy()
    mask_small = minutes_for_per36.notna() & (minutes_for_per36 < MIN_EFFECTIVE_MINUTES)
    minutes_for_per36.loc[mask_small] = MIN_EFFECTIVE_MINUTES
    
    per36 = raw_score.divide(minutes_for_per36).multiply(36)
    minutes_total = _safe_col('mp').fillna(0)
    
    return per36, minutes_total


def _build_rookie_priors(df: pd.DataFrame) -> Tuple[float, float, Dict[str, float]]:
    """
    Constrói priors para rookies baseado em dados históricos.
    
    Returns:
        Tuple contendo:
        - global_per36_mean: média global de per36
        - global_rookie_mean: média global de per36 apenas para rookies
        - rookie_team_prior: dicionário {tmID: mean_per36} para rookies por time
    """
    global_per36_mean = df['_per36'].mean(skipna=True)
    
    rookies_df = df[df['rookie']].copy()
    global_rookie_mean = rookies_df['_per36'].mean(skipna=True)
    
    if pd.isna(global_rookie_mean):
        global_rookie_mean = global_per36_mean
    
    # Vectorized: group rookies by team and calculate mean
    rookie_team_prior = (
        rookies_df
        .groupby('tmID')['_per36']
        .mean()
        .to_dict()
    )
    
    return global_per36_mean, global_rookie_mean, rookie_team_prior


def _compute_weighted_history(
    df: pd.DataFrame,
    per36_col: str,
    minutes_col: str,
    seasons_back: int,
    decay: float,
    weight_by_minutes: bool
) -> pd.Series:
    """
    Calcula performance histórica ponderada para cada jogador/temporada.
    
    Rookies usam apenas temporada atual.
    Não-rookies usam temporada atual + até seasons_back temporadas anteriores.
    """
    global_mean = df[per36_col].mean(skipna=True)
    out = pd.Series(index=df.index, dtype=float)
    
    grouped = df.sort_values(['bioID', 'year']).groupby('bioID', sort=False)
    
    for _player, g in grouped:
        years = g['year'].values
        per36_vals = g[per36_col].values
        minutes_vals = g[minutes_col].values
        
        for i, idx in enumerate(g.index):
            year_i = years[i]
            prev_indices = [j for j in range(i) if years[j] < year_i]
            
            weights_list = []
            values_list = []
            
            # Current season (always included)
            current_val = per36_vals[i]
            current_mins = minutes_vals[i]
            if not np.isnan(current_val):
                w = 1.0
                if weight_by_minutes:
                    w *= (current_mins if current_mins > 0 else 0.0)
                weights_list.append(w)
                values_list.append(current_val)
            
            # Previous seasons (only for non-rookies)
            if prev_indices:
                recent_prev = prev_indices[-seasons_back:][::-1]
                for age, pi in enumerate(recent_prev):
                    val = per36_vals[pi]
                    mins = minutes_vals[pi]
                    if np.isnan(val):
                        continue
                    w = (decay ** (age + 1))
                    if weight_by_minutes:
                        w *= (mins if mins > 0 else 0.0)
                    weights_list.append(w)
                    values_list.append(val)
            
            # Calculate weighted average
            if not weights_list:
                out.at[idx] = global_mean if not np.isnan(global_mean) else 0.0
            else:
                weights_arr = np.array(weights_list, dtype=float)
                vals_arr = np.array(values_list, dtype=float)
                if weights_arr.sum() > 0:
                    out.at[idx] = float((weights_arr * vals_arr).sum() / weights_arr.sum())
                else:
                    out.at[idx] = float(vals_arr.mean())
    
    return out


def _apply_team_factor(df: pd.DataFrame, perf_series: pd.Series) -> pd.Series:
    """
    Aplica fator de ajuste baseado na pontuação do time.
    """
    tf = pd.Series(1.0, index=df.index)
    
    if 'team_pts' in df.columns:
        med = df['team_pts'].median(skipna=True)
        if med and med > 0:
            tf = df['team_pts'] / med
    elif 'season_team_pts' in df.columns:
        med = df['season_team_pts'].median(skipna=True)
        if med and med > 0:
            tf = df['season_team_pts'] / med
    
    return perf_series * tf


def _apply_shrinkage_corrections(
    df: pd.DataFrame,
    global_per36_mean: float,
    global_rookie_mean: float,
    rookie_team_prior: Dict[str, float],
    rookie_min_minutes: float,
    rookie_prior_strength: float
) -> pd.DataFrame:
    """
    Aplica correções de shrinkage para jogadores com poucos minutos.
    """
    per36 = df['_per36']
    minutes = df['_minutes']
    rookie_mask = df['rookie']
    
    def _calculate_shrinkage(obs_per36, obs_minutes, prior_mean, is_rookie_flag):
        """Calcula valor com shrinkage (Empirical Bayes)."""
        if pd.isna(obs_per36) or obs_minutes == 0:
            return prior_mean
        
        MIN_TRUST_MINUTES = 36.0
        threshold = max(rookie_min_minutes, MIN_TRUST_MINUTES) if is_rookie_flag else max(rookie_min_minutes * 2, MIN_TRUST_MINUTES)
        
        if obs_minutes >= threshold:
            return obs_per36
        
        # Weights for Bayesian shrinkage
        w_obs = max(float(obs_minutes), 0.0) / 36.0
        w_prior = rookie_prior_strength / 36.0
        
        # Adjust prior weight for rookies based on uncertainty
        if is_rookie_flag:
            w_prior *= rookie_min_minutes / max(obs_minutes, 1.0)
            
            # Extra reinforcement for very low minutes
            if obs_minutes < 50:
                w_prior *= 5.0
            if obs_minutes < 10:
                w_prior *= 2.0
        
        denom = w_obs + w_prior
        if denom <= 0:
            return prior_mean
        
        blended = (w_obs * obs_per36 + w_prior * prior_mean) / denom
        
        # Cap extreme values for rookies with very few minutes
        if is_rookie_flag and obs_minutes < 20:
            cap = global_per36_mean + 1.5 * df['_per36'].std(skipna=True)
            blended = min(blended, cap)
        
        return blended
    
    # Apply shrinkage to players with low minutes
    for idx in df.index:
        obs_per36_val = per36.at[idx]
        obs_minutes_val = minutes.at[idx]
        is_rookie_player = rookie_mask.at[idx]
        
        # Determine appropriate prior
        if is_rookie_player:
            tmID = df.at[idx, 'tmID'] if 'tmID' in df.columns else None
            prior_mean_val = rookie_team_prior.get(tmID, global_rookie_mean) if tmID else global_rookie_mean
        else:
            prior_mean_val = global_per36_mean
        
        # Apply shrinkage if below threshold
        if obs_minutes_val < rookie_min_minutes * 2:
            df.at[idx, 'performance'] = _calculate_shrinkage(
                obs_per36_val, obs_minutes_val, prior_mean_val, is_rookie_player
            )
    
    return df


def _finalize_performance_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Finaliza o DataFrame: arredonda, organiza colunas e remove temporárias.
    """
    # Round performance to 3 decimals
    df['performance'] = df['performance'].round(3)
    
    # Ensure rookie column is boolean
    df['rookie'] = df['rookie'].astype(bool)
    
    # Reorder columns: put 'rookie' right after 'performance'
    cols = list(df.columns)
    if 'performance' in cols and 'rookie' in cols:
        cols.remove('rookie')
        perf_idx = cols.index('performance')
        cols.insert(perf_idx + 1, 'rookie')
        df = df[cols]
    
    # Remove temporary columns
    df.drop(columns=[c for c in ['_per36', '_minutes'] if c in df.columns], inplace=True)
    
    return df