from __future__ import annotations
from typing import Tuple, Dict

import numpy as np
import pandas as pd

def is_rookie(player_stats: pd.DataFrame) -> pd.Series:
    """
    return a boolean series indicating whether each row is a rookie season.

    a row is a rookie if it is the player's first appearance in the dataset
    (no earlier season for that bioid) and the season is strictly greater than
    the dataset's minimum year (so we don't label the very first dataset year,
    where there is no prior info, as rookie).

    returns: a pd.series aligned to player_stats.index with true/false values.
    """
    if 'bioID' not in player_stats.columns or 'year' not in player_stats.columns:
        raise ValueError("player_stats must contain 'bioID' and 'year' to determine rookies")

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
    rookie_prior_strength: float = 3600.0
) -> pd.DataFrame:
    """
    compute a per-player performance metric using season history and explicit rookie rules.

    high-level logic:
    - for each player-season (row with year = y), performance blends the current season
      and up to `seasons_back` prior seasons from the same player (rookies only use the current season).
    - seasons are combined with exponential time weights:
        current season weight = 1.0
        prior season k steps back weight = decay ** k  (k=1 is the immediately previous season)
      if weight_by_minutes is true, each season's weight is also multiplied by minutes from that season.
    - each season contributes a per-36 value built from a weighted linear combo of stats:
        weights: pts=1.0, reb=0.7, ast=0.7, stl=1.2, blk=1.2, tov=-0.7
        per36 = (raw_score / minutes) * 36, with minutes from 'mp' (fallback 'g'); tiny minutes are floored.
    - a team factor adjustment is applied if 'team_pts' or 'season_team_pts' exists:
        multiply by team_pts / median(team_pts) (or season_team_pts / median(season_team_pts)).

    rookie-specific rules:
    - a rookie is the first dataset season for a player (excluding the dataset's minimum year).
    - if a rookie has minutes >= rookie_min_minutes, use observed per36 (fully trusted).
    - if minutes < rookie_min_minutes, apply empirical bayes shrinkage:
        performance = (w_obs * per36_obs + w_prior * prior_team) / (w_obs + w_prior)
        w_obs   = minutes / 36
        w_prior = rookie_prior_strength / 36
        prior_team = mean per36 for rookies on the same team (fallback to global rookie mean)
      if minutes == 0, use 100% prior.

    parameters:
    - seasons_back (int): how many previous seasons to consider (default 9)
    - decay (float): temporal decay factor (0 < decay < 1)
    - weight_by_minutes (bool): whether to scale weights by minutes per season
    - rookie_min_minutes (float): minutes threshold to fully trust rookie per36
    - rookie_prior_strength (float): prior strength in "equivalent minutes" (default 3600 ≈ 100 full games)

    returns:
    - dataframe with columns 'performance' (float, 3 decimals) and 'rookie' (bool), preserving other columns.
    """
    # step 1: copy and basic validation
    df = player_stats.copy()

    if 'year' not in df.columns or 'bioID' not in df.columns:
        # fallback: if required keys are missing, just compute raw per36 and fill missing with mean
        per36, _ = _compute_per36_metrics(df)
        df['performance'] = per36.fillna(per36.mean()).astype(float).round(3)
        return df

    # step 2: base metrics (per36 and minutes) and rookie flag
    df['_per36'], df['_minutes'] = _compute_per36_metrics(df)
    df['rookie'] = is_rookie(df)

    # step 3: build rookie priors (global and by team)
    global_per36_mean, global_rookie_mean, rookie_team_prior = _build_rookie_priors(df)

    # step 4: historical performance (time-weighted, optional minute-weighted)
    perf_series = _compute_weighted_history(
        df,
        per36_col='_per36',
        minutes_col='_minutes',
        seasons_back=seasons_back,
        decay=decay,
        weight_by_minutes=weight_by_minutes
    )

    # step 5: team environment adjustment
    final_perf = _apply_team_factor(df, perf_series)
    df['performance'] = final_perf.astype(float)

    # step 6: empirical bayes shrinkage for low-minute players
    df = _apply_shrinkage_corrections(
        df,
        global_per36_mean=global_per36_mean,
        global_rookie_mean=global_rookie_mean,
        rookie_team_prior=rookie_team_prior,
        rookie_min_minutes=rookie_min_minutes,
        rookie_prior_strength=rookie_prior_strength
    )

    # step 7: finalize output columns and cleanup
    df = _finalize_performance_dataframe(df)
    return df


def _compute_per36_metrics(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    compute per-36 contribution and total minutes for each row.

    returns:
        (per36, minutes_total)
    """
    # stat weights for a simple linear score
    stat_weights = {
        'pts': 1.0, 'reb': 0.7, 'ast': 0.7,
        'stl': 1.2, 'blk': 1.2, 'tov': -0.7,
    }

    def _safe_col(name: str) -> pd.Series:
        # return column if present, else a zero series with correct index and float dtype
        return df[name] if name in df.columns else pd.Series(0, index=df.index, dtype=float)

    # pick rebounds: 'trb' if available, else sum of 'orb' + 'drb'
    reb_col = _safe_col('trb') if 'trb' in df.columns else (_safe_col('orb') + _safe_col('drb'))

    # linear raw score using the weights above
    raw_score = (
        stat_weights['pts'] * _safe_col('pts') +
        stat_weights['reb'] * reb_col +
        stat_weights['ast'] * _safe_col('ast') +
        stat_weights['stl'] * _safe_col('stl') +
        stat_weights['blk'] * _safe_col('blk') +
        stat_weights['tov'] * _safe_col('tov')
    )

    # minutes: prefer 'mp'; if not positive, fall back to 'g' (games) as a rough proxy
    minutes = _safe_col('mp').where(_safe_col('mp') > 0, _safe_col('g'))
    minutes = minutes.replace({0: np.nan})

    # enforce a minimum effective minutes for per-36 to avoid exploding tiny samples
    MIN_EFFECTIVE_MINUTES = 12.0
    minutes_for_per36 = minutes.copy()
    mask_small = minutes_for_per36.notna() & (minutes_for_per36 < MIN_EFFECTIVE_MINUTES)
    minutes_for_per36.loc[mask_small] = MIN_EFFECTIVE_MINUTES

    # per-36 formula and total minutes (use mp; nan -> 0)
    per36 = raw_score.divide(minutes_for_per36).multiply(36)
    minutes_total = _safe_col('mp').fillna(0)

    return per36, minutes_total


def _build_rookie_priors(df: pd.DataFrame) -> Tuple[float, float, Dict[str, float]]:
    """
    build rookie priors from the current dataframe.

    returns:
      - global_per36_mean: overall mean of per36
      - global_rookie_mean: mean per36 among rookies (fallback to global if none)
      - rookie_team_prior: dict tmid -> mean rookie per36 for that team
    """
    global_per36_mean = df['_per36'].mean(skipna=True)

    rookies_df = df[df['rookie']].copy()
    global_rookie_mean = rookies_df['_per36'].mean(skipna=True)

    if pd.isna(global_rookie_mean):
        global_rookie_mean = global_per36_mean

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
    compute time-weighted historical performance for each player-season.

    rookies: only use current season.
    non-rookies: use current season + up to `seasons_back` prior seasons.
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

            # always include current season
            current_val = per36_vals[i]
            current_mins = minutes_vals[i]
            if not np.isnan(current_val):
                w = 1.0
                if weight_by_minutes:
                    w *= (current_mins if current_mins > 0 else 0.0)
                weights_list.append(w)
                values_list.append(current_val)

            # add up to seasons_back previous seasons for non-rookies
            if prev_indices:
                recent_prev = prev_indices[-seasons_back:][::-1]  # most recent first
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

            # weighted average (fallbacks if needed)
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
    apply a multiplicative team environment factor based on team scoring.
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
    apply empirical bayes shrinkage for players with low minutes.
    """
    per36 = df['_per36']
    minutes = df['_minutes']
    rookie_mask = df['rookie']

    def _calculate_shrinkage(obs_per36, obs_minutes, prior_mean, is_rookie_flag):
        """
        blend observed per36 and prior using weights proportional to minutes.
        returns the shrunken estimate.
        """
        if pd.isna(obs_per36) or obs_minutes == 0:
            return prior_mean

        MIN_TRUST_MINUTES = 36.0
        threshold = (
            max(rookie_min_minutes, MIN_TRUST_MINUTES)
            if is_rookie_flag
            else max(rookie_min_minutes * 2, MIN_TRUST_MINUTES)
        )

        if obs_minutes >= threshold:
            return obs_per36

        # base weights
        w_obs = max(float(obs_minutes), 0.0) / 36.0
        w_prior = rookie_prior_strength / 36.0

        # for rookies, increase prior weight as minutes get very small
        if is_rookie_flag:
            w_prior *= rookie_min_minutes / max(obs_minutes, 1.0)
            if obs_minutes < 50:
                w_prior *= 5.0
            if obs_minutes < 10:
                w_prior *= 2.0

        denom = w_obs + w_prior
        if denom <= 0:
            return prior_mean

        blended = (w_obs * obs_per36 + w_prior * prior_mean) / denom

        # cap extreme outliers for very low-minute rookies
        if is_rookie_flag and obs_minutes < 20:
            cap = global_per36_mean + 1.5 * df['_per36'].std(skipna=True)
            blended = min(blended, cap)

        return blended

    # iterate rows where minutes are low and apply shrinkage
    for idx in df.index:
        obs_per36_val = per36.at[idx]
        obs_minutes_val = minutes.at[idx]
        is_rookie_player = rookie_mask.at[idx]

        # choose the appropriate prior
        if is_rookie_player:
            tmID = df.at[idx, 'tmID'] if 'tmID' in df.columns else None
            prior_mean_val = rookie_team_prior.get(tmID, global_rookie_mean) if tmID else global_rookie_mean
        else:
            prior_mean_val = global_per36_mean

        # apply shrinkage when minutes are below the non-rookie threshold gate
        if obs_minutes_val < rookie_min_minutes * 2:
            df.at[idx, 'performance'] = _calculate_shrinkage(
                obs_per36_val, obs_minutes_val, prior_mean_val, is_rookie_player
            )

    return df


def _finalize_performance_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    finalize the output: round performance, order columns, drop temp columns.
    """
    # round performance to 3 decimals
    df['performance'] = df['performance'].round(3)

    # ensure rookie is boolean
    df['rookie'] = df['rookie'].astype(bool)

    # place 'rookie' right after 'performance' if both exist
    cols = list(df.columns)
    if 'performance' in cols and 'rookie' in cols:
        cols.remove('rookie')
        perf_idx = cols.index('performance')
        cols.insert(perf_idx + 1, 'rookie')
        df = df[cols]

    # drop temporary columns
    df.drop(columns=[c for c in ['_per36', '_minutes'] if c in df.columns], inplace=True)

    return df
