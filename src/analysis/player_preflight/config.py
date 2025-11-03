#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Preflight-calibrated parameters for player performance model.

These values are determined by running the preflight analysis (run_preflight.py)
and should be used consistently across:
  - src/performance/player_performance.py
  - src/features/rookies.py
  - Any other player performance modeling

To recalibrate, run: make preflight
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PreflightConfig:
    """Calibrated parameters from player performance preflight analysis."""
    
    # Minimum minutes threshold to avoid extreme per-36 rates
    MIN_EFFECTIVE_MINUTES: int = 12
    
    # Minimum minutes for including rookies in calibration (RMSE-optimized)
    ROOKIE_MIN_MINUTES: int = 400
    
    # Bayesian shrinkage strength for rookies (equivalent minutes of prior)
    ROOKIE_PRIOR_STRENGTH: int = 900
    
    # Number of previous seasons to include in weighted average
    SEASONS_BACK: int = 3
    
    # Exponential decay weight for older seasons
    # Note: R² maximizes at 0.40, but we use 0.60 for interpretability (ΔR² < 0.01)
    # Interpretation: year t-1 gets 60% weight of year t, t-2 gets 36%, etc.
    DECAY: float = 0.60
    
    # Weight seasons by minutes played (vs equal weighting)
    # If True: season with 2000 minutes weighs more than season with 100 minutes
    WEIGHT_BY_MINUTES: bool = True


# Singleton instance
PREFLIGHT_PARAMS = PreflightConfig()


# Convenience accessors
def get_config() -> PreflightConfig:
    """Get the current preflight configuration."""
    return PREFLIGHT_PARAMS


def get_decay() -> float:
    """Get the configured temporal decay parameter."""
    return PREFLIGHT_PARAMS.DECAY


def get_rookie_threshold() -> int:
    """Get the configured rookie minutes threshold."""
    return PREFLIGHT_PARAMS.ROOKIE_MIN_MINUTES

