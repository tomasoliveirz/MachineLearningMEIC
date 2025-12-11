"""
Preflight-calibrated parameters for player performance model.

These values are determined by running the preflight analysis (run_preflight.py)
and should be used consistently across:
  - src/performance/player_performance.py
  - src/features/rookies.py
  - Any other player performance modeling

"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PreflightConfig:
    """Core parameters for player performance analysis (basic mode)."""
    
    # Minimum minutes threshold to avoid extreme per-36 rates
    MIN_EFFECTIVE_MINUTES: int = 12
    
    # Number of previous seasons to include in weighted average
    SEASONS_BACK: int = 3
    
    # Exponential decay weight for older seasons
    # Interpretation: year t-1 gets 60% weight of year t, t-2 gets 36%, etc.
    DECAY: float = 0.60
    
    # Weight seasons by minutes played (vs equal weighting)
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

