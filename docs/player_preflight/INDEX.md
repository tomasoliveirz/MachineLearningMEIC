# Player Performance Preflight Documentation

Comprehensive documentation for the player performance calibration system.

## 📚 Documentation Files

### [README.md](README.md)
Complete overview of the preflight system:
- Purpose and goals
- Package structure
- Usage instructions
- Key outputs and their interpretation
- Integration with main model

### [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
Quick reference guide:
- Run commands
- Key outputs table
- Parameter interpretations
- Common issues and fixes
- Troubleshooting

### [TECHNICAL_DECISIONS.md](TECHNICAL_DECISIONS.md)
Deep dive into technical decisions:
- Why we chose decay=0.6 over 0.4
- IPW clamping vs truncation
- Rookie threshold rationale
- Prior strength calibration
- All trade-offs documented

## 🚀 Quick Start

```bash
# Run calibration
make preflight

# Review outputs
cat reports/player_preflight/preflight_report.md

# Use in code
from src.analysis.player_preflight.config import PREFLIGHT_PARAMS
```

## 📂 Related Documentation

- [Migration Guide](../../MIGRATION_PREFLIGHT.md) - How we got here from the monolithic script
- [Refactoring Summary](../../REFACTORING_SUMMARY.md) - What changed and why
- [Main README](../../README.md) - Project overview

## 🔍 Key Concepts

### Preflight Parameters

All calibrated parameters live in `src/analysis/player_preflight/config.py`:

```python
@dataclass(frozen=True)
class PreflightConfig:
    MIN_EFFECTIVE_MINUTES: int = 12
    ROOKIE_MIN_MINUTES: int = 400
    ROOKIE_PRIOR_STRENGTH: int = 900
    SEASONS_BACK: int = 3
    DECAY: float = 0.60
    WEIGHT_BY_MINUTES: bool = True
    MAX_IPW_WEIGHT: float = 4.0
```

### Workflow

1. **Data Audit** → Check quality, missingness, outliers
2. **Stability Analysis** → Find optimal rookie threshold
3. **Prior Calibration** → Tune Bayesian shrinkage
4. **Temporal Optimization** → Find best (k, decay)
5. **Survival Analysis** → Compute IPW weights + warnings
6. **Validation** → Measure predictive performance
7. **Report Generation** → Consolidated markdown report

### Integration

The preflight system calibrates parameters that are then used by:
- `src/performance/player_performance.py` (main model)
- `src/features/rookies.py` (rookie features)

Simply import `PREFLIGHT_PARAMS` to ensure consistency.

---

**Last Updated:** 2025-11-03  
**Version:** 1.0.0

