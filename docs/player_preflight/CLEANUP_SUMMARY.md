# 🧹 Limpeza do Survival Bias/IPW

## O Que Foi Removido

### ❌ Ficheiros/Módulos Removidos
- `survival_bias.py` → **mantido mas não usado** (para referência futura)

### ❌ Do `config.py`
```python
# REMOVIDO:
MAX_IPW_WEIGHT: float = 4.0
```

### ❌ Do `run_preflight.py`

**Imports removidos:**
```python
from src.analysis.player_preflight.survival_bias import write_survival_weights
```

**Etapas removidas:**
- `[7/8] Survival bias (IPW)...`
- Geração de `survival_weights.csv`
- Geração de `survival_ipw_warnings.txt`

**Do relatório final:**
- Secção "## Survival bias"
- Menções a IPW
- Warnings sobre max_weight

**Dos prints finais:**
- `max_ipw_weight = 4.0`
- "Check IPW warnings"

---

## ✅ O Que Ficou (6 Parâmetros Essenciais)

```python
@dataclass(frozen=True)
class PreflightConfig:
    MIN_EFFECTIVE_MINUTES: int = 12
    ROOKIE_MIN_MINUTES: int = 400
    ROOKIE_PRIOR_STRENGTH: int = 900
    SEASONS_BACK: int = 3
    DECAY: float = 0.60
    WEIGHT_BY_MINUTES: bool = True
```

---

## 📊 Pipeline Simplificada

```
[1/7] Loading data
[2/7] Data quality audit
[3/7] Computing correlations
[4/7] Per-36 stability analysis  → ROOKIE_MIN_MINUTES
[5/7] Rookie prior calibration   → ROOKIE_PRIOR_STRENGTH
[6/7] Temporal dependence        → SEASONS_BACK, DECAY
[7/7] Predictive validation      → confirmar tudo funciona
```

**Removido:** `[8/8] Survival bias (IPW)`

---

## 🎯 Justificação para Defesa

Se perguntarem **"Porque não corrigiram survival bias?"**:

> "Survival bias existe: jogadores fracos saem da liga e desaparecem dos dados.
> A correção típica usa Inverse Probability Weighting (IPW), mas isso pode
> gerar pesos extremos (até 9×) que dominam o modelo e reduzem interpretabilidade.
>
> Para manter o trabalho focado nos parâmetros essenciais (rookie priors,
> temporal decay) e garantir que todos os componentes são compreensíveis
> e defensáveis, optámos por deixar a correção de survival bias como
> **extensão futura**, potencialmente com métodos mais robustos que IPW
> (e.g., propensity score matching, stratification)."

---

## 📁 Outputs Agora Gerados

```
reports/player_preflight/
├── preflight_report.md           ✅ SEM survival bias
├── figures/
│   ├── missingness_heatmap.png
│   ├── correlations_heatmap.png
│   ├── per36_vs_minutes.png
│   ├── rookie_prior_grid.png
│   └── r2_vs_seasons_back.png
├── tables/
│   ├── yearly_coverage.csv
│   ├── outliers_top20_z.csv
│   ├── rookie_prior_grid.csv
│   ├── walkforward_k_decay.csv
│   └── validation_strata.csv
└── meta/
    ├── audit_summary.txt
    ├── correlations.txt
    ├── stability.txt
    ├── k_decay_best.txt
    ├── sensitivity.txt
    ├── validation.txt
    └── leakage_checklist.txt
```

**Removidos:**
- ❌ `survival_weights.csv`
- ❌ `survival_ipw_warnings.txt`

---

## 🚀 Output do Script

```bash
============================================================
PLAYER PERFORMANCE PREFLIGHT
============================================================

[1/7] Loading data...
  ✓ Loaded 1876 player-year-team rows

[2/7] Data quality audit...
  ✓ Audit summary, missingness heatmap, outliers

[3/7] Computing correlations...
  ✓ Correlation matrix

[4/7] Per-36 stability analysis...
  ✓ Chosen rookie_min_minutes = 400 (RMSE = 3.268)

[5/7] Rookie prior calibration...
  ✓ Rookie prior grid (see figures/rookie_prior_grid.png)

[6/7] Temporal dependence (k, decay)...
  ✓ Best k=3, decay=0.40, R²=0.490 (n=954)

[7/7] Predictive validation...
  ✓ Validation metrics

============================================================
✅ PREFLIGHT COMPLETE
============================================================

Reports saved to: /home/tomio/Documents/UNI/AC/reports/player_preflight

Calibrated parameters (see config.py):
  - MIN_EFFECTIVE_MINUTES = 12
  - rookie_min_minutes = 400
  - rookie_prior_strength = 900
  - seasons_back = 3
  - decay = 0.6 (R² max at 0.40, ΔR²<0.01)
  - weight_by_minutes = True

Next steps:
  1. Review preflight_report.md
  2. Import PREFLIGHT_PARAMS in your models
```

---

## ✅ Status Final

**Código:**
- ✅ Limpo, sem survival bias
- ✅ 6 parâmetros essenciais bem calibrados
- ✅ Pipeline 7 etapas (era 8)
- ✅ Tudo documentado e testado

**Documentação:**
- ✅ `DEFENSE_GUIDE.md` → perguntas típicas
- ✅ `CONCEPTS_SIMPLE.md` → conceitos em linguagem simples
- ✅ `CLEANUP_SUMMARY.md` → este ficheiro
- ✅ Relatórios atualizados sem IPW

**Pronto para:** Defesa e produção 🚀

