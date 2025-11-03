# 🎯 Preflight Basic Mode

## O Que É

**Modo básico** = só o essencial para entender e melhorar o per-36.

**Removido:**
- ❌ Rookie prior calibration (Bayesian shrinkage)
- ❌ RMSE threshold selection (auto-escolha de rookie_min_minutes)
- ❌ Survival bias / IPW
- ❌ Predictive validation (MAE, RMSE, sensitivity)

**Mantido:**
- ✅ Data audit (missingness, outliers, ranges)
- ✅ Correlations (per36 vs stats)
- ✅ Per-36 stability plot (visual, sem auto-threshold)
- ✅ Temporal optimization (k, decay)

---

## Pipeline (4 etapas)

```
[1/4] Load data (aggregate stints, label rookies)
[2/4] Data quality audit
[3/4] Correlations + per36 stability plot
[4/4] Temporal dependence (optimize k, decay)
```

---

## Parâmetros (config.py)

```python
MIN_EFFECTIVE_MINUTES = 12  # floor para per-36
SEASONS_BACK = 3            # janela temporal
DECAY = 0.60                # desconto para épocas antigas
WEIGHT_BY_MINUTES = True    # ponderar por minutos
```

**Removidos:**
- `ROOKIE_MIN_MINUTES`
- `ROOKIE_PRIOR_STRENGTH`
- `MAX_IPW_WEIGHT`

---

## Como Correr

```bash
source venv/bin/activate
python src/analysis/player_preflight/run_preflight.py
```

**Output:**
```
[1/4] Loading data...
  ✓ Loaded 1876 player-year-team rows

[2/4] Data quality audit...
  ✓ Audit summary, missingness heatmap, outliers

[3/4] Computing correlations...
  ✓ Correlation matrix
  ✓ Per-36 vs minutes plot (visual inspection)

[4/4] Temporal dependence (k, decay)...
  ✓ Best k=3, decay=0.40, R²=0.490 (n=954)

✅ PREFLIGHT COMPLETE (BASIC MODE)
```

---

## Relatórios Gerados

```
reports/player_preflight/
├── preflight_report.md
├── figures/
│   ├── missingness_heatmap.png
│   ├── correlations_heatmap.png
│   ├── per36_vs_minutes.png      ← INSPEÇÃO VISUAL
│   └── r2_vs_seasons_back.png
├── tables/
│   ├── yearly_coverage.csv
│   ├── outliers_top20_z.csv
│   └── walkforward_k_decay.csv
└── meta/
    ├── audit_summary.txt
    ├── correlations.txt
    ├── k_decay_best.txt
    └── leakage_checklist.txt
```

**Removidos:**
- `rookie_prior_grid.png / .csv`
- `survival_weights.csv`
- `validation_strata.csv`
- `sensitivity.txt`

---

## O Que Fazer Agora

### 1️⃣ **Entender e melhorar o per-36**

- **Ver:** `figures/correlations_heatmap.png`
- **Pergunta:** Quais stats pesam mais? Faz sentido?
- **Ação:** Ajustar pesos em `src/utils/players.py` → `compute_per36`

**Exemplo atual:**
```python
per36 = points + 0.7*reb + 0.7*ast + 1.2*stl + 1.2*blk - 0.7*tov
```

Podes testar:
- Dar mais peso a assistências?
- Penalizar mais turnovers?
- Incluir oRebounds/dRebounds separadamente?

### 2️⃣ **Escolher threshold visual**

- **Ver:** `figures/per36_vs_minutes.png`
- **Pergunta:** A partir de quantos minutos o per36 fica estável?
- **Ação:** Escolher à mão (eg. 300? 400? 600?)

### 3️⃣ **Temporal weights**

- **Ver:** `tables/walkforward_k_decay.csv`
- **Pergunta:** k=3 faz sentido? decay=0.60 ou 0.40?
- **Ação:** Se quiseres outro valor, editar `config.py`

---

## Quando Ativar Modo Avançado?

Quando:
- ✅ Estiveres confiante no per-36
- ✅ Tiveres escolhido um threshold de minutos
- ✅ Entenderes bem os 4 parâmetros básicos

Aí podes reativar:
1. **Rookie priors** (para Bayesian shrinkage)
2. **Validation** (para medir RMSE/MAE rigoroso)
3. **Survival bias** (se quiseres IPW)

---

## Filosofia

> "Primeiro faz uma métrica que TU percebes e em que confias.  
> Depois validas rigorosamente com RMSE, MAE, IPW, etc."

**Modo básico** = foco no **entendimento**.  
**Modo avançado** = foco na **validação rigorosa**.

---

## Ficheiros-Chave

| Ficheiro | O Que É |
|----------|---------|
| `config.py` | 4 parâmetros básicos |
| `run_preflight.py` | Pipeline 4 etapas |
| `data_audit.py` | Qualidade de dados |
| `temporal_dependence.py` | Otimização k/decay |
| `stability_minutes.py` | Plot per36 vs minutes |

**Desativados mas não apagados:**
- `rookie_priors.py`
- `validation.py`
- `survival_bias.py`

---

**Status:** ✅ Modo básico ativo e funcional

