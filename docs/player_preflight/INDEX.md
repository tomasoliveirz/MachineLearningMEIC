# 📚 Player Preflight Documentation Index

**Versão:** Simplificada (sem survival bias/IPW)

---

## 🚀 START HERE

**Novo ao projeto?** Lê por esta ordem:

1. 📖 **[README.md](README.md)** → Visão geral, como correr, estrutura
2. 🧠 **[CONCEPTS_SIMPLE.md](CONCEPTS_SIMPLE.md)** → Conceitos em linguagem simples
3. 🎓 **[DEFENSE_GUIDE.md](DEFENSE_GUIDE.md)** → Guia de defesa, perguntas típicas
4. 📊 **[../reports/player_preflight/preflight_report.md](../../reports/player_preflight/preflight_report.md)** → Relatório completo

---

## 📁 Todos os Documentos

### Essenciais

| Ficheiro | Conteúdo | Quando usar |
|----------|----------|-------------|
| **README.md** | Overview, estrutura, como correr | Sempre que começares |
| **DEFENSE_GUIDE.md** | Perguntas típicas, respostas prontas | Antes da defesa |
| **CONCEPTS_SIMPLE.md** | Conceitos técnicos explicados simples | Quando não percebes algo |
| **preflight_report.md** | Resultados da calibração | Para ver os valores finais |

### Técnicos

| Ficheiro | Conteúdo | Quando usar |
|----------|----------|-------------|
| **TECHNICAL_DECISIONS.md** | Justificações técnicas detalhadas | Para entender "porquê" |
| **QUICK_REFERENCE.md** | Comandos rápidos, troubleshooting | Quando algo não funciona |
| **CLEANUP_SUMMARY.md** | O que foi removido (survival bias) | Contexto histórico |

---

## 🎯 Por Objetivo

### "Preciso defender isto"
1. **DEFENSE_GUIDE.md** → perguntas e respostas
2. **CONCEPTS_SIMPLE.md** → conceitos explicados
3. **preflight_report.md** → resultados para mostrar

### "Preciso correr isto"
1. **QUICK_REFERENCE.md** → comandos essenciais
2. **README.md** → estrutura e outputs

### "Preciso perceber o código"
1. **README.md** → mapa geral
2. **TECHNICAL_DECISIONS.md** → decisões técnicas
3. **Código:** `src/analysis/player_preflight/`

### "Preciso modificar parâmetros"
1. **config.py** → alterar valores
2. **preflight_report.md** → ver justificações
3. **TECHNICAL_DECISIONS.md** → entender impacto

---

## 🔑 Ficheiros-Chave do Código

```
src/analysis/player_preflight/
├── run_preflight.py          🎯 MAIN (orchestrator)
├── config.py                 ⚙️  PARÂMETROS CALIBRADOS
├── data_audit.py             ✓  Qualidade de dados
├── stability_minutes.py      ⚖️  Rookie min threshold
├── rookie_priors.py          🎲 Bayesian shrinkage
├── temporal_dependence.py    ⏰ k/decay optimization
└── validation.py             📊 Validação preditiva
```

---

## 📊 Relatórios Gerados

```
reports/player_preflight/
├── preflight_report.md       📝 Relatório principal
├── figures/                  📊 Visualizações
│   ├── missingness_heatmap.png
│   ├── correlations_heatmap.png
│   ├── per36_vs_minutes.png
│   ├── rookie_prior_grid.png
│   └── r2_vs_seasons_back.png
├── tables/                   📋 Dados detalhados
│   ├── yearly_coverage.csv
│   ├── outliers_top20_z.csv
│   ├── rookie_prior_grid.csv
│   ├── walkforward_k_decay.csv
│   └── validation_strata.csv
└── meta/                     📝 Sumários
    ├── audit_summary.txt
    ├── correlations.txt
    ├── stability.txt
    ├── k_decay_best.txt
    ├── sensitivity.txt
    ├── validation.txt
    └── leakage_checklist.txt
```

---

## 🎓 Parâmetros Calibrados (Referência Rápida)

```python
MIN_EFFECTIVE_MINUTES = 12      # Floor para per-36
ROOKIE_MIN_MINUTES = 400         # Threshold para calibração
ROOKIE_PRIOR_STRENGTH = 900      # Força do prior Bayesiano
SEASONS_BACK = 3                 # Janela temporal
DECAY = 0.60                     # Desconto para épocas antigas
WEIGHT_BY_MINUTES = True         # Ponderar por minutos
```

**Origem:** Walk-forward validation, minimizando RMSE/maximizando R²

---

## ❓ FAQs Rápidas

**Q: Onde estão os valores calibrados?**  
A: `src/analysis/player_preflight/config.py` → `PREFLIGHT_PARAMS`

**Q: Como re-calibrar?**  
A: `python src/analysis/player_preflight/run_preflight.py`

**Q: O que é DECAY?**  
A: Ver **CONCEPTS_SIMPLE.md** → "Temporal Dependence"

**Q: Porque não tem survival bias?**  
A: Ver **CLEANUP_SUMMARY.md** → secção "Justificação"

**Q: Como uso isto no meu modelo?**  
A: `from src.analysis.player_preflight.config import PREFLIGHT_PARAMS`

---

## 🔗 Links Úteis

- **Código:** `/home/tomio/Documents/UNI/AC/src/analysis/player_preflight/`
- **Reports:** `/home/tomio/Documents/UNI/AC/reports/player_preflight/`
- **Docs:** `/home/tomio/Documents/UNI/AC/docs/player_preflight/`
- **Utils:** `/home/tomio/Documents/UNI/AC/src/utils/players.py`

---

## 📈 Pipeline Visual

```
1. Load data (1876 rows)
         ↓
2. Data audit (missingness, outliers)
         ↓
3. Correlations (per36 vs stats)
         ↓
4. Stability (rookie_min_minutes)
         ↓
5. Rookie priors (prior_strength)
         ↓
6. Temporal (k, decay)
         ↓
7. Validation (R², RMSE, MAE)
         ↓
   ✅ CALIBRATED PARAMETERS
```

---

## ✅ Checklist de Compreensão

Antes da defesa, confirma que sabes responder:

- [ ] O que é per-36 e porque normalizamos?
- [ ] O que é DECAY e como funciona? (exemplo numérico)
- [ ] O que é rookie prior? (Bayesian shrinkage)
- [ ] Porque 400 min e não 150 ou 600?
- [ ] Como validaste (walk-forward)?
- [ ] O que é R² e RMSE?
- [ ] Porque não usaste survival bias correction?

**Se respondeste SIM a tudo:** Estás pronto! 🚀  
**Se algum NÃO:** Lê **CONCEPTS_SIMPLE.md** e **DEFENSE_GUIDE.md**

---

**Última atualização:** Após simplificação (remoção de survival bias/IPW)  
**Status:** ✅ Pronto para defesa e produção
