# 🎓 Guia de Defesa: Player Performance Preflight

**Objetivo:** Explicar cada parâmetro calibrado de forma simples e direta.

---

## 🎯 O Que É Isto?

Este módulo **calibra parâmetros** para o modelo de performance de jogadores.

É como **afinar um instrumento** antes de tocar:
- Garantes que os dados estão limpos
- Descobres os valores ótimos para usar no modelo
- Validas que as decisões fazem sentido

---

## 📋 Parâmetros Calibrados

### 1. **MIN_EFFECTIVE_MINUTES = 12**

**O que é:**
> Floor (mínimo) de minutos para calcular taxas per-36.

**Porquê 12?**
- Um jogador que jogou 2 minutos e marcou 6 pontos → 108 pts/36min (absurdo!)
- Com floor de 12 min: tratamos como se tivesse jogado pelo menos 12 min
- Evita rates extremas e instáveis

**Exemplo numérico:**
```
Jogador A: 2 min, 6 pts
  Sem floor:  (6/2)  × 36 = 108 pts/36  ❌ irrealista
  Com floor:  (6/12) × 36 = 18  pts/36  ✅ conservador

Jogador B: 1000 min, 500 pts
  Sem/com floor: (500/1000) × 36 = 18 pts/36  ✅ igual (não afeta)
```

---

### 2. **ROOKIE_MIN_MINUTES = 400**

**O que é:**
> Mínimo de minutos que um rookie deve jogar para ser incluído na **calibração** de parâmetros.

**Porquê 400?**
- Testámos [150, 300, 400, 600] minutos
- 400 minimiza o **RMSE** (erro) ao prever o per36 do ano seguinte
- Rookies com <400 min são muito instáveis (muito ruído)

**Tabela de decisão:**

| Threshold | RMSE  | Interpretação |
|-----------|-------|---------------|
| 150       | ~5.2  | Muito ruído   |
| 300       | ~3.5  | Melhor        |
| **400**   | **3.3** | **Ótimo** ✅ |
| 600       | ~3.4  | Perdes dados  |

**Nota importante:**
- Isto é só para **calibrar** o modelo
- No modelo final, podes avaliar rookies com <400 min, mas com **mais shrinkage ao prior**

---

### 3. **ROOKIE_PRIOR_STRENGTH = 900**

**O que é:**
> Força do prior Bayesiano para rookies (em "minutos equivalentes").

**Em linguagem simples:**
Quando um rookie tem poucos minutos jogados, não sabemos muito sobre ele.
Então **combinamos** o que ele fez com uma "baseline da liga" (prior).

**Como funciona:**

```python
# Exemplo: rookie com 300 minutos
peso_observado = 300 / 36 ≈ 8.3 "jogos equivalentes"
peso_prior     = 900 / 36 = 25 "jogos equivalentes"

# Média ponderada
per36_final = (8.3 × per36_observado + 25 × per36_prior) / (8.3 + 25)
            = mais peso ao prior (porque 300 min é pouco)
```

**Porquê 900?**
- Testámos [900, 1800, 3600, 7200]
- 900 minimiza o **RMSE** vs ano seguinte
- É o melhor compromisso entre "confiar no observado" e "regressão à média"

**Interpretação prática:**
- Um rookie com **900 minutos** → peso 50/50 entre o que fez e o prior
- Um rookie com **300 minutos** → peso 75% prior, 25% observado
- Um rookie com **2700 minutos** → peso 25% prior, 75% observado

---

### 4. **SEASONS_BACK = 3**

**O que é:**
> Quantos anos históricos usar para prever performance futura.

**Porquê 3?**
- Testámos k = [1, 2, 3, 4, 5, 6, 7]
- R² aumenta até k=3 (R²=0.490)
- Depois estabiliza (k=4 → R²=0.491, diferença <0.001)
- Mais anos = mais complexidade, sem ganho real

**Visualização:**

```
k=1: só ano anterior        R² = 0.477
k=2: últimos 2 anos         R² = 0.486
k=3: últimos 3 anos         R² = 0.490  ✅ ÓTIMO
k=4: últimos 4 anos         R² = 0.491  (ganho ~0%)
```

---

### 5. **DECAY = 0.60**

**O que é:**
> Fator de desconto para épocas mais antigas.

**Em linguagem simples:**
Quando fazes média histórica, **épocas recentes devem pesar mais** que épocas antigas.

**Como funciona:**

Imagina um jogador em 2023:
- 2023 (t):   peso = 0.60^0 = 1.00  (100%)
- 2022 (t-1): peso = 0.60^1 = 0.60  (60%)
- 2021 (t-2): peso = 0.60^2 = 0.36  (36%)

Depois normalizas para somar 1:
- 2023: 51% do peso total
- 2022: 31%
- 2021: 18%

**Exemplo numérico concreto:**

```
Jogador X:
  2023: per36 = 20
  2022: per36 = 15
  2021: per36 = 10

Média ponderada (decay=0.6, k=3):
  = (1.00×20 + 0.60×15 + 0.36×10) / (1.00 + 0.60 + 0.36)
  = (20 + 9 + 3.6) / 1.96
  = 32.6 / 1.96
  ≈ 16.6

Nota: O ano mais recente (20) tem MUITO mais influência.
```

**Porquê 0.60 e não 0.40?**
- R² maximiza em decay=0.40 (R²=0.490)
- Mas decay=0.60 dá R²=0.489 (diferença <0.01)
- Preferimos **0.60 por interpretabilidade**:
  - 0.60 = "ano anterior conta 60%"
  - 0.40 = "ano anterior conta 40%" (muito pouco peso ao passado)

**Outros exemplos de decay:**

| Decay | t-1  | t-2  | t-3  | Interpretação |
|-------|------|------|------|---------------|
| 1.0   | 100% | 100% | 100% | Passado = presente (não faz sentido) |
| 0.8   | 80%  | 64%  | 51%  | Passado pesa bastante |
| **0.6** | **60%** | **36%** | **22%** | **Balanço razoável** ✅ |
| 0.4   | 40%  | 16%  | 6%   | Passado quase irrelevante |

---

### 6. **WEIGHT_BY_MINUTES = True**

**O que é:**
> Ponderar épocas pelo número de minutos jogados.

**Porquê True?**
Uma época com 2000 minutos é **muito mais informativa** que uma com 50 minutos.

**Exemplo:**
```
Jogador Y (sem weight_by_minutes):
  2023: 2000 min, per36 = 18  →  peso = 1.0
  2022:   50 min, per36 = 25  →  peso = 1.0
  Média = (18 + 25) / 2 = 21.5  ❌ 50 min conta igual a 2000!

Jogador Y (com weight_by_minutes):
  2023: 2000 min, per36 = 18  →  peso = 2000
  2022:   50 min, per36 = 25  →  peso = 50
  Média = (2000×18 + 50×25) / (2000+50) ≈ 18.2  ✅ 2000 min domina
```

---

## 🔍 Como Foram Calibrados?

### Método geral:
1. **Walk-forward validation:** treinar em anos passados, testar no ano seguinte
2. **Métrica:** R² (correlação ao quadrado), MAE, RMSE
3. **Grid search:** testar múltiplos valores, escolher o melhor
4. **Trade-off:** simplicidade vs ganho (se ganho <1%, escolher o mais simples)

### Pipeline:
```
Dados raw
  ↓
Limpeza (audit)
  ↓
Stability analysis (rookie_min_minutes)
  ↓
Rookie prior calibration (prior_strength)
  ↓
Temporal dependence (k, decay)
  ↓
Predictive validation (confirmar que funciona)
  ↓
Parâmetros finais ✅
```

---

## 🎤 Perguntas Típicas do Professor

### Q1: "O que é DECAY?"
**R:** É o fator que controla quanto peso damos a épocas antigas vs recentes.
Com decay=0.6, o ano anterior conta 60% do atual, o de há 2 anos conta 36%, etc.
Maximiza R² em 0.40, mas usamos 0.60 para ter mais interpretabilidade (diferença <1%).

---

### Q2: "Porque 3 seasons back e não 5?"
**R:** Testámos k=1 até 7. R² aumenta até k=3 (0.490) e depois estabiliza.
k=4 dá R²=0.491 (ganho <0.001), então escolhemos k=3 por simplicidade.

---

### Q3: "O que é rookie prior?"
**R:** É uma forma de Bayesian shrinkage: quando um rookie tem poucos minutos,
combinamos o que ele fez com a média da liga. Isso reduz variância e melhora previsões.
900 minutos é o valor que minimiza RMSE vs ano seguinte.

---

### Q4: "Porque não usaram survival bias correction?"
**R:** Survival bias existe (jogadores fracos saem da liga), mas a correção (IPW)
requer pesos que podem explodir (até 9×) e dominar o modelo. Para manter o trabalho
focado e interpretável, deixámos essa correção como trabalho futuro.

---

### Q5: "Como validaram?"
**R:** Walk-forward validation: para cada ano t, usamos dados até t-1 para prever t.
Medimos R², MAE, RMSE. Também estratificamos por minutos jogados (<150, 150-600, >600)
para confirmar que o modelo funciona bem em diferentes regimes.

---

## ✅ Mensagem-Chave para a Defesa

> "Implementámos uma pipeline sistemática de calibração de parâmetros para o modelo
> de performance de jogadores. Todos os valores foram escolhidos através de
> walk-forward validation, minimizando RMSE e maximizando R² preditivo.
> 
> Os parâmetros finais balanceiam **precisão preditiva** com **interpretabilidade**,
> e são consumidos pelo modelo principal de forma automática através de um config centralizado."

---

## 📚 Ficheiros de Referência

- **Código:** `src/analysis/player_preflight/run_preflight.py`
- **Config:** `src/analysis/player_preflight/config.py`
- **Relatório:** `reports/player_preflight/preflight_report.md`
- **Decisões técnicas:** `docs/player_preflight/TECHNICAL_DECISIONS.md`

---

**Última atualização:** Após remoção de survival bias/IPW para simplificar

