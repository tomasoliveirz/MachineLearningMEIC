# Coach Performance Analysis - Arquitetura Completa

## 📊 Visão Geral

Sistema de análise de performance de treinadores da WNBA com 3 níveis de granularidade:
1. **Team Performance** (época)
2. **Coach Season Performance** (stint-aware)
3. **Coach Career Performance** (agregado)

---

## 🗂️ Estrutura de Ficheiros

### Inputs (Raw)
- `data/raw/teams.csv` - Stats de época das equipas
- `data/raw/coaches.csv` - Records dos treinadores (com stints)
- `data/raw/teams_post.csv` - Resultados de playoffs
- `data/raw/awards_players.csv` - Prémios (inc. Coach of the Year)
- `data/processed/team_season_statistics.csv` - Stats processadas
- `data/processed/player_performance.csv` - Performance dos jogadores

### Outputs (Processed)
- `data/processed/team_performance.csv` (142 rows, 16 cols)
- `data/processed/coach_season_performance.csv` (162 rows, 18 cols)
- `data/processed/coach_career_performance.csv` (57 rows, 11 cols)

### Scripts
- `src/performance/team_performance.py`
- `src/performance/coach_season_performance.py`
- `src/performance/coach_career_performance.py`

---

## 📐 Métricas Implementadas

### 1. Pythagorean Win% (Bill James adaptado)
**Fórmula:** `(PF^x) / (PF^x + PA^x)` onde PF=pontos/jogo, PA=sofridos/jogo

**Expoente ajustado:** x = **10.80** (otimizado por grid search nos dados WNBA)

**Interpretação:** Expectativa de vitórias baseada apenas em pontos marcados/sofridos

### 2. Roster Strength
**Fórmula:** `Σ(player_performance × minutes) / Σ(minutes)`

**Regressão:** `rs_win_pct ~ team_strength` → R² = **0.283**

**Interpretação:** Qualidade do plantel medida pelas performances individuais ponderadas por minutos

### 3. Empirical Bayes Smoothing
**Fórmula:** `(won + α×league_mu) / (gp + α)` com α=34 (1 época)

**League mean:** μ = **0.501**

**Interpretação:** Win% ajustado para evitar ruído em amostras pequenas (shrink para média da liga)

### 4. Overachievement
**Duas variantes:**
- `overach_pythag = rs_win_pct - pythag_win_pct` (vs expectativa Pythag)
- `overach_roster = rs_win_pct - rs_win_pct_expected_roster` (vs qualidade do roster)

**Ranges observados:**
- Pythag: [-12.7%, +13.3%]
- Roster: [-38.7%, +29.0%]

### 5. Consistency & Trend
- `consistency_sd`: Desvio-padrão de `overach_pythag` ao longo das épocas
- `trend`: Slope (regressão linear) de `overach_pythag` vs tempo

---

## 🔄 Pipeline de Execução

```bash
# ORDEM OBRIGATÓRIA (dependências em cadeia)

cd /home/tomio/Documents/UNI/AC
source venv/bin/activate

# 1️⃣ Team Performance (base)
python3 src/performance/team_performance.py
# → Gera team_performance.csv com Pythag e roster baselines

# 2️⃣ Coach Season (depende de 1️⃣)
python3 src/performance/coach_season_performance.py
# → Gera coach_season_performance.csv com overach por stint

# 3️⃣ Coach Career (depende de 2️⃣)
python3 src/performance/coach_career_performance.py
# → Gera coach_career_performance.csv com agregações GP-weighted
```

---

## 🎯 Casos de Uso (Análises Prontas)

### 1. "Who beats expectation?"
```python
import pandas as pd

# Career-level
cc = pd.read_csv('data/processed/coach_career_performance.csv')
top_overachievers = cc.nlargest(10, 'avg_overach_pythag')[
    ['coachID', 'seasons', 'games', 'avg_overach_pythag', 'coy_awards']
]
print(top_overachievers)
```

**Resultado atual:**
| coachID | seasons | games | avg_overach_pythag | coy_awards |
|---------|---------|-------|-------------------|------------|
| dailesh99w | 2 | 20 | +0.165 | 0 |
| weisery99w | 1 | 14 | +0.141 | 0 |
| bryanjo01w | 2 | 40 | +0.140 | 0 |

### 2. Correlation Matrix (Coach Season Level)
```python
cs = pd.read_csv('data/processed/coach_season_performance.csv')

vars = [
    'eb_rs_win_pct', 'rs_win_pct_coach', 'po_win_pct_coach',
    'coach_overach_pythag', 'coach_overach_roster',
    'is_first_year_with_team', 'is_coy_winner', 'gp'
]

corr = cs[vars].corr()
print(corr)
```

### 3. Immediate Impact (First-Year Coaches)
```python
first_year = cs[cs['is_first_year_with_team'] == 1]

import matplotlib.pyplot as plt
first_year['delta_vs_prev_team'].hist(bins=20, edgecolor='black')
plt.xlabel('Win% Change vs Previous Year')
plt.ylabel('Count')
plt.title('First-Year Coach Impact')
plt.axvline(0, color='red', linestyle='--')
plt.show()
```

**Stats atuais:**
- 69 first-year stints
- Mean delta: calculável no-fly

### 4. Regular Season vs Playoff Performance
```python
# Career level
import matplotlib.pyplot as plt

cc_valid = cc.dropna(subset=['eb_career_win_pct', 'career_po_win_pct'])

plt.scatter(
    cc_valid['eb_career_win_pct'], 
    cc_valid['career_po_win_pct'],
    s=cc_valid['games'],  # Size by sample size
    alpha=0.6
)
plt.plot([0,1], [0,1], 'r--', alpha=0.3)  # y=x line
plt.xlabel('Career RS Win% (EB-adjusted)')
plt.ylabel('Career Playoff Win%')
plt.title('RS vs PO Performance (Coach Career)')
plt.show()
```

### 5. COY Award Predictors
```python
# What predicts Coach of the Year?
cs_coy = cs[cs['is_coy_winner'] == 1]
cs_not = cs[cs['is_coy_winner'] == 0]

print("COY Winners:")
print(cs_coy[['coachID', 'year', 'rs_win_pct_coach', 'coach_overach_pythag', 
              'delta_vs_prev_team', 'team_id']].to_string())

print("\nMean comparison:")
print(f"COY overach_pythag: {cs_coy['coach_overach_pythag'].mean():.3f}")
print(f"Non-COY overach_pythag: {cs_not['coach_overach_pythag'].mean():.3f}")
print(f"COY delta_vs_prev: {cs_coy['delta_vs_prev_team'].mean():.3f}")
print(f"Non-COY delta_vs_prev: {cs_not['delta_vs_prev_team'].mean():.3f}")
```

---

## 🔍 Resolução de Problemas Específicos

### Múltiplos Coaches na Mesma Época
**Situação:** Team X troca de treinador mid-season (stint 0 e stint 1)

**Solução implementada:**
- Cada stint tem linha própria em `coach_season_performance.csv`
- Baselines (pythag_win_pct, roster_expected) são da **equipa inteira**
- Overachievement = `rs_win_pct_do_stint - baseline_da_equipa`

**Aproximação:** Simples mas estável. Alternativa seria ponderar baselines por tempo de cada coach, mas aumenta complexidade.

### Coach em 2 Equipas no Mesmo Ano
**Situação:** Coach Y sai de Team A e vai para Team B na mesma época

**Resultado:** Duas linhas em `coach_season_performance.csv` (uma por team_id)

**Agregação career:** Soma ponderada por `gp` de ambas as linhas

### Epochs Sem Dados de Playoffs
**Tratamento:**
- `po_win_pct_coach` = NaN (não 0)
- `career_po_win_pct` = média **apenas das épocas com PO**
- Count de épocas com PO disponível via `cs['po_win_pct_coach'].notna().sum()`

---

## 📊 Estatísticas do Dataset

### Team Performance
- **142 team-seasons** (10 épocas WNBA)
- **62 playoff appearances** (43.7%)
- Pythag overach: [-12.7%, +13.3%]
- Roster overach: [-38.7%, +29.0%]

### Coach Season
- **162 stints** de 57 treinadores únicos
- **69 first-year stints** (42.6%)
- **10 COY awards**
- **81 stints com dados de PO** (50.0%)

### Coach Career
- **57 coaches**
- Média: **2.8 seasons**, **83.3 games**
- **8 coaches com COY** (14.0%)
- Top overachiever: **+16.5% vs Pythag** (dailesh99w)

---

## 🛠️ Extensões Futuras

1. **Home Court Advantage no Pythag:** Ajustar fórmula para `(PF_home^x1, PA_away^x2)`
2. **Roster Strength dinâmico:** Recomputar por game (accounting for injuries)
3. **Context-Adjusted Metrics:** Strength of schedule, rest days
4. **Playoff-specific models:** Pythag exponent diferente para PO
5. **Coach "signature":** Clustering por perfis de overach (offense vs defense)

---

## ✅ Checks de Qualidade Implementados

- [x] Fitted Pythagorean exponent reportado (10.80)
- [x] Roster regression R² reportado (0.283)
- [x] EB league_mu e alpha printados (0.501, 34)
- [x] COY count verificado (10 attached)
- [x] First-year flag testado (69 stints)
- [x] Sample outputs mostrados (top-5/top-10)
- [x] Validação de ranges (overach dentro do esperado)
- [x] NaN handling documentado (PO data)

---

**Última atualização:** 2025-11-05  
**Autor:** Pipeline automático AC/WNBA  
**Versão:** 1.0.0 (stable)
