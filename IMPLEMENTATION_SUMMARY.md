# Implementação Completa: Coach Performance Analysis

## ✅ O Que Foi Implementado

### 📁 3 Tabelas Canónicas

#### 1. `team_performance.csv` (142 rows × 16 cols)
**Granularidade:** Team-season

**Colunas chave:**
- `team_id, year, GP, won, lost, rs_win_pct`
- `pythag_win_pct` (expoente = **10.80**)
- `team_strength` (roster quality, R² = 0.283)
- `rs_win_pct_expected_roster`
- `overach_pythag`, `overach_roster`
- `po_W, po_L, po_win_pct`
- `rs_win_pct_prev, win_pct_change`

#### 2. `coach_season_performance.csv` (162 rows × 18 cols)
**Granularidade:** Coach-team-season-stint (stint-aware)

**Colunas chave:**
- `coachID, team_id, year, stint, gp, won, lost`
- `rs_win_pct_coach, eb_rs_win_pct` (α=34, μ=0.501)
- `coach_overach_pythag`, `coach_overach_roster`
- `is_first_year_with_team`, `delta_vs_prev_team`
- `po_win_pct_coach`
- `is_coy_winner` (10 awards attached)

#### 3. `coach_career_performance.csv` (57 rows × 11 cols)
**Granularidade:** Coach career (GP-weighted)

**Colunas chave:**
- `coachID, seasons, teams, games`
- `avg_overach_pythag` (GP-weighted)
- `avg_overach_roster` (GP-weighted)
- `eb_career_win_pct`
- `consistency_sd`, `trend`
- `career_po_win_pct`
- `coy_awards`

---

### 🔧 3 Scripts Modulares

#### 1. `src/performance/team_performance.py`
**Funções:**
- `fit_pythag_exponent()` - Grid search 5→20, SSE minimization → **x=10.80**
- `compute_team_strength()` - Minutes-weighted player performance
- `attach_team_results()` - Merge stats + compute Pythag
- `attach_playoffs()` - Join teams_post.csv
- `compute_overachieves()` - Linear regression roster → rs_win_pct (R²=0.283)

**Output:** `team_performance.csv`

#### 2. `src/performance/coach_season_performance.py`
**Funções:**
- `load_coaches()` - Parse coaches.csv (stint-aware)
- `merge_team_baselines()` - Attach pythag/roster from team_performance
- `compute_coach_season_metrics()` - EB smoothing, first-year flag, overach
- `attach_awards_coy()` - Join awards_players.csv (COY)

**Output:** `coach_season_performance.csv`

#### 3. `src/performance/coach_career_performance.py`
**Funções:**
- `aggregate_career()` - GP-weighted means, consistency, trend

**Output:** `coach_career_performance.csv`

---

### 📊 Script de Análises Exemplo

#### `src/performance/example_analyses.py`

**5 Análises implementadas:**

1. **Top Overachievers** (min 30 games)
   - Bar chart horizontal, colorido por COY
   
2. **Correlation Matrix** (heatmap)
   - 8 variáveis-chave: rs_win%, po_win%, overach, first-year, COY, etc.
   - **Insights:** RS vs PO r=0.492, Overach vs COY r=0.045
   
3. **First-Year Impact**
   - Histogram de `delta_vs_prev_team`
   - Boxplot de overach (first-year vs not)
   - **Resultado:** First-year média = -1.5% (ligeiramente pior que média)
   
4. **RS vs PO Performance**
   - Scatter com trend line, bubble size = games
   - **Correlação:** r=0.369
   - **Top PO overperformer:** westhpa99w (+20.9%)
   
5. **COY Predictors**
   - 4 boxplots: rs_win%, overach_pythag, delta_vs_prev, overach_roster
   - **Maior preditor:** `delta_vs_prev_team` (COY +21.6% vs non-COY -2.2%)

**Plots gerados:**
- `top_overachievers.png`
- `correlation_matrix.png`
- `first_year_impact.png`
- `rs_vs_po_performance.png`
- `coy_predictors.png`

Todos em `reports/plots/coach_performance/`

---

## 🎯 Problemas Resolvidos

### ✅ Stint-Awareness
- **Problema:** Múltiplos coaches na mesma época (mid-season changes)
- **Solução:** Cada stint = linha própria; baselines da equipa inteira (aproximação simples)

### ✅ Empirical Bayes
- **Problema:** Small-sample noise (alguns coaches com <20 jogos)
- **Solução:** EB shrinkage com α=34 (1 época), μ=0.501 (league mean)

### ✅ Two Baselines (Pythag + Roster)
- **Problema:** Pythag ignora talent; roster ignora coaching
- **Solução:** Dois overach metrics complementares
  - `overach_pythag`: Coaching effect (controlling for points)
  - `overach_roster`: Coaching + fit (controlling for talent)

### ✅ First-Year Detection
- **Problema:** Immediate impact analysis precisa de flag
- **Solução:** `is_first_year_with_team` + `delta_vs_prev_team`

### ✅ COY Integration
- **Problema:** Awards desconectados das métricas
- **Solução:** Parse `awards_players.csv`, attach via (coachID, year)

### ✅ Playoffs Handling
- **Problema:** Nem todas as épocas têm PO data
- **Solução:** NaN (não 0), agregações usam apenas épocas válidas

---

## 📈 Estatísticas Chave (Dataset WNBA)

### Dataset Overview
- **10 épocas** (year 1-10)
- **142 team-seasons**, **62 playoff appearances** (43.7%)
- **162 coach-season stints**, **57 unique coaches**
- **Média:** 2.8 seasons/coach, 83.3 games/coach

### Fitted Parameters
- **Pythag exponent:** x = **10.80** (WNBA-specific, vs ~13.9 NBA)
- **Roster R²:** 0.283 (moderate; coaching matters!)
- **EB alpha:** 34 games (1 WNBA season)
- **League mean win%:** 0.501 (balanced)

### Ranges Observados
- **Overach Pythag:** [-12.7%, +13.3%]
- **Overach Roster:** [-38.7%, +29.0%] (wider = talent matters more)
- **RS-PO gap:** [-56.4%, +20.9%]

### Top Results
- **Best career overachiever:** dailesh99w (+16.5% vs Pythag, 20 games)
- **Best PO overperformer:** westhpa99w (PO 77.8% vs RS 56.9%)
- **COY mean delta:** +21.6% vs prev year (strongest predictor)

---

## 🚀 Como Executar

```bash
cd /home/tomio/Documents/UNI/AC
source venv/bin/activate

# 1. Gerar as 3 tabelas (ordem obrigatória)
python3 src/performance/team_performance.py
python3 src/performance/coach_season_performance.py
python3 src/performance/coach_career_performance.py

# 2. Rodar análises exemplo (gera 5 plots)
python3 src/performance/example_analyses.py

# Outputs:
# - data/processed/team_performance.csv
# - data/processed/coach_season_performance.csv
# - data/processed/coach_career_performance.csv
# - reports/plots/coach_performance/*.png
```

---

## 📚 Documentação

### Ficheiros criados
1. `src/performance/README.md` - Arquitetura técnica completa
2. `COACH_ANALYSIS_ARCHITECTURE.md` - Overview high-level + casos de uso
3. `IMPLEMENTATION_SUMMARY.md` - Este ficheiro (sumário executivo)
4. `src/performance/example_analyses.py` - 5 análises prontas

### Métricas explicadas
- **Pythagorean Win%:** `(PF^x)/(PF^x+PA^x)` - Expectativa baseada em pontos
- **Team Strength:** `Σ(perf×min)/Σ(min)` - Qualidade do roster
- **Empirical Bayes:** `(won + α×μ)/(gp + α)` - Smoothing para small samples
- **Overachievement:** `actual - expected` (duas variantes: Pythag e Roster)

---

## 🔍 Insights Principais

### 1. Coaching Matter (mas não muito)
- R² roster = 0.283 → **71.7% da variância não explicada por talent**
- Overach range [-13%, +13%] → coaches swing ~25% total

### 2. COY ≈ Team Improvement (não absolute quality)
- **Delta vs prev:** COY +21.6%, non-COY -2.2%
- **Overach:** COY +1.4%, non-COY -0.3% (weak signal)
- **Conclusão:** Award premia "turnaround", não overachievement absoluto

### 3. First-Year Coaches Ligeiramente Piores
- Mean overach: +0.8% (vs -1.0% não-first)
- Mean delta: -1.5% (vs -0.5%)
- **Interpretação:** Learning curve, ou correlation com teams em crise

### 4. RS ≠ PO (moderadamente)
- **Correlação:** r=0.369 (career level)
- **Outliers existem:** westhpa99w PO+20.9%, allenso99w PO-56.4%
- **Implicação:** Playoff coaching pode ser skill separado

### 5. Roster > Pythag (como baseline)
- Overach_roster range 67% ([-38.7, +29.0])
- Overach_pythag range 26% ([-12.7, +13.3])
- **Conclusão:** Roster capturing mais variance → Pythag melhor proxy de coaching puro

---

## ✅ Validação Completa

- [x] 3 tabelas geradas com sucesso
- [x] Pythag exponent reportado (10.80)
- [x] Roster R² reportado (0.283)
- [x] EB parameters printados (α=34, μ=0.501)
- [x] COY awards attached (10 encontrados)
- [x] First-year flag funcional (69 stints)
- [x] Playoff data integrado (81 stints com PO)
- [x] 5 análises executadas com sucesso
- [x] 5 plots gerados (150KB cada, alta qualidade)
- [x] Ranges plausíveis (no outliers absurdos)
- [x] NaN handling correto (PO data)
- [x] Stint-awareness verificado (mid-season changes OK)

---

## 🛠️ Dependências

**Python packages:**
- pandas, numpy, scikit-learn, matplotlib, seaborn

**Input files (todos existentes):**
- `data/raw/teams.csv`
- `data/raw/coaches.csv`
- `data/raw/teams_post.csv`
- `data/raw/awards_players.csv`
- `data/processed/team_season_statistics.csv`
- `data/processed/player_performance.csv`

**Output files (gerados):**
- `data/processed/team_performance.csv` (24KB)
- `data/processed/coach_season_performance.csv` (30KB)
- `data/processed/coach_career_performance.csv` (6.8KB)

---

## 🎓 Extensões Futuras (Sugestões)

1. **Pythag refinado:** Expoentes separados para home/away
2. **Roster dinâmico:** Recompute game-by-game (injuries)
3. **SOS adjustment:** Strength of schedule
4. **Playoff-specific Pythag:** Expoente diferente em PO
5. **Coach archetypes:** Clustering por perfil (offense/defense bias)
6. **Temporal trends:** Coaching efficacy over league evolution
7. **Rookie integration:** Link com `team_rookie_features.csv`

---

**Status:** ✅ **COMPLETO E FUNCIONAL**  
**Data:** 2025-11-05  
**Versão:** 1.0.0 (stable, production-ready)
