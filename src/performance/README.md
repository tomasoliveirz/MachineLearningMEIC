# Performance Analysis Module

Módulo de análise de performance de equipas e treinadores com métricas avançadas (Pythag, roster strength, Empirical Bayes, etc.).

## Arquitetura

### 1. `team_performance.py` → `data/processed/team_performance.csv`

**Granularidade:** Por equipa-época (team-season level)

**Pipeline:**
1. Carrega estatísticas de época das equipas (`team_season_statistics.csv` ou `teams.csv`)
2. Calcula **Pythagorean Win%** com expoente ajustado aos dados (grid search)
3. Computa **Team Strength** (roster) a partir de `player_performance.csv` (minutos-ponderado)
4. Regressão linear `rs_win_pct ~ team_strength` para obter expectativa baseada em roster
5. Junta resultados de playoffs (`teams_post.csv`)
6. Calcula overachievement vs Pythag e vs Roster

**Colunas principais:**
- `team_id, year, GP, won, lost, rs_win_pct`
- `pythag_win_pct` (expoente Pythag = **10.80** nos dados WNBA)
- `team_strength` (roster quality, R² = **0.283** vs rs_win_pct)
- `rs_win_pct_expected_roster`
- `overach_pythag = rs_win_pct - pythag_win_pct`
- `overach_roster = rs_win_pct - rs_win_pct_expected_roster`
- `po_W, po_L, po_win_pct` (playoffs)
- `rs_win_pct_prev, win_pct_change`

**Diagnostics:**
- Fitted Pythagorean exponent: **10.80**
- Roster strength R² = **0.283**

---

### 2. `coach_season_performance.py` → `data/processed/coach_season_performance.csv`

**Granularidade:** Por coach-equipa-época-stint (respeita mudanças de treinador mid-season)

**Pipeline:**
1. Carrega `coaches.csv` (com `coachID, year, tmID, stint, won, lost, post_wins, post_losses`)
2. Merge com `team_performance.csv` para obter baselines (pythag, roster)
3. Calcula overachievement por treinador-stint
4. **Empirical Bayes smoothing:** `eb_rs_win_pct` (α=34 jogos, shrink para média da liga)
5. Identifica **first-year com equipa** e `delta_vs_prev_team`
6. Junta **Coach of the Year** de `awards_players.csv`
7. Playoff win% por treinador (quando disponível)

**Colunas principais:**
- `coachID, team_id, year, stint, gp, won, lost`
- `rs_win_pct_coach, eb_rs_win_pct`
- `coach_overach_pythag = rs_win_pct_coach - pythag_win_pct`
- `coach_overach_roster = rs_win_pct_coach - rs_win_pct_expected_roster`
- `is_first_year_with_team` (flag)
- `delta_vs_prev_team = rs_win_pct_coach - rs_win_pct_prev` (impacto imediato)
- `po_win_pct_coach`
- `is_coy_winner` (1 se ganhou COY naquele ano)
- Baselines para referência: `pythag_win_pct, rs_win_pct_expected_roster, team_strength`

**Diagnostics:**
- 162 coach-season stints
- 57 unique coaches
- 10 COY awards attached
- 69 first-year stints
- 81 with playoff data

**Nota sobre múltiplos coaches na mesma época:**
- Cada stint tem sua própria linha
- Baselines (pythag, roster) são da **equipa inteira** (aproximação simples e estável)
- Overachievement = performance do stint vs baseline da equipa

---

### 3. `coach_career_performance.py` → `data/processed/coach_career_performance.csv`

**Granularidade:** Por treinador (carreira completa)

**Pipeline:**
1. Carrega `coach_season_performance.csv`
2. Agrega por `coachID` com **pesos por jogos (gp)**
3. Calcula médias GP-weighted de overachievement
4. EB career win% com α=34
5. Consistency (std dev de overach por época)
6. Trend (slope de overach ao longo das épocas)

**Colunas principais:**
- `coachID, seasons, teams, games`
- `avg_overach_pythag` (GP-weighted)
- `avg_overach_roster` (GP-weighted)
- `eb_career_win_pct` (Empirical Bayes smoothed career win%)
- `consistency_sd` (std dev de `coach_overach_pythag`)
- `trend` (slope de overach ao longo do tempo)
- `career_po_win_pct` (média de po_win_pct nas épocas com playoffs)
- `coy_awards` (count)

**Diagnostics:**
- 57 coaches
- Avg 83.3 games/coach
- Avg 2.8 seasons/coach
- 8 coaches with COY awards

**Top coach (overach_pythag):** `dailesh99w` (+0.165)

---

## Métricas Chave Explicadas

### Pythagorean Win%
Expectativa baseada em pontos marcados vs sofridos:
```
pythag_win_pct = (PF^x) / (PF^x + PA^x)
```
onde `PF = o_pts/GP`, `PA = d_pts/GP`, e **x = 10.80** (ajustado aos dados WNBA).

### Team Strength (Roster)
Média ponderada por minutos da performance dos jogadores:
```
team_strength = Σ(player_performance × minutes) / Σ(minutes)
```
R² = 0.283 vs `rs_win_pct` (regressão linear).

### Empirical Bayes Smoothing
Ajusta win% para evitar overfit em amostras pequenas:
```
eb_win_pct = (won + α×league_mu) / (gp + α)
```
onde `α = 34` (1 época WNBA) e `league_mu = 0.501`.

### Overachievement
- **vs Pythag:** `overach_pythag = rs_win_pct - pythag_win_pct`
- **vs Roster:** `overach_roster = rs_win_pct - rs_win_pct_expected_roster`

Range observado:
- Pythag: [-0.127, +0.133]
- Roster: [-0.387, +0.290]

---

## Como Usar

### 1. Gerar os ficheiros (ordem obrigatória):

```bash
cd /home/tomio/Documents/UNI/AC
source venv/bin/activate

# 1. Team performance (base para os outros)
python3 src/performance/team_performance.py

# 2. Coach season (depende de team_performance)
python3 src/performance/coach_season_performance.py

# 3. Coach career (depende de coach_season)
python3 src/performance/coach_career_performance.py
```

### 2. Análises recomendadas:

#### a) "Who beats expectation?"
```python
df = pd.read_csv('data/processed/coach_career_performance.csv')
top = df.nlargest(10, 'avg_overach_pythag')
```

#### b) Correlation matrix
```python
cs = pd.read_csv('data/processed/coach_season_performance.csv')
corr_vars = [
    'eb_rs_win_pct', 'rs_win_pct_coach', 'po_win_pct_coach',
    'coach_overach_pythag', 'consistency_sd', 'games'
]
cs[corr_vars].corr()
```

#### c) Immediate impact (first-year coaches)
```python
first_yr = cs[cs['is_first_year_with_team'] == 1]
first_yr['delta_vs_prev_team'].hist(bins=20)
```

#### d) RS vs PO correlation
```python
cc = pd.read_csv('data/processed/coach_career_performance.csv')
cc.plot.scatter('eb_career_win_pct', 'career_po_win_pct')
```

---

## Dependências

- `pandas`, `numpy`, `scikit-learn`
- Ficheiros de entrada:
  - `data/processed/team_season_statistics.csv` (ou `data/raw/teams.csv`)
  - `data/processed/player_performance.csv`
  - `data/raw/teams_post.csv`
  - `data/raw/coaches.csv`
  - `data/raw/awards_players.csv`

---

## Validação

Todos os ficheiros gerados com sucesso:
- ✓ `team_performance.csv`: 142 rows, 16 columns
- ✓ `coach_season_performance.csv`: 162 rows, 18 columns
- ✓ `coach_career_performance.csv`: 57 rows, 11 columns

Ranges plausíveis, sem NaNs inesperados, métricas consistentes.

