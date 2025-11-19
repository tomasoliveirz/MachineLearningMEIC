# AC — Sports Analytics Pipeline

Repositório com pipeline end-to-end para análise de desempenho de jogadores, equipas e rookies:
limpeza → engenharia de features → agregação por época → análise e relatórios.

---

## Estrutura do projecto (resumo)
- data/raw/                      — CSVs originais (players.csv, teams.csv, players_teams.csv, ...)
- data/processed/                — resultados da limpeza e features
- src/ac/cleaning/               — scripts de limpeza
- src/ac/features/               — engenharia de features (team season, rookies)
- src/ac/analysis/               — geração de gráficos e relatórios
- src/analysis/player_preflight/ — calibração de parâmetros de performance
- src/utils/                     — funções reutilizáveis
- reports/                       — figuras e tabelas (raw / cleaned)
- docs/                          — documentação detalhada
- Makefile                       — automação da pipeline
- requirements.txt

---

## Pré-requisitos (Windows)
1. Python 3.8+ instalado.
2. A partir da root do projecto:
   - Criar venv e instalar dependências:
     ```powershell
     python -m venv venv
     venv\Scripts\activate
     pip install -r requirements.txt
     ```
   - Ou usar `make venv` (Make deve estar disponível).

---

## Caminho do pipeline (ordem recomendada)
1. Limpeza
   - make clean_players
     - gera: data/processed/players_cleaned.csv
   - make clean_teams
     - gera: data/processed/teams_cleaned.csv

2. Engenharia de features
   - make team_season
     - a partir de teams_cleaned → data/processed/team_season_statistics.csv
   - make rookies
     - usa players_cleaned + players_teams → data/processed/team_rookie_features.csv e player_rookie_year.csv

3. Calibração de parâmetros (Player Performance Preflight)
   - make preflight
     - Calibra: rookie_min_minutes, rookie_prior_strength, seasons_back, decay
     - Gera: reports/player_preflight/preflight_report.md
     - Documentação: docs/player_preflight/
   - **Importante:** Executar sempre que dados mudarem significativamente

4. Cálculo de performance por jogador
   - Workflow principal implementado em src/player_performance.py
   - Usa parâmetros de src/analysis/player_preflight/config.py
   - make all executa a sequência: clean_players → clean_teams → team_season → rookies → analyze_cleaned

5. Análise e relatórios
   - make analyze_raw     — gera figuras/tabelas usando dados raw
   - make analyze_cleaned — gera figuras/tabelas usando dados limpos
   - Saídas: reports/figures/{raw,cleaned}/ e reports/tables/{raw,cleaned}/

---

## Notas operacionais (decisões implementadas)
- Rookies (shrinkage com prior)
  - Porquê: métricas per36 inflacionam quando há poucos minutos (alta variância).
  - Como: mistura Bayesiana entre a observação e um prior (média de rookies por equipa; fallback global), conforme [`_apply_shrinkage_corrections`](src/player_performance.py).
    - Fórmula: $$\hat{\mu}=\frac{w_{obs}\,\mu_{obs}+w_{prior}\,\mu_{prior}}{w_{obs}+w_{prior}}$$
      com $w_{obs}=\frac{\text{minutos}}{36}$ e $w_{prior}=\frac{\text{rookie\_prior\_strength}}{36}$ (reforçado quando minutos são muito baixos).
  - Exemplo: rookie com 18 min e $\mu_{obs}=28$; prior da equipa $\mu_{prior}=20$.
    - $w_{obs}=0.5$ e $w_{prior}\approx100$ ⇒ $\hat{\mu}\approx\frac{0.5\cdot28+100\cdot20}{100.5}\approx20.04$ (evita superestimar com amostra pequena).

- Históricos (decaimento exponencial)
  - Porquê: épocas recentes devem pesar mais; estabiliza a métrica com contexto do histórico.
  - Como: média ponderada temporal por jogador, com decaimento $0<\delta<1$ e, opcionalmente, ponderação por minutos, conforme [`_compute_weighted_history`](src/player_performance.py).
    - Peso da época $k$ anos atrás: $w_k=\delta^{k}\times \text{minutos}_k$ (se ativado).
  - Exemplo: épocas com per36 [20, 18, 25] e minutos [800, 600, 200], $\delta=0.7$.
    - Pesos ≈ [800, 0.7×600, 0.7^2×200] = [800, 420, 98]; média ≈ $(800·20+420·18+98·25)/(1318)\approx19.63$.

- Fator de contexto da equipa
  - Porquê: ambientes ofensivos diferentes afetam contagens (ritmo/eficiência).
  - Como: ajusta multiplicativamente pela força ofensiva da equipa na época, conforme [`_apply_team_factor`](src/player_performance.py):
    - $adj = \text{perf}\times \frac{\text{team\_pts}}{\text{mediana(team\_pts)}}$.
  - Exemplo: perf bruta 20 numa equipa 10% acima da mediana de pontos ⇒ 22.

- Ajuste por posição/papel
  - Porquê: o “valor” das estatísticas varia por função (ex.: AST pesa mais para guards; BLK/Reb mais para centers).
  - Como: pontuação base ponderada por papel a partir de pos, conforme [`_compute_per36_metrics`](src/player_performance.py) e [`_pos_to_role_series`](src/player_performance.py).
    - Exemplo: para guard, pesos aproximados: PTS 1.0, REB 0.4, AST 1.1, STL 1.5, BLK 0.4, TOV −0.9; para center: REB 1.1, BLK 1.5, AST 0.4.
  - Efeito: dois jogadores com mesmas contagens brutas, mas papéis distintos, terão per36 diferentes, refletindo impacto tático.

- Orquestração
  - Toda a lógica acima é aplicada por [`calculate_player_performance`](src/player_performance.py), usada em [src/main.py](src/main.py).

---

## Comandos úteis
- Criar ambiente e instalar: make venv
- Executar pipeline completo: make all
- Limpar outputs: make clean_data
- Ver árvore de reports: make reports_tree
- Ajuda do Makefile: make help

---

## Próximos passos (priorizados)
1. Implementar skeleton em src/model/ranking_model.py:
   - usar data/processed/player_performance.csv e team_season_statistics.csv para treinar modelo de ranking.
2. Testes unitários:
   - adicionar pytest + testes para cleaning, features e player_performance.
3. Integração contínua:
   - GitHub Actions para rodar testes e pipeline parcial (fast checks).
4. Containerização:
   - Dockerfile para garantir ambiente reproduzível e execução em Windows/Linux.
5. Melhorias de pipeline:
   - parametrizar hyperparâmetros (decay, shrinkage) via config YAML.
   - salvar artefactos (modelos, scalers) em models/ e versionamento.
6. Documentação e exemplos:
   - notebook de baseline (notebooks/baseline.ipynb) e tutorial passo-a-passo.
7. Análises adicionais:
   - calibrar importância do treinador, explorar features derivadas de jogo (lineups).

---

## Maintainers
- Tomás Oliveira — up202208415
- Lucas Greco — up202208296

--- 

Observação: executar `make help` para referência rápida dos targets e usar `make venv` antes de correr os targets se ainda não criou o ambiente virtual.




APRESENTACAO:
PARTES QUE TIVEMOS DIFICULDADE:
FOCAR EM 1 MODELO
QUAO BOM ELE É COM COISAS QUE JA ACONTECERAM
MELHOR MODELO MAIS ESTAVEL DO QUE MODELO INSTAVEL