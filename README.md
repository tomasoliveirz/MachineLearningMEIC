# MachineLearningMEIC

## Project Overview

This project explores machine learning techniques applied to sports analytics. Over the past 10 years, comprehensive data has been collected on players, teams, coaches, games, and various performance metrics. The objective is to leverage this dataset to make predictions for the upcoming test season.

## Objectives

In the upcoming season, our goals are to develop predictive models for:

- **Regular Season Rankings:** Estimate the final standings for each conference based on historical and current season data.
- **Coaching Changes:** Identify teams with a high probability of changing coaches, using trends and performance indicators.
- **Individual Awards:** Forecast recipients of major individual awards by analyzing player statistics and performance metrics.

These objectives will guide our analysis and model development throughout the project.

## Dataset

The dataset includes:
- Player statistics
- Team performance metrics
- Coaching history
- Game results
- Award recipients

## Students

- Tomás Oliveira - up202208415
- Lucas Greco - up202208296

## Próximos passos

Seguem-se os próximos passos recomendados para avançar com a modelagem da classificação da regular season (explicado em português):

- 1) Preparar o dataset por equipa-época (team-season)
	- Agregar os resultados dos jogos da época regular por equipa: vitórias, derrotas, pontos marcados/sofridos, possessions (se disponíveis), diferença de pontos, estatísticas de ataque/defesa e splits casa/fora.
	- Calcular janelas temporais/forma (por exemplo % de vitórias nas últimas 5/10 partidas).

- 2) Enriquecer com features de roster e treinador
	- Fazer merge com os ficheiros limpos de `players` e `teams` para obter idade média do plantel, experiência média, minutos médios, mudanças de roster e dados do treinador (tenure, experiência).

- 3) Definir o rótulo (target)
	- Rótulo primário: número de vitórias da regular season por equipa (regressão). Alternativa: rank final dentro da conferência (ordenação/learning-to-rank).

- 4) Prevenir data leakage e validação temporal
	- Usar apenas informação disponível até o fim da época regular para cada season.
	- Validar com split temporal (train em anos anteriores, validar no ano seguinte) ou leave-one-season-out.

- 5) Baselines e modelos
	- Baselines simples: prever vitórias igual à época anterior ou média móvel.
	- Modelos: regressão linear, Random Forest, LightGBM/XGBoost. Para ordenação direta, considerar LightGBM com objetivo de ranking.

- 6) Métricas de avaliação
	- Para ordenação: Spearman rho, Kendall tau, nDCG@K (importante para top-K que qualificam aos playoffs).
	- Para previsões de vitórias: RMSE/MAE.

- 7) Pipeline mínimo sugerido (arquitectura)
	- scripts/aggregate_team_season.py  -> produz `data_processing/team_season.csv`
	- notebooks/analysis.ipynb         -> exploração, features, validação
	- notebooks/baseline.ipynb         -> baseline (ex.: LightGBM para prever vitórias) e avaliação (Spearman/nDCG)

Comandos úteis (PowerShell) para repetir a limpeza que já foi feita:

```powershell
# executar limpeza de jogadores e equipas (já existem em `data_cleaning`)
python .\data_cleaning\clean_players.py
python .\data_cleaning\clean_teams.py

# ficheiros gerados (saída):
# data_cleaning_output\players_cleaned.csv
# data_cleaning_output\teams_cleaned.csv
```

Quer que eu gere o script de agregação `scripts/aggregate_team_season.py` e um notebook com um baseline (LightGBM + métricas Spearman/nDCG)? Posso criar os ficheiros e correr um teste rápido usando os dados de treino existentes.