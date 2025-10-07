
# AC — Sports Analytics Pipeline

This repository contains a full **data pipeline for sports analytics**, focused on players, teams, rookies, and seasonal performance.  
It is structured to allow a clean end-to-end workflow: from raw data to cleaned datasets, feature engineering, and final analysis reports with visualizations.

---

## Project Structure

```bash
.
├── data
│   ├── raw/                # original CSVs
│   │   ├── players.csv
│   │   ├── teams.csv
│   │   ├── players_teams.csv
│   │   └── ...
│   ├── processed/          # cleaned + feature engineered datasets
│   │   ├── players_cleaned.csv
│   │   ├── teams_cleaned.csv
│   │   ├── team_season.csv
│   │   └── team_rookie_features.csv
├── reports
│   ├── figures/            # visualizations (charts)
│   │   ├── raw/
│   │   │   ├── height_distribution.png
│   │   │   ├── ...
│   │   └── cleaned/
│   │       ├── height_distribution.png
│   │       ├── ...
│   └── tables/             # text reports
│       ├── raw/
│       │   └── analysis_report.txt
│       └── cleaned/
│           └── analysis_report.txt
├── src
│   └── ac
│       ├── analysis/       # reporting & visualization
│       │   └── analyzer.py
│       ├── cleaning/       # cleaning scripts
│       │   ├── clean_players.py
│       │   └── clean_teams.py
│       ├── features/       # feature engineering
│       │   ├── aggregate_team_season.py
│       │   └── rookies.py
│       └── utils/
├── notebooks/              # exploratory analysis & baselines
├── requirements.txt
├── Makefile                # pipeline automation
└── README.md
````

---

## Installation

```bash
# clone repository
git clone https://github.com/<your-user>/AC.git
cd AC

# create environment
python3 -m venv venv
source venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

---

## Pipeline Usage

Everything is automated via the `Makefile`.

<details>
<summary><b>1. Clean Data</b></summary>

```bash
make clean_players
make clean_teams
```

Generated:

* `data/processed/players_cleaned.csv`
* `data/processed/teams_cleaned.csv`

</details>

<details>
<summary><b>2. Feature Engineering</b></summary>

```bash
make team_season   # aggregates team-level features
make rookies       # detects rookies & creates features
```

Generated:

* `data/processed/team_season.csv`
* `data/processed/team_rookie_features.csv`

</details>

<details>
<summary><b>3. Analysis</b></summary>

```bash
make analyze_raw       # analysis using raw data
make analyze_cleaned   # analysis using cleaned data
```

Generated:

* Figures → `reports/figures/{raw|cleaned}/`
* Tables  → `reports/tables/{raw|cleaned}/`

</details>

<details>
<summary><b>4. Run Full Pipeline</b></summary>

```bash
make all
```

This will:
`clean_players → clean_teams → team_season → rookies → analyze_cleaned`

</details>

---

## Quick Visual Check

To preview the reports structure:

```bash
make reports_tree
```

---

## Design Philosophy

* **Separation of concerns**
  Cleaning, feature engineering, and analysis live in their own modules.

* **Reproducibility**
  All steps are reproducible via `Makefile`.

* **Dual reports**
  Every analysis is produced for both **raw** and **cleaned** data, for comparison.

---

## Roadmap

* Add predictive modeling (ranking, coach changes, awards forecasts).
* Integrate notebooks → `notebooks/baseline.ipynb` for ML baselines.
* Continuous validation with leave-one-season-out strategy.

# Aula q o lucao foi
- A temporada 10 é utilizada como conjunto de teste.
- Considerar apenas o nome do time, os jogadores que iniciaram a temporada e o treinador; descartar as demais informações.
- Identificar jogadores novos (rookies) na temporada de teste - nao é possivel calcular a performance deles com base na temporada anterior. - ao invez de dar nota 0, usar a media dos rookies das temporadas anteriores
- Calcular a média de performance dos jogadores para a temporada de teste com base nos dados das temporadas anteriores.
- A média simples das performances pode ser distorcida por jogadores com poucos minutos em campo.
- Para avaliar a performance do time, recomenda-se calcular uma média ponderada das performances dos jogadores, utilizando como peso o número de jogos (ou minutos jogados) de cada atleta.
- Coach vai ter menos importancia

---

## Maintainers

* Tomás Oliveira — up202208415
* Lucas Greco — up202208296




