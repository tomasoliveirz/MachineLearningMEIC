from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"


def calculate_winning_efficiency(df_players: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula a eficiência de resultados (Winning Efficiency).
    Mede se o time teve desempenho melhor ou pior que no ano anterior.
    """
    df = df_players.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["performance"] = pd.to_numeric(df["performance"], errors="coerce")
    df["tmID"] = df["tmID"].astype(str)

    # média da performance por equipe/ano
    team_perf = (
        df.groupby(["tmID", "year"], as_index=False)["performance"]
        .mean()
        .rename(columns={"performance": "team_mean_performance"})
    )

    # calcular variação ano a ano
    team_perf = team_perf.sort_values(by=["tmID", "year"])
    team_perf["prev_year_perf"] = team_perf.groupby("tmID")["team_mean_performance"].shift(1)

    # eficiência relativa (Δ% de performance)
    team_perf["winning_efficiency"] = (
        (team_perf["team_mean_performance"] - team_perf["prev_year_perf"])
        / team_perf["prev_year_perf"].abs()
    )
    team_perf["winning_efficiency"] = (
        team_perf["winning_efficiency"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )

    return team_perf[["tmID", "year", "winning_efficiency"]]


def combine_coach_metrics(metrics: list) -> pd.DataFrame:
    """
    Combina várias métricas em um único DataFrame, fazendo merge por tmID e year.
    """
    result = metrics[0]
    for df in metrics[1:]:
        result = pd.merge(result, df, on=["tmID", "year"], how="outer").fillna(0.0)

    # performance final (por enquanto só Winning Efficiency)
    result["final_performance"] = (
        result["winning_efficiency"]
        + result.get("player_development", 0)
        + result.get("surprise_factor", 0)
    )

    return result.sort_values(by=["tmID", "year"]).reset_index(drop=True)


def calculate_coach_performance():
    """
    Lê player_performance.csv, calcula métricas de coach e salva o resultado final.
    """
    player_path = PROC / "player_performance.csv"
    df_players = pd.read_csv(player_path)

    # calcular métricas individuais
    metrics = [
        calculate_winning_efficiency(df_players),
    ]

    # combinar tudo
    df_final = combine_coach_metrics(metrics)

    # salvar resultado
    out_path = PROC / "coach_performance.csv"
    df_final.to_csv(out_path, index=False)

    print(f"✅ Saved coach performance to: {out_path}")
    print(df_final.head(10))


if __name__ == "__main__":
    calculate_coach_performance()
