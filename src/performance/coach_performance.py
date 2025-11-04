from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

def calculate_winning_efficiency(df_coaches: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula a eficiência de resultados (Winning Efficiency):
    compara o desempenho do time no ano atual com o ano anterior.
    """
    df = df_coaches.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["win_rate"] = df["won"] / (df["won"] + df["lost"])
    df["tmID"] = df["tmID"].astype(str)

    team_perf = (
        df.groupby(["tmID", "year"], as_index=False)["win_rate"]
        .mean()
        .rename(columns={"win_rate": "team_win_rate"})
    )

    # calcular variação ano a ano
    team_perf = team_perf.sort_values(by=["tmID", "year"])
    team_perf["prev_win_rate"] = team_perf.groupby("tmID")["team_win_rate"].shift(1)

    team_perf["winning_efficiency"] = (
        (team_perf["team_win_rate"] - team_perf["prev_win_rate"])
        / team_perf["prev_win_rate"].abs()
    )
    team_perf["winning_efficiency"] = (
        team_perf["winning_efficiency"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )

    return team_perf[["tmID", "year", "winning_efficiency"]]

def calculate_win_ratio(df_coaches: pd.DataFrame, df_teams: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula a quantidade de vitórias comparada ao número de jogos por ano.
    Mostra se o time ganhou mais do que perdeu.
    """
    df_c = df_coaches.copy()
    df_t = df_teams.copy()

    # garantir tipos
    for col in ["year", "W", "L", "won", "lost", "post_wins", "post_losses"]:
        if col in df_c.columns:
            df_c[col] = pd.to_numeric(df_c[col], errors="coerce").fillna(0)
        if col in df_t.columns:
            df_t[col] = pd.to_numeric(df_t[col], errors="coerce").fillna(0)

    # total de jogos (regular + playoffs)
    df_c["total_wins"] = df_c["won"] + df_c["post_wins"]
    df_c["total_losses"] = df_c["lost"] + df_c["post_losses"]
    df_c["games_played"] = df_c["total_wins"] + df_c["total_losses"]

    # taxa de vitórias do coach
    df_c["win_rate"] = 0.0
    mask = df_c["games_played"] > 0
    df_c.loc[mask, "win_rate"] = df_c.loc[mask, "total_wins"] / df_c.loc[mask, "games_played"]

    # indicar se o técnico teve mais vitórias que derrotas
    df_c["winning_season"] = df_c["win_rate"] > 0.5

    # juntar info do teams_post (pra comparação adicional se quiser)
    df_c = pd.merge(
        df_c,
        df_t[["year", "tmID", "W", "L"]],
        on=["year", "tmID"],
        how="left",
        suffixes=("", "_team"),
    )

    return df_c[
        ["coachID", "tmID", "year", "total_wins", "total_losses", "win_rate", "winning_season"]
    ]

#Main

def combine_coach_metrics(metrics: list) -> pd.DataFrame:
    """Combina várias métricas (merge por tmID e year)."""
    result = metrics[0]
    for df in metrics[1:]:
        result = pd.merge(result, df, on=["tmID", "year"], how="outer").fillna(0.0)

    # performance final (por enquanto soma simples)
    result["final_performance"] = (
        result.get("winning_efficiency", 0) + result.get("win_rate", 0)
    )

    return result.sort_values(by=["tmID", "year"]).reset_index(drop=True)

def calculate_coach_performance():
    """Executa o pipeline completo e salva o CSV."""
    coaches_path = RAW / "coaches.csv"
    teams_path = RAW / "teams_post.csv"

    df_coaches = pd.read_csv(coaches_path)
    df_teams = pd.read_csv(teams_path)

    metrics = [
        calculate_win_ratio(df_coaches, df_teams),
        calculate_winning_efficiency(df_coaches),
    ]

    df_final = combine_coach_metrics(metrics)

    out_path = PROC / "coach_performance.csv"
    df_final.to_csv(out_path, index=False)

    print(f"✅ Saved coach performance to: {out_path}")
    print(df_final.head(10))


if __name__ == "__main__":
    calculate_coach_performance()
