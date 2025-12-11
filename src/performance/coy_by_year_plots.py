"""
COY Scatter por Ano
===================

Para cada ano, gera um scatter:

  X = delta_vs_prev_team  (Δ win% vs ano anterior)
  Y = coach_overach_roster (RS win% vs expectativa do roster)

- Pontos azuis: todos os coaches
- Pontos laranja: COY naquele ano
- Labels nos pontos COY

Input:  data/processed/coach_season_performance.csv
Output: reports/plots/coach_performance/coy_scatter_year_<year>.png
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = ROOT / "data" / "processed"
PLOTS_DIR = ROOT / "reports" / "plots" / "coach_performance"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_coach_season() -> pd.DataFrame:
    path = PROC_DIR / "coach_season_performance.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} não encontrado. Gera primeiro com coach_season_performance.py."
        )
    df = pd.read_csv(path)

    # sanity check de colunas necessárias
    required = {
        "coachID",
        "year",
        "delta_vs_prev_team",
        "coach_overach_roster",
        "is_coy_winner",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Faltam colunas em coach_season_performance.csv: {missing}")

    return df


def plot_year(df_year: pd.DataFrame, year: int) -> None:
    """Gera e salva o scatter para um único ano."""
    # Só plota linhas com overach_roster válido
    df_plot = df_year.copy()
    df_plot = df_plot[df_plot["coach_overach_roster"].notna()].copy()

    if df_plot.empty:
        print(f"[{year}] Sem dados válidos, skip.")
        return

    # X: delta vs prev (NaN → 0 só para visual; primeiro ano normalmente)
    x = df_plot["delta_vs_prev_team"].fillna(0.0)
    y = df_plot["coach_overach_roster"]

    is_coy = df_plot["is_coy_winner"] == 1

    plt.figure(figsize=(8, 8))

    # não-COY
    plt.scatter(
        x[~is_coy],
        y[~is_coy],
        alpha=0.7,
        label="Não-COY",
    )

    # COY
    if is_coy.any():
        plt.scatter(
            x[is_coy],
            y[is_coy],
            alpha=0.9,
            label="COY",
        )

        # labels nos COY
        for _, row in df_plot[is_coy].iterrows():
            x_pt = 0.0 if pd.isna(row["delta_vs_prev_team"]) else row["delta_vs_prev_team"]
            y_pt = row["coach_overach_roster"]
            plt.text(
                x_pt,
                y_pt,
                row["coachID"],
                fontsize=9,
                ha="left",
                va="bottom",
            )

    # linhas de referência
    plt.axvline(0, linestyle="--", linewidth=1, alpha=0.5)
    plt.axhline(0, linestyle="--", linewidth=1, alpha=0.5)

    plt.xlabel("Δ win% vs ano anterior (RS)")
    plt.ylabel("Overach vs roster (RS)")
    plt.title(f"Year {year} – COY (laranja) vs Não-COY (azul)")
    plt.legend()
    plt.grid(alpha=0.2)

    out_path = PLOTS_DIR / f"coy_scatter_year_{year}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"[{year}] Plot salvo em: {out_path}")


def main():
    print("\n" + "=" * 66)
    print("COY SCATTERS POR ANO")
    print("=" * 66)

    df = load_coach_season()

    years = sorted(df["year"].dropna().unique())
    print(f"Anos encontrados: {years}")

    for year in years:
        df_year = df[df["year"] == year]
        plot_year(df_year, int(year))

    print("\n✓ Todos os plots gerados em:", PLOTS_DIR)
    print("=" * 66 + "\n")


if __name__ == "__main__":
    main()
