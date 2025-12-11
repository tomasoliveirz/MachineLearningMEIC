"""
COY Drivers Visualization
=========================

Lê coach_season_performance.csv e gera plots mostrando
por que tipo de perfil os treinadores ganham Coach of the Year.

Saída:
- reports/plots/coach_performance/coy_drivers_boxplots.png
- reports/plots/coach_performance/coy_drivers_scatter.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Paths
ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = ROOT / "data" / "processed"
PLOTS_DIR = ROOT / "reports" / "plots" / "coach_performance"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_coach_season_perf() -> pd.DataFrame:
    """Carrega coach_season_performance.csv e valida colunas necessárias."""
    path = PROC_DIR / "coach_season_performance.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} não encontrado. "
            "Gera primeiro com: python src/performance/coach_season_performance.py"
        )

    df = pd.read_csv(path)

    required = {
        "coachID",
        "team_id",
        "year",
        "rs_win_pct_coach",
        "coach_overach_roster",
        "delta_vs_prev_team",
        "is_coy_winner",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"Faltam colunas em coach_season_performance.csv: {missing}"
        )

    return df


def print_coy_vs_non_coy_stats(df: pd.DataFrame) -> None:
    """Imprime médias de métricas chave para COY vs Non-COY."""
    coy = df[df["is_coy_winner"] == 1]
    non = df[df["is_coy_winner"] == 0]

    print("\n============================================================")
    print("COY vs Non-COY - Estatísticas Resumidas")
    print("============================================================")

    def mean_safe(s):
        return float(s.mean()) if len(s) > 0 else float("nan")

    metrics = {
        "rs_win_pct_coach": "RS win% (coach)",
        "coach_overach_roster": "Overach vs roster",
        "delta_vs_prev_team": "Δ win% vs ano anterior",
    }

    for col, label in metrics.items():
        m_coy = mean_safe(coy[col])
        m_non = mean_safe(non[col])
        delta = m_coy - m_non
        print(
            f"{label:25s} : COY = {m_coy: .3f} | Non-COY = {m_non: .3f} | Δ = {delta: .3f}"
        )

    print(f"\nN COY     : {len(coy)}")
    print(f"N Non-COY : {len(non)}")


def plot_boxplots_coy_vs_non_coy(df: pd.DataFrame) -> Path:
    """Gera boxplots comparando COY vs Non-COY em 3 métricas chave."""
    coy = df[df["is_coy_winner"] == 1]
    non = df[df["is_coy_winner"] == 0]

    metrics = [
        ("rs_win_pct_coach", "RS win% (coach)"),
        ("coach_overach_roster", "Overach vs roster"),
        ("delta_vs_prev_team", "Δ win% vs ano anterior"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (col, label) in zip(axes, metrics):
        data = [coy[col].dropna(), non[col].dropna()]
        ax.boxplot(data, labels=["COY", "Non-COY"])
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.3)

        # Linha zero só faz sentido para deltas/overach, mas não atrapalha na win%
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)

    fig.suptitle("Coach of the Year vs Non-COY - Distribuição de Métricas")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_path = PLOTS_DIR / "coy_drivers_boxplots.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return out_path


def plot_scatter_coy_vs_non_coy(df: pd.DataFrame) -> Path:
    """
    Scatter de Δ vs_prev_team (X) vs overach_roster (Y),
    colorindo COY vs Non-COY e anotando os COY.
    """
    cols = ["delta_vs_prev_team", "coach_overach_roster", "is_coy_winner", "coachID"]
    sub = df[cols].dropna(subset=["delta_vs_prev_team", "coach_overach_roster"]).copy()

    fig, ax = plt.subplots(figsize=(7, 7))

    colors = np.where(sub["is_coy_winner"] == 1, "tab:orange", "tab:blue")
    ax.scatter(
        sub["delta_vs_prev_team"],
        sub["coach_overach_roster"],
        c=colors,
        alpha=0.7,
        edgecolors="none",
    )

    # Eixos e linhas de referência
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Δ win% vs ano anterior (RS)")
    ax.set_ylabel("Overach vs roster (RS)")
    ax.set_title("COY (laranja) vs Non-COY (azul)")

    # Anotar apenas os COY para ver onde eles caem
    coy_points = sub[sub["is_coy_winner"] == 1]
    for _, row in coy_points.iterrows():
        ax.text(
            row["delta_vs_prev_team"],
            row["coach_overach_roster"],
            row["coachID"],
            fontsize=8,
            ha="left",
            va="bottom",
        )

    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_path = PLOTS_DIR / "coy_drivers_scatter.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return out_path


def main():
    print("\n" + "=" * 70)
    print("COACH OF THE YEAR - DRIVERS VISUALIZATION")
    print("=" * 70)

    df = load_coach_season_perf()
    print_coy_vs_non_coy_stats(df)

    print("\n[1/2] Gerando boxplots COY vs Non-COY...")
    box_path = plot_boxplots_coy_vs_non_coy(df)
    print(f"      ✓ Salvo em: {box_path}")

    print("\n[2/2] Gerando scatter Δ vs overach (COY vs Non-COY)...")
    scatter_path = plot_scatter_coy_vs_non_coy(df)
    print(f"      ✓ Salvo em: {scatter_path}")

    print("\nTudo pronto!")
    print("Abre esses ficheiros para ver com os teus olhos:")
    print(f"  - {box_path}")
    print(f"  - {scatter_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
