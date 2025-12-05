import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def create_and_save_graph(csv_path):
    """
    Cria um gráfico comparando os minutos jogados e a performance dos jogadores e salva em uma subpasta 'player_performance'.
    
    Args:
        csv_path (str): Caminho para o arquivo CSV com os dados de desempenho dos jogadores.
    """
    # Carregar os dados do CSV
    df = pd.read_csv(csv_path)

    # Diretório para salvar o gráfico
    # Assuming script is in src/graphs, go up to project root
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    output_dir = BASE_DIR / "reports" / "performance_graphs" / "players_performance"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Criar o gráfico
    plt.figure(figsize=(12, 8))
    plt.scatter(df["minutes"], df["performance"], alpha=0.7)
    plt.title("Comparação de Minutos Jogados e Performance")
    plt.xlabel("Minutos Jogados")
    plt.ylabel("Performance")
    plt.grid(True)

    # Salvar o gráfico
    output_path = output_dir / "comparison_minutes_performance.png"
    plt.savefig(output_path)
    plt.close()

    print(f"Gráfico salvo em: {output_path}")

# Exemplo de uso
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    csv_path = BASE_DIR / "data" / "processed" / "player_performance.csv"
    create_and_save_graph(str(csv_path))