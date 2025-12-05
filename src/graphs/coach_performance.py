import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==============================================================================
# CONFIGURAÇÃO DE CAMINHOS
# ==============================================================================
# Determine the project root directory (absolute path)
# Assuming script is in src/graphs/, so we go up 3 levels to reach project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CSV_PATH = os.path.join(BASE_DIR, "data", "processed", "coach_season_facts_performance.csv")
GRAPH_DIR = os.path.join(BASE_DIR, "reports", "performance_graphs", "coach_performance")

os.makedirs(GRAPH_DIR, exist_ok=True)

# Configuração visual para slides (fontes maiores, fundo limpo)
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['figure.dpi'] = 150

def load_data(csv_path):
    """Carrega e prepara os dados calculando métricas auxiliares."""
    try:
        df = pd.read_csv(csv_path)
        
        # Calcular 'Overachievement' (Performance Real vs Esperada pelo Elenco)
        # Se o técnico ganhou mais do que o elenco sugere, ele agregou valor.
        if 'rs_win_pct_expected_roster' in df.columns:
            df['overachievement'] = df['rs_win_pct_coach'] - df['rs_win_pct_expected_roster']
        else:
            df['overachievement'] = 0.0
            
        print(f"✓ Dados carregados: {len(df)} linhas.")
        return df
    except FileNotFoundError:
        print(f"❌ Arquivo não encontrado: {csv_path}")
        return None

def generate_high_value_graphics(df):
    """
    Gera gráficos focados em insights analíticos para apresentação.
    """
    print("Gerando gráficos analíticos...")

    # ==========================================================================
    # GRÁFICO 1: O Efeito da Suavização Bayesiana (Shrinkage)
    # MOTIVO: Mostrar como o modelo corrige técnicos com poucos jogos (Outliers).
    # ==========================================================================
    plt.figure(figsize=(10, 6))
    
    # Scatter plot: Eixo X = Win% Real, Eixo Y = Win% Bayesiana
    # O tamanho dos pontos é o número de jogos (GP)
    sns.scatterplot(
        data=df, 
        x="rs_win_pct_coach", 
        y="eb_rs_win_pct", 
        size="gp", 
        sizes=(20, 400), 
        alpha=0.6, 
        color="royalblue",
        legend="brief"
    )
    
    # Linha de identidade (onde X = Y)
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=1.5, label='Sem Ajuste')
    
    plt.title("Correção Bayesiana: Efeito em Amostras Pequenas", fontsize=16, fontweight='bold')
    plt.xlabel("Taxa de Vitórias Real (Raw)", fontsize=12)
    plt.ylabel("Taxa de Vitórias Ajustada (Bayesiana)", fontsize=12)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(title="Jogos (GP)", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    save_path = os.path.join(GRAPH_DIR, "1_analise_bayesiana_shrinkage.png")
    plt.savefig(save_path)
    plt.close()
    print(f"  ✓ Gráfico 1 salvo: {save_path}")

    # ==========================================================================
    # GRÁFICO 2: Performance do Técnico vs. Talento do Elenco
    # MOTIVO: Identificar quem faz "mais com menos" (tática vs. elenco).
    # ==========================================================================
    if 'rs_win_pct_expected_roster' in df.columns:
        plt.figure(figsize=(10, 6))
        
        # Filtrar apenas técnicos com mais de 20 jogos para limpar o gráfico
        df_filtered = df[df['gp'] >= 20]
        
        sns.scatterplot(
            data=df_filtered,
            x="rs_win_pct_expected_roster",
            y="rs_win_pct_coach",
            hue="overachievement",
            palette="vlag_r", # Vermelho = Ruim, Azul = Bom
            size="gp",
            sizes=(50, 300),
            alpha=0.8
        )
        
        # Linha de expectativa (y=x)
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=2, label='Expectativa')
        
        plt.text(0.8, 0.9, "Superação (Bom Técnico)", color='blue', fontsize=10, ha='center')
        plt.text(0.8, 0.6, "Sub-performance", color='red', fontsize=10, ha='center')
        
        plt.title("Valor Agregado: Técnico vs. Qualidade do Elenco", fontsize=16, fontweight='bold')
        plt.xlabel("Expectativa de Vitórias (Baseado no Elenco)", fontsize=12)
        plt.ylabel("Vitórias Reais do Técnico", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        save_path = os.path.join(GRAPH_DIR, "2_tecnico_vs_elenco.png")
        plt.savefig(save_path)
        plt.close()
        print(f"  ✓ Gráfico 2 salvo: {save_path}")

    # ==========================================================================
    # GRÁFICO 3: Impacto do "Choque de Gestão" (1º Ano vs Veteranos)
    # MOTIVO: Responder "Técnicos novos melhoram o time imediatamente?"
    # ==========================================================================
    plt.figure(figsize=(8, 6))
    
    # Criar labels legíveis
    df['Status'] = df['is_first_year_with_team'].map({1: 'Estreia no Time (1º Ano)', 0: 'Veterano no Cargo'})
    
    # Boxplot da Superação
    sns.boxplot(
        data=df,
        x="Status",
        y="overachievement",
        palette="Set2",
        showfliers=False # Ocultar outliers extremos para focar na mediana
    )
    
    # Linha zero (neutro)
    plt.axhline(0, color='black', linestyle=':', linewidth=1)
    
    plt.title("Impacto Imediato: Superação no 1º Ano vs. Seguinte", fontsize=14, fontweight='bold')
    plt.xlabel("", fontsize=12)
    plt.ylabel("Superação (Real - Esperado)", fontsize=12)
    plt.tight_layout()
    
    save_path = os.path.join(GRAPH_DIR, "3_impacto_primeiro_ano.png")
    plt.savefig(save_path)
    plt.close()
    print(f"  ✓ Gráfico 3 salvo: {save_path}")
    
    # ==========================================================================
    # GRÁFICO 4: Top 10 Técnicos (Ranking Bayesiano)
    # MOTIVO: Mostrar a utilidade do ranking final.
    # ==========================================================================
    plt.figure(figsize=(10, 8))
    
    # Filtrar técnicos com volume razoável (>50 jogos na carreira somada no dataset)
    coach_career = df.groupby('coachID').agg({
        'eb_rs_win_pct': 'mean', # Média das temporadas suavizadas
        'gp': 'sum'
    }).reset_index()
    
    top_coaches = coach_career[coach_career['gp'] > 82].nlargest(10, 'eb_rs_win_pct')
    
    sns.barplot(
        data=top_coaches,
        y="coachID",
        x="eb_rs_win_pct",
        palette="viridis"
    )
    
    plt.title("Top 10 Técnicos Mais Consistentes (Métrica Bayesiana)", fontsize=16, fontweight='bold')
    plt.xlabel("Win % Bayesiana Média", fontsize=12)
    plt.ylabel("Coach ID", fontsize=12)
    plt.xlim(0, 1)
    plt.tight_layout()
    
    save_path = os.path.join(GRAPH_DIR, "4_top_10_coaches.png")
    plt.savefig(save_path)
    plt.close()
    print(f"  ✓ Gráfico 4 salvo: {save_path}")

def main():
    df = load_data(CSV_PATH)
    if df is not None:
        generate_high_value_graphics(df)
        print("\nProcesso concluído! Os gráficos estão prontos para sua apresentação.")

if __name__ == "__main__":
    main()