"""
Módulo de Gráficos para o Modelo de Classificação (Change Coaches)

Este script gera e salva os 4 gráficos de diagnóstico essenciais:
1. Importância das Features (Feature Importance)
2. Matriz de Confusão (Confusion Matrix)
3. Curva ROC (ROC Curve)
4. Curva de Precisão-Recall (Precision-Recall Curve)
"""

import pandas as pd
from pathlib import Path
from typing import List, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    confusion_matrix
)

# Configurar o estilo dos gráficos
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100

def plot_feature_importance(
    model: Any, 
    feature_names: List[str], 
    save_path: Path
):
    """Gera um gráfico de barras da importância das features."""
    try:
        importances = model.feature_importances_
        fi_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values(by="importance", ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x="importance",
            y="feature",
            data=fi_df.head(15), # Limita às top 15
            palette="viridis"
        )
        plt.title("Importância das Features (RandomForest)")
        plt.xlabel("Importância (MDI)")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Erro ao gerar gráfico de importância: {e}")

def plot_confusion_matrix(
    model: Any, 
    X: pd.DataFrame, 
    y_true: pd.Series, 
    save_path: Path
):
    """Gera um heatmap da matriz de confusão."""
    try:
        y_pred = model.predict(X)
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(7, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d", # Formato de inteiro
            cmap="Blues",
            xticklabels=["Manteve (0)", "Despedido (1)"],
            yticklabels=["Manteve (0)", "Despedido (1)"]
        )
        plt.title("Matriz de Confusão (Dados de Teste)")
        plt.xlabel("Previsão do Modelo")
        plt.ylabel("Realidade")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Erro ao gerar matriz de confusão: {e}")

def plot_roc_curve(
    model: Any, 
    X: pd.DataFrame, 
    y_true: pd.Series, 
    save_path: Path
):
    """Gera a curva ROC e calcula o AUC."""
    try:
        y_proba = model.predict_proba(X)[:, 1] # Probabilidade da classe "1"
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(7, 6))
        plt.plot(
            fpr, 
            tpr, 
            color="darkorange", 
            lw=2, 
            label=f"Curva ROC (AUC = {roc_auc:.2f})"
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Aleatório")
        plt.xlabel("Taxa de Falsos Positivos")
        plt.ylabel("Taxa de Verdadeiros Positivos")
        plt.title("Curva ROC (Receiver Operating Characteristic)")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Erro ao gerar curva ROC: {e}")

def plot_precision_recall_curve(
    model: Any, 
    X: pd.DataFrame, 
    y_true: pd.Series, 
    save_path: Path
):
    """Gera a curva de Precisão-Recall."""
    try:
        y_proba = model.predict_proba(X)[:, 1]
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        
        plt.figure(figsize=(7, 6))
        plt.plot(recall, precision, color="darkblue", lw=2)
        plt.xlabel("Recall (Sensibilidade)")
        plt.ylabel("Precision (Precisão)")
        plt.title("Curva de Precisão-Recall")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Erro ao gerar curva P-R: {e}")

def generate_all_graphics(
    model: Any, 
    X_test: pd.DataFrame, 
    y_test: pd.Series, 
    feature_names: List[str], 
    report_dir: Path
):
    """
    Função principal para orquestrar a geração de todos os gráficos.
    """
    print("\n Gerando gráficos de diagnóstico do modelo...")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Feature Importance
    plot_feature_importance(model, feature_names, report_dir / "feature_importance.png")
    
    # 2. Confusion Matrix
    plot_confusion_matrix(model, X_test, y_test, report_dir / "confusion_matrix.png")
    
    # 3. ROC Curve
    plot_roc_curve(model, X_test, y_test, report_dir / "roc_curve.png")
    
    # 4. Precision-Recall Curve
    plot_precision_recall_curve(model, X_test, y_test, report_dir / "precision_recall_curve.png")
    
    print(f" Gráficos salvos em: {report_dir}")