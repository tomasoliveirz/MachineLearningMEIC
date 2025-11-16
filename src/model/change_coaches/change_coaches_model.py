from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, classification_report
from sklearn.exceptions import UndefinedMetricWarning

# Paths
ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports" / "models" / "change_coaches"
RANDOM_STATE = 42

# Features
FEATURE_COLS = [
    'overach_roster',            # MAIS FORTE: Superou expectativa do roster
    'is_first_year',             # Primeiro ano do coach com o time
    'lag1_point_diff',           # Point diff da temporada ANTERIOR
    'current_point_diff',        # Point diff atual
    'made_playoffs',             # Se classificou para playoffs (0/1)
    'attend_pg',                 # Público médio (indicador de suporte da torcida)
]
TARGET_COL = 'coach_changed'

# Split temporal estrito
TRAIN_END_YEAR = 6
VAL_END_YEAR = 8


# --- 1. Carregamento e Preparação de Dados ---

def _create_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Função auxiliar para criar a variável-alvo (label)."""
    if 'coachID' not in df.columns or 'team_id' not in df.columns or 'year' not in df.columns:
        raise ValueError("create_labels: Faltam colunas 'coachID', 'team_id' ou 'year'.")

    df = df.sort_values(by=["team_id", "year"])
    df["next_coachID"] = df.groupby("team_id")["coachID"].shift(-1)
    
    df[TARGET_COL] = (
        (df["next_coachID"].notna()) & (df["coachID"] != df["next_coachID"])
    ).astype(int)
    
    return df.drop(columns=["next_coachID"])

def _normalize_keys(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Normaliza nomes de colunas comuns para 'team_id' e 'year'."""
    if df.empty:
        return df
        
    rename_map = {}
    if "team_id" not in df.columns and "tmID" in df.columns:
        rename_map["tmID"] = "team_id"
    if "year" not in df.columns and "season_year" in df.columns:
        rename_map["season_year"] = "year"
    
    if rename_map:
        df = df.rename(columns=rename_map)
        
    return df

def load_and_prepare_data() -> pd.DataFrame:
    """
    Carrega, normaliza chaves, funde (merge) todos os datasets e cria a label.
    Retorna um único DataFrame "mestre" pronto para a engenharia de features.
    """
    print(" Carregando e preparando dados...")
    # 1. Carregar
    coach_df = pd.read_csv(DATA_DIR / "coach_season_performance.csv")
    team_stats_df = pd.read_csv(DATA_DIR / "team_season_statistics.csv")
    team_perf_df = pd.read_csv(DATA_DIR / "team_performance.csv")


    # 2. Normalizar Chaves (antes de fundir)
    coach_df = _normalize_keys(coach_df, "coach_performance")
    team_stats_df = _normalize_keys(team_stats_df, "team_statistics")
    team_perf_df = _normalize_keys(team_perf_df, "team_performance")

    # 3. Criar Labels
    coach_df = _create_labels(coach_df)

    # 4. Fundir (Merge)
    join_keys = ["team_id", "year"]
    
    # Começa com os treinadores (que têm a label)
    master_df = coach_df
    
    # Funde com team_stats
    if not team_stats_df.empty:
        master_df = pd.merge(
            master_df,
            team_stats_df,
            on=join_keys,
            how="left",
            suffixes=("", "_stats")
        )
        
    # Funde com team_performance
    if not team_perf_df.empty:
        # Seleciona colunas únicas para evitar sobreposição, exceto as chaves
        perf_cols = list(team_perf_df.columns.difference(master_df.columns)) + join_keys
        master_df = pd.merge(
            master_df,
            team_perf_df[perf_cols].drop_duplicates(),
            on=join_keys,
            how="left",
            suffixes=("", "_perf")
        )

    print(f" Dados carregados: {len(master_df)} registros")
    return master_df

# --- 2. Engenharia de Features ---

def _first_col(df: pd.DataFrame, *cols: str) -> pd.Series:
    """Auxiliar: Retorna a primeira coluna da lista que existe no DataFrame."""
    for c in cols:
        if c in df.columns:
            return df[c]
    return pd.Series([np.nan] * len(df), index=df.index)

def build_feature_matrix(master_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Constrói a matriz de features (X) e o vetor alvo (y).
    """
    print(" Construindo features anti-leakage...")
    if master_df.empty or TARGET_COL not in master_df.columns:
        print(" DataFrame mestre vazio ou sem a coluna alvo.")
        return pd.DataFrame(), pd.Series()

    # Ordenar por time e ano para criar lags corretamente
    df = master_df.sort_values(['team_id', 'year']).copy()
    
    # Primeiro, extrair as colunas base que vamos usar
    win_pct_raw = _first_col(df, "season_win_pct", "rs_win_pct").fillna(0)
    point_diff_raw = _first_col(df, "point_diff").fillna(0)
    
    features = pd.DataFrame(index=df.index)

    # 1. Performance ATUAL da temporada (já conhecida no final)
    features["current_win_pct"] = win_pct_raw
    features["current_point_diff"] = point_diff_raw
    
    # 2. Performance da temporada ANTERIOR (lag1 = t-1)
    # Shift(1) pega o valor da LINHA ANTERIOR do mesmo time
    features["lag1_win_pct"] = df.groupby("team_id")["team_id"].transform(
        lambda x: win_pct_raw.loc[x.index].shift(1)
    ).fillna(0.5)
    
    features["lag1_point_diff"] = df.groupby("team_id")["team_id"].transform(
        lambda x: point_diff_raw.loc[x.index].shift(1)
    ).fillna(0)
    
    # 3. Expectativas (calculadas com dados atuais, sem futuro)
    features["pythag_win_pct"] = _first_col(df, "pythag_win_pct").fillna(0.5)
    features["team_strength"] = _first_col(df, "team_strength").fillna(0)
    
    # 4. Playoffs (resultado atual)
    playoff_col = _first_col(df, "playoff")
    if pd.api.types.is_string_dtype(playoff_col):
        features["made_playoffs"] = playoff_col.map({"Y": 1, "N": 0}).fillna(0).astype(int)
    else:
        features["made_playoffs"] = (playoff_col.fillna(0) > 0).astype(int)

    # 5. Contexto do coach (histórico, sem futuro)
    # is_first_year: se stint == 0 (ou calcular baseado em histórico)
    if "stint" in df.columns:
        features["is_first_year"] = (df["stint"] == 0).astype(int)
    else:
        features["is_first_year"] = 0
    
    # 6. Overachievement (diferença entre resultado e expectativa - ATUAL)
    # overach_pythag = current_win_pct - pythag_win_pct
    features["overach_pythag"] = _first_col(df, "overach_pythag").fillna(
        features["current_win_pct"] - features["pythag_win_pct"]
    )
    
    # overach_roster: se disponível (diferença entre resultado e expectativa do roster)
    features["overach_roster"] = _first_col(df, "overach_roster").fillna(0)
    
    # 7. Attendance (público médio da temporada atual)
    if "attend" in df.columns and "GP" in df.columns:
        features["attend_pg"] = (pd.to_numeric(df["attend"], errors='coerce') / 
                                 pd.to_numeric(df["GP"], errors='coerce')).fillna(0)
    else:
        features["attend_pg"] = _first_col(df, "attend_pg").fillna(0)

    # Garantir que todas as features são numéricas
    X = features[FEATURE_COLS].copy()
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Alvo
    y = df[TARGET_COL].astype(int)
    
    # Adiciona 'year' para a divisão temporal (será removido depois do split)
    X['year'] = df['year']
    
    print(f" Features construídas: {X.shape[0]} linhas, {len(FEATURE_COLS)} features")
    return X, y

# --- 3. Divisão Temporal ---

def split_data_temporal(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Divide X e y temporalmente.
    RandomForest não precisa de normalização (árvores são invariantes a escala).
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    print(f"\n Split temporal: Treino (≤{TRAIN_END_YEAR}), Val ({TRAIN_END_YEAR+1}-{VAL_END_YEAR}), Teste (>{VAL_END_YEAR})")
    
    if 'year' not in X.columns:
        raise ValueError("X precisa ter 'year' para split temporal.")

    train_mask = X['year'] <= TRAIN_END_YEAR
    val_mask = (X['year'] > TRAIN_END_YEAR) & (X['year'] <= VAL_END_YEAR)
    test_mask = X['year'] > VAL_END_YEAR
    
    # Remover 'year' antes de retornar
    X_no_year = X.drop(columns=['year'])
    
    X_train = X_no_year[train_mask].copy()
    y_train = y[train_mask].copy()
    
    X_val = X_no_year[val_mask].copy()
    y_val = y[val_mask].copy()
    
    X_test = X_no_year[test_mask].copy()
    y_test = y[test_mask].copy()
    
    print(f"   Treino: {len(y_train)} samples (positivos: {y_train.sum()} = {100*y_train.sum()/len(y_train):.1f}%)")
    print(f"   Val:    {len(y_val)} samples (positivos: {y_val.sum()} = {100*y_val.sum()/len(y_val):.1f}%)")
    print(f"   Teste:  {len(y_test)} samples (positivos: {y_test.sum()} = {100*y_test.sum()/len(y_test):.1f}%)")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# --- 4. Treino do Modelo (RandomForest com Regularização) ---

def train_model(X_train: pd.DataFrame, y_train: pd.Series, max_depth: int = 5, min_samples_leaf: int = 5) -> RandomForestClassifier:
    print(f"\n Treinando RandomForest (max_depth={max_depth}, min_samples_leaf={min_samples_leaf})...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    print(f" Modelo treinado: {len(model.estimators_)} árvores")
    return model

# --- 5. Avaliação do Modelo ---

def evaluate_model(model: Any, X: pd.DataFrame, y: pd.Series, split_name: str) -> Dict[str, float]:
    """
    Calcula métricas de classificação para um conjunto de dados.
    
    Retorna métricas realistas (não infladas por leakage):
    - AUC: capacidade de rankear corretamente
    - Precision: quando prevê despedimento, acerta quanto?
    - Recall: de todos os despedimentos reais, quantos captura?
    - F1: balanço entre precision e recall
    """        
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    # Evitar divisão por zero se não houver positivos
    n_positive = y.sum()
    n_total = len(y)
    
    try:
        auc = roc_auc_score(y, y_proba)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
    except Exception as e:
        print(f"⚠️  Erro ao calcular métricas: {e}")
        return {}

    metrics = {
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
    }
    
    print(f"\n{'='*60}")
    print(f"Métricas - {split_name}")
    print(f"{'='*60}")
    print(f"   Total de exemplos:     {n_total}")
    print(f"   Positivos (mudanças):  {n_positive} ({100*n_positive/n_total:.1f}%)")
    print(f"   Negativos (manteve):   {n_total - n_positive} ({100*(n_total-n_positive)/n_total:.1f}%)")
    print(f"\n   {'Métrica':<15} {'Valor':>10}")
    print(f"   {'-'*25}")
    for name, value in metrics.items():
        print(f"   {name:<15} {value:>10.4f}")
    print(f"{'='*60}\n")
        
    return metrics

# --- 6. Report ---

def save_report(model: RandomForestClassifier, feature_names: list, train_metrics: dict, val_metrics: dict, test_metrics: dict, wf_metrics: dict = None):
    """
    Mostra quais features são mais importantes para o modelo e salva relatório em REPORT_DIR.
    Gera:
     - arquivo texto com métricas e importância das features
     - CSV com importância das features
    """

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    txt_path = REPORT_DIR / f"change_coaches_report.txt"
    csv_path = REPORT_DIR / f"feature_importance.csv"

    # Feature importances do RandomForest
    importances = model.feature_importances_
    fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi_df = fi_df.sort_values("importance", ascending=False)

    # Salva CSV com importância
    fi_df.to_csv(csv_path, index=False, encoding="utf-8")

    # Construir conteúdo do relatório
    lines = []
    lines.append(f"Relatório Change Coaches - RandomForest")
    lines.append("=" * 80)
    lines.append(f"{'Feature':<30} {'Importância':>15}")
    lines.append("-" * 80)
    for _, row in fi_df.iterrows():
        feat = row["feature"]
        imp = row["importance"]
        lines.append(f"{feat:<30} {imp:>15.4f}")

    lines.append("\nMétricas (treino - sanity check):")
    for k, v in (train_metrics or {}).items():
        lines.append(f"  {k:15s}: {v:.4f}")
    
    if wf_metrics:
        lines.append("\nMétricas (walk-forward validation):")
        for k, v in wf_metrics.items():
            lines.append(f"  {k:15s}: {v:.4f}")
    
    lines.append("\nMétricas (TESTE - resultado final):")
    for k, v in (test_metrics or {}).items():
        lines.append(f"  {k:15s}: {v:.4f}")

    
    # Escrever arquivo texto
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Imprimir resumo no console
    print(f"\nRelatório salvo em:\n - Texto: {txt_path}\n - CSV:   {csv_path}")

# --- 7. Validação Walk-Forward  ---

def run_walk_forward_validation(X: pd.DataFrame, y: pd.Series, max_depth: int = 5, min_samples_leaf: int = 5) -> Dict[str, float]:
    """
    Validação temporal "walk-forward" para estimativa ROBUSTA de desempenho.
    Problema resolvido: Conjuntos de val/test muito pequenos geram métricas instáveis.
    Estratégia:
    - Fold 1: Treina em anos 1-6 → Valida em ano 7
    - Fold 2: Treina em anos 1-7 → Valida em ano 8
    
    Calcula média das métricas nos folds. Esta é a estimativa REAL de generalização.
    
    Returns:
        Dict com médias de AUC, Precision, Recall, F1
    """
    print("\n" + "="*80)
    print(" VALIDAÇÃO WALK-FORWARD (Estimativa Robusta)")
    print("="*80)
    
    if 'year' not in X.columns:
        raise ValueError("X precisa ter 'year' para walk-forward validation.")
    
    folds_results = []
    
    # Fold 1: Treina 1-6, Valida 7
    fold_configs = [
        {"train_end": 6, "val_year": 7, "name": "Fold 1 (treino 1-6, val 7)"},
        {"train_end": 7, "val_year": 8, "name": "Fold 2 (treino 1-7, val 8)"},
    ]
    
    for config in fold_configs:
        print(f"\n--- {config['name']} ---")
        
        # Split temporal para este fold
        train_mask = X['year'] <= config['train_end']
        val_mask = X['year'] == config['val_year']
        
        X_no_year = X.drop(columns=['year'])
        X_fold_train = X_no_year[train_mask].copy()
        y_fold_train = y[train_mask].copy()
        X_fold_val = X_no_year[val_mask].copy()
        y_fold_val = y[val_mask].copy()
        
        print(f"   Treino: {len(y_fold_train)} samples (positivos: {y_fold_train.sum()})")
        print(f"   Val:    {len(y_fold_val)} samples (positivos: {y_fold_val.sum()})")
        
        # Treinar e avaliar
        model = train_model(X_fold_train, y_fold_train, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        metrics = evaluate_model(model, X_fold_val, y_fold_val, split_name=config['name'])
        
        folds_results.append(metrics)
    
    # Calcular médias
    avg_metrics = {}
    for metric_name in ['AUC', 'Precision', 'Recall', 'F1-Score']:
        values = [fold[metric_name] for fold in folds_results if metric_name in fold]
        avg_metrics[metric_name] = np.mean(values) if values else 0.0
    

    return avg_metrics

# --- 8. Pipeline Principal ---

def main():    
    # 1. Carregar e Preparar
    master_df = load_and_prepare_data()

    # 2. Construir Features (6 features simplificadas, sem leakage)
    X, y = build_feature_matrix(master_df)

    # 3. Dividir temporalmente (SEM normalização para RandomForest)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data_temporal(X, y)
    
    # Usa dados de treino+val (anos 1-8) para validação cruzada temporal
    X_for_wf = X[X['year'] <= VAL_END_YEAR].copy()
    y_for_wf = y[X['year'] <= VAL_END_YEAR].copy()
    wf_metrics = run_walk_forward_validation(X_for_wf, y_for_wf, max_depth=5, min_samples_leaf=5)
    
    # 5. Treinar Modelo FINAL com TODOS os dados de treino+val (anos 1-8)
    print("\n Treinando modelo FINAL com anos 1-8...")
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])
    model_final = train_model(X_train_full, y_train_full, max_depth=5, min_samples_leaf=5)
    
    # 6. Avaliar modelo final
    print("\n Avaliando modelo final...")
    train_metrics = evaluate_model(model_final, X_train_full, y_train_full, split_name="TREINO FULL (1-8) - sanity check")
    val_metrics = evaluate_model(model_final, X_val, y_val, split_name="VALIDAÇÃO (8) - single fold")
    test_metrics = evaluate_model(model_final, X_test, y_test, split_name="TESTE (9+) - RESULTADO FINAL")
    
    # 7. Relatório com walk-forward metrics
    save_report(model_final, FEATURE_COLS, train_metrics, val_metrics, test_metrics, wf_metrics)

if __name__ == "__main__":
    main()