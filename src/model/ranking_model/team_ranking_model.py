from pathlib import Path
from typing import Tuple, Dict
from datetime import datetime
import warnings

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import mean_absolute_error

warnings.filterwarnings('ignore')

# Paths
ROOT = Path(__file__).resolve().parents[3]
PROC_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "src" / "model" / "ranking_model"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Random state for reproducibility
RANDOM_STATE = 42


# =============================================================================
# 1. DATA LOADING AND VALIDATION
# =============================================================================

def load_and_merge() -> pd.DataFrame:
    """
    Load team_season_statistics.csv and team_performance.csv, merge them.
    """
    print("[TeamRanking] Loading data...")
    
    # Load teams data (enriched with features)
    stats_path = PROC_DIR / "team_season_statistics.csv"
    if not stats_path.exists():
        raise FileNotFoundError(f"team_season_statistics.csv not found in {PROC_DIR}")
    
    df_stats = pd.read_csv(stats_path)
    print(f"  ✓ Loaded {len(df_stats)} records from team_season_statistics.csv")
    
    # Load team performance data (Pythag, roster strength, etc.)
    perf_path = PROC_DIR / "team_performance.csv"
    if not perf_path.exists():
        raise FileNotFoundError(f"team_performance.csv not found in {PROC_DIR}")
    
    df_perf = pd.read_csv(perf_path)
    print(f"  ✓ Loaded {len(df_perf)} records from team_performance.csv")
    
    # Merge on year and team
    df_all = df_stats.merge(
        df_perf,
        left_on=['year', 'tmID'],
        right_on=['year', 'team_id'],
        how='left'
    )
    
    print(f"  ✓ Merged dataset: {len(df_all)} rows")
    
    # Validate required columns exist
    required = ['year', 'tmID', 'confID', 'rank', 'name',
                'pythag_win_pct', 'overach_roster', 'overach_pythag',
                'team_strength', 'rs_win_pct_expected_roster']
    missing = set(required) - set(df_all.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    
    # Remove rows without rank or confID
    df_all = df_all.dropna(subset=['rank', 'confID'])
    print(f"  ✓ After removing missing rank/confID: {len(df_all)} rows")
    
    return df_all


# =============================================================================
# 2. TEMPORAL FEATURES (Rolling Averages & Trends)
# =============================================================================

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal features for each team:
    - Rolling averages (3 and 5 year windows)
    - Trend (slope) over 3 and 5 years
    
    Features to compute:
    - point_diff, off_eff, def_eff, pythag_win_pct, team_strength
    
    CRITICAL: Uses only past data (no leakage)
    """
    print("[TeamRanking] Adding temporal features (rolling averages & trends)...")
    
    df = df.sort_values(['tmID', 'year']).copy()
    
    # Columns to compute temporal features for
    temporal_cols = ['point_diff', 'off_eff', 'def_eff', 'pythag_win_pct', 'team_strength']
    
    # Ensure columns exist
    for col in temporal_cols:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    
    # Group by team
    for col in temporal_cols:
        # Rolling mean (3 and 5 years) - using only past data
        df[f'{col}_ma3'] = df.groupby('tmID')[col].transform(
            lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
        )
        df[f'{col}_ma5'] = df.groupby('tmID')[col].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
        )
        
        # Trend (slope) over 3 and 5 years
        def calculate_slope(series, window):
            """Calculate slope of linear regression over window"""
            if len(series) < 2:
                return 0.0
            x = np.arange(len(series))
            y = series.values
            if np.std(y) == 0:
                return 0.0
            # Simple linear regression: slope = cov(x,y) / var(x)
            slope = np.cov(x, y)[0, 1] / (np.var(x) + 1e-10)
            return slope
        
        df[f'{col}_trend3'] = df.groupby('tmID')[col].transform(
            lambda x: x.shift(1).rolling(window=3, min_periods=2).apply(
                lambda series: calculate_slope(series, 3), raw=False
            )
        )
        df[f'{col}_trend5'] = df.groupby('tmID')[col].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=2).apply(
                lambda series: calculate_slope(series, 5), raw=False
            )
        )
    
    # Fill NaN values with 0 (for teams with insufficient history)
    temporal_feature_cols = [
        f'{col}_{suffix}' 
        for col in temporal_cols 
        for suffix in ['ma3', 'ma5', 'trend3', 'trend5']
    ]
    df[temporal_feature_cols] = df[temporal_feature_cols].fillna(0.0)
    
    print(f"  ✓ Added {len(temporal_feature_cols)} temporal features")
    
    return df


# =============================================================================
# 3. TEMPORAL SPLIT
# =============================================================================

def split_train_test(
    df_all: pd.DataFrame, 
    max_train_year: int = 8
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset by year (temporal split, NO shuffle).
    """    
    train_df = df_all[df_all['year'] <= max_train_year].copy()
    test_df = df_all[df_all['year'] > max_train_year].copy()
        
    return train_df, test_df


# =============================================================================
# 4. FEATURE ENGINEERING (NO LEAKAGE)
# =============================================================================

def build_feature_matrix(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Extract features (X), target (y), and metadata from DataFrame.
    Features: Union of best features from both old scripts + temporal features.
    EXCLUDED: rank, won, lost, GP, season_win_pct, playoff flags,
                        po_W, po_L, homeW, homeL, awayW, awayL, confW, confL
    """
    feature_cols_numeric = [
        # From team_season_statistics.csv
        'point_diff', 'off_eff', 'def_eff',
        'fg_pct', 'three_pct', 'ft_pct', 'opp_fg_pct',
        'prop_3pt_shots',
        'reb_diff', 'stl_diff', 'blk_diff', 'to_diff',
        'attend_pg',
        'franchise_changed',
        'prev_win_pct_1', 'prev_win_pct_3', 'prev_win_pct_5',
        'prev_point_diff_3', 'prev_point_diff_5',
        'win_pct_change',
        'off_eff_norm', 'def_eff_norm', 'fg_pct_norm', 'three_pct_norm',
        'ft_pct_norm', 'opp_fg_pct_norm', 'point_diff_norm',
        # From team_performance.csv
        'pythag_win_pct', 'team_strength', 'rs_win_pct_expected_roster',
        'overach_pythag', 'overach_roster',
        # Temporal features (rolling averages and trends)
        'point_diff_ma3', 'point_diff_ma5', 'point_diff_trend3', 'point_diff_trend5',
        'off_eff_ma3', 'off_eff_ma5', 'off_eff_trend3', 'off_eff_trend5',
        'def_eff_ma3', 'def_eff_ma5', 'def_eff_trend3', 'def_eff_trend5',
        'pythag_win_pct_ma3', 'pythag_win_pct_ma5', 'pythag_win_pct_trend3', 'pythag_win_pct_trend5',
        'team_strength_ma3', 'team_strength_ma5', 'team_strength_trend3', 'team_strength_trend5'
    ]
    
    df_work = df.copy()
    
    # Ensure numeric features exist and are float (no silent NaNs)
    for col in feature_cols_numeric:
        if col not in df_work.columns:
            # Fill missing feature with 0 (safe default)
            df_work[col] = 0.0
        else:
            df_work[col] = pd.to_numeric(df_work[col], errors='coerce').fillna(0.0)
    
    # One-hot encode confID (East/West)
    conf_dummies = pd.get_dummies(df_work['confID'], prefix='conf', drop_first=False)
    
    # Combine features
    X = pd.concat([
        df_work[feature_cols_numeric],
        conf_dummies
    ], axis=1)
    
    # Target (rank within conference)
    y = pd.to_numeric(df_work['rank'], errors='coerce')
    
    # Metadata for analysis and reporting
    meta_df = df_work[['year', 'confID', 'tmID', 'name', 'rank']].copy()
    
    return X, y, meta_df


# =============================================================================
# 5. PAIRWISE TRAINING (Learning-to-Rank)
# =============================================================================

def generate_pairwise_data(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generate pairwise training data for learning-to-rank.
    
    For each (year, confID) group:
    - Generate pairs (A, B) for all combinations
    - X_pair = X_A - X_B
    - y_pair = 1 if rank_A < rank_B (A is better), 0 otherwise
    
    Returns:
        X_pairs: DataFrame with pairwise feature differences
        y_pairs: Binary labels (1 if first team should rank higher, 0 otherwise)
    """
    print("[TeamRanking] Generating pairwise training data...")
    
    df_work = df.copy()
    df_work['index_original'] = df_work.index
    
    X_pairs_list = []
    y_pairs_list = []
    
    # Convert X to numpy for numeric operations
    X_np = X.values.astype(float)
    X_index = X.index.tolist()
    
    # Group by (year, confID)
    for (year, conf), group in df_work.groupby(['year', 'confID']):
        indices = group['index_original'].values
        ranks = y.loc[indices].values
        
        # Generate all pairs (i, j) where i != j
        for i, idx_i in enumerate(indices):
            for j, idx_j in enumerate(indices):
                if i >= j:
                    continue
                
                rank_i = ranks[i]
                rank_j = ranks[j]
                
                # Get positions in X
                pos_i = X_index.index(idx_i)
                pos_j = X_index.index(idx_j)
                
                # Always create pair as (i, j) with appropriate label
                X_pair = X_np[pos_i] - X_np[pos_j]
                
                # Label: 1 if team i is better (lower rank), 0 if team j is better
                if rank_i < rank_j:
                    y_pair = 1  # Team i wins
                elif rank_j < rank_i:
                    y_pair = 0  # Team j wins (team i loses)
                else:
                    # Equal ranks - create both labels with 50% probability
                    y_pair = 0.5
                
                X_pairs_list.append(X_pair)
                y_pairs_list.append(y_pair)
    
    X_pairs = pd.DataFrame(X_pairs_list, columns=X.columns)
    y_pairs = np.array(y_pairs_list)
    
    # Remove ties (0.5 labels)
    mask = y_pairs != 0.5
    X_pairs = X_pairs[mask].reset_index(drop=True)
    y_pairs = y_pairs[mask]
    
    print(f"  ✓ Generated {len(X_pairs)} pairwise samples")
    print(f"    - Class 1 (better): {np.sum(y_pairs == 1)}")
    print(f"    - Class 0 (worse): {np.sum(y_pairs == 0)}")
    
    return X_pairs, y_pairs


def predict_ranks_pairwise(
    model,
    df: pd.DataFrame,
    X: pd.DataFrame
) -> np.ndarray:
    """
    Predict ranks using pairwise model.
    
    For each (year, confID) group:
    - Compute score for each team as sum of P(team_i > team_j) for all j
    - Rank teams by score (higher score = better rank)
    
    Returns:
        Array of predicted scores (not ranks - conversion done later)
    """
    df_work = df.copy()
    df_work = df_work.reset_index(drop=True)
    
    scores = np.zeros(len(df_work))
    
    # Convert X to numpy for numeric operations (reset index to match df_work)
    X_reset = X.reset_index(drop=True)
    X_np = X_reset.values.astype(float)
    
    # Group by (year, confID)
    for (year, conf), group in df_work.groupby(['year', 'confID']):
        indices = group.index.tolist()
        
        # Compute pairwise probabilities
        for i in indices:
            score_i = 0.0
            
            for j in indices:
                if i == j:
                    continue
                
                # X_pair = X_i - X_j
                X_pair = (X_np[i] - X_np[j]).reshape(1, -1)
                
                # P(team_i > team_j)
                prob_i_better = model.predict_proba(X_pair)[0, 1]
                score_i += prob_i_better
            
            scores[i] = score_i
    
    return scores


# =============================================================================
# 6. MODEL FACTORY & HYPERPARAMETER OPTIMIZATION
# =============================================================================

def create_pairwise_model() -> GradientBoostingClassifier:
    """
    Create pairwise model for learning-to-rank.
    Default hyperparameters (will be optimized).
    """
    return GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        min_samples_leaf=2,
        random_state=RANDOM_STATE
    )


# NOTE: hyperparameter optimization removed. For large training sets the
# RandomizedSearchCV-based optimization was deemed unnecessary (expensive
# and yields diminishing returns). We train a default model with
# `create_pairwise_model()` instead.


# =============================================================================
# 7. RANKING CONVERSION (score → rank within conference)
# =============================================================================

def add_predicted_rank(meta_df: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
    """
    Add predicted scores and convert to ranks within (year, confID) groups.
    For pairwise: higher score = better rank (rank 1 is best).
    """
    df_result = meta_df.copy()
    df_result['pred_score'] = y_pred
    
    # Convert scores to ranks within each (year, confID) group
    # Higher score = better team = lower rank number
    df_result['pred_rank'] = df_result.groupby(['year', 'confID'])['pred_score'].rank(
        method='first', 
        ascending=False  # Higher score gets rank 1
    ).astype(int)
    
    return df_result


# =============================================================================
# 8. EVALUATION METRICS (conference-aware)
# =============================================================================

def evaluate(df_with_ranks: pd.DataFrame, split_name: str) -> Dict:
    """
    Compute evaluation metrics for predicted rankings.
    
    Metrics (all conference-aware):
    - MAE of rank (global average)
    - Mean Spearman correlation per (year, confID) group
    - Top-K accuracy (K=1 to 10): % of groups where true champion in predicted top-K
    """
    
    # Global MAE
    mae_rank = mean_absolute_error(df_with_ranks['rank'], df_with_ranks['pred_rank'])
    
    # Per-group metrics
    spearman_corrs = []
    top_k_correct = {k: 0 for k in range(1, 11)}  # top-1 to top-10
    total_groups = 0
    
    for (year, conf), group in df_with_ranks.groupby(['year', 'confID']):
        # Spearman correlation (rank preservation)
        if len(group) > 1:
            corr, _ = spearmanr(group['rank'], group['pred_rank'])
            if not np.isnan(corr):
                spearman_corrs.append(corr)
        
        # True champion
        true_top1 = group[group['rank'] == 1]['tmID'].values
        
        # Top-K accuracy for K=1 to 10
        for k in range(1, 11):
            pred_top_k = group[group['pred_rank'] <= k]['tmID'].values
            if len(true_top1) > 0 and len(pred_top_k) > 0:
                if true_top1[0] in pred_top_k:
                    top_k_correct[k] += 1
        
        total_groups += 1
    
    mean_spearman = np.mean(spearman_corrs) if spearman_corrs else 0.0
    
    # Calculate top-K accuracies
    top_k_acc = {k: top_k_correct[k] / total_groups if total_groups > 0 else 0.0 
                 for k in range(1, 11)}
    
    metrics = {
        'mae_rank': mae_rank,
        'mean_spearman': mean_spearman,
        'n_groups': total_groups
    } 

    return metrics


# =============================================================================
# SAVE OUTPUTS (corrigido) + RELATÓRIO CONSISTENTE LENDO O CSV (fonte da verdade)
# =============================================================================


def save_predictions(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Path:
    """
    Salva predictions em CSV com formato consistente e retorna o Path do arquivo.
    Colunas garantidas: year, confID, tmID, name, rank, pred_rank, pred_score, split
    """
    print("\n[TeamRanking] Saving predictions to CSV...")

    # copy pra não poluir objetos externos
    train = train_df.copy()
    test = test_df.copy()

    train["split"] = "train"
    test["split"] = "test"

    df_all = pd.concat([train, test], ignore_index=True)

    # colunas obrigatórias na ordem desejada
    out_cols = ["year", "confID", "tmID", "name", "rank", "pred_rank", "pred_score", "split"]
    for c in out_cols:
        if c not in df_all.columns:
            df_all[c] = np.nan

    # normaliza tipos: ranks inteiros se possível, scores floats
    def safe_to_int(series):
        try:
            return series.astype(float).round().astype(int)
        except Exception:
            # se falhar, retorna NaNs
            return pd.Series([np.nan] * len(series))

    df_all["rank"] = safe_to_int(df_all["rank"])
    # pred_rank pode já existir; mantenha como inteiro quando possível
    df_all["pred_rank"] = safe_to_int(df_all["pred_rank"])
    df_all["pred_score"] = pd.to_numeric(df_all["pred_score"], errors="coerce")

    # ordenar de forma estável: year, confID, pred_rank (nulos no final)
    df_all = df_all[out_cols].sort_values(
        ["year", "confID", "pred_rank"], na_position="last", ignore_index=True
    )

    out_path = PROC_DIR / "team_ranking_predictions.csv"
    df_all.to_csv(out_path, index=False, float_format="%.6f", encoding="utf-8")

    print(f"  ✓ Saved predictions to {out_path}")
    return out_path


def save_report(
    max_train_year: int,
    best_params: Dict,
    best_cv_score: float,
    train_metrics: Dict,
    test_metrics: Dict,
    test_df: pd.DataFrame,
    report_name: str = "team_ranking_report_metrics.txt"
) -> Path:
    """
    Gera relatório com métricas calculadas a partir do CSV salvo (team_ranking_predictions.csv).
    
    Top-K Accuracy: Para cada conferência de teste, verifica se o time verdadeiro de rank K
    está previsto no rank K. A porcentagem final é a média sobre todas as conferências de teste.
    
    Args:
        max_train_year: Último ano de treino (ex: 8)
        best_params: Melhores hiperparâmetros encontrados
        best_cv_score: Melhor score de CV
        train_metrics: Métricas de treino (não usado)
        test_metrics: Métricas de teste (não usado)
        test_df: DataFrame de teste (não usado - lê do CSV)
        report_name: Nome do arquivo de relatório
        
    Returns:
        Path do arquivo de relatório gerado
    """
    report_path = REPORTS_DIR / report_name
    csv_path = PROC_DIR / "team_ranking_predictions.csv"

    # Ler CSV salvo (fonte da verdade)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV não encontrado: {csv_path}")
    
    df_all = pd.read_csv(csv_path, encoding="utf-8")
    
    # Validar colunas necessárias
    required_cols = ["year", "confID", "tmID", "name", "rank", "pred_rank", "split"]
    for col in required_cols:
        if col not in df_all.columns:
            raise ValueError(f"Coluna '{col}' não encontrada no CSV")

    # Filtrar apenas dados de teste (split == 'test' e year > max_train_year)
    df_test = df_all.loc[
        (df_all["split"] == "test") &
        (df_all["year"] > max_train_year)
    ].copy()

    if df_test.empty:
        raise ValueError(f"Nenhum dado de teste encontrado para years > {max_train_year}")

    # Normalizar tipos
    df_test["rank"] = pd.to_numeric(df_test["rank"], errors="coerce").astype('Int64')
    df_test["pred_rank"] = pd.to_numeric(df_test["pred_rank"], errors="coerce").astype('Int64')

    # ============================================================================
    # CALCULAR MÉTRICAS
    # ============================================================================
    
    # MAE (Mean Absolute Error)
    valid_mask = df_test["rank"].notna() & df_test["pred_rank"].notna()
    if valid_mask.sum() > 0:
        mae_rank = mean_absolute_error(
            df_test.loc[valid_mask, "rank"].astype(int),
            df_test.loc[valid_mask, "pred_rank"].astype(int)
        )
    else:
        mae_rank = float("nan")

    # Mean Spearman Correlation (por grupo year-confID)
    spearman_corrs = []
    for (year, conf), g in df_test.groupby(["year", "confID"]):
        g = g.dropna(subset=["rank", "pred_rank"])
        if len(g) <= 1:
            continue
        try:
            corr, _ = spearmanr(g["rank"].astype(int), g["pred_rank"].astype(int))
            if not np.isnan(corr):
                spearman_corrs.append(corr)
        except Exception:
            continue
    
    mean_spearman = float(np.mean(spearman_corrs)) if spearman_corrs else float("nan")

    # Top-K Accuracy
    # Para cada K de 1 a 10, calculamos a porcentagem de acertos considerando:
    # - Acerto: O time com rank verdadeiro K está previsto no rank K
    # - Calculado por conferência (year, confID)
    
    total_groups = 0
    top_k_correct = {k: 0 for k in range(1, 11)}
    
    for (year, conf), g in df_test.groupby(["year", "confID"]):
        # Limpar dados inválidos
        g = g.dropna(subset=["rank", "pred_rank"])
        if len(g) == 0:
            continue
        
        total_groups += 1
        
        # Converter ranks para inteiros
        g = g.copy()
        g["rank"] = g["rank"].astype(int)
        g["pred_rank"] = g["pred_rank"].astype(int)
        
        # Para cada K de 1 a 10
        for k in range(1, 11):
            # Encontrar times verdadeiros nas top-K posições (ranks 1 a K)
            true_top_k = set(g.loc[g["rank"] <= k, "tmID"].astype(str))
            
            # Encontrar times previstos nas top-K posições (pred_ranks 1 a K)
            pred_top_k = set(g.loc[g["pred_rank"] <= k, "tmID"].astype(str))
            
            # Calcular interseção (acertos)
            correct_in_top_k = len(true_top_k & pred_top_k)
            
            # Se todos os K times verdadeiros estão nos K previstos, é um acerto completo
            if correct_in_top_k == k:
                top_k_correct[k] += 1

    # Calcular porcentagens
    if total_groups == 0:
        top_k_acc = {k: 0.0 for k in range(1, 11)}
    else:
        top_k_acc = {k: top_k_correct[k] / total_groups for k in range(1, 11)}

    # ============================================================================
    # SALVAR RELATÓRIO
    # ============================================================================
    
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"GENERATED: {now} UTC\n")
        f.write(f"TRAIN_SEASONS: 1-{max_train_year}\n")
        f.write(f"TEST_SEASONS: {max_train_year + 1}+\n\n")
        f.write(f"MAE_rank: {mae_rank:.4f}\n")
        f.write(f"Mean_Spearman: {mean_spearman:.4f}\n")
        f.write(f"n_groups: {total_groups}\n\n")
        f.write("Top-K Accuracy:\n")
        for k in range(1, 11):
            f.write(f"  Top-{k:2d}: {top_k_acc[k]:.2%}\n")

    print(f"[TeamRanking] Report saved to {report_path}")
    return report_path


# =============================================================================
# 10. MAIN PIPELINE
# =============================================================================

def run_team_ranking_model(
    max_train_year: int = 8,
    report_name: str = "team_ranking_report_enhanced.txt"
) -> None:
    """
    Main pipeline: Enhanced team ranking model with temporal features and pairwise learning.
    This pipeline always trains the default pairwise model (no hyperparameter search).
    
    Args:
        max_train_year: Last season for training (default: 8)
        report_name: Output report filename
    """
    print("\n" + "=" * 80)
    print("TEAM RANKING MODEL")
    print("=" * 80)
    
    # 1. Load and merge data
    df_all = load_and_merge()
    
    # 2. Add temporal features
    df_all = add_temporal_features(df_all)
    
    # 3. Temporal split
    train_raw, test_raw = split_train_test(df_all, max_train_year)
    
    # 4. Build features
    X_train, y_train, meta_train = build_feature_matrix(train_raw)
    X_test, y_test, meta_test = build_feature_matrix(test_raw)
    
    # 5. Generate pairwise training data
    X_pairs_train, y_pairs_train = generate_pairwise_data(train_raw, X_train, y_train)
    
    # 6. Train final model (default pairwise model)
    final_model = create_pairwise_model()
    final_model.fit(X_pairs_train, y_pairs_train)
    best_params = final_model.get_params()
    best_cv_score = 0.0
    print("  ✓ Model trained")
    
    # 7. Predict using pairwise model
    y_pred_train = predict_ranks_pairwise(final_model, train_raw, X_train)
    y_pred_test = predict_ranks_pairwise(final_model, test_raw, X_test)
    print("  ✓ Predictions generated")
    
    # 8. Convert to ranks (within conference)
    train_with_ranks = add_predicted_rank(meta_train, y_pred_train)
    test_with_ranks = add_predicted_rank(meta_test, y_pred_test)
    
    # 9. Evaluate
    train_metrics = evaluate(train_with_ranks, "TRAIN (full)")
    test_metrics = evaluate(test_with_ranks, "TEST (holdout)")
    
    # 10. Save outputs
    save_predictions(train_with_ranks, test_with_ranks)
    save_report(
        max_train_year, best_params, best_cv_score,
        train_metrics, test_metrics,
        test_with_ranks, report_name
    )


# =============================================================================
# 11. CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # ======== Configurations ========
    MAX_TRAIN_YEAR = 9                 # Last
    REPORT_NAME = "team_ranking_report.txt"  # Nome do arquivo de saída
    # =====================================

    run_team_ranking_model(
        max_train_year=MAX_TRAIN_YEAR,
        report_name=REPORT_NAME
    )
