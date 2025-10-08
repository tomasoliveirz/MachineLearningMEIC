"""
Modelo de Machine Learning para prever o ranking das equipes com base na performance dos jogadores.

Este módulo implementa:
1. Agregação da performance dos jogadores por equipe/temporada
2. Criação do ranking das equipes (target variable)
3. Preparação dos dados de treino (temporadas 1-9) e teste (temporada 10)
4. Treinamento e avaliação do modelo
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings

from player_performance import calculate_player_performance, is_rookie

warnings.filterwarnings('ignore')


def aggregate_team_performance(player_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega a performance dos jogadores por equipe/temporada.
    
    Para cada equipe em cada temporada, calcula estatísticas agregadas da performance
    dos jogadores (média, mediana, std, max, min, soma) e outras features relevantes
    como número de jogadores, número de rookies, etc.
    
    Args:
        player_stats: DataFrame com performance calculada por jogador/temporada
        
    Returns:
        DataFrame com uma linha por equipe/temporada e features agregadas
    """
    
    # Garantir que temos as colunas necessárias
    required_cols = ['tmID', 'year', 'performance', 'rookie']
    if not all(col in player_stats.columns for col in required_cols):
        raise ValueError(f"player_stats deve conter as colunas: {required_cols}")
    
    # Criar DataFrame de agregação
    team_features = []
    
    for (team, year), group in player_stats.groupby(['tmID', 'year']):
        features = {
            'tmID': team,
            'year': year,
            
            # Estatísticas de performance geral
            'perf_mean': group['performance'].mean(),
            'perf_median': group['performance'].median(),
            'perf_std': group['performance'].std(),
            'perf_max': group['performance'].max(),
            'perf_min': group['performance'].min(),
            'perf_sum': group['performance'].sum(),
            'perf_q25': group['performance'].quantile(0.25),
            'perf_q75': group['performance'].quantile(0.75),
            
            # Contagens
            'n_players': len(group),
            'n_rookies': group['rookie'].sum(),
            'rookie_ratio': group['rookie'].mean(),
            
            # Performance dos rookies vs não-rookies
            'rookies_perf_mean': group[group['rookie']]['performance'].mean() if group['rookie'].any() else 0,
            'non_rookies_perf_mean': group[~group['rookie']]['performance'].mean() if (~group['rookie']).any() else 0,
            
            # Performance ponderada por minutos (se disponível)
            'perf_weighted_by_minutes': (
                (group['performance'] * group['mp']).sum() / group['mp'].sum() 
                if 'mp' in group.columns and group['mp'].sum() > 0 
                else group['performance'].mean()
            ),
        }
        
        # Estatísticas adicionais de stats brutos se disponíveis
        for stat in ['pts', 'trb', 'ast', 'stl', 'blk', 'tov', 'mp']:
            if stat in group.columns:
                features[f'{stat}_total'] = group[stat].sum()
                features[f'{stat}_mean'] = group[stat].mean()
        
        team_features.append(features)
    
    team_df = pd.DataFrame(team_features)
    
    # Preencher NaNs (pode acontecer com std quando há um único jogador)
    team_df = team_df.fillna(0)
    
    return team_df


def create_team_rankings(teams_data: pd.DataFrame) -> pd.DataFrame:
    """
    Cria o ranking das equipes baseado em wins/losses.
    
    O ranking é calculado por temporada, onde 1 = melhor equipe, 2 = segunda melhor, etc.
    Usa win percentage como critério principal.
    
    Args:
        teams_data: DataFrame com dados das equipes (deve conter 'year', 'tmID', 'won', 'lost')
        
    Returns:
        DataFrame com coluna 'rank' adicionada
    """
    
    df = teams_data.copy()
    
    # Calcular win percentage se não existir
    if 'season_win_pct' not in df.columns:
        if 'won' in df.columns and 'lost' in df.columns:
            df['season_win_pct'] = df['won'] / (df['won'] + df['lost'])
        elif 'won' in df.columns and 'GP' in df.columns:
            df['season_win_pct'] = df['won'] / df['GP']
        else:
            raise ValueError("Não foi possível calcular win percentage. Colunas 'won' e 'lost' ou 'GP' necessárias.")
    
    # Criar ranking por temporada (1 = melhor)
    df['rank'] = df.groupby('year')['season_win_pct'].rank(ascending=False, method='min')
    
    return df


def prepare_train_test_data(
    player_stats: pd.DataFrame,
    teams_data: pd.DataFrame,
    test_season: int = 10,
    seasons_back: int = 3,
    decay: float = 0.7
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepara dados de treino e teste, garantindo tratamento adequado de rookies.
    
    Args:
        player_stats: DataFrame com estatísticas dos jogadores
        teams_data: DataFrame com dados das equipes
        test_season: Temporada a ser usada como teste (default: 10)
        seasons_back: Número de temporadas anteriores a considerar para performance
        decay: Fator de decaimento temporal
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    
    print(f"🔧 PREPARANDO DADOS DE TREINO/TESTE...")
    print(f"  📅 Divisão temporal: Temporadas 1-{test_season-1} = TREINO | Temporada {test_season} = TESTE")
    print(f"  ⏰ Isso evita vazamento de dados: o modelo não 'vê' o futuro durante o treino")
    
    # 1. Calcular performance dos jogadores
    print("\nCalculando performance dos jogadores...")
    player_perf = calculate_player_performance(
        player_stats,
        seasons_back=seasons_back,
        decay=decay,
        weight_by_minutes=True,
        rookie_seasons_back=3,
        rookie_fillna='global_mean'
    )
    
    print(f"  - Total de jogadores processados: {len(player_perf)}")
    print(f"  - Rookies identificados: {player_perf['rookie'].sum()}")
    
    # Verificar rookies na temporada de teste
    test_rookies = player_perf[(player_perf['year'] == test_season) & (player_perf['rookie'])]
    print(f"  - Rookies na temporada de teste (season {test_season}): {len(test_rookies)}")
    if len(test_rookies) > 0:
        print(f"    * Performance média dos rookies de teste: {test_rookies['performance'].mean():.3f}")
    
    # 2. Agregar performance por equipe
    print("\nAgregando performance por equipe...")
    team_perf = aggregate_team_performance(player_perf)
    print(f"  - Total de equipes/temporadas: {len(team_perf)}")
    
    # 3. Adicionar dados das equipes e criar rankings
    print("\nCriando rankings das equipes...")
    teams_ranked = create_team_rankings(teams_data)
    
    # 4. Merge team performance com team rankings
    full_data = team_perf.merge(
        teams_ranked[['year', 'tmID', 'rank', 'season_win_pct', 'won', 'lost']],
        on=['year', 'tmID'],
        how='left'
    )
    
    print(f"  - Dados após merge: {len(full_data)} linhas")
    
    # Verificar se temos dados faltantes no target
    missing_rank = full_data['rank'].isna().sum()
    if missing_rank > 0:
        print(f"  - AVISO: {missing_rank} equipes sem ranking (removidas)")
        full_data = full_data.dropna(subset=['rank'])
    
    # 5. Separar treino e teste
    train_data = full_data[full_data['year'] < test_season].copy()
    test_data = full_data[full_data['year'] == test_season].copy()
    
    print(f"\nDivisão treino/teste:")
    print(f"  - Treino: {len(train_data)} equipes/temporadas")
    print(f"  - Teste: {len(test_data)} equipes/temporadas")
    
    # 6. Selecionar features (versão melhorada - menos features, mais relevantes)
    exclude_cols = ['tmID', 'year', 'rank', 'season_win_pct', 'won', 'lost']
    
    # Features mais importantes baseadas na análise anterior
    core_features = [
        'perf_mean', 'perf_std', 'perf_max', 'perf_weighted_by_minutes',
        'n_players', 'rookie_ratio', 'rookies_perf_mean', 'non_rookies_perf_mean',
        'pts_mean', 'trb_mean', 'ast_mean', 'stl_mean', 'blk_mean', 'tov_mean'
    ]
    
    # Adicionar features históricas se disponíveis
    historical_features = []
    if 'prev_season_win_pct_1' in full_data.columns:
        historical_features.extend(['prev_season_win_pct_1', 'prev_season_win_pct_3'])
    
    feature_cols = core_features + historical_features
    feature_cols = [col for col in feature_cols if col in full_data.columns]
    
    print(f"\nFeatures selecionadas ({len(feature_cols)} - reduzido para evitar overfitting):")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {col}")
    
    X_train = train_data[feature_cols]
    X_test = test_data[feature_cols]
    y_train = train_data['rank']
    y_test = test_data['rank']
    
    return X_train, X_test, y_train, y_test, test_data


def cross_validate_model(model, X_train, y_train, cv=5):
    """Validação cruzada para detectar overfitting"""
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_absolute_error')
    return -scores.mean(), scores.std()


def create_ensemble_model():
    """Cria um modelo ensemble combinando múltiplos algoritmos"""
    
    # Modelos base com regularização
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    gb = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42
    )
    
    ridge = Ridge(alpha=1.0, random_state=42)
    
    # Ensemble com pesos baseados na performance esperada
    ensemble = VotingRegressor([
        ('rf', rf),
        ('gb', gb), 
        ('ridge', ridge)
    ], weights=[0.4, 0.4, 0.2])  # RF e GB têm mais peso
    
    return ensemble


def train_and_evaluate_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    test_data: pd.DataFrame
) -> Dict:
    """
    Treina e avalia modelo de ranking.
    
    Args:
        X_train, X_test, y_train, y_test: Dados de treino e teste
        test_data: DataFrame completo de teste (para análise detalhada)
        
    Returns:
        Dicionário com métricas e modelo treinado
    """
    
    print("\n🤖 TREINAMENTO E AVALIAÇÃO DO MODELO")
    print("="*80)
    print("🎯 Testando 4 algoritmos diferentes para prever rankings das equipes")
    print("📊 Métricas: MAE (erro médio), RMSE (erro quadrático médio), R² (qualidade do ajuste)")
    print("✅ MAE baixo = predições precisas | R² alto = bom ajuste aos dados")
    print()
    
    # Normalizar features
    print("\nNormalizando features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Treinar múltiplos modelos incluindo ensemble
    models = {
        'Random Forest (Regularizado)': RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting (Regularizado)': GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42
        ),
        'Ridge Regression': Ridge(alpha=1.0, random_state=42),
        'Ensemble (RF + GB + Ridge)': create_ensemble_model()
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Modelo: {model_name}")
        print('='*60)
        
        # Treinar
        print("Treinando...")
        model.fit(X_train_scaled, y_train)
        
        # Validação cruzada para detectar overfitting
        cv_mae, cv_std = cross_validate_model(model, X_train_scaled, y_train)
        print(f"  Validação Cruzada MAE: {cv_mae:.3f} (+/- {cv_std:.3f})")
        
        # Predições
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Métricas de treino
        train_mae = mean_absolute_error(y_train, y_pred_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        train_r2 = r2_score(y_train, y_pred_train)
        
        # Métricas de teste
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_r2 = r2_score(y_test, y_pred_test)
        
        print(f"\n📈 MÉTRICAS DE TREINO (como o modelo performou nos dados de treino):")
        print(f"  • MAE:  {train_mae:.3f} (erro médio de {train_mae:.1f} posições no ranking)")
        print(f"  • RMSE: {train_rmse:.3f} (penaliza erros grandes)")
        print(f"  • R²:   {train_r2:.3f} (1.0 = ajuste perfeito, 0.0 = baseline)")
        
        print(f"\n🧪 MÉTRICAS DE TESTE (performance REAL do modelo - o que importa!):")
        print(f"  • MAE:  {test_mae:.3f} (erro médio de {test_mae:.1f} posições no ranking)")
        print(f"  • RMSE: {test_rmse:.3f} (penaliza erros grandes)")
        print(f"  • R²:   {test_r2:.3f} (1.0 = ajuste perfeito, 0.0 = baseline)")
        
        # Análise de overfitting
        overfitting_ratio = train_mae / max(test_mae, 0.1)  # Evitar divisão por zero
        print(f"\n🔍 ANÁLISE DE OVERFITTING (modelo memorizou dados ao invés de aprender?):")
        print(f"  • Razão Treino/Teste: {overfitting_ratio:.2f}")
        if overfitting_ratio > 2.0:
            print("  ⚠️  POSSÍVEL OVERFITTING (modelo muito ajustado ao treino)")
        elif overfitting_ratio < 1.2:
            print("  ✅ BOM EQUILÍBRIO entre treino e teste")
        else:
            print("  ⚠️  OVERFITTING MODERADO")
        
        # Feature importance (para modelos que suportam)
        if hasattr(model, 'feature_importances_'):
            importances = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop 10 Features Mais Importantes:")
            for idx, row in importances.head(10).iterrows():
                print(f"  {row['feature']:30s} {row['importance']:.4f}")
        
        # Resultados detalhados por equipe
        test_results = test_data[['year', 'tmID', 'rank']].copy()
        test_results['predicted_rank'] = y_pred_test
        test_results['error'] = np.abs(test_results['rank'] - test_results['predicted_rank'])
        test_results = test_results.sort_values('rank')
        
        print(f"\nPredições Detalhadas (Temporada {test_data['year'].iloc[0]}):")
        print("-" * 80)
        print(f"{'Equipe':<10} {'Rank Real':>10} {'Rank Pred':>10} {'Erro':>10}")
        print("-" * 80)
        for _, row in test_results.iterrows():
            print(f"{row['tmID']:<10} {row['rank']:>10.0f} {row['predicted_rank']:>10.1f} {row['error']:>10.1f}")
        
        results[model_name] = {
            'model': model,
            'scaler': scaler,
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'train_r2': train_r2,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'predictions': test_results,
            'feature_importance': importances if hasattr(model, 'feature_importances_') else None
        }
    
    # Selecionar melhor modelo baseado em test MAE
    best_model_name = min(results.keys(), key=lambda k: results[k]['test_mae'])
    print(f"\n{'='*80}")
    print(f"MELHOR MODELO: {best_model_name}")
    print(f"  Test MAE: {results[best_model_name]['test_mae']:.3f}")
    print("="*80)
    
    return results

