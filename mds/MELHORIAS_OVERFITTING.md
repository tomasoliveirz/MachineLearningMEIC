# Melhorias Implementadas no Team Ranking Model

## Data: 12 de Novembro de 2025

## Problema Identificado
Análise dos gráficos revelou **overfitting severo** no modelo de ranking de times:
- MAE treino (0.99) vs teste (1.56) → gap de 0.57
- Spearman treino (0.706) vs teste (0.248) → correlação fraca no teste
- Instabilidade temporal e por conferência
- Métricas extremas (MAE=0, Spearman=1.0) indicando casos degenerados
- Top-K accuracy rápida demais (100% em K=4-5) → métrica pouco informativa

## Soluções Implementadas

### 1. **Validação Temporal (Train-Val-Test Split)**
```python
def split_train_val_test(df_all, max_train_year=8, val_years=2)
```
- **Antes**: Train (anos 1-8) → Test (anos 9+)
- **Depois**: Train (anos 1-6) → Val (anos 7-8) → Test (anos 9+)
- **Benefício**: Permite monitorar overfitting durante treino e ajustar hiperparâmetros

### 2. **Regularização Forte do Modelo**
```python
def create_pairwise_model(
    n_estimators=300,        # ↓ de 500
    learning_rate=0.05,      # mantido
    max_depth=3,             # ↓ árvores mais rasas
    min_samples_split=20,    # ↑ mínimo para split
    min_samples_leaf=10,     # ↑ mínimo por folha
    subsample=0.8,           # 80% dos dados por árvore
    max_features='sqrt'      # √n features por split
)
```

**Mecanismos de Regularização:**
- **max_depth=3**: Árvores rasas previnem memorização de ruído
- **min_samples_split/leaf**: Evita nós com poucos exemplos (overfitting local)
- **subsample=0.8**: Gradient Boosting estocástico (reduz correlação entre árvores)
- **max_features='sqrt'**: Feature bagging (aumenta diversidade)

### 3. **Early Stopping**
```python
validation_fraction=0.1,    # 10% dos dados de treino para validação interna
n_iter_no_change=20,        # Para se não melhorar por 20 iterações
tol=1e-4                    # Tolerância mínima de melhoria
```
- Para o treinamento automaticamente quando o modelo começa a overfit
- Evita iterações desnecessárias

### 4. **Métrica NDCG (Normalized Discounted Cumulative Gain)**
```python
def calculate_ndcg_at_k(y_true, y_pred, k=10)
```
- **Problema com Top-K**: Binária (acertou ou errou), perde nuances
- **NDCG**: Penaliza erros de acordo com a posição (errar top-1 pesa mais)
- **Range**: 0.0 (pior) a 1.0 (perfeito)
- Métrica padrão em sistemas de ranking (e.g., search engines)

### 5. **Diagnóstico de Overfitting**
```python
train_test_gap = train_metrics['mae_rank'] - test_metrics['mae_rank']
if abs(train_test_gap) < 0.3:
    "✓ Good generalization"
elif abs(train_test_gap) < 0.6:
    "⚠ Moderate overfitting"
else:
    "✗ Severe overfitting"
```
- Reporta automaticamente a qualidade da generalização
- Thresholds baseados em experiência prática

### 6. **Relatório Enriquecido**
Agora inclui:
- Métricas separadas para Train / Val / Test
- NDCG@10 para cada split
- Diagnóstico de overfitting
- Gaps entre conjuntos

### 7. **Validação Walk‑Forward**
Além da validação temporal fixa, foi adicionada uma rotina de *walk‑forward validation*.

- O que foi implementado: `generate_walk_forward_splits(df_all, max_train_year, val_years)` gera múltiplos folds onde cada fold treina em todas as temporadas anteriores e valida em uma temporada específica (ex.: val years = 7..8 gera folds para val=7 e val=8).
- Por que: validação por folds temporais fornece estimativas de desempenho mais robustas em séries temporais e evita penalizar o treino final ao "reservar" as temporadas mais recentes.
- Comportamento final: os folds são usados para escolher/configurar o modelo; depois o modelo final é treinado em todas as temporadas até `max_train_year` (inclui os anos usados nas validações) antes de avaliar no teste (anos > `max_train_year`).

Benefício prático: preservamos o sinal das temporadas mais recentes no treino final enquanto ainda obtemos validação realista e temporalmente consistente.

## Resultados Esperados

### Antes (Modelo Original)
```
TRAIN   - MAE: 0.99 | Spearman: 0.706
TEST    - MAE: 1.56 | Spearman: 0.248
Gap: 0.57 → SEVERE OVERFITTING
```

### Depois (Modelo Regularizado) - Esperado
```
TRAIN   - MAE: 1.10-1.20 | Spearman: 0.60-0.70 | NDCG: 0.75-0.85
VAL     - MAE: 1.15-1.25 | Spearman: 0.55-0.65 | NDCG: 0.70-0.80
TEST    - MAE: 1.20-1.35 | Spearman: 0.50-0.60 | NDCG: 0.65-0.75
Gap: 0.10-0.25 → GOOD/MODERATE GENERALIZATION
```

**Trade-offs:**
- ✓ Menor overfitting (gap < 0.3)
- ✓ Desempenho mais consistente entre splits
- ✓ Correlação de Spearman no teste melhorada
- ⚠ Desempenho no treino ligeiramente pior (aceitável)
- ⚠ Pode não atingir correlação perfeita no treino (esperado)

## Próximos Passos (Se Ainda Houver Problemas)

### Se Val-Test gap ainda for alto:
1. **Feature Engineering**:
   - Remover features muito específicas de ano
   - Adicionar mais features estáveis (histórico de longo prazo)

2. **Balanceamento por Conferência**:
   - Stratified sampling por confID
   - Treinar modelos separados por conferência

3. **Ensemble Temporal**:
   - Treinar modelos separados por período
   - Combinar predições com peso temporal

4. **Calibração**:
   - Isotonic regression sobre scores finais
   - Ajusta distribuição de predições

### Se correlação de Spearman ainda for baixa:
1. **Learning-to-Rank Algorithms**:
   - LambdaMART (mais robusto que GradientBoosting)
   - RankNet ou ListNet

2. **Ordinal Regression**:
   - Modelos específicos para variáveis ordinais
   - Mango (ordinal neural networks)

3. **Feature Selection**:
   - Remover features com baixa importância
   - Análise de multicolinearidade

## Como Executar

```bash
cd "c:\Users\gluca\OneDrive\Ambiente de Trabalho\Master FEUP\MachineLearningMEIC"
python src/model/ranking_model/team_ranking_model.py
```

## Arquivos Gerados

1. **data/processed/team_ranking_predictions.csv**
   - Predições para train, val e test
   - Coluna `split` identifica cada conjunto

2. **reports/models/team_ranking_report.txt**
   - Métricas completas com diagnóstico
   - NDCG@10 e Top-K accuracy

3. **reports/models/graphics/**
   - Gráficos atualizados com conjunto de validação
   - Heatmaps temporais
   - Scatter plots pred vs actual

## Referências

- **Gradient Boosting Regularization**: Friedman (2001) - "Greedy Function Approximation: A Gradient Boosting Machine"
- **NDCG**: Järvelin & Kekäläinen (2002) - "Cumulated gain-based evaluation of IR techniques"
- **Learning-to-Rank**: Liu (2011) - "Learning to Rank for Information Retrieval"
- **Temporal Validation**: Bergmeir & Benítez (2012) - "On the use of cross-validation for time series predictor evaluation"
