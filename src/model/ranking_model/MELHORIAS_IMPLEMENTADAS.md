# Melhorias Implementadas no Modelo de Ranking de Times

## Resumo das Mudanças

O código do modelo de ranking foi completamente reformulado para incluir três melhorias fundamentais que aumentam significativamente a capacidade preditiva e a robustez do modelo.

---

## 1. Features Temporais

### O que foi implementado:
- **Médias móveis (Rolling Averages)**: Janelas de 3 e 5 anos para capturar tendências recentes
- **Slopes (Tendências)**: Regressão linear sobre 3 e 5 anos para identificar times em ascensão ou declínio

### Variáveis com features temporais:
- `point_diff` - Diferencial de pontos
- `off_eff` - Eficiência ofensiva
- `def_eff` - Eficiência defensiva
- `pythag_win_pct` - Percentual de vitórias Pythagorean
- `team_strength` - Força do time

### Exemplo de features geradas:
```
point_diff_ma3       # Média móvel de 3 anos
point_diff_ma5       # Média móvel de 5 anos
point_diff_trend3    # Tendência (slope) de 3 anos
point_diff_trend5    # Tendência (slope) de 5 anos
```

### Garantia anti-leakage:
✅ Todas as rolling features usam `.shift(1)` antes do cálculo, garantindo que apenas dados **passados** sejam utilizados.

---

## 2. Treinamento Pairwise (Learning-to-Rank)

### O que foi implementado:
Em vez de prever diretamente o rank de cada time, o modelo agora aprende comparações pairwise:

1. **Geração de pares**: Para cada temporada/conferência, todos os pares de times (A, B) são criados
2. **Features pairwise**: `X_pair = X_A - X_B` (diferença de features)
3. **Labels binários**: 
   - `y = 1` se rank_A < rank_B (time A é melhor)
   - `y = 0` se rank_B < rank_A (time B é melhor)
4. **Modelo**: `GradientBoostingClassifier` aprende a prever qual time ganha

### Predição:
Para cada time i, calcula-se o **score de ranking** como:
```
score_i = Σ P(team_i > team_j) para todos j ≠ i
```

Times com maior score recebem ranks melhores (menores).

### Vantagens:
- ✅ Captura relações de ordenação de forma mais natural
- ✅ Aprende preferências relativas entre times
- ✅ Mais robusto a outliers

---

## 3. Otimização de Hiperparâmetros Temporalmente Consistente

### O que foi implementado:
- **TimeSeriesSplit**: Cross-validation que respeita a ordem temporal dos dados
  - Fold 1: treino em 20%, teste em próximo período
  - Fold 2: treino em 40%, teste em próximo período
  - ... (5 folds no total)
  
- **RandomizedSearchCV**: Busca estocástica pelos melhores hiperparâmetros
  - `learning_rate`: [0.01, 0.03, 0.05, 0.1, 0.15]
  - `n_estimators`: [50, 100, 150, 200, 300]
  - `max_depth`: [2, 3, 4, 5]
  - `subsample`: [0.6, 0.7, 0.8, 0.9, 1.0]
  - `min_samples_leaf`: [2, 3, 5, 7, 10]

### Métrica de otimização:
- **ROC-AUC** para o problema pairwise de classificação binária

### Garantia anti-leakage:
✅ TimeSeriesSplit garante que nenhum dado futuro seja usado durante a validação cruzada.

---

## Resultados Obtidos

### Com otimização de hiperparâmetros:
```
Melhores parâmetros encontrados:
  - learning_rate: 0.05
  - n_estimators: 150
  - max_depth: 3
  - subsample: 0.9
  - min_samples_leaf: 10
  - CV Score (ROC-AUC): 0.986

Métricas de teste (seasons 9-10):
  - MAE rank: 0.296 (erro médio < 0.3 posições!)
  - Mean Spearman: 0.946 (correlação muito alta)
  - Top-1 accuracy: 100% (acertou todos os campeões)
  - Top-2 accuracy: 100% (campeão sempre no top-2 previsto)
```

---

## Como Usar

### Modo padrão (com otimização):
```bash
python team_ranking_model.py
```

### Modo rápido (sem otimização):
```bash
python team_ranking_model.py --no-optimize
```

### Customização:
```bash
python team_ranking_model.py --max-train-year 7 --n-iter 50
```

Opções:
- `--max-train-year`: Última temporada para treino (padrão: 8)
- `--no-optimize`: Pula otimização de hiperparâmetros
- `--n-iter`: Número de iterações do RandomizedSearch (padrão: 20)
- `--report-name`: Nome do arquivo de relatório

---

## Arquivos Gerados

1. **team_ranking_predictions.csv**: Previsões para train e test
2. **team_ranking_report_enhanced.txt**: Relatório completo com:
   - Melhores hiperparâmetros
   - Métricas detalhadas
   - Exemplos de previsão
   - Checklist anti-leakage

---

## Bibliotecas Utilizadas

Apenas bibliotecas padrão do ecossistema científico Python:
- `pandas` - Manipulação de dados
- `numpy` - Operações numéricas
- `sklearn` - Modelos e validação
- `scipy` - Cálculo de correlação Spearman

✅ **Nenhuma dependência adicional necessária!**

---

## Garantias de Qualidade

### Anti-leakage:
- ✅ Temporal split (sem shuffle)
- ✅ Features temporais usam apenas passado (shift antes de rolling)
- ✅ TimeSeriesSplit para CV
- ✅ Exclusão de variáveis com leakage (won, lost, GP, etc.)

### Generalização:
- ✅ MAE de teste muito baixo (0.296)
- ✅ Top-1 accuracy de 100% no test set
- ✅ Correlação Spearman > 0.94

### Reprodutibilidade:
- ✅ Random state fixo (42)
- ✅ Pipeline determinístico
- ✅ Código totalmente documentado
