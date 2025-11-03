# Technical Decisions: Player Performance Preflight

Este documento justifica as principais decisões técnicas tomadas na refatoração.

## 1. Estrutura Modular

### Decisão
Dividir o monólito de 987 linhas em 7 módulos especializados.

### Rationale
- **Manutenibilidade**: Cada módulo <200 linhas, responsabilidade única
- **Testabilidade**: Funções isoladas são mais fáceis de testar
- **Reutilização**: Outros scripts podem importar funções específicas
- **Legibilidade**: Intenção clara pela organização

### Trade-offs
- **+**: Código organizado, fácil de modificar
- **-**: Mais ficheiros para navegar (mitigado por README e QUICK_REFERENCE)

---

## 2. Parâmetro `decay`: 0.4 vs 0.6-0.65

### Contexto
Walk-forward validation encontra **R²=0.508 com decay=0.4**, mas o original recomendava 0.6-0.7.

### Análise
```
decay  | R²    | Interpretação
-------|-------|-------------
0.40   | 0.508 | Season t-1: 40% de t, t-2: 16% de t
0.60   | 0.507 | Season t-1: 60% de t, t-2: 36% de t
0.70   | 0.505 | Season t-1: 70% de t, t-2: 49% de t
```

Diferenças de R²: **<0.01** (praticamente irrelevantes).

### Decisão Final
**Recomendar 0.6-0.65** com disclaimer transparente:

```python
decay = 0.6-0.65 (max R²=0.508 at decay=0.40, but differences <0.01; 
                  prefer higher decay for interpretability)
```

### Rationale
1. **Interpretabilidade**: decay=0.6 é mais intuitivo ("cada ano perde 40% de peso")
2. **Conservadorismo**: Penalizar épocas antigas faz sentido (skill degradation, rule changes)
3. **Transparência**: Mostramos o máximo real e deixamos decisão informada ao utilizador

### Recomendação
- **Produção padrão**: usar 0.65
- **Otimização máxima**: testar 0.4 (ganho marginal)
- **Muito conservador**: usar 0.7

---

## 3. IPW Weights: Clamping vs Truncation

### Contexto
Weights explode para anos avançados:
```
k=6: weight=4.28
k=7: weight=4.57
k=8: weight=5.23
k=9: weight=9.04
```

### Opções Consideradas

#### A. Usar IPW sem modificações
- **Prós**: Teoricamente correto (unbiased se modelo está certo)
- **Contras**: 1 observação com weight=9 = 9 observações normais; sensível a outliers

#### B. Clamping (w ← min(w, 4.0))
- **Prós**: Limita influência extrema, mantém todas observações
- **Contras**: Introduz viés (mas controlado)

#### C. Truncation (remover k>5)
- **Prós**: Remove problema na raiz
- **Contras**: Perde informação de veteranos

#### D. Não usar IPW
- **Prós**: Simples, robusto
- **Contras**: Viés de seleção não corrigido

### Decisão
**Gerar warning, recomendar clamping ou truncation**. Não aplicar automaticamente.

### Rationale
1. **Transparência**: Utilizador vê o problema e escolhe solução
2. **Flexibilidade**: Diferentes modelos podem precisar de diferentes abordagens
3. **Educativo**: Warning explica o problema e consequências

### Implementação Sugerida (para modelo final)
```python
# Option 1: Clamping
ipw_clamped = np.minimum(ipw_weights, 4.0)
model.fit(X, y, sample_weight=ipw_clamped)

# Option 2: Truncation
df_truncated = df[df['years_from_rookie'] <= 5]

# Option 3: Robust regression
from sklearn.linear_model import HuberRegressor
model = HuberRegressor()  # Downweights outliers automatically
```

---

## 4. `rookie_min_minutes`: 400 vs outros

### Contexto
Candidatos testados: 150, 300, 400, 600

### Resultados
```
Threshold | RMSE  | N rookies | Trade-off
----------|-------|-----------|----------
150       | 3.19  | ~200      | Muitos rookies, ruidoso
300       | 2.85  | ~150      | Balanço razoável
400       | 2.67  | ~100      | Melhor RMSE ✓
600       | 2.71  | ~60       | Poucos rookies, overfitting?
```

### Decisão
**400 minutos** (minimiza RMSE).

### Rationale
1. **Dados**: RMSE mais baixo de forma consistente
2. **Contexto WNBA**: ~34 jogos/época × 20 min/jogo = 680 min para starter
   - 400 min ≈ 60% de uma época completa
   - Razoável para identificar "rookies com amostra suficiente"
3. **Validação cruzada**: Strata analysis confirma <400 min tem corr<0.6

### Caveats
- Para outras ligas (NBA: 82 jogos), este threshold seria diferente
- Podia ser parametrizado por % da época média

---

## 5. `rookie_prior_strength`: 900 vs outros

### Contexto
Grid search: 900, 1800, 3600, 7200

### Resultados
```
Strength | Corr  | RMSE  | Interpretação
---------|-------|-------|---------------
900      | 0.652 | 3.77  | 900 min prior ✓
1800     | 0.638 | 4.06  | 1800 min prior
3600     | 0.612 | 4.32  | 3600 min prior
7200     | 0.543 | 4.50  | 7200 min prior
```

### Decisão
**900 minutos** (melhor corr + RMSE).

### Interpretação
Rookie com X minutos jogados:
```
X=300:  peso_obs=8.3,  peso_prior=25  → 75% prior
X=900:  peso_obs=25,   peso_prior=25  → 50% prior
X=1800: peso_obs=50,   peso_prior=25  → 33% prior
```

### Rationale
1. **Dados**: Claramente melhor que alternativas
2. **Intuição**: 900 min ≈ 1.3 épocas completas na WNBA
   - Faz sentido precisar de ~1 época para "confiar" no rookie
3. **Bayesiano**: Prior suficientemente forte para estabilizar, mas não dominar

---

## 6. Funções Compartilhadas em `utils/`

### Decisão
`compute_per36`, `label_rookies`, `aggregate_stints`, `per36_next_year` → `src/utils/players.py`

### Rationale
1. **DRY**: Usadas em preflight E em `player_performance.py`
2. **Consistência**: Mesma lógica em todos os scripts
3. **Testabilidade**: Um lugar para testar, funciona em todos

### Alternativa Não Escolhida
Manter duplicadas em cada script → risk de divergência.

---

## 7. Remover Código de Per-100 Pace-Adjusted

### Decisão
Remover `estimate_team_possessions_per_game`, `per36_vs_per100_check`.

### Rationale
1. **Dados insuficientes**: Comentado no main como "Skip... until better data"
2. **Complexidade**: Dean Oliver formula precisa de offensive/defensive stats detalhadas
3. **Manutenibilidade**: Código não usado confunde

### Como Recuperar
Se no futuro tiveres dados de possessões:
1. Recuperar do git history: `pre_performance_audit.py.old`
2. Criar módulo `experimental/pace_adjusted.py`
3. Integrar quando validado

---

## 8. Reports: `performance_pre` → `player_performance_preflight`

### Decisão
Renomear diretório de reports.

### Rationale
1. **Clareza**: "preflight" comunica melhor a intenção
2. **Consistência**: Alinha com nome do package `player_preflight`
3. **Futuro**: Se criares `coach_preflight`, `team_preflight`, estrutura fica clara

### Migração
Smooth: renomeado com `mv`, paths atualizados no código.

---

## 9. Dynamic Report Generation

### Decisão
Gerar `preflight_report.md` com valores reais dos cálculos, não hardcoded.

### Antes (Problemático)
```python
# Hardcoded no código
"- decay ∈ [0.6, 0.7] (choose 0.65)"  # Não bate com k_decay_best.txt
```

### Depois (Correto)
```python
# Lê de ficheiros gerados
decay_best = read_from_k_decay_best()
decay_recommendation = f"{decay_best:.2f}" + rationale
```

### Rationale
1. **Consistência**: Relatório sempre alinhado com cálculos
2. **Reprodutibilidade**: Se dados mudarem, relatório atualiza automaticamente
3. **Auditoria**: Rastreável de onde vêm os valores

---

## 10. Manter `pre_performance_audit.py.old`

### Decisão
Não apagar, mover para `.old`.

### Rationale
1. **Segurança**: Backup se algo der errado
2. **Referência**: Comparar implementações se surgir dúvida
3. **Git**: Preserva história sem poluir working tree

### Quando Apagar
Após 2-3 runs bem-sucedidos do novo sistema e confirmação que outputs batem certo.

---

## Resumo: Princípios Seguidos

1. ✅ **Transparência**: Decisões ambíguas são explicitadas no output
2. ✅ **Reproducibilidade**: Relatórios são gerados, não escritos à mão
3. ✅ **Manutenibilidade**: Código modular, <200 linhas/módulo
4. ✅ **Robustez**: Warnings para problemas (IPW, zeros, etc.)
5. ✅ **Documentação**: README, migration guide, quick reference, este doc
6. ✅ **Testabilidade**: Funções puras, importáveis independentemente
7. ✅ **Pragmatismo**: Quando diferenças são <0.01, priorizar interpretabilidade

---

**Para Discussão Futura:**

1. **Devemos usar decay=0.4 ou 0.65?**
   - Minha recomendação: 0.65 (diferença é marginal, mais interpretável)
   - Teste A/B: run modelo com ambos, comparar resultados finais

2. **IPW: clamping a 4.0 ou truncar em k=5?**
   - Minha recomendação: Clamping (mais dados > menos dados)
   - Teste: rodar com e sem, ver se muda conclusões

3. **Composite per36 scoring-driven (corr=0.73 com pontos)?**
   - Considerar: aumentar peso de assists, steals, blocks
   - Ou: criar múltiplos composites (offensive, defensive, playmaking)

4. **Remover linhas com stats=0?**
   - Minha recomendação: Sim (3 linhas só causam ruído)
   - Trivial: `df = df[df[['minutes','points']].sum(axis=1) > 0]`

---

**Autor:** Code review + refactoring  
**Data:** 2025-11-03

