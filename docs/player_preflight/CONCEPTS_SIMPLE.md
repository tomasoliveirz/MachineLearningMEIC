# 🧠 Conceitos em Linguagem Simples

> Para quando precisares de explicar a alguém (ou a ti próprio às 3 da manhã antes da defesa)

---

## 🎯 Per-36 Metric

**O que é:**
Um "composite score" que resume a contribuição de um jogador, normalizado para 36 minutos.

**Fórmula:**
```
per36 = (pts + 0.7×reb + 0.7×ast + 1.2×stl + 1.2×blk - 0.7×tov) / minutos × 36
```

**Porquê normalizar?**
- Jogador A: 1000 min, 500 pts totais → 18 pts/36
- Jogador B: 100 min, 50 pts totais → 18 pts/36
- Conclusão: mesma **eficiência**, diferentes **volumes**

**Cuidado:**
- Per-36 com poucos minutos é instável (daí o floor de 12 min)

---

## 🎲 Bayesian Shrinkage (Rookie Prior)

**Problema:**
Rookies com poucos minutos → estatísticas muito ruidosas.

**Solução:**
Combinar o observado com um "prior" (baseline da liga).

**Analogia:**
Imagina que nunca provaste comida de um restaurante novo.
- Opção A: confiar 100% na única review que tem (pode ser fake)
- Opção B: combinar essa review com a média de restaurantes na cidade
→ **Opção B é Bayesian shrinkage!**

**No código:**
```python
prior = média_dos_rookies_todos
força_prior = 900 minutos equivalentes

se rookie jogou 300 minutos:
    peso_observado = 300
    peso_prior = 900
    per36_final = (300×observado + 900×prior) / 1200
                = 25% observado + 75% prior
```

**Resultado:**
- Rookies com poucos minutos → puxados para a média
- Rookies com muitos minutos → usam mais o seu próprio desempenho

---

## ⏰ Temporal Dependence (Decay)

**Problema:**
Queres prever o per36 do ano que vem. Usas média histórica, mas...
- O que o jogador fez há 5 anos é menos relevante que o que fez no ano passado.

**Solução:**
Dar **mais peso** a épocas recentes.

**Como:**
```
peso_época = decay^(anos_atrás) × minutos_jogados
```

**Exemplo visual:**
```
Jogador em 2024:

                     PESO
2024 (t):   [████████████] 1.00
2023 (t-1): [███████]      0.60
2022 (t-2): [████]         0.36
2021 (t-3): [██]           0.22

Decay = 0.6
```

**Interpretação:**
- Decay alto (0.8) → passado conta bastante
- Decay médio (0.6) → balanço razoável ✅
- Decay baixo (0.3) → só o recente importa

---

## 📊 Walk-Forward Validation

**O que é:**
Método de validação para séries temporais.

**Como funciona:**
```
Anos disponíveis: 2015, 2016, 2017, 2018, 2019, 2020

Fold 1:
  Treino: 2015-2017
  Teste:  2018
  
Fold 2:
  Treino: 2015-2018
  Teste:  2019
  
Fold 3:
  Treino: 2015-2019
  Teste:  2020

Métrica final: média dos 3 folds
```

**Porquê não usar K-Fold normal?**
- K-Fold mistura passado e futuro → data leakage!
- Walk-forward respeita a ordem temporal ✅

---

## 🎯 R² (R-squared)

**O que é:**
Métrica que diz "quanto da variação nos dados o modelo consegue explicar".

**Interpretação:**
- R² = 0.00 → modelo não explica nada (inútil)
- R² = 0.50 → modelo explica 50% da variação (ok)
- R² = 0.70 → modelo explica 70% (bom)
- R² = 0.90 → modelo explica 90% (muito bom, pode ser overfit)
- R² = 1.00 → modelo explica tudo (perfeito, quase sempre suspeito)

**No teu caso:**
```
R² = 0.49 (temporal dependence)
```
→ Consegues explicar ~49% da variação no per36 do ano seguinte usando o histórico.
Isso é **razoável** para desporto (há muito ruído: lesões, mudanças de equipa, etc.).

---

## 📉 RMSE (Root Mean Squared Error)

**O que é:**
Erro médio do modelo, na **mesma escala** que a variável que estás a prever.

**Fórmula:**
```
RMSE = sqrt(média dos erros ao quadrado)
```

**Exemplo:**
```
Previsões vs Real:
Jogador A: previsto=15, real=18 → erro=3
Jogador B: previsto=20, real=18 → erro=2
Jogador C: previsto=10, real=18 → erro=8

RMSE = sqrt((3² + 2² + 8²) / 3) = sqrt((9+4+64)/3) ≈ 5.1
```

**Interpretação:**
"Em média, o modelo erra por ~5.1 pontos de per36."

**No teu caso:**
```
RMSE = 3.27 (rookie threshold = 400 min)
```
→ Em média, erras por ~3.3 pontos ao prever o per36 do ano seguinte.
Isso é bom (considerando que per36 médio ~ 12-15).

---

## 🏀 Survival Bias (removido, mas importante saber)

**O que é:**
Viés causado por só veres quem "sobreviveu" na liga.

**Problema:**
```
Ano 1: entram 100 rookies
  - 80 jogam mal, saem da liga
  - 20 jogam bem, continuam

Ano 5: só tens dados dos 20 que ficaram
→ Médias do "ano 5" estão enviesadas para cima!
```

**Solução (não implementada):**
Inverse Probability Weighting (IPW):
- Descobres P(chegar ao ano 5) = 20%
- Dás peso = 1/0.2 = 5 a cada sobrevivente
- Assim "representas" os 80 que saíram

**Porque não usaste:**
- IPW pode dar pesos absurdos (até 9×)
- Alguns jogadores dominariam o modelo inteiro
- Complexidade vs benefício → deixámos para trabalho futuro

---

## 🔑 Leakage (Data Leakage)

**O que é:**
Usar informação do futuro para prever o futuro (batota acidental).

**Exemplo de leakage:**
```python
# ERRADO ❌
df['per36_avg_all_time'] = df.groupby('playerID')['per36'].transform('mean')
# isto usa dados do futuro!

# CERTO ✅
df['per36_avg_past'] = df.groupby('playerID')['per36'].shift(1).expanding().mean()
# só usa dados até t-1
```

**No teu código:**
- ✅ Usas `.shift(1)` para criar `per36_next`
- ✅ Só usas épocas passadas para prever futuro
- ✅ Walk-forward validation respeita ordem temporal

---

## 📊 Stratified Validation

**O que é:**
Validar o modelo **separadamente** em sub-grupos dos dados.

**Porquê:**
Modelo pode funcionar bem "em média" mas mal em casos específicos.

**Exemplo:**
```
Validação global:
  R² = 0.69 ✅ (parece bom)

Validação estratificada por minutos:
  <150 min:    R² = 0.29 ❌ (terrível)
  150-600 min: R² = 0.57 ⚠️  (ok)
  >600 min:    R² = 0.76 ✅ (ótimo)
```

→ Descobres que o modelo **não funciona** para jogadores com poucos minutos!
→ Justifica o threshold de 400 minutos.

---

## 🔧 Grid Search

**O que é:**
Testar vários valores de um parâmetro e escolher o melhor.

**Exemplo:**
```python
# Qual o melhor rookie_min_minutes?
candidatos = [150, 300, 400, 600]

for threshold in candidatos:
    df_filtered = df[df['minutes'] >= threshold]
    rmse = avaliar_modelo(df_filtered)
    print(f"{threshold}: RMSE={rmse}")

# Output:
# 150: RMSE=5.23
# 300: RMSE=3.50
# 400: RMSE=3.27  ← MELHOR ✅
# 600: RMSE=3.35

# Escolher: 400
```

---

## 📈 Autocorrelation

**O que é:**
Correlação de uma variável **consigo mesma** ao longo do tempo.

**Exemplo:**
```
per36 do ano t  vs  per36 do ano t+1
    15                  16
    20                  19
    10                  12
    ...
    
corr = 0.69 → alta autocorrelação
```

**Interpretação:**
- Alta autocorrelação (0.7+) → desempenho é bastante estável
- Baixa autocorrelação (0.3-) → desempenho é muito volátil

**No teu caso:**
```
autocorr(per36_t, per36_t+1) ≈ 0.69
```
→ Jogadores tendem a manter-se consistentes ano-a-ano
→ Justifica usar histórico para prever futuro

---

## 🎓 Resumo dos Resumos

| Conceito | Em 5 palavras |
|----------|---------------|
| **Per-36** | Eficiência normalizada por minutos |
| **Bayesian shrinkage** | Puxar outliers para média |
| **Decay** | Passado pesa menos progressivamente |
| **Walk-forward** | Validar respeitando ordem temporal |
| **R²** | Percentagem de variação explicada |
| **RMSE** | Erro médio em unidades originais |
| **Survival bias** | Só vês quem sobrevive |
| **Leakage** | Usar futuro para prever |
| **Grid search** | Testar tudo, escolher melhor |

---

**Dica final:**
Se te perguntarem algo que não sabes responder na defesa:

> "Essa é uma extensão interessante que não explorámos, mas está documentada
> no ficheiro TECHNICAL_DECISIONS.md como trabalho futuro."

😎👌

