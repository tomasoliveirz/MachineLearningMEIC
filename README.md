# AC — WNBA Sports Analytics Pipeline

Este repositório aloja uma solução *end-to-end* de Machine Learning para a análise de desempenho desportivo (WNBA). O sistema implementa um pipeline completo que abrange desde a ingestão e limpeza de dados até à modelação preditiva de rankings de equipas, utilizando técnicas avançadas de engenharia de *features* e algoritmos de *Learning-to-Rank*.

---

## Índice
1. [Visão Geral do Projeto](#visão-geral-do-projeto)
2. [Estrutura do Repositório](#estrutura-do-repositório)
3. [Metodologia e Decisões de Modelação](#metodologia-e-decisões-de-modelação)
4. [Implementação Técnica](#implementação-técnica)
5. [Instalação e Pré-requisitos](#instalação-e-pré-requisitos)
6. [Execução do Pipeline](#execução-do-pipeline)
7. [Resultados e Artefactos](#resultados-e-artefactos)
8. [Equipa](#equipa)

---

## Visão Geral do Projeto

O objetivo central deste projeto é prever a classificação final das equipas da WNBA para uma época alvo (e.g., Época 11), baseando-se em dados históricos das épocas anteriores. Para tal, foi desenvolvido um sistema que trata a variabilidade de minutos jogados (especialmente em *rookies*), ajusta métricas ao contexto da equipa e utiliza validação temporal rigorosa para evitar *data leakage*.

---

## Estrutura do Repositório

A organização do código fonte segue uma arquitetura modular para garantir a manutenibilidade e reprodutibilidade:

* **`data/`**: Armazenamento de dados.
    * `raw/`: Ficheiros CSV originais (`players.csv`, `teams.csv`, etc.).
    * `processed/`: Dados limpos e *features* geradas.
* **`src/`**: Código fonte.
    * `ac/cleaning/`: Scripts de limpeza e normalização de dados.
    * `ac/features/`: Engenharia de *features* (estatísticas de equipa, métricas de *rookies*).
    * `ac/analysis/`: Módulos de visualização e geração de relatórios.
    * `analysis/player_preflight/`: Calibração de hiperparâmetros para métricas de desempenho.
    * `player_performance.py`: Núcleo de cálculo de métricas ajustadas de jogadores.
    * `main.py`: Ponto de entrada para a orquestração do pipeline.
    * `utils/`: Funções auxiliares reutilizáveis.
* **`reports/`**: *Outputs* do sistema (gráficos, tabelas e relatórios de texto).
* **`Makefile`**: Automação de tarefas unitárias.

---

## Metodologia e Decisões de Modelação

Para garantir a robustez das métricas utilizadas no modelo preditivo, foram implementadas diversas técnicas estatísticas para tratamento de ruído e variância.

### 1. Estabilização de Métricas para Rookies (Shrinkage Bayesiano)
Métricas extrapoladas para 36 minutos (*per36*) tendem a apresentar alta variância em jogadores com poucos minutos jogados. Para mitigar este efeito, aplica-se uma abordagem Bayesiana ("Shrinkage") que pondera a observação do jogador com um *prior* (média histórica).

**Fórmula de Ajuste:**
$$\hat{\mu} = \frac{w_{obs} \cdot \mu_{obs} + w_{prior} \cdot \mu_{prior}}{w_{obs} + w_{prior}}$$

Onde:
* $w_{obs} = \frac{\text{minutos jogados}}{36}$
* $w_{prior} = \frac{\text{rookie\_prior\_strength}}{36}$ (Parâmetro calibrado)

*Exemplo:* Um *rookie* com métricas excecionais em apenas 18 minutos terá a sua média "puxada" em direção à média da liga/equipa, evitando projeções irreais.

### 2. Ponderação Histórica (Decaimento Exponencial)
O desempenho passado é relevante, mas épocas recentes devem ter maior influência. Aplica-se uma média ponderada temporal com fator de decaimento $\delta$ ($0 < \delta < 1$).

**Peso da época $k$ (anos atrás):**
$$w_k = \delta^{k} \times \text{minutos}_k$$

Isto assegura que a consistência é valorizada, mas o estado atual do jogador é predominante.

### 3. Ajuste ao Contexto da Equipa
Métricas brutas são influenciadas pelo ritmo (*pace*) e eficiência ofensiva da equipa. As métricas individuais são ajustadas multiplicativamente:

$$\text{Adj} = \text{Perf}_{raw} \times \frac{\text{Pontos}_{\text{Equipa}}}{\text{Mediana}(\text{Pontos}_{\text{Liga}})}$$

### 4. Normalização por Posição (Role Adjustment)
O valor de uma estatística depende da posição do jogador (e.g., assistências são mais críticas para *Guards*, ressaltos para *Centers*). O sistema aplica pesos específicos por posição para calcular um *score* de desempenho unificado.

---

## Implementação Técnica

O pipeline de Machine Learning consiste nas seguintes etapas sequenciais:

1.  **Pré-processamento**: Limpeza de dados, tratamento de valores nulos e normalização de identificadores (`tmID`, `playerID`).
2.  **Engenharia de Features**:
    * Criação de métricas agregadas por época.
    * Cálculo de médias móveis (3 e 5 anos) e tendências lineares (*slopes*).
    * Features de estabilidade técnica (*Coach tenure*).
3.  **Modelação (Learning to Rank)**:
    * **Algoritmo**: Gradient Boosting Classifier.
    * **Abordagem**: *Pairwise Ranking*. O modelo aprende a função $P(A \succ B)$ (probabilidade da Equipa A ser superior à Equipa B) com base no vetor de diferenças das suas *features*.
    * **Validação**: *Walk-Forward Validation* (Janela deslizante) para respeitar estritamente a ordem temporal e impedir o uso de dados futuros no treino.

---

## Instalação e Pré-requisitos

O projeto foi desenvolvido para ambiente Windows com Python 3.8+.

1.  **Clonar o repositório e aceder à diretoria raiz.**
2.  **Configurar o ambiente virtual:**

    ```powershell
    # Criação do ambiente virtual
    python -m venv venv

    # Ativação do ambiente (Windows)
    .\venv\Scripts\activate

    # Instalação das dependências
    pip install -r requirements.txt
    ```

    *Alternativamente, se o utilitário `make` estiver disponível, execute `make venv`.*

---

## Execução do Pipeline

A execução integral do projeto é orquestrada pelo script `main.py` localizado na diretoria `src`.

```powershell
cd src
python main.py