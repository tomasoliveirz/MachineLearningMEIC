Individual Awards: Forecast the winners of each individual award.

Que ano que as awards foram ganhas? E como os jogadores ganharam, o que fizeram de especial?

## Limpeza do Dataset (clean_dataset.py)

Este arquivo realiza a limpeza inicial dos datasets `players.csv` e `teams.csv` para preparar os dados para análise e modelagem de previsões de prêmios individuais e rankings de temporada. As etapas incluem limpeza para jogadores e times.

### Limpeza do Dataset de Jogadores:
- **Remoção de linhas inválidas**: Elimina registros com altura ou peso iguais a 0.0, pois são impossíveis.
- **Tratamento de posições vazias**: Substitui posições vazias por "Unknown".
- **Remoção de colunas irrelevantes**: Remove `deathDate` (sempre "0000-00-00") e `collegeOther` (campo opcional esparso).
- **Validação de datas**: Converte `birthDate` para datetime, remove linhas com datas inválidas e calcula idade aproximada baseada na data atual (23/09/2025).
- **Remoção de duplicatas**: Elimina registros duplicados com base em `bioID`.
- **Salvamento**: Gera `../output/players_cleaned.csv` com os dados limpos.

### Limpeza do Dataset de Times:
- **Remoção de colunas irrelevantes**: Remove `playoff`, `seeded`, `firstRound`, `semis`, `finals` (dependem do ano), `min` (minutos jogados), `attend` (público) e `arena` (não essenciais para previsões).
- **Verificação de consistência**: Remove linhas onde `GP` (jogos disputados) não é igual a `won + lost`.
- **Remoção de linhas inválidas**: Elimina registros com `GP = 0` (times sem jogos).
- **Remoção de duplicatas**: Elimina registros duplicados com base em `tmID` e `year`.
- **Salvamento**: Gera `../output/teams_cleaned.csv` com os dados limpos.


