import pandas as pd
from datetime import datetime

# Carregar o dataset de jogadores
df_players = pd.read_csv('../data/players.csv')

# 1. Remover linhas com altura ou peso inválidos (0.0)
df_players = df_players[(df_players['height'] != 0.0) & (df_players['weight'] != 0.0)]

# 2. Tratar posições vazias
df_players['pos'] = df_players['pos'].replace('', 'Unknown')

# 3. Remover colunas irrelevantes
df_players = df_players.drop(columns=['deathDate', 'collegeOther'])

# 4. Validar e tratar birthDate
df_players['birthDate'] = pd.to_datetime(df_players['birthDate'], errors='coerce')  # Converte inválidos para NaT
df_players = df_players.dropna(subset=['birthDate'])  # Remove linhas com birthDate inválido

# Opcional: Calcular idade aproximada (usando ano médio da temporada, assumindo dados históricos ~2006)
# Como o dataset é histórico, usar uma data de referência passada para idades realistas
reference_date = datetime(2006, 9, 23)  # Ajuste se necessário baseado no dataset
df_players['age'] = (reference_date - df_players['birthDate']).dt.days // 365  # Idade em anos

# 5. Verificar duplicatas por bioID (remover se houver)
df_players = df_players.drop_duplicates(subset=['bioID'])

# Salvar dataset limpo de jogadores
df_players.to_csv('../data_cleaning_output/players_cleaned.csv', index=False)

# Carregar o dataset de times
df_teams = pd.read_csv('../data/teams.csv')

# 1. Remover colunas irrelevantes
df_teams = df_teams.drop(columns=['playoff', 'seeded', 'firstRound', 'semis', 'finals', 'min', 'attend', 'arena'])

# 2. Verificar consistência: GP deve ser igual a won + lost
df_teams = df_teams[df_teams['GP'] == df_teams['won'] + df_teams['lost']]

# 3. Remover linhas com jogos inválidos (GP=0)
df_teams = df_teams[df_teams['GP'] > 0]

# 4. Remover colunas onde todos os valores são zero (dados inválidos)
stats_columns = ['tmORB','tmDRB','tmTRB','opptmORB','opptmDRB','opptmTRB']
cols_to_drop = [col for col in stats_columns if (df_teams[col] == 0).all()]
df_teams = df_teams.drop(columns=cols_to_drop)

df_teams = df_teams.drop(columns=['divID'])

# 4. Verificar duplicatas por tmID e year (remover se houver)
df_teams = df_teams.drop_duplicates(subset=['tmID', 'year'])

# Salvar dataset limpo de times
df_teams.to_csv('../data_cleaning_output/teams_cleaned.csv', index=False)