.SILENT:

# Detecta Windows vs POSIX
ifeq ($(OS),Windows_NT)
PYTHON := venv/Scripts/python.exe
PIP := venv/Scripts/pip.exe
RM := rmdir /S /Q
SEP := \
else
PYTHON := venv/bin/python3
PIP := venv/bin/pip
RM := rm -rf
SEP := /
endif

# Alvos principais
.PHONY: help venv clean_players clean_teams team_season rookies analyze_raw analyze_cleaned reports_tree clean_data all

# Ajuda rápida
help:
    @echo "Targets disponíveis:"
    @echo "  make venv            -> cria ambiente virtual e instala dependências"
    @echo "  make clean_players   -> executa src/ac/cleaning/clean_players.py"
    @echo "  make clean_teams     -> executa src/ac/cleaning/clean_teams.py"
    @echo "  make team_season     -> agrega features de equipa (depende de clean_teams)"
    @echo "  make rookies         -> extrai features de rookies (depende de clean_players, clean_teams)"
    @echo "  make analyze_raw     -> gera análises usando dados raw"
    @echo "  make analyze_cleaned -> gera análises usando dados limpos"
    @echo "  make reports_tree    -> mostra estrutura de reports"
    @echo "  make clean_data      -> remove outputs gerados (data/processed e reports)"
    @echo "  make all             -> executa pipeline completo (venv + pipeline)"

# Cria/atualiza ambiente virtual e instala requirements
venv:
    @echo "Criando ambiente virtual (venv) e instalando dependências..."
    python -m venv venv
    $(PIP) install --upgrade pip || true
    $(PIP) install -r requirements.txt

# Limpeza dos jogadores
clean_players: venv
    @echo "Rodando limpeza de jogadores..."
    $(PYTHON) src/ac/cleaning/clean_players.py

# Limpeza das equipas
clean_teams: venv
    @echo "Rodando limpeza de equipas..."
    $(PYTHON) src/ac/cleaning/clean_teams.py

# Agregação por época / equipa
team_season: clean_teams
    @echo "Agregando features de época por equipa..."
    $(PYTHON) src/ac/features/aggregate_team_season.py

# Features de rookies
rookies: clean_players clean_teams
    @echo "Gerando features de rookies..."
    $(PYTHON) src/ac/features/rookies.py

# Análise com dados raw
analyze_raw: venv clean_players clean_teams
    @echo "Gerando análises (raw)..."
    $(PYTHON) src/ac/analysis/analyzer.py --mode raw

# Análise com dados limpos
analyze_cleaned: venv clean_players clean_teams
    @echo "Gerando análises (cleaned)..."
    $(PYTHON) src/ac/analysis/analyzer.py --mode cleaned

# Mostra a árvore de reports (usa tree se disponível)
reports_tree:
    @if command -v tree >/dev/null 2>&1; then \
        tree reports || true; \
    else \
        echo "tree não encontrado; listando com find/dir:"; \
        if [ -d reports ]; then find reports -type d -print -o -type f -print | sed "s,^,reports/,"; else echo "reports/ não existe"; fi; \
    fi

# Remove outputs gerados (atenção: apaga pastas processed e reports)
clean_data:
    @echo "Removendo outputs em data/processed e reports/..."
ifeq ($(OS),Windows_NT)
    if exist data\processed $(RM) data\processed || true
    if exist reports $(RM) reports || true
else
    $(RM) data/processed reports || true
endif

# Pipeline completo: cria venv (se necessário) e executa etapas principais
all: venv clean_players clean_teams team_season rookies analyze_cleaned
    @echo "Pipeline completo executado."