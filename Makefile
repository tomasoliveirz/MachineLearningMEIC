.PHONY: clean_players clean_teams team_season rookies analyze_raw analyze_cleaned reports_tree all
VENV_ACT=. venv/bin/activate;

clean_players:
	$(VENV_ACT) python3 src/ac/cleaning/clean_players.py

clean_teams:
	$(VENV_ACT) python3 src/ac/cleaning/clean_teams.py

team_season: clean_teams
	$(VENV_ACT) python3 src/ac/features/aggregate_team_season.py

rookies: clean_players clean_teams
	$(VENV_ACT) python3 src/ac/features/rookies.py

analyze_raw: clean_players clean_teams
	$(VENV_ACT) python3 src/ac/analysis/analyzer.py --mode raw

analyze_cleaned: clean_players clean_teams
	$(VENV_ACT) python3 src/ac/analysis/analyzer.py --mode cleaned

reports_tree:
	@if command -v tree >/dev/null 2>&1; then \
		tree reports; \
	else \
		echo "tree not found; using find:"; \
		find reports -type d -print -o -type f -print | sed "s,^,reports/,"; \
	fi

all: clean_players clean_teams team_season rookies analyze_cleaned
