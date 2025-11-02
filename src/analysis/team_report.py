import pandas as pd
from pathlib import Path

def generate_teams_report():
    """
    Analisa o ficheiro teams_cleaned.csv e gera um relatório detalhado sobre as equipes.
    """
    # Carregar os dados
    data_path = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'teams_cleaned.csv'
    df = pd.read_csv(data_path)
    
    # Preparar um resumo curto conforme pedido pelo utilizador
    min_year = df['year'].min()
    max_year = df['year'].max()

    teams = df.groupby('tmID')
    team_info_list = []
    for team_id, team_data in teams:
        team_data = team_data.sort_values('year')
        team_name = team_data.iloc[-1]['name']
        first_year = int(team_data['year'].min())
        last_year = int(team_data['year'].max())
        total_seasons = len(team_data)
        total_wins = int(team_data['won'].sum())
        total_losses = int(team_data['lost'].sum())
        total_games = int(team_data['GP'].sum())
        still_active = (last_year == max_year)
        team_info_list.append({
            'team_id': team_id,
            'team_name': team_name,
            'first_year': first_year,
            'last_year': last_year,
            'still_active': still_active,
            'total_seasons': total_seasons,
            'total_wins': total_wins,
            'total_losses': total_losses,
            'total_games': total_games
        })

    # Ordenar por nome para apresentação consistente
    team_info_list.sort(key=lambda x: x['team_name'])

    total_teams = len(team_info_list)
    active_teams = sum(1 for t in team_info_list if t['still_active'])
    inactive_teams = total_teams - active_teams

    # Listas de equipes (formatadas)
    active_lines = []
    inactive_lines = []
    for t in team_info_list:
        if t['still_active']:
            active_lines.append(f"  - {t['team_name']} ({t['team_id']})")
        else:
            inactive_lines.append(f"  - {t['team_name']} ({t['team_id']}) - Última temporada: {t['last_year']}")

    total_wins_all = sum(t['total_wins'] for t in team_info_list)
    total_losses_all = sum(t['total_losses'] for t in team_info_list)
    total_games_all = sum(t['total_games'] for t in team_info_list)

    # Encontrar equipe com mais vitórias e melhor taxa (mín. 3 temporadas)
    team_most_wins = max(team_info_list, key=lambda x: x['total_wins'])
    teams_min_seasons = [t for t in team_info_list if t['total_seasons'] >= 3]
    best_rate_line = ""
    if teams_min_seasons:
        team_best_rate = max(teams_min_seasons, key=lambda x: x['total_wins'] / x['total_games'] if x['total_games'] > 0 else 0)
        best_rate = (team_best_rate['total_wins'] / team_best_rate['total_games']) * 100
        best_rate_line = f"Equipe com melhor taxa de vitórias (mín. 3 temporadas): {team_best_rate['team_name']} ({best_rate:.2f}%)"

    # Montar as linhas finais exatamente no formato pedido
    report_lines = []
    report_lines.append(f"Número total de equipes analisadas: {total_teams}")
    report_lines.append(f"Equipes ainda em atividade: {active_teams}")
    report_lines.append(f"Equipes que deixaram de existir ou mudaram: {inactive_teams}")
    report_lines.append(f"Período total coberto: {min_year} até {max_year} ({max_year - min_year + 1} anos)")
    report_lines.append("")
    report_lines.append("Equipes em atividade:")
    report_lines.extend(active_lines)
    report_lines.append("")
    report_lines.append("Equipes que deixaram de existir ou mudaram:")
    report_lines.extend(inactive_lines)
    report_lines.append("")
    report_lines.append("Estatísticas gerais de todas as equipes:")
    report_lines.append(f"  Total de vitórias: {total_wins_all}")
    report_lines.append(f"  Total de derrotas: {total_losses_all}")
    report_lines.append(f"  Total de jogos: {total_games_all}")
    report_lines.append("")
    report_lines.append(f"Equipe com mais vitórias: {team_most_wins['team_name']} ({team_most_wins['total_wins']} vitórias)")
    if best_rate_line:
        report_lines.append(best_rate_line)

    report_path = Path(__file__).parent.parent.parent / 'reports' / 'teams_report.txt'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print(f"Relatório gerado com sucesso em: {report_path}")
    return str(report_path)

if __name__ == "__main__":
    generate_teams_report()
