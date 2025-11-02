import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV


# Suprimir warnings desnecessários
warnings.filterwarnings("ignore")


# Pesos para calcular a força da equipa 
TEAM_STRENGTH_WEIGHTS = {
    "avg_player_perf": 0.30,
    "total_player_perf": 0.15,
    "top5_avg_perf": 0.20,  # Performance dos top 5 jogadores
    "player_consistency": 0.08,  # Consistência das performances
    "prev_season_win_pct": 0.10,  # Histórico recente
    "coach_performance": 0.17,  # Performance do treinador 
}


# -----------------------------------------------------
# Utilidades de IO
# -----------------------------------------------------
def _candidate_dirs() -> List[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", "..", ".."))  # projeto raiz
    return [
        here,
        root,
        os.path.join(root, "data"),
        os.path.join(root, "data", "processed"),
        os.path.join(root, "data", "raw"),
    ]


def _find_csv(possible_names: List[str]) -> str:
    for name in possible_names:
        # 1) nome literal no CWD
        if os.path.isfile(name):
            return name
        # 2) procurar nos diretórios candidatos
        for d in _candidate_dirs():
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return p
    # não encontrado; devolver o primeiro como padrão (permitindo falhar cedo com mensagem clara)
    return possible_names[0]


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # normalizar nomes para minúsculas
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    rename_map = {c: c.lower() for c in df.columns}
    df.rename(columns=rename_map, inplace=True)

    # sinónimos comuns
    synonyms = {
        "teamid": "tmid",
        "team_id": "tmid",
        "team": "tmid",
        "idteam": "tmid",
        "teamname": "name",
        "tm": "tmid",
        "conference": "confid",
        "conf": "confid",
    }
    for k, v in synonyms.items():
        if k in df.columns and v not in df.columns:
            df.rename(columns={k: v}, inplace=True)
    return df


def _coerce_numeric(df: pd.DataFrame, exclude: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if c in exclude:
            continue
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_data() -> Dict[str, pd.DataFrame]:
    """Carrega todos os CSVs, aplicando correções de nomes e tipos.

    Localizações (com tolerância a nomes):
    - data/processed/player_performance.csv
    - data/processed/team_season_statistics.csv
    - data/processed/teams_cleaned.csv
    - data/processed/coach_performance.csv
    - data/raw/series_post.csv
    - data/raw/teams_post.csv
    """

    # diretórios base conforme a árvore do projeto
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    processed_dir = os.path.join(root, "data", "processed")
    raw_dir = os.path.join(root, "data", "raw")

    paths = {

        "players": _find_csv([
            os.path.join(processed_dir, "player_performance.csv"),
            "player_performance.csv",
        ]),
        "team_season": _find_csv([
            os.path.join(processed_dir, "team_season_statistics.csv"),
            "team_season_statistics.csv",
        ]),
        "teams_cleaned": _find_csv([
            os.path.join(processed_dir, "teams_cleaned.csv"),
            "teams_cleaned.csv",
        ]),
        "coach_performance": _find_csv([
            os.path.join(processed_dir, "coach_performance.csv"),
            "coach_performance.csv",
        ]),
        "series_post": _find_csv([
            os.path.join(raw_dir, "series_post.csv"),
            "series_post.csv",
        ]),
        "team_post": _find_csv([
            os.path.join(raw_dir, "teams_post.csv"),
            "teams_post.csv",
        ]),
    }

    dfs = {k: _standardize_columns(pd.read_csv(v)) for k, v in paths.items()}

    # garantir presença de year e tmid
    for key in [ "players", "team_season", "teams_cleaned", "coach_performance"]:
        df = dfs[key]
        missing = [c for c in ["year", "tmid"] if c not in df.columns]
        if missing:
            raise ValueError(f"CSV '{key}' está a faltar colunas obrigatórias: {missing}")

        # tipos básicos
        df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
        df["tmid"] = df["tmid"].astype(str)

    # preencher NaN com 0 depois das coerções numéricas
    for k in dfs.keys():
        # converter numéricos quando possível (excluindo ids e nomes)
        dfs[k] = _coerce_numeric(dfs[k], exclude=["tmid", "name", "confid", "arena", "coachid"]).fillna(0)

    return dfs

# -----------------------------------------------------
# Feature Engineering
# -----------------------------------------------------
def _pick_first_present(df: pd.DataFrame, candidates: List[str], default_name: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    # se nenhum existir, criar coluna 0
    df[default_name] = 0.0
    return default_name


def prepare_features(
    players: pd.DataFrame,
    team_season: pd.DataFrame,
    teams_cleaned: pd.DataFrame,
    coach_performance: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str], str]:
    """Gera features agregadas e devolve dataframe final, lista de features e target.

    - Aggregations avançadas de jogadores por (tmid, year)
    - Incorpora histórico do time (win%, point differential, tendências)
    - Incorpora performance dos coaches (coaching quality)
    - Calcula team_strength com múltiplos fatores
    """

    # 1) Player aggregates - features mais sofisticadas
    perf_col = _pick_first_present(players, ["performance", "perf", "score"], "performance")
    rookie_col = _pick_first_present(players, ["rookie", "is_rookie", "rookies"], "rookie")
    
    # Função para pegar top N jogadores por performance
    def top_n_avg(group, n=5):
        top_vals = group.nlargest(n)
        return top_vals.mean() if len(top_vals) > 0 else 0.0
    
    def performance_std(group):
        return group.std() if len(group) > 1 else 0.0

    player_aggs = (
        players.groupby(["tmid", "year"]).agg(
            avg_player_perf=(perf_col, "mean"),
            total_player_perf=(perf_col, "sum"),
            top5_avg_perf=(perf_col, lambda x: top_n_avg(x, 5)),
            top3_avg_perf=(perf_col, lambda x: top_n_avg(x, 3)),
            player_consistency=(perf_col, performance_std),  # menor = mais consistente
            rookie_count=(rookie_col, "sum"),
            team_size=(perf_col, "count"),
            max_player_perf=(perf_col, "max"),
            min_player_perf=(perf_col, "min"),
        )
        .reset_index()
    )
    
    # Inverter consistência (maior std = menor consistência normalizada)
    player_aggs["player_consistency"] = 1.0 / (1.0 + player_aggs["player_consistency"])

    # 2) Coach aggregates - USAR APENAS DADOS DA SEASON ANTERIOR para evitar leakage
    # Shift coach metrics em 1 ano para garantir que só usamos informação passada
    if "performance" in coach_performance.columns and "games" in coach_performance.columns:
        # Weighted average de coach performance por games coached
        coach_aggs_raw = (
            coach_performance.groupby(["tmid", "year"])
            .apply(lambda g: np.average(g["performance"], weights=g["games"]) if g["games"].sum() > 0 else 0.0)
            .reset_index()
            .rename(columns={0: "coach_performance_raw"})
        )
        # Shift para year anterior (year + 1 para ter dados do ano anterior disponíveis no ano atual)
        coach_aggs_raw["year"] = coach_aggs_raw["year"] + 1
        coach_aggs = coach_aggs_raw.rename(columns={"coach_performance_raw": "coach_performance"})
        
        # Número de coaches (também shifted)
        coach_extras_raw = (
            coach_performance.groupby(["tmid", "year"]).agg(
                num_coaches=("performance", "count"),  # número de coaches na season
            )
            .reset_index()
        )
        coach_extras_raw["year"] = coach_extras_raw["year"] + 1
        coach_aggs = coach_aggs.merge(coach_extras_raw, on=["tmid", "year"], how="left")
    else:
        # Fallback se coach_performance não tiver as colunas esperadas
        coach_aggs = pd.DataFrame(columns=["tmid", "year", "coach_performance", "num_coaches"])

    # 3) Juntar com team_season primeiro para ter acesso ao histórico
    df = team_season.copy()
    name_col = "name" if "name" in df.columns else None

    df = df.merge(player_aggs, on=["tmid", "year"], how="left")
    df = df.merge(coach_aggs, on=["tmid", "year"], how="left")

    # Trazer confid de teams_cleaned
    keep_cols = [c for c in ["tmid", "year", "confid"] if c in teams_cleaned.columns]
    conf_part = teams_cleaned[keep_cols].drop_duplicates()
    df = df.merge(conf_part, on=["tmid", "year"], how="left")

    # 4) Features de histórico do time (APENAS features de seasons anteriores - sem leakage)
    # REMOVIDAS: point_diff, home_win_pct, away_win_pct (são da season atual!)
    # MANTIDAS: apenas prev_season_* (dados históricos legítimos)
    historical_features = [
        "prev_season_win_pct_1", "prev_season_win_pct_3", "prev_season_win_pct_5",
        "prev_point_diff_3", "prev_point_diff_5", "win_pct_change_from_prev",
    ]
    
    for feat in historical_features:
        if feat not in df.columns:
            df[feat] = 0.0

    # Preencher coach features se não existirem (coach_win_pct REMOVIDA - era leakage)
    for feat in ["coach_performance", "num_coaches"]:
        if feat not in df.columns:
            df[feat] = 0.0

    # 5) Criar features derivadas (APENAS de dados históricos - sem leakage)
    # REMOVIDAS: momentum (usa win_pct_change da season atual), home_advantage (usa stats da season atual)
    # MANTIDAS: apenas features derivadas de dados passados
    
    # Experiência vs Renovação (ratio rookies/team_size)
    df["rookie_ratio"] = df["rookie_count"] / df["team_size"].clip(lower=1)
    
    # Profundidade do elenco (diferença entre top 5 e média geral)
    df["roster_depth"] = df["avg_player_perf"] / df["top5_avg_perf"].clip(lower=0.01)
    
    # Estabilidade do coaching staff (menos coaches = mais estável)
    df["coaching_stability"] = 1.0 / df["num_coaches"].clip(lower=1)

    # 6) Calcular team_strength com múltiplos fatores
    for c in ["avg_player_perf", "total_player_perf", "top5_avg_perf", 
              "player_consistency", "prev_season_win_pct_1", "coach_performance"]:
        if c not in df.columns:
            df[c] = 0.0
    
    df["team_strength"] = (
        df["avg_player_perf"] * TEAM_STRENGTH_WEIGHTS.get("avg_player_perf", 0.0)
        + df["total_player_perf"] * TEAM_STRENGTH_WEIGHTS.get("total_player_perf", 0.0)
        + df["top5_avg_perf"] * TEAM_STRENGTH_WEIGHTS.get("top5_avg_perf", 0.0)
        + df["player_consistency"] * TEAM_STRENGTH_WEIGHTS.get("player_consistency", 0.0)
        + df["prev_season_win_pct_1"] * TEAM_STRENGTH_WEIGHTS.get("prev_season_win_pct", 0.0)
        + df["coach_performance"] * TEAM_STRENGTH_WEIGHTS.get("coach_performance", 0.0)
    )

    # Target
    target = "season_win_pct"
    if target not in df.columns:
        raise ValueError("Coluna target 'season_win_pct' não encontrada em team_season_statistics.csv")

    # Features: APENAS features sem leakage (dados disponíveis ANTES da season)
    feature_cols = [
        # Player-based features (calculadas com roster da season, mas são preditivas)
        "avg_player_perf", "total_player_perf", "top5_avg_perf", "top3_avg_perf",
        "player_consistency", "rookie_count", "team_size", "max_player_perf",
        "min_player_perf", "rookie_ratio", "roster_depth",
        # Coach-based features (shifted para season anterior)
        "coach_performance", "num_coaches", "coaching_stability",
        # Historical features (APENAS prev_season_* - dados passados legítimos)
        "prev_season_win_pct_1", "prev_season_win_pct_3", "prev_season_win_pct_5",
        "prev_point_diff_3", "prev_point_diff_5", "win_pct_change_from_prev",
        # Composite
        "team_strength",
    ]
    # REMOVIDAS para evitar leakage:
    # - coach_win_pct (correlação 0.9998 com target - calculada da mesma season)
    # - point_diff, home_win_pct, away_win_pct (estatísticas da season atual)
    # - momentum, home_advantage (derivadas de stats da season atual)
    
    # Filtrar apenas features que existem
    feature_cols = [c for c in feature_cols if c in df.columns]

    # Converter para numérico e preencher NaN com 0
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    df[target] = pd.to_numeric(df[target], errors="coerce").fillna(0)

    # Preservar meta para ranking
    meta_cols = [c for c in ["tmid", "confid", "year", name_col] if c]
    df_meta = df[meta_cols + feature_cols + [target]].copy()

    return df_meta, feature_cols, target


# -----------------------------------------------------
# Treino do modelo
# -----------------------------------------------------
def train_model(
    df: pd.DataFrame, feature_cols: List[str], target: str, test_year: int | None = None
) -> Tuple[Pipeline, pd.DataFrame, pd.DataFrame]:
    """Treina um Pipeline(StandardScaler + GradientBoostingRegressor).

    Retorna: (modelo, X_test_com_meta, y_test_com_meta)
    """
    # split por season, senão aleatório
    X = df[feature_cols].copy()
    y = df[target].copy()
    meta_cols = [c for c in df.columns if c not in feature_cols + [target]]
    if test_year is not None and "year" in df.columns:
        train_mask = df["year"] < test_year  # seasons anteriores
        test_mask = df["year"] == test_year  # season de teste
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            # fallback para split aleatório caso falte dado
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            print("Aviso: split por season indisponível; usando split aleatório 80/20.")
        else:
            X_train, X_test = X.loc[train_mask], X.loc[test_mask]
            y_train, y_test = y.loc[train_mask], y.loc[test_mask]
            print(f"Split por season: treino (year < {test_year}) = {len(X_train)}, teste (year == {test_year}) = {len(X_test)}")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    # pipeline
    base_pipe = Pipeline(
        steps=[("scaler", StandardScaler()), ("regressor", GradientBoostingRegressor(random_state=42))]
    )
    param_dist = {
        "regressor__n_estimators": [100, 200, 400],
        "regressor__max_depth": [3, 5, 7],
        "regressor__learning_rate": [0.01, 0.05, 0.1],
        "regressor__subsample": [0.6, 0.8, 1.0],
    }
    search = RandomizedSearchCV(base_pipe, param_distributions=param_dist, n_iter=12, cv=3, random_state=42, n_jobs=-1)
    search.fit(X_train, y_train)
    model = search.best_estimator_

    # avaliação
    r2 = model.score(X_test, y_test)
    pred_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred_test)

    # devolver X_test com meta para ranking
    X_test_with_meta = X_test.copy()
    y_test_with_meta = y_test.copy()
    # juntar meta de volta por índice
    meta_test = df.loc[X_test.index, meta_cols]
    X_test_with_meta = pd.concat([meta_test.reset_index(drop=True), X_test_with_meta.reset_index(drop=True)], axis=1)
    y_test_with_meta = y_test_with_meta.reset_index(drop=True)

    return model, X_test_with_meta, y_test_with_meta


# -----------------------------------------------------
# Predição e ranking
# -----------------------------------------------------
def predict_and_rank(
    model: Pipeline,
    X_meta: pd.DataFrame,
    y_true: pd.Series,
    feature_cols: List[str],
    output_csv: str = "predicted_rankings.csv",
) -> pd.DataFrame:
    preds = model.predict(X_meta[feature_cols])
    out = X_meta[[c for c in X_meta.columns if c in ["tmid", "confid", "name"]]].copy()
    out["predicted_win_pct"] = preds

    # ranking por conferência
    if "confid" not in out.columns:
        out["confid"] = "UNKNOWN"
    out["rank"] = (
        out.groupby("confid")["predicted_win_pct"].rank(ascending=False, method="first").astype(int)
    )

    # ordenar para legibilidade
    out = out.sort_values(["confid", "rank", "predicted_win_pct"], ascending=[True, True, False])

    # garantir pasta results dentro de src/model e exportar CSV lá
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    output_csv_path = os.path.join(results_dir, os.path.basename(output_csv))
    out[["confid", "rank", "tmid", "name", "predicted_win_pct"]].to_csv(output_csv_path, index=False)

    return out


# -----------------------------------------------------
# Comparar e reportar performance
# -----------------------------------------------------
def compare_and_report(
    model: Pipeline,
    X_meta: pd.DataFrame,
    y_true: pd.Series,
    feature_cols: List[str] | None = None,
    output_report: str = "report.txt",
) -> pd.DataFrame:
    """
    Compara previsões com valores reais, calcula métricas e gera report.txt.
    - Ranking por conferência (previsto vs real)
    - Métricas: R2, MAE, RMSE, MAPE
    - Precisão de ranking: acerto exato, dentro de 1 posição, top-1 por conferência,
      overlap top-3, correlação de Spearman (global e por conferência)
    Salva o ficheiro report.txt no diretório de execução (CWD).
    Retorna um DataFrame com colunas: tmid, confid, year, name, predicted_win_pct, actual_win_pct, predicted_rank, actual_rank, rank_error.
    """
    meta_cols = [c for c in ["tmid", "confid", "year", "name"] if c in X_meta.columns]
    if feature_cols is None:
        feature_cols = [c for c in X_meta.columns if c not in meta_cols]

    # Prever
    preds = model.predict(X_meta[feature_cols])

    # Montar dataframe de comparação
    res = X_meta[meta_cols].copy()
    res["predicted_win_pct"] = preds
    res["actual_win_pct"] = np.asarray(y_true)
    if "confid" not in res.columns:
        res["confid"] = "UNKNOWN"
    if "name" not in res.columns:
        res["name"] = res["tmid"]

    # Ranks por conferência
    res["predicted_rank"] = res.groupby("confid")["predicted_win_pct"].rank(ascending=False, method="first").astype(int)
    res["actual_rank"] = res.groupby("confid")["actual_win_pct"].rank(ascending=False, method="first").astype(int)
    res["rank_error"] = (res["predicted_rank"] - res["actual_rank"]).abs()

    # Métricas de regressão
    r2 = r2_score(res["actual_win_pct"], res["predicted_win_pct"])
    mae = mean_absolute_error(res["actual_win_pct"], res["predicted_win_pct"])
    rmse = np.sqrt(mean_squared_error(res["actual_win_pct"], res["predicted_win_pct"]))
    denom = np.clip(np.abs(res["actual_win_pct"].values), 1e-8, None)
    mape = float(np.mean(np.abs((res["actual_win_pct"].values - res["predicted_win_pct"].values) / denom)) * 100.0)

    # Métricas de ranking (globais)
    exact_rank_acc = float((res["rank_error"] == 0).mean() * 100.0)
    within1_rank_acc = float((res["rank_error"] <= 1).mean() * 100.0)
    # Spearman global
    try:
        spearman_global = float(res[["predicted_win_pct", "actual_win_pct"]].corr(method="spearman").iloc[0, 1])
    except Exception:
        spearman_global = float("nan")

    # Métricas por conferência
    conf_rows = []
    for conf, g in res.groupby("confid"):
        # top-1
        pred_top = g.loc[g["predicted_rank"].idxmin(), "tmid"]
        true_top = g.loc[g["actual_rank"].idxmin(), "tmid"]
        top1_hit = int(pred_top == true_top)
        # overlap top-3
        pred_top3 = set(g.loc[g["predicted_rank"] <= 3, "tmid"])
        true_top3 = set(g.loc[g["actual_rank"] <= 3, "tmid"])
        top3_overlap = len(pred_top3 & true_top3) / max(1, len(true_top3))
        # spearman por conferência
        try:
            spearman_c = float(g[["predicted_win_pct", "actual_win_pct"]].corr(method="spearman").iloc[0, 1])
        except Exception:
            spearman_c = float("nan")
        conf_rows.append(
            {
                "confid": conf,
                "n_teams": len(g),
                "top1_accuracy": top1_hit,
                "top3_overlap": top3_overlap,
                "spearman": spearman_c,
            }
        )
    conf_df = pd.DataFrame(conf_rows)
    top1_by_conf_pct = float(conf_df["top1_accuracy"].mean() * 100.0) if not conf_df.empty else float("nan")
    top3_overlap_pct = float(conf_df["top3_overlap"].mean() * 100.0) if not conf_df.empty else float("nan")
    spearman_mean = float(conf_df["spearman"].mean()) if not conf_df.empty else float("nan")

    # Construir relatório
    lines = []
    lines.append("Relatório de Avaliação do Modelo (season_win_pct)")
    lines.append("-" * 60)
    lines.append(f"Nº equipas (teste): {len(res)}")
    if "year" in res.columns:
        years = sorted(res['year'].unique().tolist())
        lines.append(f"Seasons no teste: {years}")
    lines.append("")
    lines.append("Métricas de Regressão:")
    lines.append(f" - R²:  {r2:.4f}")
    lines.append(f" - MAE: {mae:.4f}")
    lines.append(f" - RMSE:{rmse:.4f}")
    lines.append(f" - MAPE:{mape:.2f}%")
    lines.append("")
    lines.append("Métricas de Ranking (global):")
    lines.append(f" - Acerto exato de ranking: {exact_rank_acc:.2f}%")
    lines.append(f" - Acerto dentro de 1 posição: {within1_rank_acc:.2f}%")
    lines.append(f" - Spearman (global): {spearman_global:.4f}")
    lines.append("")
    lines.append("Métricas por Conferência (médias):")
    lines.append(f" - Top-1 acertado (média por conf): {top1_by_conf_pct:.2f}%")
    lines.append(f" - Overlap Top-3 (média por conf): {top3_overlap_pct:.2f}%")
    lines.append(f" - Spearman (média por conf): {spearman_mean:.4f}")
    lines.append("")
    lines.append("Pesos de team_strength usados:")
    lines.append(f" - avg_player_perf: {TEAM_STRENGTH_WEIGHTS.get('avg_player_perf', 0.0)}")
    lines.append(f" - total_player_perf: {TEAM_STRENGTH_WEIGHTS.get('total_player_perf', 0.0)}")
    lines.append(f" - top5_avg_perf: {TEAM_STRENGTH_WEIGHTS.get('top5_avg_perf', 0.0)}")
    lines.append(f" - player_consistency: {TEAM_STRENGTH_WEIGHTS.get('player_consistency', 0.0)}")
    lines.append(f" - prev_season_win_pct: {TEAM_STRENGTH_WEIGHTS.get('prev_season_win_pct', 0.0)}")
    lines.append(f" - coach_performance: {TEAM_STRENGTH_WEIGHTS.get('coach_performance', 0.0)}")

    # Guardar report dentro de src/model/results
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    output_report_path = os.path.join(results_dir, os.path.basename(output_report))
    try:
        with open(output_report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"\nRelatório salvo em: {os.path.abspath(output_report_path)}")
    except Exception as e:
        print(f"Falha ao salvar o relatório: {e}")

    return res

def main():
    # 1) Load
    data = load_data()

    # 2) Prepare 
    df, feature_cols, target = prepare_features(
        players=data["players"],
        team_season=data["team_season"],
        teams_cleaned=data["teams_cleaned"],
        coach_performance=data["coach_performance"],
    )

    # 3) Train model: seasons 1–9; test: season 10
    model, X_test_with_meta, y_test = train_model(df, feature_cols, target, test_year=10)

    # 4) Predict & Rank
    predict_and_rank(model, X_test_with_meta, y_test, feature_cols)

    # 5) Compare and report performance against actual rankings
    compare_and_report(model, X_test_with_meta, y_test)

if __name__ == "__main__":
    main()
