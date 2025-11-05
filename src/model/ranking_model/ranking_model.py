import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from pathlib import Path

# 1. Load and prepare data
# Find the repository root (3 levels up from this script)
script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent.parent.parent
data_path = repo_root / "data" / "processed" / "team_performance.csv"

df = pd.read_csv(data_path)

# Remove incomplete rows
df = df.dropna(subset=["rs_win_pct"])

# Select features (sem dados que olhem pro futuro)
features = [
    "team_strength",
    "pythag_win_pct",
    "rs_win_pct_expected_roster",
    "overach_pythag",
    "overach_roster",
    "rs_win_pct_prev",
    "win_pct_change"
]
target = "rs_win_pct"

# 2. Split by year (train 1–8, test 9–10)
train_df = df[df["year"].between(1, 8)]
test_df = df[df["year"].between(9, 10)]

X_train, y_train = train_df[features], train_df[target]
X_test, y_test = test_df[features], test_df[target]

# 3. Train model
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# 4. Predictions
y_pred = model.predict(X_test)

# 5. Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 6. Generate ranking
test_df = test_df.copy()
test_df["predicted_rs_win_pct"] = y_pred
test_df["predicted_rank"] = test_df.groupby("year")["predicted_rs_win_pct"].rank(ascending=False)
test_df["actual_rank"] = test_df.groupby("year")["rs_win_pct"].rank(ascending=False)

# 7. Ranking accuracy (Spearman correlation per year)
from scipy.stats import spearmanr
spearman_scores = []
for year, group in test_df.groupby("year"):
    corr, _ = spearmanr(group["actual_rank"], group["predicted_rank"])
    spearman_scores.append((year, corr))
mean_spearman = np.mean([s for _, s in spearman_scores])

# 8. Save report
with open("report.txt", "w", encoding="utf-8") as f:
    f.write("Team Ranking Prediction Report\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Model: Gradient Boosting Regressor (XGBoost)\n")
    f.write(f"Training seasons: 1–8 | Testing seasons: 9–10\n\n")
    f.write(f"Mean Absolute Error (MAE): {mae:.4f}\n")
    f.write(f"R² Score: {r2:.4f}\n")
    f.write(f"Mean Spearman Correlation (Ranking Accuracy): {mean_spearman:.4f}\n\n")
    f.write("Per-Year Spearman correlations:\n")
    for year, score in spearman_scores:
        f.write(f"  Year {year}: {score:.4f}\n")
    f.write("\nFeature Importance:\n")
    importance = sorted(zip(features, model.feature_importances_), key=lambda x: x[1], reverse=True)
    for feat, val in importance:
        f.write(f"  {feat}: {val:.4f}\n\n")
    f.write("Explanation:\n")
    f.write("- The model predicts each team's expected win percentage for a season.\n")
    f.write("- Rankings are derived by sorting teams by predicted win percentage.\n")
    f.write("- Spearman correlation measures how well predicted rankings match actual ones.\n")
    f.write("- The model currently uses team-based features only (no coach/player inputs).\n")
    f.write("- It is ready for future expansion to include coach performance or player stats.\n")

print("✅ Report saved as report.txt")
