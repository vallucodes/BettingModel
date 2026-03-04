import pandas as pd
import numpy as np
import duckdb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from datetime import datetime
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler

import xgboost as xgb


SAVE_RESULTS = False
RESULTS_FILE = "/media/vallu/Storage/Coding/Own_projects/betting_model/model/results_log.csv"
RUN_NOTE = "xgb_l15_map_190_90_17_team_130_50_25"

PARQUET_DIR = "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/parquet"
FEATURES_DIR = "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/features"
MAP_ELO_FILE = (
    "/media/vallu/Storage/Coding/Own_projects/betting_model/"
    "vallu_scraper/data/features/map_elo/features_map_elo_190_90_17.parquet"
)
TEAM_ELO_FILE = (
    "/media/vallu/Storage/Coding/Own_projects/betting_model/"
    "vallu_scraper/data/features/team_elo/features_team_elo_130_50_25.parquet"
)

con = duckdb.connect()

# Base match-level data (same as random_forest_1.py, but we could add LAN/BO flags later if desired)
df = con.execute(
    f"""
    SELECT
        hltv_match_id,
        team1_name,
        team2_name,
        team1_score,
        team2_score
    FROM '{PARQUET_DIR}/matches.parquet'
    ORDER BY hltv_match_id ASC
"""
).df()

df["result"] = np.where(df["team1_score"] > df["team2_score"], 0, 1)

# Games played per team (no Elo calculation here)
df["team1_games"] = 0
df["team2_games"] = 0
games_played = {
    team: 0
    for team in pd.unique(df[["team1_name", "team2_name"]].values.ravel())
}

for idx, row in df.iterrows():
    team1 = row["team1_name"]
    team2 = row["team2_name"]
    df.at[idx, "team1_games"] = games_played[team1]
    df.at[idx, "team2_games"] = games_played[team2]
    games_played[team1] += 1
    games_played[team2] += 1

# Rolling player/team performance features (l15)
rolling_df = pd.read_parquet(f"{FEATURES_DIR}/features_rolling_l15.parquet")[
    [
        "hltv_match_id",
        "team1_rolling_kast_l15",
        "team2_rolling_kast_l15",
        "team1_rolling_swing_l15",
        "team2_rolling_swing_l15",
        "team1_rolling_win_rate_l15",
        "team2_rolling_win_rate_l15",
    ]
]

rolling_df["kast_diff_l15"] = (
    rolling_df["team1_rolling_kast_l15"] - rolling_df["team2_rolling_kast_l15"]
)
rolling_df["swing_diff_l15"] = (
    rolling_df["team1_rolling_swing_l15"] - rolling_df["team2_rolling_swing_l15"]
)

df = df.merge(rolling_df, on="hltv_match_id", how="left")

# Map ELO features per team / per map
map_elo_df = pd.read_parquet(MAP_ELO_FILE)

MAPS_ELO = [
    "ancient",
    "anubis",
    "dust2",
    "inferno",
    "mirage",
    "nuke",
    "overpass",
    "train",
    "vertigo",
]

map_elo_cols = [f"{m}_elo_diff" for m in MAPS_ELO]

missing_cols = [c for c in map_elo_cols if c not in map_elo_df.columns]
if missing_cols:
    raise ValueError(f"Missing expected map ELO columns in parquet: {missing_cols}")

df = df.merge(
    map_elo_df[["hltv_match_id"] + map_elo_cols],
    on="hltv_match_id",
    how="left",
)

# Team Elo features from selected parquet file
team_elo_df = pd.read_parquet(TEAM_ELO_FILE)
required_team_cols = {"hltv_match_id", "team1_elo", "team2_elo"}
missing_team_cols = required_team_cols - set(team_elo_df.columns)
if missing_team_cols:
    raise ValueError(
        f"Missing expected team ELO columns in parquet: {sorted(missing_team_cols)}"
    )

df = df.merge(
    team_elo_df[["hltv_match_id", "team1_elo", "team2_elo"]],
    on="hltv_match_id",
    how="left",
)
df["elo_diff"] = df["team1_elo"] - df["team2_elo"]

# Filter to matches with reasonable data coverage
filtered_df = df[
    (df["team1_games"] > 10)
    & (df["team2_games"] > 10)
].reset_index(drop=True)

feature_cols = [
    "elo_diff",
    "kast_diff_l15",
    "swing_diff_l15",
    "team1_rolling_win_rate_l15",
    "team2_rolling_win_rate_l15",
] + map_elo_cols

train_df, test_df = train_test_split(filtered_df, test_size=0.2, shuffle=False)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Gradient boosting does not strictly need scaling, but modest scaling can help stability.
scaler = StandardScaler()
train_inputs = scaler.fit_transform(train_df[feature_cols])
test_inputs = scaler.transform(test_df[feature_cols])

train_targets = train_df["result"].astype(int)
test_targets = test_df["result"].astype(int)

print(f"Number of training matches: {len(train_df)}")
print(f"Number of testing matches: {len(test_df)}")

# Build DMatrix objects for XGBoost
dtrain = xgb.DMatrix(train_inputs, label=train_targets, feature_names=feature_cols)
dtest = xgb.DMatrix(test_inputs, label=test_targets, feature_names=feature_cols)

# XGBoost parameters tuned for betting-style probability calibration / robustness
params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",  # primary
    "tree_method": "hist",
    "max_depth": 5,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.0,
    "lambda": 1.0,
    "alpha": 0.0,
    "scale_pos_weight": 1.0,
}

evals = [(dtrain, "train"), (dtest, "test")]

bst = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=evals,
    early_stopping_rounds=50,
    verbose_eval=50,
)


def evaluate_split(pred_probs, targets, name: str = ""):
    ll = log_loss(targets, pred_probs)
    roc = roc_auc_score(targets, pred_probs)
    brier = brier_score_loss(targets, pred_probs)
    print(f"{name} Log Loss: {ll:.3f}")
    print(f"{name} ROC-AUC: {roc:.3f}")
    print(f"{name} Brier: {brier:.3f}")
    return {"log_loss": ll, "roc_auc": roc, "brier": brier}


train_probs = bst.predict(dtrain)
test_probs = bst.predict(dtest)

train_metrics = evaluate_split(train_probs, train_targets, "Train")
test_metrics = evaluate_split(test_probs, test_targets, "Test")

print("\n--- Overfit gap (train - test), positive = overfitting ---")
print(
    f"Log Loss delta: {test_metrics['log_loss'] - train_metrics['log_loss']:+.3f} "
    "(overfit if >0.05-0.1)"
)
print(
    f"ROC-AUC delta: {train_metrics['roc_auc'] - test_metrics['roc_auc']:+.3f} "
    "(overfit if >0.02-0.05)"
)
print(
    f"Brier delta: {test_metrics['brier'] - train_metrics['brier']:+.3f} "
    "(overfit if >0.01-0.03)\n"
)

# Feature importance (gain-based)
importance_dict = bst.get_score(importance_type="gain")
fi_rows = []
for fname in feature_cols:
    fi_rows.append(
        {"feature": fname, "importance": importance_dict.get(fname, 0.0)}
    )
fi_df = pd.DataFrame(fi_rows)
fi_df = fi_df.sort_values("importance", ascending=False)

print("\n=== XGBoost feature importance (gain) ===")
print(fi_df.to_string(index=False, float_format="{:.4f}".format))

TOP_N = 20
plot_df = fi_df.head(TOP_N).sort_values("importance")

plt.figure(figsize=(10, 6))
plt.barh(plot_df["feature"], plot_df["importance"])
plt.title(f"Top {TOP_N} XGBoost feature importances (gain)")
plt.xlabel("Gain importance")
plt.tight_layout()
plt.show()

# Odds-style output for recent matches
odds_df = test_df[["hltv_match_id", "team1_name", "team2_name", "result"]].copy()
odds_df["team2_win_prob"] = test_probs
odds_df["team1_win_prob"] = 1 - test_probs
odds_df["team1_odds"] = 1 / odds_df["team1_win_prob"]
odds_df["team2_odds"] = 1 / odds_df["team2_win_prob"]
odds_df = odds_df.round(2)

start_id = 2385965
start_idx = odds_df[odds_df["hltv_match_id"] >= start_id].index.min()
if pd.notna(start_idx):
    print(
        odds_df[
            [
                "hltv_match_id",
                "team1_name",
                "team2_name",
                "team1_win_prob",
                "team2_win_prob",
                "team1_odds",
                "team2_odds",
                "result",
            ]
        ].iloc[int(start_idx) : int(start_idx) + 100]
    )

if SAVE_RESULTS:
    run_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": RUN_NOTE,
    }
    for i, f in enumerate(feature_cols, 1):
        run_data[f"feature_{i}"] = f
    run_data.update(
        {
            "train_matches": len(train_df),
            "test_matches": len(test_df),
            "train_log_loss": round(train_metrics["log_loss"], 4),
            "train_roc_auc": round(train_metrics["roc_auc"], 4),
            "train_brier": round(train_metrics["brier"], 4),
            "test_log_loss": round(test_metrics["log_loss"], 4),
            "test_roc_auc": round(test_metrics["roc_auc"], 4),
            "test_brier": round(test_metrics["brier"], 4),
        }
    )

    new_col = pd.Series(run_data, name=datetime.now().strftime("%Y%m%d_%H%M%S"))

    if os.path.exists(RESULTS_FILE):
        existing = pd.read_csv(RESULTS_FILE, index_col=0)
        updated = pd.concat([existing, new_col], axis=1)
    else:
        updated = new_col.to_frame()

    all_features = sorted(
        [idx for idx in updated.index if idx.startswith("feature_")],
        key=lambda x: int(x.split("_")[1]),
    )

    fixed_order = [
        "timestamp",
        "model",
        *all_features,
        "train_matches",
        "test_matches",
        "train_log_loss",
        "train_roc_auc",
        "train_brier",
        "test_log_loss",
        "test_roc_auc",
        "test_brier",
    ]

    updated = updated.reindex([r for r in fixed_order if r in updated.index])
    updated.to_csv(RESULTS_FILE)

