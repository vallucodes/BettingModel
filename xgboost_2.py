"""
Usage, from the project root:
    python xgboost_2.py
"""

import os
import re
from typing import Dict, Any, List

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

PARQUET_DIR = (
    "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/parquet"
)
FEATURES_ROOT = (
    "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/features"
)
MAP_ELO_FILE = os.path.join(FEATURES_ROOT, "features_map_elo_190_90_17.parquet")
# Single TrueSkill features parquet to use (no sweeping)
TRUESKILL_FILE = os.path.join(
    FEATURES_ROOT,
    "team_trueskill",
    "features_team_trueskill_15.000_8.000_0.500.parquet",
)

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
MAX_BOOST_ROUNDS = 3000
EARLY_STOPPING_ROUNDS = 100

# These limits compare training with validation, not the held-out test set.
MAX_LL_DELTA = 0.05  # validation_log_loss - train_log_loss
MAX_ROC_DELTA = 0.02  # train_roc_auc - validation_roc_auc
MAX_BR_DELTA = 0.01  # validation_brier - train_brier

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "max_depth": 3,
    "eta": 0.01,
    "subsample": 0.5,
    "colsample_bytree": 0.5,
    "min_child_weight": 20,
    "gamma": 1.0,
    "lambda": 2.0,
    "alpha": 0.0,
    "scale_pos_weight": 1.0,
}


def split_chronologically(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split rows into train, validation, and test in start-time order."""
    df = df.sort_values(
        ["started_at", "hltv_match_id"], na_position="last"
    ).reset_index(drop=True)

    train_end = int(len(df) * TRAIN_FRACTION)
    validation_end = int(len(df) * (TRAIN_FRACTION + VALIDATION_FRACTION))
    if train_end == 0 or validation_end <= train_end or validation_end >= len(df):
        raise ValueError(
            "Not enough dated matches for non-empty train, validation, and test splits."
        )

    return (
        df.iloc[:train_end].reset_index(drop=True),
        df.iloc[train_end:validation_end].reset_index(drop=True),
        df.iloc[validation_end:].reset_index(drop=True),
    )


def best_boost_rounds(bst: xgb.Booster) -> int:
    best_iteration = getattr(bst, "best_iteration", None)
    if best_iteration is not None:
        return int(best_iteration) + 1

    best_ntree_limit = getattr(bst, "best_ntree_limit", None)
    if best_ntree_limit:
        return int(best_ntree_limit)

    return MAX_BOOST_ROUNDS


def predict_at_best_iteration(bst: xgb.Booster, data: xgb.DMatrix) -> np.ndarray:
    best_ntree_limit = getattr(bst, "best_ntree_limit", None)
    if best_ntree_limit:
        return bst.predict(data, ntree_limit=best_ntree_limit)

    best_iteration = getattr(bst, "best_iteration", None)
    if best_iteration is not None:
        return bst.predict(data, iteration_range=(0, int(best_iteration) + 1))

    return bst.predict(data)


def build_base_dataframe() -> tuple[pd.DataFrame, List[str]]:
    """
    Build the part of the dataframe that is common for all runs, mirroring
    logistic_4_l15_team_elo_sweep:
    - match-level data (incl. LAN/BO flags)
    - team game counts for eligibility filtering
    - rolling l15 features
    - fixed map ELO features from MAP_ELO_FILE
    """
    con = duckdb.connect()

    df = con.execute(
        f"""
        SELECT
            hltv_match_id,
            TRY_CAST(date AS TIMESTAMP) AS started_at,
            team1_name,
            team2_name,
            team1_score,
            team2_score,
            event_type,
            is_bo1,
            is_bo3,
            is_bo5
        FROM '{PARQUET_DIR}/matches.parquet'
        WHERE TRY_CAST(date AS TIMESTAMP) IS NOT NULL
        ORDER BY started_at ASC, hltv_match_id ASC
    """
    ).df()
    con.close()

    df["result"] = np.where(df["team1_score"] > df["team2_score"], 0, 1)
    df["is_lan"] = (df["event_type"] == "LAN").astype(int)

    for col in ["is_bo1", "is_bo3", "is_bo5"]:
        df[col] = df[col].fillna(0).astype(int)

    df["team1_games"] = None
    df["team2_games"] = None

    unique_teams = pd.unique(df[["team1_name", "team2_name"]].values.ravel())
    games_played = {team: 0 for team in unique_teams}

    for idx, row in df.iterrows():
        team1, team2 = row["team1_name"], row["team2_name"]
        df.at[idx, "team1_games"] = games_played[team1]
        df.at[idx, "team2_games"] = games_played[team2]
        games_played[team1] += 1
        games_played[team2] += 1

    # Rolling l15 features
    rolling_df = pd.read_parquet(
        os.path.join(FEATURES_ROOT, "features_rolling_l15.parquet")
    )[
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

    # Fixed map ELO features (MAP_ELO_FILE constant)
    map_elo_df = pd.read_parquet(MAP_ELO_FILE)

    maps_elo = [
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
    map_elo_cols = [f"{m}_elo_diff" for m in maps_elo]

    missing_cols = [c for c in map_elo_cols if c not in map_elo_df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing expected map ELO columns in MAP_ELO_FILE: {missing_cols}"
        )

    df = df.merge(
        map_elo_df[["hltv_match_id"] + map_elo_cols],
        on="hltv_match_id",
        how="left",
    )

    return df, map_elo_cols


def run_single_trueskill_model(
    base_df: pd.DataFrame, map_elo_cols: List[str], ts_path: str
) -> Dict[str, Any]:
    """
    Run an XGBoost model for a single features_team_trueskill_*.parquet file.
    - Uses fixed map ELO features from MAP_ELO_FILE.
    - Adds TrueSkill-based pre-match rating differences as features.
    Returns a dict of metrics and metadata about this run.
    """
    ts_df = pd.read_parquet(ts_path)

    # Only require conservative ratings for each team
    required_cols = {
        "hltv_match_id",
        "team1_trueskill_conservative",
        "team2_trueskill_conservative",
    }
    missing = required_cols - set(ts_df.columns)
    if missing:
        raise ValueError(
            f"{os.path.basename(ts_path)} is missing required columns: {sorted(missing)}"
        )

    df = base_df.merge(
        ts_df[["hltv_match_id", "team1_trueskill_conservative", "team2_trueskill_conservative"]],
        on="hltv_match_id",
        how="left",
    )

    filtered_df = df[
        (df["team1_games"] > 10)
        & (df["team2_games"] > 10)
    ].reset_index(drop=True)

    # Feature set: match context, map ELOs, rolling l15 performance, and TrueSkill ratings
    feature_cols = [
        "is_lan",
        "is_bo1",
        "is_bo3",
        "is_bo5",
        "kast_diff_l15",
        "swing_diff_l15",
        "team1_rolling_win_rate_l15",
        "team2_rolling_win_rate_l15",
        "team1_trueskill_conservative",
        "team2_trueskill_conservative",
    ]

    train_df, validation_df, test_df = split_chronologically(filtered_df)

    scaler = StandardScaler()
    train_inputs = scaler.fit_transform(train_df[feature_cols])
    validation_inputs = scaler.transform(validation_df[feature_cols])
    train_targets = train_df["result"].astype(int)
    validation_targets = validation_df["result"].astype(int)

    dtrain = xgb.DMatrix(train_inputs, label=train_targets, feature_names=feature_cols)
    dvalidation = xgb.DMatrix(
        validation_inputs, label=validation_targets, feature_names=feature_cols
    )

    validation_bst = xgb.train(
        XGB_PARAMS,
        dtrain,
        num_boost_round=MAX_BOOST_ROUNDS,
        evals=[(dtrain, "train"), (dvalidation, "validation")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )

    train_probs = predict_at_best_iteration(validation_bst, dtrain)
    validation_probs = predict_at_best_iteration(validation_bst, dvalidation)
    train_ll = log_loss(train_targets, train_probs)
    validation_ll = log_loss(validation_targets, validation_probs)
    train_roc = roc_auc_score(train_targets, train_probs)
    validation_roc = roc_auc_score(validation_targets, validation_probs)
    train_brier = brier_score_loss(train_targets, train_probs)
    validation_brier = brier_score_loss(validation_targets, validation_probs)

    ll_delta = validation_ll - train_ll
    roc_delta = train_roc - validation_roc
    br_delta = validation_brier - train_brier
    overfit = not (
        ll_delta <= MAX_LL_DELTA
        and roc_delta <= MAX_ROC_DELTA
        and br_delta <= MAX_BR_DELTA
    )

    # Refit with the validation-selected round count before touching test data.
    best_rounds = best_boost_rounds(validation_bst)
    train_validation_df = pd.concat([train_df, validation_df], ignore_index=True)
    scaler = StandardScaler()
    train_validation_inputs = scaler.fit_transform(
        train_validation_df[feature_cols]
    )
    test_inputs = scaler.transform(test_df[feature_cols])
    train_validation_targets = train_validation_df["result"].astype(int)
    test_targets = test_df["result"].astype(int)
    dtrain_validation = xgb.DMatrix(
        train_validation_inputs,
        label=train_validation_targets,
        feature_names=feature_cols,
    )
    dtest = xgb.DMatrix(test_inputs, label=test_targets, feature_names=feature_cols)
    final_bst = xgb.train(
        XGB_PARAMS,
        dtrain_validation,
        num_boost_round=best_rounds,
        verbose_eval=False,
    )
    test_probs = final_bst.predict(dtest)

    test_ll = log_loss(test_targets, test_probs)
    test_roc = roc_auc_score(test_targets, test_probs)
    test_brier = brier_score_loss(test_targets, test_probs)

    fname = os.path.basename(ts_path)
    m = re.match(
        r"features_team_trueskill_([0-9.]+)_([0-9.]+)_([0-9.]+)\.parquet", fname
    )
    sigma = float(m.group(1)) if m else None
    beta = float(m.group(2)) if m else None
    tau = float(m.group(3)) if m else None

    result: Dict[str, Any] = {
        "file": fname,
        "sigma": sigma,
        "beta": beta,
        "tau": tau,
        "train_log_loss": train_ll,
        "train_roc_auc": train_roc,
        "train_brier": train_brier,
        "validation_log_loss": validation_ll,
        "validation_roc_auc": validation_roc,
        "validation_brier": validation_brier,
        "test_log_loss": test_ll,
        "test_roc_auc": test_roc,
        "test_brier": test_brier,
        "ll_delta": ll_delta,
        "roc_delta": roc_delta,
        "br_delta": br_delta,
        "overfit": overfit,
        "train_matches": len(train_df),
        "validation_matches": len(validation_df),
        "test_matches": len(test_df),
        "best_rounds": best_rounds,
    }

    print(
        f"\nRun {fname}: "
        f"Train LL={train_ll:.4f}, ROC={train_roc:.4f}, Brier={train_brier:.4f} | "
        f"Validation LL={validation_ll:.4f}, ROC={validation_roc:.4f}, "
        f"Brier={validation_brier:.4f} | "
        f"dLL={ll_delta:+.4f}, dROC={roc_delta:+.4f}, dBrier={br_delta:+.4f}, "
        f"overfit={overfit}"
    )
    print(
        f"Final test (refit on train+validation, {best_rounds} rounds): "
        f"LL={test_ll:.4f}, ROC={test_roc:.4f}, Brier={test_brier:.4f}"
    )

    return result


def main() -> None:
    base_df, map_elo_cols = build_base_dataframe()

    if not os.path.exists(TRUESKILL_FILE):
        print(f"TrueSkill features file does not exist: {TRUESKILL_FILE}")
        return

    print(f"Using TrueSkill features from: {TRUESKILL_FILE}")
    result = run_single_trueskill_model(base_df, map_elo_cols, TRUESKILL_FILE)

    print("\n=== Single TrueSkill XGBoost run summary ===")
    print(
        f"File={result['file']}, "
        f"sigma={result['sigma']}, beta={result['beta']}, tau={result['tau']}"
    )
    print(
        f"Train: LL={result['train_log_loss']:.4f}, "
        f"ROC={result['train_roc_auc']:.4f}, "
        f"Brier={result['train_brier']:.4f}"
    )
    print(
        f"Validation: LL={result['validation_log_loss']:.4f}, "
        f"ROC={result['validation_roc_auc']:.4f}, "
        f"Brier={result['validation_brier']:.4f}"
    )
    print(
        f"Test:  LL={result['test_log_loss']:.4f}, "
        f"ROC={result['test_roc_auc']:.4f}, "
        f"Brier={result['test_brier']:.4f}"
    )
    print(
        f"Train-to-validation dLL={result['ll_delta']:+.4f}, "
        f"dROC={result['roc_delta']:+.4f}, "
        f"dBrier={result['br_delta']:+.4f}, "
        f"overfit={result['overfit']}, "
        f"best_rounds={result['best_rounds']}"
    )


if __name__ == "__main__":
    main()
