import os
import re
from typing import Dict, Any, List

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import xgboost as xgb


PARQUET_DIR = (
    "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/parquet"
)
FEATURES_ROOT = (
    "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/features"
)
MAP_ELO_FILE = (
    "/media/vallu/Storage/Coding/Own_projects/betting_model/"
    "vallu_scraper/data/features/map_elo/features_map_elo_190_90_17.parquet"
)
TRUESKILL_DIR = os.path.join(FEATURES_ROOT, "team_trueskill")

# Overfitting thresholds (same as logistic_4_l15_team_elo_sweep)
MAX_LL_DELTA = 0.05  # test_log_loss - train_log_loss
MAX_ROC_DELTA = 0.02  # train_roc_auc - test_roc_auc
MAX_BR_DELTA = 0.01  # test_brier - train_brier


def build_base_dataframe() -> tuple[pd.DataFrame, List[str]]:
    """
    Build the part of the dataframe that is common for all runs, mirroring
    logistic_4_l15_team_elo_sweep:
    - match-level data (incl. LAN/BO flags)
    - simple team Elo for games-played filtering
    - rolling l15 features
    - fixed map ELO features from MAP_ELO_FILE
    """
    con = duckdb.connect()

    df = con.execute(
        f"""
        SELECT
            hltv_match_id,
            team1_name,
            team2_name,
            team1_score,
            team2_score,
            event_type,
            is_bo1,
            is_bo3,
            is_bo5
        FROM '{PARQUET_DIR}/matches.parquet'
        ORDER BY hltv_match_id ASC
    """
    ).df()
    con.close()

    df["result"] = np.where(df["team1_score"] > df["team2_score"], 0, 1)
    df["is_lan"] = (df["event_type"] == "LAN").astype(int)

    for col in ["is_bo1", "is_bo3", "is_bo5"]:
        df[col] = df[col].fillna(0).astype(int)

    # Simple team-level Elo (used for games-played filtering)
    HIGH_K = 100
    LOW_K = 20
    THRESHOLD = 30

    df["team1_elo_simple"] = None
    df["team2_elo_simple"] = None
    df["elo_diff_simple"] = None
    df["team1_games"] = None
    df["team2_games"] = None

    unique_teams = pd.unique(df[["team1_name", "team2_name"]].values.ravel())
    elo = {team: {"elo": 1500.0, "games_played": 0} for team in unique_teams}

    def expected_score(elo_1: float, elo_2: float) -> float:
        return 1.0 / (1.0 + 10 ** ((elo_2 - elo_1) / 400.0))

    def update_elo(
        elo_1: float, elo_2: float, res: int, k1: float, k2: float
    ) -> tuple[float, float]:
        ea = expected_score(elo_1, elo_2)
        elo_1_new = elo_1 + k1 * ((1 - res) - ea)
        elo_2_new = elo_2 + k2 * (res - (1 - ea))
        return elo_1_new, elo_2_new

    for idx, row in df.iterrows():
        team1, team2 = row["team1_name"], row["team2_name"]
        df.at[idx, "team1_elo_simple"] = elo[team1]["elo"]
        df.at[idx, "team2_elo_simple"] = elo[team2]["elo"]
        df.at[idx, "elo_diff_simple"] = elo[team1]["elo"] - elo[team2]["elo"]
        df.at[idx, "team1_games"] = elo[team1]["games_played"]
        df.at[idx, "team2_games"] = elo[team2]["games_played"]

        res = df.at[idx, "result"]

        k1 = HIGH_K if elo[team1]["games_played"] < THRESHOLD else LOW_K
        k2 = HIGH_K if elo[team2]["games_played"] < THRESHOLD else LOW_K

        elo[team1]["elo"], elo[team2]["elo"] = update_elo(
            elo[team1]["elo"], elo[team2]["elo"], res, k1, k2
        )
        elo[team1]["games_played"] += 1
        elo[team2]["games_played"] += 1

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

    # Very simple feature set: map ELOs + per-team conservative TrueSkill ratings
    feature_cols = map_elo_cols + [
        "team1_trueskill_conservative",
        "team2_trueskill_conservative",
    ]

    train_df, test_df = train_test_split(filtered_df, test_size=0.2, shuffle=False)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    scaler = StandardScaler()
    train_inputs = scaler.fit_transform(train_df[feature_cols])
    test_inputs = scaler.transform(test_df[feature_cols])
    train_targets = train_df["result"].astype(int)
    test_targets = test_df["result"].astype(int)

    dtrain = xgb.DMatrix(train_inputs, label=train_targets, feature_names=feature_cols)
    dtest = xgb.DMatrix(test_inputs, label=test_targets, feature_names=feature_cols)

    params = {
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

    evals = [(dtrain, "train"), (dtest, "test")]

    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=3000,
        evals=evals,
        early_stopping_rounds=100,
        verbose_eval=False,
    )

    if hasattr(bst, "best_ntree_limit") and bst.best_ntree_limit is not None:
        train_probs = bst.predict(dtrain, ntree_limit=bst.best_ntree_limit)
        test_probs = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)
    else:
        train_probs = bst.predict(dtrain)
        test_probs = bst.predict(dtest)

    train_ll = log_loss(train_targets, train_probs)
    test_ll = log_loss(test_targets, test_probs)
    train_roc = roc_auc_score(train_targets, train_probs)
    test_roc = roc_auc_score(test_targets, test_probs)
    train_brier = brier_score_loss(train_targets, train_probs)
    test_brier = brier_score_loss(test_targets, test_probs)

    ll_delta = test_ll - train_ll
    roc_delta = train_roc - test_roc
    br_delta = test_brier - train_brier

    within_overfit = (
        ll_delta <= MAX_LL_DELTA
        and roc_delta <= MAX_ROC_DELTA
        and br_delta <= MAX_BR_DELTA
    )

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
        "test_log_loss": test_ll,
        "test_roc_auc": test_roc,
        "test_brier": test_brier,
        "ll_delta": ll_delta,
        "roc_delta": roc_delta,
        "br_delta": br_delta,
        "within_overfit_limits": within_overfit,
        "train_matches": len(train_df),
        "test_matches": len(test_df),
        "best_iteration": getattr(bst, "best_iteration", None),
    }

    print(
        f"\nRun {fname}: "
        f"Train LL={train_ll:.4f}, ROC={train_roc:.4f}, Brier={train_brier:.4f} | "
        f"Test LL={test_ll:.4f}, ROC={test_roc:.4f}, Brier={test_brier:.4f} | "
        f"dLL={ll_delta:+.4f}, dROC={roc_delta:+.4f}, dBrier={br_delta:+.4f}, "
        f"within_overfit_limits={within_overfit}"
    )

    return result


def main() -> None:
    base_df, map_elo_cols = build_base_dataframe()

    if not os.path.isdir(TRUESKILL_DIR):
        print(f"TRUESKILL_DIR does not exist: {TRUESKILL_DIR}")
        return

    ts_files: List[str] = sorted(
        os.path.join(TRUESKILL_DIR, f)
        for f in os.listdir(TRUESKILL_DIR)
        if f.startswith("features_team_trueskill_") and f.endswith(".parquet")
    )

    if not ts_files:
        print(f"No features_team_trueskill_*.parquet files found in {TRUESKILL_DIR}")
        return

    print(f"Found {len(ts_files)} TrueSkill feature files.")

    all_results: List[Dict[str, Any]] = []

    for i, path in enumerate(ts_files, start=1):
        fname = os.path.basename(path)
        try:
            res = run_single_trueskill_model(base_df, map_elo_cols, path)
            all_results.append(res)
            print(f"=== [{i}/{len(ts_files)}] Done {fname} ===")
        except Exception as e:
            print(f"❌ Failed for {fname}: {e}")

    if not all_results:
        print("No successful runs, nothing to summarize.")
        return

    results_df = pd.DataFrame(all_results)

    # Competition-style ranking: favour low loss/Brier and high ROC.
    # Lower score is better.
    results_df["score"] = (
        results_df["test_log_loss"]
        + results_df["test_brier"]
        - results_df["test_roc_auc"]
    )

    # Sort by:
    # 1) within_overfit_limits (True first)
    # 2) lowest competition score
    # 3) lowest test_log_loss
    # 4) highest test_roc_auc
    results_df = results_df.sort_values(
        by=[
            "within_overfit_limits",
            "score",
            "test_log_loss",
            "test_roc_auc",
        ],
        ascending=[False, True, True, False],
    ).reset_index(drop=True)

    print("\n=== Summary of all TrueSkill XGBoost runs (best first by competition score) ===")
    display_cols = [
        "sigma",
        "beta",
        "tau",
        "train_log_loss",
        "test_log_loss",
        "train_roc_auc",
        "test_roc_auc",
        "train_brier",
        "test_brier",
        "ll_delta",
        "roc_delta",
        "br_delta",
        "score",
        "within_overfit_limits",
        "train_matches",
        "test_matches",
    ]
    print(
        results_df[display_cols].to_string(
            index=False, float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\nTop 10 TrueSkill combinations within overfitting limits:")
    top_within = results_df[results_df["within_overfit_limits"]].head(10)
    if top_within.empty:
        print("No combinations satisfied the overfitting constraints.")
    else:
        print(
            top_within[display_cols].to_string(
                index=False, float_format=lambda x: f"{x:.4f}"
            )
        )


if __name__ == "__main__":
    main()
