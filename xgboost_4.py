"""Train the XGBoost match model with precomputed joint map-strength features.

Usage:
    python xgboost_4.py
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
import matplotlib.pyplot as plt


MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
PARQUET_DIR = (
    "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/parquet"
)
FEATURES_ROOT = (
    "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/features"
)
ROSTER_TRUESKILL_DIR = os.path.join(FEATURES_ROOT, "roster_trueskill")
MAP_STRENGTH_FILE = os.path.join(
    FEATURES_ROOT, "features_map_strength_joint_v3.parquet"
)
PLOTS_DIR = os.path.join(MODEL_DIR, "plots")

MAP_FEATURES = [
    "map_p_team2",
    "map_edge_mean",
    "map_edge_best",
    "map_edge_worst",
    "map_offset_edge_mean",
    "map_offset_se_mean",
]

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
MAX_BOOST_ROUNDS = 3000
EARLY_STOPPING_ROUNDS = 100

# These limits are applied to the validation metrics during candidate selection.
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


def save_feature_importance_plots(
    bst: xgb.Booster,
    feature_cols: List[str],
    model_label: str,
) -> None:
    """Save gain and split-frequency importance plots for a fitted model."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    for importance_type, importance_label in (
        ("gain", "Average gain"),
        ("weight", "Split frequency"),
    ):
        scores = bst.get_score(importance_type=importance_type)
        importance_df = pd.DataFrame(
            {
                "feature": feature_cols,
                "importance": [scores.get(feature, 0.0) for feature in feature_cols],
            }
        ).sort_values("importance", ascending=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(importance_df["feature"], importance_df["importance"])
        ax.set_title(f"XGBoost feature importance ({importance_label})\n{model_label}")
        ax.set_xlabel(importance_label)
        fig.tight_layout()

        filename = f"xgboost_4_{model_label}_{importance_type}.png"
        output_path = os.path.join(PLOTS_DIR, filename)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Saved feature importance plot -> {output_path}")


def build_base_dataframe() -> pd.DataFrame:
    """
    Build match, rolling, and precomputed map-strength features.
    """
    con = duckdb.connect()

    df = con.execute(
        f"""
        SELECT
            hltv_match_id,
            TRY_CAST(date AS TIMESTAMP) AS started_at,
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
        rolling_df["team1_rolling_swing_l15"]
        - rolling_df["team2_rolling_swing_l15"]
    )
    df = df.merge(rolling_df, on="hltv_match_id", how="left")

    if not os.path.exists(MAP_STRENGTH_FILE):
        raise FileNotFoundError(
            f"Missing map-strength features: {MAP_STRENGTH_FILE}. "
            "Run model/feature_generation/features_map_strength.py first."
        )
    map_strength_df = pd.read_parquet(MAP_STRENGTH_FILE)
    required_map_cols = {"hltv_match_id", *MAP_FEATURES}
    missing_map_cols = required_map_cols - set(map_strength_df.columns)
    if missing_map_cols:
        raise ValueError(
            f"{os.path.basename(MAP_STRENGTH_FILE)} is missing required columns: "
            f"{sorted(missing_map_cols)}"
        )
    if map_strength_df["hltv_match_id"].duplicated().any():
        raise ValueError(
            f"{os.path.basename(MAP_STRENGTH_FILE)} contains duplicate match IDs."
        )
    df = df.merge(
        map_strength_df[["hltv_match_id", *MAP_FEATURES]],
        on="hltv_match_id",
        how="left",
        validate="one_to_one",
    )

    return df


def run_single_trueskill_model(
    base_df: pd.DataFrame, ts_path: str
) -> Dict[str, Any]:
    """
    Run an XGBoost model for a single roster-TrueSkill feature file.
    Uses the two lineups' conservative TrueSkill ratings as features.
    Returns a dict of metrics and metadata about this run.
    """
    ts_df = pd.read_parquet(ts_path)

    # Only require conservative ratings for each team
    required_cols = {
        "hltv_match_id",
        "team1_roster_trueskill_conservative",
        "team2_roster_trueskill_conservative",
    }
    missing = required_cols - set(ts_df.columns)
    if missing:
        raise ValueError(
            f"{os.path.basename(ts_path)} is missing required columns: {sorted(missing)}"
        )

    df = base_df.merge(
        ts_df[
            [
                "hltv_match_id",
                "team1_roster_trueskill_conservative",
                "team2_roster_trueskill_conservative",
            ]
        ],
        on="hltv_match_id",
        how="left",
    )

    feature_cols = [
        "is_lan",
        "is_bo1",
        "is_bo3",
        "is_bo5",
        "kast_diff_l15",
        "swing_diff_l15",
        "team1_rolling_win_rate_l15",
        "team2_rolling_win_rate_l15",
        "team1_roster_trueskill_conservative",
        "team2_roster_trueskill_conservative",
        *MAP_FEATURES,
    ]
    # Keep debuts and rebuilt teams; only exclude matches with missing ratings.
    filtered_df = df.dropna(
        subset=[
            "team1_roster_trueskill_conservative",
            "team2_roster_trueskill_conservative",
            *MAP_FEATURES,
        ]
    ).reset_index(drop=True)

    train_df, validation_df, _ = split_chronologically(filtered_df)

    scaler = StandardScaler()
    train_inputs = scaler.fit_transform(train_df[feature_cols])
    validation_inputs = scaler.transform(validation_df[feature_cols])
    train_targets = train_df["result"].astype(int)
    validation_targets = validation_df["result"].astype(int)

    dtrain = xgb.DMatrix(train_inputs, label=train_targets, feature_names=feature_cols)
    dvalidation = xgb.DMatrix(
        validation_inputs, label=validation_targets, feature_names=feature_cols
    )

    bst = xgb.train(
        XGB_PARAMS,
        dtrain,
        num_boost_round=MAX_BOOST_ROUNDS,
        evals=[(dtrain, "train"), (dvalidation, "validation")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )

    train_probs = predict_at_best_iteration(bst, dtrain)
    validation_probs = predict_at_best_iteration(bst, dvalidation)

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

    fname = os.path.basename(ts_path)
    m = re.match(
        r"features_roster_trueskill_([0-9.]+)_([0-9.]+)_([0-9.]+)\.parquet",
        fname,
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
        "ll_delta": ll_delta,
        "roc_delta": roc_delta,
        "br_delta": br_delta,
        "overfit": overfit,
        "train_matches": len(train_df),
        "validation_matches": len(validation_df),
        "best_rounds": best_boost_rounds(bst),
    }

    print(
        f"\nRun {fname}: "
        f"Train LL={train_ll:.4f}, ROC={train_roc:.4f}, Brier={train_brier:.4f} | "
        f"Validation LL={validation_ll:.4f}, ROC={validation_roc:.4f}, "
        f"Brier={validation_brier:.4f} | "
        f"dLL={ll_delta:+.4f}, dROC={roc_delta:+.4f}, dBrier={br_delta:+.4f}, "
        f"overfit={overfit}"
    )

    return result


def evaluate_selected_on_test(
    base_df: pd.DataFrame,
    ts_path: str,
    boost_rounds: int,
) -> Dict[str, Any]:
    """Refit the selected candidate on train+validation and evaluate test once."""
    ts_df = pd.read_parquet(ts_path)
    df = base_df.merge(
        ts_df[
            [
                "hltv_match_id",
                "team1_roster_trueskill_conservative",
                "team2_roster_trueskill_conservative",
            ]
        ],
        on="hltv_match_id",
        how="left",
    )
    feature_cols = [
        "is_lan",
        "is_bo1",
        "is_bo3",
        "is_bo5",
        "kast_diff_l15",
        "swing_diff_l15",
        "team1_rolling_win_rate_l15",
        "team2_rolling_win_rate_l15",
        "team1_roster_trueskill_conservative",
        "team2_roster_trueskill_conservative",
        *MAP_FEATURES,
    ]
    filtered_df = df.dropna(
        subset=[
            "team1_roster_trueskill_conservative",
            "team2_roster_trueskill_conservative",
            *MAP_FEATURES,
        ]
    ).reset_index(drop=True)
    train_df, validation_df, test_df = split_chronologically(filtered_df)
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

    bst = xgb.train(
        XGB_PARAMS,
        dtrain_validation,
        num_boost_round=boost_rounds,
        verbose_eval=False,
    )
    model_label = os.path.splitext(os.path.basename(ts_path))[0].replace(
        "features_roster_trueskill_", "roster_trueskill_"
    )
    save_feature_importance_plots(bst, feature_cols, model_label)
    test_probs = bst.predict(dtest)

    result = {
        "test_log_loss": log_loss(test_targets, test_probs),
        "test_roc_auc": roc_auc_score(test_targets, test_probs),
        "test_brier": brier_score_loss(test_targets, test_probs),
        "test_matches": len(test_df),
    }
    print(
        f"Test: LL={result['test_log_loss']:.4f}, "
        f"ROC={result['test_roc_auc']:.4f}, "
        f"Brier={result['test_brier']:.4f}, "
        f"matches={result['test_matches']}",
        flush=True,
    )
    return result


def main() -> None:
    base_df = build_base_dataframe()

    if not os.path.isdir(ROSTER_TRUESKILL_DIR):
        print(f"ROSTER_TRUESKILL_DIR does not exist: {ROSTER_TRUESKILL_DIR}")
        return

    ts_files: List[str] = sorted(
        os.path.join(ROSTER_TRUESKILL_DIR, f)
        for f in os.listdir(ROSTER_TRUESKILL_DIR)
        if f.startswith("features_roster_trueskill_") and f.endswith(".parquet")
    )

    if not ts_files:
        print(
            "No features_roster_trueskill_*.parquet files found in "
            f"{ROSTER_TRUESKILL_DIR}"
        )
        return

    print(f"Found {len(ts_files)} roster TrueSkill feature files.")

    all_results: List[Dict[str, Any]] = []

    for i, path in enumerate(ts_files, start=1):
        fname = os.path.basename(path)
        try:
            res = run_single_trueskill_model(base_df, path)
            all_results.append(res)
            print(f"=== [{i}/{len(ts_files)}] Done {fname} ===")
        except Exception as e:
            print(f"❌ Failed for {fname}: {e}")

    if not all_results:
        print("No successful runs, nothing to summarize.")
        return

    results_df = pd.DataFrame(all_results)

    # Rank candidates using validation only; the test set stays untouched.
    results_df["validation_score"] = (
        results_df["validation_log_loss"]
        + results_df["validation_brier"]
        - results_df["validation_roc_auc"]
    )

    # Sort by:
    # 1) overfit (False first)
    # 2) lowest validation score
    # 3) lowest validation log loss
    # 4) highest validation ROC AUC
    results_df = results_df.sort_values(
        by=[
            "overfit",
            "validation_score",
            "validation_log_loss",
            "validation_roc_auc",
        ],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)

    print("\n=== Roster TrueSkill candidates ranked by validation metrics ===")
    display_cols = [
        "sigma",
        "beta",
        "tau",
        "train_log_loss",
        "validation_log_loss",
        "train_roc_auc",
        "validation_roc_auc",
        "train_brier",
        "validation_brier",
        "ll_delta",
        "roc_delta",
        "br_delta",
        "validation_score",
        "overfit",
        "train_matches",
        "validation_matches",
    ]
    print(
        results_df[display_cols].to_string(
            index=False, float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\nTop 10 roster TrueSkill combinations without overfit:")
    top_without_overfit = results_df[~results_df["overfit"]].head(10)
    if top_without_overfit.empty:
        print("No combinations stayed within the overfitting constraints.")
    else:
        print(
            top_without_overfit[display_cols].to_string(
                index=False, float_format=lambda x: f"{x:.4f}"
            )
        )

    selected = results_df.iloc[0]
    selected_path = os.path.join(ROSTER_TRUESKILL_DIR, selected["file"])
    print(
        f"\n=== Final held-out test result for selected candidate "
        f"(sigma={selected['sigma']}, beta={selected['beta']}, "
        f"tau={selected['tau']}; {int(selected['best_rounds'])} boosting rounds) ===",
        flush=True,
    )
    evaluate_selected_on_test(
        base_df,
        selected_path,
        int(selected["best_rounds"]),
    )


if __name__ == "__main__":
    main()
