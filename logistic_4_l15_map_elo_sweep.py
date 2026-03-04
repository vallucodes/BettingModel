import os
import re
from typing import Dict, Any, List
from concurrent.futures import ProcessPoolExecutor, as_completed

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.model_selection import train_test_split

PARQUET_DIR = "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/parquet"
FEATURES_DIR = "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/features"

# Overfitting thresholds
MAX_LL_DELTA = 0.05   # test_log_loss - train_log_loss
MAX_ROC_DELTA = 0.02  # train_roc_auc - test_roc_auc
MAX_BR_DELTA = 0.01   # test_brier - train_brier


def build_base_dataframe() -> pd.DataFrame:
    """
    Build the part of the dataframe that is common for all runs:
    - match-level data
    - simple team Elo
    - rolling l15 features
    """
    con = duckdb.connect()

    df = con.execute(
        f"""
        SELECT hltv_match_id, team1_name, team2_name, team1_score, team2_score, event_type
        FROM '{PARQUET_DIR}/matches.parquet'
        ORDER BY hltv_match_id ASC
    """
    ).df()

    df["result"] = np.where(df["team1_score"] > df["team2_score"], 0, 1)
    df["is_lan"] = (df["event_type"] == "LAN").astype(int)

    # Simple team-level Elo
    HIGH_K = 100
    LOW_K = 20
    THRESHOLD = 30

    df["team1_elo"] = None
    df["team2_elo"] = None
    df["elo_diff"] = None
    df["team1_games"] = None
    df["team2_games"] = None

    unique_teams = pd.unique(df[["team1_name", "team2_name"]].values.ravel())
    elo = {team: {"elo": 1500.0, "games_played": 0} for team in unique_teams}

    def expected_score(elo_1: float, elo_2: float) -> float:
        return 1.0 / (1.0 + 10 ** ((elo_2 - elo_1) / 400.0))

    def update_elo(elo_1: float, elo_2: float, res: int, k1: float, k2: float) -> tuple[float, float]:
        ea = expected_score(elo_1, elo_2)
        elo_1_new = elo_1 + k1 * ((1 - res) - ea)
        elo_2_new = elo_2 + k2 * (res - (1 - ea))
        return elo_1_new, elo_2_new

    for idx, row in df.iterrows():
        team1, team2 = row["team1_name"], row["team2_name"]
        df.at[idx, "team1_elo"] = elo[team1]["elo"]
        df.at[idx, "team2_elo"] = elo[team2]["elo"]
        df.at[idx, "elo_diff"] = elo[team1]["elo"] - elo[team2]["elo"]
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
    rolling_df = pd.read_parquet(os.path.join(FEATURES_DIR, "features_rolling_l15.parquet"))[
        [
            "hltv_match_id",
            "team1_rolling_rating_l15",
            "team2_rolling_rating_l15",
            "team1_rolling_kast_l15",
            "team2_rolling_kast_l15",
            "team1_rolling_swing_l15",
            "team2_rolling_swing_l15",
            "team1_rolling_win_rate_l15",
            "team2_rolling_win_rate_l15",
        ]
    ]

    rolling_df["rating_diff_l15"] = (
        rolling_df["team1_rolling_rating_l15"] - rolling_df["team2_rolling_rating_l15"]
    )
    rolling_df["kast_diff_l15"] = (
        rolling_df["team1_rolling_kast_l15"] - rolling_df["team2_rolling_kast_l15"]
    )
    rolling_df["swing_diff_l15"] = (
        rolling_df["team1_rolling_swing_l15"] - rolling_df["team2_rolling_swing_l15"]
    )

    df = df.merge(rolling_df, on="hltv_match_id", how="left")

    return df


def run_single_map_elo_model(base_df: pd.DataFrame, map_elo_path: str) -> Dict[str, Any]:
    """
    Run the logistic_4_l15-style model for a single features_map_elo_XX_XX_XX.parquet file.
    Returns a dict of metrics and metadata about this run.
    """
    map_elo_df = pd.read_parquet(map_elo_path)

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
        raise ValueError(f"{os.path.basename(map_elo_path)} is missing expected columns: {missing_cols}")

    df = base_df.merge(
        map_elo_df[["hltv_match_id"] + map_elo_cols],
        on="hltv_match_id",
        how="left",
    )

    filtered_df = df[
        (df["team1_games"] > 10)
        & (df["team2_games"] > 10)
        & (df["team1_rolling_rating_l15"].notna())
        & (df["team2_rolling_rating_l15"].notna())
    ].reset_index(drop=True)

    feature_cols = [
        "elo_diff",
        "rating_diff_l15",
        "kast_diff_l15",
        "swing_diff_l15",
        "team1_rolling_win_rate_l15",
        "team2_rolling_win_rate_l15",
        "is_lan",
    ] + map_elo_cols

    train_df, test_df = train_test_split(filtered_df, test_size=0.2, shuffle=False)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_inputs = train_df[feature_cols]
    test_inputs = test_df[feature_cols]
    train_targets = train_df["result"]
    test_targets = test_df["result"]

    model = LogisticRegression(solver="liblinear")
    model.fit(train_inputs, train_targets)

    # Coefficient-based feature importance for this run (printed summary, no plots to avoid spam)
    coef = model.coef_[0]
    coef_df = pd.DataFrame({"feature": feature_cols, "coef": coef})
    coef_df["abs_coef"] = coef_df["coef"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False)

    top_n = min(10, len(coef_df))
    print(f"\nTop {top_n} features by |coef| for {os.path.basename(map_elo_path)}:")
    print(coef_df.head(top_n).to_string(index=False, float_format="{:.4f}".format))

    def eval_split(inputs, targets):
        probs = model.predict_proba(inputs)[:, 1]
        return {
            "log_loss": log_loss(targets, probs),
            "roc_auc": roc_auc_score(targets, probs),
            "brier": brier_score_loss(targets, probs),
        }

    train_metrics = eval_split(train_inputs, train_targets)
    test_metrics = eval_split(test_inputs, test_targets)

    ll_delta = test_metrics["log_loss"] - train_metrics["log_loss"]
    roc_delta = train_metrics["roc_auc"] - test_metrics["roc_auc"]
    br_delta = test_metrics["brier"] - train_metrics["brier"]

    within_overfit = (
        ll_delta <= MAX_LL_DELTA
        and roc_delta <= MAX_ROC_DELTA
        and br_delta <= MAX_BR_DELTA
    )

    fname = os.path.basename(map_elo_path)
    m = re.match(r"features_map_elo_(\d+)_(\d+)_(\d+)\.parquet", fname)
    high_k = int(m.group(1)) if m else None
    low_k = int(m.group(2)) if m else None
    threshold = int(m.group(3)) if m else None

    result: Dict[str, Any] = {
        "file": fname,
        "high_k": high_k,
        "low_k": low_k,
        "threshold": threshold,
        "train_log_loss": train_metrics["log_loss"],
        "train_roc_auc": train_metrics["roc_auc"],
        "train_brier": train_metrics["brier"],
        "test_log_loss": test_metrics["log_loss"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_brier": test_metrics["brier"],
        "ll_delta": ll_delta,
        "roc_delta": roc_delta,
        "br_delta": br_delta,
        "within_overfit_limits": within_overfit,
        "train_matches": len(train_df),
        "test_matches": len(test_df),
    }
    return result


def main() -> None:
    base_df = build_base_dataframe()

    # Locate all features_map_elo_XX_XX_XX.parquet files
    map_elo_files: List[str] = sorted(
        os.path.join(FEATURES_DIR, f)
        for f in os.listdir(FEATURES_DIR)
        if f.startswith("features_map_elo_") and f.endswith(".parquet")
    )

    if not map_elo_files:
        print(f"No features_map_elo_*.parquet files found in {FEATURES_DIR}")
        return

    print(f"Found {len(map_elo_files)} map ELO feature files.")

    # Use multiple processes but always leave one core free.
    cpu_count = os.cpu_count() or 4
    max_workers = max(1, cpu_count - 1)
    print(f"Running models in parallel with up to {max_workers} workers.")

    all_results: List[Dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(run_single_map_elo_model, base_df, path): path
            for path in map_elo_files
        }

        for i, future in enumerate(as_completed(future_to_file), start=1):
            path = future_to_file[future]
            fname = os.path.basename(path)
            try:
                res = future.result()
                all_results.append(res)
                print(
                    f"\n=== [{i}/{len(map_elo_files)}] Done {fname} ===\n"
                    f"Train LL={res['train_log_loss']:.4f}, ROC={res['train_roc_auc']:.4f}, "
                    f"Brier={res['train_brier']:.4f}\n"
                    f"Test  LL={res['test_log_loss']:.4f}, ROC={res['test_roc_auc']:.4f}, "
                    f"Brier={res['test_brier']:.4f}\n"
                    f"dLL={res['ll_delta']:+.4f}, dROC={res['roc_delta']:+.4f}, "
                    f"dBrier={res['br_delta']:+.4f}, "
                    f"within_overfit_limits={res['within_overfit_limits']}"
                )
            except Exception as e:
                print(f"❌ Failed for {fname}: {e}")

    if not all_results:
        print("No successful runs, nothing to summarize.")
        return

    results_df = pd.DataFrame(all_results)

    # Competition-style ranking: favor low loss/Brier and high ROC.
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

    print("\n=== Summary of all runs (best first by competition score) ===")
    display_cols = [
        "high_k",
        "low_k",
        "threshold",
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
    ]
    print(results_df[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nTop 10 combinations within overfitting limits (best competition scores):")
    top_within = results_df[results_df["within_overfit_limits"]].head(10)
    if top_within.empty:
        print("No combinations satisfied the overfitting constraints.")
    else:
        print(top_within[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()

