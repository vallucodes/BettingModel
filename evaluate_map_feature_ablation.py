"""Compare v1/v3 map-strength features with a fixed chronological evaluation."""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import xgboost_4 as model


MAP_FILES = {
    "v1": os.path.join(
        model.FEATURES_ROOT, "features_map_strength_joint_v1.parquet"
    ),
    "v3": os.path.join(
        model.FEATURES_ROOT, "features_map_strength_joint_v3.parquet"
    ),
}
BASE_FEATURES = [
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
]
TRUESKILL_FEATURES = [
    "team1_roster_trueskill_conservative",
    "team2_roster_trueskill_conservative",
]


def read_map_features(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Map-strength parquet not found: {path}")
    frame = pd.read_parquet(path)
    required = {"hltv_match_id", *model.MAP_FEATURES}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame["hltv_match_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate match IDs.")
    return frame[["hltv_match_id", *model.MAP_FEATURES]]


def attach_map_features(
    base_without_map: pd.DataFrame, map_features: pd.DataFrame
) -> pd.DataFrame:
    return base_without_map.merge(
        map_features,
        on="hltv_match_id",
        how="left",
        validate="one_to_one",
    )


def select_v1_candidate(
    v1_frame: pd.DataFrame, trueskill_paths: Sequence[Path]
) -> tuple[str, int]:
    results = [
        model.run_single_trueskill_model(v1_frame, str(path))
        for path in trueskill_paths
    ]
    if not results:
        raise RuntimeError("No roster-TrueSkill parquet files were found.")

    ranked = pd.DataFrame(results)
    ranked["validation_score"] = (
        ranked["validation_log_loss"]
        + ranked["validation_brier"]
        - ranked["validation_roc_auc"]
    )
    ranked = ranked.sort_values(
        by=[
            "overfit",
            "validation_score",
            "validation_log_loss",
            "validation_roc_auc",
        ],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)
    selected = ranked.iloc[0]
    return str(selected["file"]), int(selected["best_rounds"])


def prepare_version_frame(
    base_without_map: pd.DataFrame,
    map_features: pd.DataFrame,
    trueskill: pd.DataFrame,
) -> pd.DataFrame:
    frame = attach_map_features(base_without_map, map_features).merge(
        trueskill,
        on="hltv_match_id",
        how="left",
        validate="one_to_one",
    )
    frame = frame.dropna(
        subset=[*TRUESKILL_FEATURES, *model.MAP_FEATURES]
    )
    return frame.sort_values(
        ["started_at", "hltv_match_id"], na_position="last"
    ).reset_index(drop=True)


def assert_same_splits(
    splits_by_version: Dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]
) -> None:
    reference = splits_by_version["v1"]
    for version in ("v3",):
        candidate = splits_by_version[version]
        for label, expected, actual in zip(
            ("train", "validation", "test"), reference, candidate
        ):
            if not np.array_equal(
                expected["hltv_match_id"].to_numpy(),
                actual["hltv_match_id"].to_numpy(),
            ):
                raise ValueError(
                    f"{version} {label} IDs differ from v1; comparison is not paired."
                )


def score(y: np.ndarray, probabilities: np.ndarray) -> Dict[str, float]:
    return {
        "log_loss": float(model.log_loss(y, probabilities)),
        "brier": float(model.brier_score_loss(y, probabilities)),
        "roc_auc": float(model.roc_auc_score(y, probabilities)),
    }


def train_and_score(
    splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    feature_columns: List[str],
    rounds: int,
) -> tuple[Dict[str, float], Dict[str, float], np.ndarray]:
    train_df, validation_df, test_df = splits
    train_y = train_df["result"].astype(int).to_numpy()
    validation_y = validation_df["result"].astype(int).to_numpy()
    test_y = test_df["result"].astype(int).to_numpy()

    validation_scaler = StandardScaler()
    train_inputs = validation_scaler.fit_transform(train_df[feature_columns])
    validation_inputs = validation_scaler.transform(
        validation_df[feature_columns]
    )
    validation_model = model.xgb.train(
        model.XGB_PARAMS,
        model.xgb.DMatrix(
            train_inputs, label=train_y, feature_names=feature_columns
        ),
        num_boost_round=rounds,
        verbose_eval=False,
    )
    validation_probabilities = validation_model.predict(
        model.xgb.DMatrix(
            validation_inputs,
            label=validation_y,
            feature_names=feature_columns,
        )
    )

    train_validation_df = pd.concat(
        [train_df, validation_df], ignore_index=True
    )
    train_validation_y = train_validation_df["result"].astype(int).to_numpy()
    test_scaler = StandardScaler()
    train_validation_inputs = test_scaler.fit_transform(
        train_validation_df[feature_columns]
    )
    test_inputs = test_scaler.transform(test_df[feature_columns])
    test_model = model.xgb.train(
        model.XGB_PARAMS,
        model.xgb.DMatrix(
            train_validation_inputs,
            label=train_validation_y,
            feature_names=feature_columns,
        ),
        num_boost_round=rounds,
        verbose_eval=False,
    )
    test_probabilities = test_model.predict(
        model.xgb.DMatrix(
            test_inputs, label=test_y, feature_names=feature_columns
        )
    )
    return (
        score(validation_y, validation_probabilities),
        score(test_y, test_probabilities),
        test_probabilities,
    )


def stratified_bootstrap_indices(
    labels: np.ndarray, repetitions: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    samples = np.empty((repetitions, len(labels)), dtype=np.int32)
    for row in range(repetitions):
        samples[row] = np.concatenate(
            [
                rng.choice(positive, size=len(positive), replace=True),
                rng.choice(negative, size=len(negative), replace=True),
            ]
        )
    return samples


def paired_delta_intervals(
    labels: np.ndarray,
    prediction: np.ndarray,
    baseline_prediction: np.ndarray,
    bootstrap_samples: np.ndarray,
) -> Dict[str, float]:
    epsilon = 1e-8
    prediction = np.clip(prediction, epsilon, 1.0 - epsilon)
    baseline_prediction = np.clip(
        baseline_prediction, epsilon, 1.0 - epsilon
    )
    per_row_logloss = (
        -labels * np.log(prediction)
        - (1 - labels) * np.log(1 - prediction)
        + labels * np.log(baseline_prediction)
        + (1 - labels) * np.log(1 - baseline_prediction)
    )
    per_row_brier = (
        (prediction - labels) ** 2 - (baseline_prediction - labels) ** 2
    )
    deltas = {"log_loss": [], "brier": [], "roc_auc": []}
    for indices in bootstrap_samples:
        sample_labels = labels[indices]
        sample_prediction = prediction[indices]
        sample_baseline = baseline_prediction[indices]
        deltas["log_loss"].append(float(per_row_logloss[indices].mean()))
        deltas["brier"].append(float(per_row_brier[indices].mean()))
        deltas["roc_auc"].append(
            float(
                model.roc_auc_score(sample_labels, sample_prediction)
                - model.roc_auc_score(sample_labels, sample_baseline)
            )
        )

    intervals: Dict[str, float] = {}
    for metric, values in deltas.items():
        low, high = np.quantile(values, [0.025, 0.975])
        intervals[f"delta_{metric}_ci_low"] = float(low)
        intervals[f"delta_{metric}_ci_high"] = float(high)
    return intervals


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run paired v1/v3 map-feature ablations."
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--bootstrap-reps", type=int, default=300)
    parser.add_argument(
        "--output-dir",
        default=os.path.join(model.MODEL_DIR, "map_feature_ablation"),
    )
    args = parser.parse_args()
    if args.bootstrap_reps < 1:
        raise ValueError("--bootstrap-reps must be positive.")

    model.XGB_PARAMS = {**model.XGB_PARAMS, "seed": args.seed}
    original_map_file = model.MAP_STRENGTH_FILE
    try:
        model.MAP_STRENGTH_FILE = MAP_FILES["v1"]
        base_with_v1 = model.build_base_dataframe()
    finally:
        model.MAP_STRENGTH_FILE = original_map_file
    base_without_map = base_with_v1.drop(columns=model.MAP_FEATURES)

    v1_features = read_map_features(MAP_FILES["v1"])
    v3_features = read_map_features(MAP_FILES["v3"])
    v1_frame = attach_map_features(base_without_map, v1_features)
    trueskill_dir = Path(model.ROSTER_TRUESKILL_DIR)
    trueskill_paths = sorted(
        trueskill_dir.glob("features_roster_trueskill_*.parquet")
    )
    selected_file, rounds = select_v1_candidate(v1_frame, trueskill_paths)
    trueskill_path = trueskill_dir / selected_file
    trueskill = pd.read_parquet(trueskill_path)[
        ["hltv_match_id", *TRUESKILL_FEATURES]
    ]
    print(
        f"Fixed candidate from v1 validation: {selected_file}; "
        f"boosting rounds={rounds}; seed={args.seed}",
        flush=True,
    )

    frames = {
        "v1": prepare_version_frame(
            base_without_map, v1_features, trueskill
        ),
        "v3": prepare_version_frame(
            base_without_map, v3_features, trueskill
        ),
    }
    splits_by_version = {
        version: model.split_chronologically(frame)
        for version, frame in frames.items()
    }
    assert_same_splits(splits_by_version)
    train_df, validation_df, test_df = splits_by_version["v1"]
    labels = test_df["result"].astype(int).to_numpy()
    bootstrap_samples = stratified_bootstrap_indices(
        labels, args.bootstrap_reps, args.seed + 1
    )

    specifications = []
    for version in ("v1", "v3"):
        specifications.append((f"{version}_full", version, list(model.MAP_FEATURES)))
        specifications.append((f"{version}_no_map_features", version, []))
        for omitted in model.MAP_FEATURES:
            included = [
                feature for feature in model.MAP_FEATURES if feature != omitted
            ]
            specifications.append(
                (f"{version}_drop_{omitted}", version, included)
            )

    summaries = []
    predictions: Dict[str, np.ndarray] = {}
    for variant, version, map_columns in specifications:
        splits = splits_by_version[version]
        feature_columns = [*BASE_FEATURES, *map_columns]
        validation_scores, test_scores, test_probabilities = train_and_score(
            splits, feature_columns, rounds
        )
        predictions[variant] = test_probabilities
        summaries.append(
            {
                "variant": variant,
                "map_version": version,
                "dropped_features": ",".join(
                    feature
                    for feature in model.MAP_FEATURES
                    if feature not in map_columns
                ),
                "trueskill_file": selected_file,
                "seed": args.seed,
                "rounds": rounds,
                "validation_log_loss": validation_scores["log_loss"],
                "validation_brier": validation_scores["brier"],
                "validation_roc_auc": validation_scores["roc_auc"],
                "test_log_loss": test_scores["log_loss"],
                "test_brier": test_scores["brier"],
                "test_roc_auc": test_scores["roc_auc"],
                "test_matches": len(test_df),
            }
        )
        print(
            f"{variant}: test LL={test_scores['log_loss']:.4f}, "
            f"Brier={test_scores['brier']:.4f}, "
            f"ROC={test_scores['roc_auc']:.4f}",
            flush=True,
        )

    summary = pd.DataFrame(summaries)
    for version in ("v1", "v3"):
        other_version = "v3" if version == "v1" else "v1"
        for baseline_version, suffix in (
            (version, ""),
            (other_version, "_vs_other_full"),
        ):
            baseline_name = f"{baseline_version}_full"
            baseline = summary.loc[summary["variant"] == baseline_name].iloc[0]
            baseline_prediction = predictions[baseline_name]
            for index, row in summary[
                summary["map_version"] == version
            ].iterrows():
                prefix = "delta_test"
                if suffix:
                    prefix = f"delta_test{suffix}"
                for metric in ("log_loss", "brier", "roc_auc"):
                    summary.loc[index, f"{prefix}_{metric}"] = (
                        row[f"test_{metric}"] - baseline[f"test_{metric}"]
                    )
                intervals = paired_delta_intervals(
                    labels,
                    predictions[row["variant"]],
                    baseline_prediction,
                    bootstrap_samples,
                )
                for metric in ("log_loss", "brier", "roc_auc"):
                    interval_prefix = f"delta_test_{metric}{suffix}"
                    summary.loc[
                        index, f"{interval_prefix}_ci_low"
                    ] = intervals[f"delta_{metric}_ci_low"]
                    summary.loc[
                        index, f"{interval_prefix}_ci_high"
                    ] = intervals[f"delta_{metric}_ci_high"]

    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "summary.csv")
    predictions_path = os.path.join(args.output_dir, "test_predictions.csv")
    summary.to_csv(summary_path, index=False)
    prediction_frame = pd.DataFrame(
        {
            "hltv_match_id": test_df["hltv_match_id"].to_numpy(),
            "result": labels,
            **{f"p_{name}": values for name, values in predictions.items()},
        }
    )
    prediction_frame.to_csv(predictions_path, index=False)
    print(f"Saved {summary_path}", flush=True)
    print(f"Saved {predictions_path}", flush=True)


if __name__ == "__main__":
    main()
