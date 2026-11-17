"""Generate pre-match, per-map TrueSkill features over a parameter grid.

Reads matches.parquet and maps.parquet from vallu_scraper/data/parquet, then
writes one features_map_trueskill_<sigma>_<beta>_<tau>.parquet file per
configured parameter combination to vallu_scraper/data/features/map_trueskill.

Usage, from the project root:
    python model/feature_generation/features_map_trueskill_multiple.py

The parameter grids are defined in main(); no arguments are required.
Existing combinations are regenerated, and obsolete map-TrueSkill files are
removed after a successful sweep.
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
from trueskill import Rating, TrueSkill

from feature_output_cleanup import remove_stale_outputs


PARQUET_DIR = (
    "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/parquet"
)
FEATURES_DIR = (
    "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/features"
)
MAP_TRUESKILL_DIR = os.path.join(FEATURES_DIR, "map_trueskill")
os.makedirs(MAP_TRUESKILL_DIR, exist_ok=True)

START_MU = 25.0
MAPS = [
    "ancient",
    "anubis",
    "cache",
    "dust2",
    "inferno",
    "mirage",
    "nuke",
    "overpass",
    "train",
    "vertigo",
]

_WORKER_MATCHES: Optional[pd.DataFrame] = None
_WORKER_MAP_RESULTS: Optional[pd.DataFrame] = None


def load_map_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load chronologically ordered matches and their individual map results."""
    con = duckdb.connect()
    try:
        matches_df = con.execute(
            f"""
            SELECT hltv_match_id, date AS started_at, team1_name, team2_name
            FROM '{PARQUET_DIR}/matches.parquet'
            ORDER BY started_at ASC NULLS LAST, hltv_match_id ASC
        """
        ).df()
        map_results = con.execute(
            f"""
            SELECT
                mp.hltv_match_id,
                m.date AS started_at,
                mp.map_name,
                mp.score,
                mp.map_number
            FROM '{PARQUET_DIR}/maps.parquet' mp
            JOIN '{PARQUET_DIR}/matches.parquet' m
                ON mp.hltv_match_id = m.hltv_match_id
            ORDER BY m.date ASC NULLS LAST,
                     mp.hltv_match_id ASC,
                     mp.map_number ASC
        """
        ).df()
    finally:
        con.close()

    map_results["map_name"] = (
        map_results["map_name"].fillna("").astype(str).str.lower().str.strip()
    )
    return matches_df, map_results


def parse_map_score(score: object) -> Optional[Tuple[int, int]]:
    """Parse a team1:team2 map score, returning None for invalid scores."""
    parts = str(score).strip().split(":")
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def compute_map_trueskill_features(
    matches_df: pd.DataFrame,
    map_results: pd.DataFrame,
    sigma: float,
    beta: float,
    tau: float,
) -> pd.DataFrame:
    """Compute per-map TrueSkill ratings immediately before each match."""
    matches_df = matches_df.sort_values(
        ["started_at", "hltv_match_id"], na_position="last"
    ).reset_index(drop=True)
    map_results = map_results.sort_values(
        ["started_at", "hltv_match_id", "map_number"], na_position="last"
    ).reset_index(drop=True)
    maps_by_match = {
        match_id: rows
        for match_id, rows in map_results.groupby("hltv_match_id", sort=False)
    }

    env = TrueSkill(
        mu=START_MU,
        sigma=sigma,
        beta=beta,
        tau=tau,
        draw_probability=0.0,
    )
    unique_teams = pd.unique(
        matches_df[["team1_name", "team2_name"]].values.ravel()
    )
    rating_state: dict[str, dict[str, Rating]] = {
        team: {map_name: env.create_rating() for map_name in MAPS}
        for team in unique_teams
    }

    feature_values: dict[str, np.ndarray] = {}
    for map_name in MAPS:
        for metric in ("mu", "sigma", "conservative"):
            for team_number in (1, 2):
                feature_values[
                    f"team{team_number}_{map_name}_trueskill_{metric}"
                ] = np.empty(len(matches_df), dtype=float)
            feature_values[
                f"{map_name}_trueskill_{metric}_diff"
            ] = np.empty(len(matches_df), dtype=float)

    for index, row in enumerate(matches_df.itertuples(index=False)):
        team1 = row.team1_name
        team2 = row.team2_name

        # Snapshot every map before processing any maps played in this match.
        for map_name in MAPS:
            rating1 = rating_state[team1][map_name]
            rating2 = rating_state[team2][map_name]
            values1 = {
                "mu": rating1.mu,
                "sigma": rating1.sigma,
                "conservative": rating1.mu - 3.0 * rating1.sigma,
            }
            values2 = {
                "mu": rating2.mu,
                "sigma": rating2.sigma,
                "conservative": rating2.mu - 3.0 * rating2.sigma,
            }
            for metric in values1:
                feature_values[
                    f"team1_{map_name}_trueskill_{metric}"
                ][index] = values1[metric]
                feature_values[
                    f"team2_{map_name}_trueskill_{metric}"
                ][index] = values2[metric]
                feature_values[
                    f"{map_name}_trueskill_{metric}_diff"
                ][index] = values1[metric] - values2[metric]

        match_maps = maps_by_match.get(row.hltv_match_id)
        if match_maps is None:
            continue

        for map_result in match_maps.itertuples(index=False):
            map_name = map_result.map_name
            if map_name not in MAPS:
                continue

            score = parse_map_score(map_result.score)
            if score is None or score[0] == score[1]:
                continue

            ranks = [0, 1] if score[0] > score[1] else [1, 0]
            (new_rating1,), (new_rating2,) = env.rate(
                [
                    (rating_state[team1][map_name],),
                    (rating_state[team2][map_name],),
                ],
                ranks=ranks,
            )
            rating_state[team1][map_name] = new_rating1
            rating_state[team2][map_name] = new_rating2

    out_df = matches_df[["hltv_match_id", "started_at"]].copy()
    for column, values in feature_values.items():
        out_df[column] = values
    return out_df


def initialize_worker(
    matches_df: pd.DataFrame, map_results: pd.DataFrame
) -> None:
    """Share the loaded input tables with each sweep worker."""
    global _WORKER_MATCHES, _WORKER_MAP_RESULTS
    _WORKER_MATCHES = matches_df
    _WORKER_MAP_RESULTS = map_results


def build_map_trueskill_features(
    sigma: float, beta: float, tau: float
) -> None:
    """Build and save one map-TrueSkill feature file for a parameter set."""
    if _WORKER_MATCHES is None or _WORKER_MAP_RESULTS is None:
        raise RuntimeError("Map TrueSkill worker data was not initialized.")

    filename = (
        f"features_map_trueskill_{sigma:.3f}_{beta:.3f}_{tau:.3f}.parquet"
    )
    output_path = os.path.join(MAP_TRUESKILL_DIR, filename)
    out_df = compute_map_trueskill_features(
        _WORKER_MATCHES, _WORKER_MAP_RESULTS, sigma, beta, tau
    )
    out_df.to_parquet(output_path, index=False)
    print(f"Saved {len(out_df)} rows → {output_path}", flush=True)


def main() -> None:
    """Sweep over the same TrueSkill parameter grid as the team generator."""
    sigma_values = [4.0, 6.0, 8.333, 10.0, 12.0, 15.0]
    beta_values = [1.0, 2.0, 4.167, 6.0, 8.0, 12.0, 16.0]
    tau_values = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0]

    combos: list[Tuple[float, float, float]] = [
        (sigma, beta, tau)
        for sigma in sigma_values
        for beta in beta_values
        for tau in tau_values
    ]
    expected_filenames = {
        f"features_map_trueskill_{sigma:.3f}_{beta:.3f}_{tau:.3f}.parquet"
        for sigma, beta, tau in combos
    }
    print(f"Total combinations to run: {len(combos)}")

    if not combos:
        return

    matches_df, map_results = load_map_data()
    cpu_count = os.cpu_count() or 4
    max_workers = max(1, cpu_count - 1)
    print(f"Using up to {max_workers} processes.")

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=initialize_worker,
        initargs=(matches_df, map_results),
    ) as executor:
        future_to_params = {
            executor.submit(build_map_trueskill_features, *params): params
            for params in combos
        }

        failed = False
        for index, future in enumerate(as_completed(future_to_params), start=1):
            sigma, beta, tau = future_to_params[future]
            try:
                future.result()
                print(
                    f"[{index}/{len(combos)}] Done SIGMA={sigma:.3f}, "
                    f"BETA={beta:.3f}, TAU={tau:.3f}"
                )
            except Exception as error:
                failed = True
                print(
                    f"[{index}/{len(combos)}] FAILED SIGMA={sigma:.3f}, "
                    f"BETA={beta:.3f}, TAU={tau:.3f}: {error}"
                )

    if failed:
        print("Some combinations failed; preserving stale map TrueSkill files.")
    else:
        remove_stale_outputs(
            FEATURES_DIR,
            "features_map_trueskill_",
            ".parquet",
            expected_filenames,
            expected_output_dir=MAP_TRUESKILL_DIR,
        )


if __name__ == "__main__":
    main()
