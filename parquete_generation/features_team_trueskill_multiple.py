import os
from typing import Iterable, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import duckdb
import numpy as np
import pandas as pd
from trueskill import TrueSkill, Rating


PARQUET_DIR = "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/parquet"
FEATURES_DIR = "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/features"
TEAM_ELO_DIR = os.path.join(FEATURES_DIR, "team_trueskill")

os.makedirs(TEAM_ELO_DIR, exist_ok=True)

# Default TrueSkill configuration (can be changed)
START_MU = 25.0


def load_matches() -> pd.DataFrame:
    """Load base matches data in chronological order."""
    con = duckdb.connect()
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
    con.close()

    # No draws are possible: result is either team1 win or team2 win.
    df["result"] = np.where(df["team1_score"] > df["team2_score"], 0, 1)
    return df


def compute_team_trueskill_features(
    matches_df: pd.DataFrame,
    sigma: float,
    beta: float,
    tau: float,
) -> pd.DataFrame:
    """
    Compute team1 / team2 TrueSkill trajectories for a given
    (sigma, beta, tau) configuration.

    We expose mu (mean), sigma (uncertainty), and a conservative rating
    (mu - 3 * sigma) as features.
    """
    # No draws → draw_probability=0.0
    env = TrueSkill(
        mu=START_MU,
        sigma=sigma,
        beta=beta,
        tau=tau,
        draw_probability=0.0,
    )

    n = len(matches_df)
    team1_mu_vals = np.empty(n, dtype=float)
    team2_mu_vals = np.empty(n, dtype=float)
    team1_sigma_vals = np.empty(n, dtype=float)
    team2_sigma_vals = np.empty(n, dtype=float)
    team1_conservative_vals = np.empty(n, dtype=float)
    team2_conservative_vals = np.empty(n, dtype=float)

    unique_teams = pd.unique(
        matches_df[["team1_name", "team2_name"]].values.ravel()
    )
    rating_state: dict[str, Rating] = {
        team: env.create_rating() for team in unique_teams
    }

    for i, row in enumerate(matches_df.itertuples(index=False)):
        team1 = row.team1_name
        team2 = row.team2_name

        r1 = rating_state[team1]
        r2 = rating_state[team2]

        # Ratings before this match
        team1_mu_vals[i] = r1.mu
        team2_mu_vals[i] = r2.mu
        team1_sigma_vals[i] = r1.sigma
        team2_sigma_vals[i] = r2.sigma
        team1_conservative_vals[i] = r1.mu - 3.0 * r1.sigma
        team2_conservative_vals[i] = r2.mu - 3.0 * r2.sigma

        # Result: 0 = team1 win, 1 = team2 win
        res = 0 if row.team1_score > row.team2_score else 1

        if res == 0:
            ranks = [0, 1]  # team1 wins, team2 loses
        else:
            ranks = [1, 0]  # team2 wins, team1 loses

        (new_r1,), (new_r2,) = env.rate([(r1,), (r2,)], ranks=ranks)
        rating_state[team1] = new_r1
        rating_state[team2] = new_r2

    out_df = matches_df[["hltv_match_id"]].copy()
    out_df["team1_trueskill_mu"] = team1_mu_vals
    out_df["team2_trueskill_mu"] = team2_mu_vals
    out_df["team1_trueskill_sigma"] = team1_sigma_vals
    out_df["team2_trueskill_sigma"] = team2_sigma_vals
    out_df["team1_trueskill_conservative"] = team1_conservative_vals
    out_df["team2_trueskill_conservative"] = team2_conservative_vals
    return out_df


def build_team_trueskill_features(
    sigma: float,
    beta: float,
    tau: float,
) -> None:
    """
    Build team-level TrueSkill features with given parameters and save to
    features_team_trueskill_SIGMA_BETA_TAU.parquet.
    """
    filename = f"features_team_trueskill_{sigma:.3f}_{beta:.3f}_{tau:.3f}.parquet"
    # Keep TrueSkill files under the same 'team_elo' subdirectory for consistency
    output_path = os.path.join(TEAM_ELO_DIR, filename)

    if os.path.exists(output_path):
        print(f"⏭️ Skipping existing file {output_path}")
        return

    matches_df = load_matches()
    out_df = compute_team_trueskill_features(matches_df, sigma, beta, tau)
    out_df.to_parquet(output_path, index=False)
    print(f"✅ Saved {len(out_df)} rows → {output_path}")


def frange(start: float, stop: float, step: float) -> Iterable[float]:
    """
    Simple float range generator inclusive of stop (within a tolerance).
    """
    value = start
    # Use a small epsilon to avoid float accumulation issues
    eps = step / 1000.0
    while value <= stop + eps:
        yield value
        value += step


def main() -> None:
    """
    Sweep over (sigma, beta, tau) TrueSkill parameters.

    Adjust the lists of values below to control the sweep.
    """
    # Explicit grids for each parameter; edit these directly.
    sigma_values = [4.0, 6.0, 8.333, 10.0, 12.0, 15.0]
    beta_values = [1.0, 2.0, 4.167, 6.0, 8.0, 12.0, 16.0]
    tau_values = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0]

    combos: list[Tuple[float, float, float]] = []
    for s in sigma_values:
        for b in beta_values:
            for t in tau_values:
                filename = f"features_team_trueskill_{s:.3f}_{b:.3f}_{t:.3f}.parquet"
                path = os.path.join(TEAM_ELO_DIR, filename)
                if os.path.exists(path):
                    print(f"Already have {filename}, skipping in queue build.")
                    continue
                combos.append((s, b, t))

    total = len(combos)
    print(f"Total combinations to run: {total}")

    cpu_count = os.cpu_count() or 4
    max_workers = max(1, cpu_count - 1)
    print(f"Using up to {max_workers} processes.")

    if total == 0:
        return

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_params = {
            executor.submit(build_team_trueskill_features, s, b, t): (s, b, t)
            for (s, b, t) in combos
        }

        for i, future in enumerate(as_completed(future_to_params), start=1):
            s, b, t = future_to_params[future]
            try:
                future.result()
                print(
                    f"[{i}/{total}] Done SIGMA={s:.3f}, "
                    f"BETA={b:.3f}, TAU={t:.3f}"
                )
            except Exception as e:
                print(
                    f"[{i}/{total}] FAILED SIGMA={s:.3f}, "
                    f"BETA={b:.3f}, TAU={t:.3f}: {e}"
                )


if __name__ == "__main__":
    main()

