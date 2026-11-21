"""Pre-match TrueSkill from the players who actually played.

Team-name TrueSkill treats a rename and a rebuilt roster as a new team, and
it keeps a rating when the name stays but the lineup does not. This script
rates players. A team's pre-match strength is the mean skill of that match's
lineup, so:

- the same five players under a new name (ENCE -> ex-ENCE) keep their skill
- a name that stays while the players are replaced takes the new players' skill
- a new lineup made of veterans starts near those veterans, not at the default

No external bridge table is required. The bridge is the player nickname on
the box score. Nicknames have no HLTV player id in this dataset, so a player
who changes nickname still looks new.

Reads matches.parquet and players.parquet. Writes
features_roster_trueskill_<sigma>_<beta>_<tau>.parquet to
vallu_scraper/data/features/roster_trueskill.

Usage, from the model directory:
    python feature_generation/features_roster_trueskill.py

The parameter grid is defined in main().
"""

import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple

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
ROSTER_TRUESKILL_DIR = os.path.join(FEATURES_DIR, "roster_trueskill")

os.makedirs(ROSTER_TRUESKILL_DIR, exist_ok=True)

START_MU = 25.0


def load_matches_and_rosters() -> tuple[pd.DataFrame, dict]:
    """Load matches in start-time order and the lineup for each side.

    Player keys are casefolded nicknames. A side with six or seven names is a
    substitute listed on the box score; every listed player is part of the
    lineup because the parquet has no separate starter flag.
    """
    con = duckdb.connect()
    matches = con.execute(
        f"""
        SELECT
            hltv_match_id,
            date AS started_at,
            team1_name,
            team2_name,
            team1_score,
            team2_score
        FROM '{PARQUET_DIR}/matches.parquet'
        ORDER BY started_at ASC NULLS LAST, hltv_match_id ASC
        """
    ).df()
    player_rows = con.execute(
        f"""
        SELECT hltv_match_id, team, player_name
        FROM '{PARQUET_DIR}/players.parquet'
        """
    ).df()
    con.close()

    rosters: dict[tuple, tuple[str, ...]] = {}
    grouped: dict[tuple, set[str]] = {}
    for row in player_rows.itertuples(index=False):
        nickname = str(row.player_name).strip().casefold()
        if not nickname:
            continue
        grouped.setdefault((row.hltv_match_id, row.team), set()).add(nickname)
    for key, names in grouped.items():
        rosters[key] = tuple(sorted(names))
    return matches, rosters


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _lineup_features(ratings: list[Rating]) -> tuple[float, float, float]:
    """Mean skill, uncertainty of that mean, and conservative mean.

    Stored on the mean scale so a six-player box score does not look stronger
    than a five-player one. Win probability uses the same mean. beta is match
    noise and is applied only in the win probability, not in stored sigma.
    """
    n = len(ratings)
    mu_sum = 0.0
    var_sum = 0.0
    for rating in ratings:
        mu_sum += rating.mu
        var_sum += rating.sigma * rating.sigma
    mu = mu_sum / n
    sigma = math.sqrt(var_sum / (n * n))
    return mu, sigma, mu - 3.0 * sigma


def lineup_win_probability(
    beta: float,
    left: list[Rating],
    right: list[Rating],
) -> float:
    """P(left wins) from mean performance, with per-player match noise beta."""

    def mean_and_var(ratings: list[Rating]) -> tuple[float, float]:
        n = len(ratings)
        mu = sum(rating.mu for rating in ratings) / n
        var = sum(rating.sigma * rating.sigma + beta * beta for rating in ratings) / (
            n * n
        )
        return mu, var

    left_mu, left_var = mean_and_var(left)
    right_mu, right_var = mean_and_var(right)
    variance = left_var + right_var
    if variance <= 0.0:
        return 0.5
    return _phi((left_mu - right_mu) / math.sqrt(variance))


def compute_roster_trueskill_features(
    matches_df: pd.DataFrame,
    rosters: dict,
    sigma: float,
    beta: float,
    tau: float,
) -> pd.DataFrame:
    """Walk matches in start-time order and record pre-match lineup ratings.

    Returns the match-level feature table.
    """
    matches_df = matches_df.sort_values(
        ["started_at", "hltv_match_id"], na_position="last"
    ).reset_index(drop=True)

    env = TrueSkill(
        mu=START_MU,
        sigma=sigma,
        beta=beta,
        tau=tau,
        draw_probability=0.0,
    )
    prior = env.create_rating()

    n = len(matches_df)
    team1_mu = np.empty(n, dtype=float)
    team2_mu = np.empty(n, dtype=float)
    team1_sigma = np.empty(n, dtype=float)
    team2_sigma = np.empty(n, dtype=float)
    team1_conservative = np.empty(n, dtype=float)
    team2_conservative = np.empty(n, dtype=float)
    team1_known = np.empty(n, dtype=np.int16)
    team2_known = np.empty(n, dtype=np.int16)
    team1_prev_shared = np.empty(n, dtype=np.int16)
    team2_prev_shared = np.empty(n, dtype=np.int16)
    team1_name_games = np.empty(n, dtype=np.int32)
    team2_name_games = np.empty(n, dtype=np.int32)
    win_prob = np.empty(n, dtype=float)

    player_state: dict[str, Rating] = {}
    player_games: dict[str, int] = {}
    last_roster: dict[str, frozenset] = {}
    name_games: dict[str, int] = {}

    def ratings_for(lineup: tuple[str, ...]) -> list[Rating]:
        if not lineup:
            return [prior]
        return [player_state.get(player, prior) for player in lineup]

    def known_count(lineup: tuple[str, ...]) -> int:
        return sum(1 for player in lineup if player_games.get(player, 0) > 0)

    for i, row in enumerate(matches_df.itertuples(index=False)):
        left = rosters.get((row.hltv_match_id, "team1"), ())
        right = rosters.get((row.hltv_match_id, "team2"), ())
        left_ratings = ratings_for(left)
        right_ratings = ratings_for(right)

        left_mu, left_sigma, left_cons = _lineup_features(left_ratings)
        right_mu, right_sigma, right_cons = _lineup_features(right_ratings)
        team1_mu[i] = left_mu
        team2_mu[i] = right_mu
        team1_sigma[i] = left_sigma
        team2_sigma[i] = right_sigma
        team1_conservative[i] = left_cons
        team2_conservative[i] = right_cons
        team1_known[i] = known_count(left)
        team2_known[i] = known_count(right)
        win_prob[i] = lineup_win_probability(beta, left_ratings, right_ratings)

        left_name = str(row.team1_name).strip().casefold()
        right_name = str(row.team2_name).strip().casefold()
        left_set = frozenset(left)
        right_set = frozenset(right)
        team1_name_games[i] = name_games.get(left_name, 0)
        team2_name_games[i] = name_games.get(right_name, 0)
        team1_prev_shared[i] = len(left_set & last_roster.get(left_name, frozenset()))
        team2_prev_shared[i] = len(right_set & last_roster.get(right_name, frozenset()))

        ranks = [0, 1] if row.team1_score > row.team2_score else [1, 0]
        (new_left, new_right) = env.rate(
            [tuple(left_ratings), tuple(right_ratings)],
            ranks=ranks,
        )
        for player, rating in zip(left, new_left):
            player_state[player] = rating
            player_games[player] = player_games.get(player, 0) + 1
        for player, rating in zip(right, new_right):
            player_state[player] = rating
            player_games[player] = player_games.get(player, 0) + 1

        last_roster[left_name] = left_set
        last_roster[right_name] = right_set
        name_games[left_name] = name_games.get(left_name, 0) + 1
        name_games[right_name] = name_games.get(right_name, 0) + 1

    out = matches_df[["hltv_match_id", "started_at"]].copy()
    out["team1_roster_trueskill_mu"] = team1_mu
    out["team2_roster_trueskill_mu"] = team2_mu
    out["team1_roster_trueskill_sigma"] = team1_sigma
    out["team2_roster_trueskill_sigma"] = team2_sigma
    out["team1_roster_trueskill_conservative"] = team1_conservative
    out["team2_roster_trueskill_conservative"] = team2_conservative
    out["team1_roster_known_players"] = team1_known
    out["team2_roster_known_players"] = team2_known
    out["team1_roster_prev_shared"] = team1_prev_shared
    out["team2_roster_prev_shared"] = team2_prev_shared
    out["team1_name_games"] = team1_name_games
    out["team2_name_games"] = team2_name_games
    out["roster_win_prob_team1"] = win_prob

    return out


def _filename(sigma: float, beta: float, tau: float) -> str:
    return f"features_roster_trueskill_{sigma:.3f}_{beta:.3f}_{tau:.3f}.parquet"


def build_roster_trueskill_features(sigma: float, beta: float, tau: float) -> None:
    """Build one match-level roster-TrueSkill feature file."""
    matches, rosters = load_matches_and_rosters()
    out = compute_roster_trueskill_features(matches, rosters, sigma, beta, tau)
    output_path = os.path.join(ROSTER_TRUESKILL_DIR, _filename(sigma, beta, tau))
    out.to_parquet(output_path, index=False)
    print(f"Saved {len(out)} rows -> {output_path}")


def main() -> None:
    """Write roster TrueSkill feature files for the configured parameter grid.

    Compared on pre-match win probability, chronological 70/15/15, against
    name-level TrueSkill at sigma 6, beta 2, tau 0.1. Tau 0.5 was worse on
    stable lineups and is not included. Team strength is the mean of the
    lineup, so these betas are not the atomic-team grid.
    """
    sigmas = (8, 10, 12)
    betas = (6, 8)
    taus = (0.1, 0.25)
    combos: list[Tuple[float, float, float]] = [
        (sigma, beta, tau)
        for sigma in sigmas
        for beta in betas
        for tau in taus
    ]
    expected_filenames = {_filename(sigma, beta, tau) for sigma, beta, tau in combos}
    total = len(combos)
    print(f"Total combinations to run: {total}")

    cpu_count = os.cpu_count() or 4
    max_workers = max(1, min(total, cpu_count - 1))
    print(f"Using up to {max_workers} processes.")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_params = {
            executor.submit(build_roster_trueskill_features, sigma, beta, tau): (
                sigma,
                beta,
                tau,
            )
            for sigma, beta, tau in combos
        }
        failed = False
        for i, future in enumerate(as_completed(future_to_params), start=1):
            sigma, beta, tau = future_to_params[future]
            try:
                future.result()
                print(
                    f"[{i}/{total}] Done SIGMA={sigma:.3f}, "
                    f"BETA={beta:.3f}, TAU={tau:.3f}"
                )
            except Exception as exc:
                failed = True
                print(
                    f"[{i}/{total}] FAILED SIGMA={sigma:.3f}, "
                    f"BETA={beta:.3f}, TAU={tau:.3f}: {exc}"
                )

    if failed:
        print("Some combinations failed; preserving stale roster TrueSkill files.")
        return

    remove_stale_outputs(
        FEATURES_DIR,
        "features_roster_trueskill_",
        ".parquet",
        expected_filenames,
        expected_output_dir=ROSTER_TRUESKILL_DIR,
    )


if __name__ == "__main__":
    main()
