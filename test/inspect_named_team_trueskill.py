"""Print recent pre-match TrueSkill ratings for selected teams.

Run from the project root:
    python test/inspect_named_team_trueskill.py
"""

import argparse
from pathlib import Path

import duckdb


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_FEATURES_PARQUET = (
    ROOT_DIR
    / "vallu_scraper/data/features/team_trueskill/"
    "features_team_trueskill_15.000_8.000_0.500.parquet"
)
DEFAULT_MATCHES_PARQUET = ROOT_DIR / "vallu_scraper/data/parquet/matches.parquet"
DEFAULT_TEAMS = ["Grayhound"]
MATCH_LIMIT = 40


def load_recent_matches(
    features_parquet: Path,
    matches_parquet: Path,
    teams: list[str],
    limit: int,
):
    placeholders = ", ".join("?" for _ in teams)
    query = f"""
        WITH target_matches AS (
            SELECT
                m.hltv_match_id,
                m.date AS match_date,
                CASE
                    WHEN lower(m.team1_name) IN ({placeholders}) THEN m.team1_name
                    ELSE m.team2_name
                END AS team,
                CASE
                    WHEN lower(m.team1_name) IN ({placeholders}) THEN m.team2_name
                    ELSE m.team1_name
                END AS opponent,
                CASE
                    WHEN lower(m.team1_name) IN ({placeholders})
                        THEN f.team1_trueskill_mu
                    ELSE f.team2_trueskill_mu
                END AS team_mu,
                CASE
                    WHEN lower(m.team1_name) IN ({placeholders})
                        THEN f.team1_trueskill_sigma
                    ELSE f.team2_trueskill_sigma
                END AS team_sigma,
                CASE
                    WHEN lower(m.team1_name) IN ({placeholders})
                        THEN f.team2_trueskill_mu
                    ELSE f.team1_trueskill_mu
                END AS opponent_mu,
                CASE
                    WHEN lower(m.team1_name) IN ({placeholders})
                        THEN f.team2_trueskill_sigma
                    ELSE f.team1_trueskill_sigma
                END AS opponent_sigma
            FROM read_parquet(?) AS f
            JOIN read_parquet(?) AS m USING (hltv_match_id)
            WHERE lower(m.team1_name) IN ({placeholders})
                OR lower(m.team2_name) IN ({placeholders})
        ),
        ranked_matches AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY lower(team)
                    ORDER BY match_date DESC NULLS LAST, hltv_match_id DESC
                ) AS match_rank
            FROM target_matches
        )
        SELECT
            match_date,
            team,
            team_mu,
            team_sigma,
            opponent,
            opponent_mu,
            opponent_sigma
        FROM ranked_matches
        WHERE match_rank <= ?
        ORDER BY lower(team), match_rank
    """
    normalized_teams = [team.lower() for team in teams]
    team_params = normalized_teams * 6
    params = [
        *team_params,
        str(features_parquet),
        str(matches_parquet),
        *normalized_teams,
        *normalized_teams,
        limit,
    ]

    with duckdb.connect() as connection:
        return connection.execute(query, params).df()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show recent pre-match TrueSkill ratings for each requested team."
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=DEFAULT_FEATURES_PARQUET,
        help="Path to a team TrueSkill features parquet file.",
    )
    parser.add_argument(
        "--matches-parquet",
        type=Path,
        default=DEFAULT_MATCHES_PARQUET,
        help="Path to matches.parquet.",
    )
    parser.add_argument(
        "--teams",
        nargs="+",
        default=DEFAULT_TEAMS,
        help="Team names to inspect (default: ENCE ex-ENCE).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=MATCH_LIMIT,
        help=f"Maximum number of recent matches per team (default: {MATCH_LIMIT}).",
    )
    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be at least 1")
    for path in (args.parquet, args.matches_parquet):
        if not path.is_file():
            raise FileNotFoundError(f"Parquet file does not exist: {path}")

    matches = load_recent_matches(
        args.parquet,
        args.matches_parquet,
        args.teams,
        args.limit,
    )

    for team in args.teams:
        team_matches = matches[matches["team"].str.casefold() == team.casefold()]
        print(f"\n=== {team} (up to {args.limit} most recent matches) ===")
        if team_matches.empty:
            print("No matching matches found.")
            continue

        display = team_matches.drop(columns="team").rename(
            columns={"match_date": "date"}
        )
        rating_columns = [
            "team_mu",
            "team_sigma",
            "opponent_mu",
            "opponent_sigma",
        ]
        display[rating_columns] = display[rating_columns].round(2)
        print(display.to_string(index=False))


if __name__ == "__main__":
    main()
