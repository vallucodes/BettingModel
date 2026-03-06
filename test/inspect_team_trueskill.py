# RUN:
# python test/inspect_team_trueskill.py --parquet /media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/features/team_trueskill/features_team_trueskill_4.000_2.000_0.050.parquet

import argparse
import glob
import os

import duckdb
import pandas as pd


PARQUET_DIR = "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/parquet"
FEATURES_DIR = "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/features"
TEAM_TS_DIR = os.path.join(FEATURES_DIR, "team_elo_trueskill")


def load_matches() -> pd.DataFrame:
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
    return df


def pick_latest_trueskill_file() -> str:
    pattern = os.path.join(TEAM_TS_DIR, "features_team_trueskill_*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No TrueSkill feature files found in {TEAM_TS_DIR}")
    # Last lexicographically usually corresponds to latest parameter combo written
    return files[-1]


def compute_latest_team_ratings(features_df: pd.DataFrame, matches_df: pd.DataFrame) -> pd.DataFrame:
    merged = features_df.merge(matches_df, on="hltv_match_id", how="left", validate="one_to_one")

    records = []
    for row in merged.itertuples(index=False):
        records.append(
            {
                "team": row.team1_name,
                "hltv_match_id": row.hltv_match_id,
                "mu": row.team1_trueskill_mu,
                "sigma": row.team1_trueskill_sigma,
                "conservative": row.team1_trueskill_conservative,
            }
        )
        records.append(
            {
                "team": row.team2_name,
                "hltv_match_id": row.hltv_match_id,
                "mu": row.team2_trueskill_mu,
                "sigma": row.team2_trueskill_sigma,
                "conservative": row.team2_trueskill_conservative,
            }
        )

    team_df = pd.DataFrame.from_records(records)
    # Latest rating per team (strictly before that team's last match)
    latest = (
        team_df.sort_values("hltv_match_id")
        .groupby("team", as_index=False)
        .tail(1)
    )

    top_100 = latest.sort_values("conservative", ascending=False).head(100)
    return top_100


def build_last_50_matches_table(features_df: pd.DataFrame, matches_df: pd.DataFrame) -> pd.DataFrame:
    merged = features_df.merge(matches_df, on="hltv_match_id", how="left", validate="one_to_one")
    merged = merged.sort_values("hltv_match_id")
    last_50 = merged.tail(50).copy()

    cols = [
        "hltv_match_id",
        "team1_name",
        "team2_name",
        "team1_trueskill_mu",
        "team1_trueskill_sigma",
        "team1_trueskill_conservative",
        "team2_trueskill_mu",
        "team2_trueskill_sigma",
        "team2_trueskill_conservative",
    ]
    last_50 = last_50[cols].rename(columns={
        "team1_name": "team1",
        "team2_name": "team2",
        "team1_trueskill_mu": "t1_mu",
        "team1_trueskill_sigma": "t1_sigma",
        "team1_trueskill_conservative": "t1_cons",
        "team2_trueskill_mu": "t2_mu",
        "team2_trueskill_sigma": "t2_sigma",
        "team2_trueskill_conservative": "t2_cons",
    })
    for col in ["t1_mu", "t1_sigma", "t1_cons", "t2_mu", "t2_sigma", "t2_cons"]:
        last_50[col] = last_50[col].round(1)
    return last_50


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect team TrueSkill features: top 100 teams by rating and "
            "last 50 matches with pre-match TrueSkill."
        )
    )
    parser.add_argument(
        "--parquet",
        type=str,
        default=None,
        help=(
            "Path to a features_team_trueskill_*.parquet file. "
            "If omitted, the latest file in team_elo_trueskill/ is used."
        ),
    )
    args = parser.parse_args()

    if args.parquet is None:
        parquet_path = pick_latest_trueskill_file()
        print(f"Using latest TrueSkill parquet: {parquet_path}")
    else:
        parquet_path = args.parquet
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Provided parquet file does not exist: {parquet_path}")

    features_df = pd.read_parquet(parquet_path)
    matches_df = load_matches()

    print("\n=== Top 100 teams by conservative TrueSkill (mu - 3*sigma) ===")
    top_100 = compute_latest_team_ratings(features_df, matches_df)
    print(top_100.to_string(index=False))

    print("\n=== Last 50 matches with pre-match TrueSkill ===")
    last_50 = build_last_50_matches_table(features_df, matches_df)
    print(last_50.to_string(index=False))


if __name__ == "__main__":
    main()