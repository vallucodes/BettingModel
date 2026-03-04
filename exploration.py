import pandas as pd
import numpy as np

PARQUET_DIR = "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/parquet"


def main():
    # Load matches parquet
    df = pd.read_parquet(f"{PARQUET_DIR}/matches.parquet")[
        ["hltv_match_id", "team1_name", "team2_name", "team1_score", "team2_score"]
    ].sort_values("hltv_match_id")

    # Binary outcome from team1 perspective: 1 = team1 win, 0 = team2 win
    df["team1_win"] = (df["team1_score"] > df["team2_score"]).astype(int)
    df["team2_win"] = 1 - df["team1_win"]

    # Overall winrates
    overall_team1_wr = df["team1_win"].mean()
    overall_team2_wr = df["team2_win"].mean()

    print("=== Overall winrates (by position) ===")
    print(f"Team1 winrate: {overall_team1_wr:.3f}")
    print(f"Team2 winrate: {overall_team2_wr:.3f}")
    print()

    # Winrates by specific team when they appear as team1 or team2
    team1_stats = (
        df.groupby("team1_name")["team1_win"]
        .agg(["count", "mean"])
        .rename(columns={"count": "matches_as_team1", "mean": "team1_winrate"})
    )

    team2_stats = (
        df.groupby("team2_name")["team2_win"]
        .agg(["count", "mean"])
        .rename(columns={"count": "matches_as_team2", "mean": "team2_winrate"})
    )

    per_team = team1_stats.join(team2_stats, how="outer").fillna(0)

    print("=== Example per-team stats (top 20 by matches played) ===")
    per_team["total_matches"] = (
        per_team["matches_as_team1"] + per_team["matches_as_team2"]
    )
    example = (
        per_team.sort_values("total_matches", ascending=False)
        .head(20)
        .round(3)
    )
    print(example.to_string())


if __name__ == "__main__":
    main()

