import duckdb
import pandas as pd
import numpy as np
import os

PARQUET_DIR = "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/parquet"
FEATURES_DIR = "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/features"

TEAM_ELO_FILE = os.path.join(FEATURES_DIR, "features_team_elo_100_20_30.parquet")

START_ELO = 1500
HIGH_K = 100
LOW_K = 20
THRESHOLD = 30


def expected_score(elo_1: float, elo_2: float) -> float:
    return 1 / (1 + 10 ** ((elo_2 - elo_1) / 400))


def update_elo(elo_1: float, elo_2: float, res: int, k1: float, k2: float) -> tuple[float, float]:
    ea = expected_score(elo_1, elo_2)
    elo_1_new = elo_1 + k1 * ((1 - res) - ea)
    elo_2_new = elo_2 + k2 * (res - (1 - ea))
    return elo_1_new, elo_2_new


def recompute_elos_from_matches() -> pd.DataFrame:
    con = duckdb.connect()
    df = con.execute(
        f"""
        SELECT hltv_match_id, team1_name, team2_name, team1_score, team2_score
        FROM '{PARQUET_DIR}/matches.parquet'
        ORDER BY hltv_match_id ASC
    """
    ).df()

    df["result"] = np.where(df["team1_score"] > df["team2_score"], 0, 1)

    df["team1_elo_model"] = None
    df["team2_elo_model"] = None

    unique_teams = pd.unique(df[["team1_name", "team2_name"]].values.ravel())
    elo_state = {team: {"elo": float(START_ELO), "games_played": 0} for team in unique_teams}

    for idx, row in df.iterrows():
        team1 = row["team1_name"]
        team2 = row["team2_name"]

        # Elo before this match (model-style)
        df.at[idx, "team1_elo_model"] = elo_state[team1]["elo"]
        df.at[idx, "team2_elo_model"] = elo_state[team2]["elo"]

        res = df.at[idx, "result"]

        k1 = HIGH_K if elo_state[team1]["games_played"] < THRESHOLD else LOW_K
        k2 = HIGH_K if elo_state[team2]["games_played"] < THRESHOLD else LOW_K

        new_elo1, new_elo2 = update_elo(
            elo_state[team1]["elo"],
            elo_state[team2]["elo"],
            res,
            k1,
            k2,
        )
        elo_state[team1]["elo"] = new_elo1
        elo_state[team2]["elo"] = new_elo2
        elo_state[team1]["games_played"] += 1
        elo_state[team2]["games_played"] += 1

    return df[["hltv_match_id", "team1_elo_model", "team2_elo_model"]]


def main() -> None:
    if not os.path.exists(TEAM_ELO_FILE):
        raise FileNotFoundError(f"Expected parquet not found: {TEAM_ELO_FILE}")

    generated = pd.read_parquet(TEAM_ELO_FILE).rename(
        columns={
            "team1_elo": "team1_elo_parquet",
            "team2_elo": "team2_elo_parquet",
        }
    )

    recomputed = recompute_elos_from_matches()

    merged = recomputed.merge(generated, on="hltv_match_id", how="inner")

    merged["diff_team1"] = merged["team1_elo_model"] - merged["team1_elo_parquet"]
    merged["diff_team2"] = merged["team2_elo_model"] - merged["team2_elo_parquet"]

    tol = 1e-6
    mismatches = merged[
        (merged["diff_team1"].abs() > tol) | (merged["diff_team2"].abs() > tol)
    ]

    total = len(merged)
    num_mismatch = len(mismatches)

    print(f"Total matches compared: {total}")
    print(f"Mismatches: {num_mismatch}")

    if num_mismatch > 0:
        print("First 20 mismatches:")
        print(
            mismatches[
                [
                    "hltv_match_id",
                    "team1_elo_model",
                    "team1_elo_parquet",
                    "diff_team1",
                    "team2_elo_model",
                    "team2_elo_parquet",
                    "diff_team2",
                ]
            ].head(20)
        )
    else:
        print("✅ All team Elo values match between model computation and parquet file.")


if __name__ == "__main__":
    main()

