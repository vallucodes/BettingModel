import duckdb
import pandas as pd
import numpy as np
import os

PARQUET_DIR = "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/parquet"
FEATURES_DIR = "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/features"
os.makedirs(FEATURES_DIR, exist_ok=True)

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


def main() -> None:
    con = duckdb.connect()

    df = con.execute(
        f"""
        SELECT hltv_match_id, team1_name, team2_name, team1_score, team2_score
        FROM '{PARQUET_DIR}/matches.parquet'
        ORDER BY hltv_match_id ASC
    """
    ).df()

    df["result"] = np.where(df["team1_score"] > df["team2_score"], 0, 1)

    n = len(df)
    team1_elo_vals = np.empty(n, dtype=float)
    team2_elo_vals = np.empty(n, dtype=float)

    unique_teams = pd.unique(df[["team1_name", "team2_name"]].values.ravel())
    elo_state = {team: {"elo": float(START_ELO), "games_played": 0} for team in unique_teams}

    for i, row in enumerate(df.itertuples(index=False)):
        team1 = row.team1_name
        team2 = row.team2_name

        # Elo before this match (based only on prior matches)
        team1_elo_vals[i] = elo_state[team1]["elo"]
        team2_elo_vals[i] = elo_state[team2]["elo"]

        res = 0 if row.team1_score > row.team2_score else 1

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

    out_df = df[["hltv_match_id"]].copy()
    out_df["team1_elo"] = team1_elo_vals
    out_df["team2_elo"] = team2_elo_vals

    output_path = os.path.join(FEATURES_DIR, "features_team_elo_100_20_30.parquet")
    out_df.to_parquet(output_path, index=False)
    print(f"✅ Saved {len(out_df)} rows → {output_path}")


if __name__ == "__main__":
    main()

