import pandas as pd
import numpy as np

PARQUET_DIR = "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/parquet"


def get_map_play_counts(parquet_dir: str = PARQUET_DIR) -> pd.DataFrame:
    maps = pd.read_parquet(f"{parquet_dir}/maps.parquet")[
        ["hltv_match_id", "map_number", "map_name"]
    ].copy()

    maps["map_name"] = maps["map_name"].astype("string").str.strip().str.lower()
    maps = maps.dropna(subset=["map_name"])
    maps = maps[maps["map_name"] != ""]

    out = (
        maps.groupby("map_name")
        .agg(
            maps_played=("map_name", "size"),
            matches_with_map=("hltv_match_id", "nunique"),
        )
        .sort_values(["maps_played", "matches_with_map"], ascending=False)
        .reset_index()
    )
    return out


def get_map_slot_counts(parquet_dir: str = PARQUET_DIR) -> pd.DataFrame:
    """
    Counts how often each map appears as:
    - map1 (first map)
    - map2 (second map)
    - decider in BO3, plus all BO1 maps counted as deciders

    We infer BO1/BO3 from how many map rows exist per hltv_match_id in maps.parquet.
    """
    maps = pd.read_parquet(f"{parquet_dir}/maps.parquet")[
        ["hltv_match_id", "map_number", "map_name"]
    ].copy()

    maps["map_name"] = maps["map_name"].astype("string").str.strip().str.lower()
    maps = maps.dropna(subset=["map_name"])
    maps = maps[maps["map_name"] != ""]

    maps_per_match = maps.groupby("hltv_match_id")["map_number"].size().rename("n_maps")
    maps = maps.merge(maps_per_match, on="hltv_match_id", how="left")

    # Slot assignment:
    # - BO1 (n_maps == 1): map counted as decider
    # - BO3: map_number 1 -> map1, 2 -> map2, 3 -> decider
    slot = pd.Series(pd.NA, index=maps.index, dtype="string")
    slot = slot.mask(maps["n_maps"].eq(1), "decider")
    slot = slot.mask(maps["n_maps"].ge(2) & maps["map_number"].eq(1), "map1")
    slot = slot.mask(maps["n_maps"].ge(2) & maps["map_number"].eq(2), "map2")
    slot = slot.mask(maps["n_maps"].eq(3) & maps["map_number"].eq(3), "decider")
    maps["slot"] = slot

    # Drop maps we don't classify (e.g., BO5 map3/map4/map5, etc.)
    maps = maps.dropna(subset=["slot"])

    out = (
        maps.pivot_table(
            index="map_name",
            columns="slot",
            values="hltv_match_id",
            aggfunc="size",
            fill_value=0,
        )
        .rename_axis(None, axis=1)
        .reset_index()
    )

    for col in ["map1", "map2", "decider"]:
        if col not in out.columns:
            out[col] = 0

    out["total_classified"] = out["map1"] + out["map2"] + out["decider"]
    out = out.sort_values(["total_classified", "decider", "map1", "map2"], ascending=False)
    return out[["map_name", "map1", "map2", "decider", "total_classified"]]


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
    print()

    print("=== Map play counts ===")
    map_counts = get_map_play_counts(PARQUET_DIR)
    print(map_counts.to_string(index=False))
    print()

    print("=== Map slot counts (BO3 map1/map2/decider; BO1 counts as decider) ===")
    slot_counts = get_map_slot_counts(PARQUET_DIR)
    print(slot_counts.to_string(index=False))


if __name__ == "__main__":
    main()

