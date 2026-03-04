import pandas as pd

MATCHES_PARQUET = "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/parquet/matches.parquet"


def main():
    df = pd.read_parquet(MATCHES_PARQUET)

    # Keep only the columns we care about for a quick sanity check
    cols = [
        "hltv_match_id",
        "team1_name",
        "team2_name",
        "team1_score",
        "team2_score",
        "is_bo1",
        "is_bo3",
        "is_bo5",
    ]
    df = df[cols].copy()

    # Sort to get a stable view and take a small sample
    df = df.sort_values("hltv_match_id").sample(1000).reset_index(drop=True)

    # Configure display for readable console output
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 0)

    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

