import os
import re
import pandas as pd

FEATURES_DIR = "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/features"


def build_schema_for_parquet(path: str) -> pd.DataFrame:
    """
    Read a parquet file and return a simple schema dataframe:
    columns: name, dtype (as string).
    """
    # Read only metadata and dtypes; no need to load entire file into memory
    df = pd.read_parquet(path)
    schema = pd.DataFrame(
        {
            "name": df.columns,
            "dtype": [str(dt) for dt in df.dtypes],
        }
    )
    return schema


def main() -> None:
    os.makedirs(FEATURES_DIR, exist_ok=True)

    raw_files = sorted(
        f
        for f in os.listdir(FEATURES_DIR)
        if f.endswith(".parquet") and not f.startswith("schema_")
    )

    if not raw_files:
        print(f"No parquet files found in {FEATURES_DIR}")
        return

    # Group parquet files into logical "families" so that we only
    # generate one schema per family. Examples:
    # - features_rolling_l5 / l10 / l15 → family "features_rolling"
    # - features_map_winrate_l5 / l10 / l15 → family "features_map_winrate"
    # - features_map_elo_100_20_10, features_map_elo_50_10_5 → family "features_map_elo"
    # - features_team_elo_100_20_30 → family "features_team_elo"
    family_to_file = {}

    for fname in raw_files:
        stem = os.path.splitext(fname)[0]

        # Default: use full stem as family key
        family = stem

        # Collapse rolling lX variants
        m = re.match(r"^(features_rolling)_l\d+$", stem)
        if m:
            family = m.group(1)

        # Collapse map winrate lX variants
        m = re.match(r"^(features_map_winrate)_l\d+$", stem)
        if m:
            family = m.group(1)

        # Collapse map elo variants with parameter suffixes
        if stem.startswith("features_map_elo_"):
            family = "features_map_elo"

        # Collapse team elo variants with parameter suffixes
        if stem.startswith("features_team_elo_"):
            family = "features_team_elo"

        if family in family_to_file:
            # We already selected a representative file for this family
            print(f"Grouping {fname} under family '{family}' (using {family_to_file[family]}).")
            continue

        family_to_file[family] = fname

    files = sorted(family_to_file.values())

    print(f"Found {len(raw_files)} parquet files in {FEATURES_DIR}")
    print(f"Building schemas for {len(files)} logical families.")

    for fname in files:
        full_path = os.path.join(FEATURES_DIR, fname)
        base = os.path.splitext(fname)[0]
        txt_path = os.path.join(FEATURES_DIR, f"schema_{base}.txt")

        if os.path.exists(txt_path):
            print(f"⏭️  Text schema already exists for {fname}: {os.path.basename(txt_path)}, skipping.")
            continue

        print(f"Building text schema for {fname} → {os.path.basename(txt_path)}")
        try:
            schema_df = build_schema_for_parquet(full_path)
            # Plain-text schema (easy to read in editor): one column name per line
            with open(txt_path, "w", encoding="utf-8") as f:
                for col in schema_df["name"]:
                    f.write(f"{col}\n")

            print(
                f"✅ Saved text schema with {len(schema_df)} columns to "
                f"{txt_path}"
            )
        except Exception as e:
            print(f"❌ Failed to build schema for {fname}: {e}")


if __name__ == "__main__":
    main()

