import duckdb
import pandas as pd
import os

# Paths (adjust if needed)
PARQUET_DIR = "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/parquet"
FEATURES_DIR = "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/features"
FEATURES_FILE = "features_map_winrate_l10.parquet"

# Load features
features_path = os.path.join(FEATURES_DIR, FEATURES_FILE)
df_features = pd.read_parquet(features_path)

# Connect to DuckDB
con = duckdb.connect()

# Helper: Get all prior map plays for a team on a map before a given date
def get_prior_plays(team, map_name, match_date, window=10):
    query = """
        SELECT
            m.hltv_match_id,
            m.date,
            mp.map_name,
            mp.score,
            CASE
                WHEN m.team1_name = ? THEN
                    CASE WHEN CAST(SPLIT_PART(mp.score, ':', 1) AS INT) > CAST(SPLIT_PART(mp.score, ':', 2) AS INT) THEN 1 ELSE 0 END
                WHEN m.team2_name = ? THEN
                    CASE WHEN CAST(SPLIT_PART(mp.score, ':', 2) AS INT) > CAST(SPLIT_PART(mp.score, ':', 1) AS INT) THEN 1 ELSE 0 END
                ELSE NULL
            END AS map_won
        FROM '{PARQUET_DIR}/maps.parquet' mp
        JOIN '{PARQUET_DIR}/matches.parquet' m ON mp.hltv_match_id = m.hltv_match_id
        WHERE (m.team1_name = ? OR m.team2_name = ?)
          AND mp.map_name ILIKE ?
          AND m.date < ?
        ORDER BY m.date DESC
        LIMIT ?
    """.format(PARQUET_DIR=PARQUET_DIR)   # only the paths are formatted

    params = [team, team, team, team, map_name, match_date, window]

    prior = con.execute(query, params).df()
    if prior.empty:
        return None
    return prior['map_won'].mean()

# Validation function: For a list of samples (match_id, team_position, team_name, map_name)
def validate_winrates(samples, df_features):
    results = []
    for match_id, team_pos, team_name, map_name in samples:
        # Get feature value
        row = df_features[df_features['hltv_match_id'] == match_id]
        if row.empty:
            results.append((match_id, team_name, map_name, None, None, "Match not found"))
            continue
        feature_col = f'{team_pos}_{map_name.lower()}_wr_l10'
        feature_wr = row[feature_col].values[0]

        # Get match date
        match_date = row['date'].values[0] if 'date' in row.columns else con.execute(f"SELECT date FROM '{PARQUET_DIR}/matches.parquet' WHERE hltv_match_id = {match_id}").fetchone()[0]

        # Compute manual WR
        manual_wr = get_prior_plays(team_name, map_name, match_date)

        # Compare
        status = "Match" if manual_wr == feature_wr or (pd.isna(manual_wr) and pd.isna(feature_wr)) else "Mismatch"
        results.append((match_id, team_name, map_name, feature_wr, manual_wr, status))

    # Output as DataFrame
    return pd.DataFrame(results, columns=['hltv_match_id', 'team_name', 'map_name', 'feature_wr', 'manual_wr', 'status'])

# Example samples: Pick some from features where not null, for every map
maps = ['ancient', 'anubis', 'cache', 'dust2', 'inferno', 'mirage', 'nuke', 'overpass', 'train', 'vertigo']
samples = []
for map_name in maps:
    # Team1 samples
    team1_col = f'team1_{map_name}_wr_l10'
    if team1_col in df_features.columns:
        team1_samples = df_features[df_features[team1_col].notna()].sample(min(100, len(df_features[df_features[team1_col].notna()]))).iterrows()
        for _, row in team1_samples:
            samples.append((row['hltv_match_id'], 'team1', row['team1_name'], map_name.capitalize()))  # Capitalize map_name for query if needed

    # Team2 samples
    team2_col = f'team2_{map_name}_wr_l10'
    if team2_col in df_features.columns:
        team2_samples = df_features[df_features[team2_col].notna()].sample(min(100, len(df_features[df_features[team2_col].notna()]))).iterrows()
        for _, row in team2_samples:
            samples.append((row['hltv_match_id'], 'team2', row['team2_name'], map_name.capitalize()))  # Capitalize map_name for query if needed

# Run validation
validation_df = validate_winrates(samples, df_features)
print(validation_df.to_string())

# Check for mismatches
mismatches = validation_df[validation_df['status'] == 'Mismatch']
if not mismatches.empty:
    print("\nMismatches found:")
    print(mismatches.to_string())
else:
    print("\nNo mismatches found in samples.")
