import duckdb
import pandas as pd
import os

PARQUET_DIR = "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/parquet"
FEATURES_DIR = "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/features"
os.makedirs(FEATURES_DIR, exist_ok=True)

def load_base_data():
    """Extracts base match and round data, assigning strict win/loss for feature math."""
    print("📥 Loading base data from DuckDB...")
    con = duckdb.connect()

    team_match_stats = con.execute(f"""
        WITH round_counts AS (
            SELECT hltv_match_id, COUNT(*) AS total_rounds
            FROM '{PARQUET_DIR}/rounds.parquet'
            GROUP BY hltv_match_id
        ),
        team_rows AS (
            SELECT m.hltv_match_id, m.date, m.team1_name AS team_name, 'team1' AS side,
                   CASE WHEN m.team1_score > m.team2_score THEN 1 ELSE 0 END AS won,
                   m.team1_rating_3 AS rating, m.team1_kast_pct AS kast_pct,
                   m.team1_swing_pct AS swing_pct, r.total_rounds
            FROM '{PARQUET_DIR}/matches.parquet' m
            JOIN round_counts r ON m.hltv_match_id = r.hltv_match_id

            UNION ALL

            SELECT m.hltv_match_id, m.date, m.team2_name AS team_name, 'team2' AS side,
                   CASE WHEN m.team2_score > m.team1_score THEN 1 ELSE 0 END AS won,
                   m.team2_rating_3 AS rating, m.team2_kast_pct AS kast_pct,
                   m.team2_swing_pct AS swing_pct, r.total_rounds
            FROM '{PARQUET_DIR}/matches.parquet' m
            JOIN round_counts r ON m.hltv_match_id = r.hltv_match_id
        )
        SELECT * FROM team_rows
    """).df()

    # Strict chronological sort for rolling math
    team_match_stats = team_match_stats.sort_values(['team_name', 'hltv_match_id']).reset_index(drop=True)

    matches_df = con.execute(f"""
        SELECT hltv_match_id, date, team1_name, team2_name,
               team1_score, team2_score, event_type,
               CASE WHEN team1_score > team2_score THEN 0 ELSE 1 END AS target_match_winner
        FROM '{PARQUET_DIR}/matches.parquet'
        ORDER BY hltv_match_id ASC
    """).df()

    con.close()
    return team_match_stats, matches_df

def generate_rolling_features(window_size, base_team_stats, base_matches):
    """Calculates both unweighted win % and round-weighted stats for a given window size."""
    print(f"🔄 Processing rolling stats for l{window_size}...")

    # Copy to avoid mutating the original dataframe during the loop
    tm = base_team_stats.copy()

    # Set up dynamic column names
    win_col = f'rolling_win_rate_l{window_size}'
    rating_col = f'rolling_rating_l{window_size}'
    kast_col = f'rolling_kast_l{window_size}'
    swing_col = f'rolling_swing_l{window_size}'

    # ==========================================
    # A. Unweighted Match Win Rate
    # ==========================================
    tm[win_col] = tm.groupby('team_name')['won'].transform(
        lambda x: x.rolling(window=window_size, min_periods=1).mean().shift(1)
    )

    # ==========================================
    # B. Round-Weighted Stats
    # ==========================================
    rolling_weights = tm.groupby('team_name')['total_rounds'].transform(
        lambda x: x.rolling(window=window_size, min_periods=1).sum().shift(1)
    )

    stats_to_roll = {
        'rating': rating_col,
        'kast_pct': kast_col,
        'swing_pct': swing_col
    }

    for stat_col, out_col in stats_to_roll.items():
        tm['temp_numerator'] = tm[stat_col] * tm['total_rounds']
        rolling_numerator = tm.groupby('team_name')['temp_numerator'].transform(
            lambda x: x.rolling(window=window_size, min_periods=1).sum().shift(1)
        )
        tm[out_col] = rolling_numerator / rolling_weights

    if 'temp_numerator' in tm.columns:
        tm = tm.drop(columns=['temp_numerator'])

    # ==========================================
    # C. Format & Merge
    # ==========================================
    # Team 1 Split
    t1_cols = {
        rating_col: f'team1_{rating_col}',
        kast_col: f'team1_{kast_col}',
        swing_col: f'team1_{swing_col}',
        win_col: f'team1_{win_col}'
    }
    t1_stats = tm[tm['side'] == 'team1'][['hltv_match_id'] + list(t1_cols.keys())].rename(columns=t1_cols)

    # Team 2 Split
    t2_cols = {
        rating_col: f'team2_{rating_col}',
        kast_col: f'team2_{kast_col}',
        swing_col: f'team2_{swing_col}',
        win_col: f'team2_{win_col}'
    }
    t2_stats = tm[tm['side'] == 'team2'][['hltv_match_id'] + list(t2_cols.keys())].rename(columns=t2_cols)

    # Final Merge
    features_df = base_matches.copy()
    features_df = features_df.merge(t1_stats, on='hltv_match_id', how='left') \
                           .merge(t2_stats, on='hltv_match_id', how='left')

    features_df = features_df.sort_values('hltv_match_id').reset_index(drop=True)

    # Save
    output_path = os.path.join(FEATURES_DIR, f"features_rolling_l{window_size}.parquet")
    features_df.to_parquet(output_path, index=False)

    print(f"✅ Saved l{window_size} features to {output_path}")

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    base_team_stats, base_matches = load_base_data()

    windows_to_generate = [5, 10, 15]

    for w in windows_to_generate:
        generate_rolling_features(w, base_team_stats, base_matches)
