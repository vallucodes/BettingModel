import duckdb
import pandas as pd
import os

PARQUET_DIR = "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/parquet"
FEATURES_DIR = "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/features"
os.makedirs(FEATURES_DIR, exist_ok=True)

con = duckdb.connect()

# Step 1: build a per-team-per-match view with round-weighted stats
team_match_stats = con.execute(f"""
    WITH round_counts AS (
        SELECT hltv_match_id, COUNT(*) AS total_rounds
        FROM '{PARQUET_DIR}/rounds.parquet'
        GROUP BY hltv_match_id
    ),
    -- Unpivot matches so each match appears twice, once per team
    team_rows AS (
        SELECT
            m.hltv_match_id,
            m.date,
            m.team1_name AS team_name,
            'team1'      AS side,
            CASE WHEN m.team1_score > m.team2_score THEN 0 ELSE 1 END AS won,
            m.team1_rating_3  AS rating,
            m.team1_kast_pct  AS kast_pct,
            m.team1_swing_pct AS swing_pct,
            r.total_rounds
        FROM '{PARQUET_DIR}/matches.parquet' m
        JOIN round_counts r ON m.hltv_match_id = r.hltv_match_id

        UNION ALL

        SELECT
            m.hltv_match_id,
            m.date,
            m.team2_name AS team_name,
            'team2'      AS side,
            CASE WHEN m.team2_score > m.team1_score THEN 0 ELSE 1 END AS won,
            m.team2_rating_3  AS rating,
            m.team2_kast_pct  AS kast_pct,
            m.team2_swing_pct AS swing_pct,
            r.total_rounds
        FROM '{PARQUET_DIR}/matches.parquet' m
        JOIN round_counts r ON m.hltv_match_id = r.hltv_match_id
    )
    SELECT * FROM team_rows
    ORDER BY team_name, date
""").df()

print(team_match_stats.to_string())


# Step 2: compute rolling last-10 weighted averages in pandas
# (DuckDB window functions can't do weighted rolling easily, pandas is cleaner here)
team_match_stats = team_match_stats.sort_values(['team_name', 'date']).reset_index(drop=True)

def weighted_rolling_mean(group, stat_col, weight_col, window=10):
    """Compute rolling weighted average using previous N matches (no lookahead)."""
    results = []
    for i in range(len(group)):
        start = max(0, i - window)
        window_slice = group.iloc[start:i]  # excludes current match
        if len(window_slice) == 0:
            results.append(None)
        else:
            weights = window_slice[weight_col]
            values  = window_slice[stat_col]
            results.append((values * weights).sum() / weights.sum())
    return results

rolling_stats = []
for team_name, group in team_match_stats.groupby('team_name'):
    group = group.reset_index(drop=True)
    group['rolling_rating_l10']   = weighted_rolling_mean(group, 'rating',   'total_rounds')
    group['rolling_kast_l10']     = weighted_rolling_mean(group, 'kast_pct', 'total_rounds')
    group['rolling_swing_l10']    = weighted_rolling_mean(group, 'swing_pct','total_rounds')
    group['rolling_win_rate_l10'] = weighted_rolling_mean(group, 'won',      'total_rounds')
    rolling_stats.append(group)

rolling_df = pd.concat(rolling_stats).reset_index(drop=True)

# Step 3: join back to matches — one row per match with both team's rolling stats
matches_df = con.execute(f"""
    SELECT hltv_match_id, date, team1_name, team2_name,
           team1_score, team2_score, event_type
    FROM '{PARQUET_DIR}/matches.parquet'
""").df()

team1_stats = rolling_df[rolling_df['side'] == 'team1'][[
    'hltv_match_id', 'rolling_rating_l10', 'rolling_kast_l10',
    'rolling_swing_l10', 'rolling_win_rate_l10'
]].rename(columns={
    'rolling_rating_l10':   'team1_rolling_rating_l10',
    'rolling_kast_l10':     'team1_rolling_kast_l10',
    'rolling_swing_l10':    'team1_rolling_swing_l10',
    'rolling_win_rate_l10': 'team1_rolling_win_rate_l10',
})

team2_stats = rolling_df[rolling_df['side'] == 'team2'][[
    'hltv_match_id', 'rolling_rating_l10', 'rolling_kast_l10',
    'rolling_swing_l10', 'rolling_win_rate_l10'
]].rename(columns={
    'rolling_rating_l10':   'team2_rolling_rating_l10',
    'rolling_kast_l10':     'team2_rolling_kast_l10',
    'rolling_swing_l10':    'team2_rolling_swing_l10',
    'rolling_win_rate_l10': 'team2_rolling_win_rate_l10',
})

features_df = (
    matches_df
    .merge(team1_stats, on='hltv_match_id', how='left')
    .merge(team2_stats, on='hltv_match_id', how='left')
)

# Step 4: save as parquet
output_path = os.path.join(FEATURES_DIR, "features_rolling_l10.parquet")
features_df.to_parquet(output_path, index=False)
print(f"✅ Saved {len(features_df)} rows to {output_path}")
print(features_df[['hltv_match_id', 'team1_name', 'team2_name',
                    'team1_rolling_rating_l10', 'team2_rolling_rating_l10',
                    'team1_rolling_kast_l10', 'team2_rolling_kast_l10',
                    'team1_rolling_swing_l10', 'team2_rolling_swing_l10',
                    'team1_rolling_win_rate_l10', 'team2_rolling_win_rate_l10']].head(10).to_string())
