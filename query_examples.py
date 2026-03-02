import duckdb

PARQUET_DIR = "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/parquet"
FEATURES_DIR = "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/features"

con = duckdb.connect()

# df = con.execute(f"""
#     SELECT
#         m.date,
#         m.team1_name,
#         m.team2_name,
#         m.team1_score,
#         m.team2_score,
#         map_count.num_maps,
#         p.rating_3,
#         p.swing_pct,
#         SUM(map_count.num_maps) OVER () AS total_maps
#     FROM '{PARQUET_DIR}/players.parquet' p
#     JOIN '{PARQUET_DIR}/matches.parquet' m
#         ON p.hltv_match_id = m.hltv_match_id
#     JOIN (
#         SELECT hltv_match_id, COUNT(*) AS num_maps
#         FROM '{PARQUET_DIR}/maps.parquet'
#         GROUP BY hltv_match_id
#     ) map_count ON p.hltv_match_id = map_count.hltv_match_id
#     WHERE p.player_name = 'zont1x'
#     AND m.date LIKE '2025-%'
#     ORDER BY m.date DESC
# """).df()

# print(f"zont1x matches in 2025: {len(df)} | Total maps played: {df['total_maps'].iloc[0]}")
# print(df.to_string())

# df = con.execute(f"""
#     SELECT
#         p.player_name,
#         COUNT(*)                                               AS matches_played,
#         SUM(round_count.total_rounds)                          AS total_rounds_played,
#         ROUND(SUM(p.rating_3 * round_count.total_rounds) /
#               NULLIF(SUM(round_count.total_rounds), 0), 3)     AS avg_rating_weighted,
#         ROUND(SUM(p.adr * round_count.total_rounds) /
#               NULLIF(SUM(round_count.total_rounds), 0), 1)     AS avg_adr_weighted,
#         ROUND(SUM(p.kast_pct * round_count.total_rounds) /
#               NULLIF(SUM(round_count.total_rounds), 0), 1)     AS avg_kast_weighted,
#         ROUND(SUM(p.swing_pct * round_count.total_rounds) /
#               NULLIF(SUM(round_count.total_rounds), 0), 3)     AS avg_swing_weighted,
#         ROUND(SUM(p.kills) * 1.0 /
#               NULLIF(SUM(p.deaths), 0), 3)                     AS overall_kd_ratio
#     FROM '{PARQUET_DIR}/players.parquet' p
#     JOIN '{PARQUET_DIR}/matches.parquet' m
#         ON p.hltv_match_id = m.hltv_match_id
#     JOIN (
#         SELECT hltv_match_id, COUNT(*) AS total_rounds
#         FROM '{PARQUET_DIR}/rounds.parquet'
#         GROUP BY hltv_match_id
#     ) round_count ON p.hltv_match_id = round_count.hltv_match_id
#     WHERE (
#         (m.team1_name = 'Passion UA' AND p.team = 'team1')
#         OR (m.team2_name = 'Passion UA' AND p.team = 'team2')
#     )
#     AND m.date LIKE '2025-%'
#     GROUP BY p.player_name
#     ORDER BY avg_rating_weighted DESC
# """).df()

# print(df.to_string())

team = 'Spirit'

# Simple: all matches for a specific team
df = con.execute(f"""
    SELECT date, team1_name, team2_name, team1_score, team2_score
    FROM '{PARQUET_DIR}/matches.parquet'
    WHERE team1_name = '{team}' OR team2_name = '{team}'
    ORDER BY date ASC
""").df()
print(df.to_string())

df = con.execute(f"""
    SELECT
        m.date,
        m.team1_name,
        m.team2_name,
        m.team1_score,
        m.team2_score,
        -- Actual stats from that match
        CASE WHEN m.team1_name = '{team}' THEN m.team1_kast_pct
             ELSE m.team2_kast_pct END AS match_kast,
        -- Rolling L10 stats saved BEFORE this match was played
        CASE WHEN m.team1_name = '{team}' THEN f.team1_rolling_kast_l10
             ELSE f.team2_rolling_kast_l10 END AS rolling_kast_l10,
        -- Total rounds for manual weighted verification
        rc.total_rounds
    FROM '{PARQUET_DIR}/matches.parquet' m
    JOIN '{FEATURES_DIR}/features_rolling_l10.parquet' f
        ON m.hltv_match_id = f.hltv_match_id
    JOIN (
        SELECT hltv_match_id, COUNT(*) AS total_rounds
        FROM '{PARQUET_DIR}/rounds.parquet'
        GROUP BY hltv_match_id
    ) rc ON m.hltv_match_id = rc.hltv_match_id
    WHERE m.team1_name = '{team}' OR m.team2_name = '{team}'
    ORDER BY m.date ASC
""").df()

print(df.to_string())

