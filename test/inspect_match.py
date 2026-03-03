import duckdb

PARQUET_DIR = "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/parquet"
con = duckdb.connect()

result = con.execute(f"""
    SELECT mp.hltv_match_id, m.date, mp.map_name, mp.score,
           m.team1_name, m.team2_name
    FROM '{PARQUET_DIR}/maps.parquet' mp
    JOIN '{PARQUET_DIR}/matches.parquet' m ON mp.hltv_match_id = m.hltv_match_id
    WHERE mp.hltv_match_id = 2375776
""").df()

print(result.to_string())
