import pandas as pd

FEATURES_DIR = "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/features"

df = pd.read_parquet(f"{FEATURES_DIR}/features_map_elo_claude.parquet")

df = df.sort_values('hltv_match_id')

cols = [
    'hltv_match_id', 'team1_name', 'team2_name',
    'team1_ancient_elo', 'team1_anubis_elo', 'team1_cache_elo', 'team1_dust2_elo',
    'team1_inferno_elo', 'team1_mirage_elo', 'team1_nuke_elo', 'team1_overpass_elo',
    'team1_train_elo', 'team1_vertigo_elo',
    'team2_ancient_elo', 'team2_anubis_elo', 'team2_cache_elo', 'team2_dust2_elo',
    'team2_inferno_elo', 'team2_mirage_elo', 'team2_nuke_elo', 'team2_overpass_elo',
    'team2_train_elo', 'team2_vertigo_elo',
]

tail20 = df[cols].tail(10).reset_index(drop=True)

# Transpose so features are rows, matches are columns
transposed = tail20.set_index('hltv_match_id').T

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 0)
pd.set_option('display.float_format', '{:.1f}'.format)

print(transposed.to_string())
