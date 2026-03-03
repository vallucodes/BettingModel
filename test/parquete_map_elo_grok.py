import pandas as pd
import os

FEATURES_DIR = "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/features"

output_path = os.path.join(FEATURES_DIR, "features_map_elo_grok.parquet")

df = pd.read_parquet(output_path)

df = df.sort_values('hltv_match_id')

columns = [
    'hltv_match_id', 'team1_name', 'team2_name',
    'team1_ancient_elo', 'team1_anubis_elo', 'team1_cache_elo', 'team1_dust2_elo',
    'team1_inferno_elo', 'team1_mirage_elo', 'team1_nuke_elo', 'team1_overpass_elo',
    'team1_train_elo', 'team1_vertigo_elo',
    'team2_ancient_elo', 'team2_anubis_elo', 'team2_cache_elo', 'team2_dust2_elo',
    'team2_inferno_elo', 'team2_mirage_elo', 'team2_nuke_elo', 'team2_overpass_elo',
    'team2_train_elo', 'team2_vertigo_elo'
]

tail_df = df[columns].tail(10)

# Transpose so features are rows, matches are columns
transposed = tail_df.set_index('hltv_match_id').T

pd.set_option('display.float_format', '{:.1f}'.format)


print(transposed.to_string())
