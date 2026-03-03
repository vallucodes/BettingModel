import pandas as pd

FEATURES_DIR = "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/features"
PARQUET_DIR  = "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/parquet"

TEAM = "Natus Vincere"
MAP  = "inferno"

df = pd.read_parquet(f"{FEATURES_DIR}/features_map_elo_claude.parquet")

# Keep only matches where NaVi appears as team1 or team2
navi = df[(df['team1_name'] == TEAM) | (df['team2_name'] == TEAM)].copy()
navi = navi.sort_values('date').reset_index(drop=True)

# Pick the correct elo column depending on which side NaVi is on
def get_navi_elo(row):
    if row['team1_name'] == TEAM:
        return row[f'team1_{MAP}_elo'], row[f'team1_{MAP}_games'], row['team2_name']
    else:
        return row[f'team2_{MAP}_elo'], row[f'team2_{MAP}_games'], row['team1_name']

navi[['navi_elo', 'navi_games', 'opponent']] = navi.apply(
    lambda r: pd.Series(get_navi_elo(r)), axis=1
)

out = navi[['hltv_match_id', 'date', 'opponent', 'navi_elo', 'navi_games']].copy()
out.columns = ['match_id', 'date', 'opponent', f'navi_{MAP}_elo', f'navi_{MAP}_games']

pd.set_option('display.max_rows', None)
pd.set_option('display.width', 0)
pd.set_option('display.float_format', '{:.1f}'.format)

print(f"\n{TEAM} — {MAP.upper()} Elo progression ({len(out)} matches)\n")
print(out.to_string(index=True))
