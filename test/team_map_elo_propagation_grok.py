import pandas as pd

FEATURES_DIR = "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/features"

TEAM = "Natus Vincere"
MAP  = "inferno"

df = pd.read_parquet(f"{FEATURES_DIR}/features_map_elo_grok.parquet")

navi = df[(df['team1_name'] == TEAM) | (df['team2_name'] == TEAM)].copy()
navi = navi.sort_values('date').reset_index(drop=True)

def get_navi_elo(row):
    if row['team1_name'] == TEAM:
        return row[f'team1_{MAP}_elo'], row['team2_name']
    else:
        return row[f'team2_{MAP}_elo'], row['team1_name']

navi[['navi_elo', 'opponent']] = navi.apply(
    lambda r: pd.Series(get_navi_elo(r)), axis=1
)

out = navi[['hltv_match_id', 'date', 'opponent', 'navi_elo']].copy()
out.columns = ['match_id', 'date', 'opponent', f'navi_{MAP}_elo']

pd.set_option('display.max_rows', None)
pd.set_option('display.width', 0)
pd.set_option('display.float_format', '{:.1f}'.format)

print(f"\n{TEAM} — {MAP.upper()} Elo progression GROK ({len(out)} matches)\n")
print(out.to_string(index=True))
