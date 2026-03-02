import duckdb
import pandas as pd
import os

PARQUET_DIR = "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/parquet"
FEATURES_DIR = "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/features"
os.makedirs(FEATURES_DIR, exist_ok=True)

con = duckdb.connect()

# Step 1: Build per-team per-map per-match view
# Join maps.parquet with matches.parquet to get winner and date
map_results = con.execute(f"""
    SELECT
        mp.hltv_match_id,
        m.date,
        mp.map_name,
        mp.score,
        m.team1_name,
        m.team2_name,
        m.team1_score,
        m.team2_score
    FROM '{PARQUET_DIR}/maps.parquet' mp
    JOIN '{PARQUET_DIR}/matches.parquet' m ON mp.hltv_match_id = m.hltv_match_id
    ORDER BY m.date
""").df()

# Parse map score to get map winner
def parse_map_winner(row):
    try:
        t1, t2 = map(int, row['score'].split(':'))
        if t1 > t2:
            return row['team1_name'], row['team2_name']  # team1 won, team2 lost
        else:
            return row['team2_name'], row['team1_name']  # team2 won, team1 lost
    except:
        return None, None

map_results['map_winner'], map_results['map_loser'] = zip(*map_results.apply(parse_map_winner, axis=1))

# Step 2: Unpivot so each map appears twice (once per team)
winner_rows = map_results[['hltv_match_id', 'date', 'map_name', 'map_winner']].copy()
winner_rows['team_name'] = winner_rows['map_winner']
winner_rows['map_won'] = 1

loser_rows = map_results[['hltv_match_id', 'date', 'map_name', 'map_loser']].copy()
loser_rows['team_name'] = loser_rows['map_loser']
loser_rows['map_won'] = 0

team_map_rows = pd.concat([
    winner_rows[['hltv_match_id', 'date', 'map_name', 'team_name', 'map_won']],
    loser_rows[['hltv_match_id', 'date', 'map_name', 'team_name', 'map_won']]
]).sort_values(['team_name', 'map_name', 'date']).reset_index(drop=True)

# Drop rows where team_name is None (maps with no score data)
team_map_rows = team_map_rows.dropna(subset=['team_name'])

# Get all matches with teams and dates
matches_df = con.execute(f"""
    SELECT hltv_match_id, date, team1_name, team2_name
    FROM '{PARQUET_DIR}/matches.parquet'
""").df()

# Unpivot to team-matches (each team-match instance)
team1_matches = matches_df[['hltv_match_id', 'date', 'team1_name']].rename(columns={'team1_name': 'team_name'})
team2_matches = matches_df[['hltv_match_id', 'date', 'team2_name']].rename(columns={'team2_name': 'team_name'})
team_matches = pd.concat([team1_matches, team2_matches]).sort_values(['team_name', 'date']).drop_duplicates()

# Get unique maps (from your data or hardcode active ones)
maps = ['ancient', 'anubis', 'cache', 'dust2', 'inferno', 'mirage', 'nuke', 'overpass', 'train', 'vertigo']

# Initialize features_df with all matches
features_df = matches_df.copy()
for prefix in ['team1', 'team2']:
    for m in maps:
        features_df[f'{prefix}_{m}_wr_l10'] = None

# Loop over teams to compute features (efficient in practice)
for team, team_match_group in team_matches.groupby('team_name'):
    team_match_group = team_match_group.sort_values('date')  # All matches for this team
    team_plays = team_map_rows[team_map_rows['team_name'] == team]  # All plays for this team
    for map_name in maps:
        map_plays = team_plays[team_plays['map_name'].str.lower() == map_name].sort_values('date')
        if map_plays.empty:
            continue
        for idx, match in team_match_group.iterrows():
            prev_plays = map_plays[map_plays['date'] < match['date']]
            if prev_plays.empty:
                wr = None
            else:
                wr = prev_plays['map_won'].tail(10).mean()
            # Assign to features_df
            match_id = match['hltv_match_id']
            if features_df.loc[features_df['hltv_match_id'] == match_id, 'team1_name'].values[0] == team:
                features_df.loc[features_df['hltv_match_id'] == match_id, f'team1_{map_name}_wr_l10'] = wr
            else:
                features_df.loc[features_df['hltv_match_id'] == match_id, f'team2_{map_name}_wr_l10'] = wr

# Save
output_path = os.path.join(FEATURES_DIR, "features_map_winrate_l10.parquet")
features_df.to_parquet(output_path, index=False)
print(f"✅ Saved {len(features_df)} rows to {output_path}")
print(features_df.columns.tolist())
print(features_df.head(5).to_string())
