import pandas as pd

# Adjust these paths to your local files
PARQUET_DIR = "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/parquet"
FEATURES_DIR = "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/features"

# Load data
matches = pd.read_parquet(f"{PARQUET_DIR}/matches.parquet")
features = pd.read_parquet(f"{FEATURES_DIR}/features_rolling_l10.parquet")

# Merge relevant columns (assumes features has no team names; we get them from matches)
df = matches[['hltv_match_id', 'team1_name', 'team2_name']].merge(
    features[['hltv_match_id', 'team1_rolling_rating_l10', 'team2_rolling_rating_l10',
              'team1_rolling_kast_l10', 'team2_rolling_kast_l10',
              'team1_rolling_swing_l10', 'team2_rolling_swing_l10',
              'team1_rolling_win_rate_l10', 'team2_rolling_win_rate_l10']],
    on='hltv_match_id', how='left'
)

# Sort chronologically
df = df.sort_values('hltv_match_id').reset_index(drop=True)

# Track pre-match games played per team
unique_teams = pd.unique(df[['team1_name', 'team2_name']].values.ravel())
games_played = {team: 0 for team in unique_teams}

df['team1_games_pre'] = 0
df['team2_games_pre'] = 0

for idx, row in df.iterrows():
    df.at[idx, 'team1_games_pre'] = games_played[row['team1_name']]
    df.at[idx, 'team2_games_pre'] = games_played[row['team2_name']]
    games_played[row['team1_name']] += 1
    games_played[row['team2_name']] += 1

# Check second matches specifically
print("Verification for second matches (pre-games = 1):")

second_matches_team1 = df[df['team1_games_pre'] == 1]
print(f"Number of second matches as team1: {len(second_matches_team1)}")
print(f" - non-NaN rolling_rating_l10: {second_matches_team1['team1_rolling_rating_l10'].notna().sum()}")
print(f" - non-NaN rolling_kast_l10: {second_matches_team1['team1_rolling_kast_l10'].notna().sum()}")
print(f" - non-NaN rolling_swing_l10: {second_matches_team1['team1_rolling_swing_l10'].notna().sum()}")
print(f" - non-NaN rolling_win_rate_l10: {second_matches_team1['team1_rolling_win_rate_l10'].notna().sum()}")

second_matches_team2 = df[df['team2_games_pre'] == 1]
print(f"\nNumber of second matches as team2: {len(second_matches_team2)}")
print(f" - non-NaN rolling_rating_l10: {second_matches_team2['team2_rolling_rating_l10'].notna().sum()}")
print(f" - non-NaN rolling_kast_l10: {second_matches_team2['team2_rolling_kast_l10'].notna().sum()}")
print(f" - non-NaN rolling_swing_l10: {second_matches_team2['team2_rolling_swing_l10'].notna().sum()}")
print(f" - non-NaN rolling_win_rate_l10: {second_matches_team2['team2_rolling_win_rate_l10'].notna().sum()}")

# Proportion non-NaN by pre-games count (combined team1/team2)
features_list = ['rolling_rating_l10', 'rolling_kast_l10', 'rolling_swing_l10', 'rolling_win_rate_l10']
for feature in features_list:
    print(f"\nProportion non-NaN for {feature} by pre-games count (0=first match, 1=second, etc.):")
    team1_data = df[['team1_games_pre', f'team1_{feature}']].rename(columns={'team1_games_pre': 'games_pre', f'team1_{feature}': feature})
    team2_data = df[['team2_games_pre', f'team2_{feature}']].rename(columns={'team2_games_pre': 'games_pre', f'team2_{feature}': feature})
    combined = pd.concat([team1_data, team2_data])
    grouped = combined.groupby('games_pre')[feature].apply(lambda x: x.notna().mean() if len(x) > 0 else 0)
    print(grouped.head(20))  # Show up to 20 for early matches

# Optional: Example for a specific team (replace 'TeamName' with an actual team from your data)
# team_df = df[(df['team1_name'] == 'TeamName') | (df['team2_name'] == 'TeamName')]
# print(team_df[['hltv_match_id', 'team1_games_pre', 'team2_games_pre', 'team1_rolling_rating_l10', 'team2_rolling_rating_l10']])
