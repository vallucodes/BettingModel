import duckdb
import pandas as pd
import os

PARQUET_DIR = "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/parquet"
FEATURES_DIR = "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/features"
os.makedirs(FEATURES_DIR, exist_ok=True)

def load_base_data():
    """Loads and prepares the base matches and team-map rows."""
    con = duckdb.connect()

    map_results = con.execute(f"""
        SELECT mp.hltv_match_id, m.date, mp.map_name, mp.score,
               m.team1_name, m.team2_name, m.team1_score, m.team2_score
        FROM '{PARQUET_DIR}/maps.parquet' mp
        JOIN '{PARQUET_DIR}/matches.parquet' m ON mp.hltv_match_id = m.hltv_match_id
        ORDER BY m.hltv_match_id ASC
    """).df()

    def parse_map_winner(row):
        try:
            t1, t2 = map(int, row['score'].split(':'))
            if t1 > t2:
                return row['team1_name'], row['team2_name']
            else:
                return row['team2_name'], row['team1_name']
        except:
            return None, None

    map_results['map_winner'], map_results['map_loser'] = zip(*map_results.apply(parse_map_winner, axis=1))

    winner_rows = map_results[['hltv_match_id', 'map_name', 'map_winner']].copy()
    winner_rows['team_name'] = winner_rows['map_winner']
    winner_rows['map_won'] = 1

    loser_rows = map_results[['hltv_match_id', 'map_name', 'map_loser']].copy()
    loser_rows['team_name'] = loser_rows['map_loser']
    loser_rows['map_won'] = 0

    team_map_rows = pd.concat([
        winner_rows[['hltv_match_id', 'map_name', 'team_name', 'map_won']],
        loser_rows[['hltv_match_id', 'map_name', 'team_name', 'map_won']]
    ]).dropna(subset=['team_name'])

    matches_df = con.execute(f"""
        SELECT hltv_match_id, date, team1_name, team2_name
        FROM '{PARQUET_DIR}/matches.parquet'
        ORDER BY hltv_match_id ASC
    """).df()

    con.close()
    return team_map_rows, matches_df

def generate_map_winrates(window_size, team_map_rows, matches_df):
    """Calculates rolling winrates for a specific window size and saves to parquet."""
    print(f"🔄 Processing rolling map winrates for l{window_size}...")

    team1_matches = matches_df[['hltv_match_id', 'team1_name']].rename(columns={'team1_name': 'team_name'})
    team2_matches = matches_df[['hltv_match_id', 'team2_name']].rename(columns={'team2_name': 'team_name'})
    team_matches = pd.concat([team1_matches, team2_matches]).drop_duplicates()

    maps = ['ancient', 'anubis', 'cache', 'dust2', 'inferno', 'mirage', 'nuke', 'overpass', 'train', 'vertigo']

    tm = team_map_rows.copy()
    tm['map_name'] = tm['map_name'].str.lower()
    tm = tm[tm['map_name'].isin(maps)]
    tm = tm.sort_values(['team_name', 'map_name', 'hltv_match_id'])

    # Dynamic column name based on window size
    wr_col = f'wr_l{window_size}'

    tm[wr_col] = tm.groupby(['team_name', 'map_name'])['map_won'].transform(
        lambda x: x.rolling(window=window_size, min_periods=1).mean()
    )

    map_states = tm.pivot_table(
        index=['team_name', 'hltv_match_id'],
        columns='map_name',
        values=wr_col,
        aggfunc='last'
    ).reset_index()

    all_team_matches = team_matches.sort_values(['team_name', 'hltv_match_id'])
    team_history = pd.merge(all_team_matches, map_states, on=['team_name', 'hltv_match_id'], how='left')

    map_cols = [m for m in maps if m in team_history.columns]

    team_history[map_cols] = team_history.groupby('team_name')[map_cols].transform(
        lambda x: x.shift(1).ffill()
    )

    for m in maps:
        if m not in team_history.columns:
            team_history[m] = None

    features_df = matches_df.copy()

    # Dynamic renaming for Team 1
    t1_rename = {m: f'team1_{m}_{wr_col}' for m in maps}
    t1_rename['team_name'] = 'team1_name'
    t1_history = team_history[['hltv_match_id', 'team_name'] + maps].rename(columns=t1_rename)
    features_df = pd.merge(features_df, t1_history, on=['hltv_match_id', 'team1_name'], how='left')

    # Dynamic renaming for Team 2
    t2_rename = {m: f'team2_{m}_{wr_col}' for m in maps}
    t2_rename['team_name'] = 'team2_name'
    t2_history = team_history[['hltv_match_id', 'team_name'] + maps].rename(columns=t2_rename)
    features_df = pd.merge(features_df, t2_history, on=['hltv_match_id', 'team2_name'], how='left')

    features_df = features_df.sort_values('hltv_match_id').reset_index(drop=True)

    output_path = os.path.join(FEATURES_DIR, f"features_map_winrate_l{window_size}.parquet")
    features_df.to_parquet(output_path, index=False)

    print(f"✅ Saved l{window_size} features to {output_path}")

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    # Load base data once to save memory/time
    base_team_maps, base_matches = load_base_data()

    # Generate features for multiple window sizes in one go
    windows_to_generate = [10, 15]

    for w in windows_to_generate:
        generate_map_winrates(w, base_team_maps, base_matches)
