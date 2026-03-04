import duckdb
import pandas as pd
import numpy as np
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

PARQUET_DIR = "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/parquet"
FEATURES_DIR = "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/features"
os.makedirs(FEATURES_DIR, exist_ok=True)

MAP_ELO_START = 1500
MAPS = ['ancient', 'anubis', 'cache', 'dust2', 'inferno',
        'mirage', 'nuke', 'overpass', 'train', 'vertigo']

con = duckdb.connect()

map_results = con.execute(f"""
    SELECT
        mp.hltv_match_id,
        m.date,
        mp.map_name,
        mp.score,
        m.team1_name,
        m.team2_name
    FROM '{PARQUET_DIR}/maps.parquet' mp
    JOIN '{PARQUET_DIR}/matches.parquet' m
        ON mp.hltv_match_id = m.hltv_match_id
    ORDER BY mp.hltv_match_id ASC
""").df()
map_results['map_name'] = map_results['map_name'].str.lower().str.strip()

matches_df = con.execute(f"""
    SELECT hltv_match_id, date, team1_name, team2_name
    FROM '{PARQUET_DIR}/matches.parquet'
    ORDER BY hltv_match_id ASC
""").df()

maps_by_match = map_results.groupby('hltv_match_id')

def expected_score(e1, e2):
    return 1 / (1 + 10 ** ((e2 - e1) / 400))

matches_template = matches_df.copy()


def make_map_state():
    return {m: {'elo': float(MAP_ELO_START), 'games': 0} for m in MAPS}


def weighted_avg_elo(row, prefix):
    elos = np.array([row[f'{prefix}_{m}_elo'] for m in MAPS], dtype=float)
    games = np.array([row[f'{prefix}_{m}_games'] for m in MAPS], dtype=float)
    if games.sum() == 0:
        return float(MAP_ELO_START)
    return np.average(elos, weights=np.maximum(games, 0.1))


def build_map_elo_features(high_k: int, low_k: int, threshold: int) -> None:
    """
    Build map-level Elo features with given K-factor and threshold,
    and save to features_map_elo_HIGH_K_LOW_K_THRESHOLD.parquet.
    """
    matches_df = matches_template.copy()

    # Per-team, per-map elo state
    elo_state = defaultdict(make_map_state)

    # Init as float so pandas doesn't create int64 columns
    for m in MAPS:
        matches_df[f'team1_{m}_elo'] = float(MAP_ELO_START)
        matches_df[f'team2_{m}_elo'] = float(MAP_ELO_START)
        matches_df[f'{m}_elo_diff'] = 0.0
        matches_df[f'team1_{m}_games'] = 0
        matches_df[f'team2_{m}_games'] = 0

    matches_df = matches_df.set_index('hltv_match_id')

    # Loop all matches
    for mid, row in matches_df[['date', 'team1_name', 'team2_name']].iterrows():
        team1 = row['team1_name']
        team2 = row['team2_name']

        for m in MAPS:
            s1 = elo_state[team1][m]
            s2 = elo_state[team2][m]
            matches_df.at[mid, f'team1_{m}_elo'] = s1['elo']
            matches_df.at[mid, f'team2_{m}_elo'] = s2['elo']
            matches_df.at[mid, f'team1_{m}_games'] = s1['games']
            matches_df.at[mid, f'team2_{m}_games'] = s2['games']
            matches_df.at[mid, f'{m}_elo_diff'] = s1['elo'] - s2['elo']

        if mid not in maps_by_match.groups:
            continue

        for _, pmap in maps_by_match.get_group(mid).iterrows():
            map_name = pmap['map_name']
            if map_name not in MAPS:
                continue
            try:
                t1_score, t2_score = map(int, pmap['score'].split(':'))
            except Exception:
                continue
            if t1_score == t2_score:
                continue

            winner = team1 if t1_score > t2_score else team2
            loser = team2 if t1_score > t2_score else team1

            sw = elo_state[winner][map_name]
            sl = elo_state[loser][map_name]
            kw = high_k if sw['games'] < threshold else low_k
            kl = high_k if sl['games'] < threshold else low_k

            ea = expected_score(sw['elo'], sl['elo'])
            elo_state[winner][map_name]['elo'] = sw['elo'] + kw * (1 - ea)
            elo_state[loser][map_name]['elo'] = sl['elo'] + kl * (0 - (1 - ea))
            elo_state[winner][map_name]['games'] += 1
            elo_state[loser][map_name]['games'] += 1

    matches_df = matches_df.reset_index()

    # Summary features
    matches_df['team1_best_map_elo'] = matches_df[[f'team1_{m}_elo' for m in MAPS]].max(axis=1)
    matches_df['team2_best_map_elo'] = matches_df[[f'team2_{m}_elo' for m in MAPS]].max(axis=1)
    matches_df['best_map_elo_diff'] = matches_df['team1_best_map_elo'] - matches_df['team2_best_map_elo']

    for prefix in ['team1', 'team2']:
        matches_df[f'{prefix}_map_pool_depth'] = sum(
            (matches_df[f'{prefix}_{m}_elo'] > MAP_ELO_START).astype(int) for m in MAPS
        )
    matches_df['map_pool_depth_diff'] = matches_df['team1_map_pool_depth'] - matches_df['team2_map_pool_depth']

    matches_df['team1_weighted_map_elo'] = matches_df.apply(lambda r: weighted_avg_elo(r, 'team1'), axis=1)
    matches_df['team2_weighted_map_elo'] = matches_df.apply(lambda r: weighted_avg_elo(r, 'team2'), axis=1)
    matches_df['weighted_map_elo_diff'] = matches_df['team1_weighted_map_elo'] - matches_df['team2_weighted_map_elo']

    filename = f"features_map_elo_{high_k}_{low_k}_{threshold}.parquet"
    output_path = os.path.join(FEATURES_DIR, filename)

    # Safety: if file already exists, skip (supports reruns / partial runs)
    if os.path.exists(output_path):
        print(f"⏭️ Skipping existing file {output_path}")
        return

    matches_df.to_parquet(output_path, index=False)
    print(f"✅ Saved {len(matches_df)} rows → {output_path}")


if __name__ == "__main__":
    high_k_values = list(range(50, 201, 20))     # 50–200 step 20
    low_k_values = list(range(10, 101, 10))      # 10–100 step 10
    threshold_values = list(range(5, 31, 2))     # 5–30 step 2

    combos = []
    for h in high_k_values:
        for l in low_k_values:
            for t in threshold_values:
                filename = f"features_map_elo_{h}_{l}_{t}.parquet"
                path = os.path.join(FEATURES_DIR, filename)
                if os.path.exists(path):
                    print(f"Already have {filename}, skipping in queue build.")
                    continue
                combos.append((h, l, t))
    total = len(combos)
    print(f"Total combinations to run: {total}")

    # Use multiple processes to parallelize over parameter combinations,
    # but always leave at least one core free.
    cpu_count = os.cpu_count() or 4
    max_workers = max(1, cpu_count - 1)
    print(f"Using up to {max_workers} processes.")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_params = {
            executor.submit(build_map_elo_features, h, l, t): (h, l, t)
            for (h, l, t) in combos
        }

        for i, future in enumerate(as_completed(future_to_params), start=1):
            h, l, t = future_to_params[future]
            try:
                future.result()
                print(f"[{i}/{total}] Done HIGH_K={h}, LOW_K={l}, THRESHOLD={t}")
            except Exception as e:
                print(f"[{i}/{total}] FAILED HIGH_K={h}, LOW_K={l}, THRESHOLD={t}: {e}")
