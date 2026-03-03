import duckdb
import pandas as pd
import numpy as np
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
PARQUET_DIR   = "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/parquet"
FEATURES_DIR  = "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/features"
FEATURES_FILE = "features_map_winrate_l5.parquet"
WINDOW        = 5
VERBOSE_PER_MAP = 2
BULK_PER_MAP    = 50

MAPS = ['ancient', 'anubis', 'dust2', 'inferno', 'mirage', 'nuke', 'overpass', 'vertigo']

# ── Load data ─────────────────────────────────────────────────────────────────
df_features = pd.read_parquet(os.path.join(FEATURES_DIR, FEATURES_FILE))
# Cast to plain int — arrow/parquet sometimes reads IDs as large_string
df_features['hltv_match_id'] = df_features['hltv_match_id'].astype(int)

con = duckdb.connect()

map_results = con.execute(f"""
    SELECT CAST(mp.hltv_match_id AS BIGINT) AS hltv_match_id,
           mp.map_name, mp.score,
           m.team1_name, m.team2_name
    FROM '{PARQUET_DIR}/maps.parquet' mp
    JOIN '{PARQUET_DIR}/matches.parquet' m ON mp.hltv_match_id = m.hltv_match_id
    ORDER BY mp.hltv_match_id
""").df()

map_results['map_name']      = map_results['map_name'].str.lower().str.strip()
map_results['hltv_match_id'] = map_results['hltv_match_id'].astype(int)

# ── Rebuild team-map history ──────────────────────────────────────────────────
rows = []
for _, r in map_results.iterrows():
    try:
        t1s, t2s = map(int, r['score'].split(':'))
    except Exception:
        continue
    if t1s == t2s:
        continue
    t1w = 1 if t1s > t2s else 0
    rows.append({'hltv_match_id': int(r['hltv_match_id']), 'map_name': r['map_name'],
                 'team_name': r['team1_name'], 'map_won': t1w})
    rows.append({'hltv_match_id': int(r['hltv_match_id']), 'map_name': r['map_name'],
                 'team_name': r['team2_name'], 'map_won': 1 - t1w})

team_map_rows = (
    pd.DataFrame(rows)
    .dropna(subset=['team_name'])
    .sort_values(['team_name', 'map_name', 'hltv_match_id'])
    .reset_index(drop=True)
)
team_map_rows['hltv_match_id'] = team_map_rows['hltv_match_id'].astype(int)

print(f"Loaded {len(df_features)} feature rows, {len(team_map_rows)} team-map history rows")


# ── Manual recompute ──────────────────────────────────────────────────────────
def compute_manual_wr(team, map_name, match_id):
    mid = int(match_id)
    plays = team_map_rows[
        (team_map_rows['team_name']     == team) &
        (team_map_rows['map_name']      == map_name.lower()) &
        (team_map_rows['hltv_match_id'] <  mid)
    ].sort_values('hltv_match_id')

    if plays.empty:
        return None, pd.DataFrame()
    last_n = plays.tail(WINDOW)
    return float(last_n['map_won'].mean()), last_n


# ── Validate one sample ───────────────────────────────────────────────────────
def validate_sample(match_id, team_pos, team_name, map_name, verbose=True):
    match_id = int(match_id)
    row = df_features[df_features['hltv_match_id'] == match_id]
    if row.empty:
        return None, None, "NOT FOUND"

    feature_col = f"{team_pos}_{map_name.lower()}_wr_l5"
    if feature_col not in row.columns:
        return None, None, f"MISSING COL"

    raw = row[feature_col].values[0]
    feature_wr = None if pd.isna(raw) else float(raw)
    manual_wr, last_plays = compute_manual_wr(team_name, map_name, match_id)

    both_nan = feature_wr is None and manual_wr is None
    is_match = both_nan or (
        feature_wr is not None and manual_wr is not None and
        round(feature_wr, 6) == round(manual_wr, 6)
    )
    status = "MATCH" if is_match else "MISMATCH"

    if verbose:
        print("─" * 68)
        print(f"  Match ID : {match_id}  |  {team_name} ({team_pos})  |  Map: {map_name}")
        print()
        if last_plays.empty:
            print("  No prior plays found  ->  expected wr = None")
        else:
            print(f"  Last {len(last_plays)} plays on {map_name} before this match:")
            print()
            for i, (_, p) in enumerate(last_plays.iterrows(), 1):
                res = "WIN  v" if p['map_won'] == 1 else "LOSS x"
                print(f"    [{i}] match {int(p['hltv_match_id'])}  ->  {res}")
            wins  = int(last_plays['map_won'].sum())
            total = len(last_plays)
            print()
            print(f"  Calculation : {wins} wins / {total} games = {wins/total:.4f}  ({wins}/{total})")
        fw_str = f"{feature_wr:.4f}" if feature_wr is not None else "None"
        mw_str = f"{manual_wr:.4f}" if manual_wr is not None else "None"
        print()
        print(f"  Feature WR (parquet) : {fw_str}")
        print(f"  Manual  WR (recomp.) : {mw_str}")
        print(f"  Result               : {'OK' if is_match else '!! MISMATCH !!'}")
        print()

    return feature_wr, manual_wr, status


# ── Run ───────────────────────────────────────────────────────────────────────
print("\n" + "=" * 68)
print("  MAP WIN-RATE L5  -  VISUAL VALIDATION")
print("=" * 68 + "\n")

all_results = []

for map_name in MAPS:
    for team_pos in ['team1', 'team2']:
        col  = f"{team_pos}_{map_name}_wr_l5"
        if col not in df_features.columns:
            continue
        pool = df_features[df_features[col].notna()]
        if pool.empty:
            continue

        verbose_samples = pool.sample(min(VERBOSE_PER_MAP, len(pool)), random_state=42)
        for _, r in verbose_samples.iterrows():
            fw, mw, st = validate_sample(
                r['hltv_match_id'], team_pos, r[f'{team_pos}_name'], map_name, verbose=True)
            all_results.append({'map': map_name, 'team_pos': team_pos,
                                 'team': r[f'{team_pos}_name'],
                                 'match_id': int(r['hltv_match_id']),
                                 'feature_wr': fw, 'manual_wr': mw, 'status': st})

        bulk_pool    = pool.drop(verbose_samples.index, errors='ignore')
        bulk_samples = bulk_pool.sample(min(BULK_PER_MAP, len(bulk_pool)), random_state=99)
        for _, r in bulk_samples.iterrows():
            fw, mw, st = validate_sample(
                r['hltv_match_id'], team_pos, r[f'{team_pos}_name'], map_name, verbose=False)
            all_results.append({'map': map_name, 'team_pos': team_pos,
                                 'team': r[f'{team_pos}_name'],
                                 'match_id': int(r['hltv_match_id']),
                                 'feature_wr': fw, 'manual_wr': mw, 'status': st})

# ── Summary ───────────────────────────────────────────────────────────────────
res        = pd.DataFrame(all_results)
total      = len(res)
n_match    = (res['status'] == 'MATCH').sum()
n_mismatch = (res['status'] == 'MISMATCH').sum()
n_error    = total - n_match - n_mismatch

print("=" * 68)
print("  SUMMARY")
print("=" * 68)
print(f"  Total checked : {total}")
print(f"  Matches       : {n_match}  ({100*n_match/total:.1f}%)")
print(f"  Mismatches    : {n_mismatch}  ({100*n_mismatch/total:.1f}%)")
print(f"  Errors        : {n_error}")
print()
print("  Per-map breakdown:")
for m in MAPS:
    sub = res[res['map'] == m]
    if sub.empty:
        continue
    ok  = (sub['status'] == 'MATCH').sum()
    bad = (sub['status'] == 'MISMATCH').sum()
    print(f"    {m:<12}  {ok:>3} ok  {bad:>3} mismatch  (n={len(sub)})")
print()
if n_mismatch > 0:
    print("  MISMATCHES DETAIL:")
    print(res[res['status'] == 'MISMATCH'][
        ['match_id', 'team', 'map', 'feature_wr', 'manual_wr']
    ].to_string(index=False))
else:
    print("  All samples match - L5 feature looks correct!")
