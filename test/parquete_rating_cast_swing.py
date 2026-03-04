import duckdb
import pandas as pd
import os
import re
import math

# ── Paths ─────────────────────────────────────────────────────────────────────
PARQUET_DIR   = "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/parquet"
FEATURES_DIR  = "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/features"
FEATURES_FILE = "features_rolling_l10.parquet"

window_match = re.search(r'_l(\d+)\.parquet', FEATURES_FILE)
if window_match:
    WINDOW = int(window_match.group(1))
    SUFFIX = f"l{WINDOW}"
else:
    raise ValueError(f"Could not determine window size from filename: {FEATURES_FILE}")

TOTAL_SAMPLES = 2000

STATS_MAP = {
    'rating': 'rolling_rating',
    'kast_pct': 'rolling_kast',
    'swing_pct': 'rolling_swing',
    'won': 'rolling_win_rate'
}

# ── Load data ─────────────────────────────────────────────────────────────────
df_features = pd.read_parquet(os.path.join(FEATURES_DIR, FEATURES_FILE))
df_features['hltv_match_id'] = df_features['hltv_match_id'].astype(int)
# Strictly enforce chronological order on the features dataframe
df_features = df_features.sort_values('hltv_match_id').reset_index(drop=True)

con = duckdb.connect()

base_history = con.execute(f"""
    WITH round_counts AS (
        SELECT hltv_match_id, COUNT(*) AS total_rounds
        FROM '{PARQUET_DIR}/rounds.parquet'
        GROUP BY hltv_match_id
    ),
    team_rows AS (
        SELECT m.hltv_match_id, m.team1_name AS team_name,
               CASE WHEN m.team1_score > m.team2_score THEN 1 ELSE 0 END AS won,
               m.team1_rating_3 AS rating, m.team1_kast_pct AS kast_pct,
               m.team1_swing_pct AS swing_pct, r.total_rounds
        FROM '{PARQUET_DIR}/matches.parquet' m
        JOIN round_counts r ON m.hltv_match_id = r.hltv_match_id

        UNION ALL

        SELECT m.hltv_match_id, m.team2_name AS team_name,
               CASE WHEN m.team2_score > m.team1_score THEN 1 ELSE 0 END AS won,
               m.team2_rating_3 AS rating, m.team2_kast_pct AS kast_pct,
               m.team2_swing_pct AS swing_pct, r.total_rounds
        FROM '{PARQUET_DIR}/matches.parquet' m
        JOIN round_counts r ON m.hltv_match_id = r.hltv_match_id
    )
    SELECT * FROM team_rows
    ORDER BY hltv_match_id ASC
""").df()

base_history['hltv_match_id'] = base_history['hltv_match_id'].astype(int)
# Double enforce chronological sorting in pandas
base_history = base_history.sort_values(['team_name', 'hltv_match_id']).reset_index(drop=True)

print(f"Loaded {len(df_features)} feature rows, {len(base_history)} base history rows")

# ── Manual recompute ──────────────────────────────────────────────────────────
# ── Manual recompute ──────────────────────────────────────────────────────────
def compute_manual_stats(team, match_id):
    mid = int(match_id)
    # Strictly filter by matches chronologically older than the current match ID
    plays = base_history[
        (base_history['team_name'] == team) &
        (base_history['hltv_match_id'] < mid)
    ].sort_values('hltv_match_id')

    if plays.empty:
        return {s: None for s in STATS_MAP.keys()}, pd.DataFrame()

    last_n = plays.tail(WINDOW)
    res = {}

    for base_stat in STATS_MAP.keys():
        if base_stat == 'won':
            # Normal, unweighted winrate (flat average of 1s and 0s)
            res[base_stat] = float(last_n[base_stat].mean())
        else:
            # Round-weighted average for rating, kast, and swing
            weights = last_n['total_rounds']
            values = last_n[base_stat]

            if weights.sum() > 0:
                res[base_stat] = float((values * weights).sum() / weights.sum())
            else:
                res[base_stat] = None

    return res, last_n

# ── Validate one sample ───────────────────────────────────────────────────────
def validate_sample(match_id, team_pos, team_name):
    match_id = int(match_id)
    row = df_features[df_features['hltv_match_id'] == match_id]
    if row.empty:
        return {"status": "NOT FOUND"}

    manual_stats, last_plays = compute_manual_stats(team_name, match_id)

    results = {}
    all_match = True
    log_lines = []

    log_lines.append("─" * 75)
    log_lines.append(f"  ❌ MISMATCH DETECTED -> Match ID: {match_id} | {team_name} ({team_pos}) | Window: {WINDOW}")
    log_lines.append("")

    if last_plays.empty:
        log_lines.append("  No prior plays found -> expected values = None")
    else:
        log_lines.append(f"  Last {len(last_plays)} matches before this match:")
        for i, (_, p) in enumerate(last_plays.iterrows(), 1):
            log_lines.append(f"    [{i}] Match {int(p['hltv_match_id'])} | Rounds: {int(p['total_rounds'])} | Rating: {p['rating']:.2f} | Win: {int(p['won'])}")
    log_lines.append("")

    for base_stat, feature_name in STATS_MAP.items():
        feature_col = f"{team_pos}_{feature_name}_{SUFFIX}"

        if feature_col not in row.columns:
            all_match = False
            log_lines.append(f"  {feature_name:<16} | MISSING COLUMN: {feature_col}")
            continue

        raw_val = row[feature_col].values[0]
        feat_val = None if pd.isna(raw_val) else float(raw_val)
        man_val = manual_stats[base_stat]

        # PROPER ROUNDING CHECK: Ignore floating point errors up to 4 decimal places
        if pd.isna(feat_val) and pd.isna(man_val):
            is_match = True
        elif pd.isna(feat_val) or pd.isna(man_val):
            is_match = False
        else:
            is_match = math.isclose(feat_val, man_val, abs_tol=1e-4)

        if not is_match:
            all_match = False

        f_str = f"{feat_val:.4f}" if feat_val is not None else "None"
        m_str = f"{man_val:.4f}" if man_val is not None else "None"
        res_str = "OK" if is_match else "!! MISMATCH !!"
        log_lines.append(f"  {feature_name:<16} | Feat: {f_str:>8} | Man: {m_str:>8} | {res_str}")

    if not all_match:
        print("\n".join(log_lines))
        print()

    return {"status": "MATCH" if all_match else "MISMATCH"}

# ── Run ───────────────────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print(f"  ROLLING TEAM STATS L{WINDOW} - SILENT VALIDATION (Testing {TOTAL_SAMPLES} samples)")
print("=" * 75 + "\n")

all_results = []
valid_pool = df_features.dropna(subset=[f"team1_rolling_rating_{SUFFIX}"])

test_samples = valid_pool.sample(min(TOTAL_SAMPLES, len(valid_pool)), random_state=42)

for _, r in test_samples.iterrows():
    for team_pos in ['team1', 'team2']:
        res = validate_sample(r['hltv_match_id'], team_pos, r[f'{team_pos}_name'])
        all_results.append({"match_id": r['hltv_match_id'], "team": r[f'{team_pos}_name'], "status": res["status"]})

# ── Summary ───────────────────────────────────────────────────────────────────
res_df = pd.DataFrame(all_results)
total = len(res_df)
n_match = (res_df['status'] == 'MATCH').sum()
n_mismatch = (res_df['status'] == 'MISMATCH').sum()

print("=" * 75)
print("  SUMMARY")
print("=" * 75)
print(f"  Total checked : {total} (team-match instances)")
print(f"  Matches       : {n_match}  ({100*n_match/total:.1f}%)")
print(f"  Mismatches    : {n_mismatch}  ({100*n_mismatch/total:.1f}%)")
print()

if n_mismatch > 0:
    print("  WARNING: Mismatches found! See the detailed logs printed above.")
else:
    print(f"  ✅ SUCCESS: All {total} samples match perfectly. Your logic is clean.")
