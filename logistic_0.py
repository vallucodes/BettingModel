import pandas as pd
import numpy as np
import duckdb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from datetime import datetime
import os

SAVE_RESULTS = False
RESULTS_FILE = "/media/vallu/Storage/Coding/Own_projects/betting_model/model/results_log.csv"
RUN_NOTE = "logistic"

PARQUET_DIR = "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/parquet"
con = duckdb.connect()

# K parameters: How much one match infuences ELO change
# HIGH_K: Used for ELO calibration
# LOW_K: Used after ELO is calibrated
# THRESHOLD: Matches after which to switch to LOW_K
HIGH_K = 100
LOW_K = 20
THRESHOLD = 30

df = con.execute(f"""
    SELECT hltv_match_id, team1_name, team2_name, team1_score, team2_score
    FROM '{PARQUET_DIR}/matches.parquet'
    ORDER BY hltv_match_id ASC
""").df()

df['result'] = np.where(df['team1_score'] > df['team2_score'], 0, 1)
df['team1_elo'] = None
df['team2_elo'] = None
df['team1_games'] = None
df['team2_games'] = None

# Init ELO to 1500 in separate dictionary
unique_teams = pd.unique(df[['team1_name', 'team2_name']].values.ravel())
elo = {
    team: {
        'elo': 1500,
        'games_played': 0
    }
    for team in unique_teams
}

def expected_score(elo_1, elo_2):
    return 1 / (1 + 10 ** ((elo_2 - elo_1) / 400))

def update_elo(elo_1, elo_2, res, k1, k2):
    ea = expected_score(elo_1, elo_2)
    eb = 1 - ea
    elo_1_new = elo_1 + k1 * (res - ea)
    elo_2_new = elo_2 + k2 * ((1 - res) - eb)
    return elo_1_new, elo_2_new

for index, row in df.iterrows():
    team1, team2 = row['team1_name'], row['team2_name']
    score1, score2 = row['team1_score'], row['team2_score']

    # Store current ELO
    df.at[index, 'team1_elo'] = elo[team1]['elo']
    df.at[index, 'team2_elo'] = elo[team2]['elo']
    df.at[index, 'team1_games'] = elo[team1]['games_played']
    df.at[index, 'team2_games'] = elo[team2]['games_played']

    res = df.at[index, 'result']

    # Set k
    k1 = HIGH_K if elo[team1]['games_played'] < THRESHOLD else LOW_K
    k2 = HIGH_K if elo[team2]['games_played'] < THRESHOLD else LOW_K

    elo[team1]['elo'], elo[team2]['elo'] = update_elo(elo[team1]['elo'], elo[team2]['elo'], res, k1, k2)

    # Increment games played
    elo[team1]['games_played'] += 1
    elo[team2]['games_played'] += 1

# Filter to consider only >X matches played
filtered_df = df[(df['team1_games'] > 30) & (df['team2_games'] > 30)]

# Split remaining data chronologically
train_df, test_df = train_test_split(
    filtered_df,
    test_size=0.2,
    shuffle=False
)

num_rows = len(train_df)
print(f"Number of training matches: {num_rows}")

num_rows = len(test_df)
print(f"Number of testing matches: {num_rows}")

train_df = train_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)

# print(train_df.sample(50))

# Features
feature_cols = ['team1_elo', 'team2_elo']
train_inputs = train_df[feature_cols]
test_inputs = test_df[feature_cols]

# Target
train_targets = train_df['result']
test_targets = test_df['result']

model = LogisticRegression(solver='liblinear')
model.fit(train_inputs, train_targets)

def predict_and_plot(inputs, targets, name=''):
    probs = model.predict_proba(inputs)[:, 1]

    ll = log_loss(targets, probs)
    auc = roc_auc_score(targets, probs)
    brier = brier_score_loss(targets, probs)

    print(f"{name} Log Loss:    {ll:.3f}  (baseline: 0.693)")
    print(f"{name} ROC-AUC:     {auc:.3f}  (baseline: 0.500)")
    print(f"{name} Brier Score: {brier:.3f} (baseline: 0.250)")

predict_and_plot(train_inputs, train_targets, name='Train')
predict_and_plot(test_inputs, test_targets, name='Test')

if SAVE_RESULTS:
    train_probs = model.predict_proba(train_inputs)[:, 1]
    test_probs  = model.predict_proba(test_inputs)[:, 1]

    # Build this run as a dict of metric -> value
    run_data = {
        'timestamp':     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model':          RUN_NOTE,
    }
    # Add features one by one
    for i, feat in enumerate(feature_cols, 1):
        run_data[f'feature_{i}'] = feat

    # Add metrics
    run_data.update({
        'train_matches':  len(train_df),
        'test_matches':   len(test_df),
        'train_log_loss': round(log_loss(train_targets, train_probs), 4),
        'train_roc_auc':  round(roc_auc_score(train_targets, train_probs), 4),
        'train_brier':    round(brier_score_loss(train_targets, train_probs), 4),
        'test_log_loss':  round(log_loss(test_targets, test_probs), 4),
        'test_roc_auc':   round(roc_auc_score(test_targets, test_probs), 4),
        'test_brier':     round(brier_score_loss(test_targets, test_probs), 4),
    })

    # Convert to a single column Series (metric as index, value as data)
    new_col = pd.Series(run_data, name=datetime.now().strftime("%Y%m%d_%H%M%S"))

    if os.path.exists(RESULTS_FILE):
        existing = pd.read_csv(RESULTS_FILE, index_col=0)
        updated = pd.concat([existing, new_col], axis=1)
    else:
        updated = new_col.to_frame()

    # Find all feature rows across all runs
    all_features = [idx for idx in updated.index if idx.startswith('feature_')]
    # Sort them numerically
    all_features = sorted(all_features, key=lambda x: int(x.split('_')[1]))

    # Define fixed order: metadata first, then features, then metrics
    fixed_order = [
        'timestamp',
        'model',
        *all_features,
        'train_matches',
        'test_matches',
        'train_log_loss',
        'train_roc_auc',
        'train_brier',
        'test_log_loss',
        'test_roc_auc',
        'test_brier',
    ]

    # Reindex to enforce order, keeping any rows not in fixed_order at the end
    existing_rows = [r for r in fixed_order if r in updated.index]
    updated = updated.reindex(existing_rows)

    updated.to_csv(RESULTS_FILE)
