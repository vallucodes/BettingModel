import pandas as pd
import numpy as np
import duckdb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from datetime import datetime
import os
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

SAVE_RESULTS = False
RESULTS_FILE = "/media/vallu/Storage/Coding/Own_projects/betting_model/model/results_log.csv"
RUN_NOTE = "logistic_map_wr"
PARQUET_DIR = "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/parquet"
FEATURES_DIR = "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/features"

con = duckdb.connect()

# Load matches for results
df = con.execute(f"""
    SELECT hltv_match_id, team1_name, team2_name, team1_score, team2_score
    FROM '{PARQUET_DIR}/matches.parquet'
    ORDER BY hltv_match_id ASC
""").df()

df['result'] = np.where(df['team1_score'] > df['team2_score'], 0, 1)

# Merge map winrate features
map_features_df = pd.read_parquet(f"{FEATURES_DIR}/features_map_winrate_l10.parquet")

# Compute diffs for each map
maps = ['ancient', 'anubis', 'cache', 'dust2', 'inferno', 'mirage', 'nuke', 'overpass', 'train', 'vertigo']
for m in maps:
    map_features_df[f'{m}_wr_diff_l10'] = map_features_df[f'team1_{m}_wr_l10'] - map_features_df[f'team2_{m}_wr_l10']

df = df.merge(map_features_df[['hltv_match_id'] + [f'{m}_wr_diff_l10' for m in maps]], on='hltv_match_id', how='left')

# Features (only map wr diffs)
feature_cols = [f'{m}_wr_diff_l10' for m in maps]

# Filter: Drop rows with NaNs in any feature (insufficient map history for that diff)
filtered_df = df.dropna(subset=feature_cols)

# Print NaN info for debugging
print("NaNs in features before filter:")
print(df[feature_cols].isna().sum())
print("\nNaNs in features after filter:")
print(filtered_df[feature_cols].isna().sum())
print("\nPercentage of rows dropped due to NaNs or filters:")
print(f"{(len(df) - len(filtered_df)) / len(df) * 100:.2f}%")

print(filtered_df.tail(20))

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
test_df = test_df.reset_index(drop=True)

# Inputs and targets
train_inputs = train_df[feature_cols]
test_inputs = test_df[feature_cols]
train_targets = train_df['result']
test_targets = test_df['result']

# Train model
model = LogisticRegression(solver='liblinear')
model.fit(train_inputs, train_targets)

def predict_and_plot(inputs, targets, name=''):
    probs = model.predict_proba(inputs)[:, 1]
    ll = log_loss(targets, probs)
    auc = roc_auc_score(targets, probs)
    brier = brier_score_loss(targets, probs)
    print(f"{name} Log Loss: {ll:.3f} (baseline: 0.693)")
    print(f"{name} ROC-AUC: {auc:.3f} (baseline: 0.500)")
    print(f"{name} Brier Score: {brier:.3f} (baseline: 0.250)")

predict_and_plot(train_inputs, train_targets, name='Train')
predict_and_plot(test_inputs, test_targets, name='Test')

# Odds output
test_probs = model.predict_proba(test_inputs)[:, 1]
odds_df = test_df[['hltv_match_id', 'team1_name', 'team2_name', 'result']].copy()
odds_df['team2_win_prob'] = test_probs
odds_df['team1_win_prob'] = 1 - test_probs
odds_df['team1_odds'] = 1 / odds_df['team1_win_prob']
odds_df['team2_odds'] = 1 / odds_df['team2_win_prob']
odds_df = odds_df.round(2)
print(odds_df[['hltv_match_id','team1_name', 'team2_name',
               'team1_win_prob', 'team2_win_prob',
               'team1_odds', 'team2_odds',
               'result']].head(100))

# Calibration plot
prob_true, prob_pred = calibration_curve(test_targets, test_probs, n_bins=10)
plt.plot(prob_pred, prob_true, marker='o', label='Model')
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect calibration')
plt.xlabel('Predicted probability')
plt.ylabel('Actual win rate')
plt.legend()
plt.show()

print("Predicted prob | Actual win rate")
print("-" * 35)
for pred, true in zip(prob_pred, prob_true):
    print(f"{pred:.3f} | {true:.3f}")

if SAVE_RESULTS:
    train_probs = model.predict_proba(train_inputs)[:, 1]
    test_probs = model.predict_proba(test_inputs)[:, 1]
    run_data = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model': RUN_NOTE,
    }
    # Add features
    for i, feat in enumerate(feature_cols, 1):
        run_data[f'feature_{i}'] = feat
    # Add metrics
    run_data.update({
        'train_matches': len(train_df),
        'test_matches': len(test_df),
        'train_log_loss': round(log_loss(train_targets, train_probs), 4),
        'train_roc_auc': round(roc_auc_score(train_targets, train_probs), 4),
        'train_brier': round(brier_score_loss(train_targets, train_probs), 4),
        'test_log_loss': round(log_loss(test_targets, test_probs), 4),
        'test_roc_auc': round(roc_auc_score(test_targets, test_probs), 4),
        'test_brier': round(brier_score_loss(test_targets, test_probs), 4),
    })
    new_col = pd.Series(run_data, name=datetime.now().strftime("%Y%m%d_%H%M%S"))
    if os.path.exists(RESULTS_FILE):
        existing = pd.read_csv(RESULTS_FILE, index_col=0)
        updated = pd.concat([existing, new_col], axis=1)
    else:
        updated = new_col.to_frame()
    # Reorder (similar to original)
    all_features = [idx for idx in updated.index if idx.startswith('feature_')]
    all_features = sorted(all_features, key=lambda x: int(x.split('_')[1]))
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
    existing_rows = [r for r in fixed_order if r in updated.index]
    updated = updated.reindex(existing_rows)
    updated.to_csv(RESULTS_FILE)
