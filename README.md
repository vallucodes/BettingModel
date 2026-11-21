# odds_calculator
Betting odds model

Feature-engineering

Layer 1: Foundational team strength indicators
        → Elo differences, rank gaps, H2H records

Layer 2: Team-level performance metrics
        → Rolling rating, kast, swing, adr

Layer 3: Strategic map adaptations
        → Per-map Elo variances, veto preferences, side win-%, map win streaks

Layer 4: Recent team momentum signals
        → Multi-window win streaks, adjusted form diffs, current_win_streak_diff, adjusted_win_rate_lXX (weighted by opponent Elo)

Layer 5: Environmental factors
        → Event scale, online/LAN setup, series length, fatigue levels

Layer 6: Individual contributor insights
        → Player ADR averages, role impacts, clutch efficiencies, entryfragging

## Joint map-strength experiment

From the project root, first run
`python model/feature_generation/features_map_strength.py` to generate the
walk-forward `features_map_strength_joint_v3.parquet` from `matches.parquet`,
`maps.parquet`, `vetos.parquet`, and `players.parquet`. Then run
`python model/xgboost_4.py` to
train and evaluate the XGBoost model using that parquet alongside its existing
rolling and roster-TrueSkill features. Run the generator with `--check` to
verify its roster-identity rules, likelihood gradient, and series-probability
calculations.

The map-strength fit uses player nicknames only to bridge team-name aliases
and measure roster overlap; it does not add individual player map ratings.
An identical roster under a new name inherits the same map history. A 4/5
overlap links a renamed team only when it is the unique best recent roster
match; same-name lineup changes carry 80% strength for 4/5 and 60% for 3/5.
Older match and veto evidence is discounted cumulatively across roster changes.
Roster matching uses player nicknames from the available data, not stable player
IDs, so nickname changes and substitute lists can make overlap estimates noisy.

To compare the v1/v3 map features and drop each map feature individually on a
paired chronological split, run
`python model/evaluate_map_feature_ablation.py`. It fixes the TrueSkill
candidate and boosting rounds using v1 validation, then reuses that setup and a
fixed seed for every variant. It writes a metric summary and paired test
predictions to `model/map_feature_ablation/`.


# TODO

Check Map strength feature for xgboost model
Standin flag added to create_parquet.py
