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

To visualize team-level map strength, run
`python model/feature_generation/plot_map_strength_teams.py`. This refits the
latest walk-forward joint-model cutoff and writes a top-ten team/map heatmap
and CSV under `vallu_scraper/data/features/`. The chart shows strength by map,
map offsets relative to each team's overall strength, and offset uncertainty.
The six XGBoost map features are matchup-level aggregates, so they are not
individual team ratings and cannot be plotted against one team in isolation.

## Player-based map-strength experiment

`python model/feature_generation/features_player_map_strength.py` generates
`features_map_strength_player_joint_v1.parquet`. It estimates each player's
overall and map-specific strengths from earlier map results and vetoes, then
uses the mean of the players listed for each side as the side strength. Team
names do not identify or carry ratings. A stand-in only contributes while
listed, weighted by the roster mean. The generator assumes every listed player
contributed equally because `players.parquet` has no starter/substitute flag;
player nicknames are the available identity and are not stable player IDs.

The output uses the same six match-level map features as the team-based model.
Evaluate it without replacing the existing model with
`python model/xgboost_5.py`; that runner loads the roster-based player feature
parquet through the `xgboost_4` training workflow. Run the generator's
`--check` option for its likelihood-gradient and roster-aggregation checks.


# TODO

Check Map strength feature for xgboost model
Standin flag added to create_parquet.py
