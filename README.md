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