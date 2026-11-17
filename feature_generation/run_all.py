"""Run all feature generators sequentially, then refresh feature schemas.

Usage, from the model directory:
    python feature_generation/run_all.py

The runner uses the active Python interpreter and stops if any script fails.
"""

import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS = (
    "features_map_winrate_multiple.py",
    "features_rating_kast_swing_multiple.py",
    "features_map_elo.py",
    "features_team_elo_multiple.py",
    "features_team_trueskill_multiple.py",
    "features_map_trueskill_multiple.py",
    "generate_feature_schemas.py",
)


def main() -> None:
    for script_name in SCRIPTS:
        script_path = SCRIPT_DIR / script_name
        print(f"\n===== Running {script_name} =====", flush=True)
        subprocess.run([sys.executable, str(script_path)], check=True)


if __name__ == "__main__":
    main()
