import tempfile
import unittest
from pathlib import Path

from feature_output_cleanup import remove_stale_outputs


class RemoveStaleOutputsTests(unittest.TestCase):
    def test_removes_stale_and_nested_family_files_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            expected_name = "features_map_elo_190_90_17.parquet"
            expected_file = output_dir / expected_name
            stale_root_file = output_dir / "features_map_elo_100_20_10.parquet"
            misplaced_file = output_dir / "legacy" / expected_name
            unrelated_file = output_dir / "legacy" / "notes.txt"

            for path in (
                expected_file,
                stale_root_file,
                misplaced_file,
                unrelated_file,
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("test", encoding="utf-8")

            remove_stale_outputs(
                str(output_dir),
                "features_map_elo_",
                ".parquet",
                {expected_name},
            )

            self.assertTrue(expected_file.exists())
            self.assertFalse(stale_root_file.exists())
            self.assertFalse(misplaced_file.exists())
            self.assertTrue(unrelated_file.exists())

    def test_preserves_expected_nested_outputs_and_removes_legacy_root_copies(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            expected_dir = output_dir / "team_elo"
            expected_name = "features_team_elo_130_50_25.parquet"
            expected_file = expected_dir / expected_name
            stale_root_file = output_dir / "features_team_elo_100_20_30.parquet"
            stale_nested_file = expected_dir / "features_team_elo_10_10_10.parquet"

            for path in (expected_file, stale_root_file, stale_nested_file):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("test", encoding="utf-8")

            remove_stale_outputs(
                str(output_dir),
                "features_team_elo_",
                ".parquet",
                {expected_name},
                expected_output_dir=str(expected_dir),
            )

            self.assertTrue(expected_file.exists())
            self.assertFalse(stale_root_file.exists())
            self.assertFalse(stale_nested_file.exists())


if __name__ == "__main__":
    unittest.main()
