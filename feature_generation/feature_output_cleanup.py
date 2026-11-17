"""Remove stale generated files and misplaced family copies under an output root."""

import os
from typing import Optional


def _raise_walk_error(error: OSError) -> None:
    raise error


def remove_stale_outputs(
    output_dir: str,
    prefix: str,
    suffix: str,
    expected_filenames: set[str],
    expected_output_dir: Optional[str] = None,
) -> None:
    """Keep expected outputs in their canonical directory and remove misplaced copies."""
    target_dir = os.path.abspath(output_dir)
    canonical_dir = os.path.abspath(expected_output_dir or target_dir)
    if not os.path.exists(target_dir):
        raise FileNotFoundError(target_dir)
    if not os.path.isdir(target_dir):
        raise NotADirectoryError(target_dir)
    if os.path.commonpath((target_dir, canonical_dir)) != target_dir:
        raise ValueError("expected_output_dir must be inside output_dir.")
    if not os.path.exists(canonical_dir):
        raise FileNotFoundError(canonical_dir)
    if not os.path.isdir(canonical_dir):
        raise NotADirectoryError(canonical_dir)

    removed_count = 0

    for current_dir, _, filenames in os.walk(
        target_dir, onerror=_raise_walk_error
    ):
        is_canonical_dir = os.path.abspath(current_dir) == canonical_dir
        for filename in filenames:
            if not filename.startswith(prefix) or not filename.endswith(suffix):
                continue
            if is_canonical_dir and filename in expected_filenames:
                continue

            path = os.path.join(current_dir, filename)
            if os.path.isfile(path):
                os.remove(path)
                removed_count += 1

    if removed_count:
        print(
            f"Removed {removed_count} stale generated file(s) under {target_dir}."
        )
