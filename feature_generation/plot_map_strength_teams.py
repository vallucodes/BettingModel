"""Plot recent joint-model map strength for the ten strongest teams."""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features_map_strength import (
    BLOCK_DAYS,
    COLD_MAXITER,
    FEATURES_ROOT,
    MIN_HISTORY_DAYS,
    WARM_MAXITER,
    fit_latent,
    infer_pool,
    load_events,
)


MIN_TEAM_MATCHES = 20
OUTPUT_PLOT = os.path.join(FEATURES_ROOT, "map_strength_top_10_teams.png")
OUTPUT_CSV = os.path.join(FEATURES_ROOT, "map_strength_top_10_teams.csv")


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def latest_team_names_and_counts(data, cutoff: np.datetime64):
    latest_names = {}
    match_counts = {}
    for i, started_at in enumerate(data.started_at):
        if started_at >= cutoff:
            continue
        for team_id, name in (
            (data.team1_id[i], data.team1[i]),
            (data.team2_id[i], data.team2[i]),
        ):
            match_counts[team_id] = match_counts.get(team_id, 0) + 1
            latest = latest_names.get(team_id)
            if latest is None or started_at >= latest[0]:
                latest_names[team_id] = (started_at, str(name))
    return latest_names, match_counts


def fit_latest_walk_forward(data):
    first = data.started_at.min() + np.timedelta64(MIN_HISTORY_DAYS, "D")
    last = data.started_at.max()
    cutoffs = []
    cursor = first
    while cursor <= last:
        cutoffs.append(np.datetime64(cursor, "ns"))
        cursor += np.timedelta64(BLOCK_DAYS, "D")

    warm = None
    latest_fit = None
    latest_cutoff = None
    for i, cutoff in enumerate(cutoffs):
        next_cutoff = (
            cutoffs[i + 1]
            if i + 1 < len(cutoffs)
            else last + np.timedelta64(1, "D")
        )
        apply_idx = np.flatnonzero(
            (data.started_at >= cutoff) & (data.started_at < next_cutoff)
        )
        if apply_idx.size == 0:
            continue
        latest_fit = fit_latent(
            data,
            "joint",
            cutoff,
            warm=warm,
            maxiter=COLD_MAXITER if warm is None else WARM_MAXITER,
        )
        warm = latest_fit
        latest_cutoff = cutoff
    if latest_fit is None or latest_cutoff is None:
        raise RuntimeError("No walk-forward model cutoff could be fitted.")
    return latest_fit, latest_cutoff


def make_heatmap(
    axis,
    values: np.ndarray,
    teams: list[str],
    maps: list[str],
    title: str,
    color_map: str,
    value_format: str,
    suffix: str = "",
    vmin=None,
    vmax=None,
    center=None,
):
    if center is not None:
        limit = float(np.nanmax(np.abs(values)))
        limit = max(limit, 0.05)
        vmin, vmax = -limit, limit
    image = axis.imshow(
        values,
        aspect="auto",
        interpolation="nearest",
        cmap=color_map,
        vmin=vmin,
        vmax=vmax,
    )
    axis.set_title(title, fontsize=11, pad=10)
    axis.set_xticks(np.arange(len(maps)), labels=maps, rotation=35, ha="right")
    axis.set_yticks(np.arange(len(teams)), labels=teams)
    axis.tick_params(axis="both", labelsize=8)
    axis.set_xticks(np.arange(-0.5, len(maps), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(teams), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.5)
    axis.tick_params(which="minor", bottom=False, left=False)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            axis.text(
                col,
                row,
                f"{format(values[row, col], value_format)}{suffix}",
                ha="center",
                va="center",
                fontsize=7,
                color="black",
            )
    return image


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit the latest joint model and plot its top teams by map."
    )
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--min-matches", type=int, default=MIN_TEAM_MATCHES)
    parser.add_argument("--output", default=OUTPUT_PLOT)
    parser.add_argument("--csv", default=OUTPUT_CSV)
    args = parser.parse_args()
    if args.top < 1 or args.min_matches < 1:
        raise ValueError("--top and --min-matches must be positive.")

    data = load_events()
    print("Refitting the latest walk-forward joint model...", flush=True)
    fit, cutoff = fit_latest_walk_forward(data)
    print(
        f"Using cutoff {cutoff}; fit ran {fit.n_iter} iterations "
        f"({fit.seconds:.1f}s), tau={fit.tau:.3f}.",
        flush=True,
    )
    pool = infer_pool(data, cutoff)
    if not pool:
        raise RuntimeError(f"No active map pool could be inferred before {cutoff}.")

    latest_names, match_counts = latest_team_names_and_counts(data, cutoff)
    ranked = []
    for team_id, index in fit.team_index.items():
        count = match_counts.get(team_id, 0)
        if count < args.min_matches:
            continue
        ranked.append(
            (
                float(fit.a[index]),
                team_id,
                index,
                count,
                latest_names.get(team_id, (None, team_id))[1],
            )
        )
    ranked.sort(reverse=True)
    selected = ranked[: args.top]
    if len(selected) < args.top:
        print(
            f"Found only {len(selected)} teams with at least "
            f"{args.min_matches} matches.",
            flush=True,
        )
    if not selected:
        raise RuntimeError("No teams met the minimum match-count threshold.")

    team_ids = [item[1] for item in selected]
    team_indices = np.array([item[2] for item in selected], dtype=np.int32)
    labels = [
        f"{rank}. {item[4]} (n={item[3]})"
        for rank, item in enumerate(selected, start=1)
    ]
    map_indices = np.asarray(pool, dtype=np.int32)
    map_names = [data.maps[index] for index in map_indices]

    strength = fit.s[np.ix_(team_indices, map_indices)]
    offset = fit.b[np.ix_(team_indices, map_indices)]
    offset_sd = np.sqrt(
        np.maximum(fit.var_b[np.ix_(team_indices, map_indices)], 0.0)
    )
    reference_win_probability = sigmoid(
        strength + fit.h[map_indices][None, :]
    )

    output = []
    for row, item in enumerate(selected):
        for col, map_name in enumerate(map_names):
            output.append(
                {
                    "rank_overall_strength": row + 1,
                    "team_id": team_ids[row],
                    "team_name": item[4],
                    "matches_in_history": item[3],
                    "overall_strength_a": item[0],
                    "map": map_name,
                    "map_strength_s": strength[row, col],
                    "map_offset_b": offset[row, col],
                    "map_offset_sd": offset_sd[row, col],
                    "win_probability_vs_zero_strength_reference": (
                        reference_win_probability[row, col]
                    ),
                    "as_of": str(cutoff),
                }
            )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
    pd.DataFrame(output).to_csv(args.csv, index=False)

    figure, axes = plt.subplots(
        1, 3, figsize=(max(15, 1.65 * len(map_names) + 9), 8),
        constrained_layout=True,
    )
    panels = [
        (
            reference_win_probability * 100.0,
            "Win probability vs zero-strength reference",
            "RdYlGn",
            ".0f",
            "%",
            0.0,
            100.0,
            None,
        ),
        (
            offset,
            "Map offset b (above/below team baseline)",
            "RdBu_r",
            "+.2f",
            "",
            None,
            None,
            0.0,
        ),
        (
            offset_sd,
            "Map-offset uncertainty (approx. SD)",
            "magma",
            ".2f",
            "",
            0.0,
            None,
            None,
        ),
    ]
    for axis, panel in zip(axes, panels):
        values, title, cmap, fmt, suffix, vmin, vmax, center = panel
        image = make_heatmap(
            axis,
            values,
            labels,
            map_names,
            title,
            cmap,
            fmt,
            suffix=suffix,
            vmin=vmin,
            vmax=vmax,
            center=center,
        )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    as_of_date = np.datetime_as_string(cutoff, unit="D")
    figure.suptitle(
        f"Top {len(selected)} teams by overall strength, as of {as_of_date}\n"
        f"Active pool: {', '.join(map_names)} | "
        f"minimum history: {args.min_matches} matches | tau={fit.tau:.2f}",
        fontsize=13,
    )
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved plot: {args.output}", flush=True)
    print(f"Saved team/map estimates: {args.csv}", flush=True)


if __name__ == "__main__":
    main()
