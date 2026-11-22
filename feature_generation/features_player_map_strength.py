"""Generate walk-forward map-strength features from player ratings and rosters.

Each player's map strength is inferred from played-map results and veto
choices. A match-side strength is the mean of the listed players' strengths,
so team renames have no effect and a stand-in contributes only as part of the
lineup in which they appear.
"""

import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import duckdb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import csr_matrix

try:
    from . import features_map_strength as team_model
except ImportError:
    import features_map_strength as team_model


FEATURES_ROOT = team_model.FEATURES_ROOT
PARQUET_DIR = team_model.PARQUET_DIR
FEATURE_PATH = os.path.join(
    FEATURES_ROOT, "features_map_strength_player_joint_v1.parquet"
)
REPORT_PATH = os.path.splitext(FEATURE_PATH)[0] + "_report.txt"


@dataclass
class PlayerMapData:
    events: team_model.Events
    roster1: List[tuple[str, ...]]
    roster2: List[tuple[str, ...]]


@dataclass
class PlayerFit:
    players: np.ndarray
    player_index: Dict[str, int]
    a: np.ndarray
    b_raw: np.ndarray
    s: np.ndarray
    b: np.ndarray
    h: np.ndarray
    tau: float
    var_b: np.ndarray
    pi_ban: np.ndarray
    pi_pick: np.ndarray
    sig_pi: float
    al_early: float
    al_late: float
    al_pick: float
    p_team1_first: float
    n_results: int
    n_veto_steps: int
    n_iter: int
    seconds: float


def load_player_map_data() -> PlayerMapData:
    events = team_model.load_events()
    con = duckdb.connect()
    player_rows = con.execute(
        f"""
        SELECT hltv_match_id, team, player_name
        FROM '{PARQUET_DIR}/players.parquet'
        """
    ).df()
    con.close()

    match_pos = {int(match_id): i for i, match_id in enumerate(events.match_id)}
    grouped: Dict[tuple[int, str], set[str]] = {}
    for row in player_rows.itertuples(index=False):
        if pd.isna(row.player_name):
            continue
        player = str(row.player_name).strip().casefold()
        side = str(row.team).strip().casefold()
        position = match_pos.get(int(row.hltv_match_id))
        if player and player != "nan" and position is not None:
            if side in {"team1", "team2"}:
                grouped.setdefault((position, side), set()).add(player)

    roster1 = [
        tuple(sorted(grouped.get((i, "team1"), ())))
        for i in range(len(events.match_id))
    ]
    roster2 = [
        tuple(sorted(grouped.get((i, "team2"), ())))
        for i in range(len(events.match_id))
    ]
    return PlayerMapData(events=events, roster1=roster1, roster2=roster2)


def _roster_matrix(
    rosters: Sequence[tuple[str, ...]],
    player_index: Dict[str, int],
    n_players: int,
) -> tuple[csr_matrix, np.ndarray]:
    """Mean-weight roster matrix plus prior variance for unseen players."""
    row_indices: List[int] = []
    col_indices: List[int] = []
    values: List[float] = []
    unseen_weight_sq = np.zeros(len(rosters), dtype=np.float64)
    for row_i, roster in enumerate(rosters):
        if not roster:
            continue
        player_weight = 1.0 / len(roster)
        for player in roster:
            col_i = player_index.get(player)
            if col_i is None:
                unseen_weight_sq[row_i] += player_weight * player_weight
            else:
                row_indices.append(row_i)
                col_indices.append(col_i)
                values.append(player_weight)
    matrix = csr_matrix(
        (values, (row_indices, col_indices)),
        shape=(len(rosters), n_players),
        dtype=np.float64,
    )
    return matrix, unseen_weight_sq


def _unpack_player_theta(
    theta: np.ndarray, n_players: int, n_maps: int
) -> Dict[str, object]:
    theta = np.asarray(theta, dtype=np.float64)
    i = 0
    a = theta[i : i + n_players]
    i += n_players
    b_raw = theta[i : i + n_players * n_maps].reshape(n_players, n_maps)
    i += n_players * n_maps
    h = theta[i : i + n_maps]
    i += n_maps
    tau_raw = float(theta[i])
    tau = float(team_model.softplus(tau_raw))
    d_tau = team_model.expit_scalar(tau_raw)
    i += 1
    alpha_raw = theta[i : i + 3]
    alpha = np.asarray(team_model.softplus(alpha_raw), dtype=np.float64)
    alpha_derivative = np.asarray(
        [team_model.expit_scalar(value) for value in alpha_raw],
        dtype=np.float64,
    )
    i += 3
    sig_raw = float(theta[i])
    sig = float(team_model.softplus(sig_raw))
    d_sig = team_model.expit_scalar(sig_raw)
    i += 1
    pi_ban_raw = theta[i : i + n_players * n_maps].reshape(n_players, n_maps)
    i += n_players * n_maps
    pi_pick_raw = theta[i : i + n_players * n_maps].reshape(n_players, n_maps)
    i += n_players * n_maps
    if i != theta.size:
        raise ValueError(f"theta size {theta.size} does not match layout {i}")
    return {
        "a": a,
        "b_raw": b_raw,
        "h": h,
        "tau": tau,
        "d_tau": d_tau,
        "alpha": alpha,
        "alpha_derivative": alpha_derivative,
        "sig": sig,
        "d_sig": d_sig,
        "pi_ban_raw": pi_ban_raw,
        "pi_pick_raw": pi_pick_raw,
    }


def _player_nll_and_grad(
    theta: np.ndarray,
    n_players: int,
    n_maps: int,
    result_roster1: csr_matrix,
    result_roster2: csr_matrix,
    result_map: np.ndarray,
    result_y: np.ndarray,
    result_weight: np.ndarray,
    actor_roster: csr_matrix,
    opponent_roster: csr_matrix,
    veto_chosen: np.ndarray,
    veto_code: np.ndarray,
    veto_available: np.ndarray,
    veto_weight: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Player-level joint likelihood and its analytic gradient."""
    p = _unpack_player_theta(theta, n_players, n_maps)
    a = p["a"]
    b_raw = p["b_raw"]
    tau = p["tau"]
    s = a[:, None] + tau * b_raw
    nll = 0.0
    g_s = np.zeros((n_players, n_maps), dtype=np.float64)
    g_h = np.zeros(n_maps, dtype=np.float64)
    g_tau = 0.0
    g_alpha = np.zeros(3, dtype=np.float64)
    g_sig = 0.0
    g_pi_ban = np.zeros((n_players, n_maps), dtype=np.float64)
    g_pi_pick = np.zeros((n_players, n_maps), dtype=np.float64)

    if result_y.size and np.any(result_weight > 0):
        result_s1 = result_roster1 @ s
        result_s2 = result_roster2 @ s
        eta = result_s1[np.arange(result_y.size), result_map]
        eta -= result_s2[np.arange(result_y.size), result_map]
        eta += p["h"][result_map]
        nll += float(
            np.sum(
                result_weight
                * (team_model.softplus(eta) - result_y * eta)
            )
        )
        probability = 1.0 / (
            1.0 + np.exp(-np.clip(eta, -60.0, 60.0))
        )
        g_eta = result_weight * (probability - result_y)
        result_gradient = np.zeros((result_y.size, n_maps), dtype=np.float64)
        result_gradient[np.arange(result_y.size), result_map] = g_eta
        g_s += np.asarray(result_roster1.T @ result_gradient)
        g_s -= np.asarray(result_roster2.T @ result_gradient)
        np.add.at(g_h, result_map, g_eta)

    if veto_chosen.size and np.any(veto_weight > 0):
        actor_s = actor_roster @ s
        opponent_s = opponent_roster @ s
        pi_ban_raw = p["pi_ban_raw"]
        pi_pick_raw = p["pi_pick_raw"]
        actor_pi_ban_raw = actor_roster @ pi_ban_raw
        actor_pi_pick_raw = actor_roster @ pi_pick_raw
        actor_pi_ban = p["sig"] * actor_pi_ban_raw
        actor_pi_pick = p["sig"] * actor_pi_pick_raw
        diff_ban = opponent_s - actor_s
        diff_pick = -diff_ban

        early = veto_code == 0
        late = veto_code == 1
        pick = veto_code == 2
        utilities = np.zeros((veto_chosen.size, n_maps), dtype=np.float64)
        if np.any(early):
            utilities[early] = (
                p["alpha"][0] * diff_ban[early] + actor_pi_ban[early]
            )
        if np.any(late):
            utilities[late] = (
                p["alpha"][1] * diff_ban[late] + actor_pi_ban[late]
            )
        if np.any(pick):
            utilities[pick] = (
                p["alpha"][2] * diff_pick[pick] + actor_pi_pick[pick]
            )
        utilities = np.where(veto_available, utilities, -np.inf)
        probabilities, log_z = team_model._softmax_rows(utilities)
        chosen_u = utilities[np.arange(veto_chosen.size), veto_chosen]
        nll += float(-np.sum(veto_weight * (chosen_u - log_z)))
        g_u = probabilities * veto_weight[:, None]
        g_u[np.arange(veto_chosen.size), veto_chosen] -= veto_weight

        actor_gradient = np.zeros_like(g_u)
        opponent_gradient = np.zeros_like(g_u)
        ban_preference_gradient = np.zeros_like(g_u)
        pick_preference_gradient = np.zeros_like(g_u)
        if np.any(early):
            g_alpha[0] += float(np.sum(g_u[early] * diff_ban[early]))
            actor_gradient[early] -= p["alpha"][0] * g_u[early]
            opponent_gradient[early] += p["alpha"][0] * g_u[early]
            ban_preference_gradient[early] += g_u[early]
        if np.any(late):
            g_alpha[1] += float(np.sum(g_u[late] * diff_ban[late]))
            actor_gradient[late] -= p["alpha"][1] * g_u[late]
            opponent_gradient[late] += p["alpha"][1] * g_u[late]
            ban_preference_gradient[late] += g_u[late]
        if np.any(pick):
            g_alpha[2] += float(np.sum(g_u[pick] * diff_pick[pick]))
            actor_gradient[pick] += p["alpha"][2] * g_u[pick]
            opponent_gradient[pick] -= p["alpha"][2] * g_u[pick]
            pick_preference_gradient[pick] += g_u[pick]

        g_s += np.asarray(actor_roster.T @ actor_gradient)
        g_s += np.asarray(opponent_roster.T @ opponent_gradient)
        g_pi_ban += np.asarray(
            actor_roster.T @ (p["sig"] * ban_preference_gradient)
        )
        g_pi_pick += np.asarray(
            actor_roster.T @ (p["sig"] * pick_preference_gradient)
        )
        g_sig += float(
            np.sum(ban_preference_gradient * actor_pi_ban_raw)
            + np.sum(pick_preference_gradient * actor_pi_pick_raw)
        )

    g_a = g_s.sum(axis=1)
    g_a += a / (team_model.A_PRIOR_SD**2)
    g_b_raw = tau * g_s
    g_tau += float(np.sum(g_s * b_raw))
    nll += float(np.sum(a * a) / (2.0 * team_model.A_PRIOR_SD**2))
    nll += float(
        np.sum(b_raw * b_raw) / (2.0 * team_model.B_RAW_PRIOR_SD**2)
    )
    g_b_raw += b_raw / (team_model.B_RAW_PRIOR_SD**2)
    nll += float(np.sum(p["h"] ** 2) / (2.0 * team_model.H_PRIOR_SD**2))
    g_h += p["h"] / (team_model.H_PRIOR_SD**2)
    nll += tau * tau / (2.0 * team_model.TAU_PRIOR_SD**2)
    g_tau += tau / (team_model.TAU_PRIOR_SD**2)

    nll += float(np.sum(p["alpha"] ** 2) / (2.0 * team_model.ALPHA_PRIOR_SD**2))
    g_alpha += p["alpha"] / (team_model.ALPHA_PRIOR_SD**2)
    nll += p["sig"] ** 2 / (2.0 * team_model.SIG_PI_PRIOR_SD**2)
    g_sig += p["sig"] / (team_model.SIG_PI_PRIOR_SD**2)
    nll += float(
        np.sum(p["pi_ban_raw"] ** 2)
        / (2.0 * team_model.PI_RAW_PRIOR_SD**2)
    )
    nll += float(
        np.sum(p["pi_pick_raw"] ** 2)
        / (2.0 * team_model.PI_RAW_PRIOR_SD**2)
    )
    g_pi_ban += p["pi_ban_raw"] / (team_model.PI_RAW_PRIOR_SD**2)
    g_pi_pick += p["pi_pick_raw"] / (team_model.PI_RAW_PRIOR_SD**2)

    gradient = np.concatenate(
        [
            g_a,
            g_b_raw.ravel(),
            g_h,
            np.array([g_tau * p["d_tau"]]),
            g_alpha * p["alpha_derivative"],
            np.array([g_sig * p["d_sig"]]),
            g_pi_ban.ravel(),
            g_pi_pick.ravel(),
        ]
    )
    return float(nll), gradient


def _roster_weight_matrix(
    rosters: Sequence[tuple[str, ...]],
    player_index: Dict[str, int],
    n_players: int,
) -> tuple[csr_matrix, np.ndarray]:
    return _roster_matrix(rosters, player_index, n_players)


def fit_player_latent(
    data: PlayerMapData,
    cutoff: np.datetime64,
    warm: Optional[PlayerFit] = None,
    maxiter: Optional[int] = None,
) -> PlayerFit:
    cutoff = np.datetime64(cutoff, "ns")
    events = data.events
    n_maps = len(events.maps)
    before = np.flatnonzero(events.started_at < cutoff)
    player_names = sorted(
        {
            player
            for position in before
            for roster in (data.roster1[position], data.roster2[position])
            for player in roster
        }
    )
    if not player_names:
        raise RuntimeError(f"No player rosters before {cutoff}")
    players = np.asarray(player_names, dtype=object)
    player_index = {name: i for i, name in enumerate(player_names)}
    n_players = len(players)

    res_time_all = events.started_at[events.res_match]
    res_keep = np.flatnonzero(res_time_all < cutoff)
    res_pos_all = events.res_match[res_keep]
    valid_results = np.fromiter(
        (
            bool(data.roster1[pos]) and bool(data.roster2[pos])
            for pos in res_pos_all
        ),
        dtype=bool,
        count=len(res_pos_all),
    )
    res_keep = res_keep[valid_results]
    res_pos = events.res_match[res_keep]
    res_map = events.res_map[res_keep].astype(np.int32, copy=False)
    result_y = events.res_y[res_keep]
    result_weight = team_model.decay_weights(res_time_all[res_keep], cutoff)
    result_roster1, _ = _roster_matrix(
        [data.roster1[pos] for pos in res_pos], player_index, n_players
    )
    result_roster2, _ = _roster_matrix(
        [data.roster2[pos] for pos in res_pos], player_index, n_players
    )

    veto_time_all = (
        events.started_at[events.veto_match]
        if events.veto_match.size
        else np.array([], dtype="datetime64[ns]")
    )
    veto_keep = (
        np.flatnonzero(veto_time_all < cutoff)
        if veto_time_all.size
        else np.array([], dtype=np.int32)
    )
    veto_pos_all = events.veto_match[veto_keep]
    actor_is_team1_all = events.veto_actor_is_team1[veto_keep]
    actor_rosters_all = [
        data.roster1[pos] if actor_is_team1 else data.roster2[pos]
        for pos, actor_is_team1 in zip(veto_pos_all, actor_is_team1_all)
    ]
    opponent_rosters_all = [
        data.roster2[pos] if actor_is_team1 else data.roster1[pos]
        for pos, actor_is_team1 in zip(veto_pos_all, actor_is_team1_all)
    ]
    valid_vetoes = np.fromiter(
        (
            bool(actor_roster) and bool(opponent_roster)
            for actor_roster, opponent_roster in zip(
                actor_rosters_all, opponent_rosters_all
            )
        ),
        dtype=bool,
        count=len(veto_pos_all),
    )
    veto_keep = veto_keep[valid_vetoes]
    veto_pos = events.veto_match[veto_keep]
    actor_is_team1 = events.veto_actor_is_team1[veto_keep]
    actor_rosters = [
        data.roster1[pos] if actor_is_team1_row else data.roster2[pos]
        for pos, actor_is_team1_row in zip(veto_pos, actor_is_team1)
    ]
    opponent_rosters = [
        data.roster2[pos] if actor_is_team1_row else data.roster1[pos]
        for pos, actor_is_team1_row in zip(veto_pos, actor_is_team1)
    ]
    actor_roster, _ = _roster_matrix(actor_rosters, player_index, n_players)
    opponent_roster, _ = _roster_matrix(
        opponent_rosters, player_index, n_players
    )
    veto_chosen = events.veto_map[veto_keep].astype(np.int32, copy=False)
    veto_code = events.veto_code[veto_keep]
    veto_available = events.veto_avail[veto_keep]
    veto_weight = team_model.decay_weights(veto_time_all[veto_keep], cutoff)

    a0 = np.zeros(n_players, dtype=np.float64)
    b_raw0 = np.zeros((n_players, n_maps), dtype=np.float64)
    pi_ban_raw0 = np.zeros((n_players, n_maps), dtype=np.float64)
    pi_pick_raw0 = np.zeros((n_players, n_maps), dtype=np.float64)
    if warm is not None:
        for name, new_i in player_index.items():
            old_i = warm.player_index.get(name)
            if old_i is not None:
                a0[new_i] = warm.a[old_i]
                b_raw0[new_i] = warm.b_raw[old_i]
                pi_ban_raw0[new_i] = warm.pi_ban[old_i] / max(
                    warm.sig_pi if hasattr(warm, "sig_pi") else 1.0, 1e-8
                )
                pi_pick_raw0[new_i] = warm.pi_pick[old_i] / max(
                    warm.sig_pi if hasattr(warm, "sig_pi") else 1.0, 1e-8
                )

    theta0 = np.concatenate(
        [
            a0,
            b_raw0.ravel(),
            np.zeros(n_maps, dtype=np.float64),
            np.array(
                [
                    team_model.inv_softplus(0.35),
                    team_model.inv_softplus(0.5),
                    team_model.inv_softplus(0.5),
                    team_model.inv_softplus(0.5),
                    team_model.inv_softplus(0.10),
                ],
                dtype=np.float64,
            ),
            pi_ban_raw0.ravel(),
            pi_pick_raw0.ravel(),
        ]
    )
    if warm is not None:
        theta0[n_players + n_players * n_maps : n_players + n_players * n_maps + n_maps] = warm.h
        scalar_start = n_players + n_players * n_maps + n_maps + 1
        theta0[scalar_start : scalar_start + 3] = [
            team_model.inv_softplus(warm.al_early),
            team_model.inv_softplus(warm.al_late),
            team_model.inv_softplus(warm.al_pick),
        ]
        theta0[scalar_start + 3] = team_model.inv_softplus(
            max(warm.sig_pi, 1e-8)
        )

    if maxiter is None:
        maxiter = (
            team_model.WARM_MAXITER
            if warm is not None
            else team_model.COLD_MAXITER
        )

    def objective(theta):
        return _player_nll_and_grad(
            theta,
            n_players,
            n_maps,
            result_roster1,
            result_roster2,
            res_map,
            result_y,
            result_weight,
            actor_roster,
            opponent_roster,
            veto_chosen,
            veto_code,
            veto_available,
            veto_weight,
        )

    started = time.time()
    result = minimize(
        objective,
        theta0,
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": int(maxiter),
            "maxfun": int(maxiter) * 12,
            "ftol": 1e-8,
            "gtol": 1e-4,
        },
    )
    seconds = time.time() - started
    fitted = _unpack_player_theta(result.x, n_players, n_maps)
    a = np.array(fitted["a"], copy=True)
    b_raw = np.array(fitted["b_raw"], copy=True)
    s = a[:, None] + fitted["tau"] * b_raw
    b = fitted["tau"] * b_raw

    info = np.zeros((n_players, n_maps), dtype=np.float64)
    if result_y.size and np.any(result_weight > 0):
        eta = (result_roster1 @ s)[np.arange(result_y.size), res_map]
        eta -= (result_roster2 @ s)[np.arange(result_y.size), res_map]
        eta += fitted["h"][res_map]
        prob = 1.0 / (1.0 + np.exp(-np.clip(eta, -60.0, 60.0)))
        curvature = np.zeros((result_y.size, n_maps), dtype=np.float64)
        curvature[np.arange(result_y.size), res_map] = (
            result_weight * prob * (1.0 - prob)
        )
        info += np.asarray(result_roster1.power(2).T @ curvature)
        info += np.asarray(result_roster2.power(2).T @ curvature)

    if veto_chosen.size:
        pi_ban = fitted["sig"] * fitted["pi_ban_raw"]
        pi_pick = fitted["sig"] * fitted["pi_pick_raw"]
        actor_s = actor_roster @ s
        opponent_s = opponent_roster @ s
        actor_pi_ban = actor_roster @ pi_ban
        actor_pi_pick = actor_roster @ pi_pick
        diff_ban = opponent_s - actor_s
        utilities = np.zeros((veto_chosen.size, n_maps), dtype=np.float64)
        early, late, pick = veto_code == 0, veto_code == 1, veto_code == 2
        if np.any(early):
            utilities[early] = (
                fitted["alpha"][0] * diff_ban[early] + actor_pi_ban[early]
            )
        if np.any(late):
            utilities[late] = (
                fitted["alpha"][1] * diff_ban[late] + actor_pi_ban[late]
            )
        if np.any(pick):
            utilities[pick] = (
                fitted["alpha"][2] * -diff_ban[pick] + actor_pi_pick[pick]
            )
        probabilities, _ = team_model._softmax_rows(
            np.where(veto_available, utilities, -np.inf)
        )
        alpha_by_step = np.where(
            early,
            fitted["alpha"][0],
            np.where(late, fitted["alpha"][1], fitted["alpha"][2]),
        )
        curvature = (
            veto_weight[:, None]
            * alpha_by_step[:, None] ** 2
            * probabilities
            * (1.0 - probabilities)
        )
        info += np.asarray(actor_roster.power(2).T @ curvature)
        info += np.asarray(opponent_roster.power(2).T @ curvature)
    var_b = fitted["tau"] ** 2 / (
        1.0 + fitted["tau"] ** 2 * info
    )

    if veto_keep.size:
        first_step = events.veto_step[veto_keep] == 1
        if np.any(first_step) and veto_weight[first_step].sum() > 0:
            p_first = float(
                np.average(actor_is_team1[first_step], weights=veto_weight[first_step])
            )
        else:
            p_first = 0.4
    else:
        p_first = 0.4
    p_first = float(np.clip(p_first, 0.15, 0.85))

    return PlayerFit(
        players=players,
        player_index=player_index,
        a=a,
        b_raw=b_raw,
        s=s,
        b=b,
        h=np.array(fitted["h"], copy=True),
        tau=float(fitted["tau"]),
        var_b=var_b,
        pi_ban=fitted["sig"] * np.array(fitted["pi_ban_raw"], copy=True),
        pi_pick=fitted["sig"] * np.array(fitted["pi_pick_raw"], copy=True),
        sig_pi=float(fitted["sig"]),
        al_early=float(fitted["alpha"][0]),
        al_late=float(fitted["alpha"][1]),
        al_pick=float(fitted["alpha"][2]),
        p_team1_first=p_first,
        n_results=int(result_y.size),
        n_veto_steps=int(veto_chosen.size),
        n_iter=int(result.nit),
        seconds=seconds,
    )


def _aggregate_roster(
    roster: Sequence[tuple[str, ...]],
    fit: PlayerFit,
    matrix_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    matrix, unseen_weight_sq = _roster_matrix(
        roster, fit.player_index, len(fit.players)
    )
    return np.asarray(matrix @ matrix_values), unseen_weight_sq


def player_features_for_matches(
    fit: PlayerFit,
    data: PlayerMapData,
    match_idx: np.ndarray,
    pool: List[int],
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    events = data.events
    n = len(match_idx)
    empty = {name: np.full(n, np.nan) for name in team_model.MAP_FEATURES}
    if n == 0 or not pool:
        return empty
    roster1 = [data.roster1[i] for i in match_idx]
    roster2 = [data.roster2[i] for i in match_idx]
    s1, unknown1 = _aggregate_roster(roster1, fit, fit.s)
    s2, unknown2 = _aggregate_roster(roster2, fit, fit.s)
    b1, _ = _aggregate_roster(roster1, fit, fit.b)
    b2, _ = _aggregate_roster(roster2, fit, fit.b)
    pi_ban1, _ = _aggregate_roster(roster1, fit, fit.pi_ban)
    pi_ban2, _ = _aggregate_roster(roster2, fit, fit.pi_ban)
    pi_pick1, _ = _aggregate_roster(roster1, fit, fit.pi_pick)
    pi_pick2, _ = _aggregate_roster(roster2, fit, fit.pi_pick)

    # Variance of a weighted roster mean uses squared player weights.
    matrix1, unknown1 = _roster_matrix(roster1, fit.player_index, len(fit.players))
    matrix2, unknown2 = _roster_matrix(roster2, fit.player_index, len(fit.players))
    variance1 = np.asarray(matrix1.power(2) @ fit.var_b)
    variance2 = np.asarray(matrix2.power(2) @ fit.var_b)
    variance1 += unknown1[:, None] * fit.tau**2
    variance2 += unknown2[:, None] * fit.tau**2

    map_indices = np.asarray(pool, dtype=np.int32)
    edge = s1[:, map_indices] - s2[:, map_indices]
    offset = b1[:, map_indices] - b2[:, map_indices]
    standard_error = np.sqrt(
        np.maximum(variance1[:, map_indices] + variance2[:, map_indices], 0.0)
    )
    p_team2 = np.full(n, np.nan)
    for bo_value in (1, 3, 5):
        selected = events.bo[match_idx] == bo_value
        if np.any(selected):
            p_team2[selected] = team_model.simulate_series_p_team2(
                s1[selected],
                s2[selected],
                fit.h,
                pi_ban1[selected],
                pi_ban2[selected],
                pi_pick1[selected],
                pi_pick2[selected],
                fit,
                pool,
                bo_value,
                rng,
            )
    return {
        "map_p_team2": p_team2,
        "map_edge_mean": edge.mean(axis=1),
        "map_edge_best": edge.max(axis=1),
        "map_edge_worst": edge.min(axis=1),
        "map_offset_edge_mean": offset.mean(axis=1),
        "map_offset_se_mean": standard_error.mean(axis=1),
    }


def build_feature_frame(data: PlayerMapData, log) -> pd.DataFrame:
    events = data.events
    first = events.started_at.min() + np.timedelta64(
        team_model.MIN_HISTORY_DAYS, "D"
    )
    last = events.started_at.max()
    cutoffs = []
    cursor = first
    while cursor <= last:
        cutoffs.append(np.datetime64(cursor, "ns"))
        cursor += np.timedelta64(team_model.BLOCK_DAYS, "D")
    columns = {
        name: np.full(len(events.match_id), np.nan)
        for name in team_model.MAP_FEATURES
    }
    warm: Optional[PlayerFit] = None
    log(
        f"Walk-forward player map strength: {len(cutoffs)} refits, "
        f"half-life {team_model.HALF_LIFE_DAYS:.0f} days, "
        f"block {team_model.BLOCK_DAYS} days."
    )
    for i, cutoff in enumerate(cutoffs):
        next_cutoff = (
            cutoffs[i + 1]
            if i + 1 < len(cutoffs)
            else last + np.timedelta64(1, "D")
        )
        apply_idx = np.flatnonzero(
            (events.started_at >= cutoff) & (events.started_at < next_cutoff)
        )
        if apply_idx.size == 0:
            continue
        fit = fit_player_latent(
            data,
            cutoff,
            warm=warm,
            maxiter=(
                team_model.COLD_MAXITER
                if warm is None
                else team_model.WARM_MAXITER
            ),
        )
        warm = fit
        pool = team_model.infer_pool(events, cutoff)
        seed = (
            int(cutoff.astype("datetime64[D]").astype(np.int64))
            % (2**31 - 1)
        )
        features = player_features_for_matches(
            fit, data, apply_idx, pool, np.random.default_rng(seed)
        )
        for name, values in features.items():
            columns[name][apply_idx] = values
        log(
            f"  {np.datetime64(cutoff, 'D')}: matches {apply_idx.size}, "
            f"players {len(fit.players)}, tau {fit.tau:.3f}, "
            f"alpha ban {fit.al_early:.3f}/{fit.al_late:.3f}, "
            f"pick {fit.al_pick:.3f}, vetoes {fit.n_veto_steps}, "
            f"{fit.seconds:.1f}s"
        )
    frame = pd.DataFrame({"hltv_match_id": events.match_id, **columns})
    return frame


def run_self_checks() -> None:
    rng = np.random.default_rng(42)
    n_players, n_maps = 3, 2
    player_index = {"p1": 0, "p2": 1, "p3": 2}
    result_roster1, _ = _roster_matrix(
        [("p1", "p2")], player_index, n_players
    )
    result_roster2, _ = _roster_matrix([("p3",)], player_index, n_players)
    actor_roster, _ = _roster_matrix([("p1", "p2")], player_index, n_players)
    opponent_roster, _ = _roster_matrix([("p3",)], player_index, n_players)
    theta_size = n_players * (1 + 3 * n_maps) + n_maps + 5
    theta = rng.normal(0.0, 0.1, size=theta_size)
    theta[n_players + n_players * n_maps + n_maps] = team_model.inv_softplus(0.4)
    theta[n_players + n_players * n_maps + n_maps + 1 :
          n_players + n_players * n_maps + n_maps + 4] = team_model.inv_softplus(0.5)
    theta[n_players + n_players * n_maps + n_maps + 4] = team_model.inv_softplus(0.2)
    args = (
        n_players,
        n_maps,
        result_roster1,
        result_roster2,
        np.array([0], dtype=np.int32),
        np.array([1.0]),
        np.array([1.0]),
        actor_roster,
        opponent_roster,
        np.array([1], dtype=np.int32),
        np.array([0], dtype=np.int8),
        np.array([[True, True]]),
        np.array([1.0]),
    )
    value, gradient = _player_nll_and_grad(theta, *args)
    indices = rng.choice(theta.size, size=12, replace=False)
    epsilon = 1e-6
    for index in indices:
        plus = theta.copy()
        minus = theta.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        numeric = (
            _player_nll_and_grad(plus, *args)[0]
            - _player_nll_and_grad(minus, *args)[0]
        ) / (2 * epsilon)
        if not np.isclose(gradient[index], numeric, atol=2e-5, rtol=2e-4):
            raise AssertionError(
                f"Gradient mismatch at parameter {index}: "
                f"{gradient[index]} vs {numeric}"
            )
    if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
        raise AssertionError("Player-map objective returned non-finite values.")

    roster_matrix, unseen_variance = _roster_matrix(
        [("p1", "new_player")], player_index, n_players
    )
    if not np.allclose(roster_matrix.toarray(), [[0.5, 0.0, 0.0]]):
        raise AssertionError("Unknown roster members were not assigned prior mean.")
    if not np.allclose(unseen_variance, [0.25]):
        raise AssertionError("Unknown roster-member prior variance was lost.")
    print("Player map-strength self-checks passed.", flush=True)


def generate_features() -> pd.DataFrame:
    print("Loading matches, maps, vetoes, and player rosters...", flush=True)
    data = load_player_map_data()
    events = data.events
    print(
        f"Matches {len(events.match_id)}, results {len(events.res_y)}, "
        f"veto choices {len(events.veto_map)}, "
        f"players {len({p for roster in data.roster1 + data.roster2 for p in roster})}.",
        flush=True,
    )
    lines: List[str] = []

    def log(message: str) -> None:
        print(message, flush=True)
        lines.append(message)

    frame = build_feature_frame(data, log)
    if not frame["hltv_match_id"].is_unique:
        raise ValueError("Player map-strength features contain duplicate match IDs.")
    missing = set(["hltv_match_id", *team_model.MAP_FEATURES]) - set(frame.columns)
    if missing:
        raise ValueError(
            f"Generated player map-strength features are missing: {sorted(missing)}"
        )
    os.makedirs(FEATURES_ROOT, exist_ok=True)
    frame.to_parquet(FEATURE_PATH, index=False)
    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"Saved {len(frame)} match rows to {FEATURE_PATH}", flush=True)
    return frame


if __name__ == "__main__":
    if "--check" in sys.argv:
        run_self_checks()
    else:
        generate_features()
