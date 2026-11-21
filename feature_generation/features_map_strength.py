"""Generate walk-forward team/map-strength features from results and vetoes.

One latent number per team and map,

    s(team, map) = a(team) + tau * b_raw(team, map)

is fit by penalised maximum likelihood. Played maps use a Bradley-Terry
likelihood. Each ban or pick is a conditional logit over the maps still
available in that match's veto. The same s explains both, so a map that is
always banned still moves b. Overall strength a cancels inside a veto step.

NUTS is too slow to refit along the timeline, so L2 penalties stand in for
the Normal / HalfNormal priors in the PyMC sketch and L-BFGS fits the mode.
Features for a match come from a model fit only on earlier matches.

Team identities are connected by roster continuity, so renames preserve
strength and lineup changes discount old evidence. Each match receives
features fitted only on earlier matches. The resulting parquet is consumed
by model/xgboost_4.py.

Usage, from the project root:
    python feature_generation/features_map_strength.py
"""

import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import duckdb
import numpy as np
import pandas as pd


PARQUET_DIR = (
    "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/parquet"
)
FEATURES_ROOT = (
    "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/features"
)

FEATURE_VERSION = "joint_v3"
FEATURE_PATH = os.path.join(
    FEATURES_ROOT, f"features_map_strength_{FEATURE_VERSION}.parquet"
)
REPORT_PATH = os.path.splitext(FEATURE_PATH)[0] + "_report.txt"
HALF_LIFE_DAYS = 180.0
MIN_HISTORY_DAYS = 120
BLOCK_DAYS = 30
POOL_WINDOW_DAYS = 120
ROSTER_LINK_WINDOW_DAYS = 180
TRAIN_FRACTION = 0.70
COLD_MAXITER = 40
WARM_MAXITER = 12
N_SIMS = 64
RARE_MAP_PLAYS = 5

A_PRIOR_SD = 1.0
B_RAW_PRIOR_SD = 1.0
H_PRIOR_SD = 0.2
TAU_PRIOR_SD = 0.5
ALPHA_PRIOR_SD = 1.0
SIG_PI_PRIOR_SD = 0.3
PI_RAW_PRIOR_SD = 1.0

MAP_FEATURES = [
    "map_p_team2",
    "map_edge_mean",
    "map_edge_best",
    "map_edge_worst",
    "map_offset_edge_mean",
    "map_offset_se_mean",
]


def softplus(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def inv_softplus(y: float) -> float:
    y = float(y)
    if y <= 0:
        raise ValueError("inv_softplus expects a positive value")
    return float(y + np.log(-np.expm1(-y)))


def decay_weights(times: np.ndarray, cutoff: np.datetime64) -> np.ndarray:
    cutoff = np.datetime64(cutoff, "ns")
    days = (cutoff - times) / np.timedelta64(1, "D")
    return np.exp(-np.log(2.0) * days / HALF_LIFE_DAYS)


def parameter_count(kind: str, n_teams: int, n_maps: int) -> int:
    n = n_teams
    if kind != "strength":
        n += n_teams * n_maps + n_maps + 1
    if kind == "joint":
        n += 4 + 2 * n_teams * n_maps
    return n


def pack_theta(
    kind: str,
    a: np.ndarray,
    b_raw: np.ndarray,
    h: np.ndarray,
    tau_raw: float,
    al_early_raw: float,
    al_late_raw: float,
    al_pick_raw: float,
    sig_raw: float,
    pi_ban_raw: np.ndarray,
    pi_pick_raw: np.ndarray,
) -> np.ndarray:
    parts = [np.asarray(a, dtype=np.float64).ravel()]
    if kind != "strength":
        parts.append(np.asarray(b_raw, dtype=np.float64).ravel())
        parts.append(np.asarray(h, dtype=np.float64).ravel())
        parts.append(np.array([tau_raw], dtype=np.float64))
    if kind == "joint":
        parts.append(
            np.array(
                [al_early_raw, al_late_raw, al_pick_raw, sig_raw], dtype=np.float64
            )
        )
        parts.append(np.asarray(pi_ban_raw, dtype=np.float64).ravel())
        parts.append(np.asarray(pi_pick_raw, dtype=np.float64).ravel())
    return np.concatenate(parts)


def unpack(theta: np.ndarray, kind: str, n_teams: int, n_maps: int) -> Dict[str, Any]:
    theta = np.asarray(theta, dtype=np.float64)
    i = 0
    a = theta[i : i + n_teams]
    i += n_teams
    b_raw = np.zeros((n_teams, n_maps))
    h = np.zeros(n_maps)
    tau = 0.0
    d_tau = 0.0
    al_early = al_late = al_pick = sig = 0.0
    d_al_early = d_al_late = d_al_pick = d_sig = 0.0
    pi_ban_raw = np.zeros((n_teams, n_maps))
    pi_pick_raw = np.zeros((n_teams, n_maps))
    if kind != "strength":
        b_raw = theta[i : i + n_teams * n_maps].reshape(n_teams, n_maps)
        i += n_teams * n_maps
        h = theta[i : i + n_maps]
        i += n_maps
        tau = float(softplus(theta[i]))
        d_tau = float(expit_scalar(theta[i]))
        i += 1
    if kind == "joint":
        al_early = float(softplus(theta[i]))
        d_al_early = float(expit_scalar(theta[i]))
        i += 1
        al_late = float(softplus(theta[i]))
        d_al_late = float(expit_scalar(theta[i]))
        i += 1
        al_pick = float(softplus(theta[i]))
        d_al_pick = float(expit_scalar(theta[i]))
        i += 1
        sig = float(softplus(theta[i]))
        d_sig = float(expit_scalar(theta[i]))
        i += 1
        pi_ban_raw = theta[i : i + n_teams * n_maps].reshape(n_teams, n_maps)
        i += n_teams * n_maps
        pi_pick_raw = theta[i : i + n_teams * n_maps].reshape(n_teams, n_maps)
        i += n_teams * n_maps
    if i != theta.size:
        raise ValueError(f"theta size {theta.size} does not match layout {i}")
    return {
        "a": a,
        "b_raw": b_raw,
        "h": h,
        "tau": tau,
        "d_tau": d_tau,
        "al_early": al_early,
        "d_al_early": d_al_early,
        "al_late": al_late,
        "d_al_late": d_al_late,
        "al_pick": al_pick,
        "d_al_pick": d_al_pick,
        "sig": sig,
        "d_sig": d_sig,
        "pi_ban_raw": pi_ban_raw,
        "pi_pick_raw": pi_pick_raw,
    }


def expit_scalar(x: float) -> float:
    x = float(x)
    if x >= 0:
        z = np.exp(-x)
        return float(1.0 / (1.0 + z))
    z = np.exp(x)
    return float(z / (1.0 + z))


def ban_utility(s_actor, s_opp, pi_ban, alpha):
    """High when the opponent is stronger on that map."""
    return alpha * (s_opp - s_actor) + pi_ban


def pick_utility(s_actor, s_opp, pi_pick, alpha):
    """High when the acting team is stronger on that map."""
    return alpha * (s_actor - s_opp) + pi_pick


def _softmax_rows(utilities: np.ndarray):
    """Softmax over maps. -inf entries are unavailable."""
    finite = np.isfinite(utilities)
    row_max = np.max(np.where(finite, utilities, -1e9), axis=1, keepdims=True)
    shifted = np.exp(np.clip(utilities - row_max, -80.0, 0.0))
    shifted = np.where(finite, shifted, 0.0)
    total = shifted.sum(axis=1, keepdims=True)
    probs = shifted / np.maximum(total, 1e-300)
    log_z = row_max.ravel() + np.log(np.maximum(total.ravel(), 1e-300))
    return probs, log_z


def nll_and_grad(
    theta: np.ndarray,
    kind: str,
    n_teams: int,
    n_maps: int,
    t1: np.ndarray,
    t2: np.ndarray,
    res_map: np.ndarray,
    y: np.ndarray,
    w_res: np.ndarray,
    actor: np.ndarray,
    opp: np.ndarray,
    chosen: np.ndarray,
    veto_code: np.ndarray,
    avail: np.ndarray,
    w_veto: np.ndarray,
):
    """Negative log posterior and its gradient for one of the three models."""
    p = unpack(theta, kind, n_teams, n_maps)
    a = p["a"]
    b_raw = p["b_raw"]
    h = p["h"]
    tau = p["tau"]
    s = a[:, None] + tau * b_raw

    nll = 0.0
    g_a = np.zeros(n_teams)
    g_b = np.zeros((n_teams, n_maps))
    g_h = np.zeros(n_maps)
    g_tau = 0.0
    g_al_early = 0.0
    g_al_late = 0.0
    g_al_pick = 0.0
    g_sig = 0.0
    g_pi_ban = np.zeros((n_teams, n_maps))
    g_pi_pick = np.zeros((n_teams, n_maps))
    g_s = np.zeros((n_teams, n_maps))

    if y.size and np.any(w_res > 0):
        if kind == "strength":
            eta = a[t1] - a[t2]
        else:
            eta = s[t1, res_map] - s[t2, res_map] + h[res_map]
        nll += float(np.sum(w_res * (softplus(eta) - y * eta)))
        g_eta = w_res * (1.0 / (1.0 + np.exp(-np.clip(eta, -60.0, 60.0))) - y)
        if kind == "strength":
            np.add.at(g_a, t1, g_eta)
            np.add.at(g_a, t2, -g_eta)
        else:
            np.add.at(g_s, (t1, res_map), g_eta)
            np.add.at(g_s, (t2, res_map), -g_eta)
            np.add.at(g_h, res_map, g_eta)

    if kind == "joint" and actor.size:
        pi_ban = p["sig"] * p["pi_ban_raw"]
        pi_pick = p["sig"] * p["pi_pick_raw"]
        diff = s[opp] - s[actor]
        utilities = np.zeros((actor.size, n_maps), dtype=np.float64)
        early = veto_code == 0
        late = veto_code == 1
        pick = veto_code == 2
        if np.any(early):
            utilities[early] = ban_utility(
                s[actor[early]], s[opp[early]], pi_ban[actor[early]], p["al_early"]
            )
        if np.any(late):
            utilities[late] = ban_utility(
                s[actor[late]], s[opp[late]], pi_ban[actor[late]], p["al_late"]
            )
        if np.any(pick):
            utilities[pick] = pick_utility(
                s[actor[pick]], s[opp[pick]], pi_pick[actor[pick]], p["al_pick"]
            )
        utilities = np.where(avail, utilities, -np.inf)
        probs, log_z = _softmax_rows(utilities)
        chosen_u = utilities[np.arange(actor.size), chosen]
        nll += float(-np.sum(w_veto * (chosen_u - log_z)))
        g_u = probs * w_veto[:, None]
        g_u[np.arange(actor.size), chosen] -= w_veto

        if np.any(early):
            g_al_early += float(np.sum(g_u[early] * diff[early]))
            g_diff = g_u[early] * p["al_early"]
            np.add.at(g_s, opp[early], g_diff)
            np.add.at(g_s, actor[early], -g_diff)
            np.add.at(g_pi_ban, actor[early], g_u[early] * p["sig"])
            g_sig += float(np.sum(g_u[early] * p["pi_ban_raw"][actor[early]]))
        if np.any(late):
            g_al_late += float(np.sum(g_u[late] * diff[late]))
            g_diff = g_u[late] * p["al_late"]
            np.add.at(g_s, opp[late], g_diff)
            np.add.at(g_s, actor[late], -g_diff)
            np.add.at(g_pi_ban, actor[late], g_u[late] * p["sig"])
            g_sig += float(np.sum(g_u[late] * p["pi_ban_raw"][actor[late]]))
        if np.any(pick):
            g_al_pick += float(np.sum(g_u[pick] * (-diff[pick])))
            g_diff = g_u[pick] * (-p["al_pick"])
            np.add.at(g_s, opp[pick], g_diff)
            np.add.at(g_s, actor[pick], -g_diff)
            np.add.at(g_pi_pick, actor[pick], g_u[pick] * p["sig"])
            g_sig += float(np.sum(g_u[pick] * p["pi_pick_raw"][actor[pick]]))

    if kind == "strength":
        nll += float(np.sum(a * a) / (2.0 * A_PRIOR_SD**2))
        g_a = g_a + a / (A_PRIOR_SD**2)
    else:
        g_a = g_a + g_s.sum(axis=1)
        g_b = g_b + tau * g_s
        g_tau += float(np.sum(g_s * b_raw))
        nll += float(np.sum(a * a) / (2.0 * A_PRIOR_SD**2))
        g_a = g_a + a / (A_PRIOR_SD**2)
        nll += float(np.sum(b_raw * b_raw) / (2.0 * B_RAW_PRIOR_SD**2))
        g_b = g_b + b_raw / (B_RAW_PRIOR_SD**2)
        nll += float(np.sum(h * h) / (2.0 * H_PRIOR_SD**2))
        g_h = g_h + h / (H_PRIOR_SD**2)
        nll += (tau * tau) / (2.0 * TAU_PRIOR_SD**2)
        g_tau += tau / (TAU_PRIOR_SD**2)

    if kind == "joint":
        for value, grad_value in (
            (p["al_early"], g_al_early),
            (p["al_late"], g_al_late),
            (p["al_pick"], g_al_pick),
        ):
            nll += (value * value) / (2.0 * ALPHA_PRIOR_SD**2)
        g_al_early += p["al_early"] / (ALPHA_PRIOR_SD**2)
        g_al_late += p["al_late"] / (ALPHA_PRIOR_SD**2)
        g_al_pick += p["al_pick"] / (ALPHA_PRIOR_SD**2)
        nll += (p["sig"] * p["sig"]) / (2.0 * SIG_PI_PRIOR_SD**2)
        g_sig += p["sig"] / (SIG_PI_PRIOR_SD**2)
        nll += float(np.sum(p["pi_ban_raw"] ** 2) / (2.0 * PI_RAW_PRIOR_SD**2))
        nll += float(np.sum(p["pi_pick_raw"] ** 2) / (2.0 * PI_RAW_PRIOR_SD**2))
        g_pi_ban = g_pi_ban + p["pi_ban_raw"] / (PI_RAW_PRIOR_SD**2)
        g_pi_pick = g_pi_pick + p["pi_pick_raw"] / (PI_RAW_PRIOR_SD**2)

    chunks = [g_a]
    if kind != "strength":
        chunks.append(g_b.ravel())
        chunks.append(g_h)
        chunks.append(np.array([g_tau * p["d_tau"]]))
    if kind == "joint":
        chunks.append(
            np.array(
                [
                    g_al_early * p["d_al_early"],
                    g_al_late * p["d_al_late"],
                    g_al_pick * p["d_al_pick"],
                    g_sig * p["d_sig"],
                ]
            )
        )
        chunks.append(g_pi_ban.ravel())
        chunks.append(g_pi_pick.ravel())
    grad = np.concatenate(chunks)
    return float(nll), grad


@dataclass
class Events:
    maps: List[str]
    match_id: np.ndarray
    started_at: np.ndarray
    team1: np.ndarray
    team2: np.ndarray
    team1_id: np.ndarray
    team2_id: np.ndarray
    team1_roster_overlap: np.ndarray
    team2_roster_overlap: np.ndarray
    bo: np.ndarray
    has_veto: np.ndarray
    has_maps: np.ndarray
    match_pool: np.ndarray
    match_played: np.ndarray
    res_match: np.ndarray
    res_map: np.ndarray
    res_y: np.ndarray
    veto_match: np.ndarray
    veto_step: np.ndarray
    veto_actor_is_team1: np.ndarray
    veto_map: np.ndarray
    veto_code: np.ndarray
    veto_avail: np.ndarray


@dataclass
class LatentFit:
    kind: str
    teams: np.ndarray
    team_index: Dict[str, int]
    a: np.ndarray
    b_raw: np.ndarray
    h: np.ndarray
    tau: float
    tau_raw: float
    al_early: float
    al_late: float
    al_pick: float
    al_early_raw: float
    al_late_raw: float
    al_pick_raw: float
    sig_pi: float
    sig_raw: float
    pi_ban_raw: np.ndarray
    pi_pick_raw: np.ndarray
    s: np.ndarray
    b: np.ndarray
    var_b: np.ndarray
    pi_ban: np.ndarray
    pi_pick: np.ndarray
    p_team1_first: float
    train_map_logloss: float
    n_results: int
    n_veto_steps: int
    n_iter: int
    seconds: float


def _assign_team_identities(
    matches: pd.DataFrame,
    player_rows: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Link name aliases through roster overlap and record lineup continuity."""
    rosters: Dict[tuple, set[str]] = {}
    for row in player_rows.itertuples(index=False):
        if pd.isna(row.player_name):
            continue
        player = str(row.player_name).strip().casefold()
        team_side = str(row.team).strip().casefold()
        if player and player != "nan" and team_side in {"team1", "team2"}:
            rosters.setdefault((row.hltv_match_id, team_side), set()).add(player)

    entity_state: Dict[str, Dict[str, Any]] = {}
    name_to_entity: Dict[str, str] = {}
    roster_to_entities: Dict[frozenset[str], set[str]] = {}
    next_entity = 0
    n_matches = len(matches)
    team1_ids = np.empty(n_matches, dtype=object)
    team2_ids = np.empty(n_matches, dtype=object)
    team1_overlap = np.ones(n_matches, dtype=np.float64)
    team2_overlap = np.ones(n_matches, dtype=np.float64)

    def assign(name: str, roster: frozenset[str], started_at) -> tuple[str, float]:
        nonlocal next_entity
        name_key = name.strip().casefold()
        entity = name_to_entity.get(name_key)

        if entity is None and roster:
            exact = roster_to_entities.get(roster, set())
            if exact:
                entity = max(exact, key=lambda key: entity_state[key]["last_seen"])
            else:
                candidates = []
                for candidate, state in entity_state.items():
                    previous = state["roster"]
                    if not previous or state["last_seen"] is None:
                        continue
                    age_days = (started_at - state["last_seen"]).total_seconds() / 86400
                    if age_days > ROSTER_LINK_WINDOW_DAYS:
                        continue
                    shared = len(roster & previous)
                    denominator = max(5, min(len(roster), len(previous)))
                    similarity = shared / denominator
                    if shared >= 4 and similarity >= 0.8:
                        candidates.append((similarity, shared, candidate))
                if candidates:
                    best_score = max((similarity, shared) for similarity, shared, _ in candidates)
                    best_entities = [
                        candidate
                        for similarity, shared, candidate in candidates
                        if (similarity, shared) == best_score
                    ]
                    if len(best_entities) == 1:
                        entity = best_entities[0]

        if entity is None:
            entity = f"team_{next_entity:06d}:{name_key}"
            next_entity += 1
            entity_state[entity] = {
                "roster": frozenset(),
                "last_seen": None,
            }

        state = entity_state[entity]
        previous = state["roster"]
        if roster and previous:
            overlap = len(roster & previous) / max(5, len(roster), len(previous))
        else:
            overlap = 1.0
        if roster:
            state["roster"] = roster
            roster_to_entities.setdefault(roster, set()).add(entity)
            state["last_seen"] = started_at
        name_to_entity[name_key] = entity
        return entity, float(overlap)

    for position, row in enumerate(matches.itertuples(index=False)):
        left_roster = frozenset(rosters.get((row.hltv_match_id, "team1"), ()))
        right_roster = frozenset(rosters.get((row.hltv_match_id, "team2"), ()))
        team1_ids[position], team1_overlap[position] = assign(
            row.team1_name, left_roster, row.started_at
        )
        team2_ids[position], team2_overlap[position] = assign(
            row.team2_name, right_roster, row.started_at
        )

    return team1_ids, team2_ids, team1_overlap, team2_overlap


def _historical_roster_confidence(
    data: Events,
    cutoff: np.datetime64,
) -> tuple[np.ndarray, np.ndarray]:
    """Discount old observations by cumulative roster turnover after them."""
    confidence1 = np.ones(len(data.match_id), dtype=np.float64)
    confidence2 = np.ones(len(data.match_id), dtype=np.float64)
    before_cutoff = np.flatnonzero(data.started_at < np.datetime64(cutoff, "ns"))
    observations: Dict[str, List[tuple[int, bool, float]]] = {}
    for match_pos in before_cutoff:
        observations.setdefault(data.team1_id[match_pos], []).append(
            (int(match_pos), True, float(data.team1_roster_overlap[match_pos]))
        )
        observations.setdefault(data.team2_id[match_pos], []).append(
            (int(match_pos), False, float(data.team2_roster_overlap[match_pos]))
        )

    for team_observations in observations.values():
        team_observations.sort(
            key=lambda item: (data.started_at[item[0]], data.match_id[item[0]])
        )
        carry = 1.0
        for match_pos, is_team1, transition in reversed(team_observations):
            if is_team1:
                confidence1[match_pos] = carry
            else:
                confidence2[match_pos] = carry
            carry *= transition
    return confidence1, confidence2


def load_events() -> Events:
    """Load played maps and veto steps from the parquet export of the match JSON."""
    con = duckdb.connect()
    matches = con.execute(
        f"""
        SELECT
            hltv_match_id,
            TRY_CAST(date AS TIMESTAMP) AS started_at,
            TRIM(team1_name) AS team1_name,
            TRIM(team2_name) AS team2_name,
            COALESCE(is_bo1, 0) AS is_bo1,
            COALESCE(is_bo3, 0) AS is_bo3,
            COALESCE(is_bo5, 0) AS is_bo5
        FROM '{PARQUET_DIR}/matches.parquet'
        WHERE TRY_CAST(date AS TIMESTAMP) IS NOT NULL
          AND NULLIF(TRIM(team1_name), '') IS NOT NULL
          AND NULLIF(TRIM(team2_name), '') IS NOT NULL
        """
    ).df()
    map_rows = con.execute(
        f"""
        SELECT hltv_match_id, map_name, score
        FROM '{PARQUET_DIR}/maps.parquet'
        """
    ).df()
    veto_rows = con.execute(
        f"""
        SELECT hltv_match_id, step, team, action, map
        FROM '{PARQUET_DIR}/vetos.parquet'
        """
    ).df()
    player_rows = con.execute(
        f"""
        SELECT hltv_match_id, team, player_name
        FROM '{PARQUET_DIR}/players.parquet'
        """
    ).df()
    con.close()

    matches["started_at"] = pd.to_datetime(matches["started_at"], utc=True).dt.tz_convert(
        None
    )
    matches["team1_name"] = matches["team1_name"].astype(str).str.strip()
    matches["team2_name"] = matches["team2_name"].astype(str).str.strip()
    matches = matches[
        (matches["team1_name"] != "")
        & (matches["team2_name"] != "")
        & (matches["team1_name"] != matches["team2_name"])
    ]
    matches = matches.drop_duplicates("hltv_match_id", keep="first")
    matches = matches.sort_values(
        ["started_at", "hltv_match_id"], kind="mergesort"
    ).reset_index(drop=True)
    (
        team1_ids,
        team2_ids,
        team1_roster_overlap,
        team2_roster_overlap,
    ) = _assign_team_identities(matches, player_rows)

    map_names = sorted(
        set(map_rows["map_name"].astype(str).str.strip().str.lower())
        | set(veto_rows["map"].astype(str).str.strip().str.lower())
    )
    map_names = [name for name in map_names if name and name != "nan"]
    map_index = {name: i for i, name in enumerate(map_names)}
    n_matches = len(matches)
    n_maps = len(map_names)
    id_to_pos = pd.Series(np.arange(n_matches), index=matches["hltv_match_id"].to_numpy())

    map_rows = map_rows.copy()
    map_rows["map_name"] = map_rows["map_name"].astype(str).str.strip().str.lower()
    extracted = map_rows["score"].astype(str).str.extract(r"(\d+)\s*[:\-]\s*(\d+)")
    map_rows = map_rows[extracted[0].notna() & extracted[1].notna()].copy()
    score1 = extracted.loc[map_rows.index, 0].astype(int)
    score2 = extracted.loc[map_rows.index, 1].astype(int)
    decisive = score1.to_numpy() != score2.to_numpy()
    map_rows = map_rows.loc[decisive].copy()
    score1 = score1.loc[decisive]
    score2 = score2.loc[decisive]
    known_match = map_rows["hltv_match_id"].isin(id_to_pos.index)
    known_map = map_rows["map_name"].isin(map_index)
    map_rows = map_rows[known_match & known_map]
    score1 = score1.loc[map_rows.index]
    score2 = score2.loc[map_rows.index]
    res_match = id_to_pos.loc[map_rows["hltv_match_id"]].to_numpy(dtype=np.int32)
    res_map = map_rows["map_name"].map(map_index).to_numpy(dtype=np.int32)
    res_y = (score1.to_numpy() > score2.to_numpy()).astype(np.float64)

    match_played = np.zeros((n_matches, n_maps), dtype=bool)
    match_played[res_match, res_map] = True

    veto_rows = veto_rows.copy()
    veto_rows["map"] = veto_rows["map"].astype(str).str.strip().str.lower()
    veto_rows["action"] = veto_rows["action"].astype(str).str.strip().str.lower()
    veto_rows["team"] = veto_rows["team"].astype(str).str.strip().str.lower()
    veto_rows = veto_rows[
        veto_rows["hltv_match_id"].isin(id_to_pos.index)
        & veto_rows["map"].isin(map_index)
    ]
    veto_rows = veto_rows.sort_values(
        ["hltv_match_id", "step"], kind="mergesort"
    ).reset_index(drop=True)

    match_pool = np.zeros((n_matches, n_maps), dtype=bool)
    v_match: List[int] = []
    v_step: List[int] = []
    v_actor_team1: List[bool] = []
    v_map: List[int] = []
    v_code: List[int] = []
    v_avail: List[np.ndarray] = []

    for match_id, group in veto_rows.groupby("hltv_match_id", sort=False):
        pos = int(id_to_pos.loc[match_id])
        pool = []
        seen = set()
        parsed = []
        for row in group.itertuples(index=False):
            map_i = map_index.get(row.map)
            if map_i is None:
                continue
            if map_i not in seen:
                seen.add(map_i)
                pool.append(map_i)
            parsed.append(
                (int(row.step), str(row.team), str(row.action), map_i)
            )
        for map_i in pool:
            match_pool[pos, map_i] = True
        remaining = set(pool)
        for step, team, action, map_i in parsed:
            if action not in ("removed", "picked"):
                continue
            if team not in ("team1", "team2"):
                continue
            if map_i not in remaining:
                continue
            avail = np.zeros(n_maps, dtype=bool)
            for left in remaining:
                avail[left] = True
            if action == "picked":
                code = 2
            elif step <= 2:
                code = 0
            else:
                code = 1
            v_match.append(pos)
            v_step.append(step)
            v_actor_team1.append(team == "team1")
            v_map.append(map_i)
            v_code.append(code)
            v_avail.append(avail)
            remaining.remove(map_i)

    bo = np.full(n_matches, 3, dtype=np.int8)
    bo[matches["is_bo1"].to_numpy() == 1] = 1
    bo[matches["is_bo5"].to_numpy() == 1] = 5
    events = Events(
        maps=map_names,
        match_id=matches["hltv_match_id"].to_numpy(dtype=np.int64),
        started_at=matches["started_at"].to_numpy(dtype="datetime64[ns]"),
        team1=matches["team1_name"].astype(object).to_numpy(),
        team2=matches["team2_name"].astype(object).to_numpy(),
        team1_id=team1_ids,
        team2_id=team2_ids,
        team1_roster_overlap=team1_roster_overlap,
        team2_roster_overlap=team2_roster_overlap,
        bo=bo,
        has_veto=match_pool.any(axis=1),
        has_maps=match_played.any(axis=1),
        match_pool=match_pool,
        match_played=match_played,
        res_match=res_match,
        res_map=res_map,
        res_y=res_y,
        veto_match=np.asarray(v_match, dtype=np.int32),
        veto_step=np.asarray(v_step, dtype=np.int16),
        veto_actor_is_team1=np.asarray(v_actor_team1, dtype=bool),
        veto_map=np.asarray(v_map, dtype=np.int16),
        veto_code=np.asarray(v_code, dtype=np.int8),
        veto_avail=np.vstack(v_avail) if v_avail else np.zeros((0, n_maps), dtype=bool),
    )
    if events.veto_avail.size and not np.all(
        events.veto_avail[np.arange(len(events.veto_map)), events.veto_map]
    ):
        raise RuntimeError("A veto choice was not in its available set")
    return events


def _empty_fit_arrays(n_teams: int, n_maps: int):
    return {
        "a": np.zeros(n_teams),
        "b_raw": np.zeros((n_teams, n_maps)),
        "h": np.zeros(n_maps),
        "pi_ban_raw": np.zeros((n_teams, n_maps)),
        "pi_pick_raw": np.zeros((n_teams, n_maps)),
    }


def _initial_strength(
    n_teams: int, t1: np.ndarray, t2: np.ndarray, y: np.ndarray, w: np.ndarray
) -> np.ndarray:
    wins = np.zeros(n_teams)
    losses = np.zeros(n_teams)
    if y.size:
        np.add.at(wins, t1, w * y)
        np.add.at(losses, t1, w * (1.0 - y))
        np.add.at(wins, t2, w * (1.0 - y))
        np.add.at(losses, t2, w * y)
    a = np.log(wins + 1.0) - np.log(losses + 1.0)
    a -= a.mean() if a.size else 0.0
    return a


def fit_latent(
    data: Events,
    kind: str,
    cutoff: np.datetime64,
    warm: Optional[LatentFit] = None,
    maxiter: Optional[int] = None,
) -> LatentFit:
    """Fit one latent model on matches that started before cutoff."""
    from scipy.optimize import minimize

    cutoff = np.datetime64(cutoff, "ns")
    n_maps = len(data.maps)
    res_time = data.started_at[data.res_match]
    res_keep = np.flatnonzero(res_time < cutoff)
    veto_time = (
        data.started_at[data.veto_match]
        if data.veto_match.size
        else np.array([], dtype="datetime64[ns]")
    )
    veto_keep = np.flatnonzero(veto_time < cutoff) if veto_time.size else np.array([], dtype=int)

    res_pos = data.res_match[res_keep]
    t1_ids = data.team1_id[res_pos]
    t2_ids = data.team2_id[res_pos]
    team_chunks = [t1_ids, t2_ids]
    roster_confidence1, roster_confidence2 = _historical_roster_confidence(
        data, cutoff
    )
    actor_is_team1 = data.veto_actor_is_team1[veto_keep] if veto_keep.size else np.array([], dtype=bool)
    veto_pos = data.veto_match[veto_keep] if veto_keep.size else np.array([], dtype=np.int32)
    if kind == "joint" and veto_pos.size:
        actor_ids = np.where(
            actor_is_team1, data.team1_id[veto_pos], data.team2_id[veto_pos]
        )
        opp_ids = np.where(
            actor_is_team1, data.team2_id[veto_pos], data.team1_id[veto_pos]
        )
        team_chunks.extend([actor_ids, opp_ids])
    else:
        actor_ids = np.array([], dtype=object)
        opp_ids = np.array([], dtype=object)
    teams = pd.unique(np.concatenate(team_chunks))
    teams = np.asarray(
        [team for team in teams if isinstance(team, str) and team], dtype=object
    )
    if teams.size == 0:
        raise RuntimeError(f"No teams before {cutoff}")
    team_index = {name: i for i, name in enumerate(teams.tolist())}
    n_teams = len(teams)

    def encode(team_ids: np.ndarray) -> np.ndarray:
        if team_ids.size == 0:
            return np.zeros(0, dtype=np.int32)
        return np.fromiter(
            (team_index[team_id] for team_id in team_ids),
            dtype=np.int32,
            count=len(team_ids),
        )

    t1 = encode(t1_ids)
    t2 = encode(t2_ids)
    res_map = data.res_map[res_keep].astype(np.int32, copy=False)
    y = data.res_y[res_keep]
    w_res = decay_weights(res_time[res_keep], cutoff) * np.sqrt(
        roster_confidence1[res_pos] * roster_confidence2[res_pos]
    )

    if kind == "joint" and veto_pos.size:
        actor = encode(actor_ids)
        opp = encode(opp_ids)
        chosen = data.veto_map[veto_keep].astype(np.int32, copy=False)
        veto_code = data.veto_code[veto_keep]
        avail = data.veto_avail[veto_keep]
        actor_confidence = np.where(
            actor_is_team1,
            roster_confidence1[veto_pos],
            roster_confidence2[veto_pos],
        )
        opp_confidence = np.where(
            actor_is_team1,
            roster_confidence2[veto_pos],
            roster_confidence1[veto_pos],
        )
        w_veto = decay_weights(veto_time[veto_keep], cutoff) * np.sqrt(
            actor_confidence * opp_confidence
        )
    else:
        actor = opp = chosen = veto_code = w_veto = np.zeros(0, dtype=np.int32)
        avail = np.zeros((0, n_maps), dtype=bool)
        veto_code = np.zeros(0, dtype=np.int8)

    a0 = _initial_strength(n_teams, t1, t2, y, w_res)
    b_raw = np.zeros((n_teams, n_maps))
    h0 = np.zeros(n_maps)
    pi_ban_raw = np.zeros((n_teams, n_maps))
    pi_pick_raw = np.zeros((n_teams, n_maps))
    tau_raw = inv_softplus(0.35)
    al_early_raw = inv_softplus(0.5)
    al_late_raw = inv_softplus(0.5)
    al_pick_raw = inv_softplus(0.5)
    sig_raw = inv_softplus(0.10)
    if warm is not None and warm.kind == kind:
        previous = np.array([warm.team_index.get(name, -1) for name in teams.tolist()])
        known = previous >= 0
        a0[known] = warm.a[previous[known]]
        b_raw[known] = warm.b_raw[previous[known]]
        pi_ban_raw[known] = warm.pi_ban_raw[previous[known]]
        pi_pick_raw[known] = warm.pi_pick_raw[previous[known]]
        h0 = warm.h.copy()
        tau_raw = warm.tau_raw
        al_early_raw = warm.al_early_raw
        al_late_raw = warm.al_late_raw
        al_pick_raw = warm.al_pick_raw
        sig_raw = warm.sig_raw

    theta0 = pack_theta(
        kind, a0, b_raw, h0, tau_raw, al_early_raw, al_late_raw, al_pick_raw, sig_raw,
        pi_ban_raw, pi_pick_raw,
    )
    if maxiter is None:
        maxiter = WARM_MAXITER if warm is not None else COLD_MAXITER

    def objective(theta: np.ndarray):
        return nll_and_grad(
            theta, kind, n_teams, n_maps,
            t1, t2, res_map, y, w_res,
            actor, opp, chosen, veto_code, avail, w_veto,
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
    fitted = unpack(result.x, kind, n_teams, n_maps)
    a = np.array(fitted["a"], dtype=np.float64, copy=True)
    b_raw = np.array(fitted["b_raw"], dtype=np.float64, copy=True)
    h = np.array(fitted["h"], dtype=np.float64, copy=True)
    tau = float(fitted["tau"])
    s = a[:, None] + tau * b_raw
    b = tau * b_raw
    sig = float(fitted["sig"])
    pi_ban_raw = np.array(fitted["pi_ban_raw"], copy=True)
    pi_pick_raw = np.array(fitted["pi_pick_raw"], copy=True)

    # Diagonal observed-information approximation; veto steps reduce uncertainty too.
    var_b = np.full((n_teams, n_maps), tau**2)
    train_ll = float("nan")
    if kind != "strength":
        info = np.zeros((n_teams, n_maps))
        if y.size and np.any(w_res > 0):
            eta = s[t1, res_map] - s[t2, res_map] + h[res_map]
            train_ll = float(np.average(softplus(eta) - y * eta, weights=w_res))
            prob = 1.0 / (1.0 + np.exp(-np.clip(eta, -60.0, 60.0)))
            weight = w_res * prob * (1.0 - prob)
            np.add.at(info, (t1, res_map), weight)
            np.add.at(info, (t2, res_map), weight)

        if kind == "joint" and actor.size:
            pi_ban = sig * pi_ban_raw
            pi_pick = sig * pi_pick_raw
            utilities = np.zeros((actor.size, n_maps), dtype=np.float64)
            early = veto_code == 0
            late = veto_code == 1
            pick = veto_code == 2
            if np.any(early):
                utilities[early] = ban_utility(
                    s[actor[early]],
                    s[opp[early]],
                    pi_ban[actor[early]],
                    fitted["al_early"],
                )
            if np.any(late):
                utilities[late] = ban_utility(
                    s[actor[late]],
                    s[opp[late]],
                    pi_ban[actor[late]],
                    fitted["al_late"],
                )
            if np.any(pick):
                utilities[pick] = pick_utility(
                    s[actor[pick]],
                    s[opp[pick]],
                    pi_pick[actor[pick]],
                    fitted["al_pick"],
                )
            utilities = np.where(avail, utilities, -np.inf)
            veto_probs, _ = _softmax_rows(utilities)
            alpha = np.where(
                early,
                fitted["al_early"],
                np.where(late, fitted["al_late"], fitted["al_pick"]),
            )
            curvature = (
                w_veto[:, None]
                * alpha[:, None] ** 2
                * veto_probs
                * (1.0 - veto_probs)
            )
            map_indices = np.tile(np.arange(n_maps), actor.size)
            np.add.at(
                info,
                (np.repeat(actor, n_maps), map_indices),
                curvature.ravel(),
            )
            np.add.at(
                info,
                (np.repeat(opp, n_maps), map_indices),
                curvature.ravel(),
            )

        var_b = (tau**2) / (1.0 + (tau**2) * info)
    elif y.size and np.any(w_res > 0):
        eta = a[t1] - a[t2]
        train_ll = float(np.average(softplus(eta) - y * eta, weights=w_res))

    if veto_code.size:
        step1 = data.veto_step[veto_keep] == 1
        if np.any(step1) and np.sum(w_veto[step1]) > 0:
            p_first = float(np.average(actor_is_team1[step1], weights=w_veto[step1]))
        else:
            p_first = 0.4
    else:
        p_first = 0.4
    p_first = float(np.clip(p_first, 0.15, 0.85))

    return LatentFit(
        kind=kind,
        teams=teams,
        team_index=team_index,
        a=a,
        b_raw=b_raw,
        h=h,
        tau=tau,
        tau_raw=float(softplus_inverse_from_value(tau)) if kind != "strength" else 0.0,
        al_early=float(fitted["al_early"]),
        al_late=float(fitted["al_late"]),
        al_pick=float(fitted["al_pick"]),
        al_early_raw=float(softplus_inverse_from_value(fitted["al_early"])) if kind == "joint" else 0.0,
        al_late_raw=float(softplus_inverse_from_value(fitted["al_late"])) if kind == "joint" else 0.0,
        al_pick_raw=float(softplus_inverse_from_value(fitted["al_pick"])) if kind == "joint" else 0.0,
        sig_pi=sig,
        sig_raw=float(softplus_inverse_from_value(sig)) if kind == "joint" else 0.0,
        pi_ban_raw=pi_ban_raw,
        pi_pick_raw=pi_pick_raw,
        s=s,
        b=b,
        var_b=var_b,
        pi_ban=sig * pi_ban_raw,
        pi_pick=sig * pi_pick_raw,
        p_team1_first=p_first,
        train_map_logloss=train_ll,
        n_results=int(y.size),
        n_veto_steps=int(actor.size) if kind == "joint" else 0,
        n_iter=int(result.nit),
        seconds=seconds,
    )


def softplus_inverse_from_value(value: float) -> float:
    value = float(value)
    if value < 1e-8:
        return -20.0
    return inv_softplus(value)


def _lookup_rows(fit: LatentFit, names: np.ndarray, n_maps: int):
    index = np.array([fit.team_index.get(name, -1) for name in names.tolist()], dtype=np.int32)
    known = index >= 0
    s = np.zeros((len(names), n_maps))
    b = np.zeros((len(names), n_maps))
    var_b = np.full((len(names), n_maps), max(fit.tau, 0.0) ** 2)
    pi_ban = np.zeros((len(names), n_maps))
    pi_pick = np.zeros((len(names), n_maps))
    if np.any(known):
        s[known] = fit.s[index[known]]
        b[known] = fit.b[index[known]]
        var_b[known] = fit.var_b[index[known]]
        pi_ban[known] = fit.pi_ban[index[known]]
        pi_pick[known] = fit.pi_pick[index[known]]
    return s, b, var_b, pi_ban, pi_pick


def bo5_win_prob(map_prob: np.ndarray) -> np.ndarray:
    """P(team 2 wins a first-to-3) from per-map win probabilities."""
    n, n_sims, n_played = map_prob.shape
    state = np.zeros((n, n_sims, 4, 4))
    state[:, :, 0, 0] = 1.0
    for play in range(n_played):
        prob = map_prob[:, :, play]
        nxt = np.zeros_like(state)
        for wins2 in range(3):
            for wins1 in range(3):
                mass = state[:, :, wins2, wins1]
                nxt[:, :, wins2 + 1, wins1] += mass * prob
                nxt[:, :, wins2, wins1 + 1] += mass * (1.0 - prob)
        nxt[:, :, 3, :] += state[:, :, 3, :]
        nxt[:, :, :3, 3] += state[:, :, :3, 3]
        state = nxt
    return state[:, :, 3, 0] + state[:, :, 3, 1] + state[:, :, 3, 2]


def bo3_win_prob(q1, q2, q3):
    return q1 * q2 + q1 * (1.0 - q2) * q3 + (1.0 - q1) * q2 * q3


def _deterministic_series(p_map: np.ndarray, pool: List[int], bo: int) -> np.ndarray:
    """Series probability when the pool is not the standard seven-map veto."""
    probs = p_map[:, pool]
    if bo == 1:
        return np.median(probs, axis=1)
    if bo == 5:
        mean = probs.mean(axis=1)
        tiled = np.repeat(mean[:, None, None], 5, axis=2)
        return bo5_win_prob(tiled)[:, 0]
    ordered = np.sort(probs, axis=1)
    q1 = ordered[:, 0]
    q2 = ordered[:, -1]
    q3 = np.median(probs, axis=1)
    return bo3_win_prob(q1, q2, q3)


def simulate_series_p_team2(
    s1: np.ndarray,
    s2: np.ndarray,
    h: np.ndarray,
    pi_ban1: np.ndarray,
    pi_ban2: np.ndarray,
    pi_pick1: np.ndarray,
    pi_pick2: np.ndarray,
    fit: LatentFit,
    pool: List[int],
    bo: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Monte Carlo the usual veto, then the exact series win probability."""
    n, n_maps = s1.shape
    p_map = 1.0 / (1.0 + np.exp(-np.clip(s2 - s1 - h, -60.0, 60.0)))
    if bo == 3 and len(pool) != 7:
        return _deterministic_series(p_map, pool, bo)
    if bo == 5 and len(pool) != 7:
        return _deterministic_series(p_map, pool, bo)
    if bo == 1:
        n_bans = max(len(pool) - 1, 0)
        actions = [("ban", fit.al_early)] * min(2, n_bans) + [
            ("ban", fit.al_late)
        ] * max(n_bans - 2, 0)
    elif bo == 5:
        actions = [("ban", fit.al_early)] * 2 + [("pick", fit.al_pick)] * 4
    else:
        actions = [
            ("ban", fit.al_early),
            ("ban", fit.al_early),
            ("pick", fit.al_pick),
            ("pick", fit.al_pick),
            ("ban", fit.al_late),
            ("ban", fit.al_late),
        ]
    remaining = np.zeros((n, N_SIMS, n_maps), dtype=bool)
    remaining[:, :, pool] = True
    starter_is_team1 = rng.random((n, N_SIMS)) < fit.p_team1_first
    played: List[np.ndarray] = []
    s1e = s1[:, None, :]
    s2e = s2[:, None, :]
    for step, (kind_name, alpha) in enumerate(actions):
        actor_is_team1 = starter_is_team1 if step % 2 == 0 else ~starter_is_team1
        actor_flag = actor_is_team1[:, :, None]
        s_actor = np.where(actor_flag, s1e, s2e)
        s_opp = np.where(actor_flag, s2e, s1e)
        if kind_name == "pick":
            pi = np.where(actor_flag, pi_pick1[:, None, :], pi_pick2[:, None, :])
            utilities = pick_utility(s_actor, s_opp, pi, alpha)
        else:
            pi = np.where(actor_flag, pi_ban1[:, None, :], pi_ban2[:, None, :])
            utilities = ban_utility(s_actor, s_opp, pi, alpha)
        utilities = np.where(remaining, utilities, -1e9)
        noise = rng.random(utilities.shape)
        gumbel = -np.log(-np.log(np.clip(noise, 1e-9, 1.0 - 1e-9)))
        choice = np.argmax(utilities + gumbel, axis=-1)
        np.put_along_axis(remaining, choice[:, :, None], False, axis=-1)
        if kind_name == "pick":
            played.append(choice)
    played.append(np.argmax(remaining, axis=-1))
    gathered = []
    for choice in played:
        gathered.append(p_map[np.arange(n)[:, None], choice])
    map_probs = np.stack(gathered, axis=-1)
    if bo == 1:
        series = map_probs[:, :, 0]
    elif bo == 5:
        series = bo5_win_prob(map_probs)
    else:
        series = bo3_win_prob(map_probs[:, :, 0], map_probs[:, :, 1], map_probs[:, :, 2])
    return series.mean(axis=1)


def infer_pool(data: Events, cutoff: np.datetime64) -> List[int]:
    """Active maps from recent past vetoes, or from played maps once vetoes stop."""
    cutoff = np.datetime64(cutoff, "ns")
    start = cutoff - np.timedelta64(POOL_WINDOW_DAYS, "D")
    recent = (data.started_at >= start) & (data.started_at < cutoff)
    veto_recent = recent & data.has_veto
    if int(veto_recent.sum()) >= 40:
        freq = data.match_pool[veto_recent].sum(axis=0)
        support = int(veto_recent.sum())
    else:
        played_recent = recent & data.has_maps
        freq = data.match_played[played_recent].sum(axis=0)
        support = int(played_recent.sum())
    if support == 0:
        return []
    order = np.argsort(-freq)
    minimum = max(5, int(0.08 * support))
    pool = [int(i) for i in order if freq[i] >= minimum]
    if len(pool) > 7:
        pool = pool[:7]
    if len(pool) < 3:
        pool = [int(i) for i in order if freq[i] > 0][:7]
    return pool


def features_for_matches(
    fit: LatentFit,
    data: Events,
    match_idx: np.ndarray,
    pool: List[int],
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    n = len(match_idx)
    n_maps = len(data.maps)
    empty = {name: np.full(n, np.nan) for name in MAP_FEATURES}
    if n == 0 or len(pool) == 0:
        return empty
    s1, b1, var1, pi_ban1, pi_pick1 = _lookup_rows(
        fit, data.team1_id[match_idx], n_maps
    )
    s2, b2, var2, pi_ban2, pi_pick2 = _lookup_rows(
        fit, data.team2_id[match_idx], n_maps
    )
    continuity1 = data.team1_roster_overlap[match_idx]
    continuity2 = data.team2_roster_overlap[match_idx]
    s1 = s1 * continuity1[:, None]
    s2 = s2 * continuity2[:, None]
    b1 = b1 * continuity1[:, None]
    b2 = b2 * continuity2[:, None]
    var1 = (
        continuity1[:, None] ** 2 * var1
        + (1.0 - continuity1[:, None] ** 2) * fit.tau**2
    )
    var2 = (
        continuity2[:, None] ** 2 * var2
        + (1.0 - continuity2[:, None] ** 2) * fit.tau**2
    )
    pi_ban1 = pi_ban1 * continuity1[:, None]
    pi_ban2 = pi_ban2 * continuity2[:, None]
    pi_pick1 = pi_pick1 * continuity1[:, None]
    pi_pick2 = pi_pick2 * continuity2[:, None]
    edge = s1[:, pool] - s2[:, pool]
    offset = b1[:, pool] - b2[:, pool]
    se = np.sqrt(np.maximum(var1[:, pool] + var2[:, pool], 0.0))
    p_team2 = np.full(n, np.nan)
    bo = data.bo[match_idx]
    for bo_value in (1, 3, 5):
        selected = bo == bo_value
        if not np.any(selected):
            continue
        p_team2[selected] = simulate_series_p_team2(
            s1[selected], s2[selected], fit.h,
            pi_ban1[selected], pi_ban2[selected],
            pi_pick1[selected], pi_pick2[selected],
            fit, pool, bo_value, rng,
        )
    return {
        "map_p_team2": p_team2,
        "map_edge_mean": edge.mean(axis=1),
        "map_edge_best": edge.max(axis=1),
        "map_edge_worst": edge.min(axis=1),
        "map_offset_edge_mean": offset.mean(axis=1),
        "map_offset_se_mean": se.mean(axis=1),
    }


def _encode_eval_teams(fit: LatentFit, team_ids: np.ndarray) -> np.ndarray:
    return np.array(
        [fit.team_index.get(team_id, -1) for team_id in team_ids.tolist()],
        dtype=np.int32,
    )


def score_map_results(fit: LatentFit, data: Events, start, end, counts: Dict) -> Dict[str, float]:
    """Unweighted map log loss on [start, end), split by how often the map was played before start."""
    start = np.datetime64(start, "ns")
    end = np.datetime64(end, "ns")
    when = data.started_at[data.res_match]
    keep = np.flatnonzero((when >= start) & (when < end))
    if keep.size == 0:
        return {"logloss": np.nan, "n": 0, "rare_logloss": np.nan, "n_rare": 0,
                "common_logloss": np.nan, "n_common": 0}
    pos = data.res_match[keep]
    t1_ids = data.team1_id[pos]
    t2_ids = data.team2_id[pos]
    maps = data.res_map[keep]
    y = data.res_y[keep]
    i1 = _encode_eval_teams(fit, t1_ids)
    i2 = _encode_eval_teams(fit, t2_ids)
    continuity1 = data.team1_roster_overlap[pos]
    continuity2 = data.team2_roster_overlap[pos]
    if fit.kind == "strength":
        a1 = np.where(i1 >= 0, fit.a[np.maximum(i1, 0)], 0.0)
        a2 = np.where(i2 >= 0, fit.a[np.maximum(i2, 0)], 0.0)
        eta = continuity1 * a1 - continuity2 * a2
    else:
        s1 = np.zeros(keep.size)
        s2 = np.zeros(keep.size)
        known1 = i1 >= 0
        known2 = i2 >= 0
        s1[known1] = (
            fit.s[i1[known1], maps[known1]] * continuity1[known1]
        )
        s2[known2] = (
            fit.s[i2[known2], maps[known2]] * continuity2[known2]
        )
        eta = s1 - s2 + fit.h[maps]
    nll = softplus(eta) - y * eta
    rare = np.array(
        [
            min(counts.get((team_a, int(map_i)), 0), counts.get((team_b, int(map_i)), 0))
            < RARE_MAP_PLAYS
            for team_a, team_b, map_i in zip(t1_ids, t2_ids, maps)
        ]
    )
    def mean(mask):
        if not np.any(mask):
            return float("nan"), 0
        return float(nll[mask].mean()), int(mask.sum())

    overall, n = mean(np.ones(keep.size, dtype=bool))
    rare_ll, n_rare = mean(rare)
    common_ll, n_common = mean(~rare)
    return {
        "logloss": overall,
        "n": n,
        "rare_logloss": rare_ll,
        "n_rare": n_rare,
        "common_logloss": common_ll,
        "n_common": n_common,
    }


def play_counts(data: Events, cutoff) -> Dict:
    cutoff = np.datetime64(cutoff, "ns")
    keep = data.started_at[data.res_match] < cutoff
    counts: Dict = {}
    for team_a, team_b, map_i in zip(
        data.team1_id[data.res_match[keep]],
        data.team2_id[data.res_match[keep]],
        data.res_map[keep],
    ):
        counts[(team_a, int(map_i))] = counts.get((team_a, int(map_i)), 0) + 1
        counts[(team_b, int(map_i))] = counts.get((team_b, int(map_i)), 0) + 1
    return counts


def score_vetoes(fit: LatentFit, data: Events, start, end) -> Dict[str, float]:
    start = np.datetime64(start, "ns")
    end = np.datetime64(end, "ns")
    if data.veto_match.size == 0 or fit.kind != "joint":
        return {"logloss": np.nan, "accuracy": np.nan, "uniform_logloss": np.nan, "n": 0}
    when = data.started_at[data.veto_match]
    keep = np.flatnonzero((when >= start) & (when < end))
    if keep.size == 0:
        return {"logloss": np.nan, "accuracy": np.nan, "uniform_logloss": np.nan, "n": 0}
    pos = data.veto_match[keep]
    actor_is_team1 = data.veto_actor_is_team1[keep]
    actor_ids = np.where(
        actor_is_team1, data.team1_id[pos], data.team2_id[pos]
    )
    opp_ids = np.where(
        actor_is_team1, data.team2_id[pos], data.team1_id[pos]
    )
    n_maps = len(data.maps)
    s_actor, _, _, pi_ban_actor, pi_pick_actor = _lookup_rows(fit, actor_ids, n_maps)
    s_opp, _, _, _, _ = _lookup_rows(fit, opp_ids, n_maps)
    continuity_actor = np.where(
        actor_is_team1,
        data.team1_roster_overlap[pos],
        data.team2_roster_overlap[pos],
    )
    continuity_opp = np.where(
        actor_is_team1,
        data.team2_roster_overlap[pos],
        data.team1_roster_overlap[pos],
    )
    s_actor *= continuity_actor[:, None]
    s_opp *= continuity_opp[:, None]
    pi_ban_actor *= continuity_actor[:, None]
    pi_pick_actor *= continuity_actor[:, None]
    code = data.veto_code[keep]
    utilities = np.zeros((keep.size, n_maps))
    early = code == 0
    late = code == 1
    pick = code == 2
    if np.any(early):
        utilities[early] = ban_utility(
            s_actor[early], s_opp[early], pi_ban_actor[early], fit.al_early
        )
    if np.any(late):
        utilities[late] = ban_utility(
            s_actor[late], s_opp[late], pi_ban_actor[late], fit.al_late
        )
    if np.any(pick):
        utilities[pick] = pick_utility(
            s_actor[pick], s_opp[pick], pi_pick_actor[pick], fit.al_pick
        )
    avail = data.veto_avail[keep]
    utilities = np.where(avail, utilities, -np.inf)
    probs, log_z = _softmax_rows(utilities)
    chosen = data.veto_map[keep].astype(int)
    nll = -(utilities[np.arange(keep.size), chosen] - log_z)
    accuracy = probs.argmax(axis=1) == chosen
    n_avail = avail.sum(axis=1)
    return {
        "logloss": float(nll.mean()),
        "accuracy": float(accuracy.mean()),
        "uniform_logloss": float(np.log(n_avail).mean()),
        "uniform_accuracy": float(np.mean(1.0 / n_avail)),
        "n": int(keep.size),
    }


def format_pool(data: Events, pool: List[int]) -> str:
    if not pool:
        return "(none)"
    return ",".join(data.maps[i] for i in pool)


def offset_examples(fit: LatentFit, data: Events, cutoff, limit: int = 8) -> List[str]:
    cutoff = np.datetime64(cutoff, "ns")
    keep = data.started_at < cutoff
    counts: Dict[str, int] = {}
    for team_id in np.concatenate(
        [data.team1_id[keep], data.team2_id[keep]]
    ):
        counts[team_id] = counts.get(team_id, 0) + 1
    rows = []
    for name, index in fit.team_index.items():
        if counts.get(name, 0) < 20:
            continue
        for map_i, map_name in enumerate(data.maps):
            rows.append((float(fit.b[index, map_i]), name, map_name))
    rows.sort(key=lambda item: item[0], reverse=True)
    lines = []
    for value, name, map_name in rows[:limit]:
        lines.append(f"  {name} on {map_name}: {value:+.3f}")
    lines.append("  ...")
    for value, name, map_name in rows[-limit:]:
        lines.append(f"  {name} on {map_name}: {value:+.3f}")
    return lines


def strength_examples(fit: LatentFit, limit: int = 8) -> List[str]:
    order = np.argsort(-fit.a)[:limit]
    return [f"  {fit.teams[i]}: {fit.a[i]:+.3f}" for i in order]


def run_latent_diagnostic(data: Events) -> List[str]:
    """Compare strength-only, outcome offsets, and the joint model on the next time slice."""
    order = np.argsort(data.started_at, kind="mergesort")
    cut = data.started_at[order[int(len(order) * TRAIN_FRACTION)]]
    end = data.started_at[order[int(len(order) * (TRAIN_FRACTION + VALIDATION_FRACTION))]]
    lines = [
        "Latent map-strength diagnostic",
        f"Fit on matches before {np.datetime64(cut, 'm')}, "
        f"score matches from then until {np.datetime64(end, 'm')}.",
        "Map log loss is the mean negative log probability of the played-map winner.",
        "Rare means at least one of the two teams had played that map fewer than "
        f"{RARE_MAP_PLAYS} times before the cutoff.",
        "",
    ]
    counts = play_counts(data, cut)
    fits = {}
    for kind, label in (
        ("strength", "Overall strength only"),
        ("outcome", "Outcome map offsets"),
        ("joint", "Joint outcomes and vetoes"),
    ):
        started = time.time()
        fit = fit_latent(data, kind, cut, warm=None, maxiter=COLD_MAXITER)
        fits[kind] = fit
        scored = score_map_results(fit, data, cut, end, counts)
        veto = score_vetoes(fit, data, cut, end)
        lines.append(
            f"{label}: map ll {scored['logloss']:.4f} "
            f"(n={scored['n']}, rare {scored['rare_logloss']:.4f} n={scored['n_rare']}, "
            f"common {scored['common_logloss']:.4f} n={scored['n_common']})"
        )
        if kind == "joint":
            lines.append(
                f"  veto ll {veto['logloss']:.4f} vs uniform {veto['uniform_logloss']:.4f}, "
                f"top-1 {veto['accuracy']:.3f} vs uniform {veto['uniform_accuracy']:.3f} "
                f"(n={veto['n']})"
            )
            lines.append(
                f"  tau {fit.tau:.3f}, alpha early-ban {fit.al_early:.3f}, "
                f"late-ban {fit.al_late:.3f}, pick {fit.al_pick:.3f}, "
                f"pi scale {fit.sig_pi:.3f}"
            )
        lines.append(
            f"  train weighted map ll {fit.train_map_logloss:.4f}, "
            f"iters {fit.n_iter}, {time.time() - started:.0f}s, "
            f"results {fit.n_results}, veto steps {fit.n_veto_steps}"
        )
    joint = fits["joint"]
    lines.append("")
    lines.append(f"Pool at the diagnostic cutoff: {format_pool(data, infer_pool(data, cut))}")
    lines.append("Highest overall strengths a(team):")
    lines.extend(strength_examples(joint))
    lines.append("Largest and smallest map offsets b(team, map) among teams with 20+ matches:")
    lines.extend(offset_examples(joint, data, cut))
    lines.append("")
    return lines


def build_feature_frame(data: Events, log) -> pd.DataFrame:
    """Refit at each block edge and fill features for matches in the following block."""
    first = data.started_at.min() + np.timedelta64(MIN_HISTORY_DAYS, "D")
    last = data.started_at.max()
    cutoffs = []
    cursor = first
    while cursor <= last:
        cutoffs.append(np.datetime64(cursor, "ns"))
        cursor = cursor + np.timedelta64(BLOCK_DAYS, "D")
    columns = {name: np.full(len(data.match_id), np.nan) for name in MAP_FEATURES}
    warm: Optional[LatentFit] = None
    log(
        f"Walk-forward map strength: {len(cutoffs)} refits, "
        f"half-life {HALF_LIFE_DAYS:.0f} days, block {BLOCK_DAYS} days."
    )
    for i, cutoff in enumerate(cutoffs):
        nxt = cutoffs[i + 1] if i + 1 < len(cutoffs) else last + np.timedelta64(1, "D")
        apply_idx = np.flatnonzero((data.started_at >= cutoff) & (data.started_at < nxt))
        if apply_idx.size == 0:
            continue
        fit = fit_latent(
            data, "joint", cutoff, warm=warm,
            maxiter=COLD_MAXITER if warm is None else WARM_MAXITER,
        )
        warm = fit
        pool = infer_pool(data, cutoff)
        seed = int(cutoff.astype("datetime64[D]").astype(np.int64)) % (2**31 - 1)
        feats = features_for_matches(
            fit, data, apply_idx, pool, np.random.default_rng(seed)
        )
        for name, values in feats.items():
            columns[name][apply_idx] = values
        log(
            f"  {np.datetime64(cutoff, 'D')}: matches {apply_idx.size}, "
            f"tau {fit.tau:.3f}, alpha ban {fit.al_early:.3f}/{fit.al_late:.3f}, "
            f"pick {fit.al_pick:.3f}, pi {fit.sig_pi:.3f}, "
            f"pool [{format_pool(data, pool)}], {fit.seconds:.0f}s"
        )
    frame = pd.DataFrame({"hltv_match_id": data.match_id, **columns})
    covered = int(frame["map_p_team2"].notna().sum())
    log(f"Map-strength features filled for {covered} of {len(frame)} matches.")
    return frame


def generate_features() -> pd.DataFrame:
    """Fit the historical model and save one pre-match feature row per match."""
    print("Loading matches, maps, and vetoes...", flush=True)
    data = load_events()
    print(
        f"Matches {len(data.match_id)}, map results {len(data.res_y)}, "
        f"veto steps {len(data.veto_map)}, maps {len(data.maps)} "
        f"({', '.join(data.maps)}).",
        flush=True,
    )
    if data.veto_match.size:
        veto_times = data.started_at[data.veto_match]
        print(
            f"Veto steps run from {np.datetime64(veto_times.min(), 'D')} "
            f"to {np.datetime64(veto_times.max(), 'D')}.",
            flush=True,
        )
    lines: List[str] = []

    def log(message: str) -> None:
        print(message, flush=True)
        lines.append(message)

    log(
        f"Loaded {len(data.match_id)} matches, {len(data.res_y)} played maps, "
        f"{len(data.veto_map)} modeled veto choices, and {len(data.maps)} maps."
    )
    frame = build_feature_frame(data, log)
    if not frame["hltv_match_id"].is_unique:
        raise ValueError("Map-strength features contain duplicate match IDs.")
    missing = set(["hltv_match_id", *MAP_FEATURES]) - set(frame.columns)
    if missing:
        raise ValueError(f"Generated map-strength features are missing: {sorted(missing)}")

    os.makedirs(FEATURES_ROOT, exist_ok=True)
    frame.to_parquet(FEATURE_PATH, index=False)
    report = "\n".join(lines) + "\n"
    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        handle.write(report)
    print(f"Saved {len(frame)} match rows to {FEATURE_PATH}", flush=True)
    return frame


def run_self_checks() -> None:
    """Check likelihood math, series math, and roster-continuity behavior."""
    rng = np.random.default_rng(1)
    n_teams, n_maps = 3, 2
    t1 = rng.integers(0, n_teams, size=10)
    t2 = rng.integers(0, n_teams, size=10)
    t2 = np.where(t1 == t2, (t2 + 1) % n_teams, t2)
    res_map = rng.integers(0, n_maps, size=10)
    y = rng.integers(0, 2, size=10).astype(np.float64)
    w_res = rng.uniform(0.4, 1.0, size=10)
    actor = rng.integers(0, n_teams, size=8)
    opp = np.where(actor == 0, 1, 0)
    chosen = rng.integers(0, n_maps, size=8)
    veto_code = np.array([0, 0, 1, 1, 2, 2, 0, 2], dtype=np.int8)
    avail = np.ones((8, n_maps), dtype=bool)
    avail[0, 1] = False
    chosen[0] = 0
    w_veto = rng.uniform(0.4, 1.0, size=8)
    theta = rng.normal(
        scale=0.3, size=parameter_count("joint", n_teams, n_maps)
    )
    args = (
        "joint",
        n_teams,
        n_maps,
        t1.astype(np.int32),
        t2.astype(np.int32),
        res_map.astype(np.int32),
        y,
        w_res,
        actor.astype(np.int32),
        opp.astype(np.int32),
        chosen.astype(np.int32),
        veto_code,
        avail,
        w_veto,
    )

    def objective(values):
        return nll_and_grad(values, *args)

    _, gradient = objective(theta)
    numeric = np.zeros_like(theta)
    epsilon = 1e-5
    for index in range(theta.size):
        step = np.zeros_like(theta)
        step[index] = epsilon
        numeric[index] = (
            objective(theta + step)[0] - objective(theta - step)[0]
        ) / (2 * epsilon)
    if not np.allclose(numeric, gradient, rtol=1e-4, atol=1e-5):
        raise AssertionError(
            f"Joint likelihood gradient check failed; "
            f"max abs gap {np.max(np.abs(numeric - gradient))}"
        )

    empty = np.zeros(0, dtype=np.int32)
    empty_float = np.zeros(0, dtype=np.float64)

    def veto_nll(values):
        return nll_and_grad(
            values,
            "joint",
            n_teams,
            n_maps,
            empty,
            empty,
            empty,
            empty_float,
            empty_float,
            actor.astype(np.int32),
            opp.astype(np.int32),
            chosen.astype(np.int32),
            veto_code,
            avail,
            np.ones(8),
        )[0]

    baseline = np.zeros(parameter_count("joint", n_teams, n_maps))
    shifted = baseline.copy()
    shifted[0] = 1.0
    if abs((veto_nll(shifted) - veto_nll(baseline)) - 0.5) > 1e-6:
        raise AssertionError("Overall team strength leaked into the veto likelihood")

    probability = 1.0 / (1.0 + np.exp(-2.0))
    if bo3_win_prob(probability, probability, probability) <= 0.5:
        raise AssertionError("BO3 probability moved the wrong way")
    if bo5_win_prob(np.ones((1, 1, 5)))[0, 0] != 1.0:
        raise AssertionError("BO5 probability failed at the certain-win extreme")
    if bo5_win_prob(np.zeros((1, 1, 5)))[0, 0] != 0.0:
        raise AssertionError("BO5 probability failed at the zero-win extreme")

    rosters = {
        1: {"p1", "p2", "p3", "p4", "p5"},
        2: {"p1", "p2", "p3", "p4", "p5"},
        3: {"p1", "p2", "p3", "p4", "p6"},
        4: {"p1", "p2", "p6", "p7", "p8"},
        5: {"p1", "p2", "p7", "p9", "p10"},
        6: {"p1", "p2", "p7", "p9", "p10"},
    }
    matches = pd.DataFrame(
        {
            "hltv_match_id": [1, 2, 3, 4, 5, 6],
            "started_at": pd.date_range("2020-01-01", periods=6, freq="D"),
            "team1_name": ["Org", "Alias", "Alias", "Alias", "Rebuild", "Rebrand"],
            "team2_name": ["Opp1", "Opp2", "Opp3", "Opp4", "Opp5", "Opp6"],
        }
    )
    player_rows = pd.DataFrame(
        [
            (match_id, "team1", player)
            for match_id, roster in rosters.items()
            for player in roster
        ],
        columns=["hltv_match_id", "team", "player_name"],
    )
    team1_ids, _, overlap1, _ = _assign_team_identities(matches, player_rows)
    if not np.all(team1_ids[:4] == team1_ids[0]):
        raise AssertionError("A team rename or same-name roster change split identity")
    if team1_ids[4] == team1_ids[0] or team1_ids[5] != team1_ids[4]:
        raise AssertionError("Roster bridge incorrectly linked or split an alias")
    if not np.allclose(overlap1, [1.0, 1.0, 0.8, 0.6, 1.0, 1.0]):
        raise AssertionError(f"Unexpected roster overlap weights: {overlap1}")

    probe = type(
        "RosterHistory",
        (),
        {
            "match_id": np.arange(6),
            "started_at": matches["started_at"].to_numpy(dtype="datetime64[ns]"),
            "team1_id": team1_ids,
            "team2_id": np.array(
                ["opp1", "opp2", "opp3", "opp4", "opp5", "opp6"], dtype=object
            ),
            "team1_roster_overlap": overlap1,
            "team2_roster_overlap": np.ones(6),
        },
    )()
    confidence1, _ = _historical_roster_confidence(
        probe, np.datetime64("2020-01-08")
    )
    if not np.allclose(confidence1, [0.48, 0.48, 0.6, 1.0, 1.0, 1.0]):
        raise AssertionError(
            f"Historical evidence was not discounted after roster changes: {confidence1}"
        )
    print("Map-strength feature self-checks passed.", flush=True)


if __name__ == "__main__":
    if "--check" in sys.argv:
        run_self_checks()
    else:
        generate_features()
