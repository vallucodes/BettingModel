"""
Walk-forward map-strength features on top of the xgboost_1 match model.

One latent number per team and map,

    s(team, map) = a(team) + b(team, map)

is fit by penalised maximum likelihood. Played maps use a Bradley-Terry
likelihood. Each ban or pick is a conditional logit over the maps still
available in that match's veto. The same s explains both, so a map that is
always banned still moves b. Overall strength a cancels inside a veto step.

NUTS is too slow to refit along the timeline, so L2 penalties stand in for
the priors and L-BFGS fits the mode. b is penalised as Normal(0, 0.5^2) and
the veto habit terms pi as Normal(0, 0.3^2). Those scales stay fixed: a free
scale next to thousands of coefficients has a joint mode that lets the scale
grow and the penalty disappear. The veto slopes alpha are still learned.
Features for a match come from a model fit only on earlier matches.

Veto rows in this dataset stop on 2026-03-07. After that, s is still updated
from map results, veto coefficients keep their decayed history, and the
active pool is read from maps that were actually played.

Usage, from the project root:
    python model/xgboost_3.py
    python model/xgboost_3.py --check
"""

import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler


MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
PARQUET_DIR = (
    "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/parquet"
)
FEATURES_ROOT = (
    "/media/vallu/Storage/Coding/Own_projects/betting_model/vallu_scraper/data/features"
)
ROSTER_TRUESKILL_DIR = os.path.join(FEATURES_ROOT, "roster_trueskill")
PLOTS_DIR = os.path.join(MODEL_DIR, "plots")
CACHE_DIR = os.path.join(MODEL_DIR, "cache")

FEATURE_VERSION = "v2"
HALF_LIFE_DAYS = 180.0
MIN_HISTORY_DAYS = 120
BLOCK_DAYS = 30
POOL_WINDOW_DAYS = 120
COLD_MAXITER = 60
WARM_MAXITER = 15
N_SIMS = 64
RARE_MAP_PLAYS = 5

A_PRIOR_SD = 1.0
H_PRIOR_SD = 0.2
TAU_PRIOR_SD = 0.5
ALPHA_PRIOR_SD = 1.0
SIG_PI_PRIOR_SD = 0.3

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
MAX_BOOST_ROUNDS = 3000
EARLY_STOPPING_ROUNDS = 100
MAX_LL_DELTA = 0.05
MAX_ROC_DELTA = 0.02
MAX_BR_DELTA = 0.01

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "max_depth": 3,
    "eta": 0.01,
    "subsample": 0.5,
    "colsample_bytree": 0.5,
    "min_child_weight": 20,
    "gamma": 1.0,
    "lambda": 2.0,
    "alpha": 0.0,
    "scale_pos_weight": 1.0,
}

BASE_FEATURES = [
    "is_lan",
    "is_bo1",
    "is_bo3",
    "is_bo5",
    "kast_diff_l15",
    "swing_diff_l15",
    "team1_rolling_win_rate_l15",
    "team2_rolling_win_rate_l15",
    "team1_roster_trueskill_conservative",
    "team2_roster_trueskill_conservative",
]
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
        n += n_teams * n_maps + n_maps
    if kind == "joint":
        n += 3 + 2 * n_teams * n_maps
    return n


def pack_theta(
    kind: str,
    a: np.ndarray,
    b: np.ndarray,
    h: np.ndarray,
    al_early_raw: float,
    al_late_raw: float,
    al_pick_raw: float,
    pi_ban: np.ndarray,
    pi_pick: np.ndarray,
) -> np.ndarray:
    parts = [np.asarray(a, dtype=np.float64).ravel()]
    if kind != "strength":
        parts.append(np.asarray(b, dtype=np.float64).ravel())
        parts.append(np.asarray(h, dtype=np.float64).ravel())
    if kind == "joint":
        parts.append(
            np.array([al_early_raw, al_late_raw, al_pick_raw], dtype=np.float64)
        )
        parts.append(np.asarray(pi_ban, dtype=np.float64).ravel())
        parts.append(np.asarray(pi_pick, dtype=np.float64).ravel())
    return np.concatenate(parts)


def unpack(theta: np.ndarray, kind: str, n_teams: int, n_maps: int) -> Dict[str, Any]:
    theta = np.asarray(theta, dtype=np.float64)
    i = 0
    a = theta[i : i + n_teams]
    i += n_teams
    b = np.zeros((n_teams, n_maps))
    h = np.zeros(n_maps)
    al_early = al_late = al_pick = 0.0
    d_al_early = d_al_late = d_al_pick = 0.0
    pi_ban = np.zeros((n_teams, n_maps))
    pi_pick = np.zeros((n_teams, n_maps))
    if kind != "strength":
        b = theta[i : i + n_teams * n_maps].reshape(n_teams, n_maps)
        i += n_teams * n_maps
        h = theta[i : i + n_maps]
        i += n_maps
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
        pi_ban = theta[i : i + n_teams * n_maps].reshape(n_teams, n_maps)
        i += n_teams * n_maps
        pi_pick = theta[i : i + n_teams * n_maps].reshape(n_teams, n_maps)
        i += n_teams * n_maps
    if i != theta.size:
        raise ValueError(f"theta size {theta.size} does not match layout {i}")
    return {
        "a": a,
        "b": b,
        "h": h,
        "al_early": al_early,
        "d_al_early": d_al_early,
        "al_late": al_late,
        "d_al_late": d_al_late,
        "al_pick": al_pick,
        "d_al_pick": d_al_pick,
        "pi_ban": pi_ban,
        "pi_pick": pi_pick,
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
    b = p["b"]
    h = p["h"]
    s = a[:, None] + b

    nll = 0.0
    g_a = np.zeros(n_teams)
    g_b = np.zeros((n_teams, n_maps))
    g_h = np.zeros(n_maps)
    g_al_early = 0.0
    g_al_late = 0.0
    g_al_pick = 0.0
    g_pi_ban = np.zeros((n_teams, n_maps))
    g_pi_pick = np.zeros((n_teams, n_maps))
    g_s = np.zeros((n_teams, n_maps))

    if y.size:
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
        pi_ban = p["pi_ban"]
        pi_pick = p["pi_pick"]
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
            np.add.at(g_pi_ban, actor[early], g_u[early])
        if np.any(late):
            g_al_late += float(np.sum(g_u[late] * diff[late]))
            g_diff = g_u[late] * p["al_late"]
            np.add.at(g_s, opp[late], g_diff)
            np.add.at(g_s, actor[late], -g_diff)
            np.add.at(g_pi_ban, actor[late], g_u[late])
        if np.any(pick):
            g_al_pick += float(np.sum(g_u[pick] * (-diff[pick])))
            g_diff = g_u[pick] * (-p["al_pick"])
            np.add.at(g_s, opp[pick], g_diff)
            np.add.at(g_s, actor[pick], -g_diff)
            np.add.at(g_pi_pick, actor[pick], g_u[pick])

    if kind == "strength":
        nll += float(np.sum(a * a) / (2.0 * A_PRIOR_SD**2))
        g_a = g_a + a / (A_PRIOR_SD**2)
    else:
        g_b = g_b + g_s
        g_a = g_a + g_s.sum(axis=1)
        nll += float(np.sum(a * a) / (2.0 * A_PRIOR_SD**2))
        g_a = g_a + a / (A_PRIOR_SD**2)
        nll += float(np.sum(b * b) / (2.0 * TAU_PRIOR_SD**2))
        g_b = g_b + b / (TAU_PRIOR_SD**2)
        nll += float(np.sum(h * h) / (2.0 * H_PRIOR_SD**2))
        g_h = g_h + h / (H_PRIOR_SD**2)

    if kind == "joint":
        nll += (p["al_early"] ** 2 + p["al_late"] ** 2 + p["al_pick"] ** 2) / (
            2.0 * ALPHA_PRIOR_SD**2
        )
        g_al_early += p["al_early"] / (ALPHA_PRIOR_SD**2)
        g_al_late += p["al_late"] / (ALPHA_PRIOR_SD**2)
        g_al_pick += p["al_pick"] / (ALPHA_PRIOR_SD**2)
        nll += float(np.sum(p["pi_ban"] ** 2) / (2.0 * SIG_PI_PRIOR_SD**2))
        nll += float(np.sum(p["pi_pick"] ** 2) / (2.0 * SIG_PI_PRIOR_SD**2))
        g_pi_ban = g_pi_ban + p["pi_ban"] / (SIG_PI_PRIOR_SD**2)
        g_pi_pick = g_pi_pick + p["pi_pick"] / (SIG_PI_PRIOR_SD**2)

    chunks = [g_a]
    if kind != "strength":
        chunks.append(g_b.ravel())
        chunks.append(g_h)
    if kind == "joint":
        chunks.append(
            np.array(
                [
                    g_al_early * p["d_al_early"],
                    g_al_late * p["d_al_late"],
                    g_al_pick * p["d_al_pick"],
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
    h: np.ndarray
    al_early: float
    al_late: float
    al_pick: float
    al_early_raw: float
    al_late_raw: float
    al_pick_raw: float
    s: np.ndarray
    b: np.ndarray
    var_b: np.ndarray
    pi_ban: np.ndarray
    pi_pick: np.ndarray
    rms_b: float
    rms_pi: float
    p_team1_first: float
    train_map_logloss: float
    n_results: int
    n_veto_steps: int
    n_iter: int
    seconds: float


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
    t1_names = data.team1[res_pos]
    t2_names = data.team2[res_pos]
    name_chunks = [t1_names, t2_names]
    actor_is_team1 = data.veto_actor_is_team1[veto_keep] if veto_keep.size else np.array([], dtype=bool)
    veto_pos = data.veto_match[veto_keep] if veto_keep.size else np.array([], dtype=np.int32)
    if kind == "joint" and veto_pos.size:
        actor_names = np.where(actor_is_team1, data.team1[veto_pos], data.team2[veto_pos])
        opp_names = np.where(actor_is_team1, data.team2[veto_pos], data.team1[veto_pos])
        name_chunks.extend([actor_names, opp_names])
    else:
        actor_names = np.array([], dtype=object)
        opp_names = np.array([], dtype=object)
    teams = pd.unique(np.concatenate(name_chunks))
    teams = np.asarray([team for team in teams if isinstance(team, str) and team], dtype=object)
    if teams.size == 0:
        raise RuntimeError(f"No teams before {cutoff}")
    team_index = {name: i for i, name in enumerate(teams.tolist())}
    n_teams = len(teams)

    def encode(names: np.ndarray) -> np.ndarray:
        if names.size == 0:
            return np.zeros(0, dtype=np.int32)
        return np.fromiter((team_index[name] for name in names), dtype=np.int32, count=len(names))

    t1 = encode(t1_names)
    t2 = encode(t2_names)
    res_map = data.res_map[res_keep].astype(np.int32, copy=False)
    y = data.res_y[res_keep]
    w_res = decay_weights(res_time[res_keep], cutoff)

    if kind == "joint" and veto_pos.size:
        actor = encode(actor_names)
        opp = encode(opp_names)
        chosen = data.veto_map[veto_keep].astype(np.int32, copy=False)
        veto_code = data.veto_code[veto_keep]
        avail = data.veto_avail[veto_keep]
        w_veto = decay_weights(veto_time[veto_keep], cutoff)
    else:
        actor = opp = chosen = veto_code = w_veto = np.zeros(0, dtype=np.int32)
        avail = np.zeros((0, n_maps), dtype=bool)
        veto_code = np.zeros(0, dtype=np.int8)

    a0 = _initial_strength(n_teams, t1, t2, y, w_res)
    b0 = np.zeros((n_teams, n_maps))
    h0 = np.zeros(n_maps)
    pi_ban0 = np.zeros((n_teams, n_maps))
    pi_pick0 = np.zeros((n_teams, n_maps))
    al_early_raw = inv_softplus(0.5)
    al_late_raw = inv_softplus(0.5)
    al_pick_raw = inv_softplus(0.5)
    if warm is not None and warm.kind == kind:
        previous = np.array([warm.team_index.get(name, -1) for name in teams.tolist()])
        known = previous >= 0
        a0[known] = warm.a[previous[known]]
        b0[known] = warm.b[previous[known]]
        pi_ban0[known] = warm.pi_ban[previous[known]]
        pi_pick0[known] = warm.pi_pick[previous[known]]
        h0 = warm.h.copy()
        al_early_raw = warm.al_early_raw
        al_late_raw = warm.al_late_raw
        al_pick_raw = warm.al_pick_raw

    theta0 = pack_theta(
        kind, a0, b0, h0, al_early_raw, al_late_raw, al_pick_raw, pi_ban0, pi_pick0
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
    b = np.array(fitted["b"], dtype=np.float64, copy=True)
    h = np.array(fitted["h"], dtype=np.float64, copy=True)
    s = a[:, None] + b
    pi_ban = np.array(fitted["pi_ban"], copy=True)
    pi_pick = np.array(fitted["pi_pick"], copy=True)

    var_b = np.full((n_teams, n_maps), TAU_PRIOR_SD**2)
    train_ll = float("nan")
    if y.size and kind != "strength":
        eta = s[t1, res_map] - s[t2, res_map] + h[res_map]
        train_ll = float(np.average(softplus(eta) - y * eta, weights=w_res))
        prob = 1.0 / (1.0 + np.exp(-np.clip(eta, -60.0, 60.0)))
        info = np.zeros((n_teams, n_maps))
        weight = w_res * prob * (1.0 - prob)
        np.add.at(info, (t1, res_map), weight)
        np.add.at(info, (t2, res_map), weight)
        var_b = 1.0 / (1.0 / TAU_PRIOR_SD**2 + info)
    elif y.size:
        eta = a[t1] - a[t2]
        train_ll = float(np.average(softplus(eta) - y * eta, weights=w_res))

    if veto_code.size:
        step1 = data.veto_step[veto_keep] == 1
        if np.any(step1):
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
        h=h,
        al_early=float(fitted["al_early"]),
        al_late=float(fitted["al_late"]),
        al_pick=float(fitted["al_pick"]),
        al_early_raw=float(softplus_inverse_from_value(fitted["al_early"])) if kind == "joint" else 0.0,
        al_late_raw=float(softplus_inverse_from_value(fitted["al_late"])) if kind == "joint" else 0.0,
        al_pick_raw=float(softplus_inverse_from_value(fitted["al_pick"])) if kind == "joint" else 0.0,
        s=s,
        b=b,
        var_b=var_b,
        pi_ban=pi_ban,
        pi_pick=pi_pick,
        rms_b=float(np.sqrt(np.mean(b * b))) if b.size else 0.0,
        rms_pi=float(np.sqrt(0.5 * (np.mean(pi_ban * pi_ban) + np.mean(pi_pick * pi_pick))))
        if pi_ban.size
        else 0.0,
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
    var_b = np.full((len(names), n_maps), TAU_PRIOR_SD**2)
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
    s1, b1, var1, pi_ban1, pi_pick1 = _lookup_rows(fit, data.team1[match_idx], n_maps)
    s2, b2, var2, pi_ban2, pi_pick2 = _lookup_rows(fit, data.team2[match_idx], n_maps)
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


def _encode_eval_teams(fit: LatentFit, names: np.ndarray) -> np.ndarray:
    return np.array([fit.team_index.get(name, -1) for name in names.tolist()], dtype=np.int32)


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
    t1_names = data.team1[pos]
    t2_names = data.team2[pos]
    maps = data.res_map[keep]
    y = data.res_y[keep]
    i1 = _encode_eval_teams(fit, t1_names)
    i2 = _encode_eval_teams(fit, t2_names)
    if fit.kind == "strength":
        a1 = np.where(i1 >= 0, fit.a[np.maximum(i1, 0)], 0.0)
        a2 = np.where(i2 >= 0, fit.a[np.maximum(i2, 0)], 0.0)
        a1 = np.where(i1 >= 0, a1, 0.0)
        a2 = np.where(i2 >= 0, a2, 0.0)
        eta = a1 - a2
    else:
        s1 = np.zeros(keep.size)
        s2 = np.zeros(keep.size)
        known1 = i1 >= 0
        known2 = i2 >= 0
        s1[known1] = fit.s[i1[known1], maps[known1]]
        s2[known2] = fit.s[i2[known2], maps[known2]]
        eta = s1 - s2 + fit.h[maps]
    nll = softplus(eta) - y * eta
    rare = np.array(
        [
            min(counts.get((team_a, int(map_i)), 0), counts.get((team_b, int(map_i)), 0))
            < RARE_MAP_PLAYS
            for team_a, team_b, map_i in zip(t1_names, t2_names, maps)
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
        data.team1[data.res_match[keep]],
        data.team2[data.res_match[keep]],
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
    actor_names = np.where(actor_is_team1, data.team1[pos], data.team2[pos])
    opp_names = np.where(actor_is_team1, data.team2[pos], data.team1[pos])
    n_maps = len(data.maps)
    s_actor, _, _, pi_ban_actor, pi_pick_actor = _lookup_rows(fit, actor_names, n_maps)
    s_opp, _, _, _, _ = _lookup_rows(fit, opp_names, n_maps)
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
    for name in np.concatenate([data.team1[keep], data.team2[keep]]):
        counts[name] = counts.get(name, 0) + 1
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
                f"  alpha early-ban {fit.al_early:.3f}, late-ban {fit.al_late:.3f}, "
                f"pick {fit.al_pick:.3f}, rms(b) {fit.rms_b:.3f}, rms(pi) {fit.rms_pi:.3f}"
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
            f"alpha ban {fit.al_early:.3f}/{fit.al_late:.3f}, "
            f"pick {fit.al_pick:.3f}, rms(b) {fit.rms_b:.3f}, rms(pi) {fit.rms_pi:.3f}, "
            f"pool [{format_pool(data, pool)}], {fit.seconds:.0f}s"
        )
    frame = pd.DataFrame({"hltv_match_id": data.match_id, **columns})
    covered = int(frame["map_p_team2"].notna().sum())
    log(f"Map-strength features filled for {covered} of {len(frame)} matches.")
    return frame


def cache_paths():
    base = os.path.join(CACHE_DIR, f"map_strength_{FEATURE_VERSION}")
    return base + ".parquet", base + "_report.txt"


def load_or_build_features() -> tuple[pd.DataFrame, str]:
    feature_path, report_path = cache_paths()
    if os.path.exists(feature_path) and os.path.exists(report_path):
        frame = pd.read_parquet(feature_path)
        with open(report_path, "r", encoding="utf-8") as handle:
            report = handle.read()
        print(f"Loaded cached map-strength features from {feature_path}", flush=True)
        print(report, flush=True)
        return frame, report

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

    lines.extend(run_latent_diagnostic(data))
    for line in lines:
        print(line, flush=True)
    frame = build_feature_frame(data, log)
    os.makedirs(CACHE_DIR, exist_ok=True)
    frame.to_parquet(feature_path, index=False)
    report = "\n".join(lines) + "\n"
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(report)
    print(f"Saved {feature_path}", flush=True)
    return frame, report


def split_chronologically(df: pd.DataFrame):
    df = df.sort_values(
        ["started_at", "hltv_match_id"], na_position="last"
    ).reset_index(drop=True)
    train_end = int(len(df) * TRAIN_FRACTION)
    validation_end = int(len(df) * (TRAIN_FRACTION + VALIDATION_FRACTION))
    if train_end == 0 or validation_end <= train_end or validation_end >= len(df):
        raise ValueError(
            "Not enough dated matches for non-empty train, validation, and test splits."
        )
    return (
        df.iloc[:train_end].reset_index(drop=True),
        df.iloc[train_end:validation_end].reset_index(drop=True),
        df.iloc[validation_end:].reset_index(drop=True),
    )


def best_boost_rounds(bst: xgb.Booster) -> int:
    best_iteration = getattr(bst, "best_iteration", None)
    if best_iteration is not None:
        return int(best_iteration) + 1
    best_ntree_limit = getattr(bst, "best_ntree_limit", None)
    if best_ntree_limit:
        return int(best_ntree_limit)
    return MAX_BOOST_ROUNDS


def predict_at_best_iteration(bst: xgb.Booster, data: xgb.DMatrix) -> np.ndarray:
    best_ntree_limit = getattr(bst, "best_ntree_limit", None)
    if best_ntree_limit:
        return bst.predict(data, ntree_limit=best_ntree_limit)
    best_iteration = getattr(bst, "best_iteration", None)
    if best_iteration is not None:
        return bst.predict(data, iteration_range=(0, int(best_iteration) + 1))
    return bst.predict(data)


def save_feature_importance_plots(bst: xgb.Booster, feature_cols: List[str], model_label: str) -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    for importance_type, importance_label in (
        ("gain", "Average gain"),
        ("weight", "Split frequency"),
    ):
        scores = bst.get_score(importance_type=importance_type)
        importance_df = pd.DataFrame(
            {
                "feature": feature_cols,
                "importance": [scores.get(feature, 0.0) for feature in feature_cols],
            }
        ).sort_values("importance", ascending=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(importance_df["feature"], importance_df["importance"])
        ax.set_title(f"XGBoost feature importance ({importance_label})\n{model_label}")
        ax.set_xlabel(importance_label)
        fig.tight_layout()
        filename = f"xgboost_3_{model_label}_{importance_type}.png"
        output_path = os.path.join(PLOTS_DIR, filename)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Saved feature importance plot -> {output_path}", flush=True)


def build_base_dataframe() -> pd.DataFrame:
    con = duckdb.connect()
    df = con.execute(
        f"""
        SELECT
            hltv_match_id,
            TRY_CAST(date AS TIMESTAMP) AS started_at,
            team1_score,
            team2_score,
            event_type,
            is_bo1,
            is_bo3,
            is_bo5
        FROM '{PARQUET_DIR}/matches.parquet'
        WHERE TRY_CAST(date AS TIMESTAMP) IS NOT NULL
        ORDER BY started_at ASC, hltv_match_id ASC
        """
    ).df()
    con.close()
    df["result"] = np.where(df["team1_score"] > df["team2_score"], 0, 1)
    df["is_lan"] = (df["event_type"] == "LAN").astype(int)
    for col in ["is_bo1", "is_bo3", "is_bo5"]:
        df[col] = df[col].fillna(0).astype(int)
    rolling_df = pd.read_parquet(os.path.join(FEATURES_ROOT, "features_rolling_l15.parquet"))[
        [
            "hltv_match_id",
            "team1_rolling_kast_l15",
            "team2_rolling_kast_l15",
            "team1_rolling_swing_l15",
            "team2_rolling_swing_l15",
            "team1_rolling_win_rate_l15",
            "team2_rolling_win_rate_l15",
        ]
    ]
    rolling_df["kast_diff_l15"] = (
        rolling_df["team1_rolling_kast_l15"] - rolling_df["team2_rolling_kast_l15"]
    )
    rolling_df["swing_diff_l15"] = (
        rolling_df["team1_rolling_swing_l15"] - rolling_df["team2_rolling_swing_l15"]
    )
    return df.merge(rolling_df, on="hltv_match_id", how="left")


def _fit_eval_split(df: pd.DataFrame, feature_cols: List[str], boost_rounds: Optional[int] = None):
    train_df, validation_df, test_df = split_chronologically(df)
    if boost_rounds is None:
        scaler = StandardScaler()
        train_inputs = scaler.fit_transform(train_df[feature_cols])
        validation_inputs = scaler.transform(validation_df[feature_cols])
        train_targets = train_df["result"].astype(int)
        validation_targets = validation_df["result"].astype(int)
        dtrain = xgb.DMatrix(train_inputs, label=train_targets, feature_names=feature_cols)
        dvalidation = xgb.DMatrix(
            validation_inputs, label=validation_targets, feature_names=feature_cols
        )
        bst = xgb.train(
            XGB_PARAMS,
            dtrain,
            num_boost_round=MAX_BOOST_ROUNDS,
            evals=[(dtrain, "train"), (dvalidation, "validation")],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )
        train_probs = predict_at_best_iteration(bst, dtrain)
        validation_probs = predict_at_best_iteration(bst, dvalidation)
        metrics = _metric_block(
            train_targets, train_probs, validation_targets, validation_probs,
            len(train_df), len(validation_df), best_boost_rounds(bst),
        )
        return metrics, None
    train_validation_df = pd.concat([train_df, validation_df], ignore_index=True)
    scaler = StandardScaler()
    train_inputs = scaler.fit_transform(train_validation_df[feature_cols])
    test_inputs = scaler.transform(test_df[feature_cols])
    dtrain = xgb.DMatrix(
        train_inputs,
        label=train_validation_df["result"].astype(int),
        feature_names=feature_cols,
    )
    dtest = xgb.DMatrix(
        test_inputs, label=test_df["result"].astype(int), feature_names=feature_cols
    )
    bst = xgb.train(
        XGB_PARAMS, dtrain, num_boost_round=boost_rounds, verbose_eval=False
    )
    return bst, test_df["result"].astype(int), dtest, len(test_df)


def _metric_block(train_y, train_p, val_y, val_p, n_train, n_val, rounds) -> Dict[str, Any]:
    train_ll = log_loss(train_y, train_p)
    validation_ll = log_loss(val_y, val_p)
    train_roc = roc_auc_score(train_y, train_p)
    validation_roc = roc_auc_score(val_y, val_p)
    train_brier = brier_score_loss(train_y, train_p)
    validation_brier = brier_score_loss(val_y, val_p)
    ll_delta = validation_ll - train_ll
    roc_delta = train_roc - validation_roc
    br_delta = validation_brier - train_brier
    overfit = not (
        ll_delta <= MAX_LL_DELTA
        and roc_delta <= MAX_ROC_DELTA
        and br_delta <= MAX_BR_DELTA
    )
    return {
        "train_log_loss": train_ll,
        "train_roc_auc": train_roc,
        "train_brier": train_brier,
        "validation_log_loss": validation_ll,
        "validation_roc_auc": validation_roc,
        "validation_brier": validation_brier,
        "ll_delta": ll_delta,
        "roc_delta": roc_delta,
        "br_delta": br_delta,
        "overfit": overfit,
        "train_matches": n_train,
        "validation_matches": n_val,
        "best_rounds": rounds,
    }


def parse_trueskill_filename(fname: str):
    match = re.match(
        r"features_roster_trueskill_([0-9.]+)_([0-9.]+)_([0-9.]+)\.parquet", fname
    )
    if not match:
        return None, None, None
    return float(match.group(1)), float(match.group(2)), float(match.group(3))


def prepare_match_frame(base_df: pd.DataFrame, ts_path: str) -> pd.DataFrame:
    ts_df = pd.read_parquet(ts_path)
    required = {
        "hltv_match_id",
        "team1_roster_trueskill_conservative",
        "team2_roster_trueskill_conservative",
    }
    missing = required - set(ts_df.columns)
    if missing:
        raise ValueError(f"{os.path.basename(ts_path)} is missing {sorted(missing)}")
    df = base_df.merge(
        ts_df[
            [
                "hltv_match_id",
                "team1_roster_trueskill_conservative",
                "team2_roster_trueskill_conservative",
            ]
        ],
        on="hltv_match_id",
        how="left",
    )
    return df.dropna(
        subset=[
            "team1_roster_trueskill_conservative",
            "team2_roster_trueskill_conservative",
            "map_p_team2",
        ]
    ).reset_index(drop=True)


def evaluate_test(df: pd.DataFrame, feature_cols: List[str], boost_rounds: int, label: str):
    bst, test_y, dtest, n_test = _fit_eval_split(df, feature_cols, boost_rounds)
    save_feature_importance_plots(bst, feature_cols, label)
    probs = bst.predict(dtest)
    return {
        "test_log_loss": log_loss(test_y, probs),
        "test_roc_auc": roc_auc_score(test_y, probs),
        "test_brier": brier_score_loss(test_y, probs),
        "test_matches": n_test,
    }


def run_self_checks() -> None:
    """Finite-difference the joint gradient and check a few probability identities."""
    rng = np.random.default_rng(1)
    n_teams, n_maps = 3, 2
    t1 = rng.integers(0, n_teams, size=10)
    t2 = rng.integers(0, n_teams, size=10)
    same = t1 == t2
    t2 = np.where(same, (t2 + 1) % n_teams, t2)
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
    theta = rng.normal(scale=0.3, size=parameter_count("joint", n_teams, n_maps))

    def objective(th):
        return nll_and_grad(
            th, "joint", n_teams, n_maps,
            t1.astype(np.int32), t2.astype(np.int32), res_map.astype(np.int32),
            y, w_res, actor.astype(np.int32), opp.astype(np.int32),
            chosen.astype(np.int32), veto_code, avail, w_veto,
        )

    value, grad = objective(theta)
    numeric = np.zeros_like(theta)
    eps = 1e-5
    for i in range(theta.size):
        step = np.zeros_like(theta)
        step[i] = eps
        plus, _ = objective(theta + step)
        minus, _ = objective(theta - step)
        numeric[i] = (plus - minus) / (2 * eps)
    if not np.allclose(numeric, grad, rtol=1e-4, atol=1e-5):
        gap = np.max(np.abs(numeric - grad))
        raise AssertionError(f"Gradient check failed, max abs gap {gap}")

    # Adding a constant to one team's overall strength does not change veto probabilities.
    veto_only_y = np.zeros(0, dtype=np.float64)
    empty_i = np.zeros(0, dtype=np.int32)

    def veto_nll(th):
        nll, _ = nll_and_grad(
            th, "joint", n_teams, n_maps,
            empty_i, empty_i, empty_i, veto_only_y, veto_only_y,
            actor.astype(np.int32), opp.astype(np.int32), chosen.astype(np.int32),
            veto_code, avail, np.ones(8),
        )
        return nll

    base = np.zeros(parameter_count("joint", n_teams, n_maps))
    shifted = base.copy()
    shifted[0] += 1.0
    # Prior on a[0] changes by 0.5; veto term must not change.
    if abs((veto_nll(shifted) - veto_nll(base)) - 0.5) > 1e-6:
        raise AssertionError("Overall strength leaked into the veto likelihood")

    q = 1.0 / (1.0 + np.exp(-2.0))
    series = bo3_win_prob(q, q, q)
    # Team 2 is the favourite here because q is P(team 2 wins the map).
    if series <= 0.5:
        raise AssertionError("BO3 probability moved the wrong way")
    certain = bo5_win_prob(np.ones((1, 1, 5)))[0, 0]
    impossible = bo5_win_prob(np.zeros((1, 1, 5)))[0, 0]
    if abs(certain - 1) > 1e-9 or abs(impossible) > 1e-9:
        raise AssertionError("BO5 probability is wrong at the extremes")
    print(f"Self-checks passed (joint nll {value:.3f}).", flush=True)


def main() -> None:
    feature_frame, _report = load_or_build_features()
    base_df = build_base_dataframe()
    base_df["hltv_match_id"] = base_df["hltv_match_id"].astype(np.int64)
    feature_frame = feature_frame.copy()
    feature_frame["hltv_match_id"] = feature_frame["hltv_match_id"].astype(np.int64)
    base_df = base_df.merge(feature_frame, on="hltv_match_id", how="left")

    if not os.path.isdir(ROSTER_TRUESKILL_DIR):
        print(f"ROSTER_TRUESKILL_DIR does not exist: {ROSTER_TRUESKILL_DIR}")
        return
    ts_files = sorted(
        os.path.join(ROSTER_TRUESKILL_DIR, name)
        for name in os.listdir(ROSTER_TRUESKILL_DIR)
        if name.startswith("features_roster_trueskill_") and name.endswith(".parquet")
    )
    if not ts_files:
        print(f"No roster TrueSkill files in {ROSTER_TRUESKILL_DIR}")
        return
    print(f"Found {len(ts_files)} roster TrueSkill feature files.", flush=True)

    map_results = []
    baseline_results = []
    frames = {}
    for path in ts_files:
        fname = os.path.basename(path)
        try:
            frame = prepare_match_frame(base_df, path)
            frames[fname] = frame
            map_metrics, _ = _fit_eval_split(frame, BASE_FEATURES + MAP_FEATURES)
            base_metrics, _ = _fit_eval_split(frame, BASE_FEATURES)
        except Exception as exc:
            print(f"Failed for {fname}: {exc}", flush=True)
            continue
        sigma, beta, tau = parse_trueskill_filename(fname)
        for metrics in (map_metrics, base_metrics):
            metrics.update({"file": fname, "sigma": sigma, "beta": beta, "tau": tau})
        map_results.append(map_metrics)
        baseline_results.append(base_metrics)
        print(
            f"\n{fname}: map-strength val LL {map_metrics['validation_log_loss']:.4f} "
            f"ROC {map_metrics['validation_roc_auc']:.4f} | "
            f"baseline val LL {base_metrics['validation_log_loss']:.4f} "
            f"ROC {base_metrics['validation_roc_auc']:.4f}",
            flush=True,
        )

    if not map_results:
        print("No successful runs.")
        return
    map_df = pd.DataFrame(map_results)
    base_df_metrics = pd.DataFrame(baseline_results)
    map_df["validation_score"] = (
        map_df["validation_log_loss"] + map_df["validation_brier"] - map_df["validation_roc_auc"]
    )
    map_df = map_df.sort_values(
        by=["overfit", "validation_score", "validation_log_loss", "validation_roc_auc"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)
    base_lookup = base_df_metrics.set_index("file")
    map_df["baseline_validation_log_loss"] = map_df["file"].map(
        base_lookup["validation_log_loss"]
    )
    display = [
        "sigma", "beta", "tau",
        "validation_log_loss", "baseline_validation_log_loss",
        "train_log_loss", "validation_roc_auc", "validation_brier",
        "ll_delta", "overfit", "train_matches", "validation_matches",
    ]
    print("\n=== With map-strength features, ranked by validation ===", flush=True)
    print(map_df[display].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    selected = map_df.iloc[0]
    selected_frame = frames[selected["file"]]
    print(
        f"\n=== Held-out test, {selected['file']} "
        f"({int(selected['best_rounds'])} boosting rounds) ===",
        flush=True,
    )
    map_test = evaluate_test(
        selected_frame,
        BASE_FEATURES + MAP_FEATURES,
        int(selected["best_rounds"]),
        "map_strength_" + os.path.splitext(selected["file"])[0].replace(
            "features_roster_trueskill_", ""
        ),
    )
    base_row = base_lookup.loc[selected["file"]]
    base_test = evaluate_test(
        selected_frame,
        BASE_FEATURES,
        int(base_row["best_rounds"]),
        "baseline_" + os.path.splitext(selected["file"])[0].replace(
            "features_roster_trueskill_", ""
        ),
    )
    print(
        f"Map strength test LL {map_test['test_log_loss']:.4f}, "
        f"ROC {map_test['test_roc_auc']:.4f}, "
        f"Brier {map_test['test_brier']:.4f}, n={map_test['test_matches']}",
        flush=True,
    )
    print(
        f"Baseline test LL {base_test['test_log_loss']:.4f}, "
        f"ROC {base_test['test_roc_auc']:.4f}, "
        f"Brier {base_test['test_brier']:.4f}, n={base_test['test_matches']}",
        flush=True,
    )
    print(
        f"Test log-loss change from adding map strength: "
        f"{map_test['test_log_loss'] - base_test['test_log_loss']:+.4f}",
        flush=True,
    )


if __name__ == "__main__":
    if "--check" in sys.argv:
        run_self_checks()
    else:
        main()
