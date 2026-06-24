"""
test_lgvi_dynamics_grid.py

Validation script for the reduced SO(2) Acrobot LGVI simulator.

Run from:
    Numerical_Simulation_SPOT_MPC/

Example:
    python test_lgvi_dynamics_grid.py

This script performs:

1. Equilibrium test:
   thetaR = [0, 0], thetaF = [0, 0], u = 0.
   The system should remain exactly at rest.

2. Dynamics grid test:
   Many constant controls and time steps.
   Checks whether the Newton/root solution satisfies the implemented LGVI residual.

3. Time-step convergence test:
   For each control input, compares final angles against the finest time-step result.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from lie_group_so2 import angle_from_R, angle_from_cayley, cayley_from_R, det_error_so2, orth_error_so2
from solver_lgvi_acrobot import (
    AcrobotReducedState,
    LGVISolveError,
    lgvi_one_step,
    lgvi_one_step_cayley_safe,
    make_model_from_params,
    make_reduced_state_from_absolute,
)


# =============================================================================
# User settings
# =============================================================================

OUT_DIR = Path("lgvi_test_results")
METHODS = ["ab", "cayley"]

PARAMS: Dict[str, Any] = {
    "physical": {
        "m1": 1.0,
        "m2": 1.0,
        "l1": 0.5,
        "l2": 0.5,
        "deltaJ1": 0.01,
        "deltaJ2": 0.01,
        "g": 9.81,
        "p0": [0.0, 0.0],
    }
}

# General non-equilibrium validation state.
INITIAL_THETAR_DEG = np.array([5.0, 5.0], dtype=float)
INITIAL_THETAF_DEG = np.array([0.0, 0.0], dtype=float)

# Equilibrium state. In your convention this is hanging-down rest.
EQUILIBRIUM_THETAR_DEG = np.array([0.0, 0.0], dtype=float)
EQUILIBRIUM_THETAF_DEG = np.array([0.0, 0.0], dtype=float)

# Larger control grid.
CONTROL_VALUES = [
    -20.0, -15.0, -10.0, -7.5, -5.0, -2.5,
    0.0,
    2.5, 5.0, 7.5, 10.0, 15.0, 20.0,
]

# Larger dt grid.
DT_VALUES = [
    0.02,
    0.01,
    0.005,
    0.002,
    0.001,
    0.0005,
]

# Simulation durations.
GRID_TF = 0.1
EQUILIBRIUM_TF = 1.0

# Strict root settings.
ROOT_TOL = 1e-12
LGVI_MAXFEV = 5000

# Important:
# We accept by residual so SciPy cannot falsely mark machine-precision solutions as failed.
ACCEPT_RESIDUAL = True
ACCEPT_RESIDUAL_TOL = 1e-10
USE_MULTISTART = True
MULTISTART_SELECT = "local"  # options: "residual", "local"
LAMBDA_SPIKE_THRESHOLD = 1e4
CAYLEY_ALLOW_SUBSTEPPING = True
CAYLEY_MIN_SUBSTEP_H = 1e-6
CAYLEY_MAX_SUBDIVISIONS = 10
CAYLEY_SINGULARITY_MARGIN_DEG = 5.0

# Validation thresholds.
WARN_RESIDUAL_INF = 1e-8
WARN_KINEMATIC = 1e-10
WARN_SO2 = 1e-10
WARN_EQUILIBRIUM_ANGLE_DEG = 1e-8
WARN_EQUILIBRIUM_STEP_DEG = 1e-8


# =============================================================================
# Basic helpers
# =============================================================================

def deg2rad(x: np.ndarray | float) -> np.ndarray | float:
    return np.deg2rad(x)


def rad2deg(x: np.ndarray | float) -> np.ndarray | float:
    return np.rad2deg(x)


def inf_norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(x, dtype=float).reshape(-1), ord=np.inf))


def fro_norm(A: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(A, dtype=float), ord="fro"))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: List[str] = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_initial_state(
    model: Any,
    h: float,
    thetaR_deg: np.ndarray,
    thetaF_deg: np.ndarray,
) -> AcrobotReducedState:
    return make_reduced_state_from_absolute(
        model=model,
        h=h,
        thetaR=deg2rad(thetaR_deg),
        thetaF=deg2rad(thetaF_deg),
    )


def theta_pair_from_z(model: Any, z: np.ndarray) -> Tuple[float, float]:
    F1, F2, _, _ = model.unpack_reduced_solution(z)
    return angle_from_R(F1), angle_from_R(F2)


def theta_pair_from_y(y: np.ndarray) -> Tuple[float, float]:
    y = np.asarray(y, dtype=float).reshape(6)
    return angle_from_cayley(y[0]), angle_from_cayley(y[1])


def theta_guess_from_candidate(method: str, model: Any, guess: Optional[np.ndarray]) -> Tuple[float, float]:
    if guess is None:
        return math.nan, math.nan
    if method == "ab":
        return theta_pair_from_z(model, np.asarray(guess, dtype=float).reshape(8))
    return theta_pair_from_y(np.asarray(guess, dtype=float).reshape(6))


def lambda_from_z(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    z = np.asarray(z, dtype=float).reshape(8)
    return z[4:6].copy(), z[6:8].copy()


def lambda_from_y(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float).reshape(6)
    return y[2:4].copy(), y[4:6].copy()


def lambda_total_norm_from_parts(lambda0_norm: float, lambda12_norm: float) -> float:
    return float(math.hypot(float(lambda0_norm), float(lambda12_norm)))


def build_ab_multistart_guesses(
    model: Any,
    state: AcrobotReducedState,
    previous_z: Optional[np.ndarray],
) -> List[np.ndarray]:
    previous_f_guess = model.initial_step_guess(
        F1_prev=state.F1_prev,
        F2_prev=state.F2_prev,
    )
    primary = previous_z.copy() if previous_z is not None else previous_f_guess.copy()
    prev_lambda = primary[4:8].copy()
    identity_prev_lambda = np.array([1.0, 0.0, 1.0, 0.0, *prev_lambda], dtype=float)
    identity_zero_lambda = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    previous_f_zero_lambda = previous_f_guess.copy()
    previous_f_zero_lambda[4:8] = 0.0
    return [primary, identity_prev_lambda, identity_zero_lambda, previous_f_zero_lambda]


def build_cayley_multistart_guesses(
    state: AcrobotReducedState,
    previous_y: Optional[np.ndarray],
) -> List[np.ndarray]:
    previous_q = np.array([cayley_from_R(state.F1_prev), cayley_from_R(state.F2_prev)], dtype=float)
    primary = previous_y.copy() if previous_y is not None else np.r_[previous_q, 0.0, 0.0, 0.0, 0.0]
    prev_lam0, prev_lam12 = lambda_from_y(primary)
    zero_q_prev_lambda = np.r_[0.0, 0.0, prev_lam0, prev_lam12].astype(float)
    zero_q_zero_lambda = np.zeros(6, dtype=float)
    previous_q_zero_lambda = np.r_[previous_q, 0.0, 0.0, 0.0, 0.0].astype(float)
    return [primary, zero_q_prev_lambda, zero_q_zero_lambda, previous_q_zero_lambda]


def candidate_score(
    thetaF1_deg: float,
    thetaF2_deg: float,
    thetaF1_guess_deg: float,
    thetaF2_guess_deg: float,
    lambda0_norm: float,
    lambda12_norm: float,
    residual_inf: float,
) -> float:
    if MULTISTART_SELECT == "residual":
        return float(residual_inf)
    thetaF_max_abs_deg = max(abs(thetaF1_deg), abs(thetaF2_deg))
    if np.isfinite(thetaF1_guess_deg) and np.isfinite(thetaF2_guess_deg):
        delta_max = max(abs(thetaF1_deg - thetaF1_guess_deg), abs(thetaF2_deg - thetaF2_guess_deg))
    else:
        delta_max = 0.0
    lambda_total_norm = lambda_total_norm_from_parts(lambda0_norm, lambda12_norm)
    return float(thetaF_max_abs_deg + 0.1 * delta_max + 1.0e-6 * lambda_total_norm)


def solve_one_step_with_optional_multistart(
    model: Any,
    h: float,
    state: AcrobotReducedState,
    u_const: float,
    method: str,
    z_guess: Optional[np.ndarray],
    y_guess: Optional[np.ndarray],
    use_multistart: bool,
) -> Tuple[AcrobotReducedState, Any, np.ndarray, Dict[str, Any], Optional[np.ndarray], Optional[np.ndarray]]:
    if not use_multistart:
        if method == "ab":
            state_next, info, z = lgvi_one_step(
                model=model,
                h=float(h),
                state=state,
                u_k=float(u_const),
                z_guess=z_guess,
                root_tol=ROOT_TOL,
                lgvi_maxfev=LGVI_MAXFEV,
                normalized=False,
                accept_residual=ACCEPT_RESIDUAL,
                accept_residual_tol=ACCEPT_RESIDUAL_TOL,
            )
            selected_guess = z_guess
            next_z_guess = z.copy()
            next_y_guess = y_guess
        else:
            state_next, info, z = lgvi_one_step_cayley_safe(
                model=model,
                h=float(h),
                state=state,
                u_k=float(u_const),
                y_guess=y_guess,
                root_tol=ROOT_TOL,
                lgvi_maxfev=LGVI_MAXFEV,
                accept_residual=ACCEPT_RESIDUAL,
                accept_residual_tol=ACCEPT_RESIDUAL_TOL,
                singularity_margin_deg=CAYLEY_SINGULARITY_MARGIN_DEG,
                allow_substepping=CAYLEY_ALLOW_SUBSTEPPING,
                min_substep_h=CAYLEY_MIN_SUBSTEP_H,
                max_subdivisions=CAYLEY_MAX_SUBDIVISIONS,
            )
            selected_guess = y_guess
            next_z_guess = z_guess
            next_y_guess = np.array(
                [
                    cayley_from_R(state_next.F1_prev),
                    cayley_from_R(state_next.F2_prev),
                    z[4],
                    z[5],
                    z[6],
                    z[7],
                ],
                dtype=float,
            )

        meta = {
            "multistart_used": False,
            "multistart_num_candidates": 1,
            "multistart_num_converged": int(float(info.residual_inf) <= ACCEPT_RESIDUAL_TOL),
            "multistart_selected_index": 0,
            "multistart_selected_score": math.nan,
            "selected_guess": selected_guess,
        }
        return state_next, info, z, meta, next_z_guess, next_y_guess

    guesses = (
        build_ab_multistart_guesses(model, state, z_guess)
        if method == "ab"
        else build_cayley_multistart_guesses(state, y_guess)
    )
    candidates: List[Dict[str, Any]] = []
    last_error: Optional[LGVISolveError] = None

    for idx, guess in enumerate(guesses):
        try:
            if method == "ab":
                state_next, info, z = lgvi_one_step(
                    model=model,
                    h=float(h),
                    state=state,
                    u_k=float(u_const),
                    z_guess=guess,
                    root_tol=ROOT_TOL,
                    lgvi_maxfev=LGVI_MAXFEV,
                    normalized=False,
                    accept_residual=ACCEPT_RESIDUAL,
                    accept_residual_tol=ACCEPT_RESIDUAL_TOL,
                )
            else:
                state_next, info, z = lgvi_one_step_cayley_safe(
                    model=model,
                    h=float(h),
                    state=state,
                    u_k=float(u_const),
                    y_guess=guess,
                    root_tol=ROOT_TOL,
                    lgvi_maxfev=LGVI_MAXFEV,
                    accept_residual=ACCEPT_RESIDUAL,
                    accept_residual_tol=ACCEPT_RESIDUAL_TOL,
                    singularity_margin_deg=CAYLEY_SINGULARITY_MARGIN_DEG,
                    allow_substepping=CAYLEY_ALLOW_SUBSTEPPING,
                    min_substep_h=CAYLEY_MIN_SUBSTEP_H,
                    max_subdivisions=CAYLEY_MAX_SUBDIVISIONS,
                )
        except LGVISolveError as exc:
            last_error = exc
            continue

        if float(info.residual_inf) > ACCEPT_RESIDUAL_TOL:
            continue

        F1, F2, lam0, lam12 = model.unpack_reduced_solution(z)
        info_q1 = float(getattr(info, "q1", math.nan))
        info_q2 = float(getattr(info, "q2", math.nan))
        if method == "cayley" and np.isfinite(info_q1) and np.isfinite(info_q2):
            thetaF1 = angle_from_cayley(info_q1)
            thetaF2 = angle_from_cayley(info_q2)
        else:
            thetaF1, thetaF2 = angle_from_R(F1), angle_from_R(F2)
        thetaF1_deg = float(rad2deg(thetaF1))
        thetaF2_deg = float(rad2deg(thetaF2))
        thetaF1_guess, thetaF2_guess = theta_guess_from_candidate(method, model, guess)
        thetaF1_guess_deg = float(rad2deg(thetaF1_guess)) if np.isfinite(thetaF1_guess) else math.nan
        thetaF2_guess_deg = float(rad2deg(thetaF2_guess)) if np.isfinite(thetaF2_guess) else math.nan
        score = candidate_score(
            thetaF1_deg=thetaF1_deg,
            thetaF2_deg=thetaF2_deg,
            thetaF1_guess_deg=thetaF1_guess_deg,
            thetaF2_guess_deg=thetaF2_guess_deg,
            lambda0_norm=float(np.linalg.norm(lam0)),
            lambda12_norm=float(np.linalg.norm(lam12)),
            residual_inf=float(info.residual_inf),
        )
        candidates.append(
            {
                "index": idx,
                "guess": guess,
                "state_next": state_next,
                "info": info,
                "z": z,
                "score": score,
            }
        )

    if not candidates:
        if last_error is not None:
            raise last_error
        raise LGVISolveError(
            residual_inf=math.inf,
            message="No multi-start candidate reached ACCEPT_RESIDUAL_TOL.",
            nfev=0,
        )

    selected = min(candidates, key=lambda c: float(c["score"]))
    selected_info = selected["info"]
    selected_z = selected["z"]
    selected_guess = selected["guess"]

    if method == "ab":
        next_z_guess = selected_z.copy()
        next_y_guess = y_guess
    else:
        next_z_guess = z_guess
        next_y_guess = np.array(
            [
                cayley_from_R(selected["state_next"].F1_prev),
                cayley_from_R(selected["state_next"].F2_prev),
                selected_z[4],
                selected_z[5],
                selected_z[6],
                selected_z[7],
            ],
            dtype=float,
        )

    meta = {
        "multistart_used": True,
        "multistart_num_candidates": len(guesses),
        "multistart_num_converged": len(candidates),
        "multistart_selected_index": int(selected["index"]),
        "multistart_selected_score": float(selected["score"]),
        "selected_guess": selected_guess,
    }
    return selected["state_next"], selected_info, selected_z, meta, next_z_guess, next_y_guess


# =============================================================================
# Residual and diagnostics
# =============================================================================

def evaluate_raw_residual_parts(
    model: Any,
    state_k: AcrobotReducedState,
    z: np.ndarray,
    u_k: float,
    h: float,
) -> Dict[str, float]:
    r_raw = model.reduced_step_residual(
        z=z,
        R1_k=state_k.R1,
        R2_k=state_k.R2,
        F1_prev=state_k.F1_prev,
        F2_prev=state_k.F2_prev,
        u_k=float(u_k),
        h=float(h),
    )

    r_raw = np.asarray(r_raw, dtype=float).reshape(8)

    return {
        "residual_inf_raw": inf_norm(r_raw),
        "residual_trans_inf": inf_norm(r_raw[0:4]),
        "residual_rot_inf": inf_norm(r_raw[4:6]),
        "residual_so2_inf": inf_norm(r_raw[6:8]),
    }


def evaluate_step(
    model: Any,
    state_k: AcrobotReducedState,
    state_next: AcrobotReducedState,
    z: np.ndarray,
    u_k: float,
    h: float,
    tf: float,
    method: str,
    case_id: str,
    local_step: int,
    time: float,
    root_success: bool,
    accepted_by_residual: bool,
    nfev: int,
    message: str,
    info: Any,
    multistart_meta: Mapping[str, Any],
) -> Dict[str, Any]:
    F1_k, F2_k, lam0, lam12 = model.unpack_reduced_solution(z)

    if bool(getattr(info, "substepping_performed", False)):
        info_residual = float(getattr(info, "residual_inf", math.nan))
        residual_parts = {
            "residual_inf_raw": info_residual,
            "residual_trans_inf": info_residual,
            "residual_rot_inf": info_residual,
            "residual_so2_inf": max(abs(det_error_so2(F1_k)), abs(det_error_so2(F2_k))),
        }
        F1_kinematic = state_k.R1.T @ state_next.R1
        F2_kinematic = state_k.R2.T @ state_next.R2
    else:
        residual_parts = evaluate_raw_residual_parts(
            model=model,
            state_k=state_k,
            z=z,
            u_k=u_k,
            h=h,
        )
        F1_kinematic = F1_k
        F2_kinematic = F2_k

    kin_R1_error = fro_norm(state_next.R1 - state_k.R1 @ F1_kinematic)
    kin_R2_error = fro_norm(state_next.R2 - state_k.R2 @ F2_kinematic)

    thetaR = model.angles_from_rotations(state_next.R1, state_next.R2)
    info_q1 = float(getattr(info, "q1", math.nan))
    info_q2 = float(getattr(info, "q2", math.nan))
    if method == "cayley" and np.isfinite(info_q1) and np.isfinite(info_q2):
        thetaF1 = angle_from_cayley(info_q1)
        thetaF2 = angle_from_cayley(info_q2)
    else:
        thetaF1 = angle_from_R(F1_k)
        thetaF2 = angle_from_R(F2_k)
    thetaF1_deg = float(rad2deg(thetaF1))
    thetaF2_deg = float(rad2deg(thetaF2))
    thetaF1_abs_deg = abs(thetaF1_deg)
    thetaF2_abs_deg = abs(thetaF2_deg)
    thetaF_max_abs_deg = max(thetaF1_abs_deg, thetaF2_abs_deg)
    thetaF1_net_deg = float(getattr(info, "thetaF1_net_deg", math.nan))
    thetaF2_net_deg = float(getattr(info, "thetaF2_net_deg", math.nan))
    q1_net = float(getattr(info, "q1_net", math.nan))
    q2_net = float(getattr(info, "q2_net", math.nan))

    selected_guess = multistart_meta.get("selected_guess")
    thetaF1_guess, thetaF2_guess = theta_guess_from_candidate(method, model, selected_guess)
    thetaF1_guess_deg = float(rad2deg(thetaF1_guess)) if np.isfinite(thetaF1_guess) else math.nan
    thetaF2_guess_deg = float(rad2deg(thetaF2_guess)) if np.isfinite(thetaF2_guess) else math.nan
    delta_thetaF1_from_guess_deg = (
        thetaF1_deg - thetaF1_guess_deg if np.isfinite(thetaF1_guess_deg) else math.nan
    )
    delta_thetaF2_from_guess_deg = (
        thetaF2_deg - thetaF2_guess_deg if np.isfinite(thetaF2_guess_deg) else math.nan
    )
    finite_deltas = [
        abs(x) for x in [delta_thetaF1_from_guess_deg, delta_thetaF2_from_guess_deg] if np.isfinite(x)
    ]
    delta_thetaF_max_from_guess_deg = max(finite_deltas) if finite_deltas else math.nan
    lambda0_norm = float(np.linalg.norm(lam0))
    lambda12_norm = float(np.linalg.norm(lam12))
    lambda_total_norm = lambda_total_norm_from_parts(lambda0_norm, lambda12_norm)

    X_next = model.reconstruct_positions_from_rotations(
        state_next.R1,
        state_next.R2,
    )

    try:
        holonomic_constraint_norm = float(
            model.constraint_norm(X_next, state_next.R1, state_next.R2)
        )
    except Exception:
        holonomic_constraint_norm = float("nan")

    return {
        "method": str(method),
        "case_id": case_id,
        "local_step": int(local_step),
        "time": float(time),
        "h": float(h),
        "tf": float(tf),
        "u": float(u_k),

        "root_success": bool(root_success),
        "accepted_by_residual": bool(accepted_by_residual),
        "nfev": int(nfev),
        "root_message": str(message),

        **residual_parts,

        "kin_R1_error": kin_R1_error,
        "kin_R2_error": kin_R2_error,

        "R1_orth_error": orth_error_so2(state_next.R1),
        "R2_orth_error": orth_error_so2(state_next.R2),
        "F1_orth_error": orth_error_so2(F1_k),
        "F2_orth_error": orth_error_so2(F2_k),

        "R1_det_error": det_error_so2(state_next.R1),
        "R2_det_error": det_error_so2(state_next.R2),
        "F1_det_error": det_error_so2(F1_k),
        "F2_det_error": det_error_so2(F2_k),

        "holonomic_constraint_norm": holonomic_constraint_norm,

        "thetaR1_deg": float(rad2deg(thetaR[0])),
        "thetaR2_deg": float(rad2deg(thetaR[1])),
        "thetaF1_deg": thetaF1_deg,
        "thetaF2_deg": thetaF2_deg,
        "thetaF1_abs_deg": thetaF1_abs_deg,
        "thetaF2_abs_deg": thetaF2_abs_deg,
        "thetaF_max_abs_deg": thetaF_max_abs_deg,
        "thetaF1_net_deg": thetaF1_net_deg,
        "thetaF2_net_deg": thetaF2_net_deg,
        "q1_net": q1_net,
        "q2_net": q2_net,
        "thetaF1_guess_deg": thetaF1_guess_deg,
        "thetaF2_guess_deg": thetaF2_guess_deg,
        "delta_thetaF1_from_guess_deg": delta_thetaF1_from_guess_deg,
        "delta_thetaF2_from_guess_deg": delta_thetaF2_from_guess_deg,
        "delta_thetaF_max_from_guess_deg": delta_thetaF_max_from_guess_deg,
        "large_step_warning_30deg": bool(thetaF_max_abs_deg > 30.0),
        "large_step_warning_60deg": bool(thetaF_max_abs_deg > 60.0),
        "large_step_warning_90deg": bool(thetaF_max_abs_deg > 90.0),

        "lambda0_x": float(lam0[0]),
        "lambda0_y": float(lam0[1]),
        "lambda12_x": float(lam12[0]),
        "lambda12_y": float(lam12[1]),
        "lambda0_norm": lambda0_norm,
        "lambda12_norm": lambda12_norm,
        "lambda_total_norm": lambda_total_norm,
        "lambda_spike_warning": bool(lambda_total_norm > LAMBDA_SPIKE_THRESHOLD),

        "substepping_performed": bool(getattr(info, "substepping_performed", False)),
        "substep_depth": int(getattr(info, "substep_depth", 0)),
        "num_substeps": int(getattr(info, "num_substeps", 1)),
        "h_original": float(getattr(info, "h_original", h)),
        "h_min_used": float(getattr(info, "h_min_used", h)),
        "cayley_near_singularity": bool(getattr(info, "cayley_near_singularity", False)),
        "near_singularity_count": int(getattr(info, "near_singularity_count", 0)),
        "cayley_distance_to_singularity_deg": (
            float(180.0 - thetaF_max_abs_deg) if method == "cayley" else math.nan
        ),
        "multistart_used": bool(multistart_meta.get("multistart_used", False)),
        "multistart_num_candidates": int(multistart_meta.get("multistart_num_candidates", 1)),
        "multistart_num_converged": int(multistart_meta.get("multistart_num_converged", 0)),
        "multistart_selected_index": int(multistart_meta.get("multistart_selected_index", -1)),
        "multistart_selected_score": float(multistart_meta.get("multistart_selected_score", math.nan)),
    }


# =============================================================================
# Rollout
# =============================================================================

def run_case(
    model: Any,
    h: float,
    tf: float,
    u_const: float,
    thetaR_deg: np.ndarray,
    thetaF_deg: np.ndarray,
    case_prefix: str,
    method: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    n_steps_float = tf / h
    n_steps = int(round(n_steps_float))

    if abs(n_steps_float - n_steps) > 1e-12:
        raise ValueError(f"tf/h must be integer. Got tf={tf}, h={h}, tf/h={n_steps_float}")

    case_id = (
        f"{case_prefix}_method={method}_u={u_const:+.3f}_h={h:.6f}_tf={tf:.3f}"
    )

    state = make_initial_state(
        model=model,
        h=h,
        thetaR_deg=thetaR_deg,
        thetaF_deg=thetaF_deg,
    )

    z_guess: Optional[np.ndarray] = None
    y_guess: Optional[np.ndarray] = None
    rows: List[Dict[str, Any]] = []
    failure: Optional[Dict[str, Any]] = None
    use_case_multistart = bool(USE_MULTISTART and case_prefix != "equilibrium")

    for k in range(n_steps):
        t_k = k * h
        if k == 0:
            print(f"Running {case_id} ...", flush=True)

        state_k = AcrobotReducedState(
            R1=state.R1.copy(),
            R2=state.R2.copy(),
            F1_prev=state.F1_prev.copy(),
            F2_prev=state.F2_prev.copy(),
        )

        try:
            state_next, info, z, multistart_meta, next_z_guess, next_y_guess = (
                solve_one_step_with_optional_multistart(
                    model=model,
                    h=float(h),
                    state=state,
                    u_const=float(u_const),
                    method=method,
                    z_guess=z_guess,
                    y_guess=y_guess,
                    use_multistart=use_case_multistart,
                )
            )
        except LGVISolveError as exc:
            theta_failure = model.angles_from_rotations(state.R1, state.R2)
            thetaF1_prev = angle_from_R(state.F1_prev)
            thetaF2_prev = angle_from_R(state.F2_prev)
            last_lambda0 = float(rows[-1]["lambda0_norm"]) if rows else math.nan
            last_lambda12 = float(rows[-1]["lambda12_norm"]) if rows else math.nan
            failure = {
                "failure_step": int(k),
                "failure_time": float(t_k),
                "failure_residual_inf": float(exc.residual_inf),
                "failure_nfev": int(exc.nfev),
                "failure_message": str(exc.solver_message),
                "failure_thetaR1_deg": float(rad2deg(theta_failure[0])),
                "failure_thetaR2_deg": float(rad2deg(theta_failure[1])),
                "failure_thetaF1_prev_deg": float(rad2deg(thetaF1_prev)),
                "failure_thetaF2_prev_deg": float(rad2deg(thetaF2_prev)),
                "failure_lambda0_norm_last": last_lambda0,
                "failure_lambda12_norm_last": last_lambda12,
            }
            break

        row = evaluate_step(
            model=model,
            state_k=state_k,
            state_next=state_next,
            z=z,
            u_k=float(u_const),
            h=float(h),
            tf=float(tf),
            method=method,
            case_id=case_id,
            local_step=k,
            time=(k + 1) * h,
            root_success=info.success,
            accepted_by_residual=info.accepted_by_residual,
            nfev=info.nfev,
            message=info.message,
            info=info,
            multistart_meta=multistart_meta,
        )
        rows.append(row)

        z_guess = next_z_guess
        y_guess = next_y_guess
        state = state_next

    summary = summarize_case(
        case_id=case_id,
        rows=rows,
        failure=failure,
        expected_steps=n_steps,
        case_prefix=case_prefix,
        method=method,
        h=float(h),
        tf=float(tf),
        u=float(u_const),
    )

    return rows, summary


# =============================================================================
# Summaries
# =============================================================================

def max_col(rows: List[Dict[str, Any]], name: str) -> float:
    return float(max(float(r[name]) for r in rows))


def max_abs_col(rows: List[Dict[str, Any]], name: str) -> float:
    return float(max(abs(float(r[name])) for r in rows))


def max_abs_finite_col(rows: List[Dict[str, Any]], name: str) -> float:
    values = [abs(float(r[name])) for r in rows if np.isfinite(float(r.get(name, math.nan)))]
    return float(max(values)) if values else math.nan


def min_finite_col(rows: List[Dict[str, Any]], name: str) -> float:
    values = [float(r[name]) for r in rows if np.isfinite(float(r.get(name, math.nan)))]
    return float(min(values)) if values else math.nan


def count_true(rows: List[Dict[str, Any]], name: str) -> int:
    return int(sum(bool(r.get(name, False)) for r in rows))


def summarize_case(
    case_id: str,
    rows: List[Dict[str, Any]],
    failure: Optional[Dict[str, Any]],
    expected_steps: int,
    case_prefix: str,
    method: str,
    h: float,
    tf: float,
    u: float,
) -> Dict[str, Any]:
    if not rows:
        return {
            "method": method,
            "case_id": case_id,
            "case_prefix": case_prefix,
            "u": float(u),
            "h": float(h),
            "tf": float(tf),
            "success": False,
            "status": "FAIL",
            "expected_steps": expected_steps,
            "num_steps_completed": 0,
            "accepted_by_residual_count": 0,
            "failure_step": None if failure is None else failure.get("failure_step"),
            "failure_time": None if failure is None else failure.get("failure_time"),
            "failure_residual_inf": None if failure is None else failure.get("failure_residual_inf"),
            "failure_message": None if failure is None else failure.get("failure_message"),
            "failure_thetaR1_deg": None if failure is None else failure.get("failure_thetaR1_deg"),
            "failure_thetaR2_deg": None if failure is None else failure.get("failure_thetaR2_deg"),
            "failure_thetaF1_prev_deg": None if failure is None else failure.get("failure_thetaF1_prev_deg"),
            "failure_thetaF2_prev_deg": None if failure is None else failure.get("failure_thetaF2_prev_deg"),
            "failure_lambda0_norm_last": None if failure is None else failure.get("failure_lambda0_norm_last"),
            "failure_lambda12_norm_last": None if failure is None else failure.get("failure_lambda12_norm_last"),
            "max_residual_inf_raw": math.nan,
            "max_residual_trans_inf": math.nan,
            "max_residual_rot_inf": math.nan,
            "max_residual_so2_inf": math.nan,
            "max_kin_R1_error": math.nan,
            "max_kin_R2_error": math.nan,
            "max_SO2_error": math.nan,
            "max_abs_thetaR1_deg": math.nan,
            "max_abs_thetaR2_deg": math.nan,
            "max_abs_thetaF1_deg": math.nan,
            "max_abs_thetaF2_deg": math.nan,
            "max_thetaF1_abs_deg": math.nan,
            "max_thetaF2_abs_deg": math.nan,
            "max_thetaF_abs_deg": math.nan,
            "max_delta_thetaF1_from_guess_deg": math.nan,
            "max_delta_thetaF2_from_guess_deg": math.nan,
            "max_delta_thetaF_from_guess_deg": math.nan,
            "num_large_step_warning_30deg": 0,
            "num_large_step_warning_60deg": 0,
            "num_large_step_warning_90deg": 0,
            "max_lambda0_norm": math.nan,
            "max_lambda12_norm": math.nan,
            "max_lambda_total_norm": math.nan,
            "num_lambda_spike_warnings": 0,
            "final_thetaR1_deg": math.nan,
            "final_thetaR2_deg": math.nan,
            "substepping_case": False,
            "total_substeps": 0,
            "max_substep_depth": 0,
            "min_h_used": math.nan,
            "near_singularity_case": False,
            "near_singularity_count": 0,
            "min_cayley_distance_to_singularity_deg": math.nan,
        }

    max_so2 = max(
        max_col(rows, "R1_orth_error"),
        max_col(rows, "R2_orth_error"),
        max_col(rows, "F1_orth_error"),
        max_col(rows, "F2_orth_error"),
        max_col(rows, "R1_det_error"),
        max_col(rows, "R2_det_error"),
        max_col(rows, "F1_det_error"),
        max_col(rows, "F2_det_error"),
    )

    accepted_count = sum(bool(r["accepted_by_residual"]) for r in rows)
    all_steps_completed = len(rows) == expected_steps and failure is None

    strict_dynamics_ok = (
        all_steps_completed
        and max_col(rows, "residual_inf_raw") <= WARN_RESIDUAL_INF
        and max_col(rows, "kin_R1_error") <= WARN_KINEMATIC
        and max_col(rows, "kin_R2_error") <= WARN_KINEMATIC
        and max_so2 <= WARN_SO2
    )

    if not all_steps_completed:
        status = "FAIL"
    elif accepted_count > 0 and strict_dynamics_ok:
        status = "PASS_BY_RESIDUAL"
    elif strict_dynamics_ok:
        status = "PASS_ROOT_SUCCESS"
    else:
        status = "CHECK"

    last = rows[-1]
    substepping_case = any(bool(r.get("substepping_performed", False)) for r in rows)
    total_substeps = sum(int(r.get("num_substeps", 1)) for r in rows)
    max_substep_depth = max(int(r.get("substep_depth", 0)) for r in rows)
    h_values = [float(r.get("h_min_used", r.get("h", h))) for r in rows]
    near_singularity_count = sum(int(r.get("near_singularity_count", 0)) for r in rows)

    return {
        "method": method,
        "case_id": case_id,
        "case_prefix": case_prefix,
        "success": bool(strict_dynamics_ok),
        "status": status,
        "expected_steps": expected_steps,
        "num_steps_completed": len(rows),
        "accepted_by_residual_count": int(accepted_count),

        "failure_step": None if failure is None else failure.get("failure_step"),
        "failure_time": None if failure is None else failure.get("failure_time"),
        "failure_residual_inf": None if failure is None else failure.get("failure_residual_inf"),
        "failure_message": None if failure is None else failure.get("failure_message"),
        "failure_thetaR1_deg": None if failure is None else failure.get("failure_thetaR1_deg"),
        "failure_thetaR2_deg": None if failure is None else failure.get("failure_thetaR2_deg"),
        "failure_thetaF1_prev_deg": None if failure is None else failure.get("failure_thetaF1_prev_deg"),
        "failure_thetaF2_prev_deg": None if failure is None else failure.get("failure_thetaF2_prev_deg"),
        "failure_lambda0_norm_last": None if failure is None else failure.get("failure_lambda0_norm_last"),
        "failure_lambda12_norm_last": None if failure is None else failure.get("failure_lambda12_norm_last"),

        "u": float(last["u"]),
        "h": float(last["h"]),
        "tf": float(tf),
        "final_time": float(last["time"]),

        "max_residual_inf_raw": max_col(rows, "residual_inf_raw"),
        "max_residual_trans_inf": max_col(rows, "residual_trans_inf"),
        "max_residual_rot_inf": max_col(rows, "residual_rot_inf"),
        "max_residual_so2_inf": max_col(rows, "residual_so2_inf"),

        "max_kin_R1_error": max_col(rows, "kin_R1_error"),
        "max_kin_R2_error": max_col(rows, "kin_R2_error"),
        "max_SO2_error": max_so2,

        "max_holonomic_constraint_norm": max_col(rows, "holonomic_constraint_norm"),

        "max_lambda0_norm": max_col(rows, "lambda0_norm"),
        "max_lambda12_norm": max_col(rows, "lambda12_norm"),
        "max_lambda_total_norm": max_col(rows, "lambda_total_norm"),
        "num_lambda_spike_warnings": count_true(rows, "lambda_spike_warning"),

        "max_abs_thetaR1_deg": max_abs_col(rows, "thetaR1_deg"),
        "max_abs_thetaR2_deg": max_abs_col(rows, "thetaR2_deg"),
        "max_abs_thetaF1_deg": max_abs_col(rows, "thetaF1_deg"),
        "max_abs_thetaF2_deg": max_abs_col(rows, "thetaF2_deg"),
        "max_thetaF1_abs_deg": max_col(rows, "thetaF1_abs_deg"),
        "max_thetaF2_abs_deg": max_col(rows, "thetaF2_abs_deg"),
        "max_thetaF_abs_deg": max_col(rows, "thetaF_max_abs_deg"),
        "max_delta_thetaF1_from_guess_deg": max_abs_finite_col(rows, "delta_thetaF1_from_guess_deg"),
        "max_delta_thetaF2_from_guess_deg": max_abs_finite_col(rows, "delta_thetaF2_from_guess_deg"),
        "max_delta_thetaF_from_guess_deg": max_abs_finite_col(rows, "delta_thetaF_max_from_guess_deg"),
        "num_large_step_warning_30deg": count_true(rows, "large_step_warning_30deg"),
        "num_large_step_warning_60deg": count_true(rows, "large_step_warning_60deg"),
        "num_large_step_warning_90deg": count_true(rows, "large_step_warning_90deg"),

        "final_thetaR1_deg": float(last["thetaR1_deg"]),
        "final_thetaR2_deg": float(last["thetaR2_deg"]),

        "substepping_case": bool(substepping_case),
        "total_substeps": int(total_substeps),
        "max_substep_depth": int(max_substep_depth),
        "min_h_used": float(min(h_values)) if h_values else math.nan,
        "near_singularity_case": bool(near_singularity_count > 0),
        "near_singularity_count": int(near_singularity_count),
        "min_cayley_distance_to_singularity_deg": min_finite_col(rows, "cayley_distance_to_singularity_deg"),
    }


def add_equilibrium_specific_flags(summary: Dict[str, Any]) -> Dict[str, Any]:
    if summary.get("case_prefix") != "equilibrium":
        return summary

    if not summary.get("success", False):
        summary["equilibrium_ok"] = False
        return summary

    angle_ok = (
        abs(float(summary["final_thetaR1_deg"])) <= WARN_EQUILIBRIUM_ANGLE_DEG
        and abs(float(summary["final_thetaR2_deg"])) <= WARN_EQUILIBRIUM_ANGLE_DEG
        and float(summary["max_abs_thetaR1_deg"]) <= WARN_EQUILIBRIUM_ANGLE_DEG
        and float(summary["max_abs_thetaR2_deg"]) <= WARN_EQUILIBRIUM_ANGLE_DEG
    )

    step_ok = (
        float(summary["max_abs_thetaF1_deg"]) <= WARN_EQUILIBRIUM_STEP_DEG
        and float(summary["max_abs_thetaF2_deg"]) <= WARN_EQUILIBRIUM_STEP_DEG
    )

    summary["equilibrium_angle_ok"] = bool(angle_ok)
    summary["equilibrium_step_ok"] = bool(step_ok)
    summary["equilibrium_ok"] = bool(angle_ok and step_ok)

    if not summary["equilibrium_ok"]:
        summary["status"] = "CHECK_EQUILIBRIUM_DRIFT"
        summary["success"] = False

    return summary


def compute_convergence_summary(case_summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_method_u: Dict[Tuple[str, float], List[Dict[str, Any]]] = {}

    for s in case_summaries:
        if s.get("case_prefix") != "grid":
            continue
        if not s.get("success", False):
            continue
        by_method_u.setdefault((str(s["method"]), float(s["u"])), []).append(s)

    rows: List[Dict[str, Any]] = []

    for (method, u), summaries in by_method_u.items():
        summaries_sorted = sorted(summaries, key=lambda x: float(x["h"]))
        reference = summaries_sorted[0]  # smallest h

        ref_theta = np.array(
            [
                float(reference["final_thetaR1_deg"]),
                float(reference["final_thetaR2_deg"]),
            ],
            dtype=float,
        )

        for s in summaries_sorted:
            theta = np.array(
                [
                    float(s["final_thetaR1_deg"]),
                    float(s["final_thetaR2_deg"]),
                ],
                dtype=float,
            )
            err = theta - ref_theta

            rows.append(
                {
                    "u": float(u),
                    "method": method,
                    "h": float(s["h"]),
                    "reference_h": float(reference["h"]),
                    "final_thetaR1_deg": float(theta[0]),
                    "final_thetaR2_deg": float(theta[1]),
                    "error_vs_finest_thetaR1_deg": float(err[0]),
                    "error_vs_finest_thetaR2_deg": float(err[1]),
                    "error_vs_finest_norm_deg": float(np.linalg.norm(err)),
                    "case_id": s["case_id"],
                    "reference_case_id": reference["case_id"],
                }
            )

    return rows


def compute_method_comparison(case_summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, float, float, float], Dict[str, Dict[str, Any]]] = {}

    for summary in case_summaries:
        key = (
            str(summary.get("case_prefix", "")),
            float(summary.get("u", math.nan)),
            float(summary.get("h", math.nan)),
            float(summary.get("tf", summary.get("final_time", math.nan))),
        )
        grouped.setdefault(key, {})[str(summary.get("method"))] = summary

    rows: List[Dict[str, Any]] = []
    for (case_prefix, u, h, tf), by_method in sorted(grouped.items()):
        if "ab" not in by_method or "cayley" not in by_method:
            continue

        ab = by_method["ab"]
        cayley = by_method["cayley"]
        ab_theta = np.array(
            [float(ab.get("final_thetaR1_deg", math.nan)), float(ab.get("final_thetaR2_deg", math.nan))],
            dtype=float,
        )
        cayley_theta = np.array(
            [float(cayley.get("final_thetaR1_deg", math.nan)), float(cayley.get("final_thetaR2_deg", math.nan))],
            dtype=float,
        )
        ab_completed = (
            int(ab.get("num_steps_completed", 0)) == int(ab.get("expected_steps", -1))
            and str(ab.get("status")) != "FAIL"
        )
        cayley_completed = (
            int(cayley.get("num_steps_completed", 0)) == int(cayley.get("expected_steps", -1))
            and str(cayley.get("status")) != "FAIL"
        )
        both_completed = bool(ab_completed and cayley_completed)
        comparison_valid = both_completed
        cayley_substepping_case = bool(cayley.get("substepping_case", False))
        both_completed_and_no_substepping = bool(both_completed and not cayley_substepping_case)
        diff = cayley_theta - ab_theta if comparison_valid else np.array([math.nan, math.nan], dtype=float)

        rows.append(
            {
                "case_prefix": case_prefix,
                "u": u,
                "h": h,
                "tf": tf,
                "ab_status": ab.get("status"),
                "cayley_status": cayley.get("status"),
                "ab_success": bool(ab.get("success", False)),
                "cayley_success": bool(cayley.get("success", False)),
                "ab_num_steps_completed": int(ab.get("num_steps_completed", 0)),
                "ab_expected_steps": int(ab.get("expected_steps", 0)),
                "cayley_num_steps_completed": int(cayley.get("num_steps_completed", 0)),
                "cayley_expected_steps": int(cayley.get("expected_steps", 0)),
                "ab_completed": bool(ab_completed),
                "cayley_completed": bool(cayley_completed),
                "both_completed": bool(both_completed),
                "comparison_valid": bool(comparison_valid),
                "both_completed_and_no_substepping": bool(both_completed_and_no_substepping),
                "ab_final_thetaR1_deg": float(ab_theta[0]),
                "ab_final_thetaR2_deg": float(ab_theta[1]),
                "cayley_final_thetaR1_deg": float(cayley_theta[0]),
                "cayley_final_thetaR2_deg": float(cayley_theta[1]),
                "diff_final_thetaR1_deg": float(diff[0]),
                "diff_final_thetaR2_deg": float(diff[1]),
                "diff_final_thetaR_norm_deg": float(np.linalg.norm(diff)),
                "ab_max_residual_inf": float(ab.get("max_residual_inf_raw", math.nan)),
                "cayley_max_residual_inf": float(cayley.get("max_residual_inf_raw", math.nan)),
                "ab_max_abs_thetaF1_deg": float(ab.get("max_abs_thetaF1_deg", math.nan)),
                "ab_max_abs_thetaF2_deg": float(ab.get("max_abs_thetaF2_deg", math.nan)),
                "cayley_max_abs_thetaF1_deg": float(cayley.get("max_abs_thetaF1_deg", math.nan)),
                "cayley_max_abs_thetaF2_deg": float(cayley.get("max_abs_thetaF2_deg", math.nan)),
                "cayley_substepping_case": cayley_substepping_case,
                "cayley_total_substeps": int(cayley.get("total_substeps", 0)),
                "cayley_near_singularity_case": bool(cayley.get("near_singularity_case", False)),
                "cayley_near_singularity_count": int(cayley.get("near_singularity_count", 0)),
                "cayley_min_distance_to_singularity_deg": float(cayley.get("min_cayley_distance_to_singularity_deg", math.nan)),
                "ab_case_id": ab.get("case_id"),
                "cayley_case_id": cayley.get("case_id"),
            }
        )

    return rows


def print_summary_line(summary: Dict[str, Any]) -> None:
    status = summary["status"]

    print(
        f"[{status}] method={summary.get('method')} {summary['case_id']} | "
        f"steps={summary['num_steps_completed']}/{summary['expected_steps']} | "
        f"max r_inf={summary['max_residual_inf_raw']:.3e} | "
        f"max kin={max(summary['max_kin_R1_error'], summary['max_kin_R2_error']):.3e} | "
        f"max SO2={summary['max_SO2_error']:.3e} | "
        f"max |thetaR|=({summary['max_abs_thetaR1_deg']:.3e}, "
        f"{summary['max_abs_thetaR2_deg']:.3e}) deg | "
        f"max |thetaF|=({summary['max_abs_thetaF1_deg']:.3e}, "
        f"{summary['max_abs_thetaF2_deg']:.3e}) deg"
    )

    if summary["status"] == "FAIL":
        print(
            f"       failure_step={summary.get('failure_step')}, "
            f"failure_residual={summary.get('failure_residual_inf')}, "
            f"message={summary.get('failure_message')}"
        )


def print_method_console_summary(
    case_summaries: List[Dict[str, Any]],
    comparison_rows: List[Dict[str, Any]],
) -> None:
    ab_summaries = [s for s in case_summaries if s.get("method") == "ab"]
    cayley_summaries = [s for s in case_summaries if s.get("method") == "cayley"]

    ab_successes = sum(bool(s.get("success", False)) for s in ab_summaries)
    cayley_successes = sum(bool(s.get("success", False)) for s in cayley_summaries)
    ab_failures = sum(s.get("status") == "FAIL" for s in ab_summaries)
    cayley_failures = sum(s.get("status") == "FAIL" for s in cayley_summaries)
    cayley_substep_cases = [s for s in cayley_summaries if bool(s.get("substepping_case", False))]
    cayley_near_cases = [s for s in cayley_summaries if bool(s.get("near_singularity_case", False))]
    valid_comparisons = [r for r in comparison_rows if bool(r.get("comparison_valid", False))]
    invalid_comparisons = [r for r in comparison_rows if not bool(r.get("comparison_valid", False))]
    valid_with_substepping = [r for r in valid_comparisons if bool(r.get("cayley_substepping_case", False))]
    valid_without_substepping = [
        r for r in valid_comparisons if bool(r.get("both_completed_and_no_substepping", False))
    ]

    largest_diff = max(
        (float(r.get("diff_final_thetaR_norm_deg", math.nan)) for r in valid_comparisons),
        default=math.nan,
    )
    largest_diff_no_substep = max(
        (float(r.get("diff_final_thetaR_norm_deg", math.nan)) for r in valid_without_substepping),
        default=math.nan,
    )
    worst_ab = max((float(s.get("max_residual_inf_raw", math.nan)) for s in ab_summaries), default=math.nan)
    worst_cayley = max((float(s.get("max_residual_inf_raw", math.nan)) for s in cayley_summaries), default=math.nan)
    cayley_distance_values = [
        float(s.get("min_cayley_distance_to_singularity_deg", math.nan))
        for s in cayley_summaries
        if np.isfinite(float(s.get("min_cayley_distance_to_singularity_deg", math.nan)))
    ]
    min_cayley_distance = min(cayley_distance_values) if cayley_distance_values else math.nan

    print("\n" + "=" * 100)
    print("AB-vs-Cayley summary")
    print("=" * 100)
    print(f"AB successes: {ab_successes}")
    print(f"Cayley successes: {cayley_successes}")
    print(f"AB failures: {ab_failures}")
    print(f"Cayley failures: {cayley_failures}")
    print(f"Valid AB-vs-Cayley comparisons: {len(valid_comparisons)}")
    print(f"Invalid comparisons: {len(invalid_comparisons)}")
    print(f"Valid comparisons with Cayley substepping: {len(valid_with_substepping)}")
    print(f"Valid comparisons without Cayley substepping: {len(valid_without_substepping)}")
    print(f"Cayley cases with substepping: {len(cayley_substep_cases)}")
    print(f"Cayley cases near singularity: {len(cayley_near_cases)}")
    print(f"Largest valid AB-vs-Cayley final angle difference: {largest_diff:.6e} deg")
    print(
        "Largest valid AB-vs-Cayley final angle difference without Cayley substepping: "
        f"{largest_diff_no_substep:.6e} deg"
    )
    print(f"Minimum Cayley distance to singularity: {min_cayley_distance:.6e} deg")
    print(f"Worst residual AB: {worst_ab:.6e}")
    print(f"Worst residual Cayley: {worst_cayley:.6e}")

    def print_cases(title: str, rows: List[Any]) -> None:
        print(f"\n{title}")
        if not rows:
            print("  none")
            return
        for row in rows[:20]:
            if isinstance(row, dict) and "case_id" in row:
                print(f"  {row['case_id']}")
            elif isinstance(row, dict):
                print(f"  {row.get('ab_case_id')} | {row.get('cayley_case_id')}")
        if len(rows) > 20:
            print(f"  ... {len(rows) - 20} more")

    print_cases(
        "AB failed but Cayley succeeded",
        [r for r in comparison_rows if not bool(r["ab_completed"]) and bool(r["cayley_completed"])],
    )
    print_cases(
        "Cayley failed but AB succeeded",
        [r for r in comparison_rows if bool(r["ab_completed"]) and not bool(r["cayley_completed"])],
    )
    print_cases("Cayley needed substepping", cayley_substep_cases)
    print_cases("Cayley got near singularity", cayley_near_cases)
    print_cases(
        "Cases with thetaF > 30 deg",
        [s for s in case_summaries if int(s.get("num_large_step_warning_30deg", 0)) > 0],
    )
    print_cases(
        "Cases with thetaF > 60 deg",
        [s for s in case_summaries if int(s.get("num_large_step_warning_60deg", 0)) > 0],
    )
    print_cases(
        "Cases with thetaF > 90 deg",
        [s for s in case_summaries if int(s.get("num_large_step_warning_90deg", 0)) > 0],
    )
    print_cases(
        "Cases with lambda spike warning",
        [s for s in case_summaries if int(s.get("num_lambda_spike_warnings", 0)) > 0],
    )
    print_cases(
        "AB and Cayley final states differ by more than 1 degree, only among valid comparisons",
        [r for r in valid_comparisons if float(r["diff_final_thetaR_norm_deg"]) > 1.0],
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = make_model_from_params(PARAMS)

    all_rows: List[Dict[str, Any]] = []
    case_summaries: List[Dict[str, Any]] = []

    print("=" * 100)
    print("Reduced SO(2) Acrobot LGVI numerical simulation validation")
    print("=" * 100)
    print("Physical params:", PARAMS["physical"])
    print("Root settings:")
    print(f"  ROOT_TOL = {ROOT_TOL}")
    print(f"  LGVI_MAXFEV = {LGVI_MAXFEV}")
    print(f"  ACCEPT_RESIDUAL = {ACCEPT_RESIDUAL}")
    print(f"  ACCEPT_RESIDUAL_TOL = {ACCEPT_RESIDUAL_TOL}")
    print("=" * 100)

    # -------------------------------------------------------------------------
    # 1. Equilibrium tests
    # -------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("1) Equilibrium tests: thetaR=[0,0], thetaF=[0,0], u=0")
    print("=" * 100)

    equilibrium_summaries: List[Dict[str, Any]] = []

    for method in METHODS:
        for h in DT_VALUES:
            rows, summary = run_case(
                model=model,
                h=float(h),
                tf=float(EQUILIBRIUM_TF),
                u_const=0.0,
                thetaR_deg=EQUILIBRIUM_THETAR_DEG,
                thetaF_deg=EQUILIBRIUM_THETAF_DEG,
                case_prefix="equilibrium",
                method=method,
            )
            summary = add_equilibrium_specific_flags(summary)

            all_rows.extend(rows)
            case_summaries.append(summary)
            equilibrium_summaries.append(summary)
            print_summary_line(summary)

    # -------------------------------------------------------------------------
    # 2. General control/dt grid
    # -------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("2) General control and dt grid")
    print("=" * 100)
    print(f"Initial thetaR [deg]: {INITIAL_THETAR_DEG.tolist()}")
    print(f"Initial thetaF [deg]: {INITIAL_THETAF_DEG.tolist()}")
    print(f"Controls: {CONTROL_VALUES}")
    print(f"dt values: {DT_VALUES}")
    print(f"tf: {GRID_TF}")
    print("=" * 100)

    grid_summaries: List[Dict[str, Any]] = []

    for method in METHODS:
        for u_const in CONTROL_VALUES:
            for h in DT_VALUES:
                rows, summary = run_case(
                    model=model,
                    h=float(h),
                    tf=float(GRID_TF),
                    u_const=float(u_const),
                    thetaR_deg=INITIAL_THETAR_DEG,
                    thetaF_deg=INITIAL_THETAF_DEG,
                    case_prefix="grid",
                    method=method,
                )

                all_rows.extend(rows)
                case_summaries.append(summary)
                grid_summaries.append(summary)
                print_summary_line(summary)

    convergence_rows = compute_convergence_summary(case_summaries)
    comparison_rows = compute_method_comparison(case_summaries)

    # -------------------------------------------------------------------------
    # Write files
    # -------------------------------------------------------------------------
    step_csv = OUT_DIR / "lgvi_step_validation.csv"
    case_csv = OUT_DIR / "lgvi_case_summary.csv"
    eq_csv = OUT_DIR / "lgvi_equilibrium_summary.csv"
    conv_csv = OUT_DIR / "lgvi_convergence_summary.csv"
    comparison_csv = OUT_DIR / "lgvi_method_comparison.csv"
    json_path = OUT_DIR / "lgvi_validation_report.json"

    write_csv(step_csv, all_rows)
    write_csv(case_csv, case_summaries)
    write_csv(eq_csv, equilibrium_summaries)
    write_csv(conv_csv, convergence_rows)
    write_csv(comparison_csv, comparison_rows)

    report = {
        "settings": {
            "params": PARAMS,
            "equilibrium_thetaR_deg": EQUILIBRIUM_THETAR_DEG.tolist(),
            "equilibrium_thetaF_deg": EQUILIBRIUM_THETAF_DEG.tolist(),
            "grid_initial_thetaR_deg": INITIAL_THETAR_DEG.tolist(),
            "grid_initial_thetaF_deg": INITIAL_THETAF_DEG.tolist(),
            "control_values": CONTROL_VALUES,
            "dt_values": DT_VALUES,
            "grid_tf": GRID_TF,
            "equilibrium_tf": EQUILIBRIUM_TF,
            "root_tol": ROOT_TOL,
            "lgvi_maxfev": LGVI_MAXFEV,
            "accept_residual": ACCEPT_RESIDUAL,
            "accept_residual_tol": ACCEPT_RESIDUAL_TOL,
            "use_multistart": USE_MULTISTART,
            "multistart_select": MULTISTART_SELECT,
            "methods": METHODS,
            "cayley_allow_substepping": CAYLEY_ALLOW_SUBSTEPPING,
            "cayley_min_substep_h": CAYLEY_MIN_SUBSTEP_H,
            "cayley_max_subdivisions": CAYLEY_MAX_SUBDIVISIONS,
            "cayley_singularity_margin_deg": CAYLEY_SINGULARITY_MARGIN_DEG,
        },
        "thresholds": {
            "warn_residual_inf": WARN_RESIDUAL_INF,
            "warn_kinematic": WARN_KINEMATIC,
            "warn_so2": WARN_SO2,
            "warn_equilibrium_angle_deg": WARN_EQUILIBRIUM_ANGLE_DEG,
            "warn_equilibrium_step_deg": WARN_EQUILIBRIUM_STEP_DEG,
            "lambda_spike_threshold": LAMBDA_SPIKE_THRESHOLD,
        },
        "equilibrium": {
            "num_cases": len(equilibrium_summaries),
            "num_ok": sum(bool(s.get("equilibrium_ok", False)) for s in equilibrium_summaries),
            "all_ok": all(bool(s.get("equilibrium_ok", False)) for s in equilibrium_summaries),
        },
        "grid": {
            "num_cases": len(grid_summaries),
            "num_success": sum(bool(s.get("success", False)) for s in grid_summaries),
            "num_fail": sum(s.get("status") == "FAIL" for s in grid_summaries),
            "num_check": sum(s.get("status") == "CHECK" for s in grid_summaries),
            "num_pass_by_residual": sum(s.get("status") == "PASS_BY_RESIDUAL" for s in grid_summaries),
            "num_pass_root_success": sum(s.get("status") == "PASS_ROOT_SUCCESS" for s in grid_summaries),
        },
        "method_comparison": {
            "num_pairs": len(comparison_rows),
            "num_valid_pairs": sum(bool(r.get("comparison_valid", False)) for r in comparison_rows),
            "num_invalid_pairs": sum(not bool(r.get("comparison_valid", False)) for r in comparison_rows),
            "largest_valid_final_angle_difference_deg": max(
                (
                    float(r.get("diff_final_thetaR_norm_deg", math.nan))
                    for r in comparison_rows
                    if bool(r.get("comparison_valid", False))
                ),
                default=math.nan,
            ),
            "largest_valid_no_substepping_final_angle_difference_deg": max(
                (
                    float(r.get("diff_final_thetaR_norm_deg", math.nan))
                    for r in comparison_rows
                    if bool(r.get("both_completed_and_no_substepping", False))
                ),
                default=math.nan,
            ),
        },
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=True)

    # -------------------------------------------------------------------------
    # Final console summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("Files written")
    print("=" * 100)
    print(step_csv)
    print(case_csv)
    print(eq_csv)
    print(conv_csv)
    print(comparison_csv)
    print(json_path)

    print_method_console_summary(case_summaries, comparison_rows)

    print("\n" + "=" * 100)
    print("How to interpret")
    print("=" * 100)
    print(
        "Equilibrium test:\n"
        "  The hanging-down rest state with u=0 should remain at rest.\n"
        "  If equilibrium_ok is false, check gravity/sign/convention immediately.\n\n"
        "Grid test:\n"
        "  max_residual_inf_raw checks whether Newton solved the LGVI residual.\n"
        "  max_kin_R1/R2 checks R_{k+1} = R_k F_k.\n"
        "  max_SO2_error checks whether R and F remain valid SO(2) rotations.\n"
        "  PASS_BY_RESIDUAL is mathematically fine: SciPy did not like the solve,\n"
        "  but the residual was below ACCEPT_RESIDUAL_TOL.\n\n"
        "Convergence test:\n"
        "  For each u, the finest dt is used as internal reference.\n"
        "  Larger dt values are compared against that final angle result.\n"
        "  This checks discretization behavior, not Newton correctness."
    )


if __name__ == "__main__":
    main()
