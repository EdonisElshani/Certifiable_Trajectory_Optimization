"""
test_full_maximal_lgvi_dynamics_grid.py

Validation grid for the Jan/Brüdigam-style full maximal-coordinate planar
Acrobot LGVI benchmark.

Purpose
-------
This script tests the full maximal-coordinate variational integrator on the
same constant-input and time-step grid used for the reduced SO(2) LGVI test.
It is intentionally separate from test_lgvi_dynamics_grid.py, because this
model is a plant/benchmark model, not the SDP-reduced prediction model.

It writes:
    lgvi_test_results/full_maximal_step_validation.csv
    lgvi_test_results/full_maximal_case_summary.csv
    lgvi_test_results/full_maximal_equilibrium_summary.csv
    lgvi_test_results/full_maximal_validation_report.json

Run from Numerical_Simulation_SPOT_MPC:
    python test_full_maximal_lgvi_dynamics_grid.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from .solver_lgvi_acrobot import make_model_from_params
    from .full_maximal_vi_acrobot import (
        FullMaximalSolveError,
        make_full_state_from_angles,
        step_full_maximal_vi,
    )
    from .lie_group_so2 import angle_from_R, angle_diff_deg, orth_error_so2, det_error_so2
except ImportError:
    from solver_lgvi_acrobot import make_model_from_params
    from full_maximal_vi_acrobot import (
        FullMaximalSolveError,
        make_full_state_from_angles,
        step_full_maximal_vi,
    )
    from lie_group_so2 import angle_from_R, angle_diff_deg, orth_error_so2, det_error_so2


# -----------------------------------------------------------------------------
# Test configuration
# -----------------------------------------------------------------------------

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

# Same grid as the reduced validation script. Because apparently comparing apples
# with some other fruit was not enough trouble already.
CONTROL_VALUES = [-20.0, -15.0, -10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0]
H_VALUES = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
TF_GRID = 0.1

EQUILIBRIUM_H_VALUES = H_VALUES
EQUILIBRIUM_TF = 1.0

INITIAL_THETAR_DEG = np.array([5.0, 5.0], dtype=float)
INITIAL_THETAF_DEG = np.array([0.0, 0.0], dtype=float)

EQUILIBRIUM_THETAR_DEG = np.array([0.0, 0.0], dtype=float)
EQUILIBRIUM_THETAF_DEG = np.array([0.0, 0.0], dtype=float)

# Full maximal solver settings.
FULL_TOL = 1e-10
FULL_MAX_ITER = 30
ACCEPT_RESIDUAL = True
ACCEPT_RESIDUAL_TOL = 1e-8
TORQUE_MODE = "elbow"  # must match reduced convention tau1=-u, tau2=+u

# Diagnostics thresholds.
THETAF_WARN_DEG = 30.0
THETAF_BAD_DEG = 60.0
THETAF_CRITICAL_DEG = 90.0
LAMBDA_SPIKE_THRESHOLD = 1.0e5
H_LAMBDA_SPIKE_THRESHOLD = 1.0e2
H2_LAMBDA_SPIKE_THRESHOLD = 1.0
CONSTRAINT_WARN_THRESHOLD = 1.0e-8
SO2_WARN_THRESHOLD = 1.0e-8
KIN_WARN_THRESHOLD = 1.0e-8

OUTPUT_DIR = Path("lgvi_test_results")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _case_id(kind: str, u: float, h: float, tf: float) -> str:
    return f"{kind}_fullmax_u={u:+.3f}_h={h:.6f}_tf={tf:.3f}"


def _safe_max(values: List[float], default: float = float("nan")) -> float:
    finite = [float(v) for v in values if np.isfinite(v)]
    return max(finite) if finite else float(default)


def _safe_min(values: List[float], default: float = float("nan")) -> float:
    finite = [float(v) for v in values if np.isfinite(v)]
    return min(finite) if finite else float(default)


def _constraint_norms(model: Any, x1: np.ndarray, x2: np.ndarray, R1: np.ndarray, R2: np.ndarray) -> Tuple[float, float, float]:
    phi0 = np.asarray(x1, dtype=float).reshape(2) + R1 @ model.rho10 - model.p0
    phi12 = np.asarray(x1, dtype=float).reshape(2) + R1 @ model.rho112 - np.asarray(x2, dtype=float).reshape(2) - R2 @ model.rho212
    n0 = float(np.linalg.norm(phi0, ord=np.inf))
    n12 = float(np.linalg.norm(phi12, ord=np.inf))
    return n0, n12, max(n0, n12)


def _so2_error(*mats: np.ndarray) -> float:
    vals: List[float] = []
    for M in mats:
        vals.append(abs(float(orth_error_so2(M))))
        vals.append(abs(float(det_error_so2(M))))
    return _safe_max(vals, default=0.0)


def _abs_theta_deg(theta_rad: np.ndarray) -> Tuple[float, float]:
    theta_deg = np.rad2deg(np.asarray(theta_rad, dtype=float).reshape(2))
    return abs(float(theta_deg[0])), abs(float(theta_deg[1]))


def _status_from_infos(success: bool, infos: List[Any]) -> str:
    if not success:
        return "FAIL"
    if any(bool(info.accepted_by_residual) for info in infos):
        return "PASS_BY_RESIDUAL"
    return "PASS_ROOT_SUCCESS"


def _print_case_summary(row: Dict[str, Any]) -> None:
    status = str(row["status"])
    case_id = str(row["case_id"])
    print(
        f"[{status}] full_maximal {case_id} | steps={row['steps_completed']}/{row['expected_steps']} "
        f"| max r_inf={row['max_residual_inf']:.3e} "
        f"| max constraint={row['max_constraint_inf']:.3e} "
        f"| max kin={row['max_kin_error']:.3e} "
        f"| max SO2={row['max_SO2_error']:.3e} "
        f"| max |thetaR|=({row['max_abs_thetaR1_deg']:.3e}, {row['max_abs_thetaR2_deg']:.3e}) deg "
        f"| max |thetaF|=({row['max_abs_thetaF1_deg']:.3e}, {row['max_abs_thetaF2_deg']:.3e}) deg"
    )
    if not row["success"]:
        print(
            f"       failure_step={row['failure_step']}, "
            f"failure_residual={row['failure_residual']}, "
            f"message={row['failure_message']}"
        )


def run_full_maximal_case(
    model: Any,
    kind: str,
    u_value: float,
    h: float,
    tf: float,
    thetaR0_deg: np.ndarray,
    thetaF0_deg: np.ndarray,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    n_steps = int(round(float(tf) / float(h)))
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    tf_effective = float(n_steps) * float(h)
    case_id = _case_id(kind, float(u_value), float(h), tf_effective)

    state = make_full_state_from_angles(
        model=model,
        h=float(h),
        thetaR=np.deg2rad(thetaR0_deg),
        thetaF=np.deg2rad(thetaF0_deg),
    )
    z_guess: Optional[np.ndarray] = None

    infos: List[Any] = []
    step_rows: List[Dict[str, Any]] = []
    success = True
    failure_step = -1
    failure_residual = float("nan")
    failure_message = ""

    # Initial diagnostics.
    thetaR_initial = np.rad2deg(model.angles_from_rotations(state.R1, state.R2))
    thetaF_initial = np.rad2deg(np.array([angle_from_R(state.F1_prev), angle_from_R(state.F2_prev)]))

    for k in range(n_steps):
        try:
            state_prev = state
            state_next, info, z = step_full_maximal_vi(
                model=model,
                state=state,
                u_k=float(u_value),
                h=float(h),
                z_guess=z_guess,
                tol=FULL_TOL,
                max_iter=FULL_MAX_ITER,
                accept_residual=ACCEPT_RESIDUAL,
                accept_residual_tol=ACCEPT_RESIDUAL_TOL,
                torque_mode=TORQUE_MODE,
            )
            infos.append(info)
            z_guess = z.copy()
            state = state_next

            thetaR = np.rad2deg(model.angles_from_rotations(state.R1, state.R2))
            thetaF = np.rad2deg(np.array([angle_from_R(state.F1_prev), angle_from_R(state.F2_prev)]))
            phi0_norm, phi12_norm, phi_norm = _constraint_norms(model, state.x1, state.x2, state.R1, state.R2)
            kin1 = float(np.linalg.norm(state.R1 - state_prev.R1 @ state.F1_prev, ord="fro"))
            kin2 = float(np.linalg.norm(state.R2 - state_prev.R2 @ state.F2_prev, ord="fro"))
            kin_error = max(kin1, kin2)
            so2_err = _so2_error(state.R1, state.R2, state.F1_prev, state.F2_prev)

            line_search_damped = bool(np.isfinite(info.min_alpha) and info.min_alpha < 0.999999)
            step_rows.append(
                {
                    "case_id": case_id,
                    "kind": kind,
                    "k": k,
                    "t": (k + 1) * float(h),
                    "u": float(u_value),
                    "h": float(h),
                    "tf": tf_effective,
                    "success": bool(info.success),
                    "accepted_by_residual": bool(info.accepted_by_residual),
                    "residual_inf": float(info.residual_inf),
                    "n_iter": int(info.n_iter),
                    "nfev": int(info.nfev),
                    "message": str(info.message),
                    "line_search_damped": line_search_damped,
                    "min_alpha": float(info.min_alpha),
                    "final_alpha": float(info.final_alpha),
                    "thetaR1_deg": float(thetaR[0]),
                    "thetaR2_deg": float(thetaR[1]),
                    "thetaF1_deg": float(thetaF[0]),
                    "thetaF2_deg": float(thetaF[1]),
                    "abs_thetaF_max_deg": float(max(abs(thetaF[0]), abs(thetaF[1]))),
                    "lambda0_norm": float(info.lambda0_norm),
                    "lambda12_norm": float(info.lambda12_norm),
                    "lambda_total_norm": float(info.lambda_total_norm),
                    "h_lambda_total_norm": float(h * info.lambda_total_norm),
                    "h2_lambda_total_norm": float((h**2) * info.lambda_total_norm),
                    "constraint_phi0_inf": phi0_norm,
                    "constraint_phi12_inf": phi12_norm,
                    "constraint_inf": phi_norm,
                    "kin_R1_error": kin1,
                    "kin_R2_error": kin2,
                    "kin_error": kin_error,
                    "SO2_error": so2_err,
                    "thetaF_gt_30_deg": bool(max(abs(thetaF[0]), abs(thetaF[1])) > THETAF_WARN_DEG),
                    "thetaF_gt_60_deg": bool(max(abs(thetaF[0]), abs(thetaF[1])) > THETAF_BAD_DEG),
                    "thetaF_gt_90_deg": bool(max(abs(thetaF[0]), abs(thetaF[1])) > THETAF_CRITICAL_DEG),
                    "lambda_spike_warning": bool(info.lambda_total_norm > LAMBDA_SPIKE_THRESHOLD),
                    "h_lambda_spike_warning": bool(h * info.lambda_total_norm > H_LAMBDA_SPIKE_THRESHOLD),
                    "h2_lambda_spike_warning": bool((h**2) * info.lambda_total_norm > H2_LAMBDA_SPIKE_THRESHOLD),
                    "constraint_warning": bool(phi_norm > CONSTRAINT_WARN_THRESHOLD),
                    "SO2_warning": bool(so2_err > SO2_WARN_THRESHOLD),
                    "kin_warning": bool(kin_error > KIN_WARN_THRESHOLD),
                }
            )
        except FullMaximalSolveError as exc:
            success = False
            failure_step = k
            failure_residual = float(exc.residual_inf)
            failure_message = str(exc.solver_message)
            step_rows.append(
                {
                    "case_id": case_id,
                    "kind": kind,
                    "k": k,
                    "t": (k + 1) * float(h),
                    "u": float(u_value),
                    "h": float(h),
                    "tf": tf_effective,
                    "success": False,
                    "accepted_by_residual": False,
                    "residual_inf": failure_residual,
                    "n_iter": np.nan,
                    "nfev": np.nan,
                    "message": failure_message,
                    "line_search_damped": False,
                    "min_alpha": np.nan,
                    "final_alpha": np.nan,
                    "thetaR1_deg": np.nan,
                    "thetaR2_deg": np.nan,
                    "thetaF1_deg": np.nan,
                    "thetaF2_deg": np.nan,
                    "abs_thetaF_max_deg": np.nan,
                    "lambda0_norm": np.nan,
                    "lambda12_norm": np.nan,
                    "lambda_total_norm": np.nan,
                    "h_lambda_total_norm": np.nan,
                    "h2_lambda_total_norm": np.nan,
                    "constraint_phi0_inf": np.nan,
                    "constraint_phi12_inf": np.nan,
                    "constraint_inf": np.nan,
                    "kin_R1_error": np.nan,
                    "kin_R2_error": np.nan,
                    "kin_error": np.nan,
                    "SO2_error": np.nan,
                    "thetaF_gt_30_deg": False,
                    "thetaF_gt_60_deg": False,
                    "thetaF_gt_90_deg": False,
                    "lambda_spike_warning": False,
                    "h_lambda_spike_warning": False,
                    "h2_lambda_spike_warning": False,
                    "constraint_warning": False,
                    "SO2_warning": False,
                    "kin_warning": False,
                }
            )
            break

    steps_completed = len(infos)
    final_thetaR = np.rad2deg(model.angles_from_rotations(state.R1, state.R2))
    final_thetaF = np.rad2deg(np.array([angle_from_R(state.F1_prev), angle_from_R(state.F2_prev)]))

    vals = step_rows
    status = _status_from_infos(success and steps_completed == n_steps, infos)
    summary = {
        "case_id": case_id,
        "kind": kind,
        "method": "full_maximal_vi",
        "torque_mode": TORQUE_MODE,
        "u": float(u_value),
        "h": float(h),
        "tf": tf_effective,
        "expected_steps": n_steps,
        "steps_completed": steps_completed,
        "success": bool(success and steps_completed == n_steps),
        "status": status,
        "any_accepted_by_residual": bool(any(info.accepted_by_residual for info in infos)),
        "failure_step": failure_step,
        "failure_residual": failure_residual,
        "failure_message": failure_message,
        "max_residual_inf": _safe_max([r.get("residual_inf", np.nan) for r in vals]),
        "max_constraint_inf": _safe_max([r.get("constraint_inf", np.nan) for r in vals]),
        "max_kin_error": _safe_max([r.get("kin_error", np.nan) for r in vals]),
        "max_SO2_error": _safe_max([r.get("SO2_error", np.nan) for r in vals]),
        "max_abs_thetaR1_deg": _safe_max([abs(r.get("thetaR1_deg", np.nan)) for r in vals] + [abs(thetaR_initial[0])]),
        "max_abs_thetaR2_deg": _safe_max([abs(r.get("thetaR2_deg", np.nan)) for r in vals] + [abs(thetaR_initial[1])]),
        "max_abs_thetaF1_deg": _safe_max([abs(r.get("thetaF1_deg", np.nan)) for r in vals] + [abs(thetaF_initial[0])]),
        "max_abs_thetaF2_deg": _safe_max([abs(r.get("thetaF2_deg", np.nan)) for r in vals] + [abs(thetaF_initial[1])]),
        "max_abs_thetaF_deg": _safe_max([r.get("abs_thetaF_max_deg", np.nan) for r in vals]),
        "max_lambda_total_norm": _safe_max([r.get("lambda_total_norm", np.nan) for r in vals]),
        "max_h_lambda_total_norm": _safe_max([r.get("h_lambda_total_norm", np.nan) for r in vals]),
        "max_h2_lambda_total_norm": _safe_max([r.get("h2_lambda_total_norm", np.nan) for r in vals]),
        "max_newton_iterations": _safe_max([r.get("n_iter", np.nan) for r in vals]),
        "max_nfev": _safe_max([r.get("nfev", np.nan) for r in vals]),
        "min_alpha_used": _safe_min([r.get("min_alpha", np.nan) for r in vals]),
        "num_damped_steps": int(sum(bool(r.get("line_search_damped", False)) for r in vals)),
        "num_thetaF_gt_30": int(sum(bool(r.get("thetaF_gt_30_deg", False)) for r in vals)),
        "num_thetaF_gt_60": int(sum(bool(r.get("thetaF_gt_60_deg", False)) for r in vals)),
        "num_thetaF_gt_90": int(sum(bool(r.get("thetaF_gt_90_deg", False)) for r in vals)),
        "lambda_spike_warning": bool(any(bool(r.get("lambda_spike_warning", False)) for r in vals)),
        "h_lambda_spike_warning": bool(any(bool(r.get("h_lambda_spike_warning", False)) for r in vals)),
        "h2_lambda_spike_warning": bool(any(bool(r.get("h2_lambda_spike_warning", False)) for r in vals)),
        "constraint_warning": bool(any(bool(r.get("constraint_warning", False)) for r in vals)),
        "SO2_warning": bool(any(bool(r.get("SO2_warning", False)) for r in vals)),
        "kin_warning": bool(any(bool(r.get("kin_warning", False)) for r in vals)),
        "initial_thetaR1_deg": float(thetaR_initial[0]),
        "initial_thetaR2_deg": float(thetaR_initial[1]),
        "final_thetaR1_deg": float(final_thetaR[0]),
        "final_thetaR2_deg": float(final_thetaR[1]),
        "final_thetaF1_deg": float(final_thetaF[0]),
        "final_thetaF2_deg": float(final_thetaF[1]),
    }
    return summary, step_rows


def build_report(case_summary: pd.DataFrame, equilibrium_summary: pd.DataFrame) -> Dict[str, Any]:
    grid = case_summary[case_summary["kind"] == "grid"].copy()
    eq = equilibrium_summary.copy()

    def case_list(mask: pd.Series) -> List[str]:
        return [str(x) for x in case_summary.loc[mask, "case_id"].tolist()]

    report: Dict[str, Any] = {
        "physical_params": PARAMS["physical"],
        "solver_settings": {
            "FULL_TOL": FULL_TOL,
            "FULL_MAX_ITER": FULL_MAX_ITER,
            "ACCEPT_RESIDUAL": ACCEPT_RESIDUAL,
            "ACCEPT_RESIDUAL_TOL": ACCEPT_RESIDUAL_TOL,
            "TORQUE_MODE": TORQUE_MODE,
        },
        "grid": {
            "num_cases": int(len(grid)),
            "num_successes": int(grid["success"].sum()) if len(grid) else 0,
            "num_failures": int((~grid["success"]).sum()) if len(grid) else 0,
            "success_rate": float(grid["success"].mean()) if len(grid) else float("nan"),
            "failed_cases": [str(x) for x in grid.loc[~grid["success"], "case_id"].tolist()],
            "cases_thetaF_gt_30": [str(x) for x in grid.loc[grid["num_thetaF_gt_30"] > 0, "case_id"].tolist()],
            "cases_thetaF_gt_60": [str(x) for x in grid.loc[grid["num_thetaF_gt_60"] > 0, "case_id"].tolist()],
            "cases_thetaF_gt_90": [str(x) for x in grid.loc[grid["num_thetaF_gt_90"] > 0, "case_id"].tolist()],
            "cases_lambda_spike": [str(x) for x in grid.loc[grid["lambda_spike_warning"], "case_id"].tolist()],
            "cases_h_lambda_spike": [str(x) for x in grid.loc[grid["h_lambda_spike_warning"], "case_id"].tolist()],
            "cases_h2_lambda_spike": [str(x) for x in grid.loc[grid["h2_lambda_spike_warning"], "case_id"].tolist()],
            "cases_constraint_warning": [str(x) for x in grid.loc[grid["constraint_warning"], "case_id"].tolist()],
            "cases_SO2_warning": [str(x) for x in grid.loc[grid["SO2_warning"], "case_id"].tolist()],
            "cases_kin_warning": [str(x) for x in grid.loc[grid["kin_warning"], "case_id"].tolist()],
            "worst_residual_inf": float(grid["max_residual_inf"].max()) if len(grid) else float("nan"),
            "worst_constraint_inf": float(grid["max_constraint_inf"].max()) if len(grid) else float("nan"),
            "worst_SO2_error": float(grid["max_SO2_error"].max()) if len(grid) else float("nan"),
            "worst_kin_error": float(grid["max_kin_error"].max()) if len(grid) else float("nan"),
            "worst_lambda_total_norm": float(grid["max_lambda_total_norm"].max()) if len(grid) else float("nan"),
            "worst_h_lambda_total_norm": float(grid["max_h_lambda_total_norm"].max()) if len(grid) else float("nan"),
            "worst_h2_lambda_total_norm": float(grid["max_h2_lambda_total_norm"].max()) if len(grid) else float("nan"),
        },
        "equilibrium": {
            "num_cases": int(len(eq)),
            "num_successes": int(eq["success"].sum()) if len(eq) else 0,
            "num_failures": int((~eq["success"]).sum()) if len(eq) else 0,
            "failed_cases": [str(x) for x in eq.loc[~eq["success"], "case_id"].tolist()] if len(eq) else [],
            "worst_residual_inf": float(eq["max_residual_inf"].max()) if len(eq) else float("nan"),
            "worst_constraint_inf": float(eq["max_constraint_inf"].max()) if len(eq) else float("nan"),
        },
    }
    return report


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    print("\n" + "=" * 100)
    print("Full maximal-coordinate Acrobot LGVI numerical simulation validation")
    print("=" * 100)
    print(f"Physical params: {PARAMS['physical']}")
    print("Solver settings:")
    print(f"  FULL_TOL = {FULL_TOL}")
    print(f"  FULL_MAX_ITER = {FULL_MAX_ITER}")
    print(f"  ACCEPT_RESIDUAL = {ACCEPT_RESIDUAL}")
    print(f"  ACCEPT_RESIDUAL_TOL = {ACCEPT_RESIDUAL_TOL}")
    print(f"  TORQUE_MODE = {TORQUE_MODE}")
    print("=" * 100)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = make_model_from_params(PARAMS)

    step_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    equilibrium_rows: List[Dict[str, Any]] = []

    print("\n" + "=" * 100)
    print("1) Full maximal equilibrium tests: thetaR=[0,0], thetaF=[0,0], u=0")
    print("=" * 100)
    for h in EQUILIBRIUM_H_VALUES:
        case_id = _case_id("equilibrium", 0.0, float(h), EQUILIBRIUM_TF)
        print(f"Running {case_id} ...")
        summary, rows = run_full_maximal_case(
            model=model,
            kind="equilibrium",
            u_value=0.0,
            h=float(h),
            tf=EQUILIBRIUM_TF,
            thetaR0_deg=EQUILIBRIUM_THETAR_DEG,
            thetaF0_deg=EQUILIBRIUM_THETAF_DEG,
        )
        _print_case_summary(summary)
        step_rows.extend(rows)
        summary_rows.append(summary)
        equilibrium_rows.append(summary.copy())

    print("\n" + "=" * 100)
    print("2) Full maximal general control and dt grid")
    print("=" * 100)
    print(f"Initial thetaR [deg]: {INITIAL_THETAR_DEG.tolist()}")
    print(f"Initial thetaF [deg]: {INITIAL_THETAF_DEG.tolist()}")
    print(f"Controls: {CONTROL_VALUES}")
    print(f"dt values: {H_VALUES}")
    print(f"tf: {TF_GRID}")
    print("=" * 100)

    for u_value in CONTROL_VALUES:
        for h in H_VALUES:
            case_id = _case_id("grid", float(u_value), float(h), TF_GRID)
            print(f"Running {case_id} ...")
            summary, rows = run_full_maximal_case(
                model=model,
                kind="grid",
                u_value=float(u_value),
                h=float(h),
                tf=TF_GRID,
                thetaR0_deg=INITIAL_THETAR_DEG,
                thetaF0_deg=INITIAL_THETAF_DEG,
            )
            _print_case_summary(summary)
            step_rows.extend(rows)
            summary_rows.append(summary)

    step_df = pd.DataFrame(step_rows)
    summary_df = pd.DataFrame(summary_rows)
    eq_df = pd.DataFrame(equilibrium_rows)
    report = build_report(summary_df, eq_df)

    step_path = OUTPUT_DIR / "full_maximal_step_validation.csv"
    summary_path = OUTPUT_DIR / "full_maximal_case_summary.csv"
    equilibrium_path = OUTPUT_DIR / "full_maximal_equilibrium_summary.csv"
    report_path = OUTPUT_DIR / "full_maximal_validation_report.json"

    step_df.to_csv(step_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    eq_df.to_csv(equilibrium_path, index=False)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 100)
    print("Files written")
    print("=" * 100)
    print(step_path)
    print(summary_path)
    print(equilibrium_path)
    print(report_path)

    grid = summary_df[summary_df["kind"] == "grid"]
    print("\n" + "=" * 100)
    print("Full maximal summary")
    print("=" * 100)
    print(f"Grid successes: {int(grid['success'].sum())}/{len(grid)}")
    print(f"Grid failures: {int((~grid['success']).sum())}/{len(grid)}")
    print(f"Equilibrium successes: {int(eq_df['success'].sum())}/{len(eq_df)}")
    print(f"Worst grid residual: {float(grid['max_residual_inf'].max()):.6e}")
    print(f"Worst grid constraint error: {float(grid['max_constraint_inf'].max()):.6e}")
    print(f"Worst grid SO2 error: {float(grid['max_SO2_error'].max()):.6e}")
    print(f"Worst grid kinematic error: {float(grid['max_kin_error'].max()):.6e}")
    print(f"Worst grid lambda norm: {float(grid['max_lambda_total_norm'].max()):.6e}")
    print(f"Worst grid h*lambda norm: {float(grid['max_h_lambda_total_norm'].max()):.6e}")
    print(f"Worst grid h^2*lambda norm: {float(grid['max_h2_lambda_total_norm'].max()):.6e}")

    failed = grid.loc[~grid["success"], "case_id"].tolist()
    print("\nFailed grid cases:")
    if failed:
        for c in failed:
            print(f"  {c}")
    else:
        print("  none")

    print("\nCases with thetaF > 30 deg:")
    cases = grid.loc[grid["num_thetaF_gt_30"] > 0, "case_id"].tolist()
    if cases:
        for c in cases[:30]:
            print(f"  {c}")
        if len(cases) > 30:
            print(f"  ... {len(cases) - 30} more")
    else:
        print("  none")

    print("\nCases with h^2*lambda spike warning:")
    cases = grid.loc[grid["h2_lambda_spike_warning"], "case_id"].tolist()
    if cases:
        for c in cases:
            print(f"  {c}")
    else:
        print("  none")

    print("\n" + "=" * 100)
    print("How to interpret")
    print("=" * 100)
    print("This tests the full maximal-coordinate VI only, not the SDP-reduced model.")
    print("If this passes where the reduced model fails, the reduced model/solver is the fragile part.")
    print("If this fails too, the input/time step is hard for the plant benchmark as well.")
    print("Large raw lambda is less meaningful than h*lambda and h^2*lambda.")


if __name__ == "__main__":
    main()
