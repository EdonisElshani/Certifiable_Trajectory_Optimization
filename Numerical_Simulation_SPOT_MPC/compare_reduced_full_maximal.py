"""
compare_reduced_full_maximal.py

Open-loop comparison between:
  1) the reduced SDP-compatible SO(2) LGVI simulator, and
  2) the Jan-Brüdigam-style full maximal-coordinate planar VI benchmark.

Run from Numerical_Simulation_SPOT_MPC/:
    python compare_reduced_full_maximal.py

The output CSV tells you whether the reduced prediction model behaves similarly
or differently from the higher-fidelity constrained maximal-coordinate plant.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from lie_group_so2 import angle_diff_vec_deg
from solver_lgvi_acrobot import (
    make_model_from_params,
    make_reduced_state_from_absolute,
    rollout_lgvi_controls,
    LGVISolveError,
)
from full_maximal_vi_acrobot import (
    make_full_state_from_angles,
    rollout_full_maximal_vi_controls,
)

OUT_DIR = Path("lgvi_test_results")
OUT_CSV = OUT_DIR / "reduced_vs_full_maximal_comparison.csv"

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

INITIAL_THETAR_DEG = np.array([5.0, 5.0], dtype=float)
INITIAL_THETAF_DEG = np.array([0.0, 0.0], dtype=float)

CONTROL_VALUES = [-20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0]
H_VALUES = [0.005, 0.002, 0.001]
TF = 0.1


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_one(model: Any, u_const: float, h: float) -> Dict[str, Any]:
    n_steps_float = TF / h
    n_steps = int(round(n_steps_float))
    if abs(n_steps_float - n_steps) > 1e-12:
        raise ValueError(f"TF/h must be integer. Got TF={TF}, h={h}")

    thetaR0 = np.deg2rad(INITIAL_THETAR_DEG)
    thetaF0 = np.deg2rad(INITIAL_THETAF_DEG)
    u_sequence = np.full(n_steps, float(u_const), dtype=float)

    row: Dict[str, Any] = {
        "u": float(u_const),
        "h": float(h),
        "tf": float(TF),
        "n_steps": int(n_steps),
    }

    try:
        reduced_state = make_reduced_state_from_absolute(
            model=model,
            h=float(h),
            thetaR=thetaR0,
            thetaF=thetaF0,
        )
        reduced_sim = rollout_lgvi_controls(
            model=model,
            h=float(h),
            initial_state=reduced_state,
            u_sequence=u_sequence,
            method="cayley",
            allow_substepping=True,
            root_tol=1e-12,
            lgvi_maxfev=5000,
            accept_residual=True,
            accept_residual_tol=1e-10,
            use_multistart=True,
            multistart_select="local",
            root_solver="damped_newton",
        )
        reduced_ok = True
        reduced_msg = "ok"
        theta_reduced_deg = np.rad2deg(np.asarray(reduced_sim["thetaR"][-1], dtype=float))
    except LGVISolveError as exc:
        reduced_ok = False
        reduced_msg = f"reduced failed: {exc.solver_message}; residual={exc.residual_inf:.3e}"
        theta_reduced_deg = np.array([math.nan, math.nan], dtype=float)
        reduced_sim = None

    try:
        full_state = make_full_state_from_angles(
            model=model,
            h=float(h),
            thetaR=thetaR0,
            thetaF=thetaF0,
        )
        full_sim = rollout_full_maximal_vi_controls(
            model=model,
            h=float(h),
            initial_state=full_state,
            u_sequence=u_sequence,
            tol=1e-10,
            max_iter=30,
            accept_residual=True,
            accept_residual_tol=1e-8,
            torque_mode="elbow",
        )
        full_ok = bool(np.all(full_sim.get("success", np.array([True], dtype=bool))))
        full_msg = "ok" if full_ok else "full maximal had a non-strict step"
        theta_full_deg = np.rad2deg(np.asarray(full_sim["thetaR"][-1], dtype=float))
    except Exception as exc:
        full_ok = False
        full_msg = f"full failed: {exc}"
        theta_full_deg = np.array([math.nan, math.nan], dtype=float)
        full_sim = None

    if reduced_ok and full_ok:
        diff = angle_diff_vec_deg(theta_full_deg, theta_reduced_deg)
        diff_norm = float(np.linalg.norm(diff))
    else:
        diff = np.array([math.nan, math.nan], dtype=float)
        diff_norm = math.nan

    row.update(
        {
            "reduced_success": bool(reduced_ok),
            "full_success": bool(full_ok),
            "reduced_message": reduced_msg,
            "full_message": full_msg,
            "reduced_final_thetaR1_deg": float(theta_reduced_deg[0]),
            "reduced_final_thetaR2_deg": float(theta_reduced_deg[1]),
            "full_final_thetaR1_deg": float(theta_full_deg[0]),
            "full_final_thetaR2_deg": float(theta_full_deg[1]),
            "full_minus_reduced_thetaR1_deg": float(diff[0]),
            "full_minus_reduced_thetaR2_deg": float(diff[1]),
            "full_minus_reduced_norm_deg": diff_norm,
        }
    )

    if reduced_sim is not None:
        row.update(
            {
                "reduced_max_residual_inf": float(np.nanmax(reduced_sim["residual_inf"])),
                "reduced_max_lambda_total_norm": float(np.nanmax(reduced_sim["lambda_total_norm"])),
                "reduced_max_h_lambda_total_norm": float(np.nanmax(reduced_sim["h_lambda_total_norm"])),
                "reduced_max_h2_lambda_total_norm": float(np.nanmax(reduced_sim["h2_lambda_total_norm"])),
                "reduced_total_line_search_failures": int(np.nansum(reduced_sim["line_search_failures"])),
                "reduced_max_newton_iterations": int(np.nanmax(reduced_sim["newton_iterations"])),
            }
        )
    return row


def main() -> None:
    model = make_model_from_params(PARAMS)
    rows: List[Dict[str, Any]] = []
    for u in CONTROL_VALUES:
        for h in H_VALUES:
            print(f"running u={u:+.1f}, h={h:g} ...", flush=True)
            rows.append(run_one(model, u_const=float(u), h=float(h)))
    write_csv(OUT_CSV, rows)
    print(f"wrote {OUT_CSV}")

    valid = [r for r in rows if r["reduced_success"] and r["full_success"] and math.isfinite(r["full_minus_reduced_norm_deg"])]
    if valid:
        worst = max(valid, key=lambda r: r["full_minus_reduced_norm_deg"])
        print("worst valid reduced-vs-full mismatch:")
        print(worst)


if __name__ == "__main__":
    main()
