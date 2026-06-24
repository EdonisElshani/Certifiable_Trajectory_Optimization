"""
Scaffold for benchmarking reduced LGVI against a high-accuracy continuous ODE.

This file intentionally does not invent continuous Acrobot equations.  When the
Numerical_Simulation_SPOT_MPC model exposes a suitable continuous RHS, this
script can compare AB/Cayley LGVI final absolute angles against solve_ivp.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy.integrate import solve_ivp

from solver_lgvi_acrobot import (
    LGVISolveError,
    make_model_from_params,
    make_reduced_state_from_absolute,
    rollout_lgvi_controls,
)


OUT_DIR = Path("lgvi_test_results")
OUT_CSV = OUT_DIR / "ode_benchmark_summary.csv"

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

METHODS = ["ab", "cayley"]
CONTROL_VALUES = [-20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0]
H_VALUES = [0.01, 0.005, 0.002, 0.001, 0.0005]
TF = 0.1

INITIAL_THETAR_DEG = np.array([5.0, 5.0], dtype=float)
INITIAL_THETAF_DEG = np.array([0.0, 0.0], dtype=float)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "u",
        "h",
        "method",
        "final_thetaR1_lgvi_deg",
        "final_thetaR2_lgvi_deg",
        "final_thetaR1_ode_deg",
        "final_thetaR2_ode_deg",
        "error_thetaR1_deg",
        "error_thetaR2_deg",
        "error_norm_deg",
        "success",
        "message",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def solve_continuous_reference(model: Any, u_const: float, tf: float) -> np.ndarray:
    """
    Return final absolute angles from a high-accuracy continuous ODE solve.

    TODO:
        Implement this once the SPOT-MPC model exposes exact continuous Acrobot
        dynamics, for example `rhs_absolute(t, y, u_fun)` with state
        [thetaR1, thetaR2, omega1, omega2].
    """
    if not hasattr(model, "rhs_absolute"):
        raise NotImplementedError(
            "Continuous ODE benchmark is not available yet: "
            "AcrobotSO2Model has no rhs_absolute method in Numerical_Simulation_SPOT_MPC."
        )

    y0 = np.array(
        [
            *np.deg2rad(INITIAL_THETAR_DEG),
            *np.deg2rad(INITIAL_THETAF_DEG),
        ],
        dtype=float,
    )

    def u_fun(_t: float) -> float:
        return float(u_const)

    sol = solve_ivp(
        lambda t, y: model.rhs_absolute(t, y, u_fun),
        (0.0, float(tf)),
        y0,
        method="DOP853",
        rtol=1e-11,
        atol=1e-13,
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    return np.asarray(sol.y[:2, -1], dtype=float)


def run_lgvi_case(model: Any, method: str, u_const: float, h: float, tf: float) -> np.ndarray:
    n_steps = int(round(float(tf) / float(h)))
    initial_state = make_reduced_state_from_absolute(
        model=model,
        h=float(h),
        thetaR=np.deg2rad(INITIAL_THETAR_DEG),
        thetaF=np.deg2rad(INITIAL_THETAF_DEG),
    )
    sim = rollout_lgvi_controls(
        model=model,
        h=float(h),
        initial_state=initial_state,
        u_sequence=np.full(n_steps, float(u_const), dtype=float),
        method=method,
        allow_substepping=(method == "cayley"),
        root_tol=1e-12,
        lgvi_maxfev=5000,
        accept_residual=True,
        accept_residual_tol=1e-10,
        use_multistart=True,
        multistart_select="local",
    )
    return np.asarray(sim["thetaR"][-1], dtype=float)


def main() -> None:
    model = make_model_from_params(PARAMS)
    rows: List[Dict[str, Any]] = []

    for u_const in CONTROL_VALUES:
        try:
            theta_ode = solve_continuous_reference(model, u_const=u_const, tf=TF)
            ode_message = ""
        except NotImplementedError as exc:
            theta_ode = np.array([math.nan, math.nan], dtype=float)
            ode_message = str(exc)
        except Exception as exc:
            theta_ode = np.array([math.nan, math.nan], dtype=float)
            ode_message = f"ODE solve failed: {exc}"

        for h in H_VALUES:
            for method in METHODS:
                try:
                    theta_lgvi = run_lgvi_case(model, method=method, u_const=u_const, h=h, tf=TF)
                    err = np.rad2deg(theta_lgvi - theta_ode)
                    success = bool(np.all(np.isfinite(theta_ode)))
                    message = ode_message
                except LGVISolveError as exc:
                    theta_lgvi = np.array([math.nan, math.nan], dtype=float)
                    err = np.array([math.nan, math.nan], dtype=float)
                    success = False
                    message = f"LGVI failed: {exc.solver_message}"

                rows.append(
                    {
                        "u": float(u_const),
                        "h": float(h),
                        "method": method,
                        "final_thetaR1_lgvi_deg": float(np.rad2deg(theta_lgvi[0])),
                        "final_thetaR2_lgvi_deg": float(np.rad2deg(theta_lgvi[1])),
                        "final_thetaR1_ode_deg": float(np.rad2deg(theta_ode[0])),
                        "final_thetaR2_ode_deg": float(np.rad2deg(theta_ode[1])),
                        "error_thetaR1_deg": float(err[0]),
                        "error_thetaR2_deg": float(err[1]),
                        "error_norm_deg": float(np.linalg.norm(err)) if np.all(np.isfinite(err)) else math.nan,
                        "success": bool(success),
                        "message": message,
                    }
                )

    write_csv(OUT_CSV, rows)
    print(f"Wrote {OUT_CSV}")
    if rows and rows[0].get("message"):
        print(rows[0]["message"])


if __name__ == "__main__":
    main()
