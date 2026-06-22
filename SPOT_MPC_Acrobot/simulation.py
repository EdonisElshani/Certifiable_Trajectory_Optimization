from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from config.config_loader import load_yaml_config, build_common_params
from Numerical_Simulation.lie_group_so2 import angle_from_R, orth_error_so2, det_error_so2
from Numerical_Simulation.solver_lgvi_acrobot import (
    AcrobotReducedState,
    LGVISolveError,
    make_model_from_params,
    make_initial_state_from_params,
    simulate_one_control_interval_from_params,
    convert_state_to_sdp_initial_scalars,
    diagnostics_lgvi,
)


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = PROJECT_ROOT / "Results"
SIM_RESULTS_DIR = RESULTS_ROOT / "Results-Open_Loop-Simulation"
MPC_RESULTS_DIR = RESULTS_ROOT / "Results-MPC"


# -----------------------------------------------------------------------------
# JSON / CSV helpers
# -----------------------------------------------------------------------------


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items() if not callable(v)}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return str(obj)


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            vals = []
            for key in keys:
                val = row.get(key, "")
                if isinstance(val, float):
                    vals.append(f"{val:.16e}")
                else:
                    vals.append(str(val))
            f.write(",".join(vals) + "\n")


# -----------------------------------------------------------------------------
# Geometry and error helpers
# -----------------------------------------------------------------------------


def acrobot_points_from_angles(theta1: float, theta2: float, params: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full-link points using your absolute-angle convention:
        x = l sin(theta), y = -l cos(theta)
    """
    p0 = np.asarray(params.get("p_0", params.get("p0", [0.0, 0.0])), dtype=float).reshape(2)
    l1 = float(params["l1"])
    l2 = float(params["l2"])

    p1 = p0 + np.array([l1 * np.sin(theta1), -l1 * np.cos(theta1)], dtype=float)
    p2 = p1 + np.array([l2 * np.sin(theta2), -l2 * np.cos(theta2)], dtype=float)
    return p0, p1, p2


def acrobot_points_from_state(state: AcrobotReducedState, params: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return acrobot_points_from_angles(
        theta1=float(angle_from_R(state.R1)),
        theta2=float(angle_from_R(state.R2)),
        params=params,
    )


def state_to_row(
    state: AcrobotReducedState,
    params: Mapping[str, Any],
    time_value: float,
    k: int,
    u_value: float,
) -> Dict[str, float]:
    thetaR1 = float(angle_from_R(state.R1))
    thetaR2 = float(angle_from_R(state.R2))
    thetaF1 = float(angle_from_R(state.F1_prev))
    thetaF2 = float(angle_from_R(state.F2_prev))
    p0, p1, p2 = acrobot_points_from_angles(thetaR1, thetaR2, params)

    return {
        "k": float(k),
        "t": float(time_value),
        "u": float(u_value),
        "thetaR1_rad": thetaR1,
        "thetaR2_rad": thetaR2,
        "thetaR1_deg": float(np.rad2deg(thetaR1)),
        "thetaR2_deg": float(np.rad2deg(thetaR2)),
        "thetaF1_rad": thetaF1,
        "thetaF2_rad": thetaF2,
        "thetaF1_deg": float(np.rad2deg(thetaF1)),
        "thetaF2_deg": float(np.rad2deg(thetaF2)),
        "R1_orth_error": float(orth_error_so2(state.R1)),
        "R2_orth_error": float(orth_error_so2(state.R2)),
        "F1_orth_error": float(orth_error_so2(state.F1_prev)),
        "F2_orth_error": float(orth_error_so2(state.F2_prev)),
        "R1_det_error": float(det_error_so2(state.R1)),
        "R2_det_error": float(det_error_so2(state.R2)),
        "F1_det_error": float(det_error_so2(state.F1_prev)),
        "F2_det_error": float(det_error_so2(state.F2_prev)),
        "base_x": float(p0[0]),
        "base_y": float(p0[1]),
        "elbow_x": float(p1[0]),
        "elbow_y": float(p1[1]),
        "tip_x": float(p2[0]),
        "tip_y": float(p2[1]),
    }


def target_errors_from_row(row: Mapping[str, float], params: Mapping[str, Any]) -> Dict[str, float]:
    thetaR1_des = float(params["thetaR1_des"])
    thetaR2_des = float(params["thetaR2_des"])
    thetaF1_des = float(params.get("thetaF1_des", 0.0))
    thetaF2_des = float(params.get("thetaF2_des", 0.0))

    def wrap(x: float) -> float:
        return float((x + np.pi) % (2.0 * np.pi) - np.pi)

    eR1 = wrap(float(row["thetaR1_rad"]) - thetaR1_des)
    eR2 = wrap(float(row["thetaR2_rad"]) - thetaR2_des)
    eF1 = wrap(float(row["thetaF1_rad"]) - thetaF1_des)
    eF2 = wrap(float(row["thetaF2_rad"]) - thetaF2_des)

    return {
        "target_angle_error_norm_rad": float(np.sqrt(eR1 * eR1 + eR2 * eR2)),
        "target_angle_error_norm_deg": float(np.rad2deg(np.sqrt(eR1 * eR1 + eR2 * eR2))),
        "target_step_error_norm_rad": float(np.sqrt(eF1 * eF1 + eF2 * eF2)),
        "target_step_error_norm_deg": float(np.rad2deg(np.sqrt(eF1 * eF1 + eF2 * eF2))),
        "target_error_R1_deg": float(np.rad2deg(eR1)),
        "target_error_R2_deg": float(np.rad2deg(eR2)),
        "target_error_F1_deg": float(np.rad2deg(eF1)),
        "target_error_F2_deg": float(np.rad2deg(eF2)),
    }


# -----------------------------------------------------------------------------
# Simulation logging
# -----------------------------------------------------------------------------


def simulation_to_rows(sim: Mapping[str, Any], params: Mapping[str, Any]) -> list[dict[str, float]]:
    R1 = np.asarray(sim["R1"], dtype=float)
    R2 = np.asarray(sim["R2"], dtype=float)
    F1 = np.asarray(sim["F1"], dtype=float)
    F2 = np.asarray(sim["F2"], dtype=float)
    t = np.asarray(sim["t"], dtype=float)
    u = np.asarray(sim["u"], dtype=float).reshape(-1)
    X = np.asarray(sim.get("X", []), dtype=float)

    rows: list[dict[str, float]] = []
    n_nodes = R1.shape[0]
    for k in range(n_nodes):
        # At node 0 we do not have a newly solved F yet, so use identity-size NaN values.
        if k == 0:
            state = AcrobotReducedState(
                R1=R1[k],
                R2=R2[k],
                F1_prev=np.eye(2),
                F2_prev=np.eye(2),
            )
            u_value = float(u[0]) if len(u) else float("nan")
        else:
            state = AcrobotReducedState(
                R1=R1[k],
                R2=R2[k],
                F1_prev=F1[k - 1],
                F2_prev=F2[k - 1],
            )
            u_value = float(u[min(k - 1, len(u) - 1)]) if len(u) else float("nan")

        row = state_to_row(state, params=params, time_value=float(t[k]), k=k, u_value=u_value)
        if X.ndim == 2 and X.shape[0] > k and X.shape[1] >= 4:
            row["x1"] = float(X[k, 0])
            row["y1"] = float(X[k, 1])
            row["x2"] = float(X[k, 2])
            row["y2"] = float(X[k, 3])
        else:
            row["x1"] = float("nan")
            row["y1"] = float("nan")
            row["x2"] = float("nan")
            row["y2"] = float("nan")
        row.update(target_errors_from_row(row, params))
        rows.append(row)

    return rows


def write_simulation_log(
    sim: Mapping[str, Any],
    final_state: AcrobotReducedState,
    params: Mapping[str, Any],
    run_dir: Path,
    mpc_iteration: Optional[int] = None,
    sdp_initial_next: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = simulation_to_rows(sim, params)
    write_csv(run_dir / "simulation_trajectory.csv", rows)

    # Save arrays for separate plotting/debugging scripts.
    np.savez_compressed(
        run_dir / "simulation_arrays.npz",
        t=np.asarray(sim.get("t", []), dtype=float),
        X=np.asarray(sim.get("X", []), dtype=float),
        R1=np.asarray(sim.get("R1", []), dtype=float),
        R2=np.asarray(sim.get("R2", []), dtype=float),
        F1=np.asarray(sim.get("F1", []), dtype=float),
        F2=np.asarray(sim.get("F2", []), dtype=float),
        thetaR=np.asarray(sim.get("thetaR", []), dtype=float),
        thetaF=np.asarray(sim.get("thetaF", []), dtype=float),
        u=np.asarray(sim.get("u", []), dtype=float),
        residual_inf=np.asarray(sim.get("residual_inf", []), dtype=float),
        solver_success=np.asarray(sim.get("solver_success", []), dtype=bool),
        accepted_by_residual=np.asarray(
            sim.get("accepted_by_residual", []), dtype=bool
        ),
    )

    final_row = state_to_row(
        final_state,
        params=params,
        time_value=float(rows[-1]["t"]) if rows else 0.0,
        k=int(rows[-1]["k"]) if rows else 0,
        u_value=float(rows[-1]["u"]) if rows else float("nan"),
    )
    final_row.update(target_errors_from_row(final_row, params))

    accepted_mask = np.asarray(sim.get("accepted_by_residual", []), dtype=bool)
    residuals = np.asarray(sim.get("residual_inf", []), dtype=float)
    accepted_steps = np.flatnonzero(accepted_mask)
    if accepted_steps.size:
        accepted_residuals = residuals[accepted_steps]
        max_accepted_offset = int(np.argmax(accepted_residuals))
        max_accepted_residual = float(accepted_residuals[max_accepted_offset])
        max_accepted_residual_step: Optional[int] = int(
            accepted_steps[max_accepted_offset]
        )
    else:
        max_accepted_residual = 0.0
        max_accepted_residual_step = None

    summary = {
        "mpc_iteration": mpc_iteration,
        "n_substeps": int(len(sim.get("u", []))),
        "n_nodes": int(len(sim.get("t", []))),
        "max_residual_inf": float(np.max(np.asarray(sim.get("residual_inf", [0.0]), dtype=float))),
        "lgvi_accepted_failure_count": int(accepted_steps.size),
        "max_accepted_residual": max_accepted_residual,
        "max_accepted_residual_local_step": max_accepted_residual_step,
        "hard_failure_occurred": False,
        "final_state": final_row,
        "next_sdp_initial": sdp_initial_next or {},
    }

    try:
        diag = diagnostics_lgvi(model=make_model_from_params(params), sim=dict(sim))
        summary["max_constraint_norm"] = float(np.max(diag.get("phi_norm", np.array([np.nan]))))
    except Exception:
        summary["max_constraint_norm"] = float("nan")

    with open(run_dir / "simulation_summary.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(summary), f, indent=2)

    with open(run_dir / "simulation_readable_log.txt", "w", encoding="utf-8") as f:
        f.write("SIMULATION LOG\n")
        f.write("=" * 80 + "\n")
        if mpc_iteration is not None:
            f.write(f"MPC iteration: {mpc_iteration}\n")
        f.write(f"n_substeps: {summary['n_substeps']}\n")
        f.write(f"n_nodes: {summary['n_nodes']}\n")
        f.write(f"max residual inf: {summary['max_residual_inf']:.12e}\n")
        f.write(f"max constraint norm: {summary['max_constraint_norm']:.12e}\n")
        f.write(
            "LGVI solve failures accepted by residual tolerance: "
            f"{summary['lgvi_accepted_failure_count']}\n"
        )
        f.write(
            f"maximum accepted residual: {summary['max_accepted_residual']:.12e}\n"
        )
        f.write(
            "local simulation step of maximum accepted residual: "
            f"{summary['max_accepted_residual_local_step']}\n"
        )
        f.write(f"hard failure occurred: {summary['hard_failure_occurred']}\n")
        if accepted_steps.size:
            f.write("\nWARNING: accepted near-converged LGVI solves:\n")
            for local_step in accepted_steps:
                f.write(
                    f"  local step {int(local_step)}: solver success=False, "
                    f"residual_inf={residuals[local_step]:.12e}\n"
                )
        f.write("\nFinal state:\n")
        for key, val in final_row.items():
            f.write(f"  {key}: {val}\n")
        f.write("\nNext SDP initial values:\n")
        for key, val in (sdp_initial_next or {}).items():
            f.write(f"  {key}: {val}\n")

    return summary


def write_simulation_hard_failure_log(
    run_dir: Path,
    mpc_iteration: Optional[int],
    exc: LGVISolveError,
) -> None:
    """Persist LGVI hard-failure diagnostics before propagating the exception."""
    run_dir.mkdir(parents=True, exist_ok=True)
    accepted = exc.accepted_failures_before_hard_failure
    if accepted:
        max_step, max_residual = max(accepted, key=lambda item: item[1])
    else:
        max_step, max_residual = None, 0.0

    summary = {
        "mpc_iteration": mpc_iteration,
        "lgvi_accepted_failure_count": len(accepted),
        "max_accepted_residual": float(max_residual),
        "max_accepted_residual_local_step": max_step,
        "hard_failure_occurred": True,
        "hard_failure_local_step": exc.local_sim_step,
        "hard_failure_residual_inf": exc.residual_inf,
        "hard_failure_nfev": exc.nfev,
        "hard_failure_message": exc.solver_message,
    }
    with open(run_dir / "simulation_summary.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(summary), f, indent=2)

    with open(run_dir / "simulation_readable_log.txt", "w", encoding="utf-8") as f:
        f.write("SIMULATION LOG - HARD FAILURE\n")
        f.write("=" * 80 + "\n")
        if mpc_iteration is not None:
            f.write(f"MPC iteration: {mpc_iteration}\n")
        f.write(
            "LGVI solve failures accepted by residual tolerance before failure: "
            f"{len(accepted)}\n"
        )
        f.write(f"maximum accepted residual: {max_residual:.12e}\n")
        f.write(f"local simulation step where it occurred: {max_step}\n")
        f.write("hard failure occurred: True\n")
        f.write(f"hard failure local simulation step: {exc.local_sim_step}\n")
        f.write(f"hard failure residual_inf: {exc.residual_inf:.12e}\n")
        f.write(f"hard failure nfev: {exc.nfev}\n")
        f.write(f"hard failure message: {exc.solver_message}\n")


def simulate_and_log_control(
    params: Mapping[str, Any],
    model: Any,
    state: AcrobotReducedState,
    u_value: float,
    run_dir: Path,
    mpc_iteration: Optional[int] = None,
) -> Tuple[AcrobotReducedState, Dict[str, Any], Dict[str, float], Dict[str, Any]]:
    """
    Apply one MPC control input, log the simulation, and return the next SDP initial data.
    """
    try:
        final_state, sim = simulate_one_control_interval_from_params(
            params=params,
            model=model,
            state=state,
            u_j=float(u_value),
            root_tol=1e-10,
            lgvi_maxfev=int(params.get("lgvi_maxfev", 2000)),
            normalized=False,
            accept_residual=bool(params.get("accept_residual", True)),
            accept_residual_tol=float(params.get("accept_residual_tol", 1.0e-3)),
        )
    except LGVISolveError as exc:
        write_simulation_hard_failure_log(run_dir, mpc_iteration, exc)
        raise

    sdp_initial_next = convert_state_to_sdp_initial_scalars(
        state=final_state,
        model=model,
        dt_physical=float(params["dt_sim"]),
        dt_sdp=float(params["dt_sdp"]),
    )

    summary = write_simulation_log(
        sim=sim,
        final_state=final_state,
        params=params,
        run_dir=run_dir,
        mpc_iteration=mpc_iteration,
        sdp_initial_next=sdp_initial_next,
    )

    return final_state, sim, sdp_initial_next, summary


# -----------------------------------------------------------------------------
# Final MPC plotting and GIF
# -----------------------------------------------------------------------------


def plot_mpc_results(history_rows: list[Mapping[str, float]], params: Mapping[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not history_rows:
        return

    t = np.array([row["t"] for row in history_rows], dtype=float)
    u = np.array([row["u"] for row in history_rows], dtype=float)
    thetaR1 = np.array([row["thetaR1_deg"] for row in history_rows], dtype=float)
    thetaR2 = np.array([row["thetaR2_deg"] for row in history_rows], dtype=float)
    thetaF1 = np.array([row["thetaF1_deg"] for row in history_rows], dtype=float)
    thetaF2 = np.array([row["thetaF2_deg"] for row in history_rows], dtype=float)
    target_norm = np.array([row["target_angle_error_norm_deg"] for row in history_rows], dtype=float)

    R_orth = {
        "R1": np.array([row["R1_orth_error"] for row in history_rows], dtype=float),
        "R2": np.array([row["R2_orth_error"] for row in history_rows], dtype=float),
        "F1": np.array([row["F1_orth_error"] for row in history_rows], dtype=float),
        "F2": np.array([row["F2_orth_error"] for row in history_rows], dtype=float),
    }

    plt.figure()
    plt.plot(t, u, marker="o")
    plt.xlabel("time [s]")
    plt.ylabel("applied control u_1")
    plt.title("MPC applied controls")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "control_inputs.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(t, thetaR1, marker="o", label="thetaR1")
    plt.plot(t, thetaR2, marker="o", label="thetaR2")
    plt.axhline(np.rad2deg(float(params["thetaR1_des"])), linestyle="--", label="thetaR1_des")
    plt.axhline(np.rad2deg(float(params["thetaR2_des"])), linestyle=":", label="thetaR2_des")
    plt.xlabel("time [s]")
    plt.ylabel("absolute angle [deg]")
    plt.title("Absolute link angles")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "target_position_angles.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(t, target_norm, marker="o")
    plt.xlabel("time [s]")
    plt.ylabel("norm error to target R_des [deg]")
    plt.title("Target rotation error norm")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "target_rotation_error_norm.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(t, thetaF1, marker="o", label="thetaF1 prev")
    plt.plot(t, thetaF2, marker="o", label="thetaF2 prev")
    plt.xlabel("time [s]")
    plt.ylabel("step angle of F [deg]")
    plt.title("Step rotation / velocity proxy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "velocity_F_step_angles.png", dpi=200)
    plt.close()

    plt.figure()
    for name, vals in R_orth.items():
        plt.semilogy(t, np.maximum(vals, 1e-18), marker="o", label=name)
    plt.xlabel("time [s]")
    plt.ylabel("SO(2) orthogonality error")
    plt.title("SO(2) errors for R and F")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "SO2_errors_R_F.png", dpi=200)
    plt.close()

    elbow_x = np.array([row["elbow_x"] for row in history_rows], dtype=float)
    elbow_y = np.array([row["elbow_y"] for row in history_rows], dtype=float)
    tip_x = np.array([row["tip_x"] for row in history_rows], dtype=float)
    tip_y = np.array([row["tip_y"] for row in history_rows], dtype=float)

    plt.figure()
    plt.plot(elbow_x, elbow_y, marker="o", label="elbow")
    plt.plot(tip_x, tip_y, marker="o", label="tip")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Acrobot trajectories")
    plt.axis("equal")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "trajectories_xy.png", dpi=200)
    plt.close()


def create_acrobot_gif(
    history_rows: list[Mapping[str, float]],
    out_path: Path,
    show_trajectory_trace: bool = False,
) -> None:
    if len(history_rows) < 2:
        return

    base_x = np.array([row["base_x"] for row in history_rows], dtype=float)
    base_y = np.array([row["base_y"] for row in history_rows], dtype=float)
    elbow_x = np.array([row["elbow_x"] for row in history_rows], dtype=float)
    elbow_y = np.array([row["elbow_y"] for row in history_rows], dtype=float)
    tip_x = np.array([row["tip_x"] for row in history_rows], dtype=float)
    tip_y = np.array([row["tip_y"] for row in history_rows], dtype=float)

    all_x = np.concatenate([base_x, elbow_x, tip_x])
    all_y = np.concatenate([base_y, elbow_y, tip_y])
    margin = 0.2

    fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(np.min(all_x) - margin), float(np.max(all_x) + margin))
    ax.set_ylim(float(np.min(all_y) - margin), float(np.max(all_y) + margin))
    ax.grid(True)
    ax.set_title("MPC-SDP acrobot simulation")

    line, = ax.plot([], [], marker="o", linewidth=3)
    trace = ax.plot([], [], color="orange", linewidth=1)[0] if show_trajectory_trace else None

    def init():
        line.set_data([], [])
        if trace is not None:
            trace.set_data([], [])
            return line, trace
        return (line,)

    def update(frame: int):
        xs = [base_x[frame], elbow_x[frame], tip_x[frame]]
        ys = [base_y[frame], elbow_y[frame], tip_y[frame]]
        line.set_data(xs, ys)
        if trace is not None:
            trace.set_data(tip_x[: frame + 1], tip_y[: frame + 1])
            return line, trace
        return (line,)

    anim = FuncAnimation(fig, update, frames=len(history_rows), init_func=init, blit=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=10))
    plt.close(fig)


# -----------------------------------------------------------------------------
# Standalone simulation test
# -----------------------------------------------------------------------------


def main() -> None:
    yaml_path = PROJECT_ROOT / "config" / "acrobot_physical.yaml"
    cfg = load_yaml_config(yaml_path)
    params = build_common_params(cfg)

    model, state0 = make_initial_state_from_params(params, h_key="dt_sim")
    u_test = 0.0

    final_state, sim, sdp_next, summary = simulate_and_log_control(
        params=params,
        model=model,
        state=state0,
        u_value=u_test,
        run_dir=SIM_RESULTS_DIR / "standalone_simulation_test",
        mpc_iteration=None,
    )

    print("Standalone simulation finished.")
    print("Results saved to:", SIM_RESULTS_DIR / "standalone_simulation_test")
    print("Next SDP initial:")
    for k, v in sdp_next.items():
        print(f"  {k}: {v:+.12e}")


if __name__ == "__main__":
    main()
