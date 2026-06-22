from __future__ import annotations

import gc
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

from config.config_loader import load_yaml_config, build_common_params
from SDP.solve import solve_sdp
from Numerical_Simulation.solver_lgvi_acrobot import make_model_from_params, make_initial_state_from_params

from open_loop_sdp import SDP_RESULTS_DIR, write_sdp_run_logs
from simulation import (
    MPC_RESULTS_DIR,
    SIM_RESULTS_DIR,
    simulate_and_log_control,
    simulation_to_rows,
    state_to_row,
    target_errors_from_row,
    write_csv,
    plot_mpc_results,
    create_acrobot_gif,
)


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = PROJECT_ROOT / "Results"


# -----------------------------------------------------------------------------
# Helpers
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


def get_mpc_settings(params: Mapping[str, Any], cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Read optional MPC settings from YAML.

    Add this block to YAML if you want to override defaults:

    mpc:
      max_iterations: 35
      stop_angle_tol_deg: 2.0
      stop_step_angle_tol_deg: 2.0
      stable_steps_required: 3
      cleanup_solver_artifacts: true
    """
    mpc_cfg = cfg.get("mpc", {}) if isinstance(cfg.get("mpc", {}), Mapping) else {}

    return {
        "max_iterations": int(params.get("mpc_max_iterations", mpc_cfg.get("max_iterations", 20))),
        "stop_angle_tol_deg": float(
            params.get("mpc_stop_angle_tol_deg", mpc_cfg.get("stop_angle_tol_deg", 2.0))
        ),
        "stop_step_angle_tol_deg": float(
            params.get(
                "mpc_stop_step_angle_tol_deg",
                mpc_cfg.get("stop_step_angle_tol_deg", 2.0),
            )
        ),
        "stable_steps_required": int(
            params.get(
                "mpc_stable_steps_required",
                mpc_cfg.get("stable_steps_required", 3),
            )
        ),
        "cleanup_solver_artifacts": bool(
            params.get(
                "cleanup_solver_artifacts",
                mpc_cfg.get("cleanup_solver_artifacts", True),
            )
        ),
    }


def write_mpc_summary(
    out_dir: Path,
    params: Mapping[str, Any],
    settings: Mapping[str, Any],
    history_rows: list[Mapping[str, float]],
    sdp_summaries: list[Mapping[str, Any]],
    sim_summaries: list[Mapping[str, Any]],
    stopped: bool,
    stop_reason: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(out_dir / "mpc_history.csv", history_rows)

    final_state = history_rows[-1] if history_rows else {}
    stable_counter = int(final_state.get("stable_counter", 0))
    final_angle_error = final_state.get("target_angle_error_norm_deg")
    final_step_error = final_state.get("target_step_error_norm_deg")
    termination = "stabilization" if stopped else (
        "max_iterations" if stop_reason == "max_iterations reached" else "running"
    )

    compact = {
        "stopped": stopped,
        "termination": termination,
        "stop_reason": stop_reason,
        "num_mpc_iterations": max(0, len(history_rows) - 1),
        "stable_counter": stable_counter,
        "stable_steps_required": int(settings["stable_steps_required"]),
        "stop_angle_tol_deg": float(settings["stop_angle_tol_deg"]),
        "stop_step_angle_tol_deg": float(settings["stop_step_angle_tol_deg"]),
        "target_angle_error_norm_deg": final_angle_error,
        "target_step_error_norm_deg": final_step_error,
        "settings": _json_safe(settings),
        "params": _json_safe({k: v for k, v in params.items() if not callable(v)}),
        "sdp_summaries": _json_safe(sdp_summaries),
        "simulation_summaries": _json_safe(sim_summaries),
        "final_state": _json_safe(final_state),
    }

    with open(out_dir / "mpc_summary.json", "w", encoding="utf-8") as f:
        json.dump(compact, f, indent=2)

    with open(out_dir / "mpc_readable_log.txt", "w", encoding="utf-8") as f:
        f.write("MPC-SDP FINAL RESULT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Stopped: {stopped}\n")
        f.write(f"Termination: {termination}\n")
        f.write(f"Reason: {stop_reason}\n")
        f.write(f"Iterations: {max(0, len(history_rows) - 1)}\n")
        f.write(f"Stable counter: {stable_counter}\n")
        f.write(f"Stable steps required: {int(settings['stable_steps_required'])}\n")
        f.write(f"Final target angle error norm: {final_angle_error} deg\n")
        f.write(f"Final target step error norm: {final_step_error} deg\n")
        f.write("\nMPC settings:\n")
        f.write(f"  mpc_max_iterations: {int(settings['max_iterations'])}\n")
        f.write(f"  mpc_stop_angle_tol_deg: {float(settings['stop_angle_tol_deg'])}\n")
        f.write(
            "  mpc_stop_step_angle_tol_deg: "
            f"{float(settings['stop_step_angle_tol_deg'])}\n"
        )
        f.write(
            "  mpc_stable_steps_required: "
            f"{int(settings['stable_steps_required'])}\n"
        )
        f.write(
            "  cleanup_solver_artifacts: "
            f"{bool(settings['cleanup_solver_artifacts'])}\n"
        )
        f.write("\nApplied controls:\n")
        for row in history_rows[1:]:
            f.write(
                f"  j={int(row['mpc_iteration'])}: "
                f"t={row['t']:.6f}, u_1={row['u']:+.12e}, "
                f"thetaR1={row['thetaR1_deg']:+.6f} deg, "
                f"thetaR2={row['thetaR2_deg']:+.6f} deg, "
                f"target_norm={row['target_angle_error_norm_deg']:.6f} deg\n"
            )
        f.write("\nFinal state:\n")
        if history_rows:
            for key, val in history_rows[-1].items():
                f.write(f"  {key}: {val}\n")


# -----------------------------------------------------------------------------
# Main MPC loop
# -----------------------------------------------------------------------------


def run_mpc_sdp(
    yaml_path: str | Path = PROJECT_ROOT / "config" / "acrobot_physical.yaml",
    run_name: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = load_yaml_config(yaml_path)
    params = build_common_params(cfg)
    settings = get_mpc_settings(params, cfg)
    if int(settings["stable_steps_required"]) < 1:
        raise ValueError("mpc.stable_steps_required must be at least 1")

    if run_name is None:
        run_name = datetime.now().strftime("mpc_%Y-%m-%d_%H-%M-%S")

    mpc_dir = MPC_RESULTS_DIR / run_name
    sdp_parent = SDP_RESULTS_DIR / run_name
    sim_parent = SIM_RESULTS_DIR / run_name

    mpc_dir.mkdir(parents=True, exist_ok=True)
    sdp_parent.mkdir(parents=True, exist_ok=True)
    sim_parent.mkdir(parents=True, exist_ok=True)

    model, state = make_initial_state_from_params(params, h_key="dt_sim")

    history_rows: list[dict[str, float]] = []
    sdp_summaries: list[dict[str, Any]] = []
    sim_summaries: list[dict[str, Any]] = []
    full_trajectory_rows: list[dict[str, Any]] = []

    # Initial row before applying any MPC control.
    initial_row = state_to_row(state, params=params, time_value=0.0, k=0, u_value=float("nan"))
    initial_row.update(target_errors_from_row(initial_row, params))
    initial_row["mpc_iteration"] = -1.0
    initial_row["angle_ok"] = False
    initial_row["step_ok"] = False
    initial_row["stable_counter"] = 0
    history_rows.append(initial_row)

    mpc_initial: Optional[Dict[str, float]] = None
    stopped = False
    stop_reason = "max_iterations reached"
    stable_counter = 0

    for j in range(int(settings["max_iterations"])):
        print("\n" + "=" * 80)
        print(f"MPC iteration {j}")
        print("=" * 80)

        params_sdp = dict(params)
        if mpc_initial is not None:
            params_sdp.update(mpc_initial)

        # Solve SDP. solve_sdp writes temporary old artifacts; write_sdp_run_logs
        # copies compact information into Results and then deletes old artifacts.
        out = solve_sdp(params_sdp)
        u_apply = float(out["first_control"])

        sdp_run_dir = sdp_parent / f"mpc_{j:04d}"
        sdp_summary = write_sdp_run_logs(
            out,
            run_dir=sdp_run_dir,
            mpc_iteration=j,
            cleanup_solver_artifacts_enabled=bool(settings["cleanup_solver_artifacts"]),
        )
        sdp_summaries.append(sdp_summary)

        # Delete bulky solver output before simulating the next step. Keep only compact logs.
        del out
        gc.collect()

        sim_run_dir = sim_parent / f"mpc_{j:04d}"
        state, sim, mpc_initial, sim_summary = simulate_and_log_control(
            params=params,
            model=model,
            state=state,
            u_value=u_apply,
            run_dir=sim_run_dir,
            mpc_iteration=j,
        )
        sim_summaries.append(sim_summary)

        interval_rows = simulation_to_rows(sim, params)
        interval_start_time = j * float(
            params.get("control_interval", params["dt_sdp"])
        )
        # Adjacent intervals share their boundary node. Keep it only once.
        if j > 0:
            interval_rows = interval_rows[1:]
        for sim_row in interval_rows:
            full_row: dict[str, Any] = {
                "global_time": interval_start_time + float(sim_row["t"]),
                "mpc_iteration": j,
                "local_sim_step": int(sim_row["k"]),
                "u_applied": float(sim_row["u"]),
            }
            full_row.update(sim_row)
            full_trajectory_rows.append(full_row)
        write_csv(mpc_dir / "mpc_full_trajectory.csv", full_trajectory_rows)

        current_time = (j + 1) * float(params.get("control_interval", params["dt_sdp"]))
        row = state_to_row(state, params=params, time_value=current_time, k=j + 1, u_value=u_apply)
        row.update(target_errors_from_row(row, params))
        row["mpc_iteration"] = float(j)

        target_angle_error_norm_deg = float(row["target_angle_error_norm_deg"])
        target_step_error_norm_deg = float(row["target_step_error_norm_deg"])
        stop_angle_tol_deg = float(settings["stop_angle_tol_deg"])
        stop_step_angle_tol_deg = float(settings["stop_step_angle_tol_deg"])
        stable_steps_required = int(settings["stable_steps_required"])

        angle_ok = target_angle_error_norm_deg < stop_angle_tol_deg
        step_ok = target_step_error_norm_deg < stop_step_angle_tol_deg
        if angle_ok and step_ok:
            stable_counter += 1
        else:
            stable_counter = 0

        row["angle_ok"] = angle_ok
        row["step_ok"] = step_ok
        row["stable_counter"] = stable_counter
        history_rows.append(row)

        if stable_counter >= stable_steps_required:
            stopped = True
            stop_reason = (
                f"stabilized for {stable_counter} consecutive MPC steps "
                f"(angle error < {stop_angle_tol_deg} deg and "
                f"step error < {stop_step_angle_tol_deg} deg)"
            )

        # Save incrementally so a crash still leaves useful data. Computers are dramatic.
        write_mpc_summary(
            out_dir=mpc_dir,
            params=params,
            settings=settings,
            history_rows=history_rows,
            sdp_summaries=sdp_summaries,
            sim_summaries=sim_summaries,
            stopped=stopped,
            stop_reason=stop_reason if stopped else "running",
        )

        if stopped:
            print("Stopping condition reached.")
            break

        # Do not keep full simulation arrays from this iteration in memory.
        del sim
        gc.collect()

    write_mpc_summary(
        out_dir=mpc_dir,
        params=params,
        settings=settings,
        history_rows=history_rows,
        sdp_summaries=sdp_summaries,
        sim_summaries=sim_summaries,
        stopped=stopped,
        stop_reason=stop_reason,
    )

    if bool(params.get("plot_results", False)):
        plot_mpc_results(history_rows, params=params, out_dir=mpc_dir / "figures")
        if bool(params.get("generate_mpc_gif", float(params["dt_sim"]) > 1.0e-3)):
            gif_rows = full_trajectory_rows if full_trajectory_rows else history_rows
            gif_stride = max(1, int(params.get("gif_stride", 1)))
            gif_rows_strided = gif_rows[::gif_stride]
            if gif_rows_strided and gif_rows_strided[-1] is not gif_rows[-1]:
                gif_rows_strided.append(gif_rows[-1])
            create_acrobot_gif(
                gif_rows_strided,
                out_path=mpc_dir / "acrobot_mpc.gif",
                show_trajectory_trace=False,
            )
        else:
            print(
                "Skipping MPC GIF generation "
                f"(dt_sim={float(params['dt_sim']):g}; enable with "
                "simulation.generate_mpc_gif: true)."
            )
    else:
        print(
            "Skipping all MPC figures and GIF generation "
            "(simulation.plot_results: false)."
        )

    print("\n" + "=" * 80)
    print("MPC-SDP LOOP FINISHED")
    print("=" * 80)
    print(f"Stopped: {stopped}")
    print(f"Reason: {stop_reason}")
    print(f"Stable counter: {stable_counter}/{int(settings['stable_steps_required'])}")
    if history_rows:
        print(
            "Final errors: "
            f"angle={history_rows[-1]['target_angle_error_norm_deg']:.6f} deg, "
            f"step={history_rows[-1]['target_step_error_norm_deg']:.6f} deg"
        )
    print(f"Results folder: {mpc_dir}")

    return {
        "stopped": stopped,
        "stop_reason": stop_reason,
        "results_dir": str(mpc_dir),
        "history_rows": history_rows,
        "sdp_summaries": sdp_summaries,
        "sim_summaries": sim_summaries,
    }


def main() -> None:
    run_mpc_sdp()


if __name__ == "__main__":
    main()
