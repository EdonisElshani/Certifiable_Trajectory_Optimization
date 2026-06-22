from __future__ import annotations

import gc
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from config.config_loader import load_yaml_config, build_common_params
from SDP.solve import solve_sdp
from SDP.objective import evaluate_objective_from_vector
from Numerical_Simulation.lie_group_so2 import angle_from_R


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = PROJECT_ROOT / "Results"
SDP_RESULTS_DIR = RESULTS_ROOT / "Results-Open_Loop-SDP"


# -----------------------------------------------------------------------------
# Small numeric helpers
# -----------------------------------------------------------------------------


def _to_float(x: Any, default: float = float("nan")) -> float:
    try:
        arr = np.asarray(x)
        if arr.size == 0:
            return default
        return float(arr.reshape(-1)[0])
    except Exception:
        return default


def _json_safe(obj: Any) -> Any:
    """Convert NumPy-heavy objects to JSON-safe Python objects."""
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Mapping):
        out = {}
        for k, v in obj.items():
            if callable(v):
                continue
            try:
                json.dumps({str(k): _json_safe(v)})
                out[str(k)] = _json_safe(v)
            except Exception:
                out[str(k)] = str(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return str(obj)


def so2_error(M: np.ndarray) -> Dict[str, float]:
    M = np.asarray(M, dtype=float).reshape(2, 2)
    return {
        "circle_error": float(M[0, 0] ** 2 + M[1, 0] ** 2 - 1.0),
        "orth_error_fro": float(np.linalg.norm(M.T @ M - np.eye(2), ord="fro")),
        "det_error": float(np.linalg.det(M) - 1.0),
        "theta_rad": float(angle_from_R(M)),
        "theta_deg": float(np.rad2deg(angle_from_R(M))),
    }


def rotation_tracking_error(R: np.ndarray, c_des: float, s_des: float) -> Dict[str, float]:
    R_des = np.array([[c_des, -s_des], [s_des, c_des]], dtype=float)
    R = np.asarray(R, dtype=float).reshape(2, 2)
    R_rel = R_des.T @ R
    angle_err = float(angle_from_R(R_rel))
    return {
        "fro_error": float(np.linalg.norm(R - R_des, ord="fro")),
        "angle_error_rad": angle_err,
        "angle_error_deg": float(np.rad2deg(angle_err)),
    }


def kinematics_error(sol: Mapping[str, Any], N: int) -> Dict[str, Any]:
    """Check R_{i,k+1} = R_{i,k} F_{i,k} for extracted SDP solution."""
    rows = []
    max_link1 = 0.0
    max_link2 = 0.0

    for k in range(N):
        R11 = np.asarray(sol["R1"][k], dtype=float).reshape(2, 2)
        R12 = np.asarray(sol["R1"][k + 1], dtype=float).reshape(2, 2)
        F11 = np.asarray(sol["F1"][k], dtype=float).reshape(2, 2)

        R21 = np.asarray(sol["R2"][k], dtype=float).reshape(2, 2)
        R22 = np.asarray(sol["R2"][k + 1], dtype=float).reshape(2, 2)
        F21 = np.asarray(sol["F2"][k], dtype=float).reshape(2, 2)

        e1 = float(np.linalg.norm(R12 - R11 @ F11, ord="fro"))
        e2 = float(np.linalg.norm(R22 - R21 @ F21, ord="fro"))
        max_link1 = max(max_link1, e1)
        max_link2 = max(max_link2, e2)
        rows.append({"k": k, "kin_error_link1": e1, "kin_error_link2": e2})

    return {"rows": rows, "max_link1": max_link1, "max_link2": max_link2}


def analyze_moment_matrix_ranks(
    Xs: Any,
    rank_threshold: float = 1e-2,
    dominant_threshold: float = 0.999,
) -> Dict[str, Any]:
    """Analyze numerical rank-one recovery of the actual SDP moment matrices."""
    dominant_weights = []
    second_weights = []
    rank_proxies = []
    is_rank_one_by_block = []
    matrix_shapes = []

    for X in Xs:
        X_array = np.asarray(X, dtype=float)
        if X_array.ndim != 2 or X_array.shape[0] != X_array.shape[1]:
            raise ValueError(f"Moment matrix must be square; got shape {X_array.shape}")

        X_sym = 0.5 * (X_array + X_array.T)
        eigvals = np.linalg.eigvalsh(X_sym)[::-1]
        eigvals = np.clip(eigvals, 0.0, None)
        eig_sum = float(np.sum(eigvals))
        weights = eigvals / eig_sum if eig_sum > 0.0 else np.zeros_like(eigvals)

        dominant_weight = float(weights[0]) if weights.size else 0.0
        second_weight = float(weights[1]) if weights.size > 1 else 0.0
        rank_proxy = int(np.count_nonzero(weights > rank_threshold))
        is_rank_one = rank_proxy == 1 and dominant_weight >= dominant_threshold

        matrix_shapes.append(tuple(int(v) for v in X_array.shape))
        dominant_weights.append(dominant_weight)
        second_weights.append(second_weight)
        rank_proxies.append(rank_proxy)
        is_rank_one_by_block.append(bool(is_rank_one))

    return {
        "num_moment_matrices": len(matrix_shapes),
        "rank_threshold": float(rank_threshold),
        "dominant_threshold": float(dominant_threshold),
        "matrix_shape_by_block": matrix_shapes,
        "dominant_weight_by_block": dominant_weights,
        "second_weight_by_block": second_weights,
        "rank_proxy_by_block": rank_proxies,
        "is_rank_one_by_block": is_rank_one_by_block,
        "all_blocks_rank_one": bool(matrix_shapes) and all(is_rank_one_by_block),
        "worst_rank_proxy": max(rank_proxies, default=0),
        "min_dominant_weight": min(dominant_weights, default=float("nan")),
        "max_second_weight": max(second_weights, default=float("nan")),
    }


def write_moment_rank_condition(run_dir: Path, analysis: Mapping[str, Any]) -> None:
    """Write human-readable and CSV versions of the moment-rank certificate."""
    txt_path = run_dir / "moment_rank_condition.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Moment matrix rank condition\n")
        f.write("=" * 80 + "\n")
        f.write(f"Rank threshold: {float(analysis['rank_threshold']):.2e}\n")
        f.write(
            "Dominant weight threshold: "
            f"{float(analysis['dominant_threshold']):.3f}\n"
        )
        f.write(f"Number of moment matrices: {int(analysis['num_moment_matrices'])}\n\n")
        f.write(f"Worst rank proxy: {int(analysis['worst_rank_proxy'])}\n")
        f.write(f"Minimum dominant weight: {float(analysis['min_dominant_weight']):.7e}\n")
        f.write(f"Maximum second weight: {float(analysis['max_second_weight']):.7e}\n")
        f.write(f"All blocks rank-one: {bool(analysis['all_blocks_rank_one'])}\n\n")
        f.write("Per-block rank proxy\n")
        f.write("-" * 80 + "\n")
        f.write(
            "block   shape       dominant_weight      second_weight       "
            "rank_proxy   status\n"
        )
        for i, shape in enumerate(analysis["matrix_shape_by_block"]):
            status = "tight" if analysis["is_rank_one_by_block"][i] else "not tight"
            shape_text = f"{shape[0]}x{shape[1]}"
            f.write(
                f"{i:<7d} {shape_text:<11s} "
                f"{analysis['dominant_weight_by_block'][i]:<20.8e} "
                f"{analysis['second_weight_by_block'][i]:<19.8e} "
                f"{analysis['rank_proxy_by_block'][i]:<12d} {status}\n"
            )

    rank_rows = []
    for i, shape in enumerate(analysis["matrix_shape_by_block"]):
        is_rank_one = bool(analysis["is_rank_one_by_block"][i])
        rank_rows.append(
            {
                "block_index": i,
                "matrix_shape": f"{shape[0]}x{shape[1]}",
                "dominant_weight": analysis["dominant_weight_by_block"][i],
                "second_weight": analysis["second_weight_by_block"][i],
                "rank_proxy": analysis["rank_proxy_by_block"][i],
                "is_rank_one": is_rank_one,
                "status": "tight" if is_rank_one else "not tight",
            }
        )
    write_csv(run_dir / "moment_rank_condition.csv", rank_rows)


def _get_preferred_solution(out: Mapping[str, Any]) -> Tuple[str, Mapping[str, Any]]:
    preferred = str(out.get("preferred_extraction", "ordered"))
    solutions = out.get("solutions", {})
    if preferred not in solutions:
        if "ordered" in solutions:
            preferred = "ordered"
        elif "robust" in solutions:
            preferred = "robust"
        elif "naive" in solutions:
            preferred = "naive"
        else:
            raise RuntimeError("No extracted solution found in solve_sdp output.")
    return preferred, solutions[preferred]


def _result_scalar(out: Mapping[str, Any]) -> float:
    return _to_float(out.get("result", np.nan))


def compute_suboptimality_and_tightness(out: Mapping[str, Any], preferred: str) -> Dict[str, Any]:
    """
    Collect lower bound / extracted cost / SO(2) / kinematic tightness.

    The exact keys inside gap_info depend on your SDP/extraction implementation, so this
    function stores raw gap_info and also tries to compute a compact objective gap.
    """
    params = out["params"]
    v_opt = out.get("extracted_vectors", {}).get(preferred, None)

    lower_bound = _result_scalar(out)
    extracted_cost = float("nan")

    if v_opt is not None:
        try:
            extracted_cost = float(evaluate_objective_from_vector(v_opt, params))
        except Exception:
            extracted_cost = float("nan")

    abs_gap = extracted_cost - lower_bound if np.isfinite(extracted_cost) and np.isfinite(lower_bound) else float("nan")
    rel_gap = abs_gap / max(1.0, abs(extracted_cost)) if np.isfinite(abs_gap) else float("nan")

    sol = out["solutions"][preferred]
    N = int(params["N"])
    kin = kinematics_error(sol, N)

    errors_by_method = out.get("errors_by_method", {})
    so2_pref = errors_by_method.get(preferred, {})

    max_so2 = {}
    for key, values in so2_pref.items():
        try:
            max_so2[key] = float(np.max(np.asarray(values, dtype=float))) if len(values) else float("nan")
        except Exception:
            max_so2[key] = float("nan")

    return {
        "lower_bound_sdp": lower_bound,
        "extracted_cost": extracted_cost,
        "absolute_suboptimality_gap": abs_gap,
        "relative_suboptimality_gap": rel_gap,
        "gap_info_raw": _json_safe(out.get("gap_info", {})),
        "max_so2_errors": max_so2,
        "kinematics_tightness": {
            "max_kin_error_link1": kin["max_link1"],
            "max_kin_error_link2": kin["max_link2"],
        },
        "extraction_info": _json_safe(out.get("extraction_info", {}).get(preferred, {})),
    }


def matrix_entries(prefix: str, M: Optional[np.ndarray]) -> Dict[str, float]:
    if M is None:
        return {
            f"{prefix}_00": float("nan"),
            f"{prefix}_01": float("nan"),
            f"{prefix}_10": float("nan"),
            f"{prefix}_11": float("nan"),
        }
    M = np.asarray(M, dtype=float).reshape(2, 2)
    return {
        f"{prefix}_00": float(M[0, 0]),
        f"{prefix}_01": float(M[0, 1]),
        f"{prefix}_10": float(M[1, 0]),
        f"{prefix}_11": float(M[1, 1]),
    }


def solution_to_rows(sol: Mapping[str, Any], params: Mapping[str, Any]) -> list[dict[str, float]]:
    """One row per SDP node k. F, lambda, u are filled when defined."""
    N = int(params["N"])
    rows: list[dict[str, float]] = []

    for k in range(N + 1):
        R1 = sol["R1"].get(k) if isinstance(sol["R1"], Mapping) else sol["R1"][k]
        R2 = sol["R2"].get(k) if isinstance(sol["R2"], Mapping) else sol["R2"][k]

        F1 = None
        F2 = None
        if k < N:
            F1 = sol["F1"].get(k) if isinstance(sol["F1"], Mapping) else sol["F1"][k]
            F2 = sol["F2"].get(k) if isinstance(sol["F2"], Mapping) else sol["F2"][k]

        row: dict[str, float] = {"k": float(k)}
        row.update(matrix_entries("R1", R1))
        row.update(matrix_entries("R2", R2))
        row.update(matrix_entries("F1", F1))
        row.update(matrix_entries("F2", F2))

        row["thetaR1_rad"] = float(angle_from_R(R1))
        row["thetaR2_rad"] = float(angle_from_R(R2))
        row["thetaR1_deg"] = float(np.rad2deg(row["thetaR1_rad"]))
        row["thetaR2_deg"] = float(np.rad2deg(row["thetaR2_rad"]))

        if F1 is not None:
            row["thetaF1_rad"] = float(angle_from_R(F1))
            row["thetaF1_deg"] = float(np.rad2deg(row["thetaF1_rad"]))
        else:
            row["thetaF1_rad"] = float("nan")
            row["thetaF1_deg"] = float("nan")

        if F2 is not None:
            row["thetaF2_rad"] = float(angle_from_R(F2))
            row["thetaF2_deg"] = float(np.rad2deg(row["thetaF2_rad"]))
        else:
            row["thetaF2_rad"] = float("nan")
            row["thetaF2_deg"] = float("nan")

        # lambda/u only exist for k=1,...,N-1 in your thesis indexing.
        for name in ["lambda0", "lambda12"]:
            if name in sol and k in sol[name]:
                val = np.asarray(sol[name][k], dtype=float).reshape(2)
                row[f"{name}_x"] = float(val[0])
                row[f"{name}_y"] = float(val[1])
            else:
                row[f"{name}_x"] = float("nan")
                row[f"{name}_y"] = float("nan")

        if "u" in sol and k in sol["u"]:
            row["u"] = float(sol["u"][k])
        else:
            row["u"] = float("nan")

        eR1 = so2_error(R1)
        eR2 = so2_error(R2)
        row["SO2_R1_orth_error"] = eR1["orth_error_fro"]
        row["SO2_R2_orth_error"] = eR2["orth_error_fro"]
        row["SO2_R1_det_error"] = eR1["det_error"]
        row["SO2_R2_det_error"] = eR2["det_error"]

        if F1 is not None:
            eF1 = so2_error(F1)
            row["SO2_F1_orth_error"] = eF1["orth_error_fro"]
            row["SO2_F1_det_error"] = eF1["det_error"]
        else:
            row["SO2_F1_orth_error"] = float("nan")
            row["SO2_F1_det_error"] = float("nan")

        if F2 is not None:
            eF2 = so2_error(F2)
            row["SO2_F2_orth_error"] = eF2["orth_error_fro"]
            row["SO2_F2_det_error"] = eF2["det_error"]
        else:
            row["SO2_F2_orth_error"] = float("nan")
            row["SO2_F2_det_error"] = float("nan")

        rows.append(row)

    return rows


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


def cleanup_old_solver_artifacts(out: Mapping[str, Any], enabled: bool = True) -> None:
    """
    Delete the old parent-folder data/markdown/figs/logs artifacts created by SDP.solve.

    We first write our compact Results logs. Then these bulky artifacts can go.
    """
    if not enabled:
        return

    prefix = str(out.get("prefix", ""))
    if not prefix:
        return

    for folder in ["data", "markdown", "figs", "logs"]:
        path = PROJECT_ROOT / folder / prefix
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)


def write_sdp_run_logs(
    out: Mapping[str, Any],
    run_dir: Path,
    mpc_iteration: Optional[int] = None,
    cleanup_solver_artifacts_enabled: bool = True,
) -> Dict[str, Any]:
    """Write compact SDP logs inside Results/Results-Open_Loop-SDP."""
    run_dir.mkdir(parents=True, exist_ok=True)

    preferred, sol = _get_preferred_solution(out)
    params = out["params"]
    solve_metrics = compute_suboptimality_and_tightness(out, preferred)

    rows = solution_to_rows(sol, params)
    write_csv(run_dir / "sdp_trajectory.csv", rows)

    kin = kinematics_error(sol, int(params["N"]))
    write_csv(run_dir / "sdp_kinematics_errors.csv", kin["rows"])

    rank_analysis: Optional[Dict[str, Any]] = None
    moment_warning: Optional[str] = None
    Xs = out.get("Xs")
    if Xs is None or len(Xs) == 0:
        moment_warning = "Moment matrices Xs were not available from the solver output."
    else:
        try:
            moment_arrays = [np.asarray(X, dtype=float) for X in Xs]
            np.savez_compressed(
                run_dir / "moment_matrices.npz",
                **{f"X_{i:03d}": X for i, X in enumerate(moment_arrays)},
            )
            rank_analysis = analyze_moment_matrix_ranks(
                moment_arrays,
                rank_threshold=1e-2,
                dominant_threshold=0.999,
            )
            write_moment_rank_condition(run_dir, rank_analysis)
        except Exception as exc:
            moment_warning = f"Moment-rank condition was not available: {exc}"

    max_so2_errors = solve_metrics["max_so2_errors"]
    max_so2_error = max(max_so2_errors.values(), default=float("nan"))
    kin_metrics = solve_metrics["kinematics_tightness"]
    max_kinematic_error = max(kin_metrics.values(), default=float("nan"))

    # This is deliberately scalar-only. Trajectories and matrices have their own
    # CSV/NPZ files and must never be serialized into JSON.
    metrics: Dict[str, Any] = {
        "mpc_iteration": mpc_iteration,
        "N": int(params["N"]),
        "dt_sdp": float(params["dt"]),
        "lower_bound_sdp": solve_metrics["lower_bound_sdp"],
        "extracted_cost": solve_metrics["extracted_cost"],
        "absolute_suboptimality_gap": solve_metrics["absolute_suboptimality_gap"],
        "relative_suboptimality_gap": solve_metrics["relative_suboptimality_gap"],
        "max_so2_error": max_so2_error,
        "max_kinematic_error": max_kinematic_error,
    }
    if rank_analysis is not None:
        metrics.update(
            {
                "num_moment_matrices": rank_analysis["num_moment_matrices"],
                "worst_rank_proxy": rank_analysis["worst_rank_proxy"],
                "min_dominant_weight": rank_analysis["min_dominant_weight"],
                "max_second_weight": rank_analysis["max_second_weight"],
                "all_blocks_rank_one": rank_analysis["all_blocks_rank_one"],
            }
        )

    with open(run_dir / "sdp_metrics.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(metrics), f, indent=2)

    with open(run_dir / "sdp_readable_log.txt", "w", encoding="utf-8") as f:
        f.write("SDP solve summary\n")
        f.write("=" * 80 + "\n")
        if mpc_iteration is not None:
            f.write(f"MPC iteration: {mpc_iteration}\n")
        f.write(f"N: {int(params['N'])}\n")
        f.write(f"dt_sdp: {float(params['dt'])}\n\n")
        f.write(f"Lower bound SDP: {solve_metrics['lower_bound_sdp']:+.12e}\n")
        f.write(f"Extracted cost: {solve_metrics['extracted_cost']:+.12e}\n")
        f.write(
            "Absolute suboptimality gap: "
            f"{solve_metrics['absolute_suboptimality_gap']:+.12e}\n"
        )
        f.write(
            "Relative suboptimality gap: "
            f"{solve_metrics['relative_suboptimality_gap']:+.12e}\n"
        )
        f.write("\nMax SO(2) errors:\n")
        for key in ["R1", "R2", "F1", "F2"]:
            val = max_so2_errors.get(key, float("nan"))
            f.write(f"  {key}: {val:.12e}\n")
        f.write("\nKinematic tightness:\n")
        for key, val in kin_metrics.items():
            f.write(f"  {key}: {val:.12e}\n")
        f.write("\nMoment matrix rank condition:\n")
        if rank_analysis is None:
            f.write(f"  WARNING: {moment_warning}\n")
            f.write("  Moment-rank condition: not available\n")
        else:
            f.write(
                f"  Number of moment matrices: {rank_analysis['num_moment_matrices']}\n"
            )
            f.write(f"  Worst rank proxy: {rank_analysis['worst_rank_proxy']}\n")
            f.write(
                "  Minimum dominant weight: "
                f"{rank_analysis['min_dominant_weight']:.12e}\n"
            )
            f.write(
                "  Maximum second weight: "
                f"{rank_analysis['max_second_weight']:.12e}\n"
            )
            f.write(
                f"  All blocks rank-one: {rank_analysis['all_blocks_rank_one']}\n"
            )
        f.write("\nSaved files:\n")
        if rank_analysis is not None:
            f.write("  moment_matrices.npz\n")
            f.write("  moment_rank_condition.txt\n")
            f.write("  moment_rank_condition.csv\n")
        f.write("  sdp_trajectory.csv\n")
        f.write("  sdp_kinematics_errors.csv\n")
        f.write("  sdp_metrics.json\n")

    cleanup_old_solver_artifacts(out, enabled=cleanup_solver_artifacts_enabled)
    return metrics


def run_open_loop_sdp(
    yaml_path: str | Path = PROJECT_ROOT / "config" / "acrobot_physical.yaml",
    mpc_initial: Optional[Dict[str, float]] = None,
    run_name: Optional[str] = None,
    cleanup_solver_artifacts_enabled: bool = True,
) -> Dict[str, Any]:
    """
    Solve one open-loop SDP and write compact logs to Results/Results-Open_Loop-SDP.
    """
    cfg = load_yaml_config(yaml_path)
    params = build_common_params(cfg)
    if mpc_initial is not None:
        params.update(mpc_initial)

    if run_name is None:
        run_name = datetime.now().strftime("open_loop_%Y-%m-%d_%H-%M-%S")

    out = solve_sdp(params)
    run_dir = SDP_RESULTS_DIR / run_name
    summary = write_sdp_run_logs(
        out,
        run_dir=run_dir,
        mpc_iteration=None,
        cleanup_solver_artifacts_enabled=cleanup_solver_artifacts_enabled,
    )
    out["compact_summary"] = summary
    out["results_dir"] = str(run_dir)
    return out


def main() -> None:
    out = run_open_loop_sdp()
    print("\nOpen-loop SDP result saved to:")
    print(out["results_dir"])

    # Do not keep accidental large objects alive when this file is used interactively.
    del out
    gc.collect()


if __name__ == "__main__":
    main()
