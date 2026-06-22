from __future__ import annotations

import datetime
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


# Current file:
#   .../SPOT/Python-examples/SPOT_MPC_Acrobot/SDP/solve.py
THIS_FILE = Path(__file__).resolve()

# Project folder:
#   .../SPOT/Python-examples/SPOT_MPC_Acrobot
PROJECT_ROOT = THIS_FILE.parents[1]

# Outer SPOT folder:
#   .../SPOT
SPOT_OUTER_DIR = THIS_FILE.parents[3]

# SPOT Python package folder:
#   .../SPOT/SPOT/PYTHON
SPOT_PYTHON_DIR = SPOT_OUTER_DIR / "SPOT" / "PYTHON"

for path in [
    str(PROJECT_ROOT),      # for config, SDP, Numerical_Simulation
    str(SPOT_OUTER_DIR),    # for from SPOT.PYTHON...
    str(SPOT_PYTHON_DIR),   # for direct imports if needed
]:
    if path not in sys.path:
        sys.path.insert(0, path)

print(f"[solve.py] PROJECT_ROOT   = {PROJECT_ROOT}")
print(f"[solve.py] SPOT_OUTER_DIR = {SPOT_OUTER_DIR}")
print(f"[solve.py] SPOT_PYTHON_DIR = {SPOT_PYTHON_DIR}")


from SPOT.PYTHON.CSTSS_pybind import CSTSS_pybind
from SPOT.PYTHON.numpoly import NumPolySystem, NumPolyExpr, numpoly_visualize
from SPOT.PYTHON.naive_extract import naive_extract
from SPOT.PYTHON.robust_extract_CS import robust_extract_CS, ordered_extract_CS

from config.config_loader import load_yaml_config, build_common_params

from SDP.mapping import attach_mapping_to_params, get_remapped_ids
from SDP.constraints import (
    get_init_constraints,
    get_rotational_kinematics_link1,
    get_rotational_kinematics_link2,
    get_SO2_orthogonality_constraint_rotation_R,
    get_SO2_orthogonality_constraint_rotation_F,
    get_step_angle_bound_constraint_link_1,
    get_step_angle_bound_constraint_link_2,
    get_translational_dynamics_link1,
    get_translational_dynamics_link2,
    get_rotational_dynamics_link1,
    get_rotational_dynamics_link2,
    get_control_bounds,
    get_lambda_bounds,
)
from SDP.objective import build_objective
from SDP.cliques import get_cliques_for_cstss
from SDP.extraction import (
    extract_solution_variables,
    compute_SO2_errors,
    build_gap_info,
    get_first_mpc_control,
)


def complete_sdp_params(params: Dict[str, Any], mpc_initial: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Add SDP-specific derived parameters.

    YAML is still the single source of truth. This only derives:
        dt = dt_sdp
        rho vectors
        nonstandard SO(2) delta parameters
        desired c/s and F desired scalars from YAML thetaF values
        CSTSS defaults
        optional MPC initial boundary data
    """
    params = dict(params)

    # CSTSS defaults.
    params.setdefault("kappa", 2)
    params.setdefault("relax_mode", "SOS")
    params.setdefault("cs_mode", "SELF")
    params.setdefault("ts_mode", "NON")
    params.setdefault("ts_mom_mode", "NON")
    params.setdefault("ts_eq_mode", "NON")
    params.setdefault("if_solve", True)
    params.setdefault("if_mex", True)

    # SDP step uses dt_sdp.
    if "dt_sdp" in params:
        params["dt"] = float(params["dt_sdp"])
    elif "dt" in params:
        params["dt"] = float(params["dt"])
    else:
        raise KeyError("Missing dt_sdp or dt in params.")

    # Horizon.
    if "N" not in params:
        params["N"] = int(params.get("horizon_N", 5))

    # Geometry with thesis convention.
    l1 = float(params["l1"])
    l2 = float(params["l2"])
    params.setdefault("rho_10", np.array([0.0, 0.5 * l1], dtype=float))
    params.setdefault("rho_112", np.array([0.0, -0.5 * l1], dtype=float))
    params.setdefault("rho_212", np.array([0.0, 0.5 * l2], dtype=float))

    params["rho_10"] = np.asarray(params["rho_10"], dtype=float).reshape(2)
    params["rho_112"] = np.asarray(params["rho_112"], dtype=float).reshape(2)
    params["rho_212"] = np.asarray(params["rho_212"], dtype=float).reshape(2)

    if "p_0" not in params:
        if "p0" in params:
            params["p_0"] = np.asarray(params["p0"], dtype=float).reshape(2)
        else:
            params["p_0"] = np.array([0.0, 0.0], dtype=float)
    else:
        params["p_0"] = np.asarray(params["p_0"], dtype=float).reshape(2)

    # Nonstandard SO(2) inertia parameters.
    if "deltaJ1" not in params:
        if "Jd1" in params:
            params["deltaJ1"] = float(np.asarray(params["Jd1"], dtype=float).reshape(2, 2)[0, 0])
        else:
            params["deltaJ1"] = 0.05
    if "deltaJ2" not in params:
        if "Jd2" in params:
            params["deltaJ2"] = float(np.asarray(params["Jd2"], dtype=float).reshape(2, 2)[0, 0])
        else:
            params["deltaJ2"] = 0.05

    params["Jd1"] = np.diag([params["deltaJ1"], params["deltaJ1"]])
    params["Jd2"] = np.diag([params["deltaJ2"], params["deltaJ2"]])

    # Desired rotations.
    params["c1_des"] = float(np.cos(params["thetaR1_des"]))
    params["s1_des"] = float(np.sin(params["thetaR1_des"]))
    params["c2_des"] = float(np.cos(params["thetaR2_des"]))
    params["s2_des"] = float(np.sin(params["thetaR2_des"]))

    # Desired final step comes from YAML-derived thetaF target values.
    if "thetaF1_des" not in params or "thetaF2_des" not in params:
        raise KeyError(
            "Missing thetaF target values. Expected 'thetaF1_des' and "
            "'thetaF2_des' from the YAML-derived params."
        )

    params["a1_des"] = float(np.cos(params["thetaF1_des"]))
    params["b1_des"] = float(np.sin(params["thetaF1_des"]))
    params["a2_des"] = float(np.cos(params["thetaF2_des"]))
    params["b2_des"] = float(np.sin(params["thetaF2_des"]))

    # Initial previous step rotation F_0 from YAML.
    if "thetaF1_0" not in params or "thetaF2_0" not in params:
        raise KeyError(
            "Missing thetaF initial values. Expected 'thetaF1_0' and "
            "'thetaF2_0' from the YAML-derived params."
        )

    # Bounds and regularization.
    params.setdefault("lambda_scale", 1.0)
    params.setdefault("alpha_lam", 0.0)

    if "max_step_angle" not in params:
        if "theta_step_max" in params:
            params["max_step_angle"] = float(params["theta_step_max"])
        else:
            params["max_step_angle"] = float(np.deg2rad(params.get("max_step_angle_deg", 20.0)))

    params["a1_min"] = float(np.cos(params["max_step_angle"]))
    params["a2_min"] = float(np.cos(params["max_step_angle"]))

    # Optional MPC boundary data from numerical simulation.
    #
    # Expected:
    #   c1_current, s1_current, c2_current, s2_current
    #   a1_prev, b1_prev, a2_prev, b2_prev
    #
    # Or directly:
    #   c1_1, s1_1, c2_1, s2_1
    #   a1_0, b1_0, a2_0, b2_0
    if mpc_initial is not None:
        params.update(mpc_initial)

    # If mpc_initial uses names from convert_state_to_sdp_initial_scalars,
    # convert them into SDP boundary names.
    if "c1_0" in params and "a1_prev" in params and "c1_1" not in params:
        # Here c1_0 from the simulation conversion means "current".
        # Rename internally to avoid confusion.
        params["c1_current"] = float(params["c1_0"])
        params["s1_current"] = float(params["s1_0"])
        params["c2_current"] = float(params["c2_0"])
        params["s2_current"] = float(params["s2_0"])

    if "c1_current" not in params:
        params["c1_current"] = float(np.cos(params["thetaR1_0"]))
        params["s1_current"] = float(np.sin(params["thetaR1_0"]))
        params["c2_current"] = float(np.cos(params["thetaR2_0"]))
        params["s2_current"] = float(np.sin(params["thetaR2_0"]))

    if "a1_prev" not in params:
        params["a1_prev"] = float(np.cos(params["thetaF1_0"]))
        params["b1_prev"] = float(np.sin(params["thetaF1_0"]))
        params["a2_prev"] = float(np.cos(params["thetaF2_0"]))
        params["b2_prev"] = float(np.sin(params["thetaF2_0"]))

    attach_mapping_to_params(params)
    params["ids_remap"] = get_remapped_ids(params)

    return params


def create_output_dirs(prefix: str):
    for folder in ["data", "markdown", "figs", "logs"]:
        path = PROJECT_ROOT / folder / prefix
        path.mkdir(parents=True, exist_ok=True)


def build_polynomial_system(params: Dict[str, Any]):
    """
    Build NumPolySystem with constraints and objective.
    """
    N = int(params["N"])
    total_var_num = int(params["total_var_num"])

    ps = NumPolySystem(n_vars=total_var_num)
    idf = params["id"]

    def v(prefix, k):
        return ps.var(idf(prefix, k) - 1)

    eq_mask_sys = []

    # ------------------------------------------------------------
    # Initial/MPC boundary constraints:
    # fix R0, F0, R1.
    # ------------------------------------------------------------
    eqs, ineqs, eq_mask = get_init_constraints(
        v("c1", 0),
        v("s1", 0),
        v("c2", 0),
        v("s2", 0),
        v("a1", 0),
        v("b1", 0),
        v("a2", 0),
        v("b2", 0),
        v("c1", 1),
        v("s1", 1),
        v("c2", 1),
        v("s2", 1),
        params,
    )

    for eq in eqs:
        ps.add_eq(eq)
    for ineq in ineqs:
        ps.add_ineq(ineq)
    eq_mask_sys.extend(eq_mask)

    # ------------------------------------------------------------
    # Constraints over horizon.
    # ------------------------------------------------------------
    for k in range(N + 1):
        # R SO(2), k=0,...,N.
        eqs, ineqs, eq_mask = get_SO2_orthogonality_constraint_rotation_R(
            v("c1", k),
            v("s1", k),
            v("c2", k),
            v("s2", k),
            params,
        )
        for eq in eqs:
            ps.add_eq(eq)
        for ineq in ineqs:
            ps.add_ineq(ineq)
        eq_mask_sys.extend(eq_mask)

        if k < N:
            # F SO(2), k=0,...,N-1.
            eqs, ineqs, eq_mask = get_SO2_orthogonality_constraint_rotation_F(
                v("a1", k),
                v("b1", k),
                v("a2", k),
                v("b2", k),
                params,
            )
            for eq in eqs:
                ps.add_eq(eq)
            for ineq in ineqs:
                ps.add_ineq(ineq)
            eq_mask_sys.extend(eq_mask)

            # Step bounds.
            eqs, ineqs, eq_mask = get_step_angle_bound_constraint_link_1(
                v("a1", k),
                params,
            )
            for eq in eqs:
                ps.add_eq(eq)
            for ineq in ineqs:
                ps.add_ineq(ineq)
            eq_mask_sys.extend(eq_mask)

            eqs, ineqs, eq_mask = get_step_angle_bound_constraint_link_2(
                v("a2", k),
                params,
            )
            for eq in eqs:
                ps.add_eq(eq)
            for ineq in ineqs:
                ps.add_ineq(ineq)
            eq_mask_sys.extend(eq_mask)

        if 1 <= k <= N:
            # Kinematics R_k = R_{k-1} F_{k-1}.
            eqs, ineqs, eq_mask = get_rotational_kinematics_link1(
                v("c1", k - 1),
                v("s1", k - 1),
                v("c1", k),
                v("s1", k),
                v("a1", k - 1),
                v("b1", k - 1),
                params,
            )
            for eq in eqs:
                ps.add_eq(eq)
            for ineq in ineqs:
                ps.add_ineq(ineq)
            eq_mask_sys.extend(eq_mask)

            eqs, ineqs, eq_mask = get_rotational_kinematics_link2(
                v("c2", k - 1),
                v("s2", k - 1),
                v("c2", k),
                v("s2", k),
                v("a2", k - 1),
                v("b2", k - 1),
                params,
            )
            for eq in eqs:
                ps.add_eq(eq)
            for ineq in ineqs:
                ps.add_ineq(ineq)
            eq_mask_sys.extend(eq_mask)

        if 1 <= k < N:
            # Reduced translational dynamics.
            eqs, ineqs, eq_mask = get_translational_dynamics_link1(
                v("c1", k - 1),
                v("s1", k - 1),
                v("b1", k - 1),
                v("c1", k),
                v("s1", k),
                v("b1", k),
                v("lam0x", k),
                v("lam0y", k),
                v("lam12x", k),
                v("lam12y", k),
                params,
            )
            for eq in eqs:
                ps.add_eq(eq)
            for ineq in ineqs:
                ps.add_ineq(ineq)
            eq_mask_sys.extend(eq_mask)

            eqs, ineqs, eq_mask = get_translational_dynamics_link2(
                v("c1", k - 1),
                v("s1", k - 1),
                v("b1", k - 1),
                v("c1", k),
                v("s1", k),
                v("b1", k),
                v("c2", k - 1),
                v("s2", k - 1),
                v("b2", k - 1),
                v("c2", k),
                v("s2", k),
                v("b2", k),
                v("lam12x", k),
                v("lam12y", k),
                params,
            )
            for eq in eqs:
                ps.add_eq(eq)
            for ineq in ineqs:
                ps.add_ineq(ineq)
            eq_mask_sys.extend(eq_mask)

            # Reduced rotational dynamics.
            eqs, ineqs, eq_mask = get_rotational_dynamics_link1(
                v("b1", k - 1),
                v("b1", k),
                v("c1", k),
                v("s1", k),
                v("lam0x", k),
                v("lam0y", k),
                v("lam12x", k),
                v("lam12y", k),
                v("u", k),
                params,
            )
            for eq in eqs:
                ps.add_eq(eq)
            for ineq in ineqs:
                ps.add_ineq(ineq)
            eq_mask_sys.extend(eq_mask)

            eqs, ineqs, eq_mask = get_rotational_dynamics_link2(
                v("b2", k - 1),
                v("b2", k),
                v("c2", k),
                v("s2", k),
                v("lam12x", k),
                v("lam12y", k),
                v("u", k),
                params,
            )
            for eq in eqs:
                ps.add_eq(eq)
            for ineq in ineqs:
                ps.add_ineq(ineq)
            eq_mask_sys.extend(eq_mask)

            # Control and lambda bounds.
            eqs, ineqs, eq_mask = get_control_bounds(v("u", k), params)
            for eq in eqs:
                ps.add_eq(eq)
            for ineq in ineqs:
                ps.add_ineq(ineq)
            eq_mask_sys.extend(eq_mask)

            eqs, ineqs, eq_mask = get_lambda_bounds(
                v("lam0x", k),
                v("lam0y", k),
                v("lam12x", k),
                v("lam12y", k),
                params,
            )
            for eq in eqs:
                ps.add_eq(eq)
            for ineq in ineqs:
                ps.add_ineq(ineq)
            eq_mask_sys.extend(eq_mask)

    # Objective.
    ps.set_obj(build_objective(v, params))

    params["eq_mask_sys"] = eq_mask_sys

    return ps, params


def solve_sdp(
    params: Dict[str, Any],
    prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build and solve the reduced Acrobot SDP.

    Returns:
        result dictionary containing:
            params
            result
            res
            coeff_info
            aux_info
            solutions
            extracted_vectors
            gap_info
            first_control
    """
    total_start = time.time()

    params = complete_sdp_params(params)

    if prefix is None:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        prefix = f"Acrobot_SO2_Reduced_MPC/{current_time}/"

    create_output_dirs(prefix)

    log_path = PROJECT_ROOT / "logs" / prefix / "log.txt"

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("Acrobot SO(2) Reduced No-X/No-V SDP-MPC Optimization\n")
        f.write("=" * 80 + "\n")
        f.write("Thesis indexing: u_k, lambda_k for k=1,...,N-1.\n")
        f.write("MPC interpretation: current state is SDP node 1; first applied control is u_1.\n")
        f.write("=" * 80 + "\n")
        f.write("params:\n")
        for key in sorted(params.keys()):
            if callable(params[key]):
                continue
            f.write(f"{key}: {params[key]}\n")

    ps, params = build_polynomial_system(params)

    kappa = int(params["kappa"])
    total_var_num = int(params["total_var_num"])
    var_mapping = params["var_mapping"]

    ps.clean_all(tol=1e-14, if_scale=True, scale_obj=False)

    poly_data = ps.get_supp_rpt_data(kappa)

    print("Construction finished.")

    if params["cs_mode"] == "SELF":
        cliques = get_cliques_for_cstss(int(params["N"]), params)
    else:
        cliques = []

    params["cliques"] = cliques

    start_time = time.time()

    result, res, coeff_info, aux_info = CSTSS_pybind(
        poly_data,
        kappa,
        total_var_num,
        params,
    )

    elapsed = time.time() - start_time

    aux_info["result"] = result
    params["aux_info"] = aux_info

    with open(log_path, "a", encoding="utf-8") as f:
        result_str = str(result) if isinstance(result, list) else f"{float(np.asarray(result).squeeze()):.20f}"
        f.write(
            f"\nResult={result_str}, operation time={elapsed:.5f}, "
            f"mosek time={aux_info.get('mosek_time', 0):.5f}\n"
        )

    # Clique rank ordering for ordered_extract_CS.
    if "cliques" in aux_info and aux_info["cliques"]:
        cliques_remapped = []
        averages = []

        for clique in aux_info["cliques"]:
            remapped = [params["ids_remap"][i - 1] for i in clique]
            cliques_remapped.append(remapped)
            averages.append(np.mean(remapped))

        params["cliques_rank"] = np.argsort(averages)
    else:
        params["cliques_rank"] = []

    # Markdown visualization.
    try:
        clique_supp_list = []
        clique_coeff_list = []
        kappa_width = 2 * kappa

        if "cliques" in aux_info and aux_info["cliques"]:
            for ii in params["cliques_rank"]:
                sorted_vars = sorted(aux_info["cliques"][ii])
                supp = np.zeros((len(sorted_vars), kappa_width), dtype=np.float64)
                for idx_v, j in enumerate(sorted_vars):
                    supp[idx_v, -1] = j
                clique_supp_list.append(supp)
                clique_coeff_list.append(np.ones(len(sorted_vars)))

        md_path = PROJECT_ROOT / "markdown" / prefix / "opt_problem.md"

        with open(md_path, "w", encoding="utf-8") as md:
            md.write("equality constraints:\n")
            numpoly_visualize(aux_info["supp_rpt_h"], aux_info["coeff_h"], var_mapping, md)

            md.write("\ninequality constraints:\n")
            numpoly_visualize(aux_info["supp_rpt_g"], aux_info["coeff_g"], var_mapping, md)

            md.write("\nobjective:\n")
            numpoly_visualize([aux_info["supp_rpt_f"]], [aux_info["coeff_f"]], var_mapping, md)

            md.write("\ncliques:\n")
            numpoly_visualize(clique_supp_list, clique_coeff_list, var_mapping, md)

    except Exception as exc:
        print(f"Markdown visualization skipped: {exc}")

    params_to_save = {k: v for k, v in params.items() if not callable(v)}

    with open(PROJECT_ROOT / "data" / prefix / "params.pkl", "wb") as f:
        pickle.dump(params_to_save, f)

    with open(PROJECT_ROOT / "data" / prefix / "res.pkl", "wb") as f:
        pickle.dump(
            {
                "result": result,
                "res": res,
                "coeff_info": coeff_info,
                "aux_info": aux_info,
            },
            f,
        )

    if not params.get("if_solve", True):
        total_time = time.time() - total_start
        print("Debugging mode: problem constructed, but if_solve=False.")
        print(f"Total time: {total_time:.5f} s")

        return {
            "params": params,
            "result": result,
            "res": res,
            "coeff_info": coeff_info,
            "aux_info": aux_info,
            "solutions": {},
            "extracted_vectors": {},
            "gap_info": {},
            "first_control": None,
            "prefix": prefix,
            "Xs": [],
        }

    # Extraction.
    if params["relax_mode"] == "MOMENT":
        Xs = res["Xopt"]
    elif params["relax_mode"] == "SOS":
        Xs = [-S for S in res["Sopt"]]
    else:
        Xs = res.get("Xopt", [])

    ts_info = aux_info["ts_info"]
    cliques_aux = aux_info["cliques"]
    mon_rpt = aux_info["mon_rpt"]

    mom_mat_num = sum(len(ts_info[i]) for i in range(len(cliques_aux)))
    mom_mat_rpt = [None] * mom_mat_num

    idx = 0
    for i in range(len(cliques_aux)):
        for j in range(len(ts_info[i])):
            rpt = mon_rpt[i][ts_info[i][j], :]
            rpt = np.hstack([np.zeros_like(rpt), rpt])
            mom_mat_rpt[idx] = rpt
            idx += 1

    extracted_vectors = {}
    solutions = {}
    extraction_info = {}

    v_opt_naive, output_info_naive = naive_extract(Xs, mon_rpt, ts_info, total_var_num)
    extracted_vectors["naive"] = v_opt_naive
    extraction_info["naive"] = output_info_naive
    solutions["naive"] = extract_solution_variables(v_opt_naive, params)

    with open(PROJECT_ROOT / "data" / prefix / "v_opt_naive.pkl", "wb") as f:
        pickle.dump(v_opt_naive, f)

    if params["ts_mode"] == "NON":
        v_opt_robust, output_info_robust = robust_extract_CS(
            Xs,
            mom_mat_rpt,
            total_var_num,
            1e-2,
        )
        extracted_vectors["robust"] = v_opt_robust
        extraction_info["robust"] = output_info_robust
        solutions["robust"] = extract_solution_variables(v_opt_robust, params)

        with open(PROJECT_ROOT / "data" / prefix / "v_opt_robust.pkl", "wb") as f:
            pickle.dump(v_opt_robust, f)

        v_opt_ordered, output_info_ordered = ordered_extract_CS(
            Xs,
            mom_mat_rpt,
            total_var_num,
            1e-2,
            params.get("cliques_rank", []),
        )
        extracted_vectors["ordered"] = v_opt_ordered
        extraction_info["ordered"] = output_info_ordered
        solutions["ordered"] = extract_solution_variables(v_opt_ordered, params)

        with open(PROJECT_ROOT / "data" / prefix / "v_opt_ordered.pkl", "wb") as f:
            pickle.dump(v_opt_ordered, f)

    gap_info = build_gap_info(result, extracted_vectors, params)

    preferred = "ordered" if "ordered" in solutions else "robust" if "robust" in solutions else "naive"
    first_control = get_first_mpc_control(solutions[preferred])

    errors_by_method = {
        name: compute_SO2_errors(sol, int(params["N"]))
        for name, sol in solutions.items()
    }

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("EXTRACTION SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"preferred extraction: {preferred}\n")
        f.write(f"first MPC control u_1: {first_control:+.12e}\n")

        for name, gap in gap_info.items():
            f.write(f"\n{name.upper()} gap:\n")
            for key, val in gap.items():
                f.write(f"  {key}: {val:+.12e}\n")

        for name, err in errors_by_method.items():
            f.write(f"\n{name.upper()} SO(2) max errors:\n")
            for key, vals in err.items():
                if vals:
                    f.write(f"  {key}: {max(vals):.12e}\n")

        f.write("\nPredicted trajectory summary:\n")
        sol = solutions[preferred]
        for k in range(int(params["N"]) + 1):
            f.write(
                f"k={k:3d}: "
                f"thetaR1={sol['thetaR1'][k]:+.8f}, "
                f"thetaR2={sol['thetaR2'][k]:+.8f}"
            )
            if 1 <= k < int(params["N"]):
                f.write(f", u={sol['u'][k]:+.8f}")
            f.write("\n")

    total_time = time.time() - total_start

    print("\n" + "=" * 80)
    print("SDP SOLVE FINISHED")
    print("=" * 80)
    print(f"preferred extraction: {preferred}")
    print(f"first MPC control u_1: {first_control:+.12e}")
    print(f"total time: {total_time:.3f} s")
    print("=" * 80)

    return {
        "params": params,
        "result": result,
        "res": res,
        "coeff_info": coeff_info,
        "aux_info": aux_info,
        "solutions": solutions,
        "extracted_vectors": extracted_vectors,
        "extraction_info": extraction_info,
        "gap_info": gap_info,
        "errors_by_method": errors_by_method,
        "first_control": first_control,
        "preferred_extraction": preferred,
        "prefix": prefix,
        # These are the actual clique moment matrices used by the extraction
        # routines above.  Logging code saves them in compressed NumPy form.
        "Xs": Xs,
    }


def solve_from_yaml(
    yaml_path: str | Path = PROJECT_ROOT / "config" / "acrobot_physical.yaml",
    mpc_initial: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Load YAML config, build params, optionally insert MPC initial state, solve SDP.
    """
    cfg = load_yaml_config(yaml_path)
    params = build_common_params(cfg)

    if mpc_initial is not None:
        params.update(mpc_initial)

    return solve_sdp(params)


if __name__ == "__main__":
    solve_from_yaml()
