from __future__ import annotations

from typing import Any, Dict

import numpy as np

from SDP.constraints import reconstruct_positions_from_cs
from SDP.objective import evaluate_objective_from_vector


def extract_solution_variables(v_opt, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract reduced SDP variables into structured dictionaries.

    lambda and u exist only for k = 1,...,N-1.
    """
    N = int(params["N"])
    idf = params["id"]

    sol = {
        "x1": {},
        "x2": {},
        "R1": {},
        "R2": {},
        "F1": {},
        "F2": {},
        "lambda0": {},
        "lambda12": {},
        "u": {},
        "thetaR1": {},
        "thetaR2": {},
        "step_theta1": {},
        "step_theta2": {},
    }

    for k in range(N + 1):
        c1 = float(v_opt[idf("c1", k) - 1])
        s1 = float(v_opt[idf("s1", k) - 1])
        c2 = float(v_opt[idf("c2", k) - 1])
        s2 = float(v_opt[idf("s2", k) - 1])

        x1_obj, x2_obj = reconstruct_positions_from_cs(c1, s1, c2, s2, params)

        sol["x1"][k] = np.array(x1_obj, dtype=float)
        sol["x2"][k] = np.array(x2_obj, dtype=float)

        sol["R1"][k] = np.array([[c1, -s1], [s1, c1]], dtype=float)
        sol["R2"][k] = np.array([[c2, -s2], [s2, c2]], dtype=float)

        sol["thetaR1"][k] = float(np.arctan2(s1, c1))
        sol["thetaR2"][k] = float(np.arctan2(s2, c2))

        if k < N:
            a1 = float(v_opt[idf("a1", k) - 1])
            b1 = float(v_opt[idf("b1", k) - 1])
            a2 = float(v_opt[idf("a2", k) - 1])
            b2 = float(v_opt[idf("b2", k) - 1])

            sol["F1"][k] = np.array([[a1, -b1], [b1, a1]], dtype=float)
            sol["F2"][k] = np.array([[a2, -b2], [b2, a2]], dtype=float)

            sol["step_theta1"][k] = float(np.arctan2(b1, a1))
            sol["step_theta2"][k] = float(np.arctan2(b2, a2))

        if 1 <= k < N:
            sol["lambda0"][k] = np.array(
                [
                    float(v_opt[idf("lam0x", k) - 1]),
                    float(v_opt[idf("lam0y", k) - 1]),
                ],
                dtype=float,
            )

            sol["lambda12"][k] = np.array(
                [
                    float(v_opt[idf("lam12x", k) - 1]),
                    float(v_opt[idf("lam12y", k) - 1]),
                ],
                dtype=float,
            )

            sol["u"][k] = float(v_opt[idf("u", k) - 1])

    return sol


def get_first_mpc_control(solution: Dict[str, Any]) -> float:
    """
    In thesis indexing, the first meaningful MPC control is u_1.

    This is the value applied in the numerical simulator over the next
    control interval.
    """
    if 1 not in solution["u"]:
        raise KeyError("Solution has no u[1]. Check that N >= 2 and extraction succeeded.")

    return float(solution["u"][1])


def extract_sdp_initial_for_next_mpc(solution: Dict[str, Any]) -> Dict[str, float]:
    """
    If you want to use the SDP-predicted next state directly, this prepares
    the next MPC boundary data.

    Usually, for closed-loop MPC, prefer the numerical simulation output instead.
    Humanity already suffers enough without feeding relaxed predictions as reality.
    """
    R1_cur = solution["R1"][2]
    R2_cur = solution["R2"][2]

    F1_prev = solution["F1"][1]
    F2_prev = solution["F2"][1]

    c1_current = float(R1_cur[0, 0])
    s1_current = float(R1_cur[1, 0])
    c2_current = float(R2_cur[0, 0])
    s2_current = float(R2_cur[1, 0])

    a1_prev = float(F1_prev[0, 0])
    b1_prev = float(F1_prev[1, 0])
    a2_prev = float(F2_prev[0, 0])
    b2_prev = float(F2_prev[1, 0])

    return {
        "c1_current": c1_current,
        "s1_current": s1_current,
        "c2_current": c2_current,
        "s2_current": s2_current,
        "a1_prev": a1_prev,
        "b1_prev": b1_prev,
        "a2_prev": a2_prev,
        "b2_prev": b2_prev,
    }


def compute_SO2_errors(solution: Dict[str, Any], N: int) -> Dict[str, list]:
    errors = {
        "R1": [],
        "R2": [],
        "F1": [],
        "F2": [],
    }

    I = np.eye(2)

    for k in range(N + 1):
        R1 = solution["R1"][k]
        R2 = solution["R2"][k]

        errors["R1"].append(float(np.linalg.norm(R1.T @ R1 - I, ord="fro")))
        errors["R2"].append(float(np.linalg.norm(R2.T @ R2 - I, ord="fro")))

        if k < N:
            F1 = solution["F1"][k]
            F2 = solution["F2"][k]

            errors["F1"].append(float(np.linalg.norm(F1.T @ F1 - I, ord="fro")))
            errors["F2"].append(float(np.linalg.norm(F2.T @ F2 - I, ord="fro")))

    return errors


def build_gap_info(result, extracted_vectors: Dict[str, np.ndarray], params: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    sdp_lower_bound = float(np.asarray(result).squeeze())
    gap_info = {}

    for name, v_extracted in extracted_vectors.items():
        extracted_obj = evaluate_objective_from_vector(v_extracted, params)

        absolute_gap = extracted_obj - sdp_lower_bound
        relative_gap = absolute_gap / max(1.0, abs(extracted_obj))

        gap_info[name] = {
            "sdp_lower_bound": sdp_lower_bound,
            "extracted_objective": extracted_obj,
            "absolute_gap": absolute_gap,
            "relative_gap": relative_gap,
        }

    return gap_info