import os
import datetime
import numpy as np
import pickle
import time

import sys
# Add the parent directory to sys.path so Python can find the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from SPOT.PYTHON.CSTSS_pybind import CSTSS_pybind
from SPOT.PYTHON.numpoly import NumPolySystem, NumPolyExpr, numpoly_visualize
from SPOT.PYTHON.naive_extract import naive_extract
from SPOT.PYTHON.robust_extract_CS import robust_extract_CS, ordered_extract_CS

###############################
# Helper Functions and Stubs  #
###############################

# Variable Groups (defined by timestep index in get_var_mapping_and_dict):
# - c_{i,k}, s_{i,k}: k=0..N, i=1,2 - SO(2) rotations for each link
# - a_{i,k}, b_{i,k}: k=0..N-1, i=1,2 - discrete SO(2) step rotations
# - lambda_{0,k}: k=1..N-1 - 2 components (constraint forces at pivot)
# - lambda_{12,k}: k=1..N-1 - 2 components (constraint forces between links)
# - u_k: k=1..N-1 - scalar control input
#
# Reduced x-free/v-free model:
#   x and v are not decision variables anymore.
#   COM positions are reconstructed from (s,c), with the thesis convention
#       R = [[c, -s], [s, c]],   x = l sin(theta),   y = -l cos(theta).
#   Translational velocities use the small-step approximation theta_dot ~= b/h,
#   where F_k = [[a_k, -b_k], [b_k, a_k]].

# k = 0,...,N
LONG_PREFIXES = ["c1", "s1", "c2", "s2"]
# k = 0,...,N-1
MID_PREFIXES = ["a1", "b1", "a2", "b2"]
# k = 1,...,N-1
SHORT_PREFIXES = ["lam0x", "lam0y", "lam12x", "lam12y", "u"]


def get_var_mapping_and_dict(N):
    """Create scalar variables for the reduced x-free/v-free SO(2) Acrobot POP."""
    var_start_dict = {}
    var_mapping = {}
    cnt = 1
    long_list = list(range(N + 1))  # R_k: k=0..N
    mid_list = list(range(N))       # F_k: k=0..N-1, interval k -> k+1
    short_list = list(range(1, N))  # lambda_k, u_k: k=1..N-1

    fmt = {
        # Rotation variables R_i in SO(2)
        "c1": "c_{{1,{k}}}", "s1": "s_{{1,{k}}}",
        "c2": "c_{{2,{k}}}", "s2": "s_{{2,{k}}}",

        # Relative rotation variables F_i in SO(2)
        "a1": "a_{{1,{k}}}", "b1": "b_{{1,{k}}}",
        "a2": "a_{{2,{k}}}", "b2": "b_{{2,{k}}}",

        # Scaled Lagrange multiplier variables and scalar control input
        "lam0x": "\\lambda_{{0,x,{k}}}", "lam0y": "\\lambda_{{0,y,{k}}}",
        "lam12x": "\\lambda_{{12,x,{k}}}", "lam12y": "\\lambda_{{12,y,{k}}}",
        "u": "u_{{{k}}}",
    }

    ordered_prefixes = [
        ("c1", long_list), ("s1", long_list),
        ("c2", long_list), ("s2", long_list),
        ("a1", mid_list), ("b1", mid_list),
        ("a2", mid_list), ("b2", mid_list),
        ("lam0x", short_list), ("lam0y", short_list),
        ("lam12x", short_list), ("lam12y", short_list),
        ("u", short_list),
    ]

    for prefix, klist in ordered_prefixes:
        var_start_dict[prefix] = cnt
        for k in klist:
            var_mapping[cnt] = fmt[prefix].format(k=k)
            cnt += 1

    total_var_num = cnt - 1
    var_start_dict["N"] = N
    return var_mapping, var_start_dict, total_var_num

def get_id(prefix, k, var_start_dict, prefix_k0):
    return var_start_dict[prefix] + (k - prefix_k0[prefix])


def get_remapped_ids(params):
    """Remap reduced variables into a timestep-grouped ordering."""
    N = params['N']
    total_var_num = params['total_var_num']
    id_func = params['id']

    ids_remap = np.zeros(total_var_num, dtype=int)
    idx = 1

    for k in range(N + 1):
        ids_remap[id_func("c1", k) - 1] = idx; idx += 1
        ids_remap[id_func("s1", k) - 1] = idx; idx += 1
        ids_remap[id_func("c2", k) - 1] = idx; idx += 1
        ids_remap[id_func("s2", k) - 1] = idx; idx += 1

        if k < N:
            ids_remap[id_func("a1", k) - 1] = idx; idx += 1
            ids_remap[id_func("b1", k) - 1] = idx; idx += 1
            ids_remap[id_func("a2", k) - 1] = idx; idx += 1
            ids_remap[id_func("b2", k) - 1] = idx; idx += 1

        if 1 <= k <= N - 1:
            ids_remap[id_func("lam0x", k) - 1] = idx; idx += 1
            ids_remap[id_func("lam0y", k) - 1] = idx; idx += 1
            ids_remap[id_func("lam12x", k) - 1] = idx; idx += 1
            ids_remap[id_func("lam12y", k) - 1] = idx; idx += 1
            ids_remap[id_func("u", k) - 1] = idx; idx += 1

    return ids_remap

def get_init_constraints(c1_0, s1_0, c2_0, s2_0,
                         a1_0, b1_0, a2_0, b2_0,
                         params):
    """
    Initial constraints for the reduced SO(2) Acrobot.

    No x variables are present. The initial COM positions are implied by
    R_0 and the fixed thesis geometry:
        x1 = p0 + [l1/2*s1, -l1/2*c1]
        x2 = p0 + [l1*s1 + l2/2*s2, -l1*c1 - l2/2*c2].

    A rest start is imposed by F_0 = I, i.e. a_i,0=1 and b_i,0=0.
    """
    eqs = [
        # Absolute link orientations at k=0
        c1_0 - np.cos(params['thetaR1_0']),
        s1_0 - np.sin(params['thetaR1_0']),
        c2_0 - np.cos(params['thetaR2_0']),
        s2_0 - np.sin(params['thetaR2_0']),

        # Initial relative rotations F_0. F_0=I gives zero initial angular velocity.
        a1_0 - np.cos(params['thetaF1_0']),
        b1_0 - np.sin(params['thetaF1_0']),
        a2_0 - np.cos(params['thetaF2_0']),
        b2_0 - np.sin(params['thetaF2_0']),

        # SO(2) constraints at k=0
        c1_0**2 + s1_0**2 - 1,
        c2_0**2 + s2_0**2 - 1,
        a1_0**2 + b1_0**2 - 1,
        a2_0**2 + b2_0**2 - 1,
    ]
    eq_mask = [1] * 8 + [0] * 2 + [1] * 2
    return eqs, [], eq_mask

def get_initial_rest_position_constraints(*args, **kwargs):
    """
    Deprecated in the reduced no-x/no-v formulation.
    Rest start is imposed by F_0 = I in get_init_constraints, i.e. b_{i,0}=0.
    """
    return [], [], []

def get_translational_dynamics_link1(c1_km1, s1_km1, b1_km1,
                                     c1_k, s1_k, b1_k,
                                     lam0x_k, lam0y_k,
                                     lam12x_k, lam12y_k,
                                     params):
    """
    Reduced x-free/v-free translational DEL for link 1, scaled by h^2.

    Thesis sign convention:
        R = [[c, -s], [s, c]],  x = l sin(theta),  y = -l cos(theta).
    Small-step velocity approximation on interval k:
        theta_dot_{i,k} ~= b_{i,k}/h.

    Link-1 COM velocity approximation:
        v1_k = (l1/(2h)) [c1_k*b1_k, s1_k*b1_k]^T.

    Constraint implemented:
        (m1*l1/2) ([c*b, s*b]_k - [c*b, s*b]_{k-1})
        + h^2 m1 g e2 - h^2 s_lambda (lambda0_k + lambda12_k) = 0.
    """
    h = params["dt"]
    m1 = params["m1"]
    l1 = params["l1"]
    g = params["g"]
    lam_scale = params["lambda_scale"]

    eqs = [
        (m1 * l1 / 2.0) * (c1_k * b1_k - c1_km1 * b1_km1)
        - h**2 * lam_scale * lam0x_k
        - h**2 * lam_scale * lam12x_k,

        (m1 * l1 / 2.0) * (s1_k * b1_k - s1_km1 * b1_km1)
        + h**2 * m1 * g
        - h**2 * lam_scale * lam0y_k
        - h**2 * lam_scale * lam12y_k,
    ]
    return eqs, [], [1, 1]

def get_translational_dynamics_link2(c1_km1, s1_km1, b1_km1,
                                     c1_k, s1_k, b1_k,
                                     c2_km1, s2_km1, b2_km1,
                                     c2_k, s2_k, b2_k,
                                     lam12x_k, lam12y_k,
                                     params):
    """Reduced x-free/v-free translational DEL for link 2, scaled by h^2."""
    h = params["dt"]
    m2 = params["m2"]
    l1 = params["l1"]
    l2 = params["l2"]
    g = params["g"]
    lam_scale = params["lambda_scale"]

    link1_x = c1_k * b1_k - c1_km1 * b1_km1
    link1_y = s1_k * b1_k - s1_km1 * b1_km1
    link2_x = c2_k * b2_k - c2_km1 * b2_km1
    link2_y = s2_k * b2_k - s2_km1 * b2_km1

    eqs = [
        m2 * (l1 * link1_x + (l2 / 2.0) * link2_x)
        + h**2 * lam_scale * lam12x_k,

        m2 * (l1 * link1_y + (l2 / 2.0) * link2_y)
        + h**2 * m2 * g
        + h**2 * lam_scale * lam12y_k,
    ]
    return eqs, [], [1, 1]

def reconstruct_positions_from_cs(c1_k, s1_k, c2_k, s2_k, params):
    """
    Reconstruct COM positions from rotations, with the thesis convention
        x = l sin(theta), y = -l cos(theta).
    Works for both numeric scalars and NumPoly expressions.
    """
    p0 = params["p_0"]
    l1 = params["l1"]
    l2 = params["l2"]

    x1 = np.array([
        p0[0] + (l1 / 2.0) * s1_k,
        p0[1] - (l1 / 2.0) * c1_k,
    ], dtype=object)

    x2 = np.array([
        p0[0] + l1 * s1_k + (l2 / 2.0) * s2_k,
        p0[1] - l1 * c1_k - (l2 / 2.0) * c2_k,
    ], dtype=object)
    return x1, x2


def get_holonomic_constraint(*args, **kwargs):
    """
    In the reduced model there are no holonomic constraints left to add:
    x1 and x2 have been substituted by reconstruct_positions_from_cs.
    Returning no constraints intentionally avoids adding decorative 0=0 equations.
    """
    return [], [], []

def get_rotational_kinematics_link1(c1_km1, s1_km1,
                                    c1_k, s1_k,
                                    a1_km1, b1_km1,
                                    params):
    eqs = [
        c1_k - c1_km1 * a1_km1 + s1_km1 * b1_km1,
        s1_k - s1_km1 * a1_km1 - c1_km1 * b1_km1,
    ]

    ineqs = []
    eq_mask = [1, 1]

    return eqs, ineqs, eq_mask


def get_rotational_kinematics_link2(c2_km1, s2_km1,
                                    c2_k, s2_k,
                                    a2_km1, b2_km1,
                                    params):
    eqs = [
        c2_k - c2_km1 * a2_km1 + s2_km1 * b2_km1,
        s2_k - s2_km1 * a2_km1 - c2_km1 * b2_km1,
    ]

    ineqs = []
    eq_mask = [1, 1]

    return eqs, ineqs, eq_mask


def get_SO2_orthogonality_constraint_rotation_R(c1_k, s1_k, c2_k, s2_k, params):
    eqs = [
        c1_k**2 + s1_k**2 - 1,
        c2_k**2 + s2_k**2 - 1,
    ]

    ineqs = []
    eq_mask = [0, 0]

    return eqs, ineqs, eq_mask

def get_SO2_orthogonality_constraint_rotation_F(a1_k, b1_k, a2_k, b2_k, params):
    eqs = [
        a1_k**2 + b1_k**2 - 1,
        a2_k**2 + b2_k**2 - 1,
    ]

    ineqs = []
    eq_mask = [1, 1]

    return eqs, ineqs, eq_mask

def get_step_angle_bound_constraint_link_1(a1_k, params):

    a1_min = params["a1_min"]

    ineqs = [
        a1_k - a1_min,
    ]

    eqs = []
    eq_mask = []

    return eqs, ineqs, eq_mask

def get_step_angle_bound_constraint_link_2(a2_k, params):

    a2_min = params["a2_min"]

    ineqs = [
        a2_k - a2_min,
    ]

    eqs = []
    eq_mask = []

    return eqs, ineqs, eq_mask

def get_rotational_dynamics_link1(b1_km1, b1_k,
                                  c1_k, s1_k,
                                  lam0x_k, lam0y_k,
                                  lam12x_k, lam12y_k,
                                  u_k,
                                  params):

    h = params["dt"]

    delta_11 = params["delta_11"]
    delta_12 = params["delta_12"]

    rho_10 = params["rho_10"]
    rho_112 = params["rho_112"]

    rho_10_x = rho_10[0]
    rho_10_y = rho_10[1]

    rho_112_x = rho_112[0]
    rho_112_y = rho_112[1]

    # R_1,k^T lambda_0,k
    # R^T = [[c, s], [-s, c]]
    Rt_lam0_x = c1_k * lam0x_k * params["lambda_scale"] + s1_k * lam0y_k * params["lambda_scale"]
    Rt_lam0_y = -s1_k * lam0x_k * params["lambda_scale"] + c1_k * lam0y_k * params["lambda_scale"]

    # R_1,k^T lambda_12,k
    Rt_lam12_x = c1_k * lam12x_k * params["lambda_scale"] + s1_k * lam12y_k * params["lambda_scale"]
    Rt_lam12_y = -s1_k * lam12x_k * params["lambda_scale"] + c1_k * lam12y_k * params["lambda_scale"]

    # Planar cross product rho x R^T lambda
    mu_10 = rho_10_x * Rt_lam0_y - rho_10_y * Rt_lam0_x
    mu_112 = rho_112_x * Rt_lam12_y - rho_112_y * Rt_lam12_x

    eqs = [
        (delta_11 + delta_12) * (b1_km1 - b1_k)
        + h**2 * (mu_10 + mu_112 - u_k)
    ]

    ineqs = []

    # This is bilinear because of c*lambda and s*lambda terms.
    # Use [1] if you want to follow Kang's kinematic/dynamics mask style.
    eq_mask = [1]

    return eqs, ineqs, eq_mask


def get_rotational_dynamics_link2(b2_km1, b2_k,
                                  c2_k, s2_k,
                                  lam12x_k, lam12y_k,
                                  u_k,
                                  params):

    h = params["dt"]

    delta_21 = params["delta_21"]
    delta_22 = params["delta_22"]

    rho_212 = params["rho_212"]

    rho_212_x = rho_212[0]
    rho_212_y = rho_212[1]

    # R_2,k^T lambda_12,k
    # R^T = [[c, s], [-s, c]]
    Rt_lam12_x = c2_k * lam12x_k * params["lambda_scale"] + s2_k * lam12y_k * params["lambda_scale"]
    Rt_lam12_y = -s2_k * lam12x_k * params["lambda_scale"] + c2_k * lam12y_k * params["lambda_scale"]

    # Planar cross product rho x R^T lambda
    mu_212 = rho_212_x * Rt_lam12_y - rho_212_y * Rt_lam12_x

    eqs = [
        (delta_21 + delta_22) * (b2_km1 - b2_k)
        + h**2 * (u_k - mu_212)
    ]

    ineqs = []

    # This is bilinear because of c*lambda and s*lambda terms.
    # Use [1] if following Kang's kinematic/dynamics mask style.
    eq_mask = [1]

    return eqs, ineqs, eq_mask


def get_control_bounds(u_k, params):

    u_max = params["u_max"]

    ineqs = [
        u_max**2 - u_k**2
    ]

    eqs = []
    eq_mask = []

    return eqs, ineqs, eq_mask


def get_lambda_bounds(lam0x_k, lam0y_k, lam12x_k, lam12y_k, params):
    L = params["lambda_max"]

    ineqs = [
        L**2 - lam0x_k**2,
        L**2 - lam0y_k**2,
        L**2 - lam12x_k**2,
        L**2 - lam12y_k**2,
    ]

    return [], ineqs, []

def get_cliques_for_cstss(N, params):
    """
    Conservative 21-variable SELF cliques for the reduced x-free/v-free Acrobot.

    Interior clique I_k, k=1,...,N-1:
        {R_{1,k-1}, R_{2,k-1}, R_{1,k}, R_{2,k},
         F_{1,k-1}, F_{2,k-1}, F_{1,k}, F_{2,k},
         lambda_{0,k}, lambda_{12,k}, u_k}

    with R_i=(c_i,s_i), F_i=(a_i,b_i), lambda_0,lambda_12 in R^2.
    Size: 4 + 4 + 4 + 4 + 4 + 1 = 21.

    A small terminal clique is added to cover R_N, F_{N-1}, terminal objective,
    and the final kinematics R_N = R_{N-1}F_{N-1}. The largest clique stays 21.
    """
    id_func = params['id']
    cliques = []

    def vid(prefix, k):
        return id_func(prefix, k)

    def dedupe_keep_order(items):
        out, seen = [], set()
        for item in items:
            if item not in seen:
                out.append(item)
                seen.add(item)
        return out

    for k in range(1, N):
        clique = [
            # R_{1,k-1}, R_{2,k-1}
            vid("c1", k - 1), vid("s1", k - 1),
            vid("c2", k - 1), vid("s2", k - 1),

            # R_{1,k}, R_{2,k}
            vid("c1", k), vid("s1", k),
            vid("c2", k), vid("s2", k),

            # F_{1,k-1}, F_{2,k-1}
            vid("a1", k - 1), vid("b1", k - 1),
            vid("a2", k - 1), vid("b2", k - 1),

            # F_{1,k}, F_{2,k}
            vid("a1", k), vid("b1", k),
            vid("a2", k), vid("b2", k),

            # multipliers and control at k
            vid("lam0x", k), vid("lam0y", k),
            vid("lam12x", k), vid("lam12y", k),
            vid("u", k),
        ]
        cliques.append(dedupe_keep_order(clique))

    # terminal / kinematics endpoint clique for R_N and F_{N-1}
    if N >= 1:
        terminal = [
            vid("c1", N - 1), vid("s1", N - 1),
            vid("c2", N - 1), vid("s2", N - 1),
            vid("c1", N), vid("s1", N),
            vid("c2", N), vid("s2", N),
            vid("a1", N - 1), vid("b1", N - 1),
            vid("a2", N - 1), vid("b2", N - 1),
        ]
        cliques.append(dedupe_keep_order(terminal))

    unique_cliques, seen = [], set()
    for clique in cliques:
        key = tuple(clique)
        if key not in seen:
            unique_cliques.append(clique)
            seen.add(key)

    sizes = [len(c) for c in unique_cliques]
    print("\n" + "=" * 80)
    print("SELF CLIQUE DEBUG: reduced conservative 21-variable cliques")
    print("=" * 80)
    print(f"number of SELF cliques: {len(unique_cliques)}")
    if sizes:
        print(f"min clique size:        {min(sizes)}")
        print(f"max clique size:        {max(sizes)}")
        print(f"mean clique size:       {np.mean(sizes):.2f}")
        print(f"largest clique sizes:   {sorted(sizes)[-20:]}")
    else:
        print("No SELF cliques created. This happens if N < 1.")
    print("=" * 80 + "\n")

    return unique_cliques

def extract_solution_variables(v_opt, var_start_dict, N, id_func, params=None):
    """Extract reduced variables and reconstruct x1/x2 from c,s."""
    solution_dict = {
        "x1": {},
        "x2": {},
        "R1": {},
        "R2": {},
        "F1": {},
        "F2": {},
        "lambda0": {},
        "lambda12": {},
        "u": {},
        "theta1": {},
        "theta2": {},
        "step_theta1": {},
        "step_theta2": {},
    }

    for k in range(N + 1):
        c1 = v_opt[id_func("c1", k) - 1]
        s1 = v_opt[id_func("s1", k) - 1]
        c2 = v_opt[id_func("c2", k) - 1]
        s2 = v_opt[id_func("s2", k) - 1]

        if params is not None:
            x1_obj, x2_obj = reconstruct_positions_from_cs(c1, s1, c2, s2, params)
            x1 = np.array(x1_obj, dtype=float)
            x2 = np.array(x2_obj, dtype=float)
        else:
            x1 = np.array([np.nan, np.nan])
            x2 = np.array([np.nan, np.nan])

        solution_dict["x1"][k] = x1
        solution_dict["x2"][k] = x2
        solution_dict["R1"][k] = np.array([[c1, -s1], [s1, c1]])
        solution_dict["R2"][k] = np.array([[c2, -s2], [s2, c2]])
        solution_dict["theta1"][k] = np.arctan2(s1, c1)
        solution_dict["theta2"][k] = np.arctan2(s2, c2)

        if k < N:
            a1 = v_opt[id_func("a1", k) - 1]
            b1 = v_opt[id_func("b1", k) - 1]
            a2 = v_opt[id_func("a2", k) - 1]
            b2 = v_opt[id_func("b2", k) - 1]

            solution_dict["F1"][k] = np.array([[a1, -b1], [b1, a1]])
            solution_dict["F2"][k] = np.array([[a2, -b2], [b2, a2]])
            solution_dict["step_theta1"][k] = np.arctan2(b1, a1)
            solution_dict["step_theta2"][k] = np.arctan2(b2, a2)

        if 1 <= k < N:
            solution_dict["lambda0"][k] = np.array([
                v_opt[id_func("lam0x", k) - 1],
                v_opt[id_func("lam0y", k) - 1],
            ])
            solution_dict["lambda12"][k] = np.array([
                v_opt[id_func("lam12x", k) - 1],
                v_opt[id_func("lam12y", k) - 1],
            ])
            solution_dict["u"][k] = v_opt[id_func("u", k) - 1]

    return solution_dict

def _fmt_array(arr, precision=8):
    """Compact formatter for numpy arrays in the log file."""
    if arr is None:
        return "None"
    return np.array2string(
        np.asarray(arr, dtype=float),
        precision=precision,
        suppress_small=False,
        separator=", ",
        max_line_width=10_000,
    )


def write_extraction_trajectory_to_log(log_path, solutions, gap_info, errors_by_method, N, rank_info=None):
    """
    Append detailed per-timestep trajectory values to log.txt for the relevant
    extraction methods. This is mainly for thesis/debug analysis.

    Logged for each available method in ['ordered', 'robust']:
      - R1_k, R2_k for k = 0..N
      - F1_k, F2_k for k = 0..N-1
      - x1_k, x2_k for k = 0..N
      - lambda0_k, lambda12_k for k = 1..N-1
      - u_k for k = 1..N-1
    """
    methods_to_log = [m for m in ["ordered", "robust"] if m in solutions]

    with open(log_path, "a") as log_file:
        log_file.write("\n" + "=" * 100 + "\n")
        log_file.write("DETAILED EXTRACTED ACROBOT TRAJECTORIES\n")
        log_file.write("=" * 100 + "\n")

        if rank_info is not None:
            summary = rank_info.get("summary", {})
            blocks = rank_info.get("blocks", [])
            log_file.write("\n" + "#" * 100 + "\n")
            log_file.write("MOMENT MATRIX RANK / TIGHTNESS ANALYSIS\n")
            log_file.write("#" * 100 + "\n")
            log_file.write("Teng-style diagnostic: delta_21 = |lambda_2| / |lambda_1| per moment block.\n")
            log_file.write("Small max delta_21 suggests rank-one/tight moment blocks.\n")
            log_file.write(f"number of analyzed blocks:       {summary.get('num_blocks')}\n")
            log_file.write(f"number of skipped blocks:        {summary.get('num_skipped_blocks')}\n")
            log_file.write(f"max block dimension:             {summary.get('max_block_dim')}\n")
            log_file.write(f"max numerical rank:              {summary.get('max_numerical_rank')}\n")
            log_file.write(f"rank-one blocks:                 {summary.get('num_rank_one_blocks')} / {summary.get('num_blocks')}\n")
            log_file.write(f"rank-one tolerance on delta_21:  {summary.get('rank_one_tol'):.12e}\n")
            if summary.get("max_delta_21") is not None:
                log_file.write(f"max delta_21:                    {summary.get('max_delta_21'):.12e}\n")
                log_file.write(f"mean delta_21:                   {summary.get('mean_delta_21'):.12e}\n")
                log_file.write(f"median delta_21:                 {summary.get('median_delta_21'):.12e}\n")
                log_file.write(f"min eigenvalue over all blocks:  {summary.get('min_eig_over_all_blocks'):.12e}\n")
                log_file.write(f"blocks with negative eig:        {summary.get('num_blocks_with_negative_eig')}\n")

            if blocks:
                log_file.write("\nWorst 20 blocks by delta_21:\n")
                worst = sorted(blocks, key=lambda b: b["delta_21"], reverse=True)[:20]
                for b in worst:
                    log_file.write(
                        f"  block {b['block_index']:4d}: dim={b['dim']:4d}, "
                        f"rank={b['numerical_rank']:3d}, "
                        f"delta_21={b['delta_21']:.12e}, "
                        f"lambda1={b['lambda1_abs']:.12e}, "
                        f"lambda2={b['lambda2_abs']:.12e}, "
                        f"min_eig={b['min_eig']:.12e}\n"
                    )

        if not methods_to_log:
            log_file.write("No ordered or robust extraction available. Nothing to log here.\n")
            return

        for method in methods_to_log:
            sol = solutions[method]
            log_file.write("\n" + "#" * 100 + "\n")
            log_file.write(f"{method.upper()} EXTRACTION\n")
            log_file.write("#" * 100 + "\n")

            if method in gap_info:
                gi = gap_info[method]
                log_file.write("\nCandidate suboptimality gap:\n")
                log_file.write(f"  SDP lower bound:     {gi['sdp_lower_bound']:.12e}\n")
                log_file.write(f"  extracted objective: {gi['extracted_objective']:.12e}\n")
                log_file.write(f"  absolute gap:        {gi['absolute_gap']:.12e}\n")
                log_file.write(f"  relative gap:        {gi['relative_gap']:.12e}\n")

            if method in errors_by_method:
                err = errors_by_method[method]
                log_file.write("\nSO(2) orthogonality errors, max over horizon:\n")
                for key in ["R1", "R2", "F1", "F2"]:
                    vals = err.get(key, [])
                    if len(vals) > 0:
                        log_file.write(f"  max {key}: {max(vals):.12e}\n")

            log_file.write("\nPer-timestep values:\n")
            log_file.write("- R1, R2 are 2x2 SO(2) matrices.\n")
            log_file.write("- F1, F2 are relative step rotations and exist only for k = 0..N-1.\n")
            log_file.write("- lambda0, lambda12 and u exist only for k = 1..N-1.\n")

            for k in range(N + 1):
                log_file.write("\n" + "-" * 100 + "\n")
                log_file.write(f"k = {k}\n")
                log_file.write(f"theta1 = {sol['theta1'][k]:+.12e}\n")
                log_file.write(f"theta2 = {sol['theta2'][k]:+.12e}\n")
                log_file.write(f"x1     = {_fmt_array(sol['x1'][k])}\n")
                log_file.write(f"x2     = {_fmt_array(sol['x2'][k])}\n")
                log_file.write(f"R1     = {_fmt_array(sol['R1'][k])}\n")
                log_file.write(f"R2     = {_fmt_array(sol['R2'][k])}\n")

                if k < N:
                    log_file.write(f"step_theta1 = {sol['step_theta1'][k]:+.12e}\n")
                    log_file.write(f"step_theta2 = {sol['step_theta2'][k]:+.12e}\n")
                    log_file.write(f"F1          = {_fmt_array(sol['F1'][k])}\n")
                    log_file.write(f"F2          = {_fmt_array(sol['F2'][k])}\n")
                else:
                    log_file.write("step_theta1 = None\n")
                    log_file.write("step_theta2 = None\n")
                    log_file.write("F1          = None\n")
                    log_file.write("F2          = None\n")

                if 1 <= k < N:
                    log_file.write(f"lambda0     = {_fmt_array(sol['lambda0'][k])}\n")
                    log_file.write(f"lambda12    = {_fmt_array(sol['lambda12'][k])}\n")
                    log_file.write(f"u           = {float(sol['u'][k]):+.12e}\n")
                else:
                    log_file.write("lambda0     = None\n")
                    log_file.write("lambda12    = None\n")
                    log_file.write("u           = None\n")

        log_file.write("\n" + "=" * 100 + "\n")
        log_file.write("END DETAILED EXTRACTED ACROBOT TRAJECTORIES\n")
        log_file.write("=" * 100 + "\n")


def compute_SO2_errors(sol, N):
    """
    Compute SO(2) constraint violations for rotation and step matrices.
    
    For each angle pair (c, s), checks:
    - Normalization: ||(c, s)||_2 - 1| (should be 0)
    
    Args:
        sol: Solution dictionary with rotation variables
        N: Time horizon
    
    Returns:
        errors_dict: Dictionary with error statistics
    """
    
    errors = {
        "R1": [],
        "R2": [],
        "F1": [],
        "F2": [],
    }

    for k in range(N + 1):
        R1 = sol["R1"][k]
        R2 = sol["R2"][k]
        errors["R1"].append(np.linalg.norm(R1.T @ R1 - np.eye(2), "fro"))
        errors["R2"].append(np.linalg.norm(R2.T @ R2 - np.eye(2), "fro"))

        if k < N:
            F1 = sol["F1"][k]
            F2 = sol["F2"][k]
            errors["F1"].append(np.linalg.norm(F1.T @ F1 - np.eye(2), "fro"))
            errors["F2"].append(np.linalg.norm(F2.T @ F2 - np.eye(2), "fro"))

    return errors


def print_SO2_errors(errors):
    print("\n" + "=" * 60)
    print("SO(2) CONSTRAINT VIOLATION ANALYSIS")
    print("=" * 60)
    for key, values in errors.items():
        if values:
            print(f"{key}: max={np.max(values):.2e}, mean={np.mean(values):.2e}")


def wrap_to_pi(angle):
    """Map angle(s) to [-pi, pi]."""
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def compute_rotation_target_errors(sol, N, params):
    """
    Compute target tracking errors for R_k.

    Two diagnostics are returned:
      1) Frobenius error ||R_{i,k} - R_i^des||_F.
      2) Wrapped scalar angle error theta_{i,k} - theta_i^des.

    For SO(2), the Frobenius error and angle error are related by
        ||R(theta)-R(theta_des)||_F = sqrt(8) * |sin((theta-theta_des)/2)|,
    so the Frobenius version is consistent with the objective.
    """
    c1_des, s1_des = params["c1_des"], params["s1_des"]
    c2_des, s2_des = params["c2_des"], params["s2_des"]

    R1_des = np.array([[c1_des, -s1_des], [s1_des, c1_des]], dtype=float)
    R2_des = np.array([[c2_des, -s2_des], [s2_des, c2_des]], dtype=float)

    theta1_des = float(params["thetaR1_des"])
    theta2_des = float(params["thetaR2_des"])

    err = {
        "R1_target_F": [],
        "R2_target_F": [],
        "theta1_target": [],
        "theta2_target": [],
    }

    for k in range(N + 1):
        err["R1_target_F"].append(float(np.linalg.norm(sol["R1"][k] - R1_des, "fro")))
        err["R2_target_F"].append(float(np.linalg.norm(sol["R2"][k] - R2_des, "fro")))
        err["theta1_target"].append(float(wrap_to_pi(sol["theta1"][k] - theta1_des)))
        err["theta2_target"].append(float(wrap_to_pi(sol["theta2"][k] - theta2_des)))

    return err


def plot_acrobot_summary_and_diagnostics(sol_plot, preferred_method, params, prefix_str, errors=None, target_errors=None):
    """
    Create two figures:
      1) summary.png: angles, planar trajectory, and correctly indexed control u_k for k=1,...,N-1.
      2) diagnostics.png: SO(2) errors for R_k/F_k and target tracking errors for R_k.

    The control plot intentionally does not pad u_0=0 or u_N=0, because those are not decision variables.
    Otherwise the plot politely lies to you, as plots enjoy doing.
    """
    import matplotlib.pyplot as plt

    N = params["N"]
    dt = params["dt"]

    time_grid = np.arange(N + 1) * dt
    theta1 = np.array([sol_plot["theta1"][k] for k in range(N + 1)])
    theta2 = np.array([sol_plot["theta2"][k] for k in range(N + 1)])
    x1 = np.array([sol_plot["x1"][k] for k in range(N + 1)])
    x2 = np.array([sol_plot["x2"][k] for k in range(N + 1)])

    R1_stack = np.array([sol_plot["R1"][k] for k in range(N + 1)])
    R2_stack = np.array([sol_plot["R2"][k] for k in range(N + 1)])
    elbow = x1 + np.einsum("kij,j->ki", R1_stack, params["rho_112"])
    tip = x2 - np.einsum("kij,j->ki", R2_stack, params["rho_212"])
    base = np.tile(params["p_0"], (N + 1, 1))

    # u_k only exists for k=1,...,N-1 in this formulation.
    u_nodes = np.arange(1, N)
    u_time = u_nodes * dt
    u_values = np.array([sol_plot["u"][int(k)] for k in u_nodes], dtype=float) if len(u_nodes) else np.array([])

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), constrained_layout=True)

    axes[0].plot(time_grid, theta1, marker="o", label=r"$\theta_{1,k}$")
    axes[0].plot(time_grid, theta2, marker="o", label=r"$\theta_{2,k}$")
    axes[0].axhline(params["thetaR1_des"], linestyle="--", linewidth=1.0, label=r"$\theta_{1}^{des}$")
    axes[0].axhline(params["thetaR2_des"], linestyle=":", linewidth=1.0, label=r"$\theta_{2}^{des}$")
    axes[0].set_title(f"Joint Angles ({preferred_method})")
    axes[0].set_xlabel("time node $t_k$ [s]")
    axes[0].set_ylabel("angle [rad]")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(x1[:, 0], x1[:, 1], marker="o", label="link 1 COM")
    axes[1].plot(x2[:, 0], x2[:, 1], marker="o", label="link 2 COM")
    axes[1].plot(tip[:, 0], tip[:, 1], marker="o", label="tip")
    axes[1].scatter(base[0, 0], base[0, 1], s=30, label="base")
    axes[1].set_title("Planar Trajectories")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].axis("equal")
    axes[1].grid(True)
    axes[1].legend()

    if len(u_values):
        axes[2].step(u_time, u_values, where="mid", marker="o", label=r"optimized $u_k$")
    axes[2].set_title(r"Control Input, only optimized for $k=1,\ldots,N-1$")
    axes[2].set_xlabel(r"control node $t_k$, $k=1,\ldots,N-1$ [s]")
    axes[2].set_ylabel(r"$u_k$")
    axes[2].grid(True)
    axes[2].legend(loc="best")

    fig.savefig("./figs/" + prefix_str + "summary.png", dpi=200)
    plt.close(fig)

    if errors is None:
        errors = compute_SO2_errors(sol_plot, N)
    if target_errors is None:
        target_errors = compute_rotation_target_errors(sol_plot, N, params)

    fig_diag, ax = plt.subplots(3, 1, figsize=(8, 10), constrained_layout=True)
    k_R = np.arange(N + 1)
    k_F = np.arange(N)

    ax[0].semilogy(k_R, np.maximum(errors["R1"], 1e-18), marker="o", label=r"$\|R_{1,k}^T R_{1,k}-I\|_F$")
    ax[0].semilogy(k_R, np.maximum(errors["R2"], 1e-18), marker="o", label=r"$\|R_{2,k}^T R_{2,k}-I\|_F$")
    ax[0].set_title(r"SO(2) error of $R_k$, $k=0,\ldots,N$")
    ax[0].set_xlabel("node k")
    ax[0].set_ylabel("Frobenius error")
    ax[0].grid(True, which="both")
    ax[0].legend()

    ax[1].semilogy(k_F, np.maximum(errors["F1"], 1e-18), marker="o", label=r"$\|F_{1,k}^T F_{1,k}-I\|_F$")
    ax[1].semilogy(k_F, np.maximum(errors["F2"], 1e-18), marker="o", label=r"$\|F_{2,k}^T F_{2,k}-I\|_F$")
    ax[1].set_title(r"SO(2) error of $F_k$, $k=0,\ldots,N-1$")
    ax[1].set_xlabel("step k")
    ax[1].set_ylabel("Frobenius error")
    ax[1].grid(True, which="both")
    ax[1].legend()

    ax[2].plot(k_R, target_errors["R1_target_F"], marker="o", label=r"$\|R_{1,k}-R_1^{des}\|_F$")
    ax[2].plot(k_R, target_errors["R2_target_F"], marker="o", label=r"$\|R_{2,k}-R_2^{des}\|_F$")
    ax[2].set_title(r"Target rotation error of $R_k$")
    ax[2].set_xlabel("node k")
    ax[2].set_ylabel("Frobenius error")
    ax[2].grid(True)
    ax[2].legend()

    fig_diag.savefig("./figs/" + prefix_str + "diagnostics.png", dpi=200)
    plt.close(fig_diag)

    return {
        "u_nodes": u_nodes,
        "u_time": u_time,
        "u_values": u_values,
        "target_errors": target_errors,
    }


def _as_square_matrix(block):
    """
    Convert a solver block to a square numpy matrix when possible.

    Most SPOT/MOSEK blocks are already square matrices. This fallback also
    accepts a flattened n*n vector. If the object cannot be interpreted as a
    square matrix, None is returned and the block is skipped in rank analysis.
    """
    A = np.asarray(block, dtype=float)
    if A.ndim == 2 and A.shape[0] == A.shape[1]:
        return 0.5 * (A + A.T)

    if A.ndim == 1:
        n = int(round(np.sqrt(A.size)))
        if n * n == A.size:
            A = A.reshape((n, n))
            return 0.5 * (A + A.T)

    return None


def compute_moment_rank_tightness(Xs, rel_tol=1e-8, abs_tol=1e-10, rank_one_tol=1e-6):
    """
    Compute numerical rank/tightness diagnostics for the SDP moment blocks.

    Teng-style tightness diagnostic:
        delta_i = |lambda_2| / |lambda_1|
    where eigenvalues are sorted by absolute value. For a rank-one moment
    block, delta_i should be close to zero. The global diagnostic is
        delta = max_i delta_i.

    This is a numerical diagnostic, not a formal symbolic proof. If the extracted
    trajectory is feasible, the suboptimality gap is small, and delta is small,
    then the relaxation is behaving tightly. Amazing, the computer may even be
    telling the truth for once.
    """
    blocks = []
    skipped_blocks = []

    for idx, block in enumerate(Xs):
        A = _as_square_matrix(block)
        if A is None:
            skipped_blocks.append({
                "block_index": idx,
                "raw_shape": tuple(np.asarray(block).shape),
            })
            continue

        try:
            eigvals = np.linalg.eigvalsh(A)
        except np.linalg.LinAlgError:
            skipped_blocks.append({
                "block_index": idx,
                "raw_shape": tuple(A.shape),
                "reason": "eigvalsh failed",
            })
            continue

        abs_sorted = np.sort(np.abs(eigvals))[::-1]
        lambda1_abs = float(abs_sorted[0]) if abs_sorted.size >= 1 else 0.0
        lambda2_abs = float(abs_sorted[1]) if abs_sorted.size >= 2 else 0.0
        ratio_21 = lambda2_abs / max(lambda1_abs, abs_tol)

        rank_threshold = max(abs_tol, rel_tol * lambda1_abs)
        numerical_rank = int(np.sum(np.abs(eigvals) > rank_threshold))

        blocks.append({
            "block_index": idx,
            "dim": int(A.shape[0]),
            "numerical_rank": numerical_rank,
            "rank_threshold": float(rank_threshold),
            "lambda1_abs": lambda1_abs,
            "lambda2_abs": lambda2_abs,
            "delta_21": float(ratio_21),
            "min_eig": float(np.min(eigvals)) if eigvals.size else 0.0,
            "max_eig": float(np.max(eigvals)) if eigvals.size else 0.0,
            "trace": float(np.trace(A)),
        })

    if blocks:
        deltas = [b["delta_21"] for b in blocks]
        ranks = [b["numerical_rank"] for b in blocks]
        min_eigs = [b["min_eig"] for b in blocks]
        dims = [b["dim"] for b in blocks]

        summary = {
            "num_blocks": len(blocks),
            "num_skipped_blocks": len(skipped_blocks),
            "max_delta_21": float(np.max(deltas)),
            "mean_delta_21": float(np.mean(deltas)),
            "median_delta_21": float(np.median(deltas)),
            "max_numerical_rank": int(np.max(ranks)),
            "num_rank_one_blocks": int(np.sum(np.asarray(deltas) <= rank_one_tol)),
            "rank_one_tol": float(rank_one_tol),
            "max_block_dim": int(np.max(dims)),
            "min_eig_over_all_blocks": float(np.min(min_eigs)),
            "num_blocks_with_negative_eig": int(np.sum(np.asarray(min_eigs) < -abs_tol)),
        }
    else:
        summary = {
            "num_blocks": 0,
            "num_skipped_blocks": len(skipped_blocks),
            "max_delta_21": None,
            "mean_delta_21": None,
            "median_delta_21": None,
            "max_numerical_rank": None,
            "num_rank_one_blocks": 0,
            "rank_one_tol": float(rank_one_tol),
            "max_block_dim": None,
            "min_eig_over_all_blocks": None,
            "num_blocks_with_negative_eig": None,
        }

    return {
        "summary": summary,
        "blocks": blocks,
        "skipped_blocks": skipped_blocks,
        "settings": {
            "rel_tol": float(rel_tol),
            "abs_tol": float(abs_tol),
            "rank_one_tol": float(rank_one_tol),
        },
    }


def print_moment_rank_tightness(rank_info, top_k=10):
    """Print a compact rank/tightness summary to stdout."""
    summary = rank_info.get("summary", {})
    blocks = rank_info.get("blocks", [])

    print("\n" + "=" * 60)
    print("MOMENT MATRIX RANK / TIGHTNESS ANALYSIS")
    print("=" * 60)
    print(f"number of analyzed blocks:       {summary.get('num_blocks')}")
    print(f"number of skipped blocks:        {summary.get('num_skipped_blocks')}")
    print(f"max block dimension:             {summary.get('max_block_dim')}")
    print(f"max numerical rank:              {summary.get('max_numerical_rank')}")
    print(f"rank-one blocks:                 {summary.get('num_rank_one_blocks')} / {summary.get('num_blocks')}")
    print(f"rank-one tolerance on delta_21:  {summary.get('rank_one_tol'):.2e}")
    max_delta = summary.get("max_delta_21")
    if max_delta is not None:
        print(f"max delta_21 = |lambda2|/|lambda1|: {max_delta:.3e}")
        print(f"mean delta_21:                    {summary.get('mean_delta_21'):.3e}")
        print(f"median delta_21:                  {summary.get('median_delta_21'):.3e}")
        print(f"min eigenvalue over all blocks:   {summary.get('min_eig_over_all_blocks'):.3e}")

    if blocks:
        worst = sorted(blocks, key=lambda b: b["delta_21"], reverse=True)[:top_k]
        print("\nWorst blocks by delta_21:")
        for b in worst:
            print(
                f"  block {b['block_index']:4d}: dim={b['dim']:4d}, "
                f"rank={b['numerical_rank']:3d}, "
                f"delta_21={b['delta_21']:.3e}, "
                f"lambda1={b['lambda1_abs']:.3e}, lambda2={b['lambda2_abs']:.3e}, "
                f"min_eig={b['min_eig']:.3e}"
            )

def evaluate_acrobot_objective_from_vector(v_opt, N, id_func, params):
    def val(prefix, k):
        return v_opt[id_func(prefix, k) - 1]

    obj = 0.0

    # Terminal R cost
    obj += params["rho_R"] * 2.0 * (
        (val("c1", N) - params["c1_des"])**2
        + (val("s1", N) - params["s1_des"])**2
    )
    obj += params["rho_R"] * 2.0 * (
        (val("c2", N) - params["c2_des"])**2
        + (val("s2", N) - params["s2_des"])**2
    )

    # Terminal F cost
    obj += params["rho_F"] * 2.0 * (
        (val("a1", N - 1) - params["a1_des"])**2
        + (val("b1", N - 1) - params["b1_des"])**2
    )
    obj += params["rho_F"] * 2.0 * (
        (val("a2", N - 1) - params["a2_des"])**2
        + (val("b2", N - 1) - params["b2_des"])**2
    )

    # R tracking
    for k in range(N):
        obj += params["alpha_R"] * 2.0 * (
            (val("c1", k) - params["c1_des"])**2
            + (val("s1", k) - params["s1_des"])**2
        )
        obj += params["alpha_R"] * 2.0 * (
            (val("c2", k) - params["c2_des"])**2
            + (val("s2", k) - params["s2_des"])**2
        )

    # F tracking
    for k in range(N - 1):
        obj += params["alpha_F"] * 2.0 * (
            (val("a1", k) - params["a1_des"])**2
            + (val("b1", k) - params["b1_des"])**2
        )
        obj += params["alpha_F"] * 2.0 * (
            (val("a2", k) - params["a2_des"])**2
            + (val("b2", k) - params["b2_des"])**2
        )

    # Control cost
    for k in range(1, N):
        obj += (1.0 / params["gamma"]) * val("u", k)**2

    # Lambda regularization. This must match the actual polynomial objective;
    # otherwise the extracted objective and suboptimality gap are off. Tiny
    # detail, naturally placed exactly where it can ruin your interpretation.
    if "alpha_lam" in params and params["alpha_lam"] != 0.0:
        for k in range(1, N):
            obj += params["alpha_lam"] * (
                val("lam0x", k)**2 + val("lam0y", k)**2
                + val("lam12x", k)**2 + val("lam12y", k)**2
            )

    return float(obj)

def print_lambda_diagnostics(sol, params, method_name="solution"):
    """
    Print lambda values and saturation ratios.

    sol["lambda0"][k] and sol["lambda12"][k] are the unscaled SDP variables.
    Physical multipliers are lambda_scale * lambda_code.
    """
    import numpy as np

    N = params["N"]
    lam_scale = float(params["lambda_scale"])
    lam_max = float(params["lambda_max"])
    phys_lam_max = lam_scale * lam_max

    print("\n" + "=" * 90)
    print(f"LAMBDA DIAGNOSTICS ({method_name})")
    print("=" * 90)
    print(f"lambda_scale                 = {lam_scale:.6g}")
    print(f"lambda_max, code variable     = {lam_max:.6g}")
    print(f"lambda_max, physical component = {phys_lam_max:.6g}")
    print("-" * 90)
    print(
        "k | "
        "lam0_code [x,y]        | lam12_code [x,y]       | "
        "max comp ratio | "
        "lam0_phys_norm | lam12_phys_norm"
    )
    print("-" * 90)

    max_ratio = 0.0
    max_info = None

    for k in range(1, N):
        lam0_code = np.asarray(sol["lambda0"][k], dtype=float)
        lam12_code = np.asarray(sol["lambda12"][k], dtype=float)

        lam0_phys = lam_scale * lam0_code
        lam12_phys = lam_scale * lam12_code

        # Component-wise bound is what your SDP actually imposes.
        comp_abs = np.array([
            abs(lam0_code[0]),
            abs(lam0_code[1]),
            abs(lam12_code[0]),
            abs(lam12_code[1]),
        ])

        ratio = float(np.max(comp_abs) / max(lam_max, 1e-12))

        if ratio > max_ratio:
            max_ratio = ratio
            max_info = (k, lam0_code.copy(), lam12_code.copy())

        print(
            f"{k:2d} | "
            f"[{lam0_code[0]:+9.4f}, {lam0_code[1]:+9.4f}] | "
            f"[{lam12_code[0]:+9.4f}, {lam12_code[1]:+9.4f}] | "
            f"{ratio:13.4f} | "
            f"{np.linalg.norm(lam0_phys):14.6f} | "
            f"{np.linalg.norm(lam12_phys):15.6f}"
        )

    print("-" * 90)
    print(f"max component saturation ratio = {max_ratio:.6f}")

    if max_info is not None:
        k_star, lam0_star, lam12_star = max_info
        print(f"worst lambda node k = {k_star}")
        print(f"lambda0_code  at worst k = {lam0_star}")
        print(f"lambda12_code at worst k = {lam12_star}")
        print(f"lambda0_phys  at worst k = {lam_scale * lam0_star}")
        print(f"lambda12_phys at worst k = {lam_scale * lam12_star}")

    if max_ratio > 0.95:
        print("WARNING: lambda is very close to its component bound.")
    elif max_ratio > 0.80:
        print("NOTE: lambda is using a large fraction of the allowed bound.")
    else:
        print("Lambda is not close to the component bound.")

    print("=" * 90 + "\n")

def plot_lambda_diagnostics(sol, params, prefix_str=""):
    import numpy as np
    import matplotlib.pyplot as plt

    N = params["N"]
    dt = params["dt"]
    lam_scale = float(params["lambda_scale"])
    lam_max = float(params["lambda_max"])

    k_nodes = np.arange(1, N)
    t_nodes = k_nodes * dt

    lam0 = np.array([sol["lambda0"][int(k)] for k in k_nodes], dtype=float)
    lam12 = np.array([sol["lambda12"][int(k)] for k in k_nodes], dtype=float)

    lam0_phys = lam_scale * lam0
    lam12_phys = lam_scale * lam12

    # Component saturation ratio, because your constraint is component-wise.
    ratio0x = np.abs(lam0[:, 0]) / lam_max
    ratio0y = np.abs(lam0[:, 1]) / lam_max
    ratio12x = np.abs(lam12[:, 0]) / lam_max
    ratio12y = np.abs(lam12[:, 1]) / lam_max

    fig, ax = plt.subplots(3, 1, figsize=(8, 10), constrained_layout=True)

    ax[0].plot(t_nodes, lam0_phys[:, 0], marker="o", label=r"$\bar\lambda_{0x}$")
    ax[0].plot(t_nodes, lam0_phys[:, 1], marker="o", label=r"$\bar\lambda_{0y}$")
    ax[0].axhline(lam_scale * lam_max, linestyle="--", linewidth=1.0)
    ax[0].axhline(-lam_scale * lam_max, linestyle="--", linewidth=1.0)
    ax[0].set_title(r"Physical pivot multiplier $\bar\lambda_0 = s_\lambda \lambda_0$")
    ax[0].set_xlabel(r"time node $t_k$, $k=1,\ldots,N-1$ [s]")
    ax[0].set_ylabel(r"physical $\bar\lambda_0$")
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(t_nodes, lam12_phys[:, 0], marker="o", label=r"$\bar\lambda_{12x}$")
    ax[1].plot(t_nodes, lam12_phys[:, 1], marker="o", label=r"$\bar\lambda_{12y}$")
    ax[1].axhline(lam_scale * lam_max, linestyle="--", linewidth=1.0)
    ax[1].axhline(-lam_scale * lam_max, linestyle="--", linewidth=1.0)
    ax[1].set_title(r"Physical elbow multiplier $\bar\lambda_{12} = s_\lambda \lambda_{12}$")
    ax[1].set_xlabel(r"time node $t_k$, $k=1,\ldots,N-1$ [s]")
    ax[1].set_ylabel(r"physical $\bar\lambda_{12}$")
    ax[1].grid(True)
    ax[1].legend()

    ax[2].plot(t_nodes, ratio0x, marker="o", label=r"$|\lambda_{0x}|/\lambda_{\max}$")
    ax[2].plot(t_nodes, ratio0y, marker="o", label=r"$|\lambda_{0y}|/\lambda_{\max}$")
    ax[2].plot(t_nodes, ratio12x, marker="o", label=r"$|\lambda_{12x}|/\lambda_{\max}$")
    ax[2].plot(t_nodes, ratio12y, marker="o", label=r"$|\lambda_{12y}|/\lambda_{\max}$")
    ax[2].axhline(1.0, linestyle="--", linewidth=1.0)
    ax[2].set_title("Lambda component saturation ratios")
    ax[2].set_xlabel(r"time node $t_k$, $k=1,\ldots,N-1$ [s]")
    ax[2].set_ylabel("ratio")
    ax[2].grid(True)
    ax[2].legend()

    fig.savefig("./figs/" + prefix_str + "lambda_diagnostics.png", dpi=200)
    plt.close(fig)

def main():
    """
    Main function to setup and solve the Acrobot optimization problem using SPOT.
    
    Decision Variables:
    - c_{i,k}, s_{i,k} (k=0..N, i=1,2): Rotation matrices (cos, sin) for SO(2)
    - a_{i,k}, b_{i,k} (k=0..N-1, i=1,2): Step rotation matrices (cos, sin) for SO(2)
    - lambda_{0,k} (k=1..N-1): Constraint forces at pivot
    - lambda_{12,k} (k=1..N-1): Constraint forces at joint
    - u_k (k=1..N-1): Control input (scalar)
    
    Steps:
    1. Set up optimization parameters (time horizon, physical parameters, cost coefficients)
    2. Create variable mapping and indexing
    3. Create NumPolySystem and add all constraints and objective
    4. Define clique decomposition for efficient solving
    5. Call CSTSS_pybind to solve the optimization problem
    6. Extract and save solution
    """
    
    total_start = time.time()

    # --- CSTSS parameters ---
    params = {}
    kappa = 2; params['kappa'] = kappa
    relax_mode = "SOS"; params['relax_mode'] = relax_mode
    cs_mode = "SELF"; params['cs_mode'] = cs_mode
    ts_mode = "NON"; params['ts_mode'] = ts_mode
    ts_mom_mode = "NON"; params['ts_mom_mode'] = ts_mom_mode
    ts_eq_mode = "NON"; params['ts_eq_mode'] = ts_eq_mode
    if_solve = True; params['if_solve'] = if_solve
    if_mex = True; params['if_mex'] = if_mex

    # --- System parameters ---
    N = 20; params['N'] = N  # Time horizon (number of discrete steps)
    dt = 0.1; params['dt'] = dt  # Time step size

    # Variable mapping and indexing
    var_mapping, var_start_dict, total_var_num = get_var_mapping_and_dict(N)
    params['total_var_num'] = total_var_num
    params['var_mapping'] = var_mapping
    params['var_start_dict'] = var_start_dict
    
    # Create prefix_k0: maps each prefix to its starting timestep
    # x_i,k and R_i,k variables: k=0..N       -> start at k=0
    # F_i,k variables: k=0..N-1               -> start at k=0
    # lambda_k and u_k variables: k=1..N-1    -> start at k=1

    prefix_k0 = {
        # Rotation variables R_{i,k}, k=0..N
        "c1": 0, "s1": 0,
        "c2": 0, "s2": 0,

        # Step rotation variables F_{i,k}, k=0..N-1
        "a1": 0, "b1": 0,
        "a2": 0, "b2": 0,

        # Lagrange multipliers lambda_k, k=1..N-1
        "lam0x": 1, "lam0y": 1,
        "lam12x": 1, "lam12y": 1,

        # Control input u_k, k=1..N-1
        "u": 1,
    }

    params['prefix_k0'] = prefix_k0
    params['id'] = lambda prefix, k: get_id(prefix, k, var_start_dict, prefix_k0)
    
    # --- Physical parameters for Acrobot ---

    # Gravity
    g = 9.81; params["g"] = g

    # Link masses
    m1 = 1.0; params["m1"] = m1
    m2 = 1.0; params["m2"] = m2

    # Planar inertia parameters.
    # For SO(2), J_{d,i} is represented as diag(delta_i1, delta_i2).
    # The rotational dynamics use delta_i1 + delta_i2.
    delta_11 = 0.01
    delta_12 = 0.01
    delta_21 = 0.01
    delta_22 = 0.01

    params["delta_11"] = delta_11
    params["delta_12"] = delta_12
    params["delta_21"] = delta_21
    params["delta_22"] = delta_22

    params["Jd1"] = np.diag([delta_11, delta_12])
    params["Jd2"] = np.diag([delta_21, delta_22])

    # Geometry.
    # Body-fixed vectors from center of mass to attachment points.
    #
    # Convention:
    # rho_10  : vector from link 1 COM to base/pivot joint
    # rho_112 : vector from link 1 COM to elbow joint
    # rho_212 : vector from link 2 COM to elbow joint
    #
    # Example for two links of length L1, L2 whose COMs are at their midpoints.
    l1 = 0.5
    l2 = 0.5
    params["l1"] = l1
    params["l2"] = l2

    # Thesis convention: R maps body to inertial frame and hanging down is local -y.
    # rho vectors are from COM to joint locations, expressed in each body frame.
    params["rho_10"] = np.array([0.0,  l1 / 2.0])   # link-1 COM -> base
    params["rho_112"] = np.array([0.0, -l1 / 2.0])  # link-1 COM -> elbow
    params["rho_212"] = np.array([0.0,  l2 / 2.0])  # link-2 COM -> elbow

    # Fixed pivot position
    params["p_0"] = np.array([0.0, 0.0])


    # --- Initial states ---

    # Initial absolute link orientations.
    # Example: both links initially hanging downward/upward depending on your convention.
    # Choose these according to your simulation convention.
    # With x=l*sin(theta), y=-l*cos(theta), theta=0 means hanging down.
    deg = np.pi / 180.0

    # Initial absolute link orientations: 5 degrees from hanging down
    thetaR1_0 = 10.0 * deg
    thetaR2_0 = 10.0 * deg
    params["thetaR1_0"] = thetaR1_0
    params["thetaR2_0"] = thetaR2_0

    # Rest start: no initial relative rotation step.
    # F_i,0 = I_2 means thetaF_i,0 = 0.
    thetaF1_0 = 0.0; params["thetaF1_0"] = thetaF1_0
    thetaF2_0 = 0.0; params["thetaF2_0"] = thetaF2_0


    # --- Desired terminal states ---

    # Desired absolute orientations.
    # Example: swing-up target. Change these to your actual desired terminal pose.
    thetaR1_des = 180.0 * deg; params["thetaR1_des"] = thetaR1_des
    thetaR2_des = 180.0 * deg; params["thetaR2_des"] = thetaR2_des

    # Desired R_i in scalar SO(2) form
    params["c1_des"] = np.cos(thetaR1_des)
    params["s1_des"] = np.sin(thetaR1_des)

    params["c2_des"] = np.cos(thetaR2_des)
    params["s2_des"] = np.sin(thetaR2_des)

    # Desired terminal step rotations.
    # For terminal rest, choose F_i^des = I_2.
    thetaF1_des = 0.0; params["thetaF1_des"] = thetaF1_des
    thetaF2_des = 0.0; params["thetaF2_des"] = thetaF2_des

    params["a1_des"] = np.cos(thetaF1_des)
    params["b1_des"] = np.sin(thetaF1_des)

    params["a2_des"] = np.cos(thetaF2_des)
    params["b2_des"] = np.sin(thetaF2_des)


    # --- Cost weights ---

    rho_R = 50.0; params["rho_R"] = rho_R # 10
    rho_F = 1.0; params["rho_F"] = rho_F # 1.0
    alpha_R = 0.1; params["alpha_R"] = alpha_R # 0.1
    alpha_F = 0.005; params["alpha_F"] = alpha_F # 0.05
    gamma = 70.0; params["gamma"] = gamma # 5.0

    alpha_lam = 0; params["alpha_lam"] = alpha_lam # 0.0001
    

    # --- Constraint bounds ---

    # Step angle bounds.
    # Since a_i,k = cos(Delta theta_i,k), impose a_i,k >= a_i_min.
    theta_step_max = 0.4  # radians, adjust later
    params["theta_step_max"] = theta_step_max

    params["a1_min"] = np.cos(theta_step_max)
    params["a2_min"] = np.cos(theta_step_max)

    # Input bound
    u_max = 15.0
    params["u_max"] = u_max

    # Lambda bound
    lambda_scale = 10.0
    params["lambda_scale"] = lambda_scale

    lambda_max = 4.0
    params["lambda_max"] = lambda_max

    # --- File management ---
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    prefix_str = "Acrobot_SO2_Reduced_NoX_NoV/" + current_time + "/"
    for directory in ["./data/" + prefix_str, "./markdown/" + prefix_str,
                      "./figs/" + prefix_str, "./logs/" + prefix_str]:
        os.makedirs(directory, exist_ok=True)

    log_path = "./logs/" + prefix_str + "log.txt"
    with open(log_path, "w") as log_file:
        log_file.write("Acrobot SO(2) Reduced No-X/No-V Optimization Problem\n")
        log_file.write("=" * 50 + "\n")
        log_file.write("params: \n")
        log_file.write(str(params) + "\n")

    # --- Get remapping information ---
    ids_remap = get_remapped_ids(params); params['ids_remap'] = ids_remap

    # --- Create NumPolySystem ---
    ps = NumPolySystem(n_vars=total_var_num)
    id_func = params['id']

    def v(prefix, k):
        """Helper function to access variables"""
        return ps.var(id_func(prefix, k) - 1)

    eq_mask_sys = []

    # --- Initial constraints ---
    # Reduced initial state at k=0:
    # - R_{1,0}, R_{2,0}
    # - F_{1,0}, F_{2,0}; F_0=I imposes rest start
    # - SO(2) constraints at k=0

    eqs, ineqs, eq_mask = get_init_constraints(
        v("c1", 0),  v("s1", 0),
        v("c2", 0),  v("s2", 0),
        v("a1", 0),  v("b1", 0),
        v("a2", 0),  v("b2", 0),
        params
    )

    for eq in eqs:
        ps.add_eq(eq)
    for ineq in ineqs:
        ps.add_ineq(ineq)
    eq_mask_sys.extend(eq_mask)

    # --- Main constraint loop ---
    for k in range(1, N + 1):

        # ------------------------------------------------------------
        # Reduced translational dynamics constraints
        # Valid for k = 1,...,N-1
        # Uses R_{k-1}, R_k, F_{k-1}, F_k, lambda_k.
        # No x or v variables are present.
        # ------------------------------------------------------------
        if k < N:
            eqs, ineqs, eq_mask = get_translational_dynamics_link1(
                v("c1", k - 1), v("s1", k - 1), v("b1", k - 1),
                v("c1", k),     v("s1", k),     v("b1", k),
                v("lam0x", k),  v("lam0y", k),
                v("lam12x", k), v("lam12y", k),
                params
            )
            for eq in eqs:
                ps.add_eq(eq)
            for ineq in ineqs:
                ps.add_ineq(ineq)
            eq_mask_sys.extend(eq_mask)

            eqs, ineqs, eq_mask = get_translational_dynamics_link2(
                v("c1", k - 1), v("s1", k - 1), v("b1", k - 1),
                v("c1", k),     v("s1", k),     v("b1", k),
                v("c2", k - 1), v("s2", k - 1), v("b2", k - 1),
                v("c2", k),     v("s2", k),     v("b2", k),
                v("lam12x", k), v("lam12y", k),
                params
            )
            for eq in eqs:
                ps.add_eq(eq)
            for ineq in ineqs:
                ps.add_ineq(ineq)
            eq_mask_sys.extend(eq_mask)


        # ------------------------------------------------------------
        # Rotational dynamics constraints
        # Valid for k = 1,...,N-1
        # Uses b_{k-1}, b_k, R_k, lambda_k, u_k
        # ------------------------------------------------------------
        if k < N:
            eqs, ineqs, eq_mask = get_rotational_dynamics_link1(
                v("b1", k - 1),  v("b1", k),
                v("c1", k),      v("s1", k),
                v("lam0x", k),   v("lam0y", k),
                v("lam12x", k),  v("lam12y", k),
                v("u", k),
                params
            )
            for eq in eqs:
                ps.add_eq(eq)
            for ineq in ineqs:
                ps.add_ineq(ineq)
            eq_mask_sys.extend(eq_mask)

            eqs, ineqs, eq_mask = get_rotational_dynamics_link2(
                v("b2", k - 1),  v("b2", k),
                v("c2", k),      v("s2", k),
                v("lam12x", k),  v("lam12y", k),
                v("u", k),
                params
            )
            for eq in eqs:
                ps.add_eq(eq)
            for ineq in ineqs:
                ps.add_ineq(ineq)
            eq_mask_sys.extend(eq_mask)

            # lambda bounds
            eqs, ineqs, eq_mask = get_lambda_bounds(
                v("lam0x", k),  v("lam0y", k),
                v("lam12x", k), v("lam12y", k),
                params
            )

            for eq in eqs:
                ps.add_eq(eq)
            for ineq in ineqs:
                ps.add_ineq(ineq)
            eq_mask_sys.extend(eq_mask)


        # ------------------------------------------------------------
        # Rotational kinematics
        # Valid for k = 1,...,N
        # R_k = R_{k-1} F_{k-1}
        # ------------------------------------------------------------
        eqs, ineqs, eq_mask = get_rotational_kinematics_link1(
            v("c1", k - 1), v("s1", k - 1),
            v("c1", k),     v("s1", k),
            v("a1", k - 1), v("b1", k - 1),
            params
        )
        for eq in eqs:
            ps.add_eq(eq)
        for ineq in ineqs:
            ps.add_ineq(ineq)
        eq_mask_sys.extend(eq_mask)

        eqs, ineqs, eq_mask = get_rotational_kinematics_link2(
            v("c2", k - 1), v("s2", k - 1),
            v("c2", k),     v("s2", k),
            v("a2", k - 1), v("b2", k - 1),
            params
        )
        for eq in eqs:
            ps.add_eq(eq)
        for ineq in ineqs:
            ps.add_ineq(ineq)
        eq_mask_sys.extend(eq_mask)


        # ------------------------------------------------------------
        # SO(2) orthogonality constraints for R_k
        # Valid for k = 1,...,N
        # R_0 SO(2) is already inside get_init_constraints.
        # ------------------------------------------------------------
        eqs, ineqs, eq_mask = get_SO2_orthogonality_constraint_rotation_R(
            v("c1", k), v("s1", k),
            v("c2", k), v("s2", k),
            params
        )
        for eq in eqs:
            ps.add_eq(eq)
        for ineq in ineqs:
            ps.add_ineq(ineq)
        eq_mask_sys.extend(eq_mask)


        # ------------------------------------------------------------
        # SO(2) orthogonality constraints for F_{k-1}
        # Valid for k = 1,...,N, so F_{k-1} = F_0,...,F_{N-1}
        # ------------------------------------------------------------
        eqs, ineqs, eq_mask = get_SO2_orthogonality_constraint_rotation_F(
            v("a1", k - 1), v("b1", k - 1),
            v("a2", k - 1), v("b2", k - 1),
            params
        )
        for eq in eqs:
            ps.add_eq(eq)
        for ineq in ineqs:
            ps.add_ineq(ineq)
        eq_mask_sys.extend(eq_mask)


        # ------------------------------------------------------------
        # Step angle bounds for F_{k-1}
        # Valid for k = 1,...,N
        # ------------------------------------------------------------
        eqs, ineqs, eq_mask = get_step_angle_bound_constraint_link_1(
            v("a1", k - 1),
            params
        )
        for eq in eqs:
            ps.add_eq(eq)
        for ineq in ineqs:
            ps.add_ineq(ineq)
        eq_mask_sys.extend(eq_mask)

        eqs, ineqs, eq_mask = get_step_angle_bound_constraint_link_2(
            v("a2", k - 1),
            params
        )
        for eq in eqs:
            ps.add_eq(eq)
        for ineq in ineqs:
            ps.add_ineq(ineq)
        eq_mask_sys.extend(eq_mask)


        # ------------------------------------------------------------
        # Control bounds for u_k
        # Valid only for k = 1,...,N-1
        # No u_0 and no u_N.
        # ------------------------------------------------------------
        if k <= N - 1:
            eqs, ineqs, eq_mask = get_control_bounds(
                v("u", k),
                params
            )
            for eq in eqs:
                ps.add_eq(eq)
            for ineq in ineqs:
                ps.add_ineq(ineq)
            eq_mask_sys.extend(eq_mask)

    # --- Objective ---
    # From Acrobot POP objective:
    #
    # min:
    #   rho_R * (||R_{1,N} - R_1^des||_F^2 + ||R_{2,N} - R_2^des||_F^2)
    # + rho_F * (||F_{1,N-1} - F_1^des||_F^2 + ||F_{2,N-1} - F_2^des||_F^2)
    # + sum_{k=0}^{N-1} alpha_R * (||R_{1,k} - R_1^des||_F^2 + ||R_{2,k} - R_2^des||_F^2)
    # + sum_{k=0}^{N-2} alpha_F * (||F_{1,k} - F_1^des||_F^2 + ||F_{2,k} - F_2^des||_F^2)
    # + sum_{k=1}^{N-1} gamma^{-1} * u_k^2

    obj_expr = 0


    # ------------------------------------------------------------
    # Terminal cost: rho_R * ||R_{i,N} - R_i^des||_F^2
    # ------------------------------------------------------------

    # Link 1 terminal R cost
    obj_expr = obj_expr + params["rho_R"] * 2.0 * (
        (v("c1", N) - params["c1_des"])**2
        + (v("s1", N) - params["s1_des"])**2
    )

    # Link 2 terminal R cost
    obj_expr = obj_expr + params["rho_R"] * 2.0 * (
        (v("c2", N) - params["c2_des"])**2
        + (v("s2", N) - params["s2_des"])**2
    )


    # ------------------------------------------------------------
    # Terminal cost: rho_F * ||F_{i,N-1} - F_i^des||_F^2
    # ------------------------------------------------------------

    # Link 1 terminal F cost
    obj_expr = obj_expr + params["rho_F"] * 2.0 * (
        (v("a1", N - 1) - params["a1_des"])**2
        + (v("b1", N - 1) - params["b1_des"])**2
    )

    # Link 2 terminal F cost
    obj_expr = obj_expr + params["rho_F"] * 2.0 * (
        (v("a2", N - 1) - params["a2_des"])**2
        + (v("b2", N - 1) - params["b2_des"])**2
    )


    # ------------------------------------------------------------
    # Tracking cost for R_k over horizon k = 0,...,N-1
    # ------------------------------------------------------------
    for k in range(N):

        # Link 1 R tracking cost
        obj_expr = obj_expr + params["alpha_R"] * 2.0 * (
            (v("c1", k) - params["c1_des"])**2
            + (v("s1", k) - params["s1_des"])**2
        )

        # Link 2 R tracking cost
        obj_expr = obj_expr + params["alpha_R"] * 2.0 * (
            (v("c2", k) - params["c2_des"])**2
            + (v("s2", k) - params["s2_des"])**2
        )


    # ------------------------------------------------------------
    # Tracking cost for F_k over horizon k = 0,...,N-2
    # ------------------------------------------------------------
    for k in range(N - 1):

        # Link 1 F tracking cost
        obj_expr = obj_expr + params["alpha_F"] * 2.0 * (
            (v("a1", k) - params["a1_des"])**2
            + (v("b1", k) - params["b1_des"])**2
        )

        # Link 2 F tracking cost
        obj_expr = obj_expr + params["alpha_F"] * 2.0 * (
            (v("a2", k) - params["a2_des"])**2
            + (v("b2", k) - params["b2_des"])**2
        )


    # ------------------------------------------------------------
    # Control cost over horizon k = 1,...,N-1
    # ------------------------------------------------------------
    for k in range(1, N):

        obj_expr = obj_expr + (1.0 / params["gamma"]) * v("u", k)**2


    # Lambda regularization
    for k in range(1, N):
        obj_expr = obj_expr + params["alpha_lam"] * (
            v("lam0x", k)**2 + v("lam0y", k)**2
            + v("lam12x", k)**2 + v("lam12y", k)**2
        )

    ps.set_obj(obj_expr)

    # --- Clean polynomials ---
    ps.clean_all(tol=1e-14, if_scale=True, scale_obj=False)

    # --- Get supp_rpt data ---
    poly_data = ps.get_supp_rpt_data(kappa)

    print("Construction Finish!")

    # --- Get clique decomposition ---
    if params["cs_mode"] == "SELF":
        cliques = get_cliques_for_cstss(N, params)
    else:
        cliques = []

    params["cliques"] = cliques

    # --- Run CSTSS ---
    start_time = time.time()
    result, res, coeff_info, aux_info = CSTSS_pybind(
        poly_data, kappa, total_var_num, params
    )
    elapsed_time = time.time() - start_time
    aux_info["result"] = result
    params["aux_info"] = aux_info

    with open(log_path, "a") as log_file:
        result_str = str(result) if isinstance(result, list) else f"{result:.20f}"
        log_file.write(f"\nAcrobot_SO2 N={N}, Relax={relax_mode}, TS={ts_mode}, CS={str(cs_mode)}, "
                       f"result={result_str}, operation time={elapsed_time:.5f}, "
                       f"mosek time={aux_info.get('mosek_time', 0):.5f}\n")

        # --- Remap clique ids ---
    if "cliques" in aux_info and aux_info["cliques"]:
        cliques_remapped = []
        aver_remapped = []
        for clique in aux_info["cliques"]:
            remapped = [params["ids_remap"][i - 1] for i in clique]
            cliques_remapped.append(remapped)
            aver_remapped.append(np.mean(remapped))
        cliques_rank = np.argsort(aver_remapped)
        params["cliques_rank"] = cliques_rank
    else:
        params["cliques_rank"] = []

    # --- Markdown debug (for visualization of constraints/objective) ---
    clique_supp_list = []
    clique_coeff_list = []
    kappa_width = 2 * kappa
    if "cliques" in aux_info and aux_info["cliques"]:
        cliques_rank = params["cliques_rank"]
        for i in range(len(cliques_rank)):
            ii = cliques_rank[i]
            sorted_vars = sorted(aux_info["cliques"][ii])
            supp = np.zeros((len(sorted_vars), kappa_width), dtype=np.float64)
            for idx_v, j in enumerate(sorted_vars):
                supp[idx_v, -1] = j
            clique_supp_list.append(supp)
            clique_coeff_list.append(np.ones(len(sorted_vars)))

    md_path = "./markdown/" + prefix_str + "opt_problem.md"
    with open(md_path, "w") as md_file:
        md_file.write("equality constraints: \n")
        numpoly_visualize(aux_info['supp_rpt_h'], aux_info['coeff_h'], var_mapping, md_file)
        md_file.write("inequality constraints: \n")
        numpoly_visualize(aux_info['supp_rpt_g'], aux_info['coeff_g'], var_mapping, md_file)
        md_file.write("objective: \n")
        numpoly_visualize([aux_info['supp_rpt_f']], [aux_info['coeff_f']], var_mapping, md_file)
        md_file.write("cliques: \n")
        numpoly_visualize(clique_supp_list, clique_coeff_list, var_mapping, md_file)

    params['self_cliques'] = cliques

    # --- Early exit if not solving (for debugging problem formulation only) ---
    if not params['if_solve']:
        supp_rpt_f = aux_info.get("supp_rpt_f", None)
        supp_rpt_g = aux_info.get("supp_rpt_g", None)
        supp_rpt_h = aux_info.get("supp_rpt_h", None)
        coeff_f = aux_info.get("coeff_f", None)
        coeff_g = aux_info.get("coeff_g", None)
        coeff_h = aux_info.get("coeff_h", None)
        with open("./data/" + prefix_str + "polys.pkl", "wb") as f:
            pickle.dump({'supp_rpt_f': supp_rpt_f, 'supp_rpt_g': supp_rpt_g,
                         'supp_rpt_h': supp_rpt_h, 'coeff_f': coeff_f,
                         'coeff_g': coeff_g, 'coeff_h': coeff_h}, f)
        params_to_save = {k: vv for k, vv in params.items() if not callable(vv)}
        with open("./data/" + prefix_str + "params.pkl", "wb") as f:
            pickle.dump(params_to_save, f)
        print("Debugging mode: Problem formulation saved. No solving performed.")
        total_time = time.time() - total_start
        print(f"\nTotal time: {total_time:.5f} s")
        with open(log_path, "a") as log_file:
            log_file.write(f"Debugging mode (if_solve=False)\ntotal time={total_time:.5f}\n")
        return



    if relax_mode == "MOMENT":
        Xs = res["Xopt"]
    elif relax_mode == "SOS":
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

    v_opt_naive, output_info_naive = naive_extract(Xs, mon_rpt, ts_info, total_var_num)
    with open("./data/" + prefix_str + "v_opt_naive.pkl", "wb") as f:
        pickle.dump(v_opt_naive, f)

    solutions = {
        "naive": extract_solution_variables(v_opt_naive, var_start_dict, N, id_func, params),
    }
    extracted_vectors = {
        "naive": v_opt_naive,
    }
    extraction_info = {
        "naive": output_info_naive,
    }

    if ts_mode == "NON":
        v_opt_robust, output_info_robust = robust_extract_CS(Xs, mom_mat_rpt, total_var_num, 1e-2)
        with open("./data/" + prefix_str + "v_opt_robust.pkl", "wb") as f:
            pickle.dump(v_opt_robust, f)
        solutions["robust"] = extract_solution_variables(v_opt_robust, var_start_dict, N, id_func, params)
        extracted_vectors["robust"] = v_opt_robust
        extraction_info["robust"] = output_info_robust

        v_opt_ordered, output_info_ordered = ordered_extract_CS(
            Xs, mom_mat_rpt, total_var_num, 1e-2, params.get("cliques_rank", [])
        )
        with open("./data/" + prefix_str + "v_opt_ordered.pkl", "wb") as f:
            pickle.dump(v_opt_ordered, f)
        solutions["ordered"] = extract_solution_variables(v_opt_ordered, var_start_dict, N, id_func, params)
        extracted_vectors["ordered"] = v_opt_ordered
        extraction_info["ordered"] = output_info_ordered

    # --- Candidate suboptimality gap check ---
    sdp_lower_bound = float(np.asarray(result).squeeze())

    gap_info = {}

    for name, v_extracted in extracted_vectors.items():
        extracted_obj = evaluate_acrobot_objective_from_vector(
            v_extracted, N, id_func, params
        )

        abs_gap = extracted_obj - sdp_lower_bound
        rel_gap = abs_gap / max(1.0, abs(extracted_obj))

        gap_info[name] = {
            "sdp_lower_bound": sdp_lower_bound,
            "extracted_objective": extracted_obj,
            "absolute_gap": abs_gap,
            "relative_gap": rel_gap,
        }

        print("\n" + "=" * 60)
        print(f"CANDIDATE SUBOPTIMALITY GAP ({name.upper()})")
        print("=" * 60)
        print(f"SDP lower bound:       {sdp_lower_bound:.12e}")
        print(f"Extracted objective:   {extracted_obj:.12e}")
        print(f"Absolute gap:          {abs_gap:.12e}")
        print(f"Relative gap:          {rel_gap:.12e}")


    # --- Moment matrix rank / tightness check ---
    # This is the numerical version of the Teng rank-one diagnostic:
    # delta = max_i |lambda_{i,2}| / |lambda_{i,1}| over all moment blocks.
    rank_info = compute_moment_rank_tightness(
        Xs,
        rel_tol=1e-8,
        abs_tol=1e-10,
        rank_one_tol=1e-6,
    )
    print_moment_rank_tightness(rank_info)
    with open("./data/" + prefix_str + "rank_info.pkl", "wb") as f:
        pickle.dump(rank_info, f)

    errors_by_method = {name: compute_SO2_errors(sol, N) for name, sol in solutions.items()}

    for name, sol in solutions.items():
        print("\n" + "=" * 60)
        print(f"SOLUTION SUMMARY ({name.upper()} EXTRACTION)")
        print("=" * 60)
        for k in range(N + 1):
            theta1 = sol["theta1"][k]
            theta2 = sol["theta2"][k]
            print(f"k={k:2d}: theta1={theta1:+.6f}, theta2={theta2:+.6f}, x1={sol['x1'][k]}, x2={sol['x2'][k]}")
        print_SO2_errors(errors_by_method[name])

    write_extraction_trajectory_to_log(
        log_path=log_path,
        solutions=solutions,
        gap_info=gap_info,
        errors_by_method=errors_by_method,
        N=N,
        rank_info=rank_info,
    )

    preferred_method = "ordered" if "ordered" in solutions else ("robust" if "robust" in solutions else "naive")
    sol_plot = solutions[preferred_method]

    time_grid = np.arange(N + 1) * dt
    theta1 = np.array([sol_plot["theta1"][k] for k in range(N + 1)])
    theta2 = np.array([sol_plot["theta2"][k] for k in range(N + 1)])
    x1 = np.array([sol_plot["x1"][k] for k in range(N + 1)])
    x2 = np.array([sol_plot["x2"][k] for k in range(N + 1)])
    u_vals = np.array([sol_plot["u"].get(k, 0.0) for k in range(N + 1)])

    R1_stack = np.array([sol_plot["R1"][k] for k in range(N + 1)])
    R2_stack = np.array([sol_plot["R2"][k] for k in range(N + 1)])
    elbow = x1 + np.einsum("kij,j->ki", R1_stack, params["rho_112"])
    tip = x2 - np.einsum("kij,j->ki", R2_stack, params["rho_212"])
    base = np.tile(params["p_0"], (N + 1, 1))

    with open("./data/" + prefix_str + "solutions.pkl", "wb") as f:
        pickle.dump({
            "method_used_for_plots": preferred_method,
            "solutions": solutions,
            "errors": errors_by_method,
            "gap_info": gap_info,
            "rank_info": rank_info,
            "vectors": extracted_vectors,
            "extraction_info": extraction_info,
            "mom_mat_rpt": mom_mat_rpt,
            "mom_mat_num": mom_mat_num,
            "total_var_num": total_var_num,
            "aux_info": aux_info,
        }, f)

    supp_rpt_f = aux_info.get("supp_rpt_f", None)
    supp_rpt_g = aux_info.get("supp_rpt_g", None)
    supp_rpt_h = aux_info.get("supp_rpt_h", None)
    coeff_f = aux_info.get("coeff_f", None)
    coeff_g = aux_info.get("coeff_g", None)
    coeff_h = aux_info.get("coeff_h", None)
    with open("./data/" + prefix_str + "polys.pkl", "wb") as f:
        pickle.dump({
            "supp_rpt_f": supp_rpt_f,
            "supp_rpt_g": supp_rpt_g,
            "supp_rpt_h": supp_rpt_h,
            "coeff_f": coeff_f,
            "coeff_g": coeff_g,
            "coeff_h": coeff_h,
        }, f)
    params_to_save = {k: vv for k, vv in params.items() if not callable(vv)}
    with open("./data/" + prefix_str + "params.pkl", "wb") as f:
        pickle.dump(params_to_save, f)

    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter

        # Summary plot + diagnostics plot.
        # Important: u_k is plotted only for k=1,...,N-1, not padded with fake u_0/u_N zeros.
        target_errors_plot = compute_rotation_target_errors(sol_plot, N, params)
        plot_acrobot_summary_and_diagnostics(
            sol_plot=sol_plot,
            preferred_method=preferred_method,
            params=params,
            prefix_str=prefix_str,
            errors=errors_by_method[preferred_method],
            target_errors=target_errors_plot,
        )
        
        plot_lambda_diagnostics(sol_plot, params, prefix_str=prefix_str)

        fig_anim, ax_anim = plt.subplots(figsize=(6, 6))
        pad = 0.25 * (params["l1"] + params["l2"])
        xmin = min(np.min(base[:, 0]), np.min(elbow[:, 0]), np.min(tip[:, 0])) - pad
        xmax = max(np.max(base[:, 0]), np.max(elbow[:, 0]), np.max(tip[:, 0])) + pad
        ymin = min(np.min(base[:, 1]), np.min(elbow[:, 1]), np.min(tip[:, 1])) - pad
        ymax = max(np.max(base[:, 1]), np.max(elbow[:, 1]), np.max(tip[:, 1])) + pad
        ax_anim.set_xlim(xmin, xmax)
        ax_anim.set_ylim(ymin, ymax)
        ax_anim.set_aspect("equal")
        ax_anim.grid(True)
        ax_anim.set_title(f"Acrobot Motion ({preferred_method})")
        line, = ax_anim.plot([], [], "-o", lw=2)

        def update(frame):
            pts = np.vstack([base[frame], elbow[frame], tip[frame]])
            line.set_data(pts[:, 0], pts[:, 1])
            return (line,)

        ani = FuncAnimation(fig_anim, update, frames=N + 1, interval=max(1, int(1000 * dt)), blit=True)
        try:
            ani.save("./figs/" + prefix_str + "acrobot.gif", writer=PillowWriter(fps=max(1, int(round(1.0 / dt)))))
        except Exception as animation_error:
            print(f"Animation export skipped: {animation_error}")
        plt.close(fig_anim)
    except Exception as plot_error:
        print(f"Plot generation skipped: {plot_error}")

    print(f"Optimization completed in {elapsed_time:.5f}s")
    print(f"Objective value: {result_str}")

    total_time = time.time() - total_start
    print(f"\nTotal time: {total_time:.5f} s")
    with open(log_path, "a") as log_file:
        log_file.write(f"post-processing method={preferred_method}\n")
        log_file.write(f"total time={total_time:.5f}\n")


if __name__ == "__main__":
    main()
