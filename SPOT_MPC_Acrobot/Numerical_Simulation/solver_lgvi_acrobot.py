from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
from scipy.optimize import root

try:
    from Numerical_Simulation.lie_group_so2 import (
        F_from_delta,
        angle_from_R,
        orth_error_so2,
        det_error_so2,
    )

    from Numerical_Simulation.Discrete_Mechanical_Models_Lie_Groups.model_acrobot_so2 import (
        AcrobotSO2Model,
    )

except ImportError:
    # Allows running this file from inside Numerical_Simulation.
    from lie_group_so2 import (
        F_from_delta,
        angle_from_R,
        orth_error_so2,
        det_error_so2,
    )

    from Discrete_Mechanical_Models_Lie_Groups.model_acrobot_so2 import (
        AcrobotSO2Model,
    )


@dataclass
class AcrobotReducedState:
    """
    Reduced Option-B state for the SDP-matching SO(2) acrobot simulator.

    This state matches the reduced SDP logic:

        R1_k, R2_k:
            absolute rotations at node k

        F1_prev, F2_prev:
            previous step rotations F_{i,k-1}

    No independent x or v variables are stored.

    COM positions X can always be reconstructed from R1, R2 and the geometry
    in model_acrobot_so2.py.
    """

    R1: np.ndarray
    R2: np.ndarray
    F1_prev: np.ndarray
    F2_prev: np.ndarray

    def __post_init__(self) -> None:
        self.R1 = np.asarray(self.R1, dtype=float).reshape(2, 2)
        self.R2 = np.asarray(self.R2, dtype=float).reshape(2, 2)
        self.F1_prev = np.asarray(self.F1_prev, dtype=float).reshape(2, 2)
        self.F2_prev = np.asarray(self.F2_prev, dtype=float).reshape(2, 2)


# Backward-compatible name.
# The state is now reduced, but this alias prevents older imports from exploding.
AcrobotLGVIState = AcrobotReducedState


@dataclass
class LGVIStepInfo:
    """
    Diagnostic information for one reduced implicit step.
    """

    success: bool
    residual_inf: float
    nfev: int
    message: str


def _require_option_b_model(model: AcrobotSO2Model) -> None:
    """
    Check that the model file contains the Option-B reduced dynamics methods.

    If this fails, your model_acrobot_so2.py is still the old geometry-only file
    or an older maximal-coordinate version.
    """
    required = [
        "reduced_step_residual",
        "initial_step_guess",
        "advance_reduced_state",
        "reconstruct_positions_from_rotations",
        "scalars_from_step_rotation",
        "angles_from_rotations",
    ]

    missing = [name for name in required if not hasattr(model, name)]

    if missing:
        raise AttributeError(
            "AcrobotSO2Model is missing Option-B method(s): "
            + ", ".join(missing)
            + ". Update model_acrobot_so2.py first."
        )


def make_model_from_params(params: Mapping[str, Any]) -> AcrobotSO2Model:
    """
    Build numerical simulation model from the shared YAML-derived params dict.

    This is the bridge:

        YAML -> config_loader.py -> params -> AcrobotSO2Model

    So the physical parameters are changed only in the YAML file.
    """
    return AcrobotSO2Model.from_params_dict(params)


def _get_nested_or_flat(
    params: Mapping[str, Any],
    flat_key: str,
    block_key: str,
    nested_key: str,
    default: Optional[float] = None,
) -> float:
    """
    Helper for supporting both flattened config_loader params and raw YAML.
    """
    if flat_key in params:
        return float(params[flat_key])

    block = params.get(block_key, {})
    if isinstance(block, Mapping) and nested_key in block:
        return float(block[nested_key])

    if default is not None:
        return float(default)

    raise KeyError(
        f"Could not find '{flat_key}' in flattened params or "
        f"'{block_key}.{nested_key}' in raw YAML."
    )


def make_reduced_state_from_absolute(
    model: AcrobotSO2Model,
    h: float,
    thetaR: np.ndarray,
    thetaRdot: np.ndarray,
) -> AcrobotReducedState:
    """
    Create reduced state from absolute body-frame angles.

    thetaR:
        [thetaR1, thetaR2]

    thetaRdot:
        [thetaR1dot, thetaR2dot]

    Important:
        No relative acrobot angles are used.
        There is no thetaR2 = theta1 + theta2 conversion here.
    """
    thetaR = np.asarray(thetaR, dtype=float).reshape(2)
    thetaRdot = np.asarray(thetaRdot, dtype=float).reshape(2)

    R1, R2 = model.rotations_from_angles(thetaR[0], thetaR[1])

    F1_prev = F_from_delta(float(h) * thetaRdot[0])
    F2_prev = F_from_delta(float(h) * thetaRdot[1])

    return AcrobotReducedState(
        R1=R1,
        R2=R2,
        F1_prev=F1_prev,
        F2_prev=F2_prev,
    )


def make_initial_state_from_params(
    params: Mapping[str, Any],
    model: Optional[AcrobotSO2Model] = None,
    h_key: str = "dt_sim",
) -> Tuple[AcrobotSO2Model, AcrobotReducedState]:
    """
    Build initial model and reduced state from the YAML-derived params.

    Expected flattened keys from config_loader.py:

        thetaR1_0
        thetaR2_0
        thetaR1dot_0
        thetaR2dot_0
        dt_sim

    Also supports raw YAML style:

        initial:
          thetaR1_deg: ...
          thetaR2_deg: ...
          thetaR1dot: ...
          thetaR2dot: ...

        time:
          dt_sim: ...
    """
    if model is None:
        model = make_model_from_params(params)

    _require_option_b_model(model)

    if h_key in params:
        h = float(params[h_key])
    elif "time" in params and isinstance(params["time"], Mapping) and h_key in params["time"]:
        h = float(params["time"][h_key])
    elif "dt" in params:
        h = float(params["dt"])
    else:
        raise KeyError(
            f"Could not find time step '{h_key}' or fallback key 'dt' in params."
        )

    if "thetaR1_0" in params and "thetaR2_0" in params:
        thetaR = np.array(
            [
                float(params["thetaR1_0"]),
                float(params["thetaR2_0"]),
            ],
            dtype=float,
        )
    else:
        thetaR = np.array(
            [
                np.deg2rad(_get_nested_or_flat(params, "thetaR1_deg", "initial", "thetaR1_deg")),
                np.deg2rad(_get_nested_or_flat(params, "thetaR2_deg", "initial", "thetaR2_deg")),
            ],
            dtype=float,
        )

    thetaRdot = np.array(
        [
            _get_nested_or_flat(params, "thetaR1dot_0", "initial", "thetaR1dot", default=0.0),
            _get_nested_or_flat(params, "thetaR2dot_0", "initial", "thetaR2dot", default=0.0),
        ],
        dtype=float,
    )

    state = make_reduced_state_from_absolute(
        model=model,
        h=h,
        thetaR=thetaR,
        thetaRdot=thetaRdot,
    )

    return model, state


def reconstruct_X_from_R(
    model: AcrobotSO2Model,
    R1: np.ndarray,
    R2: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct maximal COM coordinates X = [x1; x2] from R1, R2.

    This is diagnostic / plotting only for Option B.
    X is not an independent simulation state.
    """
    return model.reconstruct_positions_from_rotations(R1, R2)


def _normalize_reduced_residual(
    residual: np.ndarray,
    model: AcrobotSO2Model,
) -> np.ndarray:
    """
    Optional scaling for the reduced residual.

    The zero set is unchanged. This only helps scipy.root conditioning.
    """
    residual = np.asarray(residual, dtype=float).reshape(8)

    trans_scale = max(
        1.0,
        abs(model.m1 * model.d1_com),
        abs(model.m2 * model.d1_elbow),
        abs(model.m2 * model.d2_com),
    )

    rot_scale = max(
        1.0,
        abs(model.J1),
        abs(model.J2),
    )

    scale = np.array(
        [
            trans_scale,
            trans_scale,
            trans_scale,
            trans_scale,
            rot_scale,
            rot_scale,
            1.0,
            1.0,
        ],
        dtype=float,
    )

    return residual / scale


def acrobot_reduced_step_residual(
    z: np.ndarray,
    model: AcrobotSO2Model,
    h: float,
    R1_k: np.ndarray,
    R2_k: np.ndarray,
    F1_prev: np.ndarray,
    F2_prev: np.ndarray,
    u_k: float,
    normalized: bool = False,
) -> np.ndarray:
    """
    Reduced one-step residual matching the SDP dynamics.

    Unknown vector:

        z = [
            a1_k,
            b1_k,
            a2_k,
            b2_k,
            lam0_x,
            lam0_y,
            lam12_x,
            lam12_y,
        ]

    Equations:
        4 reduced translational dynamics
        2 reduced rotational dynamics
        2 SO(2) step constraints

    No X_next is solved.
    """
    _require_option_b_model(model)

    residual = model.reduced_step_residual(
        z=z,
        R1_k=R1_k,
        R2_k=R2_k,
        F1_prev=F1_prev,
        F2_prev=F2_prev,
        u_k=u_k,
        h=h,
    )

    if normalized:
        residual = _normalize_reduced_residual(residual, model)

    return residual


def initial_guess_from_previous(
    model: AcrobotSO2Model,
    state: AcrobotReducedState,
    previous_z: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Initial guess for the 8-dimensional reduced root solve.

    If previous_z is available, use it as warm start.
    Otherwise use previous step rotations and zero multipliers.
    """
    if previous_z is not None:
        previous_z = np.asarray(previous_z, dtype=float).reshape(8)
        return previous_z.copy()

    return model.initial_step_guess(
        F1_prev=state.F1_prev,
        F2_prev=state.F2_prev,
    )


def lgvi_one_step(
    model: AcrobotSO2Model,
    h: float,
    state: AcrobotReducedState,
    u_k: float,
    z_guess: Optional[np.ndarray] = None,
    root_tol: float = 1e-10,
    maxfev: int = 100,
    normalized: bool = False,
    accept_residual: float = 1e-7,
) -> Tuple[AcrobotReducedState, LGVIStepInfo, np.ndarray]:
    """
    Propagate the reduced acrobot by one SDP-matching implicit step.

    Input state at node k:

        R1_k, R2_k, F1_{k-1}, F2_{k-1}

    Unknown solved by scipy.root:

        z = [a1_k, b1_k, a2_k, b2_k, lam0x, lam0y, lam12x, lam12y]

    Output state at node k+1:

        R1_{k+1}, R2_{k+1}, F1_k, F2_k

    This is the correct Option-B replacement for the old X_next-based solver.
    """
    _require_option_b_model(model)

    h = float(h)
    u_k = float(u_k)

    if z_guess is None:
        z_guess = initial_guess_from_previous(
            model=model,
            state=state,
            previous_z=None,
        )

    def fun(z: np.ndarray) -> np.ndarray:
        return acrobot_reduced_step_residual(
            z=z,
            model=model,
            h=h,
            R1_k=state.R1,
            R2_k=state.R2,
            F1_prev=state.F1_prev,
            F2_prev=state.F2_prev,
            u_k=u_k,
            normalized=normalized,
        )

    sol = root(
        fun,
        np.asarray(z_guess, dtype=float).reshape(8),
        method="hybr",
        options={
            "xtol": root_tol,
            "maxfev": maxfev,
        },
    )

    residual = fun(sol.x)
    residual_inf = float(np.linalg.norm(residual, ord=np.inf))

    info = LGVIStepInfo(
        success=bool(sol.success),
        residual_inf=residual_inf,
        nfev=int(sol.nfev),
        message=str(sol.message),
    )

    if not sol.success and residual_inf > accept_residual:
        raise RuntimeError(
            f"Reduced LGVI one-step solve failed: "
            f"success={sol.success}, "
            f"||r||_inf={residual_inf:.3e}, "
            f"message={sol.message}"
        )

    z = np.asarray(sol.x, dtype=float).reshape(8)

    R1_next, R2_next, F1_k, F2_k, _, _ = model.advance_reduced_state(
        R1_k=state.R1,
        R2_k=state.R2,
        z=z,
    )

    next_state = AcrobotReducedState(
        R1=R1_next,
        R2=R2_next,
        F1_prev=F1_k,
        F2_prev=F2_k,
    )

    return next_state, info, z


def rollout_lgvi_controls(
    model: AcrobotSO2Model,
    h: float,
    initial_state: AcrobotReducedState,
    u_sequence: np.ndarray,
    root_tol: float = 1e-10,
    maxfev: int = 100,
    normalized: bool = False,
    accept_residual: float = 1e-7,
) -> Dict[str, Any]:
    """
    Roll out reduced SDP-matching dynamics for a given torque sequence.

    This keeps the old function name `rollout_lgvi_controls` so your surrounding
    code does not need to change immediately.

    Internally, this is now Option B:

        state = (R1, R2, F1_prev, F2_prev)

    not:

        state = (X_prev, X, R1, R2, F1_prev, F2_prev)
    """
    _require_option_b_model(model)

    u_sequence = np.asarray(u_sequence, dtype=float).reshape(-1)
    num_steps = int(len(u_sequence))

    R1 = np.zeros((num_steps + 1, 2, 2), dtype=float)
    R2 = np.zeros((num_steps + 1, 2, 2), dtype=float)

    F1 = np.zeros((num_steps, 2, 2), dtype=float)
    F2 = np.zeros((num_steps, 2, 2), dtype=float)

    X = np.zeros((num_steps + 1, 4), dtype=float)

    thetaR = np.zeros((num_steps + 1, 2), dtype=float)
    dtheta = np.zeros((num_steps, 2), dtype=float)

    lam0 = np.zeros((num_steps, 2), dtype=float)
    lam12 = np.zeros((num_steps, 2), dtype=float)

    residual_inf = np.zeros(num_steps, dtype=float)

    infos: List[LGVIStepInfo] = []
    z_solutions: List[np.ndarray] = []

    state = initial_state

    R1[0] = state.R1
    R2[0] = state.R2
    X[0] = reconstruct_X_from_R(model, state.R1, state.R2)
    thetaR[0] = model.angles_from_rotations(state.R1, state.R2)

    z_guess: Optional[np.ndarray] = None

    for k, u_k in enumerate(u_sequence):
        state_next, info, z = lgvi_one_step(
            model=model,
            h=h,
            state=state,
            u_k=float(u_k),
            z_guess=z_guess,
            root_tol=root_tol,
            maxfev=maxfev,
            normalized=normalized,
            accept_residual=accept_residual,
        )

        F1_k, F2_k, lam0_k, lam12_k = model.unpack_reduced_solution(z)

        R1[k + 1] = state_next.R1
        R2[k + 1] = state_next.R2

        F1[k] = F1_k
        F2[k] = F2_k

        X[k + 1] = reconstruct_X_from_R(model, state_next.R1, state_next.R2)

        thetaR[k + 1] = model.angles_from_rotations(
            state_next.R1,
            state_next.R2,
        )

        dtheta[k, 0] = angle_from_R(F1_k)
        dtheta[k, 1] = angle_from_R(F2_k)

        lam0[k] = lam0_k
        lam12[k] = lam12_k

        residual_inf[k] = info.residual_inf

        infos.append(info)
        z_solutions.append(z)

        # Warm-start next root solve with current solution.
        z_guess = z.copy()
        state = state_next

    return {
        "t": np.arange(num_steps + 1, dtype=float) * float(h),
        "X": X,
        "R1": R1,
        "R2": R2,
        "F1": F1,
        "F2": F2,
        "thetaR": thetaR,
        "dtheta": dtheta,
        "lambda0": lam0,
        "lambda12": lam12,
        "u": u_sequence,
        "residual_inf": residual_inf,
        "infos": infos,
        "z_solutions": z_solutions,
        "final_state": state,
    }


def simulate_one_control_interval(
    model: AcrobotSO2Model,
    state: AcrobotReducedState,
    u_j: float,
    dt_control: float,
    dt_sim: float,
    root_tol: float = 1e-10,
    maxfev: int = 100,
    normalized: bool = False,
    accept_residual: float = 1e-7,
) -> Tuple[AcrobotReducedState, Dict[str, Any]]:
    """
    Simulate one MPC control interval.

    The SDP provides one control input u_j for:

        [t_j, t_j + dt_control]

    The simulator applies this constant torque over many small dt_sim steps.

    Example:
        dt_control = 0.1
        dt_sim     = 0.001

    Then:
        n_substeps = 100

    Important:
        Do not scale u_j by 100.
        The smaller h = dt_sim is already used inside the reduced dynamics.
    """
    _require_option_b_model(model)

    dt_control = float(dt_control)
    dt_sim = float(dt_sim)

    ratio = dt_control / dt_sim
    n_substeps = int(round(ratio))

    if abs(ratio - n_substeps) > 1e-10:
        raise ValueError(
            f"dt_control / dt_sim must be an integer. "
            f"Got {dt_control} / {dt_sim} = {ratio}"
        )

    u_sequence = np.full(n_substeps, float(u_j), dtype=float)

    sim = rollout_lgvi_controls(
        model=model,
        h=dt_sim,
        initial_state=state,
        u_sequence=u_sequence,
        root_tol=root_tol,
        maxfev=maxfev,
        normalized=normalized,
        accept_residual=accept_residual,
    )

    return sim["final_state"], sim


def simulate_one_control_interval_from_params(
    params: Mapping[str, Any],
    model: AcrobotSO2Model,
    state: AcrobotReducedState,
    u_j: float,
    root_tol: float = 1e-10,
    maxfev: int = 100,
    normalized: bool = False,
    accept_residual: float = 1e-7,
) -> Tuple[AcrobotReducedState, Dict[str, Any]]:
    """
    Convenience wrapper using YAML-derived params.

    Expected keys:
        dt_sim
        control_interval

    Raw YAML fallback:
        time:
          dt_sim
          control_interval
    """
    if "dt_sim" in params:
        dt_sim = float(params["dt_sim"])
    else:
        dt_sim = float(params["time"]["dt_sim"])

    if "control_interval" in params:
        dt_control = float(params["control_interval"])
    else:
        dt_control = float(params["time"].get("control_interval", params["time"]["dt_sdp"]))

    return simulate_one_control_interval(
        model=model,
        state=state,
        u_j=u_j,
        dt_control=dt_control,
        dt_sim=dt_sim,
        root_tol=root_tol,
        maxfev=maxfev,
        normalized=normalized,
        accept_residual=accept_residual,
    )


def simulate_lgvi_acrobot(
    model: AcrobotSO2Model,
    h: float,
    steps: int,
    alpha0: np.ndarray,
    omega0: np.ndarray,
    u_fun: Optional[Any] = None,
    first_step: str = "reduced",
    root_tol: float = 1e-10,
    maxfev: int = 100,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Backward-compatible rollout interface.

    This no longer uses RK4 or minimal-coordinate dynamics.

    It initializes the reduced state from absolute angles and absolute angular
    velocities, then rolls out the Option-B reduced SDP-matching dynamics.

    Parameters
    ----------
    alpha0:
        Absolute initial angles [thetaR1, thetaR2].

    omega0:
        Absolute angular velocities [thetaR1dot, thetaR2dot].

    u_fun:
        Optional function u_fun(t).

    first_step:
        Kept only for backward compatibility. Ignored.
    """
    if steps < 1:
        raise ValueError("steps must be at least 1")

    if first_step.lower() not in {"reduced", "euler", "rk4"} and verbose:
        print(
            f"[simulate_lgvi_acrobot] first_step='{first_step}' ignored. "
            "Using reduced Option-B initialization."
        )

    initial_state = make_reduced_state_from_absolute(
        model=model,
        h=h,
        thetaR=np.asarray(alpha0, dtype=float).reshape(2),
        thetaRdot=np.asarray(omega0, dtype=float).reshape(2),
    )

    if u_fun is None:
        u_sequence = np.zeros(steps, dtype=float)
    else:
        u_sequence = np.array(
            [float(u_fun(k * h)) for k in range(steps)],
            dtype=float,
        )

    sim = rollout_lgvi_controls(
        model=model,
        h=h,
        initial_state=initial_state,
        u_sequence=u_sequence,
        root_tol=root_tol,
        maxfev=maxfev,
    )

    if verbose:
        print(
            "[simulate_lgvi_acrobot] max residual:",
            float(np.max(sim["residual_inf"])) if len(sim["residual_inf"]) else np.nan,
        )

    return sim


def get_absolute_angles_and_rates(
    state: AcrobotReducedState,
    h: float,
) -> Dict[str, float]:
    """
    Extract absolute angles and approximate angular rates from reduced state.

    thetaR_i:
        angle from R_i

    thetaRdot_i:
        approximated from previous step:
            thetaRdot_i ~= angle(F_i_prev) / h
    """
    h = float(h)

    thetaR1 = float(angle_from_R(state.R1))
    thetaR2 = float(angle_from_R(state.R2))

    thetaR1dot = float(angle_from_R(state.F1_prev)) / h
    thetaR2dot = float(angle_from_R(state.F2_prev)) / h

    return {
        "thetaR1": thetaR1,
        "thetaR2": thetaR2,
        "thetaR1dot": thetaR1dot,
        "thetaR2dot": thetaR2dot,
    }


def convert_state_to_sdp_initial(
    state: AcrobotReducedState,
    dt_physical: float,
    dt_sdp: float,
) -> Dict[str, np.ndarray | float]:
    """
    Convert a fine simulation state into an SDP-compatible initial state.

    Why:
        The physical MPC rollout may use dt_sim = 0.001.
        The SDP prediction may use dt_sdp = 0.1.

    The rotations R1, R2 are unchanged, but the previous F must encode the
    same angular velocity over the larger SDP step:

        omega_i ~= angle(F_i_prev_sim) / dt_sim
        F_i_prev_sdp = exp(dt_sdp * omega_i)
    """
    rates = get_absolute_angles_and_rates(
        state=state,
        h=dt_physical,
    )

    F1_prev_sdp = F_from_delta(float(dt_sdp) * rates["thetaR1dot"])
    F2_prev_sdp = F_from_delta(float(dt_sdp) * rates["thetaR2dot"])

    return {
        "R1": state.R1.copy(),
        "R2": state.R2.copy(),
        "F1_prev": F1_prev_sdp,
        "F2_prev": F2_prev_sdp,
        "thetaR1": rates["thetaR1"],
        "thetaR2": rates["thetaR2"],
        "thetaR1dot": rates["thetaR1dot"],
        "thetaR2dot": rates["thetaR2dot"],
    }


def convert_state_to_sdp_initial_scalars(
    state: AcrobotReducedState,
    model: AcrobotSO2Model,
    dt_physical: float,
    dt_sdp: float,
) -> Dict[str, float]:
    """
    Convert reduced state to scalar values useful for fixing SDP initial data.

    Returns:
        c1_0, s1_0, c2_0, s2_0
        a1_prev, b1_prev, a2_prev, b2_prev

    The previous F values are rescaled from dt_physical to dt_sdp.
    """
    converted = convert_state_to_sdp_initial(
        state=state,
        dt_physical=dt_physical,
        dt_sdp=dt_sdp,
    )

    R1 = np.asarray(converted["R1"], dtype=float).reshape(2, 2)
    R2 = np.asarray(converted["R2"], dtype=float).reshape(2, 2)

    F1_prev = np.asarray(converted["F1_prev"], dtype=float).reshape(2, 2)
    F2_prev = np.asarray(converted["F2_prev"], dtype=float).reshape(2, 2)

    c1_0, s1_0 = model.scalars_from_rotation(R1)
    c2_0, s2_0 = model.scalars_from_rotation(R2)

    a1_prev, b1_prev = model.scalars_from_step_rotation(F1_prev)
    a2_prev, b2_prev = model.scalars_from_step_rotation(F2_prev)

    return {
        "c1_0": c1_0,
        "s1_0": s1_0,
        "c2_0": c2_0,
        "s2_0": s2_0,
        "a1_prev": a1_prev,
        "b1_prev": b1_prev,
        "a2_prev": a2_prev,
        "b2_prev": b2_prev,
        "thetaR1": float(converted["thetaR1"]),
        "thetaR2": float(converted["thetaR2"]),
        "thetaR1dot": float(converted["thetaR1dot"]),
        "thetaR2dot": float(converted["thetaR2dot"]),
    }


def diagnostics_lgvi(
    model: AcrobotSO2Model,
    sim: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    """
    Diagnostics for a reduced Option-B rollout.

    This computes:
        holonomic constraint norms from reconstructed X,
        SO(2) orthogonality/determinant errors,
        absolute angles,
        approximate angular rates,
        approximate energy if model provides energy_from_reduced_state.
    """
    R1 = np.asarray(sim["R1"], dtype=float)
    R2 = np.asarray(sim["R2"], dtype=float)
    F1 = np.asarray(sim["F1"], dtype=float)
    F2 = np.asarray(sim["F2"], dtype=float)
    X = np.asarray(sim["X"], dtype=float)
    t = np.asarray(sim["t"], dtype=float)

    num_nodes = R1.shape[0]
    num_steps = max(0, num_nodes - 1)

    h = float(t[1] - t[0]) if len(t) > 1 else 1.0

    phi_norm = np.zeros(num_nodes, dtype=float)
    phi0_norm = np.zeros(num_nodes, dtype=float)
    phi12_norm = np.zeros(num_nodes, dtype=float)

    orth_R1 = np.zeros(num_nodes, dtype=float)
    orth_R2 = np.zeros(num_nodes, dtype=float)

    det_R1 = np.zeros(num_nodes, dtype=float)
    det_R2 = np.zeros(num_nodes, dtype=float)

    thetaR = np.zeros((num_nodes, 2), dtype=float)

    for k in range(num_nodes):
        phi = model.constraints(X[k], R1[k], R2[k])

        phi0_norm[k] = float(np.linalg.norm(phi[0:2]))
        phi12_norm[k] = float(np.linalg.norm(phi[2:4]))
        phi_norm[k] = float(np.linalg.norm(phi))

        orth_R1[k] = float(orth_error_so2(R1[k]))
        orth_R2[k] = float(orth_error_so2(R2[k]))

        det_R1[k] = float(det_error_so2(R1[k]))
        det_R2[k] = float(det_error_so2(R2[k]))

        thetaR[k] = model.angles_from_rotations(R1[k], R2[k])

    omega = np.zeros((num_steps, 2), dtype=float)
    energy = np.full(num_steps, np.nan, dtype=float)

    for k in range(num_steps):
        omega[k, 0] = float(angle_from_R(F1[k])) / h
        omega[k, 1] = float(angle_from_R(F2[k])) / h

        if hasattr(model, "energy_from_reduced_state"):
            energy[k] = model.energy_from_reduced_state(
                R1=R1[k],
                R2=R2[k],
                F1_prev=F1[k],
                F2_prev=F2[k],
                h=h,
            )

    if num_steps > 0 and np.isfinite(energy[0]):
        energy_error = energy - energy[0]
    else:
        energy_error = energy.copy()

    return {
        "phi_norm": phi_norm,
        "phi0_norm": phi0_norm,
        "phi12_norm": phi12_norm,
        "orth_R1": orth_R1,
        "orth_R2": orth_R2,
        "det_R1": det_R1,
        "det_R2": det_R2,
        "thetaR": thetaR,
        "omega": omega,
        "energy": energy,
        "energy_error": energy_error,
    }


def print_step_summary(
    state: AcrobotReducedState,
    model: AcrobotSO2Model,
    h: float,
    label: str = "state",
) -> None:
    """
    Small debugging helper.
    """
    X = reconstruct_X_from_R(model, state.R1, state.R2)
    theta = model.angles_from_rotations(state.R1, state.R2)
    rates = get_absolute_angles_and_rates(state, h=h)

    print(f"[{label}]")
    print(f"  thetaR1 = {theta[0]: .6f} rad, {np.rad2deg(theta[0]): .3f} deg")
    print(f"  thetaR2 = {theta[1]: .6f} rad, {np.rad2deg(theta[1]): .3f} deg")
    print(f"  thetaR1dot ~= {rates['thetaR1dot']: .6f} rad/s")
    print(f"  thetaR2dot ~= {rates['thetaR2dot']: .6f} rad/s")
    print(f"  X = {X}")
    print(f"  constraint norm = {model.constraint_norm(X, state.R1, state.R2):.3e}")