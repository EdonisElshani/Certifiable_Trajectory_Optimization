from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
from scipy.optimize import root

try:
    from Numerical_Simulation.lie_group_so2 import (
        F_from_delta,
        F_from_cayley,
        angle_from_R,
        angle_from_cayley,
        cayley_from_R,
        cayley_to_ab,
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
        F_from_cayley,
        angle_from_R,
        angle_from_cayley,
        cayley_from_R,
        cayley_to_ab,
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

    R1, R2:
        Absolute rotations at node k.

    F1_prev, F2_prev:
        Previous step rotations F_{i,k-1}.

    No independent x, v, or angular-rate state is stored.
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
    accepted_by_residual: bool = False
    method: str = "ab"
    thetaF1_deg: float = np.nan
    thetaF2_deg: float = np.nan
    q1: float = np.nan
    q2: float = np.nan
    cayley_near_singularity: bool = False
    singularity_margin_deg: float = np.nan
    substepping_performed: bool = False
    substep_depth: int = 0
    num_substeps: int = 1
    h_original: float = np.nan
    h_min_used: float = np.nan
    near_singularity_count: int = 0


class LGVISolveError(RuntimeError):
    """Hard LGVI solve failure with diagnostics for pipeline logging."""

    def __init__(self, residual_inf: float, message: str, nfev: int) -> None:
        self.residual_inf = float(residual_inf)
        self.solver_message = str(message)
        self.nfev = int(nfev)
        self.local_sim_step: Optional[int] = None
        self.accepted_failures_before_hard_failure: List[Tuple[int, float]] = []
        super().__init__(
            "Reduced LGVI one-step solve failed: "
            f"success=False, ||r||_inf={self.residual_inf:.3e}, "
            f"message={self.solver_message}"
        )


def _require_option_b_model(model: AcrobotSO2Model) -> None:
    """
    Check that the model file contains the Option-B reduced dynamics methods.
    """
    required = [
        "reduced_step_residual",
        "initial_step_guess",
        "advance_reduced_state",
        "reconstruct_positions_from_rotations",
        "scalars_from_step_rotation",
        "scalars_from_rotation",
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
    """
    return AcrobotSO2Model.from_params_dict(params)


def _get_time_step(params: Mapping[str, Any], h_key: str) -> float:
    """
    Support flattened params and, as fallback, raw YAML-style params.
    """
    if h_key in params:
        return float(params[h_key])

    if "time" in params and isinstance(params["time"], Mapping):
        if h_key in params["time"]:
            return float(params["time"][h_key])

    if "dt" in params:
        return float(params["dt"])

    raise KeyError(
        f"Could not find time step '{h_key}', 'time.{h_key}', or fallback key 'dt'."
    )


def make_reduced_state_from_absolute(
    model: AcrobotSO2Model,
    h: float,
    thetaR: np.ndarray,
    thetaF: np.ndarray,
) -> AcrobotReducedState:
    """
    Create reduced state from absolute R angles and previous F step angles.

    thetaR:
        [thetaR1, thetaR2]

    thetaF:
        [thetaF1, thetaF2]

    The argument h is kept for call compatibility; thetaF already represents the
    step angle of F for the chosen time step.
    """
    _require_option_b_model(model)

    thetaR = np.asarray(thetaR, dtype=float).reshape(2)
    thetaF = np.asarray(thetaF, dtype=float).reshape(2)

    R1, R2 = model.rotations_from_angles(thetaR[0], thetaR[1])

    F1_prev = F_from_delta(thetaF[0])
    F2_prev = F_from_delta(thetaF[1])

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
    Build initial model and reduced state from params.

    Uses the old thesis convention:
        thetaR1_0, thetaR2_0
        thetaF1_0, thetaF2_0

    thetaF_i,0 is the step angle of F_i,0 from the YAML-derived params.
    """
    if model is None:
        model = make_model_from_params(params)

    _require_option_b_model(model)

    h = _get_time_step(params, h_key)

    thetaR = np.array(
        [
            float(params["thetaR1_0"]),
            float(params["thetaR2_0"]),
        ],
        dtype=float,
    )

    thetaF_sdp = np.array(
        [
            float(params["thetaF1_0"]),
            float(params["thetaF2_0"]),
        ],
        dtype=float,
    )

    # thetaF_i,0 is defined for dt_sdp. If we initialize a fine simulator with
    # dt_sim, rescale the step angle so the physical motion is consistent.
    h_ref = float(params.get("dt_sdp", h))
    thetaF_for_h = thetaF_sdp * (h / h_ref)

    state = make_reduced_state_from_absolute(
        model=model,
        h=h,
        thetaR=thetaR,
        thetaF=thetaF_for_h,
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
        abs(model.rot_inertia_1),
        abs(model.rot_inertia_2),
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
    lgvi_maxfev: int = 2000,
    normalized: bool = False,
    accept_residual: bool = True,
    accept_residual_tol: float = 1e-3,
) -> Tuple[AcrobotReducedState, LGVIStepInfo, np.ndarray]:
    """
    Propagate the reduced acrobot by one SDP-matching implicit step.

    Input state at node k:
        R1_k, R2_k, F1_{k-1}, F2_{k-1}

    Unknown solved by scipy.root:
        z = [a1_k, b1_k, a2_k, b2_k, lam0x, lam0y, lam12x, lam12y]

    Output state at node k+1:
        R1_{k+1}, R2_{k+1}, F1_k, F2_k
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
            "maxfev": int(lgvi_maxfev),
        },
    )

    residual = fun(sol.x)
    residual_inf = float(np.linalg.norm(residual, ord=np.inf))

    accepted_by_residual = bool(
        not sol.success
        and accept_residual
        and np.isfinite(residual_inf)
        and residual_inf <= float(accept_residual_tol)
    )

    info = LGVIStepInfo(
        success=bool(sol.success),
        residual_inf=residual_inf,
        nfev=int(sol.nfev),
        message=str(sol.message),
        accepted_by_residual=accepted_by_residual,
        method="ab",
    )

    if not sol.success and not accepted_by_residual:
        raise LGVISolveError(
            residual_inf=residual_inf,
            message=str(sol.message),
            nfev=int(sol.nfev),
        )

    z = np.asarray(sol.x, dtype=float).reshape(8)

    R1_next, R2_next, F1_k, F2_k, _, _ = model.advance_reduced_state(
        R1_k=state.R1,
        R2_k=state.R2,
        z=z,
    )

    info.thetaF1_deg = float(np.rad2deg(angle_from_R(F1_k)))
    info.thetaF2_deg = float(np.rad2deg(angle_from_R(F2_k)))
    info.h_original = h
    info.h_min_used = h

    next_state = AcrobotReducedState(
        R1=R1_next,
        R2=R2_next,
        F1_prev=F1_k,
        F2_prev=F2_k,
    )

    return next_state, info, z


def cayley_residual_to_ab_solution(y: np.ndarray) -> np.ndarray:
    """
    Convert Cayley unknowns [q1,q2,lambda0,lambda12] to the AB z vector.
    """
    y = np.asarray(y, dtype=float).reshape(6)
    a1, b1 = cayley_to_ab(y[0])
    a2, b2 = cayley_to_ab(y[1])
    return np.array([a1, b1, a2, b2, y[2], y[3], y[4], y[5]], dtype=float)


def acrobot_reduced_step_residual_cayley(
    y: np.ndarray,
    model: AcrobotSO2Model,
    h: float,
    state: AcrobotReducedState,
    u_k: float,
) -> np.ndarray:
    """
    Six-equation Cayley residual: reduced dynamics without explicit SO(2) rows.
    """
    z_ab = cayley_residual_to_ab_solution(y)
    residual = model.reduced_step_residual(
        z=z_ab,
        R1_k=state.R1,
        R2_k=state.R2,
        F1_prev=state.F1_prev,
        F2_prev=state.F2_prev,
        u_k=float(u_k),
        h=float(h),
    )
    return np.asarray(residual, dtype=float).reshape(8)[:6]


def initial_guess_cayley_from_previous(
    state: AcrobotReducedState,
    y_guess: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Initial guess for the 6-dimensional Cayley root solve.
    """
    if y_guess is not None:
        return np.asarray(y_guess, dtype=float).reshape(6).copy()

    return np.array(
        [
            cayley_from_R(state.F1_prev),
            cayley_from_R(state.F2_prev),
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=float,
    )


def _cayley_near_singularity(
    q1: float,
    q2: float,
    singularity_margin_deg: float,
) -> Tuple[bool, int]:
    margin = np.deg2rad(float(singularity_margin_deg))
    threshold = np.pi - margin
    theta1 = abs(angle_from_cayley(q1))
    theta2 = abs(angle_from_cayley(q2))
    flags = [theta1 > threshold, theta2 > threshold]
    return any(flags), int(sum(flags))


def lgvi_one_step_cayley(
    model: AcrobotSO2Model,
    h: float,
    state: AcrobotReducedState,
    u_k: float,
    y_guess: Optional[np.ndarray] = None,
    root_tol: float = 1e-12,
    lgvi_maxfev: int = 5000,
    accept_residual: bool = True,
    accept_residual_tol: float = 1e-10,
    singularity_margin_deg: float = 5.0,
) -> Tuple[AcrobotReducedState, LGVIStepInfo, np.ndarray]:
    """
    Propagate one step using Cayley coordinates for the step rotations.
    """
    _require_option_b_model(model)

    h = float(h)
    u_k = float(u_k)
    y0 = initial_guess_cayley_from_previous(state=state, y_guess=y_guess)

    def fun(y: np.ndarray) -> np.ndarray:
        return acrobot_reduced_step_residual_cayley(
            y=y,
            model=model,
            h=h,
            state=state,
            u_k=u_k,
        )

    sol = root(
        fun,
        y0,
        method="hybr",
        options={"xtol": root_tol, "maxfev": int(lgvi_maxfev)},
    )

    residual = fun(sol.x)
    residual_inf = float(np.linalg.norm(residual, ord=np.inf))
    accepted_by_residual = bool(
        not sol.success
        and accept_residual
        and np.isfinite(residual_inf)
        and residual_inf <= float(accept_residual_tol)
    )

    if not sol.success and not accepted_by_residual:
        raise LGVISolveError(
            residual_inf=residual_inf,
            message=str(sol.message),
            nfev=int(sol.nfev),
        )

    y = np.asarray(sol.x, dtype=float).reshape(6)
    q1 = float(y[0])
    q2 = float(y[1])
    F1 = F_from_cayley(q1)
    F2 = F_from_cayley(q2)
    R1_next = state.R1 @ F1
    R2_next = state.R2 @ F2
    near_singularity, near_count = _cayley_near_singularity(
        q1=q1,
        q2=q2,
        singularity_margin_deg=singularity_margin_deg,
    )

    info = LGVIStepInfo(
        success=bool(sol.success),
        residual_inf=residual_inf,
        nfev=int(sol.nfev),
        message=str(sol.message),
        accepted_by_residual=accepted_by_residual,
        method="cayley",
        thetaF1_deg=float(np.rad2deg(angle_from_cayley(q1))),
        thetaF2_deg=float(np.rad2deg(angle_from_cayley(q2))),
        q1=q1,
        q2=q2,
        cayley_near_singularity=bool(near_singularity),
        singularity_margin_deg=float(singularity_margin_deg),
        h_original=h,
        h_min_used=h,
        near_singularity_count=int(near_count),
    )

    next_state = AcrobotReducedState(
        R1=R1_next,
        R2=R2_next,
        F1_prev=F1,
        F2_prev=F2,
    )

    return next_state, info, cayley_residual_to_ab_solution(y)


def _combined_step_z_from_states(
    model: AcrobotSO2Model,
    state_start: AcrobotReducedState,
    state_end: AcrobotReducedState,
    fallback_z: np.ndarray,
) -> np.ndarray:
    """
    Build an AB-equivalent z for the net rotation from state_start to state_end.
    """
    fallback_z = np.asarray(fallback_z, dtype=float).reshape(8)
    F1_total = state_start.R1.T @ state_end.R1
    F2_total = state_start.R2.T @ state_end.R2
    a1, b1 = model.scalars_from_step_rotation(F1_total)
    a2, b2 = model.scalars_from_step_rotation(F2_total)
    return np.array([a1, b1, a2, b2, fallback_z[4], fallback_z[5], fallback_z[6], fallback_z[7]], dtype=float)


def lgvi_one_step_cayley_safe(
    model: AcrobotSO2Model,
    h: float,
    state: AcrobotReducedState,
    u_k: float,
    y_guess: Optional[np.ndarray] = None,
    root_tol: float = 1e-12,
    lgvi_maxfev: int = 5000,
    accept_residual: bool = True,
    accept_residual_tol: float = 1e-10,
    singularity_margin_deg: float = 5.0,
    allow_substepping: bool = False,
    min_substep_h: float = 1e-6,
    max_subdivisions: int = 10,
    _depth: int = 0,
    _h_original: Optional[float] = None,
) -> Tuple[AcrobotReducedState, LGVIStepInfo, np.ndarray]:
    """
    Cayley one-step solve with optional recursive h -> h/2+h/2 substepping.
    """
    h = float(h)
    h_original = h if _h_original is None else float(_h_original)

    try:
        next_state, info, z = lgvi_one_step_cayley(
            model=model,
            h=h,
            state=state,
            u_k=u_k,
            y_guess=y_guess,
            root_tol=root_tol,
            lgvi_maxfev=lgvi_maxfev,
            accept_residual=accept_residual,
            accept_residual_tol=accept_residual_tol,
            singularity_margin_deg=singularity_margin_deg,
        )
        should_split = bool(info.cayley_near_singularity)
        split_trigger_near_count = int(info.near_singularity_count) if should_split else 0
    except LGVISolveError as exc:
        if not allow_substepping or _depth >= int(max_subdivisions) or 0.5 * h < float(min_substep_h):
            raise
        should_split = True
        split_trigger_near_count = 0
        info = None
        z = None
        next_state = None

    if not allow_substepping or not should_split or _depth >= int(max_subdivisions) or 0.5 * h < float(min_substep_h):
        if info is None or next_state is None or z is None:
            raise RuntimeError("Cayley substepping reached an inconsistent internal state.")
        info.h_original = h_original
        info.h_min_used = h
        info.substep_depth = _depth
        return next_state, info, z

    half_h = 0.5 * h
    mid_state, info_a, z_a = lgvi_one_step_cayley_safe(
        model=model,
        h=half_h,
        state=state,
        u_k=u_k,
        y_guess=None,
        root_tol=root_tol,
        lgvi_maxfev=lgvi_maxfev,
        accept_residual=accept_residual,
        accept_residual_tol=accept_residual_tol,
        singularity_margin_deg=singularity_margin_deg,
        allow_substepping=allow_substepping,
        min_substep_h=min_substep_h,
        max_subdivisions=max_subdivisions,
        _depth=_depth + 1,
        _h_original=h_original,
    )
    end_state, info_b, z_b = lgvi_one_step_cayley_safe(
        model=model,
        h=half_h,
        state=mid_state,
        u_k=u_k,
        y_guess=None,
        root_tol=root_tol,
        lgvi_maxfev=lgvi_maxfev,
        accept_residual=accept_residual,
        accept_residual_tol=accept_residual_tol,
        singularity_margin_deg=singularity_margin_deg,
        allow_substepping=allow_substepping,
        min_substep_h=min_substep_h,
        max_subdivisions=max_subdivisions,
        _depth=_depth + 1,
        _h_original=h_original,
    )

    combined_z = _combined_step_z_from_states(model, state, end_state, z_b)
    residual_inf = max(float(info_a.residual_inf), float(info_b.residual_inf))
    near_count = split_trigger_near_count + int(info_a.near_singularity_count) + int(info_b.near_singularity_count)
    num_substeps = int(info_a.num_substeps) + int(info_b.num_substeps)
    h_min_used = min(float(info_a.h_min_used), float(info_b.h_min_used))
    max_depth = max(int(info_a.substep_depth), int(info_b.substep_depth), _depth + 1)
    q1_total = cayley_from_R(state.R1.T @ end_state.R1)
    q2_total = cayley_from_R(state.R2.T @ end_state.R2)

    combined_info = LGVIStepInfo(
        success=bool(info_a.success and info_b.success),
        residual_inf=residual_inf,
        nfev=int(info_a.nfev) + int(info_b.nfev),
        message=f"substepped: first=({info_a.message}); second=({info_b.message})",
        accepted_by_residual=bool(info_a.accepted_by_residual or info_b.accepted_by_residual),
        method="cayley",
        thetaF1_deg=float(np.rad2deg(angle_from_cayley(q1_total))),
        thetaF2_deg=float(np.rad2deg(angle_from_cayley(q2_total))),
        q1=float(q1_total),
        q2=float(q2_total),
        cayley_near_singularity=bool(split_trigger_near_count > 0 or info_a.cayley_near_singularity or info_b.cayley_near_singularity),
        singularity_margin_deg=float(singularity_margin_deg),
        substepping_performed=True,
        substep_depth=max_depth,
        num_substeps=num_substeps,
        h_original=h_original,
        h_min_used=h_min_used,
        near_singularity_count=near_count,
    )

    return end_state, combined_info, combined_z


def rollout_lgvi_controls(
    model: AcrobotSO2Model,
    h: float,
    initial_state: AcrobotReducedState,
    u_sequence: np.ndarray,
    method: str = "ab",
    allow_substepping: bool = False,
    min_substep_h: float = 1e-6,
    max_subdivisions: int = 10,
    singularity_margin_deg: float = 5.0,
    root_tol: float = 1e-10,
    lgvi_maxfev: int = 2000,
    normalized: bool = False,
    accept_residual: bool = True,
    accept_residual_tol: float = 1e-3,
) -> Dict[str, Any]:
    """
    Roll out reduced SDP-matching dynamics for a given torque sequence.

    State:
        (R1, R2, F1_prev, F2_prev)
    """
    _require_option_b_model(model)

    u_sequence = np.asarray(u_sequence, dtype=float).reshape(-1)
    num_steps = int(len(u_sequence))
    method = str(method).lower()
    if method not in {"ab", "cayley"}:
        raise ValueError(f"Unknown LGVI method '{method}'. Expected 'ab' or 'cayley'.")

    R1 = np.zeros((num_steps + 1, 2, 2), dtype=float)
    R2 = np.zeros((num_steps + 1, 2, 2), dtype=float)

    F1 = np.zeros((num_steps, 2, 2), dtype=float)
    F2 = np.zeros((num_steps, 2, 2), dtype=float)

    X = np.zeros((num_steps + 1, 4), dtype=float)

    thetaR = np.zeros((num_steps + 1, 2), dtype=float)
    thetaF = np.zeros((num_steps, 2), dtype=float)

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
    y_guess: Optional[np.ndarray] = None

    for k, u_k in enumerate(u_sequence):
        try:
            if method == "ab":
                state_next, info, z = lgvi_one_step(
                    model=model,
                    h=h,
                    state=state,
                    u_k=float(u_k),
                    z_guess=z_guess,
                    root_tol=root_tol,
                    lgvi_maxfev=lgvi_maxfev,
                    normalized=normalized,
                    accept_residual=accept_residual,
                    accept_residual_tol=accept_residual_tol,
                )
            else:
                state_next, info, z = lgvi_one_step_cayley_safe(
                    model=model,
                    h=h,
                    state=state,
                    u_k=float(u_k),
                    y_guess=y_guess,
                    root_tol=root_tol,
                    lgvi_maxfev=lgvi_maxfev,
                    accept_residual=accept_residual,
                    accept_residual_tol=accept_residual_tol,
                    singularity_margin_deg=singularity_margin_deg,
                    allow_substepping=allow_substepping,
                    min_substep_h=min_substep_h,
                    max_subdivisions=max_subdivisions,
                )
        except LGVISolveError as exc:
            exc.local_sim_step = k
            exc.accepted_failures_before_hard_failure = [
                (i, step_info.residual_inf)
                for i, step_info in enumerate(infos)
                if step_info.accepted_by_residual
            ]
            raise

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

        thetaF[k, 0] = angle_from_R(F1_k)
        thetaF[k, 1] = angle_from_R(F2_k)

        lam0[k] = lam0_k
        lam12[k] = lam12_k

        residual_inf[k] = info.residual_inf

        infos.append(info)
        z_solutions.append(z)

        # Warm-start next root solve with current solution.
        if method == "ab":
            z_guess = z.copy()
        else:
            y_guess = np.array([info.q1, info.q2, z[4], z[5], z[6], z[7]], dtype=float)
        state = state_next

    return {
        "method": method,
        "t": np.arange(num_steps + 1, dtype=float) * float(h),
        "X": X,
        "R1": R1,
        "R2": R2,
        "F1": F1,
        "F2": F2,
        "thetaR": thetaR,
        "thetaF": thetaF,
        "lambda0": lam0,
        "lambda12": lam12,
        "u": u_sequence,
        "residual_inf": residual_inf,
        "solver_success": np.asarray([info.success for info in infos], dtype=bool),
        "accepted_by_residual": np.asarray(
            [info.accepted_by_residual for info in infos], dtype=bool
        ),
        "substepping_performed": np.asarray(
            [info.substepping_performed for info in infos], dtype=bool
        ),
        "substep_depth": np.asarray([info.substep_depth for info in infos], dtype=int),
        "num_substeps": np.asarray([info.num_substeps for info in infos], dtype=int),
        "h_original": np.asarray([info.h_original for info in infos], dtype=float),
        "h_min_used": np.asarray([info.h_min_used for info in infos], dtype=float),
        "cayley_near_singularity": np.asarray(
            [info.cayley_near_singularity for info in infos], dtype=bool
        ),
        "near_singularity_count": np.asarray(
            [info.near_singularity_count for info in infos], dtype=int
        ),
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
    method: str = "ab",
    allow_substepping: bool = False,
    min_substep_h: float = 1e-6,
    max_subdivisions: int = 10,
    singularity_margin_deg: float = 5.0,
    root_tol: float = 1e-10,
    lgvi_maxfev: int = 2000,
    normalized: bool = False,
    accept_residual: bool = True,
    accept_residual_tol: float = 1e-3,
) -> Tuple[AcrobotReducedState, Dict[str, Any]]:
    """
    Simulate one MPC control interval.

    The SDP provides one control input u_j for:
        [t_j, t_j + dt_control]

    The simulator applies this constant torque over many small dt_sim steps.
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
        method=method,
        allow_substepping=allow_substepping,
        min_substep_h=min_substep_h,
        max_subdivisions=max_subdivisions,
        singularity_margin_deg=singularity_margin_deg,
        root_tol=root_tol,
        lgvi_maxfev=lgvi_maxfev,
        normalized=normalized,
        accept_residual=accept_residual,
        accept_residual_tol=accept_residual_tol,
    )

    return sim["final_state"], sim


def simulate_one_control_interval_from_params(
    params: Mapping[str, Any],
    model: AcrobotSO2Model,
    state: AcrobotReducedState,
    u_j: float,
    method: str = "ab",
    allow_substepping: bool = False,
    min_substep_h: float = 1e-6,
    max_subdivisions: int = 10,
    singularity_margin_deg: float = 5.0,
    root_tol: float = 1e-10,
    lgvi_maxfev: Optional[int] = None,
    normalized: bool = False,
    accept_residual: Optional[bool] = None,
    accept_residual_tol: Optional[float] = None,
) -> Tuple[AcrobotReducedState, Dict[str, Any]]:
    """
    Convenience wrapper using YAML-derived params.
    """
    if "dt_sim" in params:
        dt_sim = float(params["dt_sim"])
    else:
        dt_sim = float(params["time"]["dt_sim"])

    if "control_interval" in params:
        dt_control = float(params["control_interval"])
    else:
        dt_control = float(params["time"].get("control_interval", params["time"]["dt_sdp"]))

    if lgvi_maxfev is None:
        lgvi_maxfev = int(params.get("lgvi_maxfev", 2000))
    if accept_residual is None:
        accept_residual = bool(params.get("accept_residual", True))
    if accept_residual_tol is None:
        accept_residual_tol = float(params.get("accept_residual_tol", 1e-3))

    return simulate_one_control_interval(
        model=model,
        state=state,
        u_j=u_j,
        dt_control=dt_control,
        dt_sim=dt_sim,
        method=method,
        allow_substepping=allow_substepping,
        min_substep_h=min_substep_h,
        max_subdivisions=max_subdivisions,
        singularity_margin_deg=singularity_margin_deg,
        root_tol=root_tol,
        lgvi_maxfev=lgvi_maxfev,
        normalized=normalized,
        accept_residual=accept_residual,
        accept_residual_tol=accept_residual_tol,
    )


def simulate_lgvi_acrobot(
    model: AcrobotSO2Model,
    h: float,
    steps: int,
    alpha0: np.ndarray,
    thetaF0: Optional[np.ndarray] = None,
    u_fun: Optional[Any] = None,
    method: str = "ab",
    allow_substepping: bool = False,
    first_step: str = "reduced",
    root_tol: float = 1e-10,
    maxfev: int = 100,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Backward-compatible rollout interface.

    Initializes the reduced state from:
        alpha0  = absolute R angles [thetaR1, thetaR2]
        thetaF0 = initial F step angles [thetaF1, thetaF2]

    If thetaF0 is omitted, rest start is used:
        F1_prev = I
        F2_prev = I

    first_step is kept only for compatibility and is ignored.
    """
    if steps < 1:
        raise ValueError("steps must be at least 1")

    if first_step.lower() not in {"reduced", "euler", "rk4"} and verbose:
        print(
            f"[simulate_lgvi_acrobot] first_step='{first_step}' ignored. "
            "Using reduced Option-B initialization."
        )

    if thetaF0 is None:
        thetaF0 = np.zeros(2, dtype=float)

    initial_state = make_reduced_state_from_absolute(
        model=model,
        h=h,
        thetaR=np.asarray(alpha0, dtype=float).reshape(2),
        thetaF=np.asarray(thetaF0, dtype=float).reshape(2),
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
        method=method,
        allow_substepping=allow_substepping,
        root_tol=root_tol,
        lgvi_maxfev=maxfev,
    )

    if verbose:
        print(
            "[simulate_lgvi_acrobot] max residual:",
            float(np.max(sim["residual_inf"])) if len(sim["residual_inf"]) else np.nan,
        )

    return sim


def get_absolute_angles_and_step_angles(
    state: AcrobotReducedState,
) -> Dict[str, float]:
    """
    Extract absolute R angles and previous F step angles from reduced state.
    """
    thetaR1 = float(angle_from_R(state.R1))
    thetaR2 = float(angle_from_R(state.R2))

    thetaF1_prev = float(angle_from_R(state.F1_prev))
    thetaF2_prev = float(angle_from_R(state.F2_prev))

    return {
        "thetaR1": thetaR1,
        "thetaR2": thetaR2,
        "thetaF1_prev": thetaF1_prev,
        "thetaF2_prev": thetaF2_prev,
    }


def convert_state_to_sdp_initial(
    state: AcrobotReducedState,
    dt_physical: float,
    dt_sdp: float,
    interval_start_state: Optional[AcrobotReducedState] = None,
) -> Dict[str, np.ndarray | float]:
    """
    Convert a fine simulation state into an SDP-compatible initial state.

    The rotations R1, R2 are unchanged.

    For MPC, the previous SDP step is the rotation accumulated over the entire
    control interval: R_start.T @ R_end.  The last fine-step F is retained only
    as a comparison diagnostic and is never rescaled.
    """
    dt_physical = float(dt_physical)
    dt_sdp = float(dt_sdp)

    values = get_absolute_angles_and_step_angles(state)

    thetaF1_last_fine = values["thetaF1_prev"]
    thetaF2_last_fine = values["thetaF2_prev"]
    if interval_start_state is None:
        raise ValueError("interval_start_state is required for MPC SDP conversion")

    F1_prev_sdp = interval_start_state.R1.T @ state.R1
    F2_prev_sdp = interval_start_state.R2.T @ state.R2
    thetaF1_sdp = float(angle_from_R(F1_prev_sdp))
    thetaF2_sdp = float(angle_from_R(F2_prev_sdp))

    return {
        "R1": state.R1.copy(),
        "R2": state.R2.copy(),
        "F1_prev": F1_prev_sdp,
        "F2_prev": F2_prev_sdp,
        "thetaR1": values["thetaR1"],
        "thetaR2": values["thetaR2"],
        "thetaF1_prev": thetaF1_sdp,
        "thetaF2_prev": thetaF2_sdp,
        "thetaF1_last_fine": thetaF1_last_fine,
        "thetaF2_last_fine": thetaF2_last_fine,
    }


def convert_state_to_sdp_initial_scalars(
    state: AcrobotReducedState,
    model: AcrobotSO2Model,
    dt_physical: float,
    dt_sdp: float,
    interval_start_state: Optional[AcrobotReducedState] = None,
) -> Dict[str, float]:
    """
    Convert reduced state to scalar values useful for fixing SDP initial data.

    Returns:
        c1_0, s1_0, c2_0, s2_0
        a1_prev, b1_prev, a2_prev, b2_prev

    The previous F values are computed over the full MPC interval from
    interval_start_state to state.
    """
    converted = convert_state_to_sdp_initial(
        state=state,
        dt_physical=dt_physical,
        dt_sdp=dt_sdp,
        interval_start_state=interval_start_state,
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
        # These names are kept for compatibility with solve.py.
        # They mean current physical state, later placed at SDP node 1.
        "c1_0": float(c1_0),
        "s1_0": float(s1_0),
        "c2_0": float(c2_0),
        "s2_0": float(s2_0),

        # Previous step F for the next SDP.
        "a1_prev": float(a1_prev),
        "b1_prev": float(b1_prev),
        "a2_prev": float(a2_prev),
        "b2_prev": float(b2_prev),

        # Diagnostics only.
        "thetaR1": float(converted["thetaR1"]),
        "thetaR2": float(converted["thetaR2"]),
        "thetaF1_prev": float(converted["thetaF1_prev"]),
        "thetaF2_prev": float(converted["thetaF2_prev"]),
        "thetaF1_last_fine": float(converted["thetaF1_last_fine"]),
        "thetaF2_last_fine": float(converted["thetaF2_last_fine"]),
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
        previous-step angles,
        approximate energy if model provides energy_from_reduced_state.
    """
    R1 = np.asarray(sim["R1"], dtype=float)
    R2 = np.asarray(sim["R2"], dtype=float)
    F1 = np.asarray(sim["F1"], dtype=float)
    F2 = np.asarray(sim["F2"], dtype=float)
    X = np.asarray(sim["X"], dtype=float)

    num_nodes = R1.shape[0]
    num_steps = max(0, num_nodes - 1)

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

    thetaF = np.zeros((num_steps, 2), dtype=float)
    energy = np.full(num_steps, np.nan, dtype=float)

    for k in range(num_steps):
        thetaF[k, 0] = float(angle_from_R(F1[k]))
        thetaF[k, 1] = float(angle_from_R(F2[k]))

        if hasattr(model, "energy_from_reduced_state"):
            energy[k] = model.energy_from_reduced_state(
                R1=R1[k],
                R2=R2[k],
                F1_prev=F1[k],
                F2_prev=F2[k],
                h=float(sim["t"][1] - sim["t"][0]) if len(sim.get("t", [])) > 1 else 1.0,
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
        "thetaF": thetaF,
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
    step = get_absolute_angles_and_step_angles(state)

    print(f"[{label}]")
    print(f"  thetaR1 = {theta[0]: .6f} rad, {np.rad2deg(theta[0]): .3f} deg")
    print(f"  thetaR2 = {theta[1]: .6f} rad, {np.rad2deg(theta[1]): .3f} deg")
    print(f"  thetaF1_prev = {step['thetaF1_prev']: .6f} rad")
    print(f"  thetaF2_prev = {step['thetaF2_prev']: .6f} rad")
    print(f"  X = {X}")
    print(f"  constraint norm = {model.constraint_norm(X, state.R1, state.R2):.3e}")
