"""
full_maximal_vi_acrobot.py

Planar Jan-Brüdigam-style maximal-coordinate variational-integrator benchmark
for the two-link Acrobot.

This file intentionally lives next to the reduced SDP-matching simulator.  It is
not used inside the SDP relaxation.  Its purpose is to provide a higher-fidelity
plant/benchmark with independent COM positions x_i and rotations R_i, while the
base and elbow joints are enforced by Lagrange multipliers.

The implementation is the planar SO(2) analogue of the maximal-coordinate VI
philosophy used in Brüdigam-style constrained dynamics:

    independent body poses + position-level constraints + implicit solve

State at a node k stores q_{k-1} and q_k.  Given q_{k-1}, q_k and an input u_k,
the solver computes q_{k+1}, lambda_{k+1}.  Step rotations are represented with
SO(2) Cayley coordinates qF_i so R_{i,k+1} = R_{i,k} Cay(qF_i).

Important thesis framing:
    * The SDP certificate applies to your reduced polynomial model.
    * This full maximal-coordinate simulator is a benchmark/plant model.
    * Disagreement between them is model mismatch, not automatically an SDP bug.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

try:
    from .lie_group_so2 import (
        F_from_cayley,
        angle_from_R,
        cayley_from_R,
        cross2,
        orth_error_so2,
        det_error_so2,
    )
    from .Discrete_Mechanical_Models_Lie_Groups.model_acrobot_so2 import AcrobotSO2Model
except ImportError:
    from lie_group_so2 import (
        F_from_cayley,
        angle_from_R,
        cayley_from_R,
        cross2,
        orth_error_so2,
        det_error_so2,
    )
    from Discrete_Mechanical_Models_Lie_Groups.model_acrobot_so2 import AcrobotSO2Model


@dataclass
class FullMaximalAcrobotState:
    """Two-node state for the full planar maximal-coordinate VI."""

    x1_prev: np.ndarray
    x2_prev: np.ndarray
    R1_prev: np.ndarray
    R2_prev: np.ndarray
    x1: np.ndarray
    x2: np.ndarray
    R1: np.ndarray
    R2: np.ndarray
    F1_prev: np.ndarray
    F2_prev: np.ndarray

    def __post_init__(self) -> None:
        self.x1_prev = np.asarray(self.x1_prev, dtype=float).reshape(2)
        self.x2_prev = np.asarray(self.x2_prev, dtype=float).reshape(2)
        self.R1_prev = np.asarray(self.R1_prev, dtype=float).reshape(2, 2)
        self.R2_prev = np.asarray(self.R2_prev, dtype=float).reshape(2, 2)
        self.x1 = np.asarray(self.x1, dtype=float).reshape(2)
        self.x2 = np.asarray(self.x2, dtype=float).reshape(2)
        self.R1 = np.asarray(self.R1, dtype=float).reshape(2, 2)
        self.R2 = np.asarray(self.R2, dtype=float).reshape(2, 2)
        self.F1_prev = np.asarray(self.F1_prev, dtype=float).reshape(2, 2)
        self.F2_prev = np.asarray(self.F2_prev, dtype=float).reshape(2, 2)


@dataclass
class FullMaximalStepInfo:
    success: bool
    residual_inf: float
    n_iter: int
    nfev: int
    message: str
    accepted_by_residual: bool = False
    thetaF1_deg: float = np.nan
    thetaF2_deg: float = np.nan
    lambda0_norm: float = np.nan
    lambda12_norm: float = np.nan
    lambda_total_norm: float = np.nan
    line_search_used: bool = False
    min_alpha: float = np.nan
    final_alpha: float = np.nan


class FullMaximalSolveError(RuntimeError):
    def __init__(self, residual_inf: float, message: str) -> None:
        self.residual_inf = float(residual_inf)
        self.solver_message = str(message)
        super().__init__(f"Full maximal VI solve failed: ||r||_inf={residual_inf:.3e}; {message}")


def _norm_inf(x: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(x, dtype=float).reshape(-1), ord=np.inf))


def _lambda_norms(lam0: np.ndarray, lam12: np.ndarray) -> Tuple[float, float, float]:
    n0 = float(np.linalg.norm(np.asarray(lam0, dtype=float).reshape(2)))
    n12 = float(np.linalg.norm(np.asarray(lam12, dtype=float).reshape(2)))
    return n0, n12, float(np.hypot(n0, n12))


def _finite_difference_jacobian(fun: Any, z: np.ndarray, f0: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float).reshape(-1)
    f0 = np.asarray(f0, dtype=float).reshape(-1)
    J = np.zeros((f0.size, z.size), dtype=float)
    for j in range(z.size):
        step = np.sqrt(np.finfo(float).eps) * max(1.0, abs(float(z[j])))
        zp = z.copy()
        zp[j] += step
        J[:, j] = (np.asarray(fun(zp), dtype=float).reshape(-1) - f0) / step
    return J


def make_full_state_from_angles(
    model: AcrobotSO2Model,
    h: float,
    thetaR: np.ndarray,
    thetaF: Optional[np.ndarray] = None,
) -> FullMaximalAcrobotState:
    """Create a full maximal state from absolute angles and previous step angles."""
    thetaR = np.asarray(thetaR, dtype=float).reshape(2)
    if thetaF is None:
        thetaF = np.zeros(2, dtype=float)
    thetaF = np.asarray(thetaF, dtype=float).reshape(2)

    R1, R2 = model.rotations_from_angles(thetaR[0], thetaR[1])
    X = model.reconstruct_positions_from_rotations(R1, R2)
    x1, x2 = X[0:2], X[2:4]

    # Previous node is consistent with the given previous step rotations.
    F1_prev = model.step_rotation_from_scalars(np.cos(thetaF[0]), np.sin(thetaF[0]))
    F2_prev = model.step_rotation_from_scalars(np.cos(thetaF[1]), np.sin(thetaF[1]))
    R1_prev = R1 @ F1_prev.T
    R2_prev = R2 @ F2_prev.T
    X_prev = model.reconstruct_positions_from_rotations(R1_prev, R2_prev)

    return FullMaximalAcrobotState(
        x1_prev=X_prev[0:2],
        x2_prev=X_prev[2:4],
        R1_prev=R1_prev,
        R2_prev=R2_prev,
        x1=x1,
        x2=x2,
        R1=R1,
        R2=R2,
        F1_prev=F1_prev,
        F2_prev=F2_prev,
    )


def unpack_full_z(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float, np.ndarray, np.ndarray]:
    z = np.asarray(z, dtype=float).reshape(10)
    x1_next = z[0:2]
    x2_next = z[2:4]
    qF1 = float(z[4])
    qF2 = float(z[5])
    lam0 = z[6:8]
    lam12 = z[8:10]
    return x1_next, x2_next, qF1, qF2, lam0, lam12


def full_maximal_initial_guess(
    model: AcrobotSO2Model,
    state: FullMaximalAcrobotState,
    h: float,
    previous_z: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Warm-start vector for the full maximal solve."""
    if previous_z is not None:
        return np.asarray(previous_z, dtype=float).reshape(10).copy()

    # Constant-velocity predictor for x, previous-F predictor for rotations.
    x1_guess = state.x1 + (state.x1 - state.x1_prev)
    x2_guess = state.x2 + (state.x2 - state.x2_prev)
    q1_guess = cayley_from_R(state.F1_prev)
    q2_guess = cayley_from_R(state.F2_prev)
    return np.r_[x1_guess, x2_guess, q1_guess, q2_guess, 0.0, 0.0, 0.0, 0.0].astype(float)


def full_maximal_residual(
    z: np.ndarray,
    model: AcrobotSO2Model,
    state: FullMaximalAcrobotState,
    u_k: float,
    h: float,
    torque_mode: str = "elbow",
) -> np.ndarray:
    """
    10-equation full maximal-coordinate residual.

    Unknown z = [x1_next, x2_next, qF1, qF2, lambda0, lambda12].
    Equations = translational DEL, rotational DEL, next-node constraints.
    """
    h = float(h)
    u_k = float(u_k)
    x1_next, x2_next, qF1, qF2, lam0, lam12 = unpack_full_z(z)
    F1 = F_from_cayley(qF1)
    F2 = F_from_cayley(qF2)
    R1_next = state.R1 @ F1
    R2_next = state.R2 @ F2

    g_vec = np.array([0.0, -float(model.g)], dtype=float)

    # Position-level constraint forces with convention
    # phi0=x1+R1*rho10-p0, phi12=x1+R1*rho112-x2-R2*rho212.
    r_x1 = (
        model.m1 * (x1_next - 2.0 * state.x1 + state.x1_prev)
        - h**2 * model.m1 * g_vec
        - h**2 * (lam0 + lam12)
    )
    r_x2 = (
        model.m2 * (x2_next - 2.0 * state.x2 + state.x2_prev)
        - h**2 * model.m2 * g_vec
        + h**2 * lam12
    )

    # Constraint moments evaluated at the next node, consistent with the
    # next-node position constraints.
    mu1 = cross2(model.rho10, R1_next.T @ lam0) + cross2(model.rho112, R1_next.T @ lam12)
    mu2 = -cross2(model.rho212, R2_next.T @ lam12)

    _, b1_prev = model.scalars_from_step_rotation(state.F1_prev)
    _, b2_prev = model.scalars_from_step_rotation(state.F2_prev)
    _, b1 = model.scalars_from_step_rotation(F1)
    _, b2 = model.scalars_from_step_rotation(F2)

    mode = str(torque_mode).lower()
    if mode == "elbow":
        tau1 = -u_k
        tau2 = +u_k
    elif mode == "base":
        tau1 = +u_k
        tau2 = 0.0
    else:
        raise ValueError("torque_mode must be 'elbow' or 'base'.")

    r_rot1 = model.rot_inertia_1 * (b1_prev - b1) + h**2 * (mu1 + tau1)
    r_rot2 = model.rot_inertia_2 * (b2_prev - b2) + h**2 * (mu2 + tau2)

    phi0 = x1_next + R1_next @ model.rho10 - model.p0
    phi12 = x1_next + R1_next @ model.rho112 - x2_next - R2_next @ model.rho212

    return np.r_[r_x1, r_x2, r_rot1, r_rot2, phi0, phi12].astype(float)


def _damped_newton_solve(
    fun: Any,
    z0: np.ndarray,
    tol: float,
    max_iter: int,
    line_search_c: float = 1e-4,
    min_alpha: float = 1e-8,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Finite-difference Newton with backtracking line search on ||r||_2."""
    z = np.asarray(z0, dtype=float).reshape(-1).copy()
    nfev = 0
    min_alpha_seen = 1.0
    final_alpha = 1.0

    for it in range(int(max_iter) + 1):
        r = np.asarray(fun(z), dtype=float).reshape(-1)
        nfev += 1
        r2 = float(np.linalg.norm(r))
        rinf = _norm_inf(r)
        if rinf <= float(tol):
            return z, {
                "success": True,
                "residual_inf": rinf,
                "n_iter": it,
                "nfev": nfev,
                "message": "damped Newton converged",
                "min_alpha": min_alpha_seen,
                "final_alpha": final_alpha,
            }

        J = _finite_difference_jacobian(fun, z, r)
        nfev += z.size
        try:
            dz = np.linalg.solve(J, -r)
        except np.linalg.LinAlgError:
            dz = np.linalg.lstsq(J, -r, rcond=None)[0]

        alpha = 1.0
        accepted = False
        while alpha >= float(min_alpha):
            z_trial = z + alpha * dz
            r_trial = np.asarray(fun(z_trial), dtype=float).reshape(-1)
            nfev += 1
            # Armijo-type decrease in residual norm.  If the full Newton step is
            # too aggressive, alpha is damped.  This is the line-search part.
            if float(np.linalg.norm(r_trial)) <= (1.0 - line_search_c * alpha) * r2:
                z = z_trial
                final_alpha = alpha
                min_alpha_seen = min(min_alpha_seen, alpha)
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            return z, {
                "success": False,
                "residual_inf": rinf,
                "n_iter": it,
                "nfev": nfev,
                "message": "line search failed to reduce residual",
                "min_alpha": min_alpha_seen,
                "final_alpha": alpha,
            }

    r = np.asarray(fun(z), dtype=float).reshape(-1)
    return z, {
        "success": False,
        "residual_inf": _norm_inf(r),
        "n_iter": int(max_iter),
        "nfev": nfev,
        "message": "maximum Newton iterations reached",
        "min_alpha": min_alpha_seen,
        "final_alpha": final_alpha,
    }


def step_full_maximal_vi(
    model: AcrobotSO2Model,
    state: FullMaximalAcrobotState,
    u_k: float,
    h: float,
    z_guess: Optional[np.ndarray] = None,
    tol: float = 1e-10,
    max_iter: int = 30,
    accept_residual: bool = True,
    accept_residual_tol: float = 1e-8,
    torque_mode: str = "elbow",
) -> Tuple[FullMaximalAcrobotState, FullMaximalStepInfo, np.ndarray]:
    """One Jan-style full maximal-coordinate VI step."""
    z0 = full_maximal_initial_guess(model, state, h=h, previous_z=z_guess)

    def fun(z: np.ndarray) -> np.ndarray:
        return full_maximal_residual(z, model=model, state=state, u_k=u_k, h=h, torque_mode=torque_mode)

    z, meta = _damped_newton_solve(fun, z0=z0, tol=tol, max_iter=max_iter)
    residual = fun(z)
    residual_inf = _norm_inf(residual)
    accepted = bool((not meta["success"]) and accept_residual and np.isfinite(residual_inf) and residual_inf <= accept_residual_tol)
    if not meta["success"] and not accepted:
        raise FullMaximalSolveError(residual_inf=residual_inf, message=str(meta["message"]))

    x1_next, x2_next, qF1, qF2, lam0, lam12 = unpack_full_z(z)
    F1 = F_from_cayley(qF1)
    F2 = F_from_cayley(qF2)
    R1_next = state.R1 @ F1
    R2_next = state.R2 @ F2
    n0, n12, ntot = _lambda_norms(lam0, lam12)

    next_state = FullMaximalAcrobotState(
        x1_prev=state.x1,
        x2_prev=state.x2,
        R1_prev=state.R1,
        R2_prev=state.R2,
        x1=x1_next,
        x2=x2_next,
        R1=R1_next,
        R2=R2_next,
        F1_prev=F1,
        F2_prev=F2,
    )
    info = FullMaximalStepInfo(
        success=bool(meta["success"] or accepted),
        residual_inf=float(residual_inf),
        n_iter=int(meta["n_iter"]),
        nfev=int(meta["nfev"]),
        message=str(meta["message"]),
        accepted_by_residual=accepted,
        thetaF1_deg=float(np.rad2deg(angle_from_R(F1))),
        thetaF2_deg=float(np.rad2deg(angle_from_R(F2))),
        lambda0_norm=n0,
        lambda12_norm=n12,
        lambda_total_norm=ntot,
        line_search_used=True,
        min_alpha=float(meta["min_alpha"]),
        final_alpha=float(meta["final_alpha"]),
    )
    return next_state, info, z


def rollout_full_maximal_vi_controls(
    model: AcrobotSO2Model,
    h: float,
    initial_state: FullMaximalAcrobotState,
    u_sequence: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 30,
    accept_residual: bool = True,
    accept_residual_tol: float = 1e-8,
    torque_mode: str = "elbow",
) -> Dict[str, Any]:
    """Roll out the full maximal-coordinate VI for a control sequence."""
    u_sequence = np.asarray(u_sequence, dtype=float).reshape(-1)
    num_steps = int(len(u_sequence))
    state = initial_state
    z_guess: Optional[np.ndarray] = None

    x1 = np.zeros((num_steps + 1, 2), dtype=float)
    x2 = np.zeros((num_steps + 1, 2), dtype=float)
    R1 = np.zeros((num_steps + 1, 2, 2), dtype=float)
    R2 = np.zeros((num_steps + 1, 2, 2), dtype=float)
    F1 = np.zeros((num_steps, 2, 2), dtype=float)
    F2 = np.zeros((num_steps, 2, 2), dtype=float)
    thetaR = np.zeros((num_steps + 1, 2), dtype=float)
    thetaF = np.zeros((num_steps, 2), dtype=float)
    residual_inf = np.zeros(num_steps, dtype=float)
    infos = []
    z_solutions = []

    x1[0] = state.x1
    x2[0] = state.x2
    R1[0] = state.R1
    R2[0] = state.R2
    thetaR[0] = model.angles_from_rotations(state.R1, state.R2)

    for k, u_k in enumerate(u_sequence):
        state_next, info, z = step_full_maximal_vi(
            model=model,
            state=state,
            u_k=float(u_k),
            h=float(h),
            z_guess=z_guess,
            tol=tol,
            max_iter=max_iter,
            accept_residual=accept_residual,
            accept_residual_tol=accept_residual_tol,
            torque_mode=torque_mode,
        )
        x1[k + 1] = state_next.x1
        x2[k + 1] = state_next.x2
        R1[k + 1] = state_next.R1
        R2[k + 1] = state_next.R2
        F1[k] = state_next.F1_prev
        F2[k] = state_next.F2_prev
        thetaR[k + 1] = model.angles_from_rotations(state_next.R1, state_next.R2)
        thetaF[k, 0] = angle_from_R(state_next.F1_prev)
        thetaF[k, 1] = angle_from_R(state_next.F2_prev)
        residual_inf[k] = info.residual_inf
        infos.append(info)
        z_solutions.append(z)
        z_guess = z.copy()
        state = state_next

    X = np.column_stack([x1, x2])
    return {
        "method": "full_maximal_vi",
        "t": np.arange(num_steps + 1, dtype=float) * float(h),
        "x1": x1,
        "x2": x2,
        "X": X,
        "R1": R1,
        "R2": R2,
        "F1": F1,
        "F2": F2,
        "thetaR": thetaR,
        "thetaF": thetaF,
        "u": u_sequence,
        "residual_inf": residual_inf,
        "infos": infos,
        "z_solutions": z_solutions,
        "final_state": state,
        "solver_success": np.asarray([info.success for info in infos], dtype=bool),
    }


def compare_reduced_and_full_rollouts(
    reduced_sim: Dict[str, Any],
    full_sim: Dict[str, Any],
) -> Dict[str, Any]:
    """Small helper to compare final absolute angles of reduced and full simulations."""
    try:
        from .lie_group_so2 import angle_diff_deg
    except ImportError:
        from lie_group_so2 import angle_diff_deg

    reduced_final = np.rad2deg(np.asarray(reduced_sim["thetaR"][-1], dtype=float).reshape(2))
    full_final = np.rad2deg(np.asarray(full_sim["thetaR"][-1], dtype=float).reshape(2))
    err = np.asarray(angle_diff_deg(full_final, reduced_final), dtype=float)
    return {
        "reduced_final_thetaR1_deg": float(reduced_final[0]),
        "reduced_final_thetaR2_deg": float(reduced_final[1]),
        "full_final_thetaR1_deg": float(full_final[0]),
        "full_final_thetaR2_deg": float(full_final[1]),
        "full_minus_reduced_thetaR1_deg": float(err[0]),
        "full_minus_reduced_thetaR2_deg": float(err[1]),
        "full_minus_reduced_norm_deg": float(np.linalg.norm(err)),
    }
