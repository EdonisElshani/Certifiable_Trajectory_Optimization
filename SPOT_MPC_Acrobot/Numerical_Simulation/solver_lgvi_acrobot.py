from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable, Dict, List, Tuple

import numpy as np
from scipy.optimize import root

from Numerical_Simulation.lie_group_so2 import (
    F_from_delta,
    angle_from_R,
    vee2,
    orth_error_so2,
    det_error_so2,
)

from Numerical_Simulation.Discrete_Mechanical_Models_Lie_Groups.model_acrobot_so2 import (
    AcrobotSO2Model,
)

from Numerical_Simulation.state import AcrobotLGVIState


@dataclass
class LGVIStepInfo:
    success: bool
    residual_inf: float
    nfev: int
    message: str


def _Pi(F: np.ndarray, Jd: np.ndarray, h: float) -> np.ndarray:
    """Pulled-back rotational momentum-like skew matrix."""
    return (Jd @ F - F.T @ Jd) / h


def acrobot_lgvi_step_residual(
    z: np.ndarray,
    model: AcrobotSO2Model,
    h: float,
    X_prev: np.ndarray,
    X_k: np.ndarray,
    R1_k: np.ndarray,
    R2_k: np.ndarray,
    F1_prev: np.ndarray,
    F2_prev: np.ndarray,
    u_k: float = 0.0,
    normalized: bool = True,
) -> np.ndarray:
    """Return the 10-dimensional residual for one LGVI step.

    If normalized=True, the translational equation is divided by h and the
    rotational equation is divided by h. This has the same roots and improves
    conditioning for small h.
    """
    z = np.asarray(z, dtype=float).reshape(10)
    X_next = z[0:4]
    dtheta1 = z[4]
    dtheta2 = z[5]
    lam0 = z[6:8]
    lam12 = z[8:10]

    F1 = F_from_delta(dtheta1)
    F2 = F_from_delta(dtheta2)
    R1_next = R1_k @ F1
    R2_next = R2_k @ F2

    V_prev = (X_k - X_prev) / h
    V_k = (X_next - X_k) / h

    # Translational DEL: M(V_k - V_{k-1}) + h*g*M*E - h*Gx^T lambda = 0
    rT = model.M @ (V_k - V_prev) + h * model.g * (model.M @ model.E) - h * model.Gx_T_lambda(lam0, lam12)
    if normalized:
        rT = rT / h

    # Rotational DEL: vee(Pi_{k-1} - Pi_k) + h*(gamma + tau) = 0
    Pi1_prev = _Pi(F1_prev, model.Jd1, h)
    Pi2_prev = _Pi(F2_prev, model.Jd2, h)
    Pi1 = _Pi(F1, model.Jd1, h)
    Pi2 = _Pi(F2, model.Jd2, h)

    gamma = model.constraint_torques(R1_k, R2_k, lam0, lam12)
    tau = model.generalized_torque_classical_acrobot(u_k)
    rR = np.array([
        vee2(Pi1_prev - Pi1) + h * (gamma[0] + tau[0]),
        vee2(Pi2_prev - Pi2) + h * (gamma[1] + tau[1]),
    ])
    if normalized:
        rR = rR / h

    # Enforce next-node constraints for forward simulation.
    rC = model.constraints(X_next, R1_next, R2_next)

    return np.r_[rT, rR, rC]


def initial_guess_from_previous(
    X_prev: np.ndarray,
    X_k: np.ndarray,
    F1_prev: np.ndarray,
    F2_prev: np.ndarray,
) -> np.ndarray:
    """Simple predictor for z_k."""
    X_guess = X_k + (X_k - X_prev)
    dtheta1_guess = angle_from_R(F1_prev)
    dtheta2_guess = angle_from_R(F2_prev)
    lam_guess = np.zeros(4)
    return np.r_[X_guess, dtheta1_guess, dtheta2_guess, lam_guess]


def simulate_lgvi_acrobot(
    model: AcrobotSO2Model,
    h: float,
    steps: int,
    alpha0: np.ndarray,
    omega0: np.ndarray,
    u_fun: Optional[Callable[[float], float]] = None,
    first_step: str = "rk4",
    root_tol: float = 1e-10,
    maxfev: int = 100,
    verbose: bool = False,
) -> Dict[str, np.ndarray | List[LGVIStepInfo]]:
    """Simulate the Acrobot with the maximal-coordinate LGVI.

    Parameters
    ----------
    steps:
        Number of time intervals. The output has steps+1 nodes.
    first_step:
        'rk4' uses a small RK4 step from the continuous minimal model to
        initialize q_1. 'euler' uses R_1=R_0 exp(h omega_0 S).
    """
    alpha0 = np.asarray(alpha0, dtype=float).reshape(2)
    omega0 = np.asarray(omega0, dtype=float).reshape(2)
    if steps < 1:
        raise ValueError("steps must be at least 1")

    X = np.zeros((steps + 1, 4))
    R1 = np.zeros((steps + 1, 2, 2))
    R2 = np.zeros((steps + 1, 2, 2))
    F1 = np.zeros((steps, 2, 2))
    F2 = np.zeros((steps, 2, 2))
    dtheta = np.zeros((steps, 2))
    lambdas = np.zeros((steps, 4))
    residual_inf = np.full(steps, np.nan)
    infos: List[LGVIStepInfo] = []

    X[0], R1[0], R2[0], _ = model.positions_from_angles(alpha0[0], alpha0[1])

    # Initialize the second node q_1.
    if first_step.lower() == "rk4":
        # Local RK4 to generate a consistent high-quality first step.
        y0 = np.array([alpha0[0], alpha0[1], omega0[0], omega0[1]], dtype=float)
        u = u_fun if u_fun is not None else None
        k1 = model.minimal_rhs(0.0, y0, u)
        k2 = model.minimal_rhs(0.5 * h, y0 + 0.5 * h * k1, u)
        k3 = model.minimal_rhs(0.5 * h, y0 + 0.5 * h * k2, u)
        k4 = model.minimal_rhs(h, y0 + h * k3, u)
        y1 = y0 + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        alpha1 = y1[:2]
    elif first_step.lower() == "euler":
        alpha1 = alpha0 + h * omega0
    else:
        raise ValueError("first_step must be 'rk4' or 'euler'")

    X[1], R1[1], R2[1], _ = model.positions_from_angles(alpha1[0], alpha1[1])
    F1[0] = R1[0].T @ R1[1]
    F2[0] = R2[0].T @ R2[1]
    dtheta[0] = np.array([angle_from_R(F1[0]), angle_from_R(F2[0])])

    # If there is only one step, return the initialized trajectory.
    if steps == 1:
        return {
            "t": np.arange(steps + 1) * h,
            "X": X,
            "R1": R1,
            "R2": R2,
            "F1": F1,
            "F2": F2,
            "dtheta": dtheta,
            "lambda": lambdas,
            "residual_inf": residual_inf,
            "infos": infos,
        }

    # March from k=1 to steps-1.
    z_guess = initial_guess_from_previous(X[0], X[1], F1[0], F2[0])
    for k in range(1, steps):
        u_k = 0.0 if u_fun is None else float(u_fun(k * h))

        fun = lambda z: acrobot_lgvi_step_residual(
            z, model, h,
            X[k-1], X[k], R1[k], R2[k], F1[k-1], F2[k-1],
            u_k=u_k,
            normalized=True,
        )

        sol = root(fun, z_guess, method="hybr", options={"xtol": root_tol, "maxfev": maxfev})
        r = fun(sol.x)
        rinf = float(np.linalg.norm(r, ord=np.inf))
        residual_inf[k] = rinf
        info = LGVIStepInfo(bool(sol.success), rinf, int(sol.nfev), str(sol.message))
        infos.append(info)

        if verbose and (not sol.success or rinf > 1e-7):
            print(f"[LGVI] k={k}, success={sol.success}, ||r||_inf={rinf:.3e}, msg={sol.message}")

        if not sol.success and rinf > 1e-7:
            raise RuntimeError(f"LGVI root solve failed at k={k}: ||r||_inf={rinf:.3e}, message={sol.message}")

        z = sol.x
        X[k+1] = z[0:4]
        dtheta[k] = z[4:6]
        lambdas[k] = z[6:10]
        F1[k] = F_from_delta(dtheta[k, 0])
        F2[k] = F_from_delta(dtheta[k, 1])
        R1[k+1] = R1[k] @ F1[k]
        R2[k+1] = R2[k] @ F2[k]

        # Predictor for next step: reuse solved delta and lambda, linear X extrapolation.
        if k < steps - 1:
            z_guess = np.r_[X[k+1] + (X[k+1] - X[k]), dtheta[k, 0], dtheta[k, 1], lambdas[k]]

    return {
        "t": np.arange(steps + 1) * h,
        "X": X,
        "R1": R1,
        "R2": R2,
        "F1": F1,
        "F2": F2,
        "dtheta": dtheta,
        "lambda": lambdas,
        "residual_inf": residual_inf,
        "infos": infos,
    }


def diagnostics_lgvi(model: AcrobotSO2Model, sim: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Compute constraint, SO(2), and energy diagnostics."""
    X = sim["X"]
    R1 = sim["R1"]
    R2 = sim["R2"]
    F1 = sim["F1"]
    F2 = sim["F2"]
    t = sim["t"]
    h = float(t[1] - t[0]) if len(t) > 1 else 1.0
    n_nodes = X.shape[0]
    n_steps = n_nodes - 1

    phi_norm = np.zeros(n_nodes)
    phi0_norm = np.zeros(n_nodes)
    phi12_norm = np.zeros(n_nodes)
    orth_R1 = np.zeros(n_nodes)
    orth_R2 = np.zeros(n_nodes)
    det_R1 = np.zeros(n_nodes)
    det_R2 = np.zeros(n_nodes)
    alpha = np.zeros((n_nodes, 2))

    for k in range(n_nodes):
        phi = model.constraints(X[k], R1[k], R2[k])
        phi0_norm[k] = np.linalg.norm(phi[:2])
        phi12_norm[k] = np.linalg.norm(phi[2:])
        phi_norm[k] = np.linalg.norm(phi)
        orth_R1[k] = orth_error_so2(R1[k])
        orth_R2[k] = orth_error_so2(R2[k])
        det_R1[k] = det_error_so2(R1[k])
        det_R2[k] = det_error_so2(R2[k])
        alpha[k] = model.angles_from_rotations(R1[k], R2[k])

    energy = np.full(n_steps, np.nan)
    omega = np.zeros((n_steps, 2))
    for k in range(n_steps):
        V = (X[k+1] - X[k]) / h
        omega[k, 0] = angle_from_R(F1[k]) / h
        omega[k, 1] = angle_from_R(F2[k]) / h
        energy[k] = model.energy_from_maximal(X[k], V, omega[k, 0], omega[k, 1])

    return {
        "phi_norm": phi_norm,
        "phi0_norm": phi0_norm,
        "phi12_norm": phi12_norm,
        "orth_R1": orth_R1,
        "orth_R2": orth_R2,
        "det_R1": det_R1,
        "det_R2": det_R2,
        "alpha": alpha,
        "omega": omega,
        "energy": energy,
        "energy_error": energy - energy[0] if n_steps > 0 else energy,
    }

def reconstruct_X_from_R(
    model: AcrobotSO2Model,
    R1: np.ndarray,
    R2: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct maximal COM coordinates X = [x1; x2] from R1, R2
    using the holonomic constraints.

    phi0  = x1 + R1 rho10 - p0 = 0
    phi12 = x1 + R1 rho112 - x2 - R2 rho212 = 0
    """
    x1 = model.p0 - R1 @ model.rho10
    x2 = x1 + R1 @ model.rho112 - R2 @ model.rho212
    return np.r_[x1, x2]


def make_lgvi_state_from_relative(
    model: AcrobotSO2Model,
    h: float,
    q: np.ndarray,
    qdot: np.ndarray,
) -> AcrobotLGVIState:
    """
    Create an LGVI state from standard Acrobot coordinates.

    q     = [theta1, theta2]
    qdot  = [theta1dot, theta2dot]

    The model uses absolute SO(2) attitudes:
        alpha1 = theta1 - pi/2
        alpha2 = theta1 + theta2 - pi/2

    The previous F is approximated by:
        F_{k-1} = exp(h * omega_k)
    so that:
        R_{k-1} = R_k F_{k-1}^T
    """
    q = np.asarray(q, dtype=float).reshape(2)
    qdot = np.asarray(qdot, dtype=float).reshape(2)

    alpha = model.absolute_angles_from_relative(q[0], q[1])
    omega_abs = np.array(
        [
            qdot[0],
            qdot[0] + qdot[1],
        ],
        dtype=float,
    )

    X, R1, R2, _ = model.positions_from_angles(alpha[0], alpha[1])

    F1_prev = F_from_delta(h * omega_abs[0])
    F2_prev = F_from_delta(h * omega_abs[1])

    R1_prev = R1 @ F1_prev.T
    R2_prev = R2 @ F2_prev.T
    X_prev = reconstruct_X_from_R(model, R1_prev, R2_prev)

    return AcrobotLGVIState(
        X_prev=X_prev,
        X=X,
        R1=R1,
        R2=R2,
        F1_prev=F1_prev,
        F2_prev=F2_prev,
    )

def lgvi_one_step(
    model: AcrobotSO2Model,
    h: float,
    state: AcrobotLGVIState,
    u_k: float,
    z_guess: Optional[np.ndarray] = None,
    root_tol: float = 1e-10,
    maxfev: int = 100,
    normalized: bool = True,
) -> tuple[AcrobotLGVIState, LGVIStepInfo, np.ndarray]:
    """
    Propagate the acrobot by one forced LGVI step.

    Input:
        state at node k:
            X_{k-1}, X_k, R_{1,k}, R_{2,k}, F_{1,k-1}, F_{2,k-1}
        scalar torque:
            u_k

    Output:
        next_state at node k+1:
            X_k, X_{k+1}, R_{1,k+1}, R_{2,k+1}, F_{1,k}, F_{2,k}
    """
    if z_guess is None:
        z_guess = initial_guess_from_previous(
            state.X_prev,
            state.X,
            state.F1_prev,
            state.F2_prev,
        )

    def fun(z: np.ndarray) -> np.ndarray:
        return acrobot_lgvi_step_residual(
            z=z,
            model=model,
            h=h,
            X_prev=state.X_prev,
            X_k=state.X,
            R1_k=state.R1,
            R2_k=state.R2,
            F1_prev=state.F1_prev,
            F2_prev=state.F2_prev,
            u_k=float(u_k),
            normalized=normalized,
        )

    sol = root(
        fun,
        z_guess,
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

    if not sol.success and residual_inf > 1e-7:
        raise RuntimeError(
            f"LGVI one-step solve failed: "
            f"success={sol.success}, "
            f"||r||_inf={residual_inf:.3e}, "
            f"message={sol.message}"
        )

    z = sol.x
    X_next = z[0:4]
    dtheta1 = float(z[4])
    dtheta2 = float(z[5])

    F1 = F_from_delta(dtheta1)
    F2 = F_from_delta(dtheta2)

    R1_next = state.R1 @ F1
    R2_next = state.R2 @ F2

    next_state = AcrobotLGVIState(
        X_prev=state.X,
        X=X_next,
        R1=R1_next,
        R2=R2_next,
        F1_prev=F1,
        F2_prev=F2,
    )

    return next_state, info, z

def rollout_lgvi_controls(
    model: AcrobotSO2Model,
    h: float,
    initial_state: AcrobotLGVIState,
    u_sequence: np.ndarray,
    root_tol: float = 1e-10,
    maxfev: int = 100,
) -> dict:
    """
    Open-loop rollout for a given torque sequence.

    This is useful for:
        - testing one SDP control sequence,
        - testing one MPC step,
        - validating extracted controls physically.
    """
    u_sequence = np.asarray(u_sequence, dtype=float).reshape(-1)
    num_steps = len(u_sequence)

    X = np.zeros((num_steps + 1, 4))
    R1 = np.zeros((num_steps + 1, 2, 2))
    R2 = np.zeros((num_steps + 1, 2, 2))
    F1 = np.zeros((num_steps, 2, 2))
    F2 = np.zeros((num_steps, 2, 2))
    dtheta = np.zeros((num_steps, 2))
    residual_inf = np.zeros(num_steps)
    infos = []
    z_solutions = []

    state = initial_state

    X[0] = state.X
    R1[0] = state.R1
    R2[0] = state.R2

    z_guess = None

    for k, u_k in enumerate(u_sequence):
        state_next, info, z = lgvi_one_step(
            model=model,
            h=h,
            state=state,
            u_k=float(u_k),
            z_guess=z_guess,
            root_tol=root_tol,
            maxfev=maxfev,
        )

        X[k + 1] = state_next.X
        R1[k + 1] = state_next.R1
        R2[k + 1] = state_next.R2
        F1[k] = state_next.F1_prev
        F2[k] = state_next.F2_prev
        dtheta[k, 0] = angle_from_R(F1[k])
        dtheta[k, 1] = angle_from_R(F2[k])
        residual_inf[k] = info.residual_inf
        infos.append(info)
        z_solutions.append(z)

        # Good predictor for next step, because root solvers are needy little creatures.
        z_guess = np.r_[
            state_next.X + (state_next.X - state.X),
            dtheta[k, 0],
            dtheta[k, 1],
            z[6:10],
        ]

        state = state_next

    return {
        "t": np.arange(num_steps + 1) * h,
        "X": X,
        "R1": R1,
        "R2": R2,
        "F1": F1,
        "F2": F2,
        "dtheta": dtheta,
        "u": u_sequence,
        "residual_inf": residual_inf,
        "infos": infos,
        "z_solutions": z_solutions,
        "final_state": state,
    }