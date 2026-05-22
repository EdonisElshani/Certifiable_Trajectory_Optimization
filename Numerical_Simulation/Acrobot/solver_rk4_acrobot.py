"""RK4 reference simulators for the planar Acrobot.

Two variants are provided:

1. simulate_rk4_acrobot_relative
   Integrates the standard continuous Acrobot equations in minimal relative
   coordinates y=[theta1, theta2, theta1dot, theta2dot].  R1,R2 are reconstructed
   from angles after each step, so the reported rotations are exactly on SO(2).

2. simulate_rk4_acrobot_matrix
   Integrates the rotation matrices directly via Rdot=R*S*omega, analogous to
   the 3D pendulum RK4 implementation.  Without projection, R may drift from
   SO(2).  This is useful for comparing manifold preservation.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

try:
    from Acrobot.lie_group_so2 import S2, R_so2, angle_from_R, orth_error_so2, project_to_so2
    from Acrobot.Discrete_Mechanical_Models_Lie_Groups.model_acrobot_so2 import AcrobotSO2Model
except ImportError:
    from lie_group_so2 import S2, R_so2, angle_from_R, orth_error_so2, project_to_so2
    from Discrete_Mechanical_Models_Lie_Groups.model_acrobot_so2 import AcrobotSO2Model


def rk4_step_vector(rhs, t: float, y: np.ndarray, h: float, u_fun=None) -> np.ndarray:
    """Plain explicit RK4 step for a vector state."""
    k1 = rhs(t, y, u_fun)
    k2 = rhs(t + 0.5 * h, y + 0.5 * h * k1, u_fun)
    k3 = rhs(t + 0.5 * h, y + 0.5 * h * k2, u_fun)
    k4 = rhs(t + h, y + h * k3, u_fun)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate_rk4_acrobot_relative(
    model: AcrobotSO2Model,
    h: float,
    steps: int,
    q0: np.ndarray,
    qdot0: np.ndarray,
    u_fun: Optional[Callable[[float], float]] = None,
) -> Dict[str, np.ndarray]:
    """Run RK4 on the standard continuous Acrobot equations.

    Parameters
    ----------
    q0:
        [theta1, theta2], where theta2 is the relative elbow angle and zero
        means both links point directly downward.
    qdot0:
        [theta1dot, theta2dot].

    Notes
    -----
    R1 and R2 are reconstructed from q after RK4. Therefore their orthogonality
    error is machine precision. This is not RK4 preserving SO(2); it is the
    angle parametrization doing the babysitting. Tiny but important.
    """
    q0 = np.asarray(q0, dtype=float).reshape(2)
    qdot0 = np.asarray(qdot0, dtype=float).reshape(2)
    y = np.zeros((steps + 1, 4))
    y[0] = np.r_[q0, qdot0]

    for k in range(steps):
        y[k + 1] = rk4_step_vector(model.rhs_relative, k * h, y[k], h, u_fun)

    X = np.zeros((steps + 1, 4))
    R1 = np.zeros((steps + 1, 2, 2))
    R2 = np.zeros((steps + 1, 2, 2))
    alpha_abs = np.zeros((steps + 1, 2))
    energy = np.zeros(steps + 1)
    orth_R1 = np.zeros(steps + 1)
    orth_R2 = np.zeros(steps + 1)
    phi0_norm = np.zeros(steps + 1)
    phi12_norm = np.zeros(steps + 1)
    phi_norm = np.zeros(steps + 1)

    for k in range(steps + 1):
        theta1, theta2 = y[k, 0], y[k, 1]
        X[k], R1[k], R2[k], _ = model.positions_from_relative(theta1, theta2)
        alpha_abs[k] = model.absolute_angles_from_relative(theta1, theta2)
        energy[k] = model.energy_from_relative(y[k])
        orth_R1[k] = orth_error_so2(R1[k])
        orth_R2[k] = orth_error_so2(R2[k])
        phi = model.constraints(X[k], R1[k], R2[k])
        phi0_norm[k] = np.linalg.norm(phi[:2])
        phi12_norm[k] = np.linalg.norm(phi[2:])
        phi_norm[k] = np.linalg.norm(phi)

    return {
        "t": np.arange(steps + 1) * h,
        "y": y,
        "q": y[:, 0:2],
        "qdot": y[:, 2:4],
        "alpha_abs": alpha_abs,
        "X": X,
        "R1": R1,
        "R2": R2,
        "energy": energy,
        "energy_error": energy - energy[0],
        "orth_R1": orth_R1,
        "orth_R2": orth_R2,
        "phi0_norm": phi0_norm,
        "phi12_norm": phi12_norm,
        "phi_norm": phi_norm,
    }


def _matrix_rhs(model: AcrobotSO2Model, t: float, y: np.ndarray, u_fun=None) -> np.ndarray:
    """RHS for matrix-RK4 state y=[R1(:), R2(:), qdot].

    q is recovered from R1,R2 using atan2. This mirrors the 3D-pendulum style
    where the matrix itself is part of the RK4 state.
    """
    R1 = y[0:4].reshape(2, 2)
    R2 = y[4:8].reshape(2, 2)
    qdot = y[8:10]
    theta1, theta2 = model.relative_angles_from_rotations(R1, R2, wrap=False)
    u = 0.0 if u_fun is None else float(u_fun(t))
    qddot = model.qddot_relative(theta1, theta2, qdot, u)

    omega1_abs = qdot[0]
    omega2_abs = qdot[0] + qdot[1]
    S = S2()
    R1dot = R1 @ (omega1_abs * S)
    R2dot = R2 @ (omega2_abs * S)
    return np.r_[R1dot.reshape(4), R2dot.reshape(4), qddot]


def rk4_step_matrix(model: AcrobotSO2Model, t: float, y: np.ndarray, h: float, u_fun=None) -> np.ndarray:
    """Explicit RK4 step for matrix state, same spirit as the 3D pendulum code."""
    k1 = _matrix_rhs(model, t, y, u_fun)
    k2 = _matrix_rhs(model, t + 0.5 * h, y + 0.5 * h * k1, u_fun)
    k3 = _matrix_rhs(model, t + 0.5 * h, y + 0.5 * h * k2, u_fun)
    k4 = _matrix_rhs(model, t + h, y + h * k3, u_fun)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate_rk4_acrobot_matrix(
    model: AcrobotSO2Model,
    h: float,
    steps: int,
    q0: np.ndarray,
    qdot0: np.ndarray,
    u_fun: Optional[Callable[[float], float]] = None,
    project: bool = False,
) -> Dict[str, np.ndarray]:
    """Run RK4 with R1,R2 integrated as matrices.

    If project=False, the rotations may drift from SO(2), exactly like the
    unprojected SO(3) RK4 pendulum.  If project=True, R1 and R2 are projected
    back to SO(2) after each step.
    """
    q0 = np.asarray(q0, dtype=float).reshape(2)
    qdot0 = np.asarray(qdot0, dtype=float).reshape(2)
    _, R1_0, R2_0, _ = model.positions_from_relative(q0[0], q0[1])
    y = np.zeros((steps + 1, 10))
    y[0] = np.r_[R1_0.reshape(4), R2_0.reshape(4), qdot0]

    for k in range(steps):
        y_next = rk4_step_matrix(model, k * h, y[k], h, u_fun)
        if project:
            R1 = project_to_so2(y_next[0:4].reshape(2, 2))
            R2 = project_to_so2(y_next[4:8].reshape(2, 2))
            y_next[0:4] = R1.reshape(4)
            y_next[4:8] = R2.reshape(4)
        y[k + 1] = y_next

    X = np.zeros((steps + 1, 4))
    R1_hist = np.zeros((steps + 1, 2, 2))
    R2_hist = np.zeros((steps + 1, 2, 2))
    q = np.zeros((steps + 1, 2))
    qdot = np.zeros((steps + 1, 2))
    alpha_abs = np.zeros((steps + 1, 2))
    energy = np.zeros(steps + 1)
    orth_R1 = np.zeros(steps + 1)
    orth_R2 = np.zeros(steps + 1)

    for k in range(steps + 1):
        R1 = y[k, 0:4].reshape(2, 2)
        R2 = y[k, 4:8].reshape(2, 2)
        R1_hist[k] = R1
        R2_hist[k] = R2
        q[k] = model.relative_angles_from_rotations(R1, R2, wrap=False)
        qdot[k] = y[k, 8:10]
        alpha_abs[k] = np.array([angle_from_R(R1), angle_from_R(R2)])
        # For diagnostics, compute COM positions from the possibly non-orthogonal matrices.
        x1 = model.p0 - R1 @ model.rho10
        x2 = x1 + R1 @ model.rho112 - R2 @ model.rho212
        X[k] = np.r_[x1, x2]
        energy[k] = model.energy_from_relative(np.r_[q[k], qdot[k]])
        orth_R1[k] = orth_error_so2(R1)
        orth_R2[k] = orth_error_so2(R2)

    return {
        "t": np.arange(steps + 1) * h,
        "y": y,
        "q": q,
        "qdot": qdot,
        "alpha_abs": alpha_abs,
        "X": X,
        "R1": R1_hist,
        "R2": R2_hist,
        "energy": energy,
        "energy_error": energy - energy[0],
        "orth_R1": orth_R1,
        "orth_R2": orth_R2,
    }


# Backward-compatible name. By default, use the matrix version because it matches
# the user's 3D-pendulum RK4 style more closely.
def simulate_rk4_acrobot(
    model: AcrobotSO2Model,
    h: float,
    steps: int,
    q0: np.ndarray,
    qdot0: np.ndarray,
    u_fun: Optional[Callable[[float], float]] = None,
    project: bool = False,
    mode: str = "matrix",
) -> Dict[str, np.ndarray]:
    if mode == "relative":
        return simulate_rk4_acrobot_relative(model, h, steps, q0, qdot0, u_fun=u_fun)
    if mode == "matrix":
        return simulate_rk4_acrobot_matrix(model, h, steps, q0, qdot0, u_fun=u_fun, project=project)
    raise ValueError("mode must be 'matrix' or 'relative'")

# ---------------------------------------------------------------------------
# Maximal-coordinate RK4 with acceleration-level constraints only
# ---------------------------------------------------------------------------

def _constraint_diagnostics(model: AcrobotSO2Model, X: np.ndarray, R1_hist: np.ndarray, R2_hist: np.ndarray):
    """Compute phi0, phi12, and total constraint norms."""
    n = X.shape[0]
    phi0_norm = np.zeros(n)
    phi12_norm = np.zeros(n)
    phi_norm = np.zeros(n)

    for k in range(n):
        phi = model.constraints(X[k], R1_hist[k], R2_hist[k])
        phi0_norm[k] = np.linalg.norm(phi[:2])
        phi12_norm[k] = np.linalg.norm(phi[2:])
        phi_norm[k] = np.linalg.norm(phi)

    return phi0_norm, phi12_norm, phi_norm


def _initial_maximal_state_from_relative(
    model: AcrobotSO2Model,
    q0: np.ndarray,
    qdot0: np.ndarray,
) -> np.ndarray:
    """Build consistent maximal-coordinate initial state from relative Acrobot ICs.

    State:
        y = [X(4), alpha(2), V(4), omega(2)]
    """
    q0 = np.asarray(q0, dtype=float).reshape(2)
    qdot0 = np.asarray(qdot0, dtype=float).reshape(2)

    theta1, theta2 = q0
    theta1dot, theta2dot = qdot0

    alpha = model.absolute_angles_from_relative(theta1, theta2)
    omega = np.array([theta1dot, theta1dot + theta2dot], dtype=float)

    X, R1, R2, _ = model.positions_from_relative(theta1, theta2)

    S = S2()

    # x1 = p0 - R1 rho10
    x1dot = -R1 @ (S @ model.rho10) * omega[0]

    # x2 = x1 + R1 rho112 - R2 rho212
    x2dot = (
        x1dot
        + R1 @ (S @ model.rho112) * omega[0]
        - R2 @ (S @ model.rho212) * omega[1]
    )

    V = np.r_[x1dot, x2dot]

    return np.r_[X, alpha, V, omega]


def _maximal_rhs_accel_level(
    model: AcrobotSO2Model,
    t: float,
    y: np.ndarray,
    u_fun=None,
) -> np.ndarray:
    """Continuous maximal-coordinate dynamics with acceleration-level constraints only.

    State:
        y = [X(4), alpha(2), V(4), omega(2)]

    This is the method that can show 'joints falling apart':
    constraints are enforced at acceleration level, but NOT projected back
    to position level.
    """
    y = np.asarray(y, dtype=float).reshape(12)

    X = y[0:4]
    alpha = y[4:6]
    V = y[6:10]
    omega = y[10:12]

    R1 = R_so2(alpha[0])
    R2 = R_so2(alpha[1])

    u = 0.0 if u_fun is None else float(u_fun(t))
    tau = model.generalized_torque_classical_acrobot(u)  # [-u, +u]

    S = S2()
    S2mat = S @ S  # = -I for SO(2)

    def dyn_from_lambda(lam_vec: np.ndarray):
        lam_vec = np.asarray(lam_vec, dtype=float).reshape(4)
        lam0 = lam_vec[:2]
        lam12 = lam_vec[2:]

        # Translational dynamics:
        # M Xddot + g M E - Gx^T lambda = 0
        Xddot = np.linalg.solve(
            model.M,
            model.Gx_T_lambda(lam0, lam12) - model.g * (model.M @ model.E),
        )

        # Rotational dynamics:
        # J_i * omegadot_i = gamma_i + tau_i
        gamma = model.constraint_torques(R1, R2, lam0, lam12)
        omegadot = np.array([
            (gamma[0] + tau[0]) / model.J1,
            (gamma[1] + tau[1]) / model.J2,
        ], dtype=float)

        return Xddot, omegadot

    def phi_ddot_from_lambda(lam_vec: np.ndarray):
        Xddot, omegadot = dyn_from_lambda(lam_vec)
        x1dd = Xddot[:2]
        x2dd = Xddot[2:]

        # phi0 = x1 + R1 rho10 - p0
        phi0_dd = (
            x1dd
            + R1 @ (S @ model.rho10) * omegadot[0]
            + R1 @ (S2mat @ model.rho10) * (omega[0] ** 2)
        )

        # phi12 = x1 + R1 rho112 - x2 - R2 rho212
        phi12_dd = (
            x1dd
            + R1 @ (S @ model.rho112) * omegadot[0]
            + R1 @ (S2mat @ model.rho112) * (omega[0] ** 2)
            - x2dd
            - R2 @ (S @ model.rho212) * omegadot[1]
            - R2 @ (S2mat @ model.rho212) * (omega[1] ** 2)
        )

        return np.r_[phi0_dd, phi12_dd]

    # Because phi_ddot is affine in lambda, identify A*lambda + c = 0
    c = phi_ddot_from_lambda(np.zeros(4))
    A = np.column_stack([
        phi_ddot_from_lambda(np.eye(4)[j]) - c for j in range(4)
    ])

    lam = np.linalg.solve(A, -c)

    Xddot, omegadot = dyn_from_lambda(lam)

    alpha_dot = omega
    X_dot = V

    return np.r_[X_dot, alpha_dot, Xddot, omegadot]


def simulate_rk4_acrobot_maximal_accel(
    model: AcrobotSO2Model,
    h: float,
    steps: int,
    q0: np.ndarray,
    qdot0: np.ndarray,
    u_fun: Optional[Callable[[float], float]] = None,
) -> Dict[str, np.ndarray]:
    """RK4 in maximal coordinates with acceleration-level constraints only."""
    y = np.zeros((steps + 1, 12))
    y[0] = _initial_maximal_state_from_relative(model, q0, qdot0)

    rhs = lambda t, yy, uu: _maximal_rhs_accel_level(model, t, yy, uu)

    for k in range(steps):
        y[k + 1] = rk4_step_vector(rhs, k * h, y[k], h, u_fun)

    X = y[:, 0:4]
    alpha = y[:, 4:6]
    V = y[:, 6:10]
    omega = y[:, 10:12]

    R1_hist = np.zeros((steps + 1, 2, 2))
    R2_hist = np.zeros((steps + 1, 2, 2))
    q = np.zeros((steps + 1, 2))
    qdot = np.zeros((steps + 1, 2))
    energy = np.zeros(steps + 1)

    for k in range(steps + 1):
        R1_hist[k] = R_so2(alpha[k, 0])
        R2_hist[k] = R_so2(alpha[k, 1])

        q[k] = model.relative_angles_from_absolute(alpha[k, 0], alpha[k, 1], wrap=False)
        qdot[k] = np.array([
            omega[k, 0],
            omega[k, 1] - omega[k, 0],
        ])

        energy[k] = model.energy_from_maximal(
            X[k],
            V[k],
            omega[k, 0],
            omega[k, 1],
        )

    phi0_norm, phi12_norm, phi_norm = _constraint_diagnostics(model, X, R1_hist, R2_hist)

    return {
        "t": np.arange(steps + 1) * h,
        "y": y,
        "X": X,
        "alpha_abs": alpha,
        "V": V,
        "omega_abs": omega,
        "R1": R1_hist,
        "R2": R2_hist,
        "q": q,
        "qdot": qdot,
        "energy": energy,
        "energy_error": energy - energy[0],
        "phi0_norm": phi0_norm,
        "phi12_norm": phi12_norm,
        "phi_norm": phi_norm,
    }