import numpy as np
from numpy.linalg import norm

from lie_group import project_to_so3


def rk4_step(model, R: np.ndarray, Omega: np.ndarray, h: float, t: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Explicit RK4 step for the state y = (R, Omega).
    """
    k1_R, k1_Om = model.rk4_rhs(R, Omega, t)
    k2_R, k2_Om = model.rk4_rhs(R + 0.5 * h * k1_R, Omega + 0.5 * h * k1_Om, t + 0.5 * h)
    k3_R, k3_Om = model.rk4_rhs(R + 0.5 * h * k2_R, Omega + 0.5 * h * k2_Om, t + 0.5 * h)
    k4_R, k4_Om = model.rk4_rhs(R + h * k3_R, Omega + h * k3_Om, t + h)

    R_next = R + (h / 6.0) * (k1_R + 2.0 * k2_R + 2.0 * k3_R + k4_R)
    Omega_next = Omega + (h / 6.0) * (k1_Om + 2.0 * k2_Om + 2.0 * k3_Om + k4_Om)
    return R_next, Omega_next


def simulate_rk4(
    model,
    R0: np.ndarray,
    Omega0: np.ndarray,
    h: float,
    tf: float,
    project: bool = False,
) -> dict:
    N = int(tf / h)
    t = np.linspace(0.0, tf, N + 1)

    R = R0.copy()
    Omega = Omega0.copy()

    Omega_hist = np.zeros((N, 3))
    E_hist = np.zeros(N)
    mu_hist = np.zeros(N)
    orth_hist = np.zeros(N)

    mu0 = model.momentum_from_omega(R, Omega)

    for k in range(N):
        R, Omega = rk4_step(model, R, Omega, h, t[k])

        if project:
            R = project_to_so3(R)

        Omega_hist[k, :] = Omega
        E_hist[k] = model.energy(R, Omega)
        mu_hist[k] = model.momentum_from_omega(R, Omega)
        orth_hist[k] = norm(np.eye(3) - R.T @ R, ord="fro")

    return {
        "t": t[1:],
        "Omega_hist": Omega_hist,
        "E_hist": E_hist,
        "DeltaE": E_hist - E_hist[0],
        "mu_hist": mu_hist,
        "DeltaMu": mu_hist - mu0,
        "orth_hist": orth_hist,
    }