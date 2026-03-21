import numpy as np
from scipy.linalg import expm, logm, norm
from scipy.optimize import root
import matplotlib.pyplot as plt

# =========================
# Basic Lie algebra helpers
# =========================

def hat(v: np.ndarray) -> np.ndarray:
    """Map R^3 -> so(3)."""
    return np.array([
        [0.0,   -v[2],  v[1]],
        [v[2],   0.0,  -v[0]],
        [-v[1],  v[0],  0.0]
    ])


def vee(X: np.ndarray) -> np.ndarray:
    """Map so(3) -> R^3."""
    return np.array([X[2, 1], X[0, 2], X[1, 0]])


# =========================
# Problem-specific functions
# =========================

def Jd_from_J(J: np.ndarray) -> np.ndarray:
    """
    Nonstandard inertia matrix:
        J_d = 1/2 tr(J) I - J
    """
    return 0.5 * np.trace(J) * np.eye(3) - J


def moment(R: np.ndarray, m: float, g: float, rho_c: np.ndarray, e3: np.ndarray) -> np.ndarray:
    """
    M_k = m g rho_c x (R^T e3)
    """
    return m * g * np.cross(rho_c, R.T @ e3)


def omega_from_F(F: np.ndarray, h: float) -> np.ndarray:
    """
    Recover angular velocity from F using:
        hat(Omega_k) = (1/h) log(F_k)
    """
    Omega_hat = logm(F) / h
    Omega_hat = np.real_if_close(Omega_hat)
    return vee(Omega_hat)


def potential(R: np.ndarray, m: float, g: float, rho_c: np.ndarray, e3: np.ndarray) -> float:
    """
    Continuous potential used for diagnostic energy.
    Keep this sign consistent with your thesis.
    Here:
        U(R) = m g e3^T R rho_c
    """
    return float(-m * g * (e3 @ (R @ rho_c)))


def energy(R: np.ndarray, Omega: np.ndarray, J: np.ndarray, m: float, g: float,
           rho_c: np.ndarray, e3: np.ndarray) -> float:
    """
    Continuous total energy evaluated along the discrete trajectory:
        E = 1/2 Omega^T J Omega + U(R)
    """
    T = 0.5 * Omega @ J @ Omega
    U = potential(R, m, g, rho_c, e3)
    return float(T + U)


# =========================
# Implicit solve for F_k
# =========================

def solve_F(Pi_k: np.ndarray, M_k: np.ndarray, Jd: np.ndarray, h: float,
            f_guess: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Solve the implicit equation
        (1/h)(F J_d - J_d F^T) = hat(Pi_k) + (h/2) hat(M_k)
    using F = exp(hat(f)).

    Returns:
        F_k, f_k, residual_norm
    """

    def residual(f: np.ndarray) -> np.ndarray:
        F = expm(hat(f))
        X = (F @ Jd - Jd @ F.T) / h - hat(Pi_k) - 0.5 * h * hat(M_k)
        return vee(X)

    sol = root(residual, f_guess, method="hybr")

    if not sol.success:
        raise RuntimeError(f"Implicit solve failed: {sol.message}")

    f_k = sol.x
    F_k = expm(hat(f_k))
    res_norm = norm(residual(f_k))
    return F_k, f_k, res_norm


# =========================
# One LGVI step
# =========================

def lgvi_step(R_k: np.ndarray, Pi_k: np.ndarray, f_guess: np.ndarray,
              J: np.ndarray, h: float, m: float, g: float,
              rho_c: np.ndarray, e3: np.ndarray) -> tuple:
    """
    Perform one step of the discrete Hamilton update:
        (1/h)(F_k J_d - J_d F_k^T) = Pi_k^hat + (h/2) M_k^hat
        R_{k+1} = R_k F_k
        Pi_{k+1} = F_k^T Pi_k + (h/2) F_k^T M_k + (h/2) M_{k+1}
    """
    Jd = Jd_from_J(J)

    M_k = moment(R_k, m, g, rho_c, e3)

    F_k, f_k, res_norm = solve_F(Pi_k, M_k, Jd, h, f_guess)

    R_k1 = R_k @ F_k
    M_k1 = moment(R_k1, m, g, rho_c, e3)

    Pi_k1 = F_k.T @ Pi_k + 0.5 * h * (F_k.T @ M_k + M_k1)

    Omega_k = omega_from_F(F_k, h)

    return R_k1, Pi_k1, F_k, Omega_k, M_k, M_k1, f_k, res_norm


# =========================
# Main simulation
# =========================

def run_simulation():
    # Parameters from Lee's 3D pendulum example
    m = 1.0
    g = 9.81
    rho_c = np.array([0.0, 0.0, 0.3])
    e3 = np.array([0.0, 0.0, 1.0])
    J = np.diag([0.13, 0.28, 0.17])

    h = 0.01
    tf = 100.0
    N = int(tf / h)

    # Initial conditions
    R = np.eye(3)
    Omega0 = np.array([4.14, 4.14, 4.14])

    # Practical initialization of discrete momentum
    Pi = J @ Omega0

    # Initial guess for the implicit solve
    f_guess = h * Omega0

    # Storage
    t = np.linspace(0.0, tf, N + 1)
    Omega_hist = np.zeros((N, 3))
    E_hist = np.zeros(N)
    mu_hist = np.zeros(N)
    orth_hist = np.zeros(N)
    res_hist = np.zeros(N)

    # initial momentum-map reference
    mu0 = float(e3 @ R @ Pi)

    for k in range(N):
        R, Pi, F, Omega, M_k, M_k1, f_guess, res_norm = lgvi_step(
            R, Pi, f_guess, J, h, m, g, rho_c, e3
        )

        Omega_hist[k, :] = Omega
        E_hist[k] = energy(R, Omega, J, m, g, rho_c, e3)
        mu_hist[k] = float(e3 @ R @ Pi)
        orth_hist[k] = norm(np.eye(3) - R.T @ R, ord="fro")
        res_hist[k] = res_norm

    DeltaE = E_hist - E_hist[0]
    DeltaMu = mu_hist - mu0

    return t[1:], Omega_hist, E_hist, DeltaE, mu_hist, DeltaMu, orth_hist, res_hist


# =========================
# Plotting
# =========================

def make_plots(t, Omega_hist, E_hist, DeltaE, mu_hist, DeltaMu, orth_hist, res_hist):
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))

    # Angular velocity
    axs[0, 0].plot(t, Omega_hist[:, 0], label=r"$\Omega_1$")
    axs[0, 0].plot(t, Omega_hist[:, 1], label=r"$\Omega_2$")
    axs[0, 0].plot(t, Omega_hist[:, 2], label=r"$\Omega_3$")
    axs[0, 0].set_title("Angular velocity")
    axs[0, 0].set_xlabel("t")
    axs[0, 0].set_ylabel(r"$\Omega$")
    axs[0, 0].legend()

    # Total energy
    axs[0, 1].plot(t, E_hist)
    axs[0, 1].set_title("Total energy")
    axs[0, 1].set_xlabel("t")
    axs[0, 1].set_ylabel("E")

    # Orthogonality error
    axs[1, 0].plot(t, orth_hist)
    axs[1, 0].set_title(r"Orthogonality error $\|I - R^T R\|_F$")
    axs[1, 0].set_xlabel("t")
    axs[1, 0].set_ylabel("error")

    # Momentum-map error
    axs[1, 1].plot(t, DeltaMu)
    axs[1, 1].set_title(r"Momentum-map error $e_3^T R_k \Pi_k - e_3^T R_0 \Pi_0$")
    axs[1, 1].set_xlabel("t")
    axs[1, 1].set_ylabel(r"$\Delta \mu$")

    fig.tight_layout()
    plt.show()

    # Optional extra figure for energy deviation + nonlinear residual
    fig2, axs2 = plt.subplots(1, 2, figsize=(11, 4))

    axs2[0].plot(t, DeltaE)
    axs2[0].set_title(r"Energy deviation $\Delta E_k = E_k - E_0$")
    axs2[0].set_xlabel("t")
    axs2[0].set_ylabel(r"$\Delta E$")

    axs2[1].plot(t, res_hist)
    axs2[1].set_title("Implicit solve residual norm")
    axs2[1].set_xlabel("t")
    axs2[1].set_ylabel("residual")

    fig2.tight_layout()
    plt.show()


if __name__ == "__main__":
    t, Omega_hist, E_hist, DeltaE, mu_hist, DeltaMu, orth_hist, res_hist = run_simulation()
    make_plots(t, Omega_hist, E_hist, DeltaE, mu_hist, DeltaMu, orth_hist, res_hist)

    print("Final orthogonality error:", orth_hist[-1])
    print("Max orthogonality error:", np.max(orth_hist))
    print("Max |DeltaMu|:", np.max(np.abs(DeltaMu)))
    print("Max |DeltaE|:", np.max(np.abs(DeltaE)))
    print("Max residual norm:", np.max(res_hist))