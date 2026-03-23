import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm, solve


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
# Rodrigues formula
# =========================

def rodrigues(f: np.ndarray) -> np.ndarray:
    """
    Rodrigues formula:
        F = I + sin(theta)/theta * f^
              + (1-cos(theta))/theta^2 * (f^)^2
    with theta = ||f||.
    """
    theta = norm(f)
    F = np.eye(3)

    if theta < 1e-12:
        f_hat = hat(f)
        return F + f_hat + 0.5 * (f_hat @ f_hat)

    f_hat = hat(f)
    return (
        np.eye(3)
        + (np.sin(theta) / theta) * f_hat
        + ((1.0 - np.cos(theta)) / theta**2) * (f_hat @ f_hat)
    )


# =========================
# Problem-specific functions
# =========================

def moment(R: np.ndarray, m: float, g: float, rho_c: np.ndarray, e3: np.ndarray) -> np.ndarray:
    """
    M_k = m g rho_c x (R^T e3)
    """
    return -m * g * np.cross(rho_c, R.T @ e3)


def omega_from_F(F: np.ndarray, h: float) -> np.ndarray:
    """
    First-order approximation:
        hat(Omega_k) ≈ (F_k - I)/h
    but for plotting we use the skew part for robustness.
    """
    Omega_hat = 0.5 * (F - F.T) / h
    return vee(Omega_hat)


def potential(R: np.ndarray, m: float, g: float, rho_c: np.ndarray, e3: np.ndarray) -> float:
    """
    Keep sign consistent with your thesis.
    Here:
        U(R) = m g e3^T R rho_c
    """
    return float(m * g * (e3 @ (R @ rho_c)))


def energy(R: np.ndarray, Omega: np.ndarray, J: np.ndarray, m: float, g: float,
           rho_c: np.ndarray, e3: np.ndarray) -> float:
    """
    Continuous total energy diagnostic:
        E = 1/2 Omega^T J Omega + U(R)
    """
    T = 0.5 * Omega @ J @ Omega
    U = potential(R, m, g, rho_c, e3)
    return float(T + U)


# =========================
# Lee's vector equation A(f) = 0
# =========================

def A_of_f(f: np.ndarray, a: np.ndarray, J: np.ndarray) -> np.ndarray:
    """
    A(f) = -a + sin(||f||)/||f|| Jf + (1-cos(||f||))/||f||^2 (f x Jf)
    """
    theta = norm(f)
    Jf = J @ f

    if theta < 1e-12:
        # Small-angle expansion:
        # sin(theta)/theta -> 1
        # (1-cos(theta))/theta^2 -> 1/2
        return -a + Jf + 0.5 * np.cross(f, Jf)

    term1 = (np.sin(theta) / theta) * Jf
    term2 = ((1.0 - np.cos(theta)) / theta**2) * np.cross(f, Jf)
    return -a + term1 + term2


def jacobian_A(f: np.ndarray, J: np.ndarray) -> np.ndarray:
    """
    Jacobian from Lee's formula:
    ∇A(f)
      = ((cos||f|| ||f|| - sin||f||)/||f||^3) Jf f^T
        + (sin||f||/||f||) J
        + ((sin||f|| ||f|| - 2(1-cos||f||))/||f||^4) (f x Jf) f^T
        + ((1-cos||f||)/||f||^2) (-hat(Jf) + hat(f) J)

    For very small ||f|| we use the linearization ∇A(0) ≈ J.
    """
    theta = norm(f)

    if theta < 1e-10:
        return J.copy()

    Jf = J @ f
    fxJf = np.cross(f, Jf)

    c1 = (np.cos(theta) * theta - np.sin(theta)) / theta**3
    c2 = np.sin(theta) / theta
    c3 = (np.sin(theta) * theta - 2.0 * (1.0 - np.cos(theta))) / theta**4
    c4 = (1.0 - np.cos(theta)) / theta**2

    term1 = c1 * np.outer(Jf, f)
    term2 = c2 * J
    term3 = c3 * np.outer(fxJf, f)
    term4 = c4 * (-hat(Jf) + hat(f) @ J)

    return term1 + term2 + term3 + term4


# =========================
# Newton solve for f_k
# =========================

def solve_f_newton(a: np.ndarray, J: np.ndarray, f_init: np.ndarray = None,
                   tol: float = 1e-12, max_iter: int = 50) -> tuple[np.ndarray, int, float]:
    """
    Solve A(f) = 0 using Newton iteration:
        f_{i+1} = f_i - [∇A(f_i)]^{-1} A(f_i)

    Initial guess:
      - previous step solution if given
      - otherwise linearized solution J^{-1} a
    """
    if f_init is None:
        f = solve(J, a)
    else:
        f = f_init.copy()

    for it in range(max_iter):
        Af = A_of_f(f, a, J)
        JAf = jacobian_A(f, J)

        delta = solve(JAf, Af)
        f_new = f - delta

        if norm(f_new - f) < tol:
            res = norm(A_of_f(f_new, a, J))
            return f_new, it + 1, res

        f = f_new

    res = norm(A_of_f(f, a, J))
    raise RuntimeError(
        f"Newton iteration did not converge in {max_iter} iterations. Residual={res:.3e}"
    )


# =========================
# One LGVI step
# =========================

def lgvi_step(R_k: np.ndarray, Pi_k: np.ndarray, f_guess: np.ndarray,
              J: np.ndarray, h: float, m: float, g: float,
              rho_c: np.ndarray, e3: np.ndarray,
              tol: float = 1e-12, max_iter: int = 50) -> tuple:
    """
    One step using Lee's Newton solve in R^3:
        a_k = h (Pi_k + h/2 M_k)
        solve A(f_k) = 0
        F_k = Rodrigues(f_k)
        R_{k+1} = R_k F_k
        Pi_{k+1} = F_k^T Pi_k + h/2 F_k^T M_k + h/2 M_{k+1}
    """
    M_k = moment(R_k, m, g, rho_c, e3)
    a_k = h * (Pi_k + 0.5 * h * M_k)

    f_k, n_iter, residual = solve_f_newton(
        a=a_k,
        J=J,
        f_init=f_guess,
        tol=tol,
        max_iter=max_iter
    )

    F_k = rodrigues(f_k)
    R_k1 = R_k @ F_k
    M_k1 = moment(R_k1, m, g, rho_c, e3)

    Pi_k1 = F_k.T @ Pi_k + 0.5 * h * (F_k.T @ M_k + M_k1)

    # For diagnostics
    Omega_k = f_k / h

    return R_k1, Pi_k1, F_k, Omega_k, M_k, M_k1, f_k, n_iter, residual


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

    # Practical initialization
    Pi = J @ Omega0
    f_guess = h * Omega0

    # Storage
    t = np.linspace(0.0, tf, N + 1)
    Omega_hist = np.zeros((N, 3))
    E_hist = np.zeros(N)
    mu_hist = np.zeros(N)
    orth_hist = np.zeros(N)
    res_hist = np.zeros(N)
    iter_hist = np.zeros(N)

    mu0 = float(e3 @ R @ Pi)

    for k in range(N):
        R, Pi, F, Omega, M_k, M_k1, f_guess, n_iter, residual = lgvi_step(
            R, Pi, f_guess, J, h, m, g, rho_c, e3
        )

        Omega_hist[k, :] = Omega
        E_hist[k] = energy(R, Omega, J, m, g, rho_c, e3)
        mu_hist[k] = float(e3 @ R @ Pi)
        orth_hist[k] = norm(np.eye(3) - R.T @ R, ord="fro")
        res_hist[k] = residual
        iter_hist[k] = n_iter

    DeltaE = E_hist - E_hist[0]
    DeltaMu = mu_hist - mu0

    return t[1:], Omega_hist, E_hist, DeltaE, mu_hist, DeltaMu, orth_hist, res_hist, iter_hist


# =========================
# Plotting
# =========================

def make_plots(t, Omega_hist, E_hist, DeltaE, mu_hist, DeltaMu, orth_hist, res_hist, iter_hist):
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))

    axs[0, 0].plot(t, Omega_hist[:, 0], label=r"$\Omega_1$")
    axs[0, 0].plot(t, Omega_hist[:, 1], label=r"$\Omega_2$")
    axs[0, 0].plot(t, Omega_hist[:, 2], label=r"$\Omega_3$")
    axs[0, 0].set_title("Angular velocity")
    axs[0, 0].set_xlabel("t")
    axs[0, 0].set_ylabel(r"$\Omega$")
    axs[0, 0].legend()

    axs[0, 1].plot(t, E_hist)
    axs[0, 1].set_title("Total energy")
    axs[0, 1].set_xlabel("t")
    axs[0, 1].set_ylabel("E")

    axs[1, 0].plot(t, orth_hist)
    axs[1, 0].set_title(r"Orthogonality error $\|I - R^T R\|_F$")
    axs[1, 0].set_xlabel("t")
    axs[1, 0].set_ylabel("error")

    axs[1, 1].plot(t, DeltaMu)
    axs[1, 1].set_title(r"Momentum-map error $e_3^T R_k \Pi_k - e_3^T R_0 \Pi_0$")
    axs[1, 1].set_xlabel("t")
    axs[1, 1].set_ylabel(r"$\Delta \mu$")

    fig.tight_layout()
    plt.show()

    fig2, axs2 = plt.subplots(1, 3, figsize=(14, 4))

    axs2[0].plot(t, DeltaE)
    axs2[0].set_title(r"Energy deviation $\Delta E_k$")
    axs2[0].set_xlabel("t")
    axs2[0].set_ylabel(r"$\Delta E$")

    axs2[1].plot(t, res_hist)
    axs2[1].set_title("Newton residual")
    axs2[1].set_xlabel("t")
    axs2[1].set_ylabel("residual")

    axs2[2].plot(t, iter_hist)
    axs2[2].set_title("Newton iterations per step")
    axs2[2].set_xlabel("t")
    axs2[2].set_ylabel("iterations")

    fig2.tight_layout()
    plt.show()


if __name__ == "__main__":
    t, Omega_hist, E_hist, DeltaE, mu_hist, DeltaMu, orth_hist, res_hist, iter_hist = run_simulation()
    make_plots(t, Omega_hist, E_hist, DeltaE, mu_hist, DeltaMu, orth_hist, res_hist, iter_hist)

    print("Final orthogonality error:", orth_hist[-1])
    print("Max orthogonality error:", np.max(orth_hist))
    print("Max |DeltaMu|:", np.max(np.abs(DeltaMu)))
    print("Max |DeltaE|:", np.max(np.abs(DeltaE)))
    print("Max Newton residual:", np.max(res_hist))
    print("Max Newton iterations:", np.max(iter_hist))