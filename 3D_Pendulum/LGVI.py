import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm, solve


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


def rodrigues(f: np.ndarray) -> np.ndarray:
    """
    Rodrigues formula:
        F = I + sin(theta)/theta * f^ + (1-cos(theta))/theta^2 * (f^)^2
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

def moment(R: np.ndarray, m: float, g: float, rho_c: np.ndarray, e3: np.ndarray) -> np.ndarray:
    """
    M_k = m g rho_c x (R^T e3)
    """
    return m * g * np.cross(rho_c, R.T @ e3)


def potential(R: np.ndarray, m: float, g: float, rho_c: np.ndarray, e3: np.ndarray) -> float:
    """
        U(R) = -m g e3^T R rho_c
    """
    return float(-m * g * (e3 @ (R @ rho_c)))


def energy(R: np.ndarray, Omega: np.ndarray, J: np.ndarray, m: float, g: float,
           rho_c: np.ndarray, e3: np.ndarray) -> float:
    """
    Continuous total energy:
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
    Jacobian from Lee's formula ∇A(f)

    For very small ||f||: Linearization ∇A(0) approx J.
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


def solve_f_newton(a: np.ndarray, J: np.ndarray, f_init: np.ndarray = None,
                   tol: float = 1e-12, max_iter: int = 50) -> tuple[np.ndarray, int, float]:
    """
    Newton iteration solving A(f) = 0:
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


def lgvi_step(R_k: np.ndarray, Pi_k: np.ndarray, f_0: np.ndarray,
              J: np.ndarray, h: float, m: float, g: float,
              rho_c: np.ndarray, e3: np.ndarray,
              tol: float = 1e-12, max_iter: int = 50) -> tuple:
    """
    One step:
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
        f_init=f_0,
        tol=tol,
        max_iter=max_iter
    )

    F_k = rodrigues(f_k)
    R_k1 = R_k @ F_k
    M_k1 = moment(R_k1, m, g, rho_c, e3)

    Pi_k1 = F_k.T @ Pi_k + 0.5 * h * (F_k.T @ M_k + M_k1)

    # For diagnostics
    Omega_k = solve(J, Pi_k1)

    return R_k1, Pi_k1, F_k, Omega_k, M_k, M_k1, f_k, n_iter, residual


def rk4_rhs(R: np.ndarray, Omega: np.ndarray, J: np.ndarray,
            m: float, g: float, rho_c: np.ndarray, e3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Continuous 3D pendulum equations:
        R_dot = R * hat(Omega)
        J * Omega_dot = M(R) - Omega x (J Omega)
    """
    R_dot = R @ hat(Omega)
    Omega_dot = solve(J, moment(R, m, g, rho_c, e3) - np.cross(Omega, J @ Omega))
    return R_dot, Omega_dot


def rk4_step(R: np.ndarray, Omega: np.ndarray, J: np.ndarray, h: float,
             m: float, g: float, rho_c: np.ndarray, e3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Explicit RK4 step for the state y = (R, Omega).
    k1 = f(y_n)
    k2 = f(y_n + h/2 k1)
    k3 = f(y_n + h/2 k2)
    k4 = f(y_n + h k3)
    y_{n+1} = y_n + h/6 * (k1 + 2k2 + 2k3 + k4)
    """
    k1_R, k1_Om = rk4_rhs(R, Omega, J, m, g, rho_c, e3)
    k2_R, k2_Om = rk4_rhs(R + 0.5 * h * k1_R, Omega + 0.5 * h * k1_Om, J, m, g, rho_c, e3)
    k3_R, k3_Om = rk4_rhs(R + 0.5 * h * k2_R, Omega + 0.5 * h * k2_Om, J, m, g, rho_c, e3)
    k4_R, k4_Om = rk4_rhs(R + h * k3_R, Omega + h * k3_Om, J, m, g, rho_c, e3)

    R_next = R + (h / 6.0) * (k1_R + 2.0 * k2_R + 2.0 * k3_R + k4_R)
    Omega_next = Omega + (h / 6.0) * (k1_Om + 2.0 * k2_Om + 2.0 * k3_Om + k4_Om)

    return R_next, Omega_next


def project_to_so3(R: np.ndarray) -> np.ndarray:
    """
    Project a nearby 3x3 matrix onto SO(3) using the SVD / polar-factor projection.
    """
    U, _, Vt = np.linalg.svd(R)
    R_proj = U @ Vt

    # Enforce det = +1
    if np.linalg.det(R_proj) < 0:
        U[:, -1] *= -1.0
        R_proj = U @ Vt

    return R_proj

# Main

def run_simulation():
    # Parameter Initialization
    m = 1.0
    g = 9.81
    rho_c = np.array([0.0, 0.0, 0.3])
    e3 = np.array([0.0, 0.0, 1.0])
    J = np.diag([0.13, 0.28, 0.17])

    h = 0.05
    tf = 1000.0
    N = int(tf / h)

    R = np.eye(3)
    Omega0 = np.array([4.14, 4.14, 4.14])
    t = np.linspace(0.0, tf, N + 1)

    # LGVI
    Pi = J @ Omega0
    a_0 = h * (Pi + 0.5 * h * moment(R, m, g, rho_c, e3))
    f_0 = solve(J, a_0)

    # RK4
    R_rk = np.eye(3)
    Omega_rk = Omega0.copy()

    # RK4 projected onto SO(3)
    R_rk_proj = np.eye(3)
    Omega_rk_proj = Omega0.copy()

    # Plot Saving Data LGVI
    Omega_hist = np.zeros((N, 3))
    E_hist = np.zeros(N)
    mu_hist = np.zeros(N)
    orth_hist = np.zeros(N)
    res_hist = np.zeros(N)
    iter_hist = np.zeros(N)

    # Plot Saving Data RK4
    Omega_hist_rk = np.zeros((N, 3))
    E_hist_rk = np.zeros(N)
    mu_hist_rk = np.zeros(N)
    orth_hist_rk = np.zeros(N)

    # Plot Saving Data RK4 projected
    Omega_hist_rk_proj = np.zeros((N, 3))
    E_hist_rk_proj = np.zeros(N)
    mu_hist_rk_proj = np.zeros(N)
    orth_hist_rk_proj = np.zeros(N)

    # Momentum map values
    mu0 = float(e3 @ R @ Pi)
    mu0_rk = float(e3 @ R_rk @ (J @ Omega_rk))
    mu0_rk_proj = float(e3 @ R_rk_proj @ (J @ Omega_rk_proj))

    for k in range(N):
        # -----------------
        # LGVI step
        # -----------------
        R, Pi, F, Omega, M_k, M_k1, f_0, n_iter, residual = lgvi_step(
            R, Pi, f_0, J, h, m, g, rho_c, e3
        )

        Omega_hist[k, :] = Omega
        E_hist[k] = energy(R, Omega, J, m, g, rho_c, e3)
        mu_hist[k] = float(e3 @ R @ Pi)
        orth_hist[k] = norm(np.eye(3) - R.T @ R, ord="fro")
        res_hist[k] = residual
        iter_hist[k] = n_iter

        # -----------------
        # Plain RK4 step
        # -----------------
        R_rk, Omega_rk = rk4_step(R_rk, Omega_rk, J, h, m, g, rho_c, e3)

        Omega_hist_rk[k, :] = Omega_rk
        E_hist_rk[k] = energy(R_rk, Omega_rk, J, m, g, rho_c, e3)
        mu_hist_rk[k] = float(e3 @ R_rk @ (J @ Omega_rk))
        orth_hist_rk[k] = norm(np.eye(3) - R_rk.T @ R_rk, ord="fro")

        # -----------------
        # RK4 + projection onto SO(3)
        # -----------------
        R_rk_proj, Omega_rk_proj = rk4_step(R_rk_proj, Omega_rk_proj, J, h, m, g, rho_c, e3)
        R_rk_proj = project_to_so3(R_rk_proj)

        Omega_hist_rk_proj[k, :] = Omega_rk_proj
        E_hist_rk_proj[k] = energy(R_rk_proj, Omega_rk_proj, J, m, g, rho_c, e3)
        mu_hist_rk_proj[k] = float(e3 @ R_rk_proj @ (J @ Omega_rk_proj))
        orth_hist_rk_proj[k] = norm(np.eye(3) - R_rk_proj.T @ R_rk_proj, ord="fro")

    DeltaE = E_hist - E_hist[0]
    DeltaMu = mu_hist - mu0

    DeltaE_rk = E_hist_rk - E_hist_rk[0]
    DeltaMu_rk = mu_hist_rk - mu0_rk

    DeltaE_rk_proj = E_hist_rk_proj - E_hist_rk_proj[0]
    DeltaMu_rk_proj = mu_hist_rk_proj - mu0_rk_proj

    return (
        t[1:],
        Omega_hist, E_hist, DeltaE, mu_hist, DeltaMu, orth_hist, res_hist, iter_hist,
        Omega_hist_rk, E_hist_rk, DeltaE_rk, mu_hist_rk, DeltaMu_rk, orth_hist_rk,
        Omega_hist_rk_proj, E_hist_rk_proj, DeltaE_rk_proj, mu_hist_rk_proj, DeltaMu_rk_proj, orth_hist_rk_proj
    )


def make_plots(t,
               Omega_hist, E_hist, DeltaE, mu_hist, DeltaMu, orth_hist, res_hist, iter_hist,
               Omega_hist_rk, E_hist_rk, DeltaE_rk, mu_hist_rk, DeltaMu_rk, orth_hist_rk,
               Omega_hist_rk_proj, E_hist_rk_proj, DeltaE_rk_proj, mu_hist_rk_proj, DeltaMu_rk_proj, orth_hist_rk_proj):

    fig, axs = plt.subplots(2, 2, figsize=(13, 8))

    # Angular velocity comparison
    line1, = axs[0, 0].plot(t, Omega_hist[:, 0], "-", label=r"$\Omega_1$ LGVI")
    axs[0, 0].plot(t, Omega_hist_rk[:, 0], "--", color=line1.get_color(), label=r"$\Omega_1$ RK4")
    axs[0, 0].plot(t, Omega_hist_rk_proj[:, 0], ":", color=line1.get_color(), label=r"$\Omega_1$ RK4-proj")

    line2, = axs[0, 0].plot(t, Omega_hist[:, 1], "-", label=r"$\Omega_2$ LGVI")
    axs[0, 0].plot(t, Omega_hist_rk[:, 1], "--", color=line2.get_color(), label=r"$\Omega_2$ RK4")
    axs[0, 0].plot(t, Omega_hist_rk_proj[:, 1], ":", color=line2.get_color(), label=r"$\Omega_2$ RK4-proj")

    line3, = axs[0, 0].plot(t, Omega_hist[:, 2], "-", label=r"$\Omega_3$ LGVI")
    axs[0, 0].plot(t, Omega_hist_rk[:, 2], "--", color=line3.get_color(), label=r"$\Omega_3$ RK4")
    axs[0, 0].plot(t, Omega_hist_rk_proj[:, 2], ":", color=line3.get_color(), label=r"$\Omega_3$ RK4-proj")

    axs[0, 0].set_title("Angular velocity: LGVI vs RK4 vs RK4-proj")
    axs[0, 0].set_xlabel("t")
    axs[0, 0].set_ylabel(r"$\Omega$")
    axs[0, 0].legend(fontsize=7, ncol=3)

    # Total energy comparison
    axs[0, 1].plot(t, E_hist, label="LGVI")
    axs[0, 1].plot(t, E_hist_rk, "--", label="RK4")
    axs[0, 1].plot(t, E_hist_rk_proj, ":", label="RK4-proj")
    axs[0, 1].set_title("Total energy")
    axs[0, 1].set_xlabel("t")
    axs[0, 1].set_ylabel("E")
    axs[0, 1].set_ylim(1.6, 2.2)
    axs[0, 1].legend()

    # Orthogonality error comparison
    axs[1, 0].plot(t, orth_hist, label="LGVI")
    axs[1, 0].plot(t, orth_hist_rk, "--", label="RK4")
    axs[1, 0].plot(t, orth_hist_rk_proj, ":", label="RK4-proj")
    axs[1, 0].set_title(r"Orthogonality error $\|I - R^T R\|_F$")
    axs[1, 0].set_xlabel("t")
    axs[1, 0].set_ylabel("error")
    axs[1, 0].legend()

    # Momentum error comparison
    axs[1, 1].plot(t, DeltaMu, label="LGVI")
    axs[1, 1].plot(t, DeltaMu_rk, "--", label="RK4")
    axs[1, 1].plot(t, DeltaMu_rk_proj, ":", label="RK4-proj")
    axs[1, 1].set_title(r"Momentum error")
    axs[1, 1].set_xlabel("t")
    axs[1, 1].set_ylabel(r"$\Delta \mu$")
    axs[1, 1].legend()

    fig.tight_layout()
    plt.show()

    fig2, axs2 = plt.subplots(1, 3, figsize=(15, 4))

    axs2[0].plot(t, DeltaE, label="LGVI")
    axs2[0].plot(t, DeltaE_rk, "--", label="RK4")
    axs2[0].plot(t, DeltaE_rk_proj, ":", label="RK4-proj")
    axs2[0].set_title(r"Energy deviation $\Delta E_k$")
    axs2[0].set_xlabel("t")
    axs2[0].set_ylabel(r"$\Delta E$")
    axs2[0].legend()

    axs2[1].plot(t, res_hist)
    axs2[1].set_title("LGVI Newton residual")
    axs2[1].set_xlabel("t")
    axs2[1].set_ylabel("residual")

    axs2[2].plot(t, iter_hist)
    axs2[2].set_title("LGVI Newton iterations per step")
    axs2[2].set_xlabel("t")
    axs2[2].set_ylabel("iterations")

    fig2.tight_layout()
    plt.show()


if __name__ == "__main__":
    (
        t,
        Omega_hist, E_hist, DeltaE, mu_hist, DeltaMu, orth_hist, res_hist, iter_hist,
        Omega_hist_rk, E_hist_rk, DeltaE_rk, mu_hist_rk, DeltaMu_rk, orth_hist_rk,
        Omega_hist_rk_proj, E_hist_rk_proj, DeltaE_rk_proj, mu_hist_rk_proj, DeltaMu_rk_proj, orth_hist_rk_proj
    ) = run_simulation()

    make_plots(
        t,
        Omega_hist, E_hist, DeltaE, mu_hist, DeltaMu, orth_hist, res_hist, iter_hist,
        Omega_hist_rk, E_hist_rk, DeltaE_rk, mu_hist_rk, DeltaMu_rk, orth_hist_rk,
        Omega_hist_rk_proj, E_hist_rk_proj, DeltaE_rk_proj, mu_hist_rk_proj, DeltaMu_rk_proj, orth_hist_rk_proj
    )

    print("----- LGVI -----")
    print("Final orthogonality error:", orth_hist[-1])
    print("Max orthogonality error:", np.max(orth_hist))
    print("Max |DeltaMu|:", np.max(np.abs(DeltaMu)))
    print("Max |DeltaE|:", np.max(np.abs(DeltaE)))
    print("Max Newton residual:", np.max(res_hist))
    print("Max Newton iterations:", np.max(iter_hist))

    print("\n----- RK4 -----")
    print("Final orthogonality error:", orth_hist_rk[-1])
    print("Max orthogonality error:", np.max(orth_hist_rk))
    print("Max |DeltaMu|:", np.max(np.abs(DeltaMu_rk)))
    print("Max |DeltaE|:", np.max(np.abs(DeltaE_rk)))

    print("\n----- RK4 projected onto SO(3) -----")
    print("Final orthogonality error:", orth_hist_rk_proj[-1])
    print("Max orthogonality error:", np.max(orth_hist_rk_proj))
    print("Max |DeltaMu|:", np.max(np.abs(DeltaMu_rk_proj)))
    print("Max |DeltaE|:", np.max(np.abs(DeltaE_rk_proj)))