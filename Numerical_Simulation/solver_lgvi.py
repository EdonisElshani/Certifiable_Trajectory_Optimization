from xml.parsers.expat import model

import numpy as np
from numpy.linalg import norm, solve

from lie_group import hat, rodrigues


def A_of_f(f: np.ndarray, a: np.ndarray, J: np.ndarray) -> np.ndarray:
    """
    A(f) = -a + sin(||f||)/||f|| Jf + (1-cos(||f||))/||f||^2 (f x Jf)
    """
    theta = norm(f)
    Jf = J @ f

    if theta < 1e-12:
        return -a + Jf + 0.5 * np.cross(f, Jf)

    term1 = (np.sin(theta) / theta) * Jf
    term2 = ((1.0 - np.cos(theta)) / theta**2) * np.cross(f, Jf)
    return -a + term1 + term2


def jacobian_A(f: np.ndarray, J: np.ndarray) -> np.ndarray:
    """
    Jacobian from Lee's formula ∇A(f).
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


def lgvi_step(model, R_k: np.ndarray, Pi_k: np.ndarray, f_0: np.ndarray,
              h: float, tol: float = 1e-12, max_iter: int = 50) -> tuple:
    """
    One LGVI step:
        a_k = h (Pi_k + h/2 M_k)
        solve A(f_k) = 0
        F_k = Rodrigues(f_k)
        R_{k+1} = R_k F_k
        Pi_{k+1} = F_k^T Pi_k + h/2 F_k^T M_k + h/2 M_{k+1}
    """
    a_k = model.a_k(R_k, Pi_k, h, t_k)

    f_k, n_iter, residual = solve_f_newton(
        a=a_k,
        J=model.J,
        f_init=f_0,
        tol=tol,
        max_iter=max_iter
    )

    F_k = rodrigues(f_k)
    R_k1 = R_k @ F_k
    Pi_k1 = model.update_pi_lgvi(R_k, R_k1, F_k, Pi_k, h, t_k)

    Omega_k = solve(model.J, Pi_k1)

    return R_k1, Pi_k1, Omega_k, f_k, n_iter, residual


def simulate_lgvi(model, R0: np.ndarray, Omega0: np.ndarray, h: float, tf: float,
                  tol: float = 1e-12, max_iter: int = 50) -> dict:
    N = int(tf / h)
    t = np.linspace(0.0, tf, N + 1)

    R = R0.copy()
    Pi = model.J @ Omega0

    a_0 = h * (Pi + 0.5 * h * model.moment(R))
    f_0 = solve(model.J, a_0)

    Omega_hist = np.zeros((N, 3))
    E_hist = np.zeros(N)
    mu_hist = np.zeros(N)
    orth_hist = np.zeros(N)
    res_hist = np.zeros(N)
    iter_hist = np.zeros(N)

    mu0 = model.momentum_from_pi(R, Pi)

    for k in range(N):
        R, Pi, Omega, f_0, n_iter, residual = lgvi_step(
            model, R, Pi, f_0, h, tol=tol, max_iter=max_iter
        )

        Omega_hist[k, :] = Omega
        E_hist[k] = model.energy(R, Omega)
        mu_hist[k] = model.momentum_from_pi(R, Pi)
        orth_hist[k] = norm(np.eye(3) - R.T @ R, ord="fro")
        res_hist[k] = residual
        iter_hist[k] = n_iter

    return {
        "t": t[1:],
        "Omega_hist": Omega_hist,
        "E_hist": E_hist,
        "DeltaE": E_hist - E_hist[0],
        "mu_hist": mu_hist,
        "DeltaMu": mu_hist - mu0,
        "orth_hist": orth_hist,
        "res_hist": res_hist,
        "iter_hist": iter_hist,
    }