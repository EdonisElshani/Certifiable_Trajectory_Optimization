import numpy as np
from numpy.linalg import norm, solve

from lie_group import hat, cayley


def G_of_f(f: np.ndarray, q: np.ndarray, J: np.ndarray) -> np.ndarray:
    """
    Cayley-based vector equation

        G(f) = q + q x f + f (q^T f) - 2 J f = 0
        q = h (Pi_k + h/2 M_k).
    """
    return q + np.cross(q, f) + f * (q @ f) - 2.0 * (J @ f)


def jacobian_G(f: np.ndarray, q: np.ndarray, J: np.ndarray) -> np.ndarray:
    """
    Jacobian of G(f):

        ∇G(f) = q^ + (q^T f) I + f q^T - 2 J
    """
    return hat(q) + (q @ f) * np.eye(3) + np.outer(f, q) - 2.0 * J


def solve_f_newton_cayley(q: np.ndarray, J: np.ndarray, f_init: np.ndarray = None,
                          tol: float = 1e-12, max_iter: int = 50) -> tuple[np.ndarray, int, float]:
    """
    Newton iteration solving G(f) = 0:

        f_{i+1} = f_i - [∇G(f_i)]^{-1} G(f_i)

    Initial guess:
      - previous step solution if given
      - otherwise linearized guess 0.5 * J^{-1} q
    """
    if f_init is None:
        f = 0.5 * solve(J, q)
    else:
        f = f_init.copy()

    for it in range(max_iter):
        Gf = G_of_f(f, q, J)
        JG = jacobian_G(f, q, J)

        delta = solve(JG, Gf)
        f_new = f - delta

        if norm(f_new - f) < tol:
            res = norm(G_of_f(f_new, q, J))
            return f_new, it + 1, res

        f = f_new

    res = norm(G_of_f(f, q, J))
    raise RuntimeError(
        f"Cayley-Newton iteration did not converge in {max_iter} iterations. Residual={res:.3e}"
    )


def lgvi_step(model, R_k: np.ndarray, Pi_k: np.ndarray, f_0: np.ndarray,
              h: float, tol: float = 1e-12, max_iter: int = 50) -> tuple:
    """
    One LGVI step using the Cayley transformation:

        q_k = h (Pi_k + h/2 M_k)
        solve G(f_k) = 0
        F_k = cay(f_k)
        R_{k+1} = R_k F_k
        Pi_{k+1} = F_k^T Pi_k + h/2 F_k^T M_k + h/2 M_{k+1}
    """
    M_k = model.moment(R_k)
    q_k = h * (Pi_k + 0.5 * h * M_k)

    f_k, n_iter, residual = solve_f_newton_cayley(
        q=q_k,
        J=model.J,
        f_init=f_0,
        tol=tol,
        max_iter=max_iter
    )

    F_k = cayley(f_k)
    R_k1 = R_k @ F_k
    M_k1 = model.moment(R_k1)

    Pi_k1 = F_k.T @ Pi_k + 0.5 * h * (F_k.T @ M_k + M_k1)

    # Diagnostic angular velocity
    Omega_k = solve(model.J, Pi_k1)

    return R_k1, Pi_k1, Omega_k, f_k, n_iter, residual


def simulate_lgvi(model, R0: np.ndarray, Omega0: np.ndarray, h: float, tf: float,
                  tol: float = 1e-12, max_iter: int = 50) -> dict:
    """
    Same outward interface as your exponential-map solver,
    but now using the Cayley transformation internally.
    """
    N = int(tf / h)
    t = np.linspace(0.0, tf, N + 1)

    R = R0.copy()
    Pi = model.J @ Omega0

    q_0 = h * (Pi + 0.5 * h * model.moment(R))
    f_0 = 0.5 * solve(model.J, q_0)

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