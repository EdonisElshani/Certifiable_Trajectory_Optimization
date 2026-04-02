import numpy as np
from numpy.linalg import norm, solve

from lie_group import hat, cayley


def G_of_f(f: np.ndarray, a_k: np.ndarray, J: np.ndarray) -> np.ndarray:
    return a_k + np.cross(a_k, f) + f * (a_k @ f) - 2.0 * (J @ f)


def jacobian_G(f: np.ndarray, a_k: np.ndarray, J: np.ndarray) -> np.ndarray:
    return hat(a_k) + (a_k @ f) * np.eye(3) + np.outer(f, a_k) - 2.0 * J


def solve_f_newton_cayley(
    a_k: np.ndarray,
    J: np.ndarray,
    f_init: np.ndarray = None,
    tol: float = 1e-12,
    max_iter: int = 50,
) -> tuple[np.ndarray, int, float]:
    if f_init is None:
        f = solve(2.0 * J - hat(a_k), a_k)
    else:
        f = f_init.copy()

    for it in range(max_iter):
        Gf = G_of_f(f, a_k, J)
        JG = jacobian_G(f, a_k, J)
        delta = solve(JG, Gf)

        alpha = 1.0
        f_new = f - alpha * delta
        while norm(G_of_f(f_new, a_k, J)) > norm(Gf) and alpha > 1e-6:
            alpha *= 0.5
            f_new = f - alpha * delta

        if norm(f_new - f) < tol:
            res = norm(G_of_f(f_new, a_k, J))
            return f_new, it + 1, res

        f = f_new

    res = norm(G_of_f(f, a_k, J))
    raise RuntimeError(
        f"Cayley-Newton iteration did not converge in {max_iter} iterations. Residual={res:.3e}"
    )


def lgvi_step(
    model,
    R_k: np.ndarray,
    Pi_k: np.ndarray,
    f_0: np.ndarray,
    h: float,
    t_k: float,
    tol: float = 1e-12,
    max_iter: int = 50,
) -> tuple:
    a_k = model.a_k(R_k, Pi_k, h, t_k)

    f_k, n_iter, residual = solve_f_newton_cayley(
        a_k=a_k,
        J=model.J,
        f_init=f_0,
        tol=tol,
        max_iter=max_iter,
    )

    F_k = cayley(f_k)
    R_k1 = R_k @ F_k
    Pi_k1 = model.update_pi_lgvi(R_k, R_k1, F_k, Pi_k, h, t_k)
    Omega_k1 = solve(model.J, Pi_k1)

    return R_k1, Pi_k1, Omega_k1, f_k, n_iter, residual


def simulate_lgvi(
    model,
    R0: np.ndarray,
    Omega0: np.ndarray,
    h: float,
    tf: float,
    tol: float = 1e-12,
    max_iter: int = 50,
) -> dict:
    N = int(tf / h)
    t = np.linspace(0.0, tf, N + 1)

    R = R0.copy()
    Pi = model.J @ Omega0
    f_0 = model.initial_f_guess_cayley(R, Pi, h, t[0])

    R_hist = np.zeros((N, 3, 3))
    x_hist = np.zeros((N, 3))
    Omega_hist = np.zeros((N, 3))
    u_hist = np.zeros((N, 3))

    E_hist = np.zeros(N)
    mu_hist = np.zeros(N)
    orth_hist = np.zeros(N)
    res_hist = np.zeros(N)
    iter_hist = np.zeros(N)

    mu0 = model.momentum_from_pi(R, Pi)

    for k in range(N):
        R, Pi, Omega, f_0, n_iter, residual = lgvi_step(
            model,
            R,
            Pi,
            f_0,
            h,
            t[k],
            tol=tol,
            max_iter=max_iter,
        )

        t_out = t[k + 1]
        u_out = model.control(t_out, R, Omega)

        R_hist[k] = R
        x_hist[k] = R @ model.rho_c
        Omega_hist[k] = Omega
        u_hist[k] = u_out

        E_hist[k] = model.energy(R, Omega)
        mu_hist[k] = model.momentum_from_pi(R, Pi)
        orth_hist[k] = norm(np.eye(3) - R.T @ R, ord="fro")
        res_hist[k] = residual
        iter_hist[k] = n_iter

    return {
        "t": t[1:],
        "R_hist": R_hist,
        "x_hist": x_hist,
        "Omega_hist": Omega_hist,
        "u_hist": u_hist,
        "E_hist": E_hist,
        "DeltaE": E_hist - E_hist[0],
        "mu_hist": mu_hist,
        "DeltaMu": mu_hist - mu0,
        "orth_hist": orth_hist,
        "res_hist": res_hist,
        "iter_hist": iter_hist,
    }