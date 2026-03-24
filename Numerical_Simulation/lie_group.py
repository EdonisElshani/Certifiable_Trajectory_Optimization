import numpy as np
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

    if theta < 1e-12:
        f_hat = hat(f)
        return np.eye(3) + f_hat + 0.5 * (f_hat @ f_hat)

    f_hat = hat(f)
    return (
        np.eye(3)
        + (np.sin(theta) / theta) * f_hat
        + ((1.0 - np.cos(theta)) / theta**2) * (f_hat @ f_hat)
    )

def cayley(f: np.ndarray) -> np.ndarray:
    """
    Cayley transform on SO(3):
        F = cay(f) = (I + f^) (I - f^)^{-1}

    This is the convention used in the computational approach you showed.
    """
    f_hat = hat(f)
    I = np.eye(3)
    return solve(I - f_hat, I + f_hat)


def project_to_so3(R: np.ndarray) -> np.ndarray:
    """
    Project 3x3 matrix onto SO(3) using the SVD projection.
    """
    U, _, Vt = np.linalg.svd(R)
    R_proj = U @ Vt

    if np.linalg.det(R_proj) < 0:
        U[:, -1] *= -1.0
        R_proj = U @ Vt

    return R_proj