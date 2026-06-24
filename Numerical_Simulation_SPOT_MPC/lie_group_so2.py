"""Small SO(2) helper functions for the planar Acrobot simulations."""

from __future__ import annotations

import numpy as np


def S2() -> np.ndarray:
    """Generator of so(2)."""
    return np.array([[0.0, -1.0], [1.0, 0.0]])


def R_so2(theta: float) -> np.ndarray:
    """Planar rotation matrix."""
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def F_from_delta(theta_delta: float) -> np.ndarray:
    """Relative SO(2) update F = exp(theta_delta S)."""
    return R_so2(theta_delta)


def angle_from_R(R: np.ndarray) -> float:
    """Return the rotation angle of an SO(2) matrix."""
    return float(np.arctan2(R[1, 0], R[0, 0]))


def vee2(A: np.ndarray) -> float:
    """Vee map for a 2x2 skew matrix a*S.

    For a numerically non-perfect skew matrix, returns the skew projection.
    """
    return float(0.5 * (A[1, 0] - A[0, 1]))


def hat2(a: float) -> np.ndarray:
    """Hat map for so(2): scalar -> a*S."""
    return float(a) * S2()


def cross2(a: np.ndarray, b: np.ndarray) -> float:
    """Scalar out-of-plane cross product for planar vectors.

    cross2(a,b) = a_x b_y - a_y b_x.
    """
    a = np.asarray(a, dtype=float).reshape(2)
    b = np.asarray(b, dtype=float).reshape(2)
    return float(a[0] * b[1] - a[1] * b[0])


def orth_error_so2(R: np.ndarray) -> float:
    """Frobenius orthogonality error ||I - R^T R||_F."""
    return float(np.linalg.norm(np.eye(2) - R.T @ R, ord="fro"))


def det_error_so2(R: np.ndarray) -> float:
    """Absolute determinant error |det(R)-1|."""
    return float(abs(np.linalg.det(R) - 1.0))


def project_to_so2(R: np.ndarray) -> np.ndarray:
    """Project a near-SO(2) matrix to SO(2) by extracting its angle.

    This is intentionally analogous to the optional SO(3) projection used in
    the 3D pendulum RK4 script.
    """
    return R_so2(angle_from_R(R))
