from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class AcrobotLGVIState:
    """
    Physical state needed for one implicit LGVI step.

    X_prev:
        Maximal COM state at node k-1, shape (4,).
    X:
        Maximal COM state at node k, shape (4,).
    R1, R2:
        Current absolute SO(2) rotations at node k.
    F1_prev, F2_prev:
        Previous relative rotations F_{i,k-1} = R_{i,k-1}^T R_{i,k}.
        These encode the discrete angular velocity / momentum.
    """
    X_prev: np.ndarray
    X: np.ndarray
    R1: np.ndarray
    R2: np.ndarray
    F1_prev: np.ndarray
    F2_prev: np.ndarray