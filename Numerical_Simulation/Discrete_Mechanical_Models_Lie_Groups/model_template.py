from dataclasses import dataclass, field
import numpy as np


@dataclass
class MyNewModel:
    J: np.ndarray = field(default_factory=lambda: np.eye(3))

    def moment(self, R: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def potential(self, R: np.ndarray) -> float:
        raise NotImplementedError

    def energy(self, R: np.ndarray, Omega: np.ndarray) -> float:
        T = 0.5 * Omega @ self.J @ Omega
        U = self.potential(R)
        return float(T + U)

    def momentum_from_pi(self, R: np.ndarray, Pi: np.ndarray) -> float:
        raise NotImplementedError

    def momentum_from_omega(self, R: np.ndarray, Omega: np.ndarray) -> float:
        raise NotImplementedError

    def rk4_rhs(self, R: np.ndarray, Omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError