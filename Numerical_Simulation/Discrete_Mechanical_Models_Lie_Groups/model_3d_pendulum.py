from dataclasses import dataclass, field
import numpy as np
from numpy.linalg import solve


@dataclass
class Pendulum3DModel:
    m: float = 1.0
    g: float = 9.81
    rho_c: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.5]))
    e3: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    J: np.ndarray = field(default_factory=lambda: np.diag([0.333396, 0.333396, 0.00125]))

    def moment(self, R: np.ndarray) -> np.ndarray:
        return self.m * self.g * np.cross(self.rho_c, R.T @ self.e3)

    def potential(self, R: np.ndarray) -> float:
        return float(-self.m * self.g * (self.e3 @ (R @ self.rho_c)))

    def energy(self, R: np.ndarray, Omega: np.ndarray) -> float:
        T = 0.5 * Omega @ self.J @ Omega
        U = self.potential(R)
        return float(T + U)

    def momentum_from_pi(self, R: np.ndarray, Pi: np.ndarray) -> float:
        return float(self.e3 @ R @ Pi)

    def momentum_from_omega(self, R: np.ndarray, Omega: np.ndarray) -> float:
        return float(self.e3 @ R @ (self.J @ Omega))

    def rk4_rhs(self, R: np.ndarray, Omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from lie_group import hat

        R_dot = R @ hat(Omega)
        Omega_dot = solve(self.J, self.moment(R) - np.cross(Omega, self.J @ Omega))
        return R_dot, Omega_dot