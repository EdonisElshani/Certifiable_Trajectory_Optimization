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

    def control(self, t: float, R: np.ndarray, Omega: np.ndarray | None = None) -> np.ndarray:
        """
        Unforced default model: zero control moment in body frame.
        """
        return np.zeros(3)

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

    def a_k(self, R_k: np.ndarray, Pi_k: np.ndarray, h: float, t_k: float) -> np.ndarray:
        """
        Cayley / LGVI implicit right-hand side for the unforced c=1/2 scheme:
            a_k = h (Pi_k + h/2 M_k)
        """
        M_k = self.moment(R_k)
        return h * (Pi_k + 0.5 * h * M_k)

    def initial_f_guess_cayley(self, R0: np.ndarray, Pi0: np.ndarray, h: float, t0: float) -> np.ndarray:
        """
        Same linearized guess as before, but now model-dependent via a_k.
        """
        from lie_group import hat

        a0 = self.a_k(R0, Pi0, h, t0)
        return 0.5 * solve(2.0 * self.J - hat(a0), a0)

    def update_pi_lgvi(
        self,
        R_k: np.ndarray,
        R_k1: np.ndarray,
        F_k: np.ndarray,
        Pi_k: np.ndarray,
        h: float,
        t_k: float,
    ) -> np.ndarray:
        """
        Unforced c=1/2 LGVI momentum update:
            Pi_{k+1} = F_k^T Pi_k + h/2 (F_k^T M_k + M_{k+1})
        """
        M_k = self.moment(R_k)
        M_k1 = self.moment(R_k1)
        return F_k.T @ Pi_k + 0.5 * h * (F_k.T @ M_k + M_k1)

    def rk4_rhs(self, R: np.ndarray, Omega: np.ndarray, t: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        from lie_group import hat

        R_dot = R @ hat(Omega)
        Omega_dot = solve(self.J, self.moment(R) - np.cross(Omega, self.J @ Omega))
        return R_dot, Omega_dot
    
    