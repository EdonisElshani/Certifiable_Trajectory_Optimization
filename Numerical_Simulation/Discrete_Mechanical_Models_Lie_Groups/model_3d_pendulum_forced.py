from dataclasses import dataclass
import numpy as np
from numpy.linalg import solve

from Discrete_Mechanical_Models_Lie_Groups.model_3d_pendulum import Pendulum3DModel


@dataclass
class ForcedPendulum3DModel(Pendulum3DModel):
    """
    Forced 3D pendulum with c = 1/2.

    Built-in control moment in body coordinates:
        u(t) = R^T e3 x u_p(t),
    with
        u_p(t) = [0.15 sin(0.5 t), 0.10 cos(0.25 t), 0]^T.
    """

    def control(self, t: float, R: np.ndarray, Omega: np.ndarray | None = None) -> np.ndarray:
        u_p = np.array([
            0.15 * np.sin(0.5 * t),
            0.10 * np.cos(0.25 * t),
            0.0,
        ])
        return np.cross(R.T @ self.e3, u_p)

    def a_k(self, R_k: np.ndarray, Pi_k: np.ndarray, h: float, t_k: float) -> np.ndarray:
        """
        Forced c = 1/2 LGVI implicit right-hand side:
            a_k = h (Pi_k + h/2 (M_k + u_k))
        """
        M_k = self.moment(R_k)
        Omega_k = solve(self.J, Pi_k)
        u_k = self.control(t_k, R_k, Omega_k)
        return h * (Pi_k + 0.5 * h * (M_k + u_k))

    def initial_f_guess_cayley(self, R0: np.ndarray, Pi0: np.ndarray, h: float, t0: float) -> np.ndarray:
        """
        Forced model gets its own startup through its own a_k.
        """
        from lie_group import hat

        a0 = self.a_k(R0, Pi0, h, t0)
        return solve(2.0 * self.J - hat(a0), a0)

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
        Forced c = 1/2 LGVI momentum update:
            Pi_{k+1}
              = F_k^T Pi_k
                + h/2 F_k^T (M_k + u_k)
                + h/2 (M_{k+1} + u_{k+1})
        """
        M_k = self.moment(R_k)
        M_k1 = self.moment(R_k1)

        Omega_k = solve(self.J, Pi_k)
        u_k = self.control(t_k, R_k, Omega_k)
        u_k1 = self.control(t_k + h, R_k1, None)

        return F_k.T @ Pi_k + 0.5 * h * (F_k.T @ (M_k + u_k) + (M_k1 + u_k1))

    def rk4_rhs(self, R: np.ndarray, Omega: np.ndarray, t: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        from lie_group import hat

        u = self.control(t, R, Omega)
        R_dot = R @ hat(Omega)
        Omega_dot = solve(self.J, self.moment(R) + u - np.cross(Omega, self.J @ Omega))
        return R_dot, Omega_dot