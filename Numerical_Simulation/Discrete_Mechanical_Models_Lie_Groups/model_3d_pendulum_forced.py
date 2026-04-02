from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
from numpy.linalg import solve

from Discrete_Mechanical_Models_Lie_Groups.model_3d_pendulum import Pendulum3DModel


@dataclass
class ForcedPendulum3DModel(Pendulum3DModel):
    """
    Forced 3D pendulum with c = 1/2.

    The control is injected in body coordinates as
        u_k = R_k^T e3 x u_p(t_k)
    if up_fun is supplied.

    If body_torque_fun is supplied, it is used directly as the body-frame control
    moment u(t, R, Omega).

    Priority:
        1) body_torque_fun
        2) up_fun through R^T e3 x u_p
        3) zero control
    """

    def control(self, t: float, R: np.ndarray, Omega: np.ndarray | None = None) -> np.ndarray:
        """
        Built-in prescribed control moment in body coordinates.

        Example:
            u(t) = R^T e3 x u_p(t)
        with
            u_p(t) = [0.15 sin(0.5 t), 0.10 cos(0.25 t), 0]^T
        """
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

        Pi_predict = F_k.T @ Pi_k + 0.5 * h * (F_k.T @ (M_k + u_k) + M_k1)
        Omega_k1 = solve(self.J, Pi_predict)
        u_k1 = self.control(t_k + h, R_k1, Omega_k1)

        return F_k.T @ Pi_k + 0.5 * h * (F_k.T @ (M_k + u_k) + (M_k1 + u_k1))

    def rk4_rhs(self, R: np.ndarray, Omega: np.ndarray, t: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        from lie_group import hat

        u = self.control(t, R, Omega)
        R_dot = R @ hat(Omega)
        Omega_dot = solve(self.J, self.moment(R) + u - np.cross(Omega, self.J @ Omega))
        return R_dot, Omega_dot