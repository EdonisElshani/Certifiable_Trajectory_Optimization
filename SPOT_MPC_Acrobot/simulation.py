from __future__ import annotations

from pathlib import Path
import numpy as np

from Numerical_Simulation.Discrete_Mechanical_Models_Lie_Groups.model_acrobot_so2 import (
    AcrobotSO2Model,
    AcrobotSO2Params,
)
from Numerical_Simulation.solver_lgvi_acrobot import (
    make_lgvi_state_from_relative,
    rollout_lgvi_controls,
    diagnostics_lgvi,
)
from Numerical_Simulation.lie_group_so2 import angle_from_R


def main() -> None:
    # ------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------
    h = 0.05

    params = AcrobotSO2Params(
        m1=1.0,
        m2=1.0,
        l1=0.5,
        l2=0.5,
        lc1=0.25,
        lc2=0.25,
        J1=(1.0 / 12.0) * 1.0 * 0.5**2,
        J2=(1.0 / 12.0) * 1.0 * 0.5**2,
        g=9.81,
        p0=(0.0, 0.0),
    )

    model = AcrobotSO2Model(params)

    # ------------------------------------------------------------
    # Initial condition in relative Acrobot coordinates
    # q = [theta1, theta2]
    # theta1 = shoulder angle, theta2 = relative elbow angle
    # zero means hanging downward in your model convention.
    # ------------------------------------------------------------
    q0 = np.deg2rad(np.array([5.0, 0.0]))
    qdot0 = np.array([0.0, 0.0])

    state0 = make_lgvi_state_from_relative(
        model=model,
        h=h,
        q=q0,
        qdot=qdot0,
    )

    # ------------------------------------------------------------
    # One-step open-loop test with given torque u_k
    # Later this u_k will come from the SDP extraction.
    # ------------------------------------------------------------
    u_k = 1.0
    u_sequence = np.array([u_k], dtype=float)

    sim = rollout_lgvi_controls(
        model=model,
        h=h,
        initial_state=state0,
        u_sequence=u_sequence,
        root_tol=1e-10,
        maxfev=150,
    )

    diag = diagnostics_lgvi(model, sim)

    # ------------------------------------------------------------
    # Print compact diagnostics
    # ------------------------------------------------------------
    R1_next = sim["R1"][-1]
    R2_next = sim["R2"][-1]

    alpha1_next = angle_from_R(R1_next)
    alpha2_next = angle_from_R(R2_next)

    q_next = model.relative_angles_from_absolute(
        alpha1_next,
        alpha2_next,
        wrap=True,
    )

    print("=== One-step LGVI open-loop simulation ===")
    print(f"h = {h}")
    print(f"u_k = {u_k}")
    print(f"q0      = {q0}")
    print(f"q_next  = {q_next}")
    print(f"max root residual = {np.max(sim['residual_inf']):.3e}")
    print(f"constraint norm at next state = {diag['phi_norm'][-1]:.3e}")

    # ------------------------------------------------------------
    # Save result
    # ------------------------------------------------------------
    out_dir = Path("output") / "one_step_lgvi"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_dir / "one_step_result.npz",
        t=sim["t"],
        X=sim["X"],
        R1=sim["R1"],
        R2=sim["R2"],
        F1=sim["F1"],
        F2=sim["F2"],
        dtheta=sim["dtheta"],
        u=sim["u"],
        residual_inf=sim["residual_inf"],
        phi_norm=diag["phi_norm"],
    )

    print(f"Saved result to: {out_dir / 'one_step_result.npz'}")


if __name__ == "__main__":
    main()