import numpy as np
import matplotlib.pyplot as plt

from Discrete_Mechanical_Models_Lie_Groups.model_3d_pendulum import Pendulum3DModel

# For solving with exponential map-based LGVI:
# from solver_lgvi import simulate_lgvi

from solver_lgvi_cayley import simulate_lgvi
from solver_rk4 import simulate_rk4


def make_plots(lgvi: dict, rk4: dict, rk4_proj: dict):
    t = lgvi["t"]

    plt.figure(figsize=(10, 6))
    line1, = plt.plot(t, lgvi["Omega_hist"][:, 0], "-", label=r"$\Omega_1$ LGVI")
    plt.plot(t, rk4["Omega_hist"][:, 0], "--", color=line1.get_color(), label=r"$\Omega_1$ RK4")
    plt.plot(t, rk4_proj["Omega_hist"][:, 0], ":", color=line1.get_color(), label=r"$\Omega_1$ RK4-proj")

    line2, = plt.plot(t, lgvi["Omega_hist"][:, 1], "-", label=r"$\Omega_2$ LGVI")
    plt.plot(t, rk4["Omega_hist"][:, 1], "--", color=line2.get_color(), label=r"$\Omega_2$ RK4")
    plt.plot(t, rk4_proj["Omega_hist"][:, 1], ":", color=line2.get_color(), label=r"$\Omega_2$ RK4-proj")

    line3, = plt.plot(t, lgvi["Omega_hist"][:, 2], "-", label=r"$\Omega_3$ LGVI")
    plt.plot(t, rk4["Omega_hist"][:, 2], "--", color=line3.get_color(), label=r"$\Omega_3$ RK4")
    plt.plot(t, rk4_proj["Omega_hist"][:, 2], ":", color=line3.get_color(), label=r"$\Omega_3$ RK4-proj")

    plt.title("Angular velocity: LGVI vs RK4 vs RK4-proj")
    plt.xlabel("discrete time-steps k")
    plt.ylabel(r"$\Omega$")
    plt.legend(fontsize=8, ncol=3)
    plt.tight_layout()

    plt.figure(figsize=(8, 5))
    plt.plot(t, lgvi["E_hist"], label="LGVI")
    plt.plot(t, rk4["E_hist"], "--", label="RK4")
    plt.plot(t, rk4_proj["E_hist"], ":", label="RK4-proj")
    plt.title("Total energy")
    plt.xlabel("discrete time-steps k")
    plt.ylabel("E")
    plt.ylim(-2, 0)
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(8, 5))
    plt.plot(t, lgvi["orth_hist"], label="LGVI")
    plt.plot(t, rk4["orth_hist"], "--", label="RK4")
    plt.plot(t, rk4_proj["orth_hist"], ":", label="RK4-proj")
    plt.title(r"Orthogonality error $\|I - R^T R\|_F$")
    plt.xlabel("discrete time-steps k")
    plt.ylabel("error")
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(8, 5))
    plt.plot(t, lgvi["DeltaMu"], label="LGVI")
    plt.plot(t, rk4["DeltaMu"], "--", label="RK4")
    plt.plot(t, rk4_proj["DeltaMu"], ":", label="RK4-proj")
    plt.title("Momentum error")
    plt.xlabel("discrete time-steps k")
    plt.ylabel(r"$\Delta \mu$")
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(8, 5))
    plt.plot(t, lgvi["DeltaE"], label="LGVI")
    plt.plot(t, rk4["DeltaE"], "--", label="RK4")
    plt.plot(t, rk4_proj["DeltaE"], ":", label="RK4-proj")
    plt.title(r"Energy deviation $\Delta E_k$")
    plt.xlabel("discrete time-steps k")
    plt.ylabel(r"$\Delta E$")
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    model = Pendulum3DModel()

    # Simulation parameters
    h = 0.05
    tf = 1000.0

    R0 = np.eye(3)
    Omega0 = np.array([4.14, 4.14, 4.14])

    lgvi = simulate_lgvi(model, R0, Omega0, h, tf)
    rk4 = simulate_rk4(model, R0, Omega0, h, tf, project=False)
    rk4_proj = simulate_rk4(model, R0, Omega0, h, tf, project=True)

    make_plots(lgvi, rk4, rk4_proj)

    print("----- LGVI -----")
    print("Final orthogonality error:", lgvi["orth_hist"][-1])
    print("Max orthogonality error:", np.max(lgvi["orth_hist"]))
    print("Max |DeltaMu|:", np.max(np.abs(lgvi["DeltaMu"])))
    print("Max |DeltaE|:", np.max(np.abs(lgvi["DeltaE"])))
    print("Max Newton residual:", np.max(lgvi["res_hist"]))
    print("Max Newton iterations:", np.max(lgvi["iter_hist"]))

    print("\n----- RK4 -----")
    print("Final orthogonality error:", rk4["orth_hist"][-1])
    print("Max orthogonality error:", np.max(rk4["orth_hist"]))
    print("Max |DeltaMu|:", np.max(np.abs(rk4["DeltaMu"])))
    print("Max |DeltaE|:", np.max(np.abs(rk4["DeltaE"])))

    print("\n----- RK4 projected onto SO(3) -----")
    print("Final orthogonality error:", rk4_proj["orth_hist"][-1])
    print("Max orthogonality error:", np.max(rk4_proj["orth_hist"]))
    print("Max |DeltaMu|:", np.max(np.abs(rk4_proj["DeltaMu"])))
    print("Max |DeltaE|:", np.max(np.abs(rk4_proj["DeltaE"])))