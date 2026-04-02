import numpy as np
import matplotlib.pyplot as plt

from Discrete_Mechanical_Models_Lie_Groups.model_3d_pendulum import Pendulum3DModel
from Discrete_Mechanical_Models_Lie_Groups.model_3d_pendulum_forced import ForcedPendulum3DModel

# For solving with exponential map-based LGVI:
# from solver_lgvi import simulate_lgvi

from solver_lgvi_cayley import simulate_lgvi
from solver_rk4 import simulate_rk4


def make_plots(lgvi: dict, rk4: dict, rk4_proj: dict):
    t = lgvi["t"]

    # -------------------------------
    # Global plot styling for LaTeX export
    # -------------------------------
    plt.rcParams.update({
        "font.size": 18,
        "axes.labelsize": 20,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "legend.fontsize": 16,
        "axes.linewidth": 2.0,
        "xtick.major.width": 1.8,
        "ytick.major.width": 1.8,
        "xtick.major.size": 7,
        "ytick.major.size": 7,
        "lines.linewidth": 3.0,
        "lines.markersize": 7,
        "legend.frameon": True,
    })

    def style_axes():
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
        ax.tick_params(axis="both", which="major", width=1.8, length=7)
        ax.tick_params(axis="both", which="minor", width=1.4, length=4)
        ax.grid(False)

    def finalize_plot(filename: str):
        style_axes()
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

    # -------------------------------
    # 1) Angular velocity
    # -------------------------------
    plt.figure(figsize=(8, 5.5))
    line1, = plt.plot(t, lgvi["Omega_hist"][:, 0], "-", label=r"$\Omega_1$ LGVI")
    plt.plot(t, rk4["Omega_hist"][:, 0], "--", color=line1.get_color(), label=r"$\Omega_1$ RK4")
    plt.plot(t, rk4_proj["Omega_hist"][:, 0], ":", color=line1.get_color(), label=r"$\Omega_1$ RK4-proj")

    line2, = plt.plot(t, lgvi["Omega_hist"][:, 1], "-", label=r"$\Omega_2$ LGVI")
    plt.plot(t, rk4["Omega_hist"][:, 1], "--", color=line2.get_color(), label=r"$\Omega_2$ RK4")
    plt.plot(t, rk4_proj["Omega_hist"][:, 1], ":", color=line2.get_color(), label=r"$\Omega_2$ RK4-proj")

    line3, = plt.plot(t, lgvi["Omega_hist"][:, 2], "-", label=r"$\Omega_3$ LGVI")
    plt.plot(t, rk4["Omega_hist"][:, 2], "--", color=line3.get_color(), label=r"$\Omega_3$ RK4")
    plt.plot(t, rk4_proj["Omega_hist"][:, 2], ":", color=line3.get_color(), label=r"$\Omega_3$ RK4-proj")

    plt.xlabel("discrete time-steps $k$")
    plt.ylabel(r"$\Omega$")
    plt.legend(ncol=3, loc="best")
    finalize_plot("Angular_velocity.pdf")

    # -------------------------------
    # 2) Total energy
    # -------------------------------
    plt.figure(figsize=(8, 5.5))
    plt.plot(t, lgvi["E_hist"], label="LGVI")
    plt.plot(t, rk4["E_hist"], "--", label="RK4")
    plt.plot(t, rk4_proj["E_hist"], ":", label="RK4-proj")
    plt.xlabel("discrete time-steps $k$")
    plt.ylabel(r"$E_k$")
    #plt.ylim(-0.75, 1.00)
    leg = plt.legend(loc="best")
    leg.get_frame().set_linewidth(1.5)
    finalize_plot("Total_Energy.pdf")

    # -------------------------------
    # 3) Orthogonality error
    # -------------------------------
    plt.figure(figsize=(8, 5.5))
    plt.plot(t, lgvi["orth_hist"], label="LGVI")
    plt.plot(t, rk4["orth_hist"], "--", label="RK4")
    plt.plot(t, rk4_proj["orth_hist"], ":", label="RK4-proj")
    plt.xlabel("discrete time-steps $k$")
    plt.ylabel(r"$e_{\mathrm{orth}}$")
    leg = plt.legend(loc="best")
    leg.get_frame().set_linewidth(1.5)
    finalize_plot("Orthogonality_error.pdf")

    # -------------------------------
    # 4) Momentum error
    # -------------------------------
    plt.figure(figsize=(8, 5.5))
    plt.plot(t, lgvi["DeltaMu"], label="LGVI")
    plt.plot(t, rk4["DeltaMu"], "--", label="RK4")
    plt.plot(t, rk4_proj["DeltaMu"], ":", label="RK4-proj")
    plt.xlabel("discrete time-steps $k$")
    plt.ylabel(r"$\Delta \mu_k$")
    leg = plt.legend(loc="best")
    leg.get_frame().set_linewidth(1.5)
    finalize_plot("Momentum_error.pdf")

    # -------------------------------
    # 5) Energy deviation
    # -------------------------------
    plt.figure(figsize=(8, 5.5))
    plt.plot(t, lgvi["DeltaE"], label="LGVI")
    plt.plot(t, rk4["DeltaE"], "--", label="RK4")
    plt.plot(t, rk4_proj["DeltaE"], ":", label="RK4-proj")
    plt.xlabel("discrete time-steps $k$")
    plt.ylabel(r"$\Delta E_k$")
    leg = plt.legend(loc="best")
    leg.get_frame().set_linewidth(1.5)
    finalize_plot("Delta_E.pdf")

    

if __name__ == "__main__":

    # Choose the model to simulate:
    model = ForcedPendulum3DModel()
    # model = Pendulum3DModel()

    # Simulation parameters
    h = 0.02
    tf = 100.0

    R0 = np.eye(3)
    Omega0 = np.array([4.14, 4.14, 4.14])

    lgvi = simulate_lgvi(model, R0, Omega0, h, tf)
    rk4 = simulate_rk4(model, R0, Omega0, h, tf, project=False)
    rk4_proj = simulate_rk4(model, R0, Omega0, h, tf, project=True)

    # ============================================================
    # Trajectory plots: one 3D subplot for each method
    # ============================================================
    fig_traj = plt.figure(figsize=(15, 4))
    cases = [
        ("LGVI", lgvi),
        ("RK4", rk4),
        ("Projected RK4", rk4_proj),
    ]

    for i, (title, data) in enumerate(cases, start=1):
        ax = fig_traj.add_subplot(1, 3, i, projection="3d")
        x = data["x_hist"]

        ax.plot(x[:, 0], x[:, 1], x[:, 2], linewidth=1.8)
        ax.scatter(x[0, 0], x[0, 1], x[0, 2], marker="o", s=30, label="start")
        ax.scatter(x[-1, 0], x[-1, 1], x[-1, 2], marker="x", s=40, label="end")

        ax.set_title(title, fontsize=14)
        ax.set_xlabel(r"$x_1$", fontsize=12)
        ax.set_ylabel(r"$x_2$", fontsize=12)
        ax.set_zlabel(r"$x_3$", fontsize=12)
        ax.legend(fontsize=10)

    plt.tight_layout()


    # ============================================================
    # Input components u_k
    # ============================================================
    fig_u, axes_u = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

    labels = ["LGVI", "RK4", "Projected RK4"]
    datas = [lgvi, rk4, rk4_proj]

    for j in range(3):
        for label, data in zip(labels, datas):
            axes_u[j].plot(data["t"], data["u_hist"][:, j], linewidth=1.8, label=label)

        axes_u[j].set_ylabel(rf"$u_{j+1}$", fontsize=12)
        axes_u[j].grid(True, alpha=0.3)

    axes_u[0].set_title("Control Input Components", fontsize=14)
    axes_u[-1].set_xlabel("Time [s]", fontsize=12)
    axes_u[0].legend(fontsize=10)

    plt.tight_layout()


    # ============================================================
    # Angular velocity components
    # ============================================================
    fig_omega, axes_omega = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

    for j in range(3):
        for label, data in zip(labels, datas):
            axes_omega[j].plot(data["t"], data["Omega_hist"][:, j], linewidth=1.8, label=label)

        axes_omega[j].set_ylabel(rf"$\Omega_{j+1}$", fontsize=12)
        axes_omega[j].grid(True, alpha=0.3)

    axes_omega[0].set_title("Angular Velocity Components", fontsize=14)
    axes_omega[-1].set_xlabel("Time [s]", fontsize=12)
    axes_omega[0].legend(fontsize=10)

    plt.tight_layout()
    plt.show()

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