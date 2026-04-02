import numpy as np
import matplotlib.pyplot as plt

from Discrete_Mechanical_Models_Lie_Groups.model_3d_pendulum import Pendulum3DModel
from Discrete_Mechanical_Models_Lie_Groups.model_3d_pendulum_forced import ForcedPendulum3DModel

# For solving with exponential map-based LGVI:
# from solver_lgvi import simulate_lgvi

from solver_lgvi_cayley import simulate_lgvi
from solver_rk4 import simulate_rk4


# ============================================================
# User choices
# ============================================================
INCLUDE_RK4_PROJ = False


def set_axes_equal(ax):
    """
    Make a 3D plot have equal axis scaling.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def make_plots(lgvi: dict, rk4: dict, rk4_proj: dict | None = None):
    t = lgvi["t"]

    # -------------------------------
    # Global plot styling
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
        "lines.linewidth": 1.5,
        "lines.markersize": 7,
        "legend.frameon": True,
    })

    if rk4_proj is None:
        labels = ["LGVI", "RK4"]
        datas = [lgvi, rk4]
    else:
        labels = ["LGVI", "RK4", "RK4-proj"]
        datas = [lgvi, rk4, rk4_proj]

    # ============================================================
    # 1) Position trajectory comparison in 3D (pendulum-style)
    # ============================================================
    fig_traj = plt.figure(figsize=(7, 6))
    ax_traj = fig_traj.add_subplot(111, projection="3d")

    # thinner lines in this figure
    traj_lw = 1.0
    rod_lw = 0.7
    sample_step = max(1, len(lgvi["x_hist"]) // 40)   # draw only some rods

    # trajectories
    ax_traj.plot(
        lgvi["x_hist"][:, 0], lgvi["x_hist"][:, 1], lgvi["x_hist"][:, 2],
        linewidth=traj_lw, label="LGVI"
    )
    ax_traj.plot(
        rk4["x_hist"][:, 0], rk4["x_hist"][:, 1], rk4["x_hist"][:, 2],
        "--", linewidth=traj_lw, label="RK4"
    )

    if rk4_proj is not None:
        ax_traj.plot(
            rk4_proj["x_hist"][:, 0], rk4_proj["x_hist"][:, 1], rk4_proj["x_hist"][:, 2],
            ":", linewidth=traj_lw, label="RK4-proj"
        )

    # pendulum rods from origin to selected positions
    for k in range(0, len(lgvi["x_hist"]), sample_step):
        x = lgvi["x_hist"][k]
        ax_traj.plot([0, x[0]], [0, x[1]], [0, x[2]], color="C0", alpha=0.35, linewidth=rod_lw)

    for k in range(0, len(rk4["x_hist"]), sample_step):
        x = rk4["x_hist"][k]
        ax_traj.plot([0, x[0]], [0, x[1]], [0, x[2]], color="C1", alpha=0.25, linewidth=rod_lw)

    if rk4_proj is not None:
        for k in range(0, len(rk4_proj["x_hist"]), sample_step):
            x = rk4_proj["x_hist"][k]
            ax_traj.plot([0, x[0]], [0, x[1]], [0, x[2]], color="C2", alpha=0.25, linewidth=rod_lw)

    # pivot point
    ax_traj.scatter(0, 0, 0, color="black", s=40)

    # start / end markers
    x0 = lgvi["x_hist"][0]
    xN = lgvi["x_hist"][-1]
    ax_traj.scatter(x0[0], x0[1], x0[2], marker="o", s=25, color="black", label="start")
    ax_traj.scatter(xN[0], xN[1], xN[2], marker="x", s=35, color="black", label="end")

    ax_traj.set_title("Pendulum Position Trajectory", fontsize=14)
    ax_traj.set_xlabel(r"$x_1$", fontsize=12)
    ax_traj.set_ylabel(r"$x_2$", fontsize=12)
    ax_traj.set_zlabel(r"$x_3$", fontsize=12)
    set_axes_equal(ax_traj)
    ax_traj.legend(fontsize=10)

    plt.tight_layout()

    # ============================================================
    # 2) Position components x1, x2, x3
    # ============================================================
    fig_pos, axes_pos = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

    for j in range(3):
        for label, data in zip(labels, datas):
            axes_pos[j].plot(data["t"], data["x_hist"][:, j], linewidth=1.8, label=label)

        axes_pos[j].set_ylabel(rf"$x_{j+1}$", fontsize=12)
        axes_pos[j].grid(True, alpha=0.3)

    axes_pos[0].set_title("Pendulum Position Components", fontsize=14)
    axes_pos[-1].set_xlabel("Time [s]", fontsize=12)
    axes_pos[0].legend(fontsize=10)

    plt.tight_layout()

    # ============================================================
    # 3) Input components u_k
    # ============================================================
    fig_u, axes_u = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

    for j in range(3):
        for label, data in zip(labels, datas):
            axes_u[j].plot(data["t"], data["u_hist"][:, j], linewidth=1.8, label=label)

        axes_u[j].set_ylabel(rf"$u_{j+1}$", fontsize=12)
        axes_u[j].grid(True, alpha=0.3)

    axes_u[0].set_title("Applied Body Torque Components", fontsize=14)
    axes_u[-1].set_xlabel("Time [s]", fontsize=12)
    axes_u[0].legend(fontsize=10)

    plt.tight_layout()

    # ============================================================
    # 4) Angular velocity components
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

    # ============================================================
    # 5) Trajectory difference relative to LGVI
    # ============================================================
    fig_err, ax_err = plt.subplots(figsize=(8, 5.5))

    err_rk4 = np.linalg.norm(rk4["x_hist"] - lgvi["x_hist"], axis=1)
    ax_err.plot(
        lgvi["t"], err_rk4, "--", linewidth=2.0,
        label=r"$\|x_{\mathrm{RK4}} - x_{\mathrm{LGVI}}\|$"
    )

    if rk4_proj is not None:
        err_rk4_proj = np.linalg.norm(rk4_proj["x_hist"] - lgvi["x_hist"], axis=1)
        ax_err.plot(
            lgvi["t"], err_rk4_proj, ":", linewidth=2.0,
            label=r"$\|x_{\mathrm{RK4proj}} - x_{\mathrm{LGVI}}\|$"
        )

    ax_err.set_xlabel("Time [s]")
    ax_err.set_ylabel("Position error norm")
    ax_err.set_title("Trajectory Difference Relative to LGVI")
    ax_err.grid(True, alpha=0.3)
    ax_err.legend(fontsize=10)

    plt.tight_layout()

    # ============================================================
    # 6) Old benchmark plots
    # ============================================================
    # Angular velocity (old style)
    plt.figure(figsize=(8, 5.5))
    line1, = plt.plot(t, lgvi["Omega_hist"][:, 0], "-", label=r"$\Omega_1$ LGVI")
    plt.plot(t, rk4["Omega_hist"][:, 0], "--", color=line1.get_color(), label=r"$\Omega_1$ RK4")
    if rk4_proj is not None:
        plt.plot(t, rk4_proj["Omega_hist"][:, 0], ":", color=line1.get_color(), label=r"$\Omega_1$ RK4-proj")

    line2, = plt.plot(t, lgvi["Omega_hist"][:, 1], "-", label=r"$\Omega_2$ LGVI")
    plt.plot(t, rk4["Omega_hist"][:, 1], "--", color=line2.get_color(), label=r"$\Omega_2$ RK4")
    if rk4_proj is not None:
        plt.plot(t, rk4_proj["Omega_hist"][:, 1], ":", color=line2.get_color(), label=r"$\Omega_2$ RK4-proj")

    line3, = plt.plot(t, lgvi["Omega_hist"][:, 2], "-", label=r"$\Omega_3$ LGVI")
    plt.plot(t, rk4["Omega_hist"][:, 2], "--", color=line3.get_color(), label=r"$\Omega_3$ RK4")
    if rk4_proj is not None:
        plt.plot(t, rk4_proj["Omega_hist"][:, 2], ":", color=line3.get_color(), label=r"$\Omega_3$ RK4-proj")

    plt.xlabel("Time [s]")
    plt.ylabel(r"$\Omega$")
    plt.legend(ncol=3, loc="best")
    plt.tight_layout()

    # Total energy
    plt.figure(figsize=(8, 5.5))
    plt.plot(t, lgvi["E_hist"], label="LGVI")
    plt.plot(t, rk4["E_hist"], "--", label="RK4")
    if rk4_proj is not None:
        plt.plot(t, rk4_proj["E_hist"], ":", label="RK4-proj")
    plt.xlabel("Time [s]")
    plt.ylabel(r"$E_k$")
    plt.legend(loc="best")
    plt.tight_layout()

    # Orthogonality error
    plt.figure(figsize=(8, 5.5))
    plt.plot(t, lgvi["orth_hist"], label="LGVI")
    plt.plot(t, rk4["orth_hist"], "--", label="RK4")
    if rk4_proj is not None:
        plt.plot(t, rk4_proj["orth_hist"], ":", label="RK4-proj")
    plt.xlabel("Time [s]")
    plt.ylabel(r"$e_{\mathrm{orth}}$")
    plt.legend(loc="best")
    plt.tight_layout()

    # Momentum error
    plt.figure(figsize=(8, 5.5))
    plt.plot(t, lgvi["DeltaMu"], label="LGVI")
    plt.plot(t, rk4["DeltaMu"], "--", label="RK4")
    if rk4_proj is not None:
        plt.plot(t, rk4_proj["DeltaMu"], ":", label="RK4-proj")
    plt.xlabel("Time [s]")
    plt.ylabel(r"$\Delta \mu_k$")
    plt.legend(loc="best")
    plt.tight_layout()

    # Energy deviation
    plt.figure(figsize=(8, 5.5))
    plt.plot(t, lgvi["DeltaE"], label="LGVI")
    plt.plot(t, rk4["DeltaE"], "--", label="RK4")
    if rk4_proj is not None:
        plt.plot(t, rk4_proj["DeltaE"], ":", label="RK4-proj")
    plt.xlabel("Time [s]")
    plt.ylabel(r"$\Delta E_k$")
    plt.legend(loc="best")
    plt.tight_layout()


if __name__ == "__main__":

    # Choose the model to simulate:
    model = ForcedPendulum3DModel()
    # model = Pendulum3DModel()

    # Simulation parameters
    h = 0.001
    tf = 100.0

    R0 = np.eye(3)
    Omega0 = np.array([4.14, 4.14, 4.14])

    lgvi = simulate_lgvi(model, R0, Omega0, h, tf)
    rk4 = simulate_rk4(model, R0, Omega0, h, tf, project=False)

    rk4_proj = None
    if INCLUDE_RK4_PROJ:
        rk4_proj = simulate_rk4(model, R0, Omega0, h, tf, project=True)

    make_plots(lgvi, rk4, rk4_proj)

    plt.show()

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

    if rk4_proj is not None:
        print("\n----- RK4 projected onto SO(3) -----")
        print("Final orthogonality error:", rk4_proj["orth_hist"][-1])
        print("Max orthogonality error:", np.max(rk4_proj["orth_hist"]))
        print("Max |DeltaMu|:", np.max(np.abs(rk4_proj["DeltaMu"])))
        print("Max |DeltaE|:", np.max(np.abs(rk4_proj["DeltaE"])))