import numpy as np
import matplotlib.pyplot as plt

from Discrete_Mechanical_Models_Lie_Groups.model_3d_pendulum import Pendulum3DModel

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
    plt.ylim(-0.75, 1.00)
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