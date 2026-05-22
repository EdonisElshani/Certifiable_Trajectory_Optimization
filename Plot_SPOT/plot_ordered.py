# python Plot_SPOT/plot_ordered.py --N 20 --angle_deg 90 --axis x --show_diagnosics --show_window
import argparse
import re
import webbrowser
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False


def rotation_matrix_from_axis(axis, angle_rad):
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    axis = axis.lower()

    if axis == "x":
        return np.array([
            [1.0, 0.0, 0.0],
            [0.0, c,   -s ],
            [0.0, s,    c ],
        ])
    elif axis == "y":
        return np.array([
            [ c,  0.0,  s ],
            [0.0, 1.0, 0.0],
            [-s,  0.0,  c ],
        ])
    elif axis == "z":
        return np.array([
            [ c,  -s,  0.0],
            [ s,   c,  0.0],
            [0.0, 0.0, 1.0],
        ])
    else:
        raise ValueError(f"Unsupported axis '{axis}'. Choose from 'x', 'y', or 'z'.")


def parse_vector_arg(vec_str):
    parts = [float(x.strip()) for x in vec_str.split(",")]
    if len(parts) != 3:
        raise ValueError("rho_c must have exactly three comma-separated entries, e.g. '0,0,0.5'")
    return np.array(parts, dtype=float)


def get_var_mapping_and_dict(N):
    var_start_dict = {}
    cnt = 1

    long_list = list(range(N + 1))
    f_list = list(range(N))
    u_list = list(range(1, N))

    oriented_prefixes = [
        ("r11", long_list), ("r12", long_list), ("r13", long_list),
        ("r21", long_list), ("r22", long_list), ("r23", long_list),
        ("r31", long_list), ("r32", long_list), ("r33", long_list),
        ("f11", f_list), ("f12", f_list), ("f13", f_list),
        ("f21", f_list), ("f22", f_list), ("f23", f_list),
        ("f31", f_list), ("f32", f_list), ("f33", f_list),
        ("up1", u_list), ("up2", u_list), ("up3", u_list),
    ]

    for prefix, klist in oriented_prefixes:
        var_start_dict[prefix] = cnt
        cnt += len(klist)

    total_var_num = cnt - 1
    return var_start_dict, total_var_num


def get_id(prefix, k, var_start_dict, prefix_k0):
    return var_start_dict[prefix] + (k - prefix_k0[prefix])


def find_text_file(search_dir, explicit_name=None):
    search_dir = Path(search_dir)

    if explicit_name is not None:
        candidate = search_dir / explicit_name
        if not candidate.exists():
            raise FileNotFoundError(f"Could not find the specified text file: {candidate}")
        return candidate

    txt_files = sorted(search_dir.glob("*.txt"))
    if len(txt_files) == 0:
        raise FileNotFoundError(
            f"No .txt file found in {search_dir}. Put the log text file in the same folder as this script."
        )
    if len(txt_files) == 1:
        return txt_files[0]

    txt_files = sorted(txt_files, key=lambda p: p.stat().st_mtime, reverse=True)
    return txt_files[0]


def parse_ordered_vector_from_log(log_path):
    text = Path(log_path).read_text(encoding="utf-8", errors="ignore")
    pattern = re.compile(
        r"Ordered solution \(v_opt_ordered\):\s*\[([^\]]+)\]",
        re.DOTALL
    )
    match = pattern.search(text)
    if match is None:
        raise ValueError(
            f"Could not find 'Ordered solution (v_opt_ordered): [ ... ]' in {log_path}"
        )

    body = match.group(1)
    vec = np.fromstring(body.replace("\n", " "), sep=" ")
    if vec.size == 0:
        raise ValueError("Found the ordered extraction block, but failed to parse any numbers.")
    return vec


def extract_solution_variables(v_opt, N):
    prefix_k0 = {
        "r11": 0, "r12": 0, "r13": 0,
        "r21": 0, "r22": 0, "r23": 0,
        "r31": 0, "r32": 0, "r33": 0,
        "f11": 0, "f12": 0, "f13": 0,
        "f21": 0, "f22": 0, "f23": 0,
        "f31": 0, "f32": 0, "f33": 0,
        "up1": 1, "up2": 1, "up3": 1,
    }

    var_start_dict, total_var_num = get_var_mapping_and_dict(N)
    if len(v_opt) != total_var_num:
        raise ValueError(
            f"Vector length mismatch: got {len(v_opt)} entries, expected {total_var_num} for N={N}."
        )

    id_func = lambda prefix, k: get_id(prefix, k, var_start_dict, prefix_k0)

    sol = {"R": {}, "F": {}, "u_p": {}}

    for k in range(N + 1):
        R_k = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                prefix = f"r{i+1}{j+1}"
                R_k[i, j] = v_opt[id_func(prefix, k) - 1]
        sol["R"][k] = R_k

    for k in range(N):
        F_k = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                prefix = f"f{i+1}{j+1}"
                F_k[i, j] = v_opt[id_func(prefix, k) - 1]
        sol["F"][k] = F_k

    for k in range(1, N):
        u_p_k = np.zeros(3)
        for i in range(3):
            prefix = f"up{i+1}"
            u_p_k[i] = v_opt[id_func(prefix, k) - 1]
        sol["u_p"][k] = u_p_k

    return sol


def compute_R_tracking_error(sol, N, R_des):
    return np.array([np.linalg.norm(sol["R"][k] - R_des, ord="fro") for k in range(N + 1)])


def compute_so3_errors(sol, N):
    R_orth = np.zeros(N + 1)
    R_det = np.zeros(N + 1)
    for k in range(N + 1):
        R = sol["R"][k]
        R_orth[k] = np.linalg.norm(R.T @ R - np.eye(3), ord="fro")
        R_det[k] = abs(np.linalg.det(R) - 1.0)

    F_orth = np.zeros(N)
    F_det = np.zeros(N)
    for k in range(N):
        F = sol["F"][k]
        F_orth[k] = np.linalg.norm(F.T @ F - np.eye(3), ord="fro")
        F_det[k] = abs(np.linalg.det(F) - 1.0)

    return R_orth, R_det, F_orth, F_det


def compute_pendulum_positions(sol, N, rho_c):
    return np.array([sol["R"][k] @ rho_c for k in range(N + 1)])


def compute_F_speed_proxy(sol, N, dt):
    theta = np.zeros(N)
    speed = np.zeros(N)

    for k in range(N):
        F = sol["F"][k]
        c = 0.5 * (np.trace(F) - 1.0)
        c = np.clip(c, -1.0, 1.0)
        theta[k] = np.arccos(c)
        speed[k] = theta[k] / dt

    return theta, speed






def set_equal_3d_axes(ax, positions, rho_c=None):
    """
    Give a 3D axis equal scaling so the pendulum path is not visually distorted.
    Includes the origin so the pivot is always visible.
    """
    points = [np.asarray(positions, dtype=float), np.zeros((1, 3))]
    if rho_c is not None:
        points.append(np.asarray(rho_c, dtype=float).reshape(1, 3))

    all_points = np.vstack(points)
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    centers = 0.5 * (mins + maxs)

    span = np.max(maxs - mins)
    if span < 1e-12:
        span = max(1.0, np.linalg.norm(rho_c) if rho_c is not None else 1.0)

    half = 0.6 * span
    ax.set_xlim(centers[0] - half, centers[0] + half)
    ax.set_ylim(centers[1] - half, centers[1] + half)
    ax.set_zlim(centers[2] - half, centers[2] + half)

    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def get_plane_coords(positions, axis_name, N):
    """
    Get coordinates perpendicular to rotation axis + time steps.
    Returns: x_data, y_data, z_data (where y is time steps k)
    """
    k_steps = np.arange(N + 1)
    
    if axis_name == "x":
        # Rotating around x, show k-y-z
        return k_steps, positions[:, 1], positions[:, 2], "k", "y", "z"
    elif axis_name == "y":
        # Rotating around y, show k-x-z
        return k_steps, positions[:, 0], positions[:, 2], "k", "x", "z"
    else:  # axis_name == "z"
        # Rotating around z, show x-k-y
        return positions[:, 0], k_steps, positions[:, 1], "x", "k", "y"


def save_interactive_plotly_html(positions, N, axis_name, out_path):
    if not PLOTLY_AVAILABLE:
        return False

    x_data, y_data, z_data, xlabel, ylabel, zlabel = get_plane_coords(positions, axis_name, N)
    
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=x_data, y=y_data, z=z_data,
        mode="lines+markers",
        name="trajectory",
        marker=dict(size=4),
        line=dict(width=5)
    ))
    fig.add_trace(go.Scatter3d(
        x=[x_data[0]], y=[y_data[0]], z=[z_data[0]],
        mode="markers",
        name="start (k=0)",
        marker=dict(size=8, color="red", symbol="circle")
    ))
    fig.add_trace(go.Scatter3d(
        x=[x_data[-1]], y=[y_data[-1]], z=[z_data[-1]],
        mode="markers",
        name=f"end (k={N})",
        marker=dict(size=8, color="orange", symbol="diamond")
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            zaxis_title=zlabel,
            aspectmode="auto"
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.write_html(str(out_path), include_plotlyjs="cdn")
    return True



def maybe_show_matplotlib_window(positions, N, axis_name, rho_c=None):
    """
    Open a rotatable matplotlib 3D window using the same coordinate transformation as plots.
    This works on a local machine with a GUI backend.
    On a headless cluster, there is no live window unless you use X forwarding.
    """
    x_data, y_data, z_data, xlabel, ylabel, zlabel = get_plane_coords(positions, axis_name, N)
    
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(x_data, y_data, z_data,
            marker="o", linewidth=2, markersize=4, label="trajectory")
    ax.scatter([x_data[0]], [y_data[0]], [z_data[0]],
               color="red", marker="o", s=80, label="start")
    ax.scatter([x_data[-1]], [y_data[-1]], [z_data[-1]],
               color="orange", marker="^", s=90, label="end")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    # ax.set_title(f"Interactive matplotlib 3D pendulum path (rotating around {axis_name})")
    ax.legend(loc="best")
    ax.mouse_init()
    fig.tight_layout()
    plt.show()


def plot_diagnostics(sol, N, R_des, rho_c, dt, axis_name, angle_deg, outdir, make_html=True, show_combined=False):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

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

    k_R = np.arange(N + 1)
    k_F = np.arange(N)
    k_u = np.arange(1, N)

    R_err = compute_R_tracking_error(sol, N, R_des)
    R_orth, R_det, F_orth, F_det = compute_so3_errors(sol, N)
    positions = compute_pendulum_positions(sol, N, rho_c)
    theta_F, speed_F = compute_F_speed_proxy(sol, N, dt)
    U = np.array([sol["u_p"][k] for k in k_u]) if len(k_u) > 0 else np.zeros((0, 3))

    # 1) Control inputs
    fig = plt.figure(figsize=(8, 5))
    if len(k_u) > 0:
        plt.plot(k_u, U[:, 0], marker="o", label=r"$u_{p,1,k}$")
        plt.plot(k_u, U[:, 1], marker="o", label=r"$u_{p,2,k}$")
        plt.plot(k_u, U[:, 2], marker="o", label=r"$u_{p,3,k}$")
    plt.xlabel("k")
    plt.ylabel("control value")
    # plt.title("Control inputs")
    plt.grid(True)
    if len(k_u) > 0:
        plt.legend()
    fig.tight_layout()
    fig.savefig(outdir / "u_components.png", dpi=200)
    if not show_combined:
        plt.close(fig)

    # 2) Rotation tracking error
    fig = plt.figure(figsize=(8, 5))
    plt.plot(k_R, R_err, marker="o",
             label=rf"$\|R_k - R_{{\mathrm{{des}}}}\|_F$, axis={axis_name}, angle={angle_deg:g}°")
    plt.xlabel("k")
    plt.ylabel("Frobenius error")
    # plt.title(f"Rotation tracking error ({angle_deg:g}° about {axis_name})")
    plt.grid(True)
    plt.legend()
    fig.tight_layout()
    fig.savefig(outdir / "R_tracking_error.png", dpi=200)
    if not show_combined:
        plt.close(fig)

    # 3) Orthogonality
    fig = plt.figure(figsize=(8, 5))
    plt.semilogy(k_R, R_orth, marker="o", label=r"$\|R_k^\top R_k - I\|_F$")
    plt.semilogy(k_F, F_orth, marker="s", label=r"$\|F_k^\top F_k - I\|_F$")
    plt.xlabel("k")
    plt.ylabel("orthogonality error")
    # plt.title("SO(3) orthogonality errors")
    plt.grid(True, which="both")
    plt.legend()
    fig.tight_layout()
    fig.savefig(outdir / "so3_orthogonality_errors.png", dpi=200)
    if not show_combined:
        plt.close(fig)

    # 4) Determinants
    fig = plt.figure(figsize=(8, 5))
    plt.semilogy(k_R, R_det, marker="o", label=r"$|\det(R_k)-1|$")
    plt.semilogy(k_F, F_det, marker="s", label=r"$|\det(F_k)-1|$")
    plt.xlabel("k")
    plt.ylabel("determinant error")
    # plt.title("SO(3) determinant errors")
    plt.grid(True, which="both")
    plt.legend()
    fig.tight_layout()
    fig.savefig(outdir / "so3_determinant_errors.png", dpi=200)
    if not show_combined:
        plt.close(fig)

    # 5) Static 3D path: perpendicular plane vs time steps k
    x_data, y_data, z_data, xlabel, ylabel, zlabel = get_plane_coords(positions, axis_name, N)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x_data, y_data, z_data,
            marker="o", linewidth=2, markersize=4, label="trajectory")
    ax.scatter([x_data[0]], [y_data[0]], [z_data[0]],
               color="red", marker="o", s=70, label="start (k=0)")
    ax.scatter([x_data[-1]], [y_data[-1]], [z_data[-1]],
               color="orange", marker="^" , s=80, label=f"end (k={N})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    # ax.set_title(f"3D pendulum path (rotating around {axis_name})")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outdir / "pendulum_path_3d.png", dpi=200)
    if not show_combined:
        plt.close(fig)

    # 6) Speed proxy
    fig = plt.figure(figsize=(8, 5))
    plt.plot(k_F, speed_F, marker="o", label=r"$\theta(F_k)/\Delta t$")
    plt.xlabel("k")
    plt.ylabel("rad/s")
    # plt.title(r"Discrete angular speed from $F_k$")
    plt.grid(True)
    plt.legend()
    fig.tight_layout()
    fig.savefig(outdir / "F_speed_proxy.png", dpi=200)
    if not show_combined:
        plt.close(fig)

    # Combined 2x3 diagnostics page
    combined_png = outdir / "all_6_diagnostics_subplots.png"
    combined_pdf = outdir / "all_6_diagnostics_subplots.pdf"

    fig = plt.figure(figsize=(18, 10), constrained_layout=True)

    ax1 = fig.add_subplot(2, 3, 1)
    if len(k_u) > 0:
        ax1.plot(k_u, U[:, 0], marker="o", label=r"$u_{p,1,k}$")
        ax1.plot(k_u, U[:, 1], marker="o", label=r"$u_{p,2,k}$")
        ax1.plot(k_u, U[:, 2], marker="o", label=r"$u_{p,3,k}$")
        ax1.legend(fontsize=16)
    ax1.set_xlabel("k")
    ax1.set_ylabel("control")
    # ax1.set_title("Control inputs")
    ax1.grid(True)

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(k_R, R_err, marker="o",
             label=rf"$\|R_k - R_{{\mathrm{{des}}}}\|_F$")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Frobenius error")
    # ax2.set_title(f"Tracking error ({angle_deg:g}° about {axis_name})")
    ax2.grid(True)
    ax2.legend(fontsize=16)

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.semilogy(k_R, R_orth, marker="o", label=r"$\|R_k^\top R_k - I\|_F$")
    ax3.semilogy(k_F, F_orth, marker="s", label=r"$\|F_k^\top F_k - I\|_F$")
    ax3.set_xlabel("k")
    ax3.set_ylabel("orthogonality error")
    # ax3.set_title("SO(3) orthogonality")
    ax3.grid(True, which="both")
    ax3.legend(fontsize=16)

    ax4 = fig.add_subplot(2, 3, 4)
    ax4.semilogy(k_R, R_det, marker="o", label=r"$|\det(R_k)-1|$")
    ax4.semilogy(k_F, F_det, marker="s", label=r"$|\det(F_k)-1|$")
    ax4.set_xlabel("k")
    ax4.set_ylabel("determinant error")
    # ax4.set_title("SO(3) determinant")
    ax4.grid(True, which="both")
    ax4.legend(fontsize=16)

    ax5 = fig.add_subplot(2, 3, 5, projection="3d")
    x_data, y_data, z_data, xlabel, ylabel, zlabel = get_plane_coords(positions, axis_name, N)
    ax5.plot(x_data, y_data, z_data,
             marker="o", linewidth=2, markersize=4, label="trajectory")
    ax5.scatter([x_data[0]], [y_data[0]], [z_data[0]],
                color="red", marker="o", s=60, label="start (k=0)")
    ax5.scatter([x_data[-1]], [y_data[-1]], [z_data[-1]],
                color="orange", marker="^" , s=70, label=f"end (k={N})")
    ax5.set_xlabel(xlabel)
    ax5.set_ylabel(ylabel)
    ax5.set_zlabel(zlabel)
    # ax5.set_title(f"3D path ({axis_name}-axis)")
    ax5.legend(fontsize=16, loc="best")

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(k_F, speed_F, marker="o", label=r"$\theta(F_k)/\Delta t$")
    ax6.set_xlabel("k")
    ax6.set_ylabel("rad/s")
    # ax6.set_title("Discrete angular speed")
    ax6.grid(True)
    ax6.legend(fontsize=16)

    # fig.suptitle("Ordered-extraction diagnostics", fontsize=16)
    fig.savefig(combined_png, dpi=220, bbox_inches="tight")
    fig.savefig(combined_pdf, bbox_inches="tight")
    if not show_combined:
        plt.close(fig)

    html_sub = outdir / "pendulum_path_3d_interactive.html"
    interactive_saved = False
    if make_html:
        interactive_saved = save_interactive_plotly_html(positions, N, axis_name, html_sub)

    return {
        "R_tracking_error": R_err,
        "R_orthogonality_error": R_orth,
        "F_orthogonality_error": F_orth,
        "R_determinant_error": R_det,
        "F_determinant_error": F_det,
        "positions": positions,
        "F_speed_proxy": speed_F,
        "interactive_saved": interactive_saved,
        "html_sub": html_sub,
        "combined_png": combined_png,
        "combined_pdf": combined_pdf,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Plots ordered-extraction diagnostics plus 3D pendulum path and F-speed. Can also create interactive HTML."
    )
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--angle_deg", type=float, required=True)
    parser.add_argument("--axis", type=str, required=True, choices=["x", "y", "z"])
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--rho_c", type=str, default="0,0,0.5")
    parser.add_argument("--txt", type=str, default=None)
    parser.add_argument("--open_html", action="store_true",
                        help="Open the interactive HTML in the default browser after saving.")
    parser.add_argument("--show_window", action="store_true",
                        help="Open an interactive matplotlib 3D window. Works only on machines with GUI display.")
    parser.add_argument("--show_diagnostics", action="store_true",
                        help="Open the combined 6-panel diagnostics figure in a matplotlib window.")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    log_path = find_text_file(script_dir, args.txt)

    angle_rad = np.deg2rad(args.angle_deg)
    R_des = rotation_matrix_from_axis(args.axis, angle_rad)
    rho_c = parse_vector_arg(args.rho_c)

    v_opt_ordered = parse_ordered_vector_from_log(log_path)
    sol = extract_solution_variables(v_opt_ordered, args.N)

    plots_dir = script_dir / "ordered_extraction_plots"
    errors = plot_diagnostics(
        sol, args.N, R_des, rho_c, args.dt, args.axis, args.angle_deg,
        plots_dir, make_html=True, show_combined=args.show_diagnostics
    )

    print("Used text file:", log_path)
    print("Desired axis:", args.axis)
    print("Desired angle [deg]:", args.angle_deg)
    print("dt:", args.dt)
    print("rho_c:", rho_c)
    print("matplotlib backend:", matplotlib.get_backend())
    print("R_des =\n", R_des)
    print("||I - R_des||_F =", np.linalg.norm(np.eye(3) - R_des, ord="fro"))
    print("first tracking error =", errors["R_tracking_error"][0])
    print("final tracking error =", errors["R_tracking_error"][-1])
    print("Saved plots to:", plots_dir.resolve())
    print("Interactive HTML saved:", errors["html_sub"])
    print("Combined subplot page (PNG):", errors["combined_png"])
    print("Combined subplot page (PDF):", errors["combined_pdf"])

    if args.open_html and errors["interactive_saved"]:
        webbrowser.open(errors["html_sub"].resolve().as_uri())

    if args.show_window:
        maybe_show_matplotlib_window(errors["positions"], args.N, args.axis, rho_c=rho_c)

    if args.show_diagnostics:
        plt.show()


if __name__ == "__main__":
    main()
