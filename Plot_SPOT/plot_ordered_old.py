import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt



def rotation_matrix_from_axis(axis, angle_rad):
    """Return rotation matrix for a rotation about axis in {'x','y','z'}."""
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

def rot_x(angle_rad):
    """Rotation matrix for a rotation about the x-axis by angle_rad."""
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c,   -s ],
        [0.0, s,    c ],
    ])


def get_var_mapping_and_dict(N):
    var_start_dict = {}
    cnt = 1

    long_list = list(range(N + 1))   # R_k: k = 0..N
    f_list = list(range(N))          # F_k: k = 0..N-1
    u_list = list(range(1, N))       # u_p_k: k = 1..N-1

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
    R_err = np.zeros(N + 1)
    for k in range(N + 1):
        R_err[k] = np.linalg.norm(sol["R"][k] - R_des, ord="fro")
    return R_err


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


def plot_diagnostics(sol, N, R_des, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    k_R = np.arange(N + 1)
    k_F = np.arange(N)
    k_u = np.arange(1, N)

    R_err = compute_R_tracking_error(sol, N, R_des)
    R_orth, R_det, F_orth, F_det = compute_so3_errors(sol, N)

    U = np.array([sol["u_p"][k] for k in k_u]) if len(k_u) > 0 else np.zeros((0, 3))

    fig = plt.figure(figsize=(8, 5))
    if len(k_u) > 0:
        plt.plot(k_u, U[:, 0], marker="o", label=r"$u_{p,1,k}$")
        plt.plot(k_u, U[:, 1], marker="o", label=r"$u_{p,2,k}$")
        plt.plot(k_u, U[:, 2], marker="o", label=r"$u_{p,3,k}$")
    plt.xlabel("k")
    plt.ylabel("control value")
    plt.title("Control inputs from ordered extraction")
    plt.grid(True)
    plt.legend()
    fig.tight_layout()
    fig.savefig(outdir / "u_components.png", dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(k_R, R_err, marker="o", label=r"$\|R_k - R_{\mathrm{des}}\|_F$")
    plt.xlabel("k")
    plt.ylabel("Frobenius error")
    plt.title("Rotation tracking error")
    plt.grid(True)
    plt.legend()
    fig.tight_layout()
    fig.savefig(outdir / "R_tracking_error.png", dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 5))
    plt.semilogy(k_R, R_orth, marker="o", label=r"$\|R_k^\top R_k - I\|_F$")
    plt.semilogy(k_F, F_orth, marker="s", label=r"$\|F_k^\top F_k - I\|_F$")
    plt.xlabel("k")
    plt.ylabel("orthogonality error")
    plt.title("SO(3) orthogonality errors")
    plt.grid(True, which="both")
    plt.legend()
    fig.tight_layout()
    fig.savefig(outdir / "so3_orthogonality_errors.png", dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 5))
    plt.semilogy(k_R, R_det, marker="o", label=r"$|\det(R_k)-1|$")
    plt.semilogy(k_F, F_det, marker="s", label=r"$|\det(F_k)-1|$")
    plt.xlabel("k")
    plt.ylabel("determinant error")
    plt.title("SO(3) determinant errors")
    plt.grid(True, which="both")
    plt.legend()
    fig.tight_layout()
    fig.savefig(outdir / "so3_determinant_errors.png", dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(12, 10))

    ax1 = fig.add_subplot(2, 2, 1)
    if len(k_u) > 0:
        ax1.plot(k_u, U[:, 0], marker="o", label=r"$u_{p,1,k}$")
        ax1.plot(k_u, U[:, 1], marker="o", label=r"$u_{p,2,k}$")
        ax1.plot(k_u, U[:, 2], marker="o", label=r"$u_{p,3,k}$")
    ax1.set_xlabel("k")
    ax1.set_ylabel("control value")
    ax1.set_title("Control inputs")
    ax1.grid(True)
    ax1.legend()

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(k_R, R_err, marker="o", label=r"$\|R_k - R_{\mathrm{des}}\|_F$")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Frobenius error")
    ax2.set_title("Rotation tracking error")
    ax2.grid(True)
    ax2.legend()

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.semilogy(k_R, R_orth, marker="o", label=r"$\|R_k^\top R_k - I\|_F$")
    ax3.semilogy(k_F, F_orth, marker="s", label=r"$\|F_k^\top F_k - I\|_F$")
    ax3.set_xlabel("k")
    ax3.set_ylabel("orthogonality error")
    ax3.set_title("Orthogonality errors")
    ax3.grid(True, which="both")
    ax3.legend()

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.semilogy(k_R, R_det, marker="o", label=r"$|\det(R_k)-1|$")
    ax4.semilogy(k_F, F_det, marker="s", label=r"$|\det(F_k)-1|$")
    ax4.set_xlabel("k")
    ax4.set_ylabel("determinant error")
    ax4.set_title("Determinant errors")
    ax4.grid(True, which="both")
    ax4.legend()

    fig.tight_layout()
    fig.savefig(outdir / "ordered_extraction_diagnostics.png", dpi=200)
    plt.close(fig)

    return {
        "R_tracking_error": R_err,
        "R_orthogonality_error": R_orth,
        "F_orthogonality_error": F_orth,
        "R_determinant_error": R_det,
        "F_determinant_error": F_det,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Reads a .txt log and plots ordered-extraction diagnostics."
    )
    parser.add_argument("--N", type=int, required=True,
                        help="Planning horizon N used in the optimization.")
    parser.add_argument("--angle_deg", type=float, required=True,
                        help="Desired rotation angle in degrees.")
    parser.add_argument("--axis", type=str, required=True, choices=["x", "y", "z"],
                        help="Rotation axis for R_des: x, y, or z.")
    parser.add_argument("--txt", type=str, default=None,
                        help="Optional text file name in the same folder.")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    log_path = find_text_file(script_dir, args.txt)

    angle_rad = np.deg2rad(args.angle_deg)
    R_des = rotation_matrix_from_axis(args.axis, angle_rad)

    v_opt_ordered = parse_ordered_vector_from_log(log_path)
    sol = extract_solution_variables(v_opt_ordered, args.N)

    plots_dir = script_dir / "ordered_extraction_plots"
    errors = plot_diagnostics(sol, args.N, R_des, plots_dir)

    print("Used text file:", log_path)
    print("Desired axis:", args.axis)
    print("Desired angle [deg]:", args.angle_deg)
    print("R_des =\n", R_des)
    print("Parsed ordered extraction vector with length:", len(v_opt_ordered))
    print("Saved plots to:", plots_dir.resolve())
    print("\nSummary:")
    print("  max ||R_k - R_des||_F      =", np.max(errors["R_tracking_error"]))
    print("  max ||R_k^T R_k - I||_F    =", np.max(errors["R_orthogonality_error"]))
    print("  max ||F_k^T F_k - I||_F    =", np.max(errors["F_orthogonality_error"]))
    print("  max |det(R_k) - 1|         =", np.max(errors["R_determinant_error"]))
    print("  max |det(F_k) - 1|         =", np.max(errors["F_determinant_error"]))

if __name__ == "__main__":
    main()