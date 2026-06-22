from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def generate_mpc_postprocessing(
    run_folder: str | Path,
    figures_subdir: str = "figures",
    gif_stride: Optional[int] = None,
    gif_fps: int = 25,
    max_gif_frames: int = 250,
    gif_time_scale: float = 0.5,
    dpi: int = 150,
) -> Path:
    """
    Generate all desired MPC plots and a downsampled GIF from a results folder.

    Expected files inside run_folder:
        - mpc_full_trajectory.csv   (required)
        - mpc_history.csv           (recommended)
        - mpc_summary.json          (optional)

    Output:
        run_folder / figures_subdir / ...
    """

    run_folder = Path(run_folder)
    figures_dir = run_folder / figures_subdir
    figures_dir.mkdir(parents=True, exist_ok=True)

    traj_path = run_folder / "mpc_full_trajectory.csv"
    hist_path = run_folder / "mpc_history.csv"
    summ_path = run_folder / "mpc_summary.json"

    if not traj_path.exists():
        raise FileNotFoundError(f"Missing required file: {traj_path}")

    traj = pd.read_csv(traj_path)
    hist = pd.read_csv(hist_path) if hist_path.exists() else None
    summary = None
    if summ_path.exists():
        with open(summ_path, "r") as f:
            summary = json.load(f)

    # ---------- sanity ----------
    required_traj_cols = [
        "global_time",
        "thetaR1_deg", "thetaR2_deg",
        "thetaF1_deg", "thetaF2_deg",
        "R1_orth_error", "R2_orth_error",
        "F1_orth_error", "F2_orth_error",
        "elbow_x", "elbow_y",
        "tip_x", "tip_y",
        "target_angle_error_norm_deg",
        "target_step_error_norm_deg",
        "target_error_R1_deg", "target_error_R2_deg",
        "target_error_F1_deg", "target_error_F2_deg",
    ]
    missing = [c for c in required_traj_cols if c not in traj.columns]
    if missing:
        raise ValueError(f"mpc_full_trajectory.csv is missing columns: {missing}")

    # ---------- derive targets ----------
    # By convention:
    # target_error_R1_deg = thetaR1_deg - thetaR1_des_deg
    # => thetaR1_des_deg = thetaR1_deg - target_error_R1_deg
    thetaR1_des = float((traj["thetaR1_deg"] - traj["target_error_R1_deg"]).iloc[0])
    thetaR2_des = float((traj["thetaR2_deg"] - traj["target_error_R2_deg"]).iloc[0])
    thetaF1_des = float((traj["thetaF1_deg"] - traj["target_error_F1_deg"]).iloc[0])
    thetaF2_des = float((traj["thetaF2_deg"] - traj["target_error_F2_deg"]).iloc[0])

    # ---------- lengths for animation target marker ----------
    base_x = float(traj["base_x"].iloc[0]) if "base_x" in traj.columns else 0.0
    base_y = float(traj["base_y"].iloc[0]) if "base_y" in traj.columns else 0.0

    l1 = float(np.hypot(traj["elbow_x"].iloc[0] - base_x, traj["elbow_y"].iloc[0] - base_y))
    l2 = float(np.hypot(traj["tip_x"].iloc[0] - traj["elbow_x"].iloc[0],
                        traj["tip_y"].iloc[0] - traj["elbow_y"].iloc[0]))

    # target marker positions from desired R
    r1 = np.deg2rad(thetaR1_des)
    r2 = np.deg2rad(thetaR2_des)
    target_elbow_x = base_x + l1 * np.sin(r1)
    target_elbow_y = base_y - l1 * np.cos(r1)
    target_tip_x = target_elbow_x + l2 * np.sin(r2)
    target_tip_y = target_elbow_y - l2 * np.cos(r2)

    # ---------- helper ----------
    def _save(fig: plt.Figure, filename: str) -> None:
        fig.tight_layout()
        fig.savefig(figures_dir / filename, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    # =========================================================
    # 1) MPC applied controls
    # =========================================================
    if hist is not None and "t" in hist.columns and "u" in hist.columns:
        t_ctrl = hist["t"].to_numpy()
        u_ctrl = hist["u"].to_numpy()
        # usually first row is initial state with NaN control
        mask = ~np.isnan(u_ctrl)
        t_ctrl = t_ctrl[mask]
        u_ctrl = u_ctrl[mask]
    else:
        # fallback: unique controls from trajectory at MPC boundaries
        grouped = traj.groupby("mpc_iteration", sort=True).first().reset_index()
        t_ctrl = grouped["global_time"].to_numpy()
        u_ctrl = grouped["u_applied"].to_numpy()

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.plot(t_ctrl, u_ctrl, linewidth=2)
    ax.set_title("MPC applied controls")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("applied control $u_1$")
    ax.grid(True)
    _save(fig, "control_inputs.png")

    # =========================================================
    # 2) SO(2) errors
    # =========================================================
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    t = traj["global_time"].to_numpy()
    ax.semilogy(t, traj["R1_orth_error"], linewidth=2, label="R1")
    ax.semilogy(t, traj["R2_orth_error"], linewidth=2, label="R2")
    ax.semilogy(t, traj["F1_orth_error"], linewidth=2, label="F1")
    ax.semilogy(t, traj["F2_orth_error"], linewidth=2, label="F2")
    ax.set_title("SO(2) errors for R and F")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("SO(2) orthogonality error")
    ax.grid(True, which="both")
    ax.legend()
    _save(fig, "SO2_errors_R_F.png")

    # =========================================================
    # 3) Absolute link angles + desired
    # =========================================================
    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)
    ax.plot(t, traj["thetaR1_deg"], linewidth=2, label="thetaR1")
    ax.plot(t, traj["thetaR2_deg"], linewidth=2, label="thetaR2")
    ax.axhline(thetaR1_des, linestyle="--", label="thetaR1_des")
    ax.axhline(thetaR2_des, linestyle=":", label="thetaR2_des")
    ax.set_title("Absolute link angles")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("absolute angle [deg]")
    ax.grid(True)
    ax.legend()
    _save(fig, "target_position_angles.png")

    # =========================================================
    # 4) Target rotation error norm
    # =========================================================
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.plot(t, traj["target_angle_error_norm_deg"], linewidth=2)
    ax.set_title("Target rotation error norm")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("norm error to target $R_{des}$ [deg]")
    ax.grid(True)
    _save(fig, "target_rotation_error_norm.png")

    # =========================================================
    # 5) XY trajectories
    # =========================================================
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    ax.plot(traj["elbow_x"], traj["elbow_y"], linewidth=2, label="elbow")
    ax.plot(traj["tip_x"], traj["tip_y"], linewidth=2, label="tip")
    ax.scatter([base_x], [base_y], linewidth=2, label="base")
    ax.scatter([target_elbow_x], [target_elbow_y], marker="x", s=80, label="target elbow")
    ax.scatter([target_tip_x], [target_tip_y], marker="x", s=80, label="target tip")
    ax.set_title("Acrobot trajectories")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    ax.axis("equal")
    ax.legend()
    _save(fig, "trajectories_xy.png")

    # =========================================================
    # 6) Step angles of F
    # =========================================================
    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)
    ax.plot(t, traj["thetaF1_deg"], linewidth=2, label="thetaF1")
    ax.plot(t, traj["thetaF2_deg"], linewidth=2, label="thetaF2")
    ax.axhline(thetaF1_des, linestyle="--", label="thetaF1_des")
    ax.axhline(thetaF2_des, linestyle=":", label="thetaF2_des")
    ax.set_title("Step rotation / velocity proxy")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("step angle of F [deg]")
    ax.grid(True)
    ax.legend()
    _save(fig, "velocity_F_step_angles.png")

    # =========================================================
    # 7) GIF generation (downsampled)
    # =========================================================
    # automatic stride if not provided
    n = len(traj)

    t_start = float(traj["global_time"].iloc[0])
    t_end = float(traj["global_time"].iloc[-1])
    sim_duration = max(1e-9, t_end - t_start)

    # Desired GIF playback duration.
    # gif_time_scale = 0.5 means:
    # 1 simulated second is shown in 0.5 real seconds.
    desired_gif_duration = gif_time_scale * sim_duration

    # Number of frames needed for that duration at the chosen FPS.
    target_gif_frames = max(2, int(np.ceil(desired_gif_duration * gif_fps)))

    # Also cap by max_gif_frames.
    target_gif_frames = min(target_gif_frames, max_gif_frames)

    if gif_stride is None:
        gif_stride = max(1, int(np.ceil(n / target_gif_frames)))

    gif_df = traj.iloc[::gif_stride].copy()

    # Always include final state
    if gif_df.index[-1] != traj.index[-1]:
        gif_df = pd.concat([gif_df, traj.iloc[[-1]]], ignore_index=True)
    if gif_df.index[-1] != traj.index[-1]:
        gif_df = pd.concat([gif_df, traj.iloc[[-1]]], ignore_index=True)

    # axis bounds with padding
    xs = np.concatenate([
        np.array([base_x]),
        traj["elbow_x"].to_numpy(),
        traj["tip_x"].to_numpy(),
        np.array([target_elbow_x, target_tip_x]),
    ])
    ys = np.concatenate([
        np.array([base_y]),
        traj["elbow_y"].to_numpy(),
        traj["tip_y"].to_numpy(),
        np.array([target_elbow_y, target_tip_y]),
    ])
    xpad = 0.15 * max(1e-6, xs.max() - xs.min())
    ypad = 0.15 * max(1e-6, ys.max() - ys.min())

    fig, ax = plt.subplots(figsize=(7.5, 7.5), constrained_layout=True)

    ax.set_title("Acrobot MPC")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)

    # Use equal geometry, but do not let matplotlib crop the visible frame
    ax.set_aspect("equal", adjustable="box")

    # Larger padding so the full motion and axis are visible
    motion_width = max(1e-6, xs.max() - xs.min())
    motion_height = max(1e-6, ys.max() - ys.min())

    pad = 0.25 * max(motion_width, motion_height)

    x_center = 0.5 * (xs.min() + xs.max())
    y_center = 0.5 * (ys.min() + ys.max())

    half_range = 0.5 * max(motion_width, motion_height) + pad

    ax.set_xlim(x_center - half_range, x_center + half_range)
    ax.set_ylim(y_center - half_range, y_center + half_range)

    # target markers only, no orange trace line
    ax.scatter([target_elbow_x], [target_elbow_y], marker="x", s=80, label="target elbow")
    ax.scatter([target_tip_x], [target_tip_y], marker="x", s=80, label="target tip")

    # animated artists
    link1_line, = ax.plot([], [], linewidth=3)
    link2_line, = ax.plot([], [], linewidth=3)
    base_pt, = ax.plot([], [], linewidth=2, markersize=6)
    elbow_pt, = ax.plot([], [], linewidth=2, markersize=6)
    tip_pt, = ax.plot([], [], linewidth=2, markersize=6)
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top")
    err_text = ax.text(0.02, 0.88, "", transform=ax.transAxes, va="top")

    ax.legend(loc="upper right", fontsize=8)

    def init():
        link1_line.set_data([], [])
        link2_line.set_data([], [])
        base_pt.set_data([base_x], [base_y])
        elbow_pt.set_data([], [])
        tip_pt.set_data([], [])
        time_text.set_text("")
        err_text.set_text("")
        return link1_line, link2_line, base_pt, elbow_pt, tip_pt, time_text, err_text

    def update(i):
        row = gif_df.iloc[i]

        ex, ey = float(row["elbow_x"]), float(row["elbow_y"])
        tx, ty = float(row["tip_x"]), float(row["tip_y"])

        link1_line.set_data([base_x, ex], [base_y, ey])
        link2_line.set_data([ex, tx], [ey, ty])

        base_pt.set_data([base_x], [base_y])
        elbow_pt.set_data([ex], [ey])
        tip_pt.set_data([tx], [ty])

        time_text.set_text(f"t = {row['global_time']:.3f} s")
        err_text.set_text(
            f"R-error = {row['target_angle_error_norm_deg']:.2f} deg\n"
            f"F-error = {row['target_step_error_norm_deg']:.2f} deg"
        )
        return link1_line, link2_line, base_pt, elbow_pt, tip_pt, time_text, err_text

    ani = FuncAnimation(
        fig,
        update,
        frames=len(gif_df),
        init_func=init,
        blit=True,
        repeat=False,
    )

    gif_path = figures_dir / "acrobot_mpc.gif"
    ani.save(gif_path, writer=PillowWriter(fps=gif_fps))
    plt.close(fig)

    # =========================================================
    # 8) tiny summary file for convenience
    # =========================================================
    summary_txt = figures_dir / "postprocess_summary.txt"
    with open(summary_txt, "w") as f:
        f.write("Postprocessing summary\n")
        f.write("=" * 80 + "\n")
        f.write(f"Run folder: {run_folder}\n")
        f.write(f"Figures folder: {figures_dir}\n")
        f.write(f"Trajectory rows: {len(traj)}\n")
        f.write(f"GIF stride: {gif_stride}\n")
        f.write(f"GIF frames: {len(gif_df)}\n")
        f.write(f"GIF fps: {gif_fps}\n")
        f.write(f"thetaR1_des [deg]: {thetaR1_des}\n")
        f.write(f"thetaR2_des [deg]: {thetaR2_des}\n")
        f.write(f"thetaF1_des [deg]: {thetaF1_des}\n")
        f.write(f"thetaF2_des [deg]: {thetaF2_des}\n")
        if summary is not None:
            f.write(f"Stopped: {summary.get('stopped', None)}\n")
            f.write(f"Stop reason: {summary.get('stop_reason', None)}\n")

    return figures_dir


from pathlib import Path
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Folder containing mpc_full_trajectory.csv, mpc_history.csv, and mpc_summary.json.",
    )
    args = parser.parse_args()

    if args.run is None:
        run_dir = Path(__file__).resolve().parent
    else:
        run_dir = Path(args.run)

    out = generate_mpc_postprocessing(
        run_folder=run_dir,
        gif_stride=None,
        gif_fps=50,
        max_gif_frames=250,
        gif_time_scale=2,
    )

    print(f"Generated figures in: {out}")