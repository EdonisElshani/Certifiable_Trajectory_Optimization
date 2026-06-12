# SPOT_MPC_Acrobot/config/config_loader.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml
import numpy as np


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return cfg


def deg2rad(deg: float) -> float:
    return float(np.deg2rad(deg))


def build_common_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build one shared parameter dictionary used by both:
        - Numerical_Simulation
        - SDP formulation

    This avoids changing physical parameters in multiple files.
    """
    physical = cfg["physical"]
    time = cfg["time"]
    bounds = cfg["bounds"]
    cost = cfg["cost"]
    scaling = cfg.get("scaling", {})
    sdp = cfg.get("sdp", {})
    initial = cfg["initial"]
    target = cfg["target"]

    params: Dict[str, Any] = {}

    # ------------------------------------------------------------
    # Physical parameters
    # ------------------------------------------------------------
    params["m1"] = float(physical["m1"])
    params["m2"] = float(physical["m2"])

    params["l1"] = float(physical["l1"])
    params["l2"] = float(physical["l2"])

    params["lc1"] = float(physical["lc1"])
    params["lc2"] = float(physical["lc2"])

    params["J1"] = float(physical["J1"])
    params["J2"] = float(physical["J2"])

    params["g"] = float(physical["g"])
    params["p_0"] = np.array(physical["p0"], dtype=float)

    # ------------------------------------------------------------
    # Time
    # ------------------------------------------------------------
    params["dt"] = float(time["dt"])

    # ------------------------------------------------------------
    # Initial condition
    # ------------------------------------------------------------
    params["theta1_0"] = deg2rad(initial["theta1_deg"])
    params["theta2_0"] = deg2rad(initial["theta2_deg"])
    params["theta1dot_0"] = float(initial["theta1dot"])
    params["theta2dot_0"] = float(initial["theta2dot"])

    # Depending on your SDP convention, you may also want absolute angles:
    params["thetaR1_0"] = params["theta1_0"]
    params["thetaR2_0"] = params["theta1_0"] + params["theta2_0"]

    params["thetaF1_0"] = 0.0
    params["thetaF2_0"] = 0.0

    # ------------------------------------------------------------
    # Target
    # ------------------------------------------------------------
    params["theta1_des"] = deg2rad(target["theta1_deg"])
    params["theta2_des"] = deg2rad(target["theta2_deg"])

    params["thetaR1_des"] = params["theta1_des"]
    params["thetaR2_des"] = params["theta1_des"] + params["theta2_des"]

    # ------------------------------------------------------------
    # Bounds
    # ------------------------------------------------------------
    params["u_max"] = float(bounds["u_max"])
    params["lambda_max"] = float(bounds["lambda_max"])

    max_step_angle = deg2rad(bounds["max_step_angle_deg"])
    params["max_step_angle"] = max_step_angle

    # For SO(2), step bound usually becomes:
    # a_k = cos(delta_theta_k) >= cos(delta_theta_max)
    params["a1_min"] = float(np.cos(max_step_angle))
    params["a2_min"] = float(np.cos(max_step_angle))

    # ------------------------------------------------------------
    # SDP settings
    # ------------------------------------------------------------
    params["N"] = int(sdp.get("horizon_N", 5))
    params["relaxation_order"] = int(sdp.get("relaxation_order", 2))
    params["clique_mode"] = sdp.get("clique_mode", "self")
    params["extraction"] = sdp.get("extraction", "ordered")

    # ------------------------------------------------------------
    # Cost weights
    # ------------------------------------------------------------
    params["rho_R"] = float(cost["rho_R"])
    params["rho_F"] = float(cost["rho_F"])
    params["alpha_R"] = float(cost["alpha_R"])
    params["alpha_F"] = float(cost["alpha_F"])
    params["gamma"] = float(cost["gamma"])

    # ------------------------------------------------------------
    # Scaling
    # ------------------------------------------------------------
    params["lambda_scale"] = float(scaling.get("lambda_scale", 1.0))

    return params