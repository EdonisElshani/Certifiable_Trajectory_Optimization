# SPOT_MPC_Acrobot/config/config_loader.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


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
    physical = cfg["physical"]
    time = cfg["time"]
    bounds = cfg["bounds"]
    cost = cfg["cost"]
    scaling = cfg.get("scaling", {})
    sdp = cfg.get("sdp", {})
    initial = cfg["initial"]
    target = cfg["target"]

    params: Dict[str, Any] = {}

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

    params["dt_sdp"] = float(time["dt_sdp"])
    params["dt_sim"] = float(time["dt_sim"])
    params["control_interval"] = float(time.get("control_interval", params["dt_sdp"]))

    ratio = params["control_interval"] / params["dt_sim"]
    n_substeps = int(round(ratio))
    if abs(ratio - n_substeps) > 1e-10:
        raise ValueError(
            f"control_interval / dt_sim must be an integer. "
            f"Got {params['control_interval']} / {params['dt_sim']} = {ratio}"
        )

    params["sim_substeps_per_control"] = n_substeps

    # Backward compatibility for old SDP code.
    params["dt"] = params["dt_sdp"]

    params["thetaR1_0"] = deg2rad(float(initial["thetaR1_deg"]))
    params["thetaR2_0"] = deg2rad(float(initial["thetaR2_deg"]))
    params["thetaR1dot_0"] = float(initial["thetaR1dot"])
    params["thetaR2dot_0"] = float(initial["thetaR2dot"])

    params["thetaF1_0"] = 0.0
    params["thetaF2_0"] = 0.0

    params["thetaR1_des"] = deg2rad(float(target["thetaR1_deg"]))
    params["thetaR2_des"] = deg2rad(float(target["thetaR2_deg"]))

    params["u_max"] = float(bounds["u_max"])
    params["lambda_max"] = float(bounds["lambda_max"])

    max_step_angle = deg2rad(float(bounds["max_step_angle_deg"]))
    params["max_step_angle"] = max_step_angle
    params["a1_min"] = float(np.cos(max_step_angle))
    params["a2_min"] = float(np.cos(max_step_angle))

    params["N"] = int(sdp.get("horizon_N", 5))
    params["relaxation_order"] = int(sdp.get("relaxation_order", 2))
    params["clique_mode"] = sdp.get("clique_mode", "self")
    params["extraction"] = sdp.get("extraction", "ordered")

    params["rho_R"] = float(cost["rho_R"])
    params["rho_F"] = float(cost["rho_F"])
    params["alpha_R"] = float(cost["alpha_R"])
    params["alpha_F"] = float(cost["alpha_F"])
    params["gamma"] = float(cost["gamma"])

    params["lambda_scale"] = float(scaling.get("lambda_scale", 1.0))

    return params