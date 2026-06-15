import numpy as np

from config.config_loader import load_yaml_config, build_common_params
from Numerical_Simulation.solver_lgvi_acrobot import (
    make_initial_state_from_params,
    simulate_one_control_interval_from_params,
    diagnostics_lgvi,
)

cfg = load_yaml_config("config/acrobot_physical.yaml")
params = build_common_params(cfg)

model, state = make_initial_state_from_params(params, h_key="dt_sim")

# Replace this with your extracted SDP controls.
# For first testing, use zeros or small constant torque.
u_sdp = np.zeros(20)
# u_sdp = np.ones(20) * 1.0

all_theta = []
all_residuals = []
all_u = []

for j, u_j in enumerate(u_sdp):
    state, sim_interval = simulate_one_control_interval_from_params(
        params=params,
        model=model,
        state=state,
        u_j=float(u_j),
        root_tol=1e-10,
        maxfev=200,
        normalized=False,
    )

    diag = diagnostics_lgvi(model, sim_interval)

    all_theta.append(diag["thetaR"])
    all_residuals.append(sim_interval["residual_inf"])
    all_u.append(sim_interval["u"])

    print(
        f"interval {j:02d}: "
        f"u={u_j:+.4f}, "
        f"max_res={sim_interval['residual_inf'].max():.3e}, "
        f"theta_final={diag['thetaR'][-1]}"
    )

all_residuals = np.concatenate(all_residuals)
all_u = np.concatenate(all_u)

print("=" * 60)
print("total sim substeps:", len(all_u))
print("expected substeps:", 20 * int(round(params["control_interval"] / params["dt_sim"])))
print("global max residual:", all_residuals.max())