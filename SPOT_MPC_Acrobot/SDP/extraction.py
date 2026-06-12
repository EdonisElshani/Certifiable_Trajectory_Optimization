def extract_solution_variables(v_opt, var_start_dict, N, id_func, params=None):
    """Extract reduced variables and reconstruct x1/x2 from c,s."""
    solution_dict = {
        "x1": {},
        "x2": {},
        "R1": {},
        "R2": {},
        "F1": {},
        "F2": {},
        "lambda0": {},
        "lambda12": {},
        "u": {},
        "theta1": {},
        "theta2": {},
        "step_theta1": {},
        "step_theta2": {},
    }

    for k in range(N + 1):
        c1 = v_opt[id_func("c1", k) - 1]
        s1 = v_opt[id_func("s1", k) - 1]
        c2 = v_opt[id_func("c2", k) - 1]
        s2 = v_opt[id_func("s2", k) - 1]

        if params is not None:
            x1_obj, x2_obj = reconstruct_positions_from_cs(c1, s1, c2, s2, params)
            x1 = np.array(x1_obj, dtype=float)
            x2 = np.array(x2_obj, dtype=float)
        else:
            x1 = np.array([np.nan, np.nan])
            x2 = np.array([np.nan, np.nan])

        solution_dict["x1"][k] = x1
        solution_dict["x2"][k] = x2
        solution_dict["R1"][k] = np.array([[c1, -s1], [s1, c1]])
        solution_dict["R2"][k] = np.array([[c2, -s2], [s2, c2]])
        solution_dict["theta1"][k] = np.arctan2(s1, c1)
        solution_dict["theta2"][k] = np.arctan2(s2, c2)

        if k < N:
            a1 = v_opt[id_func("a1", k) - 1]
            b1 = v_opt[id_func("b1", k) - 1]
            a2 = v_opt[id_func("a2", k) - 1]
            b2 = v_opt[id_func("b2", k) - 1]

            solution_dict["F1"][k] = np.array([[a1, -b1], [b1, a1]])
            solution_dict["F2"][k] = np.array([[a2, -b2], [b2, a2]])
            solution_dict["step_theta1"][k] = np.arctan2(b1, a1)
            solution_dict["step_theta2"][k] = np.arctan2(b2, a2)

        if 1 <= k < N:
            solution_dict["lambda0"][k] = np.array([
                v_opt[id_func("lam0x", k) - 1],
                v_opt[id_func("lam0y", k) - 1],
            ])
            solution_dict["lambda12"][k] = np.array([
                v_opt[id_func("lam12x", k) - 1],
                v_opt[id_func("lam12y", k) - 1],
            ])
            solution_dict["u"][k] = v_opt[id_func("u", k) - 1]

    return solution_dict

def _fmt_array(arr, precision=8):
    """Compact formatter for numpy arrays in the log file."""
    if arr is None:
        return "None"
    return np.array2string(
        np.asarray(arr, dtype=float),
        precision=precision,
        suppress_small=False,
        separator=", ",
        max_line_width=10_000,
    )


def write_extraction_trajectory_to_log(log_path, solutions, gap_info, errors_by_method, N, rank_info=None):
    """
    Append detailed per-timestep trajectory values to log.txt for the relevant
    extraction methods. This is mainly for thesis/debug analysis.

    Logged for each available method in ['ordered', 'robust']:
      - R1_k, R2_k for k = 0..N
      - F1_k, F2_k for k = 0..N-1
      - x1_k, x2_k for k = 0..N
      - lambda0_k, lambda12_k for k = 1..N-1
      - u_k for k = 1..N-1
    """
    methods_to_log = [m for m in ["ordered", "robust"] if m in solutions]

    with open(log_path, "a") as log_file:
        log_file.write("\n" + "=" * 100 + "\n")
        log_file.write("DETAILED EXTRACTED ACROBOT TRAJECTORIES\n")
        log_file.write("=" * 100 + "\n")

        if rank_info is not None:
            summary = rank_info.get("summary", {})
            blocks = rank_info.get("blocks", [])
            log_file.write("\n" + "#" * 100 + "\n")
            log_file.write("MOMENT MATRIX RANK / TIGHTNESS ANALYSIS\n")
            log_file.write("#" * 100 + "\n")
            log_file.write("Teng-style diagnostic: delta_21 = |lambda_2| / |lambda_1| per moment block.\n")
            log_file.write("Small max delta_21 suggests rank-one/tight moment blocks.\n")
            log_file.write(f"number of analyzed blocks:       {summary.get('num_blocks')}\n")
            log_file.write(f"number of skipped blocks:        {summary.get('num_skipped_blocks')}\n")
            log_file.write(f"max block dimension:             {summary.get('max_block_dim')}\n")
            log_file.write(f"max numerical rank:              {summary.get('max_numerical_rank')}\n")
            log_file.write(f"rank-one blocks:                 {summary.get('num_rank_one_blocks')} / {summary.get('num_blocks')}\n")
            log_file.write(f"rank-one tolerance on delta_21:  {summary.get('rank_one_tol'):.12e}\n")
            if summary.get("max_delta_21") is not None:
                log_file.write(f"max delta_21:                    {summary.get('max_delta_21'):.12e}\n")
                log_file.write(f"mean delta_21:                   {summary.get('mean_delta_21'):.12e}\n")
                log_file.write(f"median delta_21:                 {summary.get('median_delta_21'):.12e}\n")
                log_file.write(f"min eigenvalue over all blocks:  {summary.get('min_eig_over_all_blocks'):.12e}\n")
                log_file.write(f"blocks with negative eig:        {summary.get('num_blocks_with_negative_eig')}\n")

            if blocks:
                log_file.write("\nWorst 20 blocks by delta_21:\n")
                worst = sorted(blocks, key=lambda b: b["delta_21"], reverse=True)[:20]
                for b in worst:
                    log_file.write(
                        f"  block {b['block_index']:4d}: dim={b['dim']:4d}, "
                        f"rank={b['numerical_rank']:3d}, "
                        f"delta_21={b['delta_21']:.12e}, "
                        f"lambda1={b['lambda1_abs']:.12e}, "
                        f"lambda2={b['lambda2_abs']:.12e}, "
                        f"min_eig={b['min_eig']:.12e}\n"
                    )

        if not methods_to_log:
            log_file.write("No ordered or robust extraction available. Nothing to log here.\n")
            return

        for method in methods_to_log:
            sol = solutions[method]
            log_file.write("\n" + "#" * 100 + "\n")
            log_file.write(f"{method.upper()} EXTRACTION\n")
            log_file.write("#" * 100 + "\n")

            if method in gap_info:
                gi = gap_info[method]
                log_file.write("\nCandidate suboptimality gap:\n")
                log_file.write(f"  SDP lower bound:     {gi['sdp_lower_bound']:.12e}\n")
                log_file.write(f"  extracted objective: {gi['extracted_objective']:.12e}\n")
                log_file.write(f"  absolute gap:        {gi['absolute_gap']:.12e}\n")
                log_file.write(f"  relative gap:        {gi['relative_gap']:.12e}\n")

            if method in errors_by_method:
                err = errors_by_method[method]
                log_file.write("\nSO(2) orthogonality errors, max over horizon:\n")
                for key in ["R1", "R2", "F1", "F2"]:
                    vals = err.get(key, [])
                    if len(vals) > 0:
                        log_file.write(f"  max {key}: {max(vals):.12e}\n")

            log_file.write("\nPer-timestep values:\n")
            log_file.write("- R1, R2 are 2x2 SO(2) matrices.\n")
            log_file.write("- F1, F2 are relative step rotations and exist only for k = 0..N-1.\n")
            log_file.write("- lambda0, lambda12 and u exist only for k = 1..N-1.\n")

            for k in range(N + 1):
                log_file.write("\n" + "-" * 100 + "\n")
                log_file.write(f"k = {k}\n")
                log_file.write(f"theta1 = {sol['theta1'][k]:+.12e}\n")
                log_file.write(f"theta2 = {sol['theta2'][k]:+.12e}\n")
                log_file.write(f"x1     = {_fmt_array(sol['x1'][k])}\n")
                log_file.write(f"x2     = {_fmt_array(sol['x2'][k])}\n")
                log_file.write(f"R1     = {_fmt_array(sol['R1'][k])}\n")
                log_file.write(f"R2     = {_fmt_array(sol['R2'][k])}\n")

                if k < N:
                    log_file.write(f"step_theta1 = {sol['step_theta1'][k]:+.12e}\n")
                    log_file.write(f"step_theta2 = {sol['step_theta2'][k]:+.12e}\n")
                    log_file.write(f"F1          = {_fmt_array(sol['F1'][k])}\n")
                    log_file.write(f"F2          = {_fmt_array(sol['F2'][k])}\n")
                else:
                    log_file.write("step_theta1 = None\n")
                    log_file.write("step_theta2 = None\n")
                    log_file.write("F1          = None\n")
                    log_file.write("F2          = None\n")

                if 1 <= k < N:
                    log_file.write(f"lambda0     = {_fmt_array(sol['lambda0'][k])}\n")
                    log_file.write(f"lambda12    = {_fmt_array(sol['lambda12'][k])}\n")
                    log_file.write(f"u           = {float(sol['u'][k]):+.12e}\n")
                else:
                    log_file.write("lambda0     = None\n")
                    log_file.write("lambda12    = None\n")
                    log_file.write("u           = None\n")

        log_file.write("\n" + "=" * 100 + "\n")
        log_file.write("END DETAILED EXTRACTED ACROBOT TRAJECTORIES\n")
        log_file.write("=" * 100 + "\n")
