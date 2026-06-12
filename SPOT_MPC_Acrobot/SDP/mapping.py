# k = 0,...,N
LONG_PREFIXES = ["c1", "s1", "c2", "s2"]
# k = 0,...,N-1
MID_PREFIXES = ["a1", "b1", "a2", "b2"]
# k = 1,...,N-1
SHORT_PREFIXES = ["lam0x", "lam0y", "lam12x", "lam12y", "u"]


def get_var_mapping_and_dict(N):
    """Create scalar variables for the reduced x-free/v-free SO(2) Acrobot POP."""
    var_start_dict = {}
    var_mapping = {}
    cnt = 1
    long_list = list(range(N + 1))  # R_k: k=0..N
    mid_list = list(range(N))       # F_k: k=0..N-1, interval k -> k+1
    short_list = list(range(1, N))  # lambda_k, u_k: k=1..N-1

    fmt = {
        # Rotation variables R_i in SO(2)
        "c1": "c_{{1,{k}}}", "s1": "s_{{1,{k}}}",
        "c2": "c_{{2,{k}}}", "s2": "s_{{2,{k}}}",

        # Relative rotation variables F_i in SO(2)
        "a1": "a_{{1,{k}}}", "b1": "b_{{1,{k}}}",
        "a2": "a_{{2,{k}}}", "b2": "b_{{2,{k}}}",

        # Scaled Lagrange multiplier variables and scalar control input
        "lam0x": "\\lambda_{{0,x,{k}}}", "lam0y": "\\lambda_{{0,y,{k}}}",
        "lam12x": "\\lambda_{{12,x,{k}}}", "lam12y": "\\lambda_{{12,y,{k}}}",
        "u": "u_{{{k}}}",
    }

    ordered_prefixes = [
        ("c1", long_list), ("s1", long_list),
        ("c2", long_list), ("s2", long_list),
        ("a1", mid_list), ("b1", mid_list),
        ("a2", mid_list), ("b2", mid_list),
        ("lam0x", short_list), ("lam0y", short_list),
        ("lam12x", short_list), ("lam12y", short_list),
        ("u", short_list),
    ]

    for prefix, klist in ordered_prefixes:
        var_start_dict[prefix] = cnt
        for k in klist:
            var_mapping[cnt] = fmt[prefix].format(k=k)
            cnt += 1

    total_var_num = cnt - 1
    var_start_dict["N"] = N
    return var_mapping, var_start_dict, total_var_num

def get_id(prefix, k, var_start_dict, prefix_k0):
    return var_start_dict[prefix] + (k - prefix_k0[prefix])


def get_remapped_ids(params):
    """Remap reduced variables into a timestep-grouped ordering."""
    N = params['N']
    total_var_num = params['total_var_num']
    id_func = params['id']

    ids_remap = np.zeros(total_var_num, dtype=int)
    idx = 1

    for k in range(N + 1):
        ids_remap[id_func("c1", k) - 1] = idx; idx += 1
        ids_remap[id_func("s1", k) - 1] = idx; idx += 1
        ids_remap[id_func("c2", k) - 1] = idx; idx += 1
        ids_remap[id_func("s2", k) - 1] = idx; idx += 1

        if k < N:
            ids_remap[id_func("a1", k) - 1] = idx; idx += 1
            ids_remap[id_func("b1", k) - 1] = idx; idx += 1
            ids_remap[id_func("a2", k) - 1] = idx; idx += 1
            ids_remap[id_func("b2", k) - 1] = idx; idx += 1

        if 1 <= k <= N - 1:
            ids_remap[id_func("lam0x", k) - 1] = idx; idx += 1
            ids_remap[id_func("lam0y", k) - 1] = idx; idx += 1
            ids_remap[id_func("lam12x", k) - 1] = idx; idx += 1
            ids_remap[id_func("lam12y", k) - 1] = idx; idx += 1
            ids_remap[id_func("u", k) - 1] = idx; idx += 1

    return ids_remap