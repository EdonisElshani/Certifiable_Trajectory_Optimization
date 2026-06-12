def get_cliques_for_cstss(N, params):
    """
    Conservative 21-variable SELF cliques for the reduced x-free/v-free Acrobot.

    Interior clique I_k, k=1,...,N-1:
        {R_{1,k-1}, R_{2,k-1}, R_{1,k}, R_{2,k},
         F_{1,k-1}, F_{2,k-1}, F_{1,k}, F_{2,k},
         lambda_{0,k}, lambda_{12,k}, u_k}

    with R_i=(c_i,s_i), F_i=(a_i,b_i), lambda_0,lambda_12 in R^2.
    Size: 4 + 4 + 4 + 4 + 4 + 1 = 21.

    A small terminal clique is added to cover R_N, F_{N-1}, terminal objective,
    and the final kinematics R_N = R_{N-1}F_{N-1}. The largest clique stays 21.
    """
    id_func = params['id']
    cliques = []

    def vid(prefix, k):
        return id_func(prefix, k)

    def dedupe_keep_order(items):
        out, seen = [], set()
        for item in items:
            if item not in seen:
                out.append(item)
                seen.add(item)
        return out

    for k in range(1, N):
        clique = [
            # R_{1,k-1}, R_{2,k-1}
            vid("c1", k - 1), vid("s1", k - 1),
            vid("c2", k - 1), vid("s2", k - 1),

            # R_{1,k}, R_{2,k}
            vid("c1", k), vid("s1", k),
            vid("c2", k), vid("s2", k),

            # F_{1,k-1}, F_{2,k-1}
            vid("a1", k - 1), vid("b1", k - 1),
            vid("a2", k - 1), vid("b2", k - 1),

            # F_{1,k}, F_{2,k}
            vid("a1", k), vid("b1", k),
            vid("a2", k), vid("b2", k),

            # multipliers and control at k
            vid("lam0x", k), vid("lam0y", k),
            vid("lam12x", k), vid("lam12y", k),
            vid("u", k),
        ]
        cliques.append(dedupe_keep_order(clique))

    # terminal / kinematics endpoint clique for R_N and F_{N-1}
    if N >= 1:
        terminal = [
            vid("c1", N - 1), vid("s1", N - 1),
            vid("c2", N - 1), vid("s2", N - 1),
            vid("c1", N), vid("s1", N),
            vid("c2", N), vid("s2", N),
            vid("a1", N - 1), vid("b1", N - 1),
            vid("a2", N - 1), vid("b2", N - 1),
        ]
        cliques.append(dedupe_keep_order(terminal))

    unique_cliques, seen = [], set()
    for clique in cliques:
        key = tuple(clique)
        if key not in seen:
            unique_cliques.append(clique)
            seen.add(key)

    sizes = [len(c) for c in unique_cliques]
    print("\n" + "=" * 80)
    print("SELF CLIQUE DEBUG: reduced conservative 21-variable cliques")
    print("=" * 80)
    print(f"number of SELF cliques: {len(unique_cliques)}")
    if sizes:
        print(f"min clique size:        {min(sizes)}")
        print(f"max clique size:        {max(sizes)}")
        print(f"mean clique size:       {np.mean(sizes):.2f}")
        print(f"largest clique sizes:   {sorted(sizes)[-20:]}")
    else:
        print("No SELF cliques created. This happens if N < 1.")
    print("=" * 80 + "\n")

    return unique_cliques