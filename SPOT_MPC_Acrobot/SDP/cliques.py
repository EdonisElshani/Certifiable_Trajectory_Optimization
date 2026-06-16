# SDP/cliques.py

from __future__ import annotations

import numpy as np


def _dedupe_keep_order(items):
    out = []
    seen = set()

    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)

    return out


def get_cliques_for_cstss(N: int, params: dict):
    """
    Conservative SELF cliques for the reduced Acrobot.

    Interior dynamics at k = 1,...,N-1 depend on:

        R_{k-1}, R_k,
        F_{k-1}, F_k,
        lambda_k,
        u_k.

    We also add small kinematic endpoint cliques.
    """
    idf = params["id"]
    cliques = []

    def vid(prefix, k):
        return idf(prefix, k)

    # Initial/MPC boundary clique: R0, F0, R1.
    init_clique = [
        vid("c1", 0),
        vid("s1", 0),
        vid("c2", 0),
        vid("s2", 0),
        vid("a1", 0),
        vid("b1", 0),
        vid("a2", 0),
        vid("b2", 0),
        vid("c1", 1),
        vid("s1", 1),
        vid("c2", 1),
        vid("s2", 1),
    ]
    cliques.append(_dedupe_keep_order(init_clique))

    # Interior dynamics cliques.
    for k in range(1, N):
        clique = [
            # R_{k-1}
            vid("c1", k - 1),
            vid("s1", k - 1),
            vid("c2", k - 1),
            vid("s2", k - 1),

            # R_k
            vid("c1", k),
            vid("s1", k),
            vid("c2", k),
            vid("s2", k),

            # F_{k-1}
            vid("a1", k - 1),
            vid("b1", k - 1),
            vid("a2", k - 1),
            vid("b2", k - 1),

            # F_k
            vid("a1", k),
            vid("b1", k),
            vid("a2", k),
            vid("b2", k),

            # lambda_k and u_k
            vid("lam0x", k),
            vid("lam0y", k),
            vid("lam12x", k),
            vid("lam12y", k),
            vid("u", k),
        ]
        cliques.append(_dedupe_keep_order(clique))

    # Terminal kinematics / terminal objective clique.
    terminal_clique = [
        vid("c1", N - 1),
        vid("s1", N - 1),
        vid("c2", N - 1),
        vid("s2", N - 1),
        vid("a1", N - 1),
        vid("b1", N - 1),
        vid("a2", N - 1),
        vid("b2", N - 1),
        vid("c1", N),
        vid("s1", N),
        vid("c2", N),
        vid("s2", N),
    ]
    cliques.append(_dedupe_keep_order(terminal_clique))

    unique_cliques = []
    seen = set()

    for clique in cliques:
        key = tuple(clique)
        if key not in seen:
            unique_cliques.append(clique)
            seen.add(key)

    sizes = [len(c) for c in unique_cliques]

    print("\n" + "=" * 80)
    print("SELF CLIQUE DEBUG: reduced Acrobot cliques")
    print("=" * 80)
    print(f"number of SELF cliques: {len(unique_cliques)}")

    if sizes:
        print(f"min clique size:      {min(sizes)}")
        print(f"max clique size:      {max(sizes)}")
        print(f"mean clique size:     {np.mean(sizes):.2f}")
        print(f"largest clique sizes: {sorted(sizes)[-20:]}")
    else:
        print("No cliques created.")

    print("=" * 80 + "\n")

    return unique_cliques