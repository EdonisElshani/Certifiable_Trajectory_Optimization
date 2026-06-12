def get_init_constraints(c1_0, s1_0, c2_0, s2_0,
                         a1_0, b1_0, a2_0, b2_0,
                         params):
    """
    Initial constraints for the reduced SO(2) Acrobot.

    No x variables are present. The initial COM positions are implied by
    R_0 and the fixed thesis geometry:
        x1 = p0 + [l1/2*s1, -l1/2*c1]
        x2 = p0 + [l1*s1 + l2/2*s2, -l1*c1 - l2/2*c2].

    A rest start is imposed by F_0 = I, i.e. a_i,0=1 and b_i,0=0.
    """
    eqs = [
        # Absolute link orientations at k=0
        c1_0 - np.cos(params['thetaR1_0']),
        s1_0 - np.sin(params['thetaR1_0']),
        c2_0 - np.cos(params['thetaR2_0']),
        s2_0 - np.sin(params['thetaR2_0']),

        # Initial relative rotations F_0. F_0=I gives zero initial angular velocity.
        a1_0 - np.cos(params['thetaF1_0']),
        b1_0 - np.sin(params['thetaF1_0']),
        a2_0 - np.cos(params['thetaF2_0']),
        b2_0 - np.sin(params['thetaF2_0']),

        # SO(2) constraints at k=0
        c1_0**2 + s1_0**2 - 1,
        c2_0**2 + s2_0**2 - 1,
        a1_0**2 + b1_0**2 - 1,
        a2_0**2 + b2_0**2 - 1,
    ]
    eq_mask = [1] * 8 + [0] * 2 + [1] * 2
    return eqs, [], eq_mask

def get_initial_rest_position_constraints(*args, **kwargs):
    """
    Deprecated in the reduced no-x/no-v formulation.
    Rest start is imposed by F_0 = I in get_init_constraints, i.e. b_{i,0}=0.
    """
    return [], [], []

def get_translational_dynamics_link1(c1_km1, s1_km1, b1_km1,
                                     c1_k, s1_k, b1_k,
                                     lam0x_k, lam0y_k,
                                     lam12x_k, lam12y_k,
                                     params):
    """
    Reduced x-free/v-free translational DEL for link 1, scaled by h^2.

    Thesis sign convention:
        R = [[c, -s], [s, c]],  x = l sin(theta),  y = -l cos(theta).
    Small-step velocity approximation on interval k:
        theta_dot_{i,k} ~= b_{i,k}/h.

    Link-1 COM velocity approximation:
        v1_k = (l1/(2h)) [c1_k*b1_k, s1_k*b1_k]^T.

    Constraint implemented:
        (m1*l1/2) ([c*b, s*b]_k - [c*b, s*b]_{k-1})
        + h^2 m1 g e2 - h^2 s_lambda (lambda0_k + lambda12_k) = 0.
    """
    h = params["dt"]
    m1 = params["m1"]
    l1 = params["l1"]
    g = params["g"]
    lam_scale = params["lambda_scale"]

    eqs = [
        (m1 * l1 / 2.0) * (c1_k * b1_k - c1_km1 * b1_km1)
        - h**2 * lam_scale * lam0x_k
        - h**2 * lam_scale * lam12x_k,

        (m1 * l1 / 2.0) * (s1_k * b1_k - s1_km1 * b1_km1)
        + h**2 * m1 * g
        - h**2 * lam_scale * lam0y_k
        - h**2 * lam_scale * lam12y_k,
    ]
    return eqs, [], [1, 1]

def get_translational_dynamics_link2(c1_km1, s1_km1, b1_km1,
                                     c1_k, s1_k, b1_k,
                                     c2_km1, s2_km1, b2_km1,
                                     c2_k, s2_k, b2_k,
                                     lam12x_k, lam12y_k,
                                     params):
    """Reduced x-free/v-free translational DEL for link 2, scaled by h^2."""
    h = params["dt"]
    m2 = params["m2"]
    l1 = params["l1"]
    l2 = params["l2"]
    g = params["g"]
    lam_scale = params["lambda_scale"]

    link1_x = c1_k * b1_k - c1_km1 * b1_km1
    link1_y = s1_k * b1_k - s1_km1 * b1_km1
    link2_x = c2_k * b2_k - c2_km1 * b2_km1
    link2_y = s2_k * b2_k - s2_km1 * b2_km1

    eqs = [
        m2 * (l1 * link1_x + (l2 / 2.0) * link2_x)
        + h**2 * lam_scale * lam12x_k,

        m2 * (l1 * link1_y + (l2 / 2.0) * link2_y)
        + h**2 * m2 * g
        + h**2 * lam_scale * lam12y_k,
    ]
    return eqs, [], [1, 1]

def reconstruct_positions_from_cs(c1_k, s1_k, c2_k, s2_k, params):
    """
    Reconstruct COM positions from rotations, with the thesis convention
        x = l sin(theta), y = -l cos(theta).
    Works for both numeric scalars and NumPoly expressions.
    """
    p0 = params["p_0"]
    l1 = params["l1"]
    l2 = params["l2"]

    x1 = np.array([
        p0[0] + (l1 / 2.0) * s1_k,
        p0[1] - (l1 / 2.0) * c1_k,
    ], dtype=object)

    x2 = np.array([
        p0[0] + l1 * s1_k + (l2 / 2.0) * s2_k,
        p0[1] - l1 * c1_k - (l2 / 2.0) * c2_k,
    ], dtype=object)
    return x1, x2


def get_holonomic_constraint(*args, **kwargs):
    """
    In the reduced model there are no holonomic constraints left to add:
    x1 and x2 have been substituted by reconstruct_positions_from_cs.
    Returning no constraints intentionally avoids adding decorative 0=0 equations.
    """
    return [], [], []

def get_rotational_kinematics_link1(c1_km1, s1_km1,
                                    c1_k, s1_k,
                                    a1_km1, b1_km1,
                                    params):
    eqs = [
        c1_k - c1_km1 * a1_km1 + s1_km1 * b1_km1,
        s1_k - s1_km1 * a1_km1 - c1_km1 * b1_km1,
    ]

    ineqs = []
    eq_mask = [1, 1]

    return eqs, ineqs, eq_mask


def get_rotational_kinematics_link2(c2_km1, s2_km1,
                                    c2_k, s2_k,
                                    a2_km1, b2_km1,
                                    params):
    eqs = [
        c2_k - c2_km1 * a2_km1 + s2_km1 * b2_km1,
        s2_k - s2_km1 * a2_km1 - c2_km1 * b2_km1,
    ]

    ineqs = []
    eq_mask = [1, 1]

    return eqs, ineqs, eq_mask


def get_SO2_orthogonality_constraint_rotation_R(c1_k, s1_k, c2_k, s2_k, params):
    eqs = [
        c1_k**2 + s1_k**2 - 1,
        c2_k**2 + s2_k**2 - 1,
    ]

    ineqs = []
    eq_mask = [0, 0]

    return eqs, ineqs, eq_mask

def get_SO2_orthogonality_constraint_rotation_F(a1_k, b1_k, a2_k, b2_k, params):
    eqs = [
        a1_k**2 + b1_k**2 - 1,
        a2_k**2 + b2_k**2 - 1,
    ]

    ineqs = []
    eq_mask = [1, 1]

    return eqs, ineqs, eq_mask

def get_step_angle_bound_constraint_link_1(a1_k, params):

    a1_min = params["a1_min"]

    ineqs = [
        a1_k - a1_min,
    ]

    eqs = []
    eq_mask = []

    return eqs, ineqs, eq_mask

def get_step_angle_bound_constraint_link_2(a2_k, params):

    a2_min = params["a2_min"]

    ineqs = [
        a2_k - a2_min,
    ]

    eqs = []
    eq_mask = []

    return eqs, ineqs, eq_mask

def get_rotational_dynamics_link1(b1_km1, b1_k,
                                  c1_k, s1_k,
                                  lam0x_k, lam0y_k,
                                  lam12x_k, lam12y_k,
                                  u_k,
                                  params):

    h = params["dt"]

    delta_11 = params["delta_11"]
    delta_12 = params["delta_12"]

    rho_10 = params["rho_10"]
    rho_112 = params["rho_112"]

    rho_10_x = rho_10[0]
    rho_10_y = rho_10[1]

    rho_112_x = rho_112[0]
    rho_112_y = rho_112[1]

    # R_1,k^T lambda_0,k
    # R^T = [[c, s], [-s, c]]
    Rt_lam0_x = c1_k * lam0x_k * params["lambda_scale"] + s1_k * lam0y_k * params["lambda_scale"]
    Rt_lam0_y = -s1_k * lam0x_k * params["lambda_scale"] + c1_k * lam0y_k * params["lambda_scale"]

    # R_1,k^T lambda_12,k
    Rt_lam12_x = c1_k * lam12x_k * params["lambda_scale"] + s1_k * lam12y_k * params["lambda_scale"]
    Rt_lam12_y = -s1_k * lam12x_k * params["lambda_scale"] + c1_k * lam12y_k * params["lambda_scale"]

    # Planar cross product rho x R^T lambda
    mu_10 = rho_10_x * Rt_lam0_y - rho_10_y * Rt_lam0_x
    mu_112 = rho_112_x * Rt_lam12_y - rho_112_y * Rt_lam12_x

    eqs = [
        (delta_11 + delta_12) * (b1_km1 - b1_k)
        + h**2 * (mu_10 + mu_112 - u_k)
    ]

    ineqs = []

    # This is bilinear because of c*lambda and s*lambda terms.
    # Use [1] if you want to follow Kang's kinematic/dynamics mask style.
    eq_mask = [1]

    return eqs, ineqs, eq_mask


def get_rotational_dynamics_link2(b2_km1, b2_k,
                                  c2_k, s2_k,
                                  lam12x_k, lam12y_k,
                                  u_k,
                                  params):

    h = params["dt"]

    delta_21 = params["delta_21"]
    delta_22 = params["delta_22"]

    rho_212 = params["rho_212"]

    rho_212_x = rho_212[0]
    rho_212_y = rho_212[1]

    # R_2,k^T lambda_12,k
    # R^T = [[c, s], [-s, c]]
    Rt_lam12_x = c2_k * lam12x_k * params["lambda_scale"] + s2_k * lam12y_k * params["lambda_scale"]
    Rt_lam12_y = -s2_k * lam12x_k * params["lambda_scale"] + c2_k * lam12y_k * params["lambda_scale"]

    # Planar cross product rho x R^T lambda
    mu_212 = rho_212_x * Rt_lam12_y - rho_212_y * Rt_lam12_x

    eqs = [
        (delta_21 + delta_22) * (b2_km1 - b2_k)
        + h**2 * (u_k - mu_212)
    ]

    ineqs = []

    # This is bilinear because of c*lambda and s*lambda terms.
    # Use [1] if following Kang's kinematic/dynamics mask style.
    eq_mask = [1]

    return eqs, ineqs, eq_mask


def get_control_bounds(u_k, params):

    u_max = params["u_max"]

    ineqs = [
        u_max**2 - u_k**2
    ]

    eqs = []
    eq_mask = []

    return eqs, ineqs, eq_mask


def get_lambda_bounds(lam0x_k, lam0y_k, lam12x_k, lam12y_k, params):
    L = params["lambda_max"]

    ineqs = [
        L**2 - lam0x_k**2,
        L**2 - lam0y_k**2,
        L**2 - lam12x_k**2,
        L**2 - lam12y_k**2,
    ]

    return [], ineqs, []