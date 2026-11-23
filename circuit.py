# ECSE 343 group project (Fall 2026) -- circuit.py
#
# Problems 1 and 2: DC solution of the transistor stage, sensitivities,
# Gauss-Newton fit.  Fill in the functions marked TODO.  Keep the function
# names and the return values as they are, the grading script imports this
# file.  Rename it to group_X_circuit.py when you hand it in.
#
# Conventions used everywhere:
#   x = [V_B, V_C, V_E]  node voltages in V
#   Y = measure(X, R_C)  columns V_B, V_C, V_E (V) and I_CC (mA)

import numpy as np

VCC = 9.0
R_B = 10e3
I_S = 1e-14
BETA_F = 100.0
BETA_R = 2.0
V_T = 0.025

VIN = np.round(np.arange(0.0, 5.0 + 1e-9, 0.1), 10)   # 0, 0.1, ..., 5.0
RC_RANGE = (500.0, 10e3)
RE_RANGE = (100.0, 2e3)


# --- given --------------------------------------------------------------

def bjt_currents(x):
    # Ebers-Moll, returns (I_B, I_C, I_E) in amperes
    VB, VC, VE = x
    ebe = np.exp((VB - VE) / V_T)
    ebc = np.exp((VB - VC) / V_T)
    IC = I_S * (ebe - ebc) - I_S / BETA_R * (ebc - 1.0)
    IB = I_S / BETA_F * (ebe - 1.0) + I_S / BETA_R * (ebc - 1.0)
    return IB, IC, IC + IB


def f_nonlin(x):
    IB, IC, IE = bjt_currents(x)
    return np.array([IB, IC, -IE])


def get_G(R_C, R_E):
    return np.diag([1.0 / R_B, 1.0 / R_C, 1.0 / R_E])


def get_b(v_in, R_C):
    return np.array([v_in / R_B, VCC / R_C, 0.0])


def residual(x, v_in, R_C, R_E):
    # F(x) = G x + f(x) - b, in amperes
    return get_G(R_C, R_E) @ x + f_nonlin(x) - get_b(v_in, R_C)


def measure(X, R_C):
    # X is n x 3 (one row per sweep point); adds the I_CC column in mA
    return np.column_stack([X, 1e3 * (VCC - X[:, 1]) / R_C])


def load_measurements(path="measurements.csv"):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1:5]      # v_in, Y


def limit_step(dx, step_limit=0.2):
    # scale dx so that its largest entry is at most step_limit (None = off)
    if step_limit is None:
        return dx
    m = np.max(np.abs(dx))
    if m > step_limit:
        dx = dx * (step_limit / m)
    return dx


# --- problem 1 ----------------------------------------------------------

def get_jac(x, R_C, R_E):
    """TODO 1(a). Jacobian of F at x, 3x3."""
    VB, VC, VE = x

    # g_BE = (I_S/V_T) e^(V_BE/V_T) and g_BC = (I_S/V_T) e^(V_BC/V_T)
    g_BE = I_S / V_T * np.exp((VB - VE) / V_T)
    g_BC = I_S / V_T * np.exp((VB - VC) / V_T)

    # each row is [d/dV_B, d/dV_C, d/dV_E], from differentiating eq. (1)
    dIB = np.array([g_BE / BETA_F + g_BC / BETA_R, -g_BC / BETA_R, -g_BE / BETA_F])
    dIC = np.array([g_BE - (1 + 1 / BETA_R) * g_BC, (1 + 1 / BETA_R) * g_BC, -g_BE])
    dIE = dIB + dIC   # I_E = I_C + I_B

    # J_F = G + df/dx 
    # f = [I_B, I_C, -I_E]
    return get_G(R_C, R_E) + np.array([dIB, dIC, -dIE])

def newton_raphson(x0, v_in, R_C, R_E, tol=1e-9, max_iter=50, step_limit=0.2,
                   return_history=False):
    """TODO 1(b). Newton-Raphson for F(x) = 0.

    Call limit_step on every step (step_limit=None turns it off).  Stop when
    ||dx||_2 < tol.  Raise RuntimeError if max_iter is reached or x is not
    finite any more.  Return (x, k) where k is the number of iterations, or
    (x, k, xs) with xs = [x0, x1, ..., xk] if return_history is True.
    """
    x = np.array(x0, dtype=float)
    xs = [x.copy()]

    for k in range(1, max_iter + 1):
        # Newton step: solve J_F(x_k) dx = -F(x_k)
        dx = np.linalg.solve(get_jac(x, R_C, R_E), -residual(x, v_in, R_C, R_E))

        # scale down if V > 0.2
        dx = limit_step(dx, step_limit)
        x = x + dx
        xs.append(x.copy())

        if not np.all(np.isfinite(x)):
            raise RuntimeError("Newton-Raphson: x is not finite any more")
        if np.linalg.norm(dx) < tol:
            if return_history:
                return x, k, xs
            return x, k

    raise RuntimeError("Newton-Raphson: max_iter reached")


def dc_sweep(R_C, R_E, v_in=VIN, step_limit=0.2):
    """TODO 1(b). Sweep over v_in with continuation, first point from [0, VCC, 0].

    Return X (n x 3) and an integer array with the Newton iterations per point.
    """
    X = np.zeros((len(v_in), 3))
    iters = np.zeros(len(v_in), dtype=int)

    x = np.array([0.0, VCC, 0.0])
    for i in range(len(v_in)):
        x, k = newton_raphson(x, v_in[i], R_C, R_E, step_limit=step_limit)
        X[i] = x
        iters[i] = k

    return X, iters


def broyden(x0, v_in, R_C, R_E, tol=1e-9, max_iter=50):
    """TODO 1(c). Broyden's method, B_0 = get_jac(x0), no step limit.

    Same stopping rule and errors as newton_raphson.  Return (x, k, xs).
    """
    x = np.array(x0, dtype=float)
    xs = [x.copy()]
    B = get_jac(x, R_C, R_E)   # B_0 = J_F(x0)
    F = residual(x, v_in, R_C, R_E)

    for k in range(1, max_iter + 1):
        dx = np.linalg.solve(B, -F)
        x = x + dx
        xs.append(x.copy())

        if not np.all(np.isfinite(x)):
            raise RuntimeError("Broyden: x is not finite any more")
        if np.linalg.norm(dx) < tol:
            return x, k, xs

        # B_k+1 = B_k + (dF - B_k dx) dx^T / (dx^T dx)
        F_new = residual(x, v_in, R_C, R_E)
        dF = F_new - F
        B = B + np.outer(dF - B @ dx, dx) / (dx @ dx)
        F = F_new

    raise RuntimeError("Broyden: max_iter reached")


# --- problem 2 ----------------------------------------------------------

def get_sensitivities(X, R_C, R_E):
    """TODO 2(a). dY/dR_C and dY/dR_E for Y = measure(X, R_C), both n x 4.

    Units are those of Y per ohm, so the I_CC column is in mA/ohm.
    """
    raise NotImplementedError("TODO")


def fit_residual(R_C, R_E, Y_meas, cols):
    """TODO 2(b). Simulate the sweep at (R_C, R_E) and return (r, J_r).

    r is (Y_meas - Y)[:, cols] stacked point by point into one vector, J_r the
    m x 2 matrix with columns R_C dY/dR_C and R_E dY/dR_E, same stacking.  If
    Newton fails inside the sweep just let the RuntimeError through.
    """
    raise NotImplementedError("TODO")


def house_lstsq(A, b):
    """TODO 2(b). min ||A beta - b||_2 by Householder QR (your PS4 code)."""
    raise NotImplementedError("TODO")


def gauss_newton(RC_guess, RE_guess, Y_meas, cols=(0, 1, 2, 3), max_iter=30, tol=1e-8):
    """TODO 2(b). Gauss-Newton in (ln R_C, ln R_E), rules as in the handout.

    cols picks the measured columns (0 V_B, 1 V_C, 2 V_E, 3 I_CC).
    Return a dict with keys
        'R_C', 'R_E'    final estimates
        'iters'         number of updates made
        'converged'     True / False
        'cost', 'r', 'J_r'   at the final estimates (cost = 0.5 ||r||^2)
        'path'          list of (R_C, R_E) visited, guess first
    """
    raise NotImplementedError("TODO")


def error_propagation(r, J_r):
    """TODO 2(c). Covariance and friends for a converged fit.

    Return a dict with 's', 'sigma_mle', 'cov' (2x2), 'u' (u_C, u_E), 'rho'
    and 'kappa', defined as in the handout.
    """
    raise NotImplementedError("TODO")
