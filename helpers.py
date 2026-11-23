# ECSE 343 group project (Fall 2026) -- helpers.py
#
# Problem 3: training data and the linear regression solvers.  Fill in the
# TODO functions, keep names and return values.  Rename to group_X_helpers.py
# when you hand it in and fix the import line below accordingly.

import pickle
import numpy as np
from circuit import dc_sweep, measure, house_lstsq, RC_RANGE, RE_RANGE

NOISE_STD = 0.02      # same noise level as measurements.csv (V and mA)


def create_dataset(N, noise_std=NOISE_STD, seed=0, path=None):
    """TODO 3(a). N simulated boards.

    Return a dict with
        'Y'        N x 51 x 4, noisy sweeps (same columns as measure)
        'Y_clean'  N x 51 x 4, without noise
        'theta'    N x 2, columns ln R_C and ln R_E
    and pickle it to `path` if a path is given.
    """
    rng = np.random.default_rng(seed)
    raise NotImplementedError("TODO")
    if path is not None:
        with open(path, "wb") as fh:
            pickle.dump(data, fh)
    return data


def load_dataset(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def features(Y, cols=(0, 1, 2, 3)):
    # flatten the chosen columns of each sweep into one row (works for a single sweep too)
    Y = np.asarray(Y)
    if Y.ndim == 2:
        Y = Y[None]
    return Y[:, :, list(cols)].reshape(len(Y), -1)


def split(n, frac_train=0.8, seed=0):
    # random train / test indices
    idx = np.random.default_rng(seed).permutation(n)
    k = int(round(frac_train * n))
    return idx[:k], idx[k:]


def design_matrix(F, mu, sd):
    # standardize with the training mean/std and put the intercept column in front
    return np.column_stack([np.ones(len(F)), (F - mu) / sd])


def rms_log_error(theta_pred, theta_true):
    """TODO 3(b). RMS of (pred - true) for each column, returns shape (2,)."""
    raise NotImplementedError("TODO")


def fit_linear_qr(A, T):
    """TODO 3(b). Weights W (p x 2) minimizing ||A W - T||_F, via house_lstsq
    on each column of T."""
    raise NotImplementedError("TODO")


def fit_linear_gd(A, T, n_iter=5000, record=(100, 1000, 5000)):
    """TODO 3(b). Same weights by gradient descent, fixed step 1/L as in the
    handout.  Start from W = 0 except the intercept row = mean of T.

    Return (W, snaps) where snaps[k] is a copy of W after k iterations, for
    each k in `record`.
    """
    raise NotImplementedError("TODO")


def fit_linear_gs(A, T, n_iter=5000, record=(100, 1000, 5000)):
    """TODO 3(b). Same weights by Gauss-Seidel on (A^T A / n) W = A^T T / n,
    one sweep over all rows of W per iteration, both columns of T at once.
    Same start and same return values as fit_linear_gd.
    """
    raise NotImplementedError("TODO")
