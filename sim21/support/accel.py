import math
import sys

import numpy as np
from numba import njit

MIN_DENOM = math.sqrt(sys.float_info.epsilon)


@njit(cache=True)
def gdem(x_2, f_x_2, x_1, f_x_1, x_0, f_x_0):
    # x_0 is most recent value
    # x_1 is one iterations back
    # x_2 is two iterations back

    delta_x_0 = f_x_0 - x_0
    delta_x_1 = f_x_1 - x_1
    delta_x_2 = f_x_2 - x_2

    b01 = np.dot(delta_x_0, delta_x_1)
    b11 = np.dot(delta_x_1, delta_x_1)
    b02 = np.dot(delta_x_0, delta_x_2)
    b12 = np.dot(delta_x_1, delta_x_2)
    b22 = np.dot(delta_x_2, delta_x_2)

    den = (b11 * b22 - b12 * b12)

    if abs(den) > MIN_DENOM:
        u1 = (b02 * b12 - b01 * b22) / den
        u2 = (b01 * b12 - b02 * b11) / den
        den = (1.0 + u1 + u2)
        x_inf = x_0 + (delta_x_0 - u2 * delta_x_1) / den
    else:
        x_inf = f_x_0

    return x_inf


@njit(cache=True)
def update_inverse_jacobian(previous_inv_jac, dx, df, threshold=0, modify_in_place=True):
    """
    Use Broyden method (following Numerical Recipes in C, 9.7) to update inverse Jacobian
    current_inv_jac is previous inverse Jacobian (n x n)
    dx is delta x for last step (n)
    df is delta errors for last step (n)
    """
    dot_dx_inv_j = np.dot(dx, previous_inv_jac)
    denom = np.dot(dot_dx_inv_j, df)
    if abs(threshold) <= 0:
        threshold = MIN_DENOM

    if abs(denom) < threshold:
        return previous_inv_jac, False

    if modify_in_place:
        previous_inv_jac += np.outer((dx - np.dot(previous_inv_jac, df)), dot_dx_inv_j) / denom
        result = previous_inv_jac
    else:
        result = previous_inv_jac + np.outer((dx - np.dot(previous_inv_jac, df)), dot_dx_inv_j) / denom

    return result, True
