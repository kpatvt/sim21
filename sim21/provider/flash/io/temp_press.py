from ..basic import basic_flash_temp_press_2phase
from ..rachford_rice import possible_rr_2phase, solve_rr_2phase, dew_point_rr_2phase, bubble_point_rr_2phase
from numba import njit
import numpy as np


def flash_temp_press_2phase(provider, temp, press, feed_comp, valid=None, previous=None):
    # There is nothing special about the 2 phase flash since there is not really much to be gained
    # from the more sophisticated formulation. Just use the basic flash here.
    previous_k_values = None
    if previous is not None:
        try:
            previous_k_values = previous.k_values_vle
        except KeyError:
            previous_k_values = None

    return basic_flash_temp_press_2phase(provider, temp, press, feed_comp, valid, previous_k_values)


@njit(cache=True)
def calc_log_kb(w, log_k_values, valid):
    log_kb = 0
    for i in valid:
        log_kb += w[i]*log_k_values[i]

    return log_kb


@njit(cache=True)
def calc_u(log_k_values, log_kb, valid):
    u = np.zeros(len(log_k_values))
    for i in valid:
        u[i] = log_k_values[i] - log_kb

    return u


