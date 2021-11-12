import math
import sys

import numpy as np
from numba import njit

__RR_INTERNAL_TOLERANCE = math.sqrt(sys.float_info.epsilon)
__RR_ITERATIONS = 32


@njit(cache=True)
def possible_rr_2phase(k_values, feed_comp, valid_comps):
    vapor_present = False
    liquid_present = False
    g_0 = 0
    g_1 = 0
    for i in valid_comps:
        g_0 += feed_comp[i] * (k_values[i] - 1.0)
        g_1 += feed_comp[i] / k_values[i]

    g_1 = 1 - g_1
    if g_0 <= 0:
        liquid_present = True
    elif g_1 >= 0:
        vapor_present = True
    else:
        vapor_present = True
        liquid_present = True

    return vapor_present, liquid_present


@njit(cache=True)
def dew_point_rr_2phase(k_values, feed_comp, valid_comps):
    x, y = np.zeros(len(k_values)), np.zeros(len(k_values))
    for i in valid_comps:
        y[i] = feed_comp[i]
        x[i] = y[i]/k_values[i]

    return x, y


@njit(cache=True)
def bubble_point_rr_2phase(k_values, feed_comp, valid_comps):
    x, y = np.zeros(len(k_values)), np.zeros(len(k_values))
    for i in valid_comps:
        x[i] = feed_comp[i]
        y[i] = k_values[i]*x[i]

    return x, y


@njit(cache=True)
def solve_rr_2phase(k_values, z, valid_comps, initial_beta=-1.0):
    # Find the interval for the likely solution
    k_values_min = np.min(k_values)
    k_values_max = np.max(k_values)
    beta_min = 1/(1 - k_values_max)
    beta_max = 1/(1 - k_values_min)
    x, y = np.zeros(len(k_values)), np.zeros(len(k_values))

    beta_min, beta_max = min(beta_min, beta_max), max(beta_min, beta_max)
    # If the initial_beta is inside this interval, use it as the guess
    # Otherwise halfway in the interval
    if beta_min < initial_beta < beta_max:
        beta_guess = initial_beta
    else:
        beta_guess = (beta_min + beta_max)/2

    # Following Michelsen, do a Newton search for the beta with a given form when beta > 0.5
    # Hard coded to 32 iterations, more than enough.

    f = 1 / __RR_INTERNAL_TOLERANCE
    converged = False

    for iter_count in range(__RR_ITERATIONS):
        f, f_prime = 0, 0
        for k in valid_comps:
            f += z[k] * (k_values[k] - 1.0) / (1.0 - beta_guess + beta_guess * k_values[k])
            f_prime += -1*(z[k] * ((k_values[k] - 1.0) ** 2) / ((1.0 - beta_guess + beta_guess * k_values[k]) ** 2))

        if abs(f) < __RR_INTERNAL_TOLERANCE and iter_count > 1:
            converged = True
            break

        delta = f/f_prime
        beta_guess_new = beta_guess - delta

        if beta_guess_new > beta_max:
            beta_guess_new = (beta_guess + beta_max)/2
        elif beta_guess_new < beta_min:
            beta_guess_new = (beta_guess + beta_min)/2

        beta_guess = beta_guess_new

    # We take the absolute value and normalize the sum of the components, just in case...
    # Can happen occasionally and cascade into the thermo routines, we can cause additional problems
    # if we don't normalize them
    x_sum, y_sum = 0, 0
    for k in valid_comps:
        den = 1 - beta_guess + beta_guess * k_values[k]
        x[k] = abs((1 - beta_guess)*z[k]/den)
        x_sum += x[k]
        y[k] = abs(beta_guess * k_values[k] * z[k] / den)
        y_sum += y[k]

    for k in valid_comps:
        x[k] /= x_sum
        y[k] /= y_sum

    return converged, beta_guess, x, y

