import numpy as np
from numba import njit
from ..basic import generate_2phase_results, generate_2phase_estimates
from .temp_press import  calc_log_kb, calc_u
from .settings import INSIDE_OUT_DAMPING, INSIDE_OUT_INNER_ITERATIONS, INSIDE_OUT_OUTER_ITERATIONS, \
    INSIDE_OUT_INNER_TOLERANCE, INSIDE_OUT_OUTER_TOLERANCE
from math import exp, log

from ...agg import AggregateByMole
from ...error import FlashConvergenceError


@njit(cache=True)
def initial_press_vap_frac_weighting_coeffs(log_k_values, vap_comp, vap_frac, valid):
    t = np.zeros(len(log_k_values))
    w = np.zeros(len(log_k_values))
    t_sum = 0
    for i in valid:
        t[i] = vap_comp[i]/(1 - vap_frac + vap_frac*exp(log_k_values[i]))
        t_sum += t[i]

    for i in valid:
        w[i] = t[i]/t_sum

    return w


def update_model_press_vap_frac_2phase(provider,
                                       temp, temp_star,
                                       press, vap_frac,
                                       liq_comp, vap_comp, valid,
                                       previous_w, previous_u,
                                       previous_a, previous_b,
                                       previous_log_kb, previous_temp,
                                       full_update=False):

    inv_temp_star = 1/temp_star
    # There is alteast one call to the provider to generate a liq and vapor phase
    # for the given condition
    log_k_values, liq, vap = generate_2phase_results(provider, temp, press, liq_comp, vap_comp, valid)

    # If there is a None in any of these terms, then we will generate the full model
    if previous_w is None or previous_u is None or previous_a is None or previous_b is None or \
        previous_log_kb is None or previous_temp is None or full_update:
        # Save the initial values
        log_k_values_1, liq_1, vap_1, temp_1 = log_k_values, liq, vap, temp

        # Perturb by some amount; by atleast 1 K or smaller depending on valid of temp
        delta_temp = 1.0
        temp_2 = temp_1 + delta_temp

        # Generate the peturbed real cofficients
        log_k_values_2, liq_2, vap_2 = generate_2phase_results(provider, temp_2, press, liq_comp, vap_comp, valid)

        # Generate the weighting factors only once
        if previous_w is None:
            w = initial_press_vap_frac_weighting_coeffs(log_k_values, vap_comp, vap_frac, valid)
        else:
            w = previous_w

        log_kb_hat_1 = calc_log_kb(w, log_k_values_1, valid)
        log_kb_hat_2 = calc_log_kb(w, log_k_values_2, valid)

        inv_temp_1 = 1/temp_1
        inv_temp_2 = 1/temp_2
        b_hat = (log_kb_hat_2 - log_kb_hat_1)/(inv_temp_2 - inv_temp_1)
        a_hat = log_kb_hat_1 - b_hat*(inv_temp_1 - inv_temp_star)
        log_kb_hat = log_kb_hat_1
    # Otherwise just do a simple update
    else:
        inv_temp = 1/temp
        w = previous_w
        log_kb_hat = calc_log_kb(w, log_k_values, valid)
        b_hat = previous_b
        a_hat = log_kb_hat - b_hat*(inv_temp - inv_temp_star)

    # Now update u_hat
    u_hat = calc_u(log_k_values, log_kb_hat, valid)

    # Return the updated model
    return w, u_hat, a_hat, b_hat, log_kb_hat, liq, vap


@njit(cache=True)
def solve_model_press_vap_frac_2phase(vap_frac, feed_comp, kb_0, u, r_guess, valid):
    # Should speed this up via jiting - can be very fast
    liq_comp_new = np.zeros(len(feed_comp))
    vap_comp_new = np.zeros(len(feed_comp))
    p = np.zeros(len(feed_comp))
    p_sum = 0
    liq_frac_spec = 1 - vap_frac
    r_value = r_guess
    sum_exp_u_p = 0
    closest_f, closest_r_value = 100, 100
    if vap_frac == 0 or vap_frac == 1:
        r_value = vap_frac
        for i in valid:
            p[i] = feed_comp[i] / (1.0 - r_value + kb_0 * r_value * exp(u[i]))
            p_sum += p[i]
            sum_exp_u_p += exp(u[i]) * p[i]

        inner_converged = True
    else:
        inner_converged = False
        f = 1
        for inner_iterations in range(INSIDE_OUT_INNER_ITERATIONS):
            if abs(f) < INSIDE_OUT_INNER_TOLERANCE:
                inner_converged = True
                break

            p_sum = 0
            sum_exp_u_p = 0
            for i in valid:
                p[i] =  feed_comp[i] / (1.0 - r_value + kb_0 * r_value * exp(u[i]))
                p_sum += p[i]
                sum_exp_u_p += exp(u[i]) * p[i]

            liq_frac_calc = (1 - r_value) * p_sum
            f = (liq_frac_calc - liq_frac_spec)**2
            if f < closest_f:
                closest_f = f
                closest_r_value = r_value

            r_value_delta = r_value + INSIDE_OUT_INNER_TOLERANCE/100
            p_sum_delta = 0
            for i in valid:
                p_sum_delta += feed_comp[i] / (1.0 - r_value + kb_0 * r_value * exp(u[i]))

            liq_frac_calc_delta = (1 - r_value_delta) * p_sum_delta
            f_delta = (liq_frac_calc_delta - liq_frac_spec)**2

            df_dx = (f_delta - f)/(r_value_delta - r_value)
            if abs(df_dx) < INSIDE_OUT_INNER_TOLERANCE:
                inner_converged = False
                break

            r_value_new = r_value - f/df_dx
            if abs(r_value_new - r_value) < INSIDE_OUT_INNER_TOLERANCE:
                inner_converged = True
                break

            if r_value_new < 0:
                r_value_new = 0.9*r_value
            elif r_value_new > 1.0:
                r_value_new = min(1.0, 1.1*r_value)

            r_value = r_value_new

    if not inner_converged:
        r_value = closest_r_value
        p_sum = 0
        sum_exp_u_p = 0
        for i in valid:
            p[i] = feed_comp[i] / (1.0 - r_value + kb_0 * r_value * exp(u[i]))
            p_sum += p[i]
            sum_exp_u_p += exp(u[i]) * p[i]

    kb_calc = p_sum/sum_exp_u_p
    for i in valid:
        liq_comp_new[i] = p[i]/p_sum
        vap_comp_new[i] = (exp(u[i]) * p[i])/sum_exp_u_p

    return inner_converged, p, p_sum, r_value, kb_calc, liq_comp_new, vap_comp_new


@njit(cache=True)
def calc_error(u, u_hat, a, a_hat, b, b_hat, valid):
    # Also can be jitted out
    result = 0
    for i in valid:
        result += abs((u_hat[i] - u[i]))

    result += abs((a_hat - a)/a)
    result += abs((b_hat - b)/b)
    return result


def flash_press_vap_frac_2phase(provider, press, vap_frac, feed_comp, valid=None, previous=None):
    if valid is None:
        valid = provider.all_valid_components

    # temp_star is a reference value
    temp_star = 298.15

    # Just a dummy value
    if previous is None:
        temp_est = provider.guess_nbp(feed_comp, valid)
        k_values, liq_comp, vap_comp, _ = generate_2phase_estimates(provider, temp_est, press, feed_comp, valid)
        # Correction to make sure that temp_star is not close to temp_est
        # can occasionally cause a problem
        if abs(temp_est - temp_star)/temp_est < 0.05:
            temp_star = 0.95*temp_est
    else:
        raise NotImplementedError

    # Initialize the values of the search
    temp = temp_est
    r_value = vap_frac
    inner_converged, outer_converged = False, False
    w, u, a, b, log_kb, liq, vap = update_model_press_vap_frac_2phase(provider, temp, temp_star,
                                                                      press, vap_frac,
                                                                      liq_comp, vap_comp, valid,
                                                                      None, None, None, None,
                                                                      None, None)

    force_full_update = False
    error = 0
    kb_0 = exp(log_kb)

    minimum_full_update_steps = 10
    pre_accelerate = 3
    accelerate = True
    history = []

    for outer_iterations in range(INSIDE_OUT_OUTER_ITERATIONS):
        # Check if the actual error is less than tolerance, if so we quit
        if outer_iterations > 1 and error < INSIDE_OUT_OUTER_TOLERANCE:
            outer_converged = True
            break

        # If we have less than 5 iterations, we do a full update
        # This helps minimize the number of iterations
        if outer_iterations < 5:
            force_full_update = True

        # Solve the inner simplified model
        try:
            inner_converged, \
            p, p_sum, \
            r_value, \
            kb_calc, \
            liq_comp_new, vap_comp_new = solve_model_press_vap_frac_2phase(vap_frac,
                                                                           feed_comp,
                                                                           kb_0, u,
                                                                           r_value, valid)
        except ZeroDivisionError:
            print('Zero Divide Error')
            pass

        # print('PV#', outer_iterations, 'temp:', temp, 'vap_frac:', vap_frac, 'error:', error, 'inner_converged:', inner_converged)

        if kb_calc > 0:
            # Calculate the new temperature
            log_kb_calc = log(kb_calc)
            temp_calc_inv = 1/temp_star + (log_kb_calc - a)/b
            temp_calc = 1/temp_calc_inv
        else:
            # if we get a negative kb_calc
            # force a full update
            temp_calc = 10*temp

        liq_comp = liq_comp_new
        vap_comp = vap_comp_new

        # if the temp_calc is negative or a sudden increase temp,
        # damp the update
        if temp_calc < 0:
            temp_calc = (1 - INSIDE_OUT_DAMPING)*temp
            force_full_update = True
        elif temp_calc > 4.0*temp:
            temp_calc = (1 + INSIDE_OUT_DAMPING)*temp
            force_full_update = True
        else:
            force_full_update = False
            liq_comp = liq_comp_new
            vap_comp = vap_comp_new

        if not inner_converged:
            # force_full_update = True
            pass

        # Get the update the model factors
        w_hat, u_hat, \
        a_hat, b_hat, \
        log_kb_hat, \
        liq_hat, vap_hat = update_model_press_vap_frac_2phase(provider, temp, temp_star,
                                                              press, vap_frac,
                                                              liq_comp, vap_comp, valid,
                                                              w,
                                                              u,
                                                              a,
                                                              b,
                                                              kb_0,
                                                              temp,
                                                              full_update=force_full_update)

        # Save the actual phases generated
        liq = liq_hat
        vap = vap_hat

        previous_error = error
        error = calc_error(u, u_hat, a, a_hat, b, b_hat, valid)

        if outer_iterations > 10:
            # a, b
            # x = np.zeros(len(u) + 2)
            # x[:len(u)] = u
            # x[len(u):] = a, b
            #
            # f_x = np.zeros(len(u_hat) + 2)
            # f_x[:len(u_hat)] = u_hat
            # f_x[len(u_hat):] = a_hat, b_hat
            #
            # history.append(x)
            # history.append(f_x)
            #
            # if len(history) == 6 and accelerate:
            #     # x_2, f_x_2, x_1, f_x_1, x_0, f_x_0 = history
            #     # x_inf = gdem(x_2, f_x_2, x_1, f_x_1, x_0, f_x_0)
            #     x_inf = (x + f_x)*0.5
            #     # print('Accelerating')
            #     u_hat = x_inf[:len(u_hat)]
            #     a_hat, b_hat = x_inf[len(u_hat):]
            #     history = []
            temp_calc = (temp + temp_calc)*0.5
            u_hat = (u + u_hat)*0.5
            a_hat = (a + a_hat)*0.5
            b_hat = (b + b_hat)*0.5

        temp = temp_calc
        u = u_hat
        a = a_hat
        b = b_hat
        # log_kb = log_kb_calc

    if not outer_converged:
        raise FlashConvergenceError

    # Otherwise we can return the solution
    return AggregateByMole(provider, [liq, vap], [1 - vap_frac, vap_frac])
