from math import log, exp
from ..basic import generate_2phase_estimates, generate_2phase_results
from .temp_press import calc_log_kb, calc_u
# The formulation for the weighting coefficients is the same for temp_vap_frac flash as well
from .settings import INSIDE_OUT_DAMPING, INSIDE_OUT_OUTER_ITERATIONS, INSIDE_OUT_OUTER_TOLERANCE
# The inner loop is the same for temp_vap_frac flash and press_vap_frac flash
from .press_vap_frac import initial_press_vap_frac_weighting_coeffs, calc_error, solve_model_press_vap_frac_2phase
import numpy as np

from ...agg import AggregateByMole
from ....error import FlashConvergenceError
from ....num.accel import gdem


def update_model_temp_vap_frac_2phase(provider, temp, press, press_star, vap_frac,
                                      liq_comp, vap_comp, valid,
                                      previous_w, previous_u,
                                      previous_a, previous_b,
                                      previous_log_kb, previous_press,
                                      full_update=False):

    log_press_star = log(press_star)
    # There is alteast one call to the provider to generate a liq and vapor phase for the given condition
    log_k_values, liq, vap = generate_2phase_results(provider, temp, press, liq_comp, vap_comp, valid)

    if previous_w is None or previous_u is None or previous_a is None or previous_b is None or \
        previous_log_kb is None or previous_press is None or full_update:

        # Save the initial values
        log_k_values_1, liq_1, vap_1, press_1 = log_k_values, liq, vap, press

        # Perturb by some amount; by atleast 5%
        delta_press = 0.05*press_1
        press_2 = press_1 + delta_press

        # Generate the peturbed real cofficients
        log_k_values_2, liq_2, vap_2 = generate_2phase_results(provider, temp, press_2, liq_comp, vap_comp, valid)

        # Generate the weighting factors only once
        if previous_w is None:
            w = initial_press_vap_frac_weighting_coeffs(log_k_values, vap_comp, vap_frac, valid)
        else:
            w = previous_w

        log_kb_hat_1 = calc_log_kb(w, log_k_values_1, valid)
        log_kb_hat_2 = calc_log_kb(w, log_k_values_2, valid)

        kb_hat_1 = exp(log_kb_hat_1)
        kb_hat_2 = exp(log_kb_hat_2)

        b_hat = (log(kb_hat_1 * press_1) - log(kb_hat_2 * press_2)) / (log(press_1 / press_star) - log(press_2 / press_star))
        a_hat = log(kb_hat_1 * press_1) - b_hat * log(press_1 / press_star)
        log_kb_hat = log_kb_hat_1

    # Otherwise just do a simple update
    else:
        w = previous_w
        log_kb_hat = calc_log_kb(w, log_k_values, valid)
        kb_hat = exp(log_kb_hat)
        b_hat = previous_b
        a_hat = log(kb_hat * press) - b_hat * log(press / press_star)

    # Now update u_hat
    u_hat = calc_u(log_k_values, log_kb_hat, valid)

    # Return the updated model
    return w, u_hat, a_hat, b_hat, log_kb_hat, liq, vap


def flash_temp_vap_frac_2phase(provider, temp, vap_frac, feed_comp, valid=None, previous=None):
    if valid is None:
        valid = provider.all_valid_components

    # press_star is a reference value
    press_star = 1e5
    press_est = 101325.0
    if previous is None:
        k_values, liq_comp, vap_comp, _ = generate_2phase_estimates(provider, temp, press_est, feed_comp, valid)
        # Correction to make sure that temp_star is not close to temp_est
        # can occasionally cause a problem
        if abs(press_est - press_star)/press_est < 0.1:
            press_star = 0.90*press_est
    else:
        raise NotImplementedError

    # Initialize the values of the search
    press = press_est
    log_press_star = log(press_star)
    r_value = vap_frac
    inner_converged, outer_converged = False, False
    w, u, a, b, log_kb, liq, vap = update_model_temp_vap_frac_2phase(provider, temp, press, press_star, vap_frac,
                                                                    liq_comp, vap_comp, valid,
                                                                    None, None,
                                                                    None, None,
                                                                    None, None,
                                                                    full_update=False)

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

        # print('#', outer_iterations, 'press:', press, 'vap_frac:', vap_frac, 'error:', error)
        # If we have less than 5 iterations, we do a full update
        # This helps minimize the number of iterations
        force_full_update = False
        if outer_iterations < 5:
            force_full_update = True

        # Solve the inner simplified model
        inner_converged, \
        p, p_sum, \
        r_value, \
        kb_calc, \
        liq_comp_new, vap_comp_new = solve_model_press_vap_frac_2phase(vap_frac,
                                                                       feed_comp,
                                                                       kb_0, u,
                                                                       r_value, valid)

        # The kb_calc is valid, lets do an update
        if kb_calc > 0:
            log_kb_calc = log(kb_calc)
            log_press_calc = (log_kb_calc + b*log_press_star - a)/(b - 1)
            press_calc = exp(log_press_calc)
            vap_frac_calc = 1.0 - ((1.0 - r_value) * p_sum)
            liq_comp = liq_comp_new
            vap_comp = vap_comp_new
        else:
            press_calc = (1 + INSIDE_OUT_DAMPING)*press
            force_full_update = True

        # Get the update to the model factors
        w_hat, u_hat, \
        a_hat, b_hat, \
        log_kb_hat, \
        liq_hat, vap_hat = update_model_temp_vap_frac_2phase(provider,
                                                             temp, press_calc, press_star,
                                                             vap_frac, liq_comp, vap_comp, valid,
                                                             w, u, a, b, log_kb, press,
                                                             full_update=force_full_update)
        # Save the actual phases generated
        liq = liq_hat
        vap = vap_hat

        # Calculate the error
        error = calc_error(u, u_hat, a, a_hat, b, b_hat, valid)

        if outer_iterations > minimum_full_update_steps+pre_accelerate and accelerate:
            # a, b
            x = np.zeros(len(u) + 2)
            x[:len(u)] = u
            x[len(u):] = a, b

            f_x = np.zeros(len(u_hat) + 2)
            f_x[:len(u_hat)] = u_hat
            f_x[len(u_hat):] = a_hat, b_hat

            history.append(x)
            history.append(f_x)

            if len(history) == 6 and accelerate:
                x_2, f_x_2, x_1, f_x_1, x_0, f_x_0 = history
                x_inf = gdem(x_2, f_x_2, x_1, f_x_1, x_0, f_x_0)
                # print('Accelerating')
                u_hat = x_inf[:len(u_hat)]
                a_hat, b_hat = x_inf[len(u_hat):]
                history = []

        # Update all the variables, this is where acceleration could occur
        press = press_calc
        u = u_hat
        a = a_hat
        b = b_hat

    if not outer_converged:
        raise FlashConvergenceError

    # print('#', outer_iterations, 'press:', press, 'vap_frac:', vap_frac, 'error:', error)
    # Otherwise we can return the solution
    return AggregateByMole([liq, vap], [1 - vap_frac, vap_frac])

