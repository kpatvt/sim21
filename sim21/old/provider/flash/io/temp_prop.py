from math import log, exp
import numpy as np
from numba import njit

from sim21.old.provider.agg import AggregateByMole
from sim21.old.provider.error import FlashConvergenceError
from sim21.old.provider.flash.basic import generate_2phase_results, generate_2phase_estimates
from sim21.old.provider.flash.io.press_prop import initial_press_prop_weighting_coeffs, calc_error_full
from sim21.old.provider.flash.io.settings import INSIDE_OUT_INNER_ITERATIONS, INSIDE_OUT_INNER_TOLERANCE, \
    INSIDE_OUT_OUTER_ITERATIONS, INSIDE_OUT_OUTER_TOLERANCE
from sim21.old.provider.flash.io.temp_press import calc_log_kb, calc_u, flash_temp_press_2phase
from sim21.old.support.accel import gdem


def update_model_temp_prop_2phase(provider, temp, press, press_star, vap_frac,
                                  prop_type, prop_target, prop_scaling,
                                  liq_comp, vap_comp, valid,
                                  previous_w, previous_u,
                                  previous_a, previous_b, previous_c, previous_d, previous_e, previous_f,
                                  previous_log_kb, previous_press, full_update=False):
    # There is alteast one call to the provider to generate a liq and vapor phase
    # for the given condition
    log_k_values, liq, vap = generate_2phase_results(provider, temp, press, liq_comp, vap_comp, valid)

    # If there is a None in any of these terms, then we will generate the full model
    if previous_w is None or previous_u is None or previous_a is None or previous_b is None or \
        previous_c is None or previous_d is None or previous_e is None or previous_f is None or \
        previous_log_kb is None or previous_press is None or full_update:
        # Save the initial values
        log_k_values_1, liq_1, vap_1, press_1 = log_k_values, liq, vap, press

        # Generate the weighting factors only once
        if previous_w is None:
            # The temperature needs to be peturbed once to get the weighting coefficients
            # Perturb by some amount; by atleast 1 K or smaller depending on valid of temp
            delta_temp = 1.0
            temp_1 = temp
            temp_2 = temp_1 + delta_temp
            # Generate the peturbed real cofficients
            log_k_values_2, liq_2, vap_2 = generate_2phase_results(provider, temp_2, press, liq_comp, vap_comp, valid)
            w = initial_press_prop_weighting_coeffs(log_k_values_1, log_k_values_2, temp_1, temp_2, vap_comp, vap_frac,
                                                    valid)
        else:
            w = previous_w

        # Generate the peturbed pressure value
        press_2 = press_1 * 1.005
        log_k_values_2, liq_2, vap_2 = generate_2phase_results(provider, temp, press_2, liq_comp, vap_comp, valid)

        log_kb_hat_1 = calc_log_kb(w, log_k_values_1, valid)
        log_kb_hat_2 = calc_log_kb(w, log_k_values_2, valid)
        log_kb_hat = log_kb_hat_1

        kb_hat_1 = exp(log_kb_hat_1)
        kb_hat_2 = exp(log_kb_hat_2)

        b_hat = (log(kb_hat_1 * press_1) - log(kb_hat_2 * press_2)) / (
                log(press_1 / press_star) - log(press_2 / press_star))
        a_hat = log(kb_hat_1 * press_1) - b_hat * log(press_1 / press_star)

        dhv_1 = getattr(vap_1, prop_type) / prop_scaling
        dhv_2 = getattr(vap_2, prop_type) / prop_scaling

        d_hat = (dhv_1 - dhv_2) / (press_1 / press_star - press_2 / press_star)
        c_hat = dhv_1 - d_hat * (press_1 / press_star)

        dhl_1 = getattr(liq_1, prop_type) / prop_scaling
        dhl_2 = getattr(liq_2, prop_type) / prop_scaling

        f_hat = (dhl_1 - dhl_2) / (press_1 / press_star - press_2 / press_star)
        e_hat = dhl_1 - f_hat * (press_1 / press_star)

    # Otherwise just do a simple update
    else:
        w = previous_w
        log_kb_hat = calc_log_kb(w, log_k_values, valid)
        kb_hat_1 = exp(log_kb_hat)
        press_1 = press

        b_hat = previous_b
        a_hat = log(kb_hat_1 * press_1) - b_hat * log(press_1 / press_star)

        dhv_1 = getattr(vap, prop_type) / prop_scaling
        d_hat = previous_d
        c_hat = dhv_1 - d_hat * (press_1 / press_star)

        dhl_1 = getattr(liq, prop_type) / prop_scaling
        f_hat = previous_f
        e_hat = dhl_1 - f_hat * (press_1 / press_star)

    # Now update u_hat
    u_hat = calc_u(log_k_values, log_kb_hat, valid)

    # Return the updated model
    return w, u_hat, a_hat, b_hat, c_hat, d_hat, e_hat, f_hat, log_kb_hat, liq, vap


@njit(cache=True)
def solve_model_temp_prop_2phase(feed_comp, kb_0, u, a, b, c, d, e, f, r_value, press, press_star, prop_target_scaled,
                                 valid):
    # Should speed this up via jiting - can be very fast
    liq_comp = np.zeros(len(feed_comp))
    vap_comp = np.zeros(len(feed_comp))

    p = np.zeros(len(feed_comp))
    p_sum = 0
    exp_u_p_sum = 0
    inner_converged = False
    kb_calc = 1
    vap_frac_calc = r_value
    vap_present, liq_present = False, False
    log_press_star = log(press_star)

    for inner_iterations in range(INSIDE_OUT_INNER_ITERATIONS):
        # -- First evaluation
        p_sum = 0
        exp_u_p_sum = 0
        for i in valid:
            p[i] = feed_comp[i] / (1 - r_value + kb_0 * r_value * exp(u[i]))
            p_sum += p[i]
            exp_u_p_sum += exp(u[i]) * p[i]

        kb_calc = p_sum / exp_u_p_sum
        if kb_calc < 0:
            break

        log_kb_calc = log(kb_calc)
        log_press_calc = (log_kb_calc + b * log_press_star - a) / (b - 1)
        press_calc = exp(log_press_calc)

        liq_frac_calc = (1 - r_value) * p_sum
        vap_frac_calc = 1 - liq_frac_calc

        h_vap = c + d * (press_calc / press_star)
        h_liq = e + f * (press_calc / press_star)

        func_eval = (vap_frac_calc * h_vap + liq_frac_calc * h_liq - prop_target_scaled)
        # print('func:', func_eval, 'vap_frac:', vap_frac_calc)

        if abs(func_eval) < INSIDE_OUT_INNER_TOLERANCE:
            inner_converged = True
            break

        # -- Second evaluation
        p_sum_prime = 0
        exp_u_p_sum_prime = 0
        r_value_prime = r_value + INSIDE_OUT_INNER_TOLERANCE / 1000

        for i in valid:
            p_i_prime = feed_comp[i] / (1 - r_value_prime + kb_0 * r_value_prime * exp(u[i]))
            p_sum_prime += p_i_prime
            exp_u_p_sum_prime += exp(u[i]) * p_i_prime

        kb_calc_prime = p_sum_prime / exp_u_p_sum_prime
        if kb_calc_prime < 0:
            inner_converged = False
            break

        log_kb_calc_prime = log(kb_calc_prime)
        log_press_calc_prime = (log_kb_calc_prime + b * log_press_star - a) / (b - 1)
        press_calc_prime = exp(log_press_calc_prime)

        liq_frac_calc_prime = (1 - r_value_prime) * p_sum_prime
        vap_frac_calc_prime = 1 - liq_frac_calc_prime

        h_vap_prime = c + d * (press_calc_prime / press_star)
        h_liq_prime = e + f * (press_calc_prime / press_star)

        func_eval_prime = (vap_frac_calc_prime * h_vap_prime + liq_frac_calc_prime * h_liq_prime - prop_target_scaled)

        df_dx = (func_eval_prime - func_eval) / (r_value_prime - r_value)
        if abs(df_dx) < INSIDE_OUT_INNER_TOLERANCE * 1e-3:
            inner_converged = False
            # Set to r = 0
            r_value = 0
            p_sum = 0
            exp_u_p_sum = 0
            for i in valid:
                p[i] = feed_comp[i] / (1 - r_value + kb_0 * r_value * exp(u[i]))
                p_sum += p[i]
                exp_u_p_sum += exp(u[i]) * p[i]

            kb_calc = p_sum / exp_u_p_sum
            if kb_calc < 0:
                break

            log_kb_calc = log(kb_calc)
            log_press_calc = (log_kb_calc + b * log_press_star - a) / (b - 1)
            press_calc = exp(log_press_calc)

            liq_frac_calc = (1 - r_value) * p_sum
            vap_frac_calc = 1 - liq_frac_calc

            h_vap = c + d * (press_calc / press_star)
            h_liq = e + f * (press_calc / press_star)

            func_eval_r0 = (vap_frac_calc * h_vap + liq_frac_calc * h_liq - prop_target_scaled)

            # Set to r = 1
            r_value = 1
            p_sum = 0
            exp_u_p_sum = 0
            for i in valid:
                p[i] = feed_comp[i] / (1 - r_value + kb_0 * r_value * exp(u[i]))
                p_sum += p[i]
                exp_u_p_sum += exp(u[i]) * p[i]

            kb_calc = p_sum / exp_u_p_sum
            if kb_calc < 0:
                break

            log_kb_calc = log(kb_calc)
            log_press_calc = (log_kb_calc + b * log_press_star - a) / (b - 1)
            press_calc = exp(log_press_calc)

            liq_frac_calc = (1 - r_value) * p_sum
            vap_frac_calc = 1 - liq_frac_calc

            h_vap = c + d * (press_calc / press_star)
            h_liq = e + f * (press_calc / press_star)

            func_eval_r1 = (vap_frac_calc * h_vap + liq_frac_calc * h_liq - prop_target_scaled)

            if abs(func_eval_r1) < abs(func_eval_r0):
                r_value = 1
                vap_present = True
                liq_present = False
            else:
                r_value = 0
                vap_present = False
                liq_present = True

            p_sum = 0
            exp_u_p_sum = 0
            for i in valid:
                p[i] = feed_comp[i] / (1 - r_value + kb_0 * r_value * exp(u[i]))
                p_sum += p[i]
                exp_u_p_sum += exp(u[i]) * p[i]

            kb_calc = p_sum / exp_u_p_sum
            if kb_calc < 0:
                break

            log_kb_calc = log(kb_calc)
            log_press_calc = (log_kb_calc + b * log_press_star - a) / (b - 1)
            press_calc = exp(log_press_calc)
            # Now break
            break

        r_value_new = r_value - (func_eval / df_dx)
        r_value = r_value_new

    # We converged to a vapor only solution
    if r_value > 1:
        r_value = 1
        p_sum = 0
        exp_u_p_sum = 0
        for i in valid:
            p[i] = feed_comp[i] / (1 - r_value + kb_0 * r_value * exp(u[i]))
            p_sum += p[i]
            exp_u_p_sum += exp(u[i]) * p[i]

        kb_calc = p_sum / exp_u_p_sum
        vap_present = True
        liq_present = False
    # We converged to liquid only solution
    elif r_value < 0:
        r_value = 0
        p_sum = 0
        exp_u_p_sum = 0
        for i in valid:
            p[i] = feed_comp[i] / (1 - r_value + kb_0 * r_value * exp(u[i]))
            p_sum += p[i]
            exp_u_p_sum += exp(u[i]) * p[i]

        kb_calc = p_sum / exp_u_p_sum
        vap_present = False
        liq_present = True

    elif inner_converged is True:
        vap_present = True
        liq_present = True

    if kb_calc > 0:
        log_kb_calc = log(kb_calc)
        log_press_calc = (log_kb_calc + b * log_press_star - a) / (b - 1)
        press_calc = exp(log_press_calc)
    else:
        press_calc = press

    for i in valid:
        liq_comp[i] = p[i] / p_sum
        vap_comp[i] = exp(u[i]) * p[i] / exp_u_p_sum

    return inner_converged, p, p_sum, r_value, vap_frac_calc, press_calc, kb_calc, liq_comp, vap_comp, vap_present, liq_present


def flash_temp_prop_2phase(provider, temp,
                           prop_type, prop_target, delta_target,
                           feed_comp, valid=None, previous=None,
                           override_k_values=None,
                           start_press=None, recursive=False):
    if valid is None:
        valid = provider.all_valid_components

    accelerate = True
    # Determine the scaled target for the property
    prop_scaling = provider.scaling(prop_type)
    prop_target_scaled = (prop_target + delta_target) / prop_scaling

    # press_star is a reference value
    press_star = 100e5
    minimum_full_update_steps = 5
    pre_accelerate = 5
    if previous is None:
        if start_press is None:
            press_est = 101325
        else:
            press_est = start_press

        k_values, \
        liq_comp, vap_comp, \
        vap_frac = generate_2phase_estimates(provider,
                                             temp, press_est, feed_comp, valid,
                                             override_k_values=override_k_values)
        # Correction to make sure that temp_star is not close to temp_est
        # can occasionally cause a problem
        if abs(press_est - press_star) / press_est < 0.05:
            press_star = 0.95 * press_est
    else:
        raise NotImplementedError

    # Initialize the values of the search
    press = press_est
    r_value = vap_frac
    inner_converged, outer_converged = False, False
    w, u, a, b, c, d, e, f, log_kb, liq, vap = update_model_temp_prop_2phase(provider, temp,
                                                                             press, press_star, vap_frac,
                                                                             prop_type, prop_target, prop_scaling,
                                                                             liq_comp, vap_comp, valid,
                                                                             None, None, None, None, None, None,
                                                                             None, None, None, None,
                                                                             full_update=False)

    force_full_update = False
    error = 0
    kb_0 = exp(log_kb)
    vap_present, liq_present = False, False
    history = []

    for outer_iterations in range(INSIDE_OUT_OUTER_ITERATIONS):
        # Check if the actual error is less than tolerance, if so we quit
        if outer_iterations > 1 and error < INSIDE_OUT_OUTER_TOLERANCE:
            outer_converged = True
            break

        # print('#', outer_iterations, 'press:', press, 'vap_frac:', vap_frac, 'error:', error, 'vap_present:', vap_present, 'liq_present:', liq_present)
        # If we have less than 5 iterations, we do a full update
        # This helps minimize the number of iterations
        if outer_iterations < minimum_full_update_steps:
            force_full_update = True

        # Solve the inner simplified model
        inner_converged, \
        p, p_sum, r_value, vap_frac, \
        press_calc, \
        kb_calc, \
        liq_comp_new, vap_comp_new, vap_present, liq_present = solve_model_temp_prop_2phase(feed_comp, kb_0,
                                                                                            u, a, b,
                                                                                            c, d,
                                                                                            e, f, r_value,
                                                                                            press, press_star,
                                                                                            prop_target_scaled, valid)

        # If we hit a one phase at any point, we switch over to one phase iterations
        if outer_iterations > 1 and False in (vap_present, liq_present):
            if liq_present and not vap_present:
                chosen_phase = 'liq'
            elif not liq_present and vap_present:
                chosen_phase = 'vap'
            else:
                raise FlashConvergenceError

            # If we called recursively, ending up here is a weird situation, error out
            if recursive:
                raise FlashConvergenceError

            # If the inner loop didn't converge, the press_calc is probably junk
            if not inner_converged:
                press_calc = 101325.0

            # Switching to one phase search
            # print('Switching to one phase search in:', LIQUID)
            return flash_temp_prop_1phase(provider, chosen_phase,
                                          temp,
                                          press_calc,
                                          prop_type,
                                          prop_target,
                                          delta_target,
                                          feed_comp,
                                          valid, liq)

        # if the temp_calc is negative or a sudden increase temp,
        # damp the update
        if press_calc > 5.0 * press:
            press_calc = 1.5 * press
            force_full_update = True
        else:
            force_full_update = False
            liq_comp = liq_comp_new
            vap_comp = vap_comp_new

        if not inner_converged:
            # force_full_update = True
            pass

        # Get the update the model factors
        w, u_hat, a_hat, b_hat, \
        c_hat, d_hat, \
        e_hat, f_hat, \
        log_kb_hat, liq_hat, vap_hat = update_model_temp_prop_2phase(provider, temp, press_calc, press_star, vap_frac,
                                                                     prop_type, prop_target, prop_scaling,
                                                                     liq_comp, vap_comp, valid,
                                                                     w, u, a, b, c, d, e, f,
                                                                     log_kb, press, full_update=force_full_update)

        # Save the actual phases generated
        liq = liq_hat
        vap = vap_hat
        error = calc_error_full(u, u_hat, a, a_hat, b, b_hat, c, c_hat, d, d_hat, e, e_hat, f, f_hat, valid)
        press = press_calc

        if outer_iterations > minimum_full_update_steps + pre_accelerate and accelerate:
            x = np.zeros(len(u) + 6)
            x[:len(u)] = u
            x[len(u):] = a, b, c, d, e, f

            f_x = np.zeros(len(u_hat) + 6)
            f_x[:len(u_hat)] = u_hat
            f_x[len(u_hat):] = a_hat, b_hat, c_hat, d_hat, e_hat, f_hat

            history.append(x)
            history.append(f_x)

            if len(history) == 6 and accelerate:
                x_2, f_x_2, x_1, f_x_1, x_0, f_x_0 = history
                x_inf = gdem(x_2, f_x_2, x_1, f_x_1, x_0, f_x_0)
                # print('Accelerating')
                u_hat = x_inf[:len(u_hat)]
                a_hat, b_hat, c_hat, d_hat, e_hat, f_hat = x_inf[len(u_hat):]
                # history.pop()
                # history.pop()
                # history.pop()
                # history.pop()
                history = []

        u = u_hat
        a = a_hat
        b = b_hat
        c = c_hat
        d = d_hat
        e = e_hat
        f = f_hat
        log_kb = log_kb_hat

    if not outer_converged:
        raise FlashConvergenceError

    if outer_converged and vap_present and liq_present:
        return AggregateByMole(provider, [liq, vap], [1 - vap_frac, vap_frac])
    else:
        raise FlashConvergenceError


def flash_temp_prop_1phase(provider, phase_type, temp, start_press, prop_type, prop_target, delta_target, feed_comp,
                           valid, ph):
    prop_scaling = provider.scaling(prop_type)
    prop_target_scaled = (prop_target + delta_target) / prop_scaling
    # print('Searching for:', phase_type)

    press_1 = start_press
    converged = False

    for iterations in range(INSIDE_OUT_OUTER_ITERATIONS):
        press_2 = press_1 + press_1 * 1e-3
        prop_1 = getattr(provider.phase(temp, press_1, feed_comp, phase_type, valid=valid), prop_type) / prop_scaling
        prop_2 = getattr(provider.phase(temp, press_2, feed_comp, phase_type, valid=valid), prop_type) / prop_scaling
        func_eval = (prop_1 - prop_target_scaled) * 1000

        # print('#', iterations, 'press:', press_1, 'error:', func_eval)

        if abs(func_eval) < INSIDE_OUT_OUTER_TOLERANCE:
            converged = True
            break

        func_eval_prime = (prop_2 - prop_target_scaled) * 1000
        df = (func_eval_prime - func_eval)
        df_dx = df / (press_2 - press_1)

        # if abs(df_dx) < INSIDE_OUT_INNER_TOLERANCE:
        #     raise FlashConvergenceError

        delta_press = func_eval / df_dx
        press_1_new = press_1 - delta_press

        if press_1_new < 0:
            press_1_new = 0.5 * press_1

        press_1 = press_1_new

    if not converged:
        raise FlashConvergenceError

    press_final = press_1
    # If we converged, now check if we can flash into multiple phases
    results = flash_temp_press_2phase(provider, temp, press_final, feed_comp, valid)
    if len(results.phases) > 1:
        # This is a rare situation but can happen, if it does we call the original flash recursively with the new
        # values, if we fail, then we are done.
        new_k_values = results.k_values_vle
        return flash_temp_prop_2phase(provider, temp, prop_type, prop_target, delta_target, feed_comp, valid=valid,
                                      previous=None, override_k_values=new_k_values,
                                      start_press=press_final, recursive=True)

    else:
        return results
