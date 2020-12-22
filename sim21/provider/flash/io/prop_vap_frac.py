import numpy as np
from numba import njit
from math import exp, log

from sim21.provider.agg import AggregateByMole
from sim21.provider.error import FlashConvergenceError
from sim21.provider.flash.basic import generate_2phase_results, generate_2phase_estimates
from sim21.provider.flash.io.press_prop import initial_press_prop_weighting_coeffs
from sim21.provider.flash.io.settings import INSIDE_OUT_INNER_ITERATIONS, INSIDE_OUT_INNER_TOLERANCE, \
    INSIDE_OUT_OUTER_ITERATIONS, INSIDE_OUT_OUTER_TOLERANCE
from sim21.provider.flash.io.temp_press import calc_log_kb, calc_u
from sim21.support.accel import gdem


@njit(cache=True)
def calc_error_complete(u, u_hat,
                        a, a_hat, b_temp, b_temp_hat, b_press, b_press_hat,
                        c, c_hat, d_temp, d_temp_hat, d_press, d_press_hat,
                        e, e_hat, f, f_hat, valid):
    result = 0
    for i in valid:
        result += abs((u_hat[i] - u[i]))

    result += abs((a_hat - a) / a)
    result += abs((b_temp_hat - b_temp) / b_temp)
    result += abs((b_press_hat - b_press) / b_press)

    result += abs((c_hat - c) / c)
    result += abs((d_temp_hat - d_temp) / d_temp)
    result += abs((d_press_hat - d_press) / d_press)

    result += abs((e_hat - e) / e)
    result += abs((f_hat - f) / f)

    return result


def update_model_prop_vap_frac_2phase(provider,
                                      temp, temp_star, press, press_star, vap_frac,
                                      prop_type, prop_target,  prop_scaling,
                                      liq_comp, vap_comp, valid,
                                      previous_w, previous_u,
                                      previous_a, previous_b_temp, previous_b_press,
                                      previous_c, previous_d_temp, previous_d_press,
                                      previous_e, previous_f,
                                      previous_log_kb,
                                      previous_temp, previous_press,
                                      full_update=False):

    inv_temp_star = 1/temp_star
    # There is alteast one call to the provider to generate a liq and vapor phase
    # for the given condition
    log_k_values, liq, vap = generate_2phase_results(provider, temp, press, liq_comp, vap_comp, valid)

    # If there is a None in any of these terms, then we will generate the full model
    if previous_w is None or previous_u is None or  \
        previous_a is None or  previous_b_temp is None or  previous_b_press is None or \
        previous_c is None or previous_d_temp is None or  previous_d_press is None or \
        previous_e is None or previous_f is None or \
        previous_log_kb is None or \
        previous_temp is None or previous_press is None or full_update:

        # Save the initial values
        log_k_values_temp_1_press_1 = log_k_values
        liq_temp_1_press_1, vap_temp_1_press_1 = liq, vap
        temp_1 = temp

        # Perturb by some amount; by atleast 1 K or smaller depending on valid of temp
        delta_temp = 1.0
        temp_2 = temp_1 + delta_temp

        press_1 = press
        delta_press = press_1*1e-2
        press_2 = press_1 + delta_press

        # Generate the peturbed real cofficients
        log_k_values_temp_2_press_1, liq_temp_2_press_1, vap_temp_2_press_1 = generate_2phase_results(provider, temp_2, press_1, liq_comp, vap_comp, valid)
        log_k_values_temp_1_press_2, liq_temp_1_press_2, vap_temp_1_press_2 = generate_2phase_results(provider, temp_1, press_2, liq_comp, vap_comp, valid)

        # Generate the weighting factors only once
        if previous_w is None:
            w = initial_press_prop_weighting_coeffs(log_k_values_temp_1_press_1,
                                                    log_k_values_temp_2_press_1,
                                                    temp_1, temp_2,
                                                    vap_comp,
                                                    vap_frac,
                                                    valid)
        else:
            w = previous_w

        log_kb_hat_temp_1_press_1 = calc_log_kb(w, log_k_values_temp_1_press_1, valid)
        log_kb_hat_temp_2_press_1 = calc_log_kb(w, log_k_values_temp_2_press_1, valid)
        log_kb_hat_temp_1_press_2 = calc_log_kb(w, log_k_values_temp_1_press_2, valid)

        b_temp_hat = (log(exp(log_kb_hat_temp_1_press_1)*press_1) - log(exp(log_kb_hat_temp_2_press_1)*press_1))/((1/temp_1) - (1/temp_2))
        b_press_hat = (log(exp(log_kb_hat_temp_1_press_1)*press_1) - log(exp(log_kb_hat_temp_1_press_2)*press_2))/(log(press_1/press_star) - log(press_2/press_star))
        a_hat = log(exp(log_kb_hat_temp_1_press_1)*press_1) - b_temp_hat * (1/temp_1 - 1/temp_star) - b_press_hat * log(press_1/press_star)

        dhv_temp_1_press_1 = getattr(vap_temp_1_press_1, prop_type)/prop_scaling
        dhv_temp_2_press_1 = getattr(vap_temp_2_press_1, prop_type)/prop_scaling
        dhv_temp_1_press_2 = getattr(vap_temp_1_press_2, prop_type)/prop_scaling

        d_temp_hat = (dhv_temp_1_press_1 - dhv_temp_2_press_1)/(temp_1 - temp_2)
        d_press_hat = (dhv_temp_1_press_1 - dhv_temp_1_press_2)/((press_1/press_star) - (press_2/press_star))

        c_hat = dhv_temp_1_press_1 - d_temp_hat*(temp_1 - temp_star) - d_press_hat*(press_1/press_star)

        dhl_temp_1_press_1 = getattr(liq_temp_1_press_1, prop_type)/prop_scaling
        dhl_temp_2_press_1 = getattr(liq_temp_2_press_1, prop_type)/prop_scaling

        f_hat = (dhl_temp_1_press_1 - dhl_temp_2_press_1) / (temp_1 - temp_2)
        e_hat = dhl_temp_1_press_1 - f_hat * (temp_1 - temp_star)

        log_kb_hat = log_kb_hat_temp_1_press_1

    # Otherwise just do a simple update
    else:
        w = previous_w
        log_kb_hat_temp_1_press_1 = log_kb_hat = calc_log_kb(w, log_k_values, valid)
        temp_1, press_1 = temp, press
        b_temp_hat = previous_b_temp
        b_press_hat = previous_b_press

        dhv_temp_1_press_1 = getattr(vap, prop_type)/prop_scaling
        a_hat = log(exp(log_kb_hat_temp_1_press_1)*press_1) - b_temp_hat * (1/temp_1 - 1/temp_star) - b_press_hat * log(press_1/press_star)

        d_temp_hat = previous_d_temp
        d_press_hat = previous_d_press
        c_hat = dhv_temp_1_press_1 - d_temp_hat*(temp_1 - temp_star) - d_press_hat*(press_1/press_star)

        dhl_temp_1_press_1 = getattr(liq, prop_type)/prop_scaling
        f_hat = previous_f
        e_hat = dhl_temp_1_press_1 - f_hat * (temp_1 - temp_star)


    # Now update u_hat
    u_hat = calc_u(log_k_values, log_kb_hat, valid)

    # Return the updated model
    return w, u_hat, a_hat, b_temp_hat, b_press_hat, c_hat, d_temp_hat, d_press_hat, e_hat, f_hat, log_kb_hat, liq, vap


@njit(cache=True)
def solve_model_prop_vap_frac_2phase(feed_comp, kb_0,
                                     u, a, b_temp, b_press, c, d_temp, d_press, e, f, r_value,
                                     start_temp, temp_star,
                                     start_press, press_star,
                                     prop_target_scaled, vap_frac_spec, valid):

    # Should speed this up via jiting - can be very fast
    liq_comp = np.zeros(len(feed_comp))
    vap_comp = np.zeros(len(feed_comp))

    p = np.zeros(len(feed_comp))
    p_sum = 0
    exp_u_p_sum = 0
    inner_converged = False
    kb_calc = 1
    vap_frac_calc = r_value
    liq_frac_spec = 1 - vap_frac_spec

    first_loop_converged = False
    second_loop_converged = False
    temp_calc = start_temp
    press_calc = start_press

    for inner_iterations in range(INSIDE_OUT_INNER_ITERATIONS):
        if vap_frac_spec == 0 or vap_frac_spec == 1:
            r_value = vap_frac_spec
            for i in valid:
                p[i] = feed_comp[i] / (1.0 - r_value + kb_0 * r_value * exp(u[i]))
                p_sum += p[i]
                exp_u_p_sum += exp(u[i]) * p[i]

            first_loop_converged = True
        else:
            first_loop_converged = False
            func_eval = 1
            closest_func_eval = 1

            for inner1_iterations in range(INSIDE_OUT_INNER_ITERATIONS):
                if abs(func_eval) < INSIDE_OUT_INNER_TOLERANCE*100:
                    first_loop_converged = True
                    break

                p_sum = 0
                exp_u_p_sum = 0
                for i in valid:
                    p[i] = feed_comp[i] / (1.0 - r_value + kb_0 * r_value * exp(u[i]))
                    p_sum += p[i]
                    exp_u_p_sum += exp(u[i]) * p[i]

                liq_frac_calc = (1 - r_value) * p_sum
                func_eval = (liq_frac_calc - liq_frac_spec) ** 2
                if func_eval < closest_func_eval:
                    closest_func_eval = func_eval
                    closest_r_value = r_value

                r_value_delta = r_value + INSIDE_OUT_INNER_TOLERANCE / 100
                p_sum_delta = 0
                for i in valid:
                    p_sum_delta += feed_comp[i] / (1.0 - r_value + kb_0 * r_value * exp(u[i]))

                liq_frac_calc_delta = (1 - r_value_delta) * p_sum_delta
                func_eval_delta = (liq_frac_calc_delta - liq_frac_spec) ** 2

                df_dx = (func_eval_delta - func_eval) / (r_value_delta - r_value)
                if abs(df_dx) < INSIDE_OUT_INNER_TOLERANCE:
                    first_loop_converged = False
                    break

                r_value_new = r_value - func_eval / df_dx
                if abs(r_value_new - r_value) < INSIDE_OUT_INNER_TOLERANCE:
                    first_loop_converged = True
                    break

                if r_value_new < 0:
                    r_value_new = 0.9 * r_value
                elif r_value_new > 1.0:
                    r_value_new = min(1.0, 1.1 * r_value)

                r_value = r_value_new

        # Loop has converged or broken out
        kb_calc = p_sum/exp_u_p_sum
        log_kb_calc = log(kb_calc)
        liq_frac_calc = (1 - r_value) * p_sum
        vap_frac_calc = 1 - liq_frac_calc

        # Start secondary loop
        temp_calc = start_temp
        press_calc = start_press
        func_eval = 1

        for inner2_iterations in range(INSIDE_OUT_INNER_ITERATIONS):
            if abs(func_eval) < INSIDE_OUT_INNER_TOLERANCE:
                # print("Converged inner loop in ", inner_iterations, " iterations", "R:", R)
                second_loop_converged = True
                break

            log_press_calc = (a + b_temp*(1/temp_calc - 1/temp_star) - b_press*log(press_star) - log_kb_calc)/(1 - b_press)
            press_calc = exp(log_press_calc)

            h_vap = c + d_temp*(temp_calc - temp_star) + d_press*(press_calc/press_star)
            h_liq = e + f*(temp_calc - temp_star)

            func_eval = (vap_frac_calc * h_vap + liq_frac_calc * h_liq - prop_target_scaled)

            temp_calc_prime = temp_calc + 1e-6
            log_press_calc_prime = (a + b_temp*(1/temp_calc_prime - 1/temp_star) - b_press*log(press_star) - log_kb_calc)/(1 - b_press)
            press_calc_prime = exp(log_press_calc_prime)

            h_vap_prime = c + d_temp*(temp_calc_prime - temp_star) + d_press*(press_calc_prime/press_star)
            h_liq_prime = e + f*(temp_calc_prime - temp_star)

            func_eval_prime = (vap_frac_calc * h_vap_prime + liq_frac_calc * h_liq_prime - prop_target_scaled)

            d_func_dtemp = (func_eval_prime - func_eval)/(temp_calc_prime - temp_calc)
            if abs(d_func_dtemp) < INSIDE_OUT_INNER_TOLERANCE:
                second_loop_converged = False
                break

            temp_calc_new = temp_calc - func_eval/d_func_dtemp
            if abs(temp_calc_new - temp_calc) < INSIDE_OUT_INNER_TOLERANCE:
                second_loop_converged = False
                break

            temp_calc = temp_calc_new
            kb_calc = (exp(a + b_temp*(1/temp_calc - 1/temp_star) + b_press*log(press_calc/press_star)))/press_calc
            log_kb_calc = log(kb_calc)

        if first_loop_converged and second_loop_converged:
            inner_converged = True
            break

    for i in valid:
        liq_comp[i] = p[i] / p_sum
        vap_comp[i] = exp(u[i]) * p[i] / exp_u_p_sum

    return inner_converged, p, p_sum, r_value, vap_frac_calc, temp_calc, press_calc, kb_calc, liq_comp, vap_comp


def flash_prop_vap_frac_2phase(provider,
                               prop_type, prop_target, delta_target,
                               vap_frac, feed_comp,
                               valid=None, previous=None, override_k_values=None):
    if valid is None:
        valid = provider.all_valid_components

    prop_scaling = provider.scaling(prop_type)
    prop_target_scaled = (prop_target + delta_target)/prop_scaling

    # temp_star is a reference value
    temp_star = 298.15
    press_star = 1e5
    press_est = 101325.0

    # Just a dummy value
    if previous is None:
        temp_est = provider.guess_nbp(feed_comp, valid)
        k_values, \
        liq_comp, vap_comp, _ = generate_2phase_estimates(provider, temp_est, press_est,
                                                          feed_comp, valid, override_k_values=override_k_values)
        # Correction to make sure that temp_star is not close to temp_est
        # can occasionally cause a problem
        if abs(temp_est - temp_star)/temp_est < 0.05:
            temp_star = 0.95*temp_est

        if abs(press_est - press_star) / press_est < 0.1:
            press_star = 0.90 * press_est

    else:
        raise NotImplementedError

    temp = temp_est
    press = press_est
    r_value = vap_frac
    inner_converged, outer_converged = False, False
    minimum_full_update_steps = 5
    pre_accelerate = 2
    accelerate = True

    w, u, \
    a, b_temp, b_press, \
    c, d_temp, d_press, \
    e, f, log_kb, liq, vap = update_model_prop_vap_frac_2phase(provider,
                                                               temp, temp_star, press, press_star, vap_frac,
                                                               prop_type, prop_target,  prop_scaling,
                                                               liq_comp, vap_comp, valid,
                                                               None, None,
                                                               None, None, None,
                                                               None, None, None,
                                                               None, None,
                                                               None, None, None,
                                                               full_update=True)

    force_full_update = False
    error = 0
    kb_0 = exp(log_kb)
    history = []

    for outer_iterations in range(INSIDE_OUT_OUTER_ITERATIONS):
        # Check if the actual error is less than tolerance, if so we quit
        if outer_iterations > 1 and error < INSIDE_OUT_OUTER_TOLERANCE:
            outer_converged = True
            break

        # print('#', outer_iterations, 'temp:', temp, 'press:', press, 'vap_frac:', vap_frac, 'error:', error)
        # If we have less than 5 iterations, we do a full update
        # This helps minimize the number of iterations
        if outer_iterations < minimum_full_update_steps:
            force_full_update = True

        # Solve the inner simplified model
        inner_converged, \
        p, p_sum, r_value, \
        vap_frac_calc, \
        temp_calc, press_calc, \
        kb_calc, liq_comp_hat, vap_comp_hat = solve_model_prop_vap_frac_2phase(feed_comp, kb_0,
                                                                               u, a, b_temp, b_press,
                                                                               c, d_temp, d_press, e, f, r_value,
                                                                               temp, temp_star,
                                                                               press, press_star,
                                                                               prop_target_scaled,
                                                                               vap_frac, valid)

        if press_calc < 0.5*press:
            press_calc = 0.9*press
        elif press_calc > 4.0*press:
            press_calc = 1.5*press

        if temp_calc < 0.5*temp:
            temp_calc = 0.9*temp
        elif temp_calc > 2.0*temp:
            temp_calc = 1.1*temp

        liq_comp = liq_comp_hat
        vap_comp = vap_comp_hat

        # Get the update the model factors
        w, u_hat, \
        a_hat, b_temp_hat, b_press_hat, \
        c_hat, d_temp_hat, d_press_hat, \
        e_hat, f_hat, log_kb_hat, liq_hat, vap_hat = update_model_prop_vap_frac_2phase(provider,
                                                                                       temp_calc, temp_star,
                                                                                       press_calc, press_star, vap_frac,
                                                                                       prop_type, prop_target, prop_scaling,
                                                                                       liq_comp, vap_comp, valid,
                                                                                       w, u, a, b_temp, b_press,
                                                                                       c, d_temp, d_press, e, f,
                                                                                       log_kb, temp, press,
                                                                                       full_update=force_full_update)

        # Save the actual phases generated

        previous_error = error
        error = calc_error_complete(u, u_hat, a, a_hat, b_temp, b_temp_hat, b_press, b_press_hat,
                                    c, c_hat, d_temp, d_temp_hat, d_press, d_press_hat,
                                    e, e_hat, f, f_hat, valid)

        if outer_iterations > minimum_full_update_steps+pre_accelerate and accelerate:
            # a, b_temp, b_press, c, d_temp, d_press, e, f
            x = np.zeros(len(u) + 8)
            x[:len(u)] = u
            x[len(u):] = a, b_temp, b_press, c, d_temp, d_press, e, f

            f_x = np.zeros(len(u_hat) + 8)
            f_x[:len(u_hat)] = u_hat
            f_x[len(u_hat):] = a_hat, b_temp_hat, b_press_hat, c_hat, d_temp_hat, d_press_hat, e_hat, f_hat

            history.append(x)
            history.append(f_x)

            if len(history) == 6 and accelerate:
                x_2, f_x_2, x_1, f_x_1, x_0, f_x_0 = history
                x_inf = gdem(x_2, f_x_2, x_1, f_x_1, x_0, f_x_0)
                # print('Accelerating')
                u_hat = x_inf[:len(u_hat)]
                a_hat, b_temp_hat, b_press_hat, c_hat, d_temp_hat, d_press_hat, e_hat, f_hat = x_inf[len(u_hat):]
                # history.pop()
                # history.pop()
                # history.pop()
                # history.pop()
                history = []

        temp = temp_calc
        press = press_calc
        u = u_hat
        a = a_hat
        b_temp = b_temp_hat
        b_press = b_press_hat
        c = c_hat
        d_temp = d_temp_hat
        d_press = d_press_hat
        e = e_hat
        f = f_hat
        log_kb = log_kb_hat
        liq = liq_hat
        vap = vap_hat

    if not outer_converged:
        raise FlashConvergenceError

    # Otherwise we can return the solution
    return AggregateByMole(provider, [liq, vap], [1 - vap_frac, vap_frac])

