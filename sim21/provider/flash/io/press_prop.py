import numpy as np
from numba import njit
from math import exp, log

from .settings import INSIDE_OUT_INNER_ITERATIONS, INSIDE_OUT_INNER_TOLERANCE, INSIDE_OUT_OUTER_ITERATIONS, \
    INSIDE_OUT_OUTER_TOLERANCE, INSIDE_OUT_DAMPING
from .temp_press import calc_log_kb, calc_u, flash_temp_press_2phase
from ..basic import generate_2phase_results, generate_2phase_estimates
from ..nested import nested_press_prop_2phase
from ...agg import AggregateByMole
from ...error import FlashConvergenceError


@njit(cache=True)
def initial_press_prop_weighting_coeffs(log_k_values_1, log_k_values_2, temp_1, temp_2, vap_comp, vap_frac, valid):
    t = np.zeros(len(log_k_values_1))
    w = np.zeros(len(log_k_values_1))
    t_sum = 0
    for i in valid:
        d_log_k_d_temp = (log_k_values_2[i] - log_k_values_1[i])/(temp_2 - temp_1)
        t[i] = vap_comp[i]*d_log_k_d_temp/(1 - vap_frac + vap_frac*exp(log_k_values_1[i]))
        t_sum += t[i]

    for i in valid:
        w[i] = t[i]/t_sum

    return w


@njit(cache=True)
def calc_error_full(u, u_hat, a, a_hat, b, b_hat, c, c_hat, d, d_hat, e, e_hat, f, f_hat, valid):
    # Also can be jitted out
    result = 0
    for i in valid:
        result += abs((u_hat[i] - u[i]))

    result += abs((a_hat - a) / a)
    result += abs((b_hat - b) / b)

    result += abs((c_hat - c) / c)
    result += abs((d_hat - d) / d)

    result += abs((e_hat - e) / e)
    result += abs((f_hat - f) / f)

    return result


def update_model_press_prop_2phase(provider,
                                   temp, temp_star,
                                   press,
                                   vap_frac,
                                   prop_type, prop_target, prop_scaling,
                                   liq_comp, vap_comp, valid,
                                   previous_w, previous_u,
                                   previous_a, previous_b,
                                   previous_c, previous_d,
                                   previous_e, previous_f,
                                   previous_log_kb,
                                   previous_temp,
                                   full_update=False):

    inv_temp_star = 1/temp_star
    # There is alteast one call to the provider to generate a liq and vapor phase
    # for the given condition
    log_k_values, liq, vap = generate_2phase_results(provider, temp, press, liq_comp, vap_comp, valid)

    # If there is a None in any of these terms, then we will generate the full model
    if previous_w is None or previous_u is None or previous_a is None or previous_b is None or \
        previous_c is None or previous_d is None or previous_e is None or previous_f is None or \
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
            w = initial_press_prop_weighting_coeffs(log_k_values_1, log_k_values_2, temp_1, temp_2, vap_comp, vap_frac, valid)
        else:
            w = previous_w

        log_kb_hat_1 = calc_log_kb(w, log_k_values_1, valid)
        log_kb_hat_2 = calc_log_kb(w, log_k_values_2, valid)

        inv_temp_1 = 1/temp_1
        inv_temp_2 = 1/temp_2
        b_hat = (log_kb_hat_2 - log_kb_hat_1)/(inv_temp_2 - inv_temp_1)
        a_hat = log_kb_hat_1 - b_hat*(inv_temp_1 - inv_temp_star)
        log_kb_hat = log_kb_hat_1

        dhv_1 = getattr(vap_1, prop_type)/prop_scaling
        dhv_2 = getattr(vap_2, prop_type)/prop_scaling

        d_hat = (dhv_1 - dhv_2) / (temp_1 - temp_2)
        c_hat = dhv_1 - d_hat * (temp_1 - temp_star)

        dhl_1 = getattr(liq_1, prop_type)/prop_scaling
        dhl_2 = getattr(liq_2, prop_type)/prop_scaling

        f_hat = (dhl_1 - dhl_2) / (temp_1 - temp_2)
        e_hat = dhl_1 - f_hat * (temp_1 - temp_star)

    # Otherwise just do a simple update
    else:
        inv_temp = 1/temp
        temp_1 = temp
        w = previous_w
        log_kb_hat = calc_log_kb(w, log_k_values, valid)
        b_hat = previous_b
        a_hat = log_kb_hat - b_hat*(inv_temp - inv_temp_star)

        dhv_1 = getattr(vap, prop_type)/prop_scaling
        d_hat = previous_d
        c_hat = dhv_1 - d_hat * (temp_1 - temp_star)

        dhl_1 = getattr(liq, prop_type)/prop_scaling
        f_hat = previous_f
        e_hat = dhl_1 - f_hat * (temp_1 - temp_star)

    # Now update u_hat
    u_hat = calc_u(log_k_values, log_kb_hat, valid)

    # Return the updated model
    return w, u_hat, a_hat, b_hat, c_hat, d_hat, e_hat, f_hat, log_kb_hat, liq, vap


@njit(cache=True)
def solve_model_press_prop_2phase(feed_comp, kb_0, u, a, b, c, d, e, f, r_value, temp, temp_star, prop_target_scaled, valid):
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

    for inner_iterations in range(INSIDE_OUT_INNER_ITERATIONS):
        # -- First evaluation
        p_sum = 0
        exp_u_p_sum = 0
        for i in valid:
            p[i] = feed_comp[i]/(1 - r_value + kb_0*r_value*exp(u[i]))
            p_sum += p[i]
            exp_u_p_sum += exp(u[i])*p[i]

        kb_calc = p_sum/exp_u_p_sum
        if kb_calc < 0:
            break

        temp_calc_inv = (1/temp_star + (log(kb_calc) - a)/b)
        if temp_calc_inv < 0:
            break

        temp_calc = 1/temp_calc_inv
        liq_frac_calc = (1 - r_value)*p_sum
        vap_frac_calc = 1 - liq_frac_calc

        h_vap = c + d*(temp_calc - temp_star)
        h_liq = e + f*(temp_calc - temp_star)

        func_eval = (vap_frac_calc*h_vap + liq_frac_calc*h_liq - prop_target_scaled)
        # print('func:', func_eval, 'vap_frac:', vap_frac_calc)

        if abs(func_eval) < INSIDE_OUT_INNER_TOLERANCE:
            inner_converged = True
            break

        # -- Second evaluation
        p_sum_prime = 0
        exp_u_p_sum_prime = 0
        r_value_prime = r_value + INSIDE_OUT_INNER_TOLERANCE/1000

        for i in valid:
            p_i_prime = feed_comp[i]/(1 - r_value_prime + kb_0*r_value_prime*exp(u[i]))
            p_sum_prime += p_i_prime
            exp_u_p_sum_prime += exp(u[i])*p_i_prime

        kb_calc_prime = p_sum_prime/exp_u_p_sum_prime
        if kb_calc_prime < 0:
            inner_converged = False
            break

        temp_calc_prime_inv = (1/temp_star + (log(kb_calc_prime) - a)/b)
        if temp_calc_prime_inv < 0:
            inner_converged = False
            break

        temp_calc_prime = 1/temp_calc_prime_inv
        liq_frac_calc_prime = (1 - r_value_prime)*p_sum_prime
        vap_frac_calc_prime = 1 - liq_frac_calc_prime

        h_vap_prime = c + d*(temp_calc_prime - temp_star)
        h_liq_prime = e + f*(temp_calc_prime - temp_star)

        func_eval_prime = (vap_frac_calc_prime*h_vap_prime + liq_frac_calc_prime*h_liq_prime - prop_target_scaled)

        df_dx = (func_eval_prime - func_eval)/(r_value_prime - r_value)
        if abs(df_dx) < INSIDE_OUT_INNER_TOLERANCE:
            # print('Derivative goes to zero')
            inner_converged = False
            break

        r_value_new = r_value - (func_eval/df_dx)
        if r_value_new < 0:
            r_value_new = r_value/2
        elif r_value_new > 1:
            r_value_new = (1 + r_value)/2

        r_value = r_value_new

    # print('r_value:', r_value)
    if not inner_converged:
        # Evaluate at r = 0
        r_value = 0

        p_sum = 0
        exp_u_p_sum = 0
        for i in valid:
            p[i] = feed_comp[i]/(1 - r_value + kb_0*r_value*exp(u[i]))
            p_sum += p[i]
            exp_u_p_sum += exp(u[i])*p[i]

        kb_calc = p_sum/exp_u_p_sum
        temp_calc_inv = (1/temp_star + (log(kb_calc) - a)/b)
        temp_calc = 1/temp_calc_inv
        liq_frac_calc = (1 - r_value)*p_sum
        vap_frac_calc = 1 - liq_frac_calc

        h_vap = c + d*(temp_calc - temp_star)
        h_liq = e + f*(temp_calc - temp_star)

        func_at_r0 = (vap_frac_calc*h_vap + liq_frac_calc*h_liq - prop_target_scaled)

        r_value = 1
        p_sum = 0
        exp_u_p_sum = 0
        for i in valid:
            p[i] = feed_comp[i] / (1 - r_value + kb_0 * r_value * exp(u[i]))
            p_sum += p[i]
            exp_u_p_sum += exp(u[i]) * p[i]

        kb_calc = p_sum / exp_u_p_sum
        temp_calc_inv = (1 / temp_star + (log(kb_calc) - a) / b)
        temp_calc = 1 / temp_calc_inv
        liq_frac_calc = (1 - r_value) * p_sum
        vap_frac_calc = 1 - liq_frac_calc

        h_vap = c + d * (temp_calc - temp_star)
        h_liq = e + f * (temp_calc - temp_star)

        func_at_r1 = (vap_frac_calc * h_vap + liq_frac_calc * h_liq - prop_target_scaled)

        if abs(func_at_r1) < abs(func_at_r0):
            # Set to some value greater than 1
            r_value = 1.1
        else:
            # Set to some value less than 0
            r_value = -0.1

    # We converged to a vapor only solution
    if r_value > 1:
        r_value = 1
        p_sum = 0
        exp_u_p_sum = 0
        for i in valid:
            p[i] = feed_comp[i]/(1 - r_value + kb_0*r_value*exp(u[i]))
            p_sum += p[i]
            exp_u_p_sum += exp(u[i])*p[i]

        kb_calc = p_sum/exp_u_p_sum
        vap_present = True
        liq_present = False
    # We converged to liquid only solution
    elif r_value < 0:
        r_value = 0
        p_sum = 0
        exp_u_p_sum = 0
        for i in valid:
            p[i] = feed_comp[i]/(1 - r_value + kb_0*r_value*exp(u[i]))
            p_sum += p[i]
            exp_u_p_sum += exp(u[i])*p[i]

        kb_calc = p_sum/exp_u_p_sum
        vap_present = False
        liq_present = True
    else:
        vap_present = True
        liq_present = True

    if kb_calc > 0:
        temp_calc_inv = (1/temp_star + (log(kb_calc) - a)/b)
        temp_calc = 1/temp_calc_inv
        liq_frac_calc = (1 - r_value)*p_sum
        vap_frac_calc = 1 - liq_frac_calc
    else:
        temp_calc = temp

    for i in valid:
        liq_comp[i] = p[i] / p_sum
        vap_comp[i] = exp(u[i]) * p[i] / exp_u_p_sum

    # print('inner_converged:', inner_converged, 'r_value:', r_value, 'temp_calc:', temp_calc, 'vap_present:', vap_present, 'liq_present:', liq_present)
    return inner_converged, p, p_sum, r_value, vap_frac_calc, temp_calc, kb_calc, liq_comp, vap_comp, vap_present, liq_present


def flash_press_prop_2phase(provider, press, prop_type, prop_target, delta_target, feed_comp, valid=None,
                            previous=None, override_k_values=None, start_temp=None, recursive=False):
    if valid is None:
        valid = provider.all_valid_components

    prop_scaling = provider.scaling(prop_type)
    prop_target_scaled = (prop_target + delta_target)/prop_scaling

    # temp_star is a reference value
    temp_star = 298.15
    # Just a dummy value
    if previous is None:
        if start_temp is None:
            temp_est = provider.guess_nbp(feed_comp, valid)
        else:
            temp_est = start_temp

        k_values, liq_comp, vap_comp, vap_frac = generate_2phase_estimates(provider, temp_est, press, feed_comp, valid,
                                                                           override_k_values=override_k_values)
    else:
        temp_est = previous.temp
        if previous.contains('vap', 'liq'):
            override_k_values = previous.k_values_vle
            k_values, liq_comp, vap_comp, vap_frac = generate_2phase_estimates(provider, temp_est,
                                                                               press, feed_comp, valid,
                                                                               override_k_values=override_k_values)
        else:
            phase_type = list(previous.phases)[0]
            ph = previous[phase_type]
            return flash_press_prop_1phase(provider, phase_type,
                                           press,
                                           temp_est,
                                           prop_type,
                                           prop_target,
                                           delta_target,
                                           feed_comp,
                                           valid, ph)

    # Correction to make sure that temp_star is not close to temp_est
    # can occasionally cause a problem
    if abs(temp_est - temp_star)/temp_est < 0.05:
        temp_star = 0.95*temp_est

    start_temp = temp_est
    # Initialize the values of the search
    temp = temp_est
    r_value = vap_frac
    inner_converged, outer_converged = False, False
    w, u, a, b, c, d, e, f, log_kb, liq, vap = update_model_press_prop_2phase(provider,
                                                                              temp, temp_star,
                                                                              press, vap_frac,
                                                                              prop_type, prop_target, prop_scaling,
                                                                              liq_comp, vap_comp, valid,
                                                                              None, None, None, None, None, None,
                                                                              None, None, None, None,
                                                                              full_update=False)

    force_full_update = False
    error = 0
    kb_0 = exp(log_kb)
    vap_present, liq_present = False, False

    for outer_iterations in range(INSIDE_OUT_OUTER_ITERATIONS):
        # Check if the actual error is less than tolerance, if so we quit
        if outer_iterations > 1 and error < INSIDE_OUT_OUTER_TOLERANCE:
            outer_converged = True
            break

        # If we have less than 5 iterations, we do a full update
        # This helps minimize the number of iterations
        if outer_iterations < 4:
            force_full_update = True

        # Solve the inner simplified model
        inner_converged, \
        p, p_sum, r_value, vap_frac, \
        temp_calc, \
        kb_calc, \
        liq_comp_new, vap_comp_new, vap_present, liq_present = solve_model_press_prop_2phase(feed_comp, kb_0,
                                                                                             u, a, b,
                                                                                             c, d,
                                                                                             e, f, r_value,
                                                                                             temp, temp_star,
                                                                                             prop_target_scaled, valid)

        # If we hit a one phase at any point, we switch over to one phase iterations
        if outer_iterations > 2 and liq_present and not vap_present:
            # If we called recursively, ending up here is a weird situation, error out
            if recursive:
                # print('PH called recursively!')
                raise FlashConvergenceError

            # Switching to one phase search
            # print('Switching to one phase search in:', 'liq')
            return flash_press_prop_1phase(provider, 'liq',
                                           press,
                                           start_temp,
                                           prop_type,
                                           prop_target,
                                           delta_target,
                                           feed_comp,
                                           valid, liq)

        elif outer_iterations > 2 and not liq_present and vap_present:
            # If we called recursively, ending up here is a weird situation, error out
            if recursive:
                raise FlashConvergenceError

            # print('Switching to one phase search in:', 'vap')
            return flash_press_prop_1phase(provider, 'vap',
                                           press,
                                           start_temp,
                                           prop_type,
                                           prop_target,
                                           delta_target,
                                           feed_comp,
                                           valid, vap)

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
        w, u_hat, a_hat, b_hat, \
        c_hat, d_hat, \
        e_hat, f_hat, \
        log_kb_hat, liq_hat, vap_hat = update_model_press_prop_2phase(provider,
                                                                      temp_calc, temp_star,
                                                                      press, vap_frac,
                                                                      prop_type, prop_target,
                                                                      prop_scaling,
                                                                      liq_comp, vap_comp, valid,
                                                                      w, u, a, b, c, d, e, f,
                                                                      log_kb, temp,
                                                                      full_update=force_full_update)

        # Save the actual phases generated
        liq = liq_hat
        vap = vap_hat
        error = calc_error_full(u, u_hat, a, a_hat, b, b_hat, c, c_hat, d, d_hat, e, e_hat, f, f_hat, valid)
        if outer_iterations > 10:
            temp = (temp + temp_calc)*0.5
            u = (u + u_hat)*0.5
            a = (a + a_hat)*0.5
            b = (b + b_hat)*0.5
            c = (c + c_hat)*0.5
            d = (d + d_hat)*0.5
            e = (e + e_hat)*0.5
            f = (f + f_hat)*0.5
            log_kb = (log_kb + log_kb_hat)*0.5
        else:
            temp = temp_calc
            u = u_hat
            a = a_hat
            b = b_hat
            c = c_hat
            d = d_hat
            e = e_hat
            f = f_hat
            log_kb = log_kb_hat

        # print('PH#', outer_iterations, 'temp:', temp, 'vap_frac:', vap_frac, 'error:', error, 'vap_present:', vap_present, 'liq_present:', liq_present)

    if not outer_converged:
        raise FlashConvergenceError

    if outer_converged and vap_present and liq_present:
        return AggregateByMole(provider, [liq, vap], [1 - vap_frac, vap_frac])

    elif outer_converged and liq_present and not vap_present:
        # If we called recursively, ending up here is a weird situation, error out
        if recursive:
            raise FlashConvergenceError

        # Switching to one phase search
        # print('Switching to one phase search in:', LIQUID)
        return flash_press_prop_1phase(provider, 'liq',
                                       press,
                                       temp,
                                       prop_type,
                                       prop_target,
                                       delta_target,
                                       feed_comp,
                                       valid, liq)

    elif outer_converged and not liq_present and vap_present:
        # If we called recursively, ending up here is a weird situation, error out
        if recursive:
            print('PH Called Recursively')
            raise FlashConvergenceError

        # print('Switching to one phase search in:', VAPOR)
        return flash_press_prop_1phase(provider, 'vap',
                                       press,
                                       temp,
                                       prop_type,
                                       prop_target,
                                       delta_target,
                                       feed_comp,
                                       valid, vap)
    else:
        raise FlashConvergenceError


def flash_press_prop_1phase(provider, phase_type, press, start_temp, prop_type, prop_target, delta_target, feed_comp, valid, ph, recursive=False):
    prop_scaling = provider.scaling(prop_type)
    prop_target_scaled = (prop_target + delta_target)/prop_scaling


    temp_1 = start_temp
    converged = False
    phase_at_temp = None

    for iterations in range(INSIDE_OUT_OUTER_ITERATIONS):
        temp_2 = temp_1 + 0.001
        phase_at_temp = provider.phase(temp_1, press, feed_comp, phase_type, valid=valid)
        prop_1 = getattr(phase_at_temp, prop_type)/prop_scaling
        prop_2 = getattr(provider.phase(temp_2, press, feed_comp, phase_type, valid=valid), prop_type)/prop_scaling
        f = (prop_1 - prop_target_scaled)

        # print('#', iterations, 'temp:', temp_1, 'error:', f, 'phase:', phase_type, 'pseudo:', phase_at_temp.pseudo)

        if abs(f) < INSIDE_OUT_OUTER_TOLERANCE:
            converged = True
            break

        f_prime = (prop_2 - prop_target_scaled)
        df_dx = (f_prime - f)/(temp_2 - temp_1)
        if abs(df_dx) < INSIDE_OUT_INNER_TOLERANCE:
            raise FlashConvergenceError

        delta_temp = f/df_dx
        if abs(delta_temp) > 4*temp_1:
            temp_1_new = delta_temp/2

        temp_1_new = temp_1 - delta_temp
        if temp_1_new < 0:
            temp_1_new = 0.9*temp_1

        temp_1 = temp_1_new

    if not converged:
        # def phase_temp(guess_t_arg):
        #     guess_t = guess_t_arg[0]
        #     guess_phase_at_temp = provider.phase(guess_t, press, feed_comp, phase_type, valid=valid)
        #     guess_prop = getattr(guess_phase_at_temp, prop_type) / prop_scaling
        #     residual = (guess_prop - prop_target_scaled)
        #     print('Using fsolve instead, guess:', guess_t, 'residual:', residual)
        #     return [residual]
        #
        # from scipy.optimize import fsolve
        # temp_1 = fsolve(phase_temp, [temp_1])

        raise FlashConvergenceError

    temp_final = temp_1

    # If we converged, now check if we can flash into multiple phases
    results = flash_temp_press_2phase(provider, temp_final, press, feed_comp, valid)
    if len(results.phases) > 1:
        # print('One phase resulted in two phases, calling 2 phase search recursively')
        # This can happen where the flash indicates that we need to be two phases
        # in that case, we go to the nested search and hope for the best
        # print('vap_frac of tested phase', results.vap_frac_mole)
        new_k_values = results.k_values_vle

        return flash_press_prop_2phase(provider, press, prop_type, prop_target, delta_target, feed_comp, valid=valid,
                                       previous=results, start_temp=temp_final, recursive=True)

        # return nested_press_prop_2phase(provider, press, prop_type, prop_target, delta_target,
        #                                 feed_comp, new_k_values, start_temp, valid=valid)

    elif len(results.phases) == 1:
        if phase_type not in results.phases:
            if recursive:
                # Should happen only once, otherwise we have a problem
                # print('Single phase recursive PH')
                raise FlashConvergenceError
            else:
                # We converged to a false phase solution
                # Repeat with the flipped_phase
                # print('One phase resulted in false/pseudo solution, flipping phase and calling 1 phase search recursively')
                flipped_phase = {'vap': 'liq', 'liq': 'vap'}
                return flash_press_prop_1phase(provider,
                                               flipped_phase[phase_type], press, start_temp, prop_type, prop_target,
                                               delta_target,
                                               feed_comp, valid, ph, recursive=True)

    return results
