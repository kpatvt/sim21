import sys

from sim21.provider.error import FlashConvergenceError
from sim21.provider.flash.basic import generate_2phase_estimates, basic_flash_temp_press_2phase

NESTED_ITERATIONS_MAX = 50
NESTED_PROPERTY_TOLERANCE = 1e-5


def nested_press_prop_2phase(provider, press, prop_type, prop_target, delta_target,
                             feed_comp, start_k_values, start_temp, valid=None):
    # This function fails horribly for a narrow boiling/single component mixture
    # since there is a big jump in the property
    if valid is None:
        valid = provider.all_valid_components

    prop_scaling = provider.scaling(prop_type)
    prop_target_scaled = (prop_target + delta_target)/prop_scaling

    #
    k_values, liq_comp, vap_comp, vap_frac = generate_2phase_estimates(provider, start_temp,
                                                                       press, feed_comp, valid,
                                                                       override_k_values=start_k_values)
    # Should be called in a two phase region
    if vap_frac < 0 or vap_frac > 1:
        raise AssertionError

    converged = False
    temp_guess = start_temp
    results = None
    for iteration in range(NESTED_ITERATIONS_MAX):
        results = basic_flash_temp_press_2phase(provider, temp_guess, press, feed_comp, valid,
                                                previous_k_values=start_k_values)
        if len(results.phases) > 1:
            start_k_values = results.k_values_vle

        prop = getattr(results, prop_type)/prop_scaling
        error = f_x = (prop - prop_target_scaled)
        if abs(error) < NESTED_PROPERTY_TOLERANCE:
            converged = True
            break

        temp_guess_prime = temp_guess*(1 + NESTED_PROPERTY_TOLERANCE)
        results_prime = basic_flash_temp_press_2phase(provider, temp_guess_prime, press, feed_comp, valid,
                                                      previous_k_values=start_k_values)
        prop_prime = getattr(results_prime, prop_type)/prop_scaling

        df_dx = (prop_prime - prop)/(temp_guess_prime - temp_guess)
        if abs(df_dx) < sys.float_info.epsilon:
            raise FlashConvergenceError

        temp_guess_new = temp_guess - f_x/df_dx
        if temp_guess_new < 0.9*temp_guess:
            temp_guess_new = 0.9*temp_guess

        elif temp_guess_new > 1.1*temp_guess:
            temp_guess_new = 1.1*temp_guess

        temp_guess = temp_guess_new

    if not converged:
        raise FlashConvergenceError

    return results


