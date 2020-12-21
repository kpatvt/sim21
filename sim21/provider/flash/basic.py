from numba import njit
import numpy as np
from math import exp

from sim21.provider.agg import AggregateByMole
from sim21.provider.error import FlashConvergenceError
from sim21.provider.flash.rachford_rice import possible_rr_2phase, solve_rr_2phase, dew_point_rr_2phase
from sim21.provider.flash.rachford_rice import bubble_point_rr_2phase
from sim21.provider.flash.stability import two_phase_stability_test

__BASIC_TEMP_PRESS_FLASH_ITERATIONS = 30
__BASIC_TEMP_PRESS_TOLERANCE = 1e-6


@njit(cache=True)
def update_two_phase_k_values(old_k, liq_log_phi, vap_log_phi, valid_comps):
    new_k = np.ones(len(old_k))
    dev = 0
    for i in valid_comps:
        new_k[i] = exp(liq_log_phi[i] - vap_log_phi[i])
        dev += max(abs(old_k[i] - new_k[i]), dev)

    return new_k, dev


def basic_flash_temp_press_2phase(provider, temp, press, feed_comp, valid, previous_k_values=None):
    # if valid is None
    if valid is None:
        valid = provider.all_valid_components

    # If no K-value estimate is provided, use the k-value correlation to provide an estimate
    k_values = None
    if previous_k_values is None:
        k_values = provider.guess_k_value_vle(temp, press)
    else:
        k_values = previous_k_values

    # Using the k-value estimate, determine if we will get a solution in the two phase region
    # If no, do a stability scripts to identify if the phase is stable. If the phase is stable, we are done.
    # If phase is not stable, use the K-value estimate generated by the stability analysis as a K-Value estimate
    vapor_present, liquid_present = possible_rr_2phase(k_values, feed_comp, valid)
    if vapor_present is False or liquid_present is False:
        # print('Doing stability check')
        converged, \
        liq_stable, vap_stable, \
        k_values_trial_liq, \
        k_values_trial_vap, \
        likely_k_values = two_phase_stability_test(provider, temp, press, feed_comp, valid)

        # If liq_stable and vapor_stable both are False
        # then the feed is not stable as either a vapor a or a liquid
        # And use likely_k_values as the initial estimate
        if not liq_stable and not vap_stable:
            # print('Unstable phase found, new k-values generated...')
            k_values = likely_k_values
        elif liq_stable:
            # print('Liquid Stable')
            ph = provider.phase(temp, press, feed_comp, 'liq')
            # Sometimes we get a flip on vapor and liquid, check if the liquid is a pseudo phase
            if ph.pseudo is True:
                ph = provider.phase(temp, press, feed_comp, 'vap')

            return AggregateByMole(provider, [ph], [1])
            # Stable as a liquid
            pass
        elif vap_stable:
            # print('Vapor Stable')
            ph = provider.phase(temp, press, feed_comp, 'vap')
            if ph.pseudo is True:
                ph = provider.phase(temp, press, feed_comp, 'liq')

            return AggregateByMole(provider, [ph], [1])
            # Stable as a vapor
            pass
    else:
        # Start the RR calculation directly
        pass

    # Converge the RR calculation, allowing for negative flash
    # For the first iteration, set to -1 to allow solve_rr_2phase to set beta_guess
    beta_guess = -1
    liq_phase, vap_phase = None, None
    liq_comp, vap_comp = None, None

    deviation = 1
    fully_converged = False
    invalid_solution_count = 0

    for flash_iteration in range(__BASIC_TEMP_PRESS_FLASH_ITERATIONS):
        if deviation < __BASIC_TEMP_PRESS_TOLERANCE:
            fully_converged = True
            break

        vapor_present, liquid_present = possible_rr_2phase(k_values, feed_comp, valid)

        if vapor_present and liquid_present:
            rr_converged, beta_guess, liq_comp, vap_comp = solve_rr_2phase(k_values, feed_comp, valid, beta_guess)

        elif vapor_present or liquid_present:
            invalid_solution_count += 1
            # print('Invalid solution found, tries remaining:', 5 - invalid_solution_count)
            if vapor_present:
                liq_comp, vap_comp = dew_point_rr_2phase(k_values, feed_comp, valid)
            else:
                liq_comp, vap_comp = bubble_point_rr_2phase(k_values, feed_comp, valid)

            if invalid_solution_count >= 5:
                # print('Breaking out, likely single phase')
                break

        # Get the new_values
        liq_phase, vap_phase = provider.phases_vle(temp, press, liq_comp, vap_comp)
        new_k_values, deviation = update_two_phase_k_values(k_values, liq_phase.log_phi, vap_phase.log_phi, valid)

        # For now we are just doing successive substitution
        # This is where we could accelerate...

        # Update the k-values from the provider and then repeat the RR caluation
        # until K-values converge or iterations run out
        k_values = new_k_values
        # print('temp:', temp, 'press:', press, 'beta:', beta_guess, 'k-values:', k_values)

    # We have broken out of the loop, but check if we are converged or forcing a stability scripts
    # We can break out to force a stability scripts or because we didn't converge
    # Check if we converged or not, happens close to the critical point;
    # Eventually shift to a full Newton approach here
    # For now we error out
    if not fully_converged and vapor_present and liquid_present:
        # If iterations run out and provider supports log_phi derivatives, use to switch over to a full Newton approach
        raise FlashConvergenceError

    # if K-values converge, we have a valid solution
    if liquid_present and not vapor_present:
        # We are pretty much one-phase if this occurs
        # Return solution, corresponding to the identified state
        # print('Liquid Solution')
        ph = provider.phase(temp, press, feed_comp, 'liq')
        # Sometimes we get a flip on vapor and liquid, check if the liquid is a pseudo phase
        if ph.pseudo is True:
            ph = provider.phase(temp, press, feed_comp, 'vap')

        return AggregateByMole(provider,[ph], [1])

    elif not liquid_present and vapor_present:
        # print('Vapor Solution')
        ph = provider.phase(temp, press, feed_comp, 'vap')
        if ph.pseudo is True:
            ph = provider.phase(temp, press, feed_comp, 'liq')

        return AggregateByMole(provider,[ph], [1])

    elif fully_converged and vapor_present and liquid_present:
        # print('Mixed')
        return AggregateByMole(provider,[liq_phase, vap_phase], [1 - beta_guess, beta_guess])
    else:
        raise FlashConvergenceError


def generate_2phase_results(provider, temp, press, liq_comp, vap_comp, valid):
    liq, vap = provider.phases_vle(temp, press, liq_comp, vap_comp, allow_pseudo=True, valid=valid)
    liq_log_phi = liq.log_phi
    vap_log_phi = vap.log_phi
    return liq_log_phi - vap_log_phi, liq, vap


def generate_2phase_estimates(provider, temp, press, feed_comp, valid, override_k_values=None):
    if override_k_values is None:
        k_values = provider.guess_k_value_vle(temp, press)
    else:
        k_values = override_k_values

    vapor_present, liquid_present = possible_rr_2phase(k_values, feed_comp, valid)
    if vapor_present and liquid_present:
        converged, beta, liq_comp, vap_comp = solve_rr_2phase(k_values, feed_comp, valid)
    elif vapor_present:
        liq_comp, vap_comp = dew_point_rr_2phase(k_values, feed_comp, valid)
        beta = 1
    else:
        liq_comp, vap_comp = bubble_point_rr_2phase(k_values, feed_comp, valid)
        beta = 0

    return k_values, liq_comp, vap_comp, beta
