import numpy as np
from math import log, exp
from numba import njit

# Lower tolerance for stability analysis so it finishes quickly
# The flash algorithm will conclude this anyway
__STABILITY_TOLERANCE = 1e-6
__STABILITY_SUBST_ITERATIONS = 10


@njit(cache=True)
def trial_two_phases(valid, k_values, z, log_phi_feed_liq_phase, log_phi_feed_vap_phase):
    w_liq_comp = np.zeros(len(z))
    w_vap_comp = np.zeros(len(z))
    d_trial_liq, d_trial_vap = np.zeros(len(z)), np.zeros(len(z))

    for i in valid:
        w_liq_comp[i] = z[i]/k_values[i]
        w_vap_comp[i] = z[i]*k_values[i]
        d_trial_liq[i] = log(z[i]) + log_phi_feed_vap_phase[i]
        d_trial_vap[i] = log(z[i]) + log_phi_feed_liq_phase[i]

    return w_liq_comp, w_vap_comp, d_trial_liq, d_trial_vap


@njit(cache=True)
def stability_update(valid, w_liq_comp, w_vap_comp,
                     log_phi_liq_phase, log_phi_vap_phase,
                     d_trial_liq, d_trial_vap,
                     k_values_trial_liq, k_values_trial_vap,
                     feed_comp):

    tpd_trial_vap = 1
    tpd_trial_liq = 1

    for i in valid:
        tpd_trial_vap += w_vap_comp[i] * (log(w_vap_comp[i]) + log_phi_vap_phase[i] - d_trial_vap[i] - 1)
        tpd_trial_liq += w_liq_comp[i] * (log(w_liq_comp[i]) + log_phi_liq_phase[i] - d_trial_liq[i] - 1)
        w_vap_comp[i] = exp(d_trial_vap[i] - log_phi_vap_phase[i])
        w_liq_comp[i] = exp(d_trial_liq[i] - log_phi_liq_phase[i])
        k_values_trial_liq[i] = feed_comp[i] / w_liq_comp[i]
        k_values_trial_vap[i] = w_vap_comp[i] / feed_comp[i]

    return tpd_trial_liq, tpd_trial_vap


def two_phase_stability_test(provider, temp, press, composition, valid, k_values=None):
    # This method follows the stability analysis in Michelsen's text
    # We are only doing the successive substitution part of the analysis and only for STABILITY_SUBTS_ITERATIONS
    # If we don't converge in chosen iterations, then it is up to the full flash algorithm to complete flash
    if k_values is None:
        k_values = provider.guess_k_value_vle(temp, press)

    feed_comp = composition

    feed_liq_phase, feed_vap_phase = provider.phases_vle(temp, press, composition, composition,
                                                         allow_pseudo=True,
                                                         valid=valid)

    w_liq_comp, w_vap_comp, d_trial_liq, d_trial_vap = trial_two_phases(valid,
                                                                        k_values,
                                                                        feed_comp,
                                                                        feed_liq_phase.log_phi,
                                                                        feed_vap_phase.log_phi)

    tpd_trial_liq_prev, tpd_trial_vap_prev = 0, 0
    tpd_trial_liq, tpd_trial_vap = 0, 0
    vap_converged, liq_converged = False, False
    k_values_trial_liq, k_values_trial_vap = np.ones(len(k_values)), np.ones(len(k_values))
    w_liq_phase, w_vap_phase = provider.phases_vle(temp, press, w_liq_comp, w_vap_comp, allow_pseudo=True, valid=valid)

    for iter_count in range(__STABILITY_SUBST_ITERATIONS):
        log_phi_vap_phase = w_vap_phase.log_phi
        log_phi_liq_phase = w_liq_phase.log_phi
        tpd_trial_liq, tpd_trial_vap = stability_update(valid,
                                                        w_liq_comp, w_vap_comp,
                                                        log_phi_liq_phase, log_phi_vap_phase,
                                                        d_trial_liq, d_trial_vap,
                                                        k_values_trial_liq, k_values_trial_vap,
                                                        feed_comp)

        if abs(tpd_trial_liq_prev - tpd_trial_liq) < __STABILITY_TOLERANCE:
            liq_converged = True

        if abs(tpd_trial_vap_prev - tpd_trial_vap) < __STABILITY_TOLERANCE:
            vap_converged = True

        if liq_converged and vap_converged:
            # print('Converged')
            break

        tpd_trial_liq_prev = tpd_trial_liq
        tpd_trial_vap_prev = tpd_trial_vap

        if not liq_converged:
            w_liq_phase = provider.phase(temp, press, w_liq_comp, 'liq', valid=valid)

        if not vap_converged:
            w_vap_phase = provider.phase(temp, press, w_vap_comp, 'vap', valid=valid)

    converged = liq_converged and vap_converged
    liq_stable, vap_stable = False, False
    if tpd_trial_liq < 0:
        liq_stable = True

    if tpd_trial_vap < 0:
        vap_stable = True

    likely_k_values = None
    if not vap_stable and not liq_stable:
        if tpd_trial_liq < tpd_trial_vap:
            likely_k_values = k_values_trial_liq
        else:
            likely_k_values = k_values_trial_vap

    if not converged:
        # print('Stability scripts did not converge, best solution returned')
        pass

    return converged, liq_stable, vap_stable, k_values_trial_liq, k_values_trial_vap, likely_k_values
