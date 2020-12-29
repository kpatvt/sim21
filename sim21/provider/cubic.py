import math

import numpy as np
from numba import njit

from sim21.data import chemsep
from sim21.data.chemsep_consts import GAS_CONSTANT
from sim21.provider.flash.basic import basic_flash_temp_press_2phase
from sim21.provider.flash.io import flash_press_prop_2phase, flash_press_vap_frac_2phase, flash_temp_vap_frac_2phase
from sim21.support.roots import solve_cubic_reals, mid
from sim21.provider.generic import press_derivs, log_phi_derivs, residual_derivs, calc_ig_props
from sim21.provider.phase import PhaseByMole
from sim21.provider.base import Provider, MIN_COMPOSITION

# The flash routines
# from ypsim.thermo.flash.basic import basic_flash_temp_press_2phase
# from ypsim.thermo.flash.io import flash_press_prop_2phase
# from .flash.basic import basic_flash_temp_press_2phase
# from .flash.io import flash_press_vap_frac_2phase
# from .flash.io import flash_temp_vap_frac_2phase
# from .flash.io import flash_press_prop_2phase
# from .flash.io import flash_temp_prop_2phase
# from .flash.io import flash_prop_vap_frac_2phase

SQRT_2 = math.sqrt(2)


@njit(cache=True)
def parameters(eos, n_count, valid_comps, gas_const, crit_temp, crit_press, omega, temp):
    """
    Calculate pure component parameters for each component for a given eos
    Currently only SRK and PR with standard alpha functions are supported
    Returns success, a, da_dT, d2a_dT2, b (sucess indicates whether call was successful
    """
    a = np.zeros(n_count)
    da_dT = np.zeros(n_count)
    d2a_dT2 = np.zeros(n_count)
    b = np.zeros(n_count)

    if eos == 0:
        c1, c2 = 0.427480, 0.086640
        kappa_coeff1, kappa_coeff2, kappa_coeff3 = 0.48508, 1.55171, -0.15613
    elif eos == 1:
        c1, c2 = 0.45724, 0.07780
        kappa_coeff1, kappa_coeff2, kappa_coeff3 = 0.37464, 1.54226, -0.26992
    else:
        return False, a, da_dT, d2a_dT2, b

    # Compute
    for i in valid_comps:
        tau = 1 - math.sqrt(temp / crit_temp[i])
        kappa = kappa_coeff1 + kappa_coeff2 * omega[i] + kappa_coeff3 * (omega[i] ** 2)
        alpha = (1 + kappa * tau) ** 2
        d_alpha_dT = -kappa * (1 - tau) * (kappa * tau + 1) / temp
        d2_alpha_dT2 = kappa ** 2 * (temp / crit_temp[i]) / (2 * temp ** 2) + kappa * (1 - tau) * (kappa * tau + 1) / (
            2 * temp ** 2)

        a[i] = c1 * (((gas_const * crit_temp[i]) ** 2) / crit_press[i]) * alpha
        da_dT[i] = c1 * (((gas_const * crit_temp[i]) ** 2) / crit_press[i]) * d_alpha_dT
        d2a_dT2[i] = c1 * (((gas_const * crit_temp[i]) ** 2) / crit_press[i]) * d2_alpha_dT2
        b[i] = c2 * gas_const * crit_temp[i] / crit_press[i]

    return True, a, da_dT, d2a_dT2, b


@njit(cache=True)
def mixture(eos, n, n_count, n_sum, valid_comps, a, da_dT, d2a_dT2, b, ip_k, ip_l):
    """
    Implements mixing rule for a; a is quadratic, b is linear
    Supports only a k_ij binary interaction parameter
    Returns a_mix, da_mix_dT, d2a_mix_dT2, b_mix
    """
    D = np.zeros(n_count)
    D_prime = np.zeros((n_count, n_count))
    D_T = np.zeros(n_count)
    B = np.zeros(n_count)
    B_prime = np.zeros((n_count, n_count))

    D_sum = 0
    D_T_sum = 0
    D_T_T_sum = 0
    nB_sum = 0

    for i in valid_comps:
        D[i] = 0
        D_T[i] = 0
        for j in valid_comps:
            a_ij = math.sqrt(a[i] * a[j]) * (1 - ip_k[i, j])
            da_ij_dT = math.sqrt(a[i] * a[j]) * (1 - ip_k[i, j]) * (a[i] * da_dT[j] / 2 + a[j] * da_dT[i] / 2) / (
                a[i] * a[j])
            d2a_ij_dT2 = (a[i] * a[j]) ** (9.0 / 2.0) * (
                (1 - ip_k[i, j]) * (a[i] * da_dT[j] + a[j] * da_dT[i]) ** 2 + 2 * (ip_k[i, j] - 1) * (
                a[i] * da_dT[j] + a[j] * da_dT[i]) * a[i] * da_dT[j] + 2 * (ip_k[i, j] - 1) * (
                    a[i] * da_dT[j] + a[j] * da_dT[i]) * a[j] * da_dT[i] - 2 * (ip_k[i, j] - 1) * (
                    a[i] * d2a_dT2[j] + a[j] * d2a_dT2[i] + 2 * da_dT[i] * da_dT[j]) * a[i] * a[j]) / (
                             4 * a[i] ** 6 * a[j] ** 6)

            D_prime[i, j] = 2 * a_ij
            D[i] += 2 * n[j] * a_ij
            D_T[i] += 2 * n[j] * da_ij_dT
            D_T_T_sum += n[i] * n[j] * d2a_ij_dT2

            b_ij = 0.5 * (b[i] + b[j]) * (1 - ip_l[i, j])
            nB_sum += n[i] * n[j] * b_ij
            B[i] += 2 * n[j] * b_ij

        D_sum += 0.5 * n[i] * D[i]
        D_T_sum += 0.5 * n[i] * D_T[i]

    B_sum = nB_sum / n_sum
    for i in valid_comps:
        B[i] = (B[i] - B_sum) / n_sum

    for i in valid_comps:
        for j in valid_comps:
            b_ij = 0.5 * (b[i] + b[j]) * (1 - ip_l[i, j])
            B_prime[i, j] = (2 * b_ij - B[i] - B[j]) / n_sum

    return D_sum, D, D_prime, D_T_sum, D_T, D_T_T_sum, B_sum, B, B_prime


@njit(cache=True)
def volume(eos, n_sum, gas_const, temp, press, desired_phase, D_sum, B_sum):
    """
    Calculates volume for the desired phase using the eos, parameters
    If desired_phase not does match the desired phase and pseudo is True, a pseudo volume representing that
    phase is calculated
    """
    R = gas_const
    a_mix = D_sum
    b_mix = B_sum

    # Express cubic eos in Z form
    coeff_A = (a_mix / (n_sum ** 2)) * press / ((R * temp) ** 2)
    coeff_B = (b_mix / n_sum) * (press / (R * temp))
    if eos == 0:
        delta_1, delta_2 = 1, 0
    elif eos == 1:
        delta_1, delta_2 = 1 + SQRT_2, 1 - SQRT_2
    else:
        return False, 0, 0

    c1 = 1
    c2 = -(1 - coeff_B * (delta_1 + delta_2 - 1))
    c3 = -(delta_1 + delta_2 - coeff_B * (delta_1 * delta_2 - delta_1 - delta_2) - coeff_A / coeff_B) * coeff_B
    c4 = -(coeff_A + coeff_B * (1 + coeff_B) * delta_1 * delta_2) * coeff_B

    # This is ugly, but needs to be this way to work around some bugs in numba
    # All we are doing is filtering the Zs where the corresponding volume is less than B_sum
    all_roots = solve_cubic_reals(c1, c2, c3, c4)
    valid_roots = np.zeros(3)
    valid_root_count = 0
    for i in range(len(all_roots)):
        test_z = all_roots[i]
        if test_z * n_sum * R * temp / press > B_sum:
            valid_roots[valid_root_count] = test_z
            valid_root_count += 1

    if valid_root_count == 0:
        return False, 0, 0
    else:
        all_Z = np.zeros(valid_root_count)
        for i in range(valid_root_count):
            all_Z[i] = valid_roots[i]

    compress_Z = 0
    if desired_phase == 'vap':
        compress_Z = all_Z[-1]
        calc_vol = compress_Z * n_sum * R * temp / press
    elif desired_phase == 'liq':
        compress_Z = all_Z[0]
        calc_vol = compress_Z * n_sum * R * temp / press
    else:
        # raise NotImplementedError
        return False, 0, 0

    return True, calc_vol, compress_Z


@njit(cache=True)
def derivs_primary(eos, n_count, n_sum, valid_comps,
                   gas_const, temp, vol,
                   B, B_sum, D, D_sum, D_T_sum, D_T_T_sum,
                   wrt_composition=True):
    _f, _f_V, _f_V_V, _f_B_V, _f_B, _f_B_B, = 0, 0, 0, 0, 0, 0
    F, F_n, F_n_V, F_n_B = 0, 0, 0, 0
    F_T, F_T_T, F_T_V = 0, 0, 0
    F_V, F_V_V = 0, 0
    F_B, F_B_T, F_B_V, F_B_B, F_B_D = 0, 0, 0, 0, 0
    F_D, F_D_V, F_D_T = 0, 0, 0
    g, g_V, g_V_V, g_B, g_B_V, g_B_B = 0, 0, 0, 0, 0, 0
    del_F_del_T_V_n, del_F_del_V_T_n = 0, 0
    del_F_del_n_T_V = np.zeros(n_count)
    del_2F_del_T2_V_n, del_2F_del_T_del_V_n, del_2F_del_V2_T_n = 0, 0, 0

    if eos == 0:
        delta_1, delta_2 = 1, 0
    elif eos == 1:
        delta_1, delta_2 = 1 + SQRT_2, 1 - SQRT_2
    else:
        return _f, _f_V, _f_V_V, _f_B_V, _f_B, _f_B_B, \
               F, F_n, F_n_V, F_n_B, \
               F_T, F_T_T, F_T_V, \
               F_V, F_V_V, \
               F_B, F_B_T, F_B_V, F_B_B, F_B_D, \
               F_D, F_D_V, F_D_T, \
               g, g_V, g_V_V, g_B, g_B_V, g_B_B, \
               del_F_del_n_T_V, del_F_del_T_V_n, del_F_del_V_T_n, \
               del_2F_del_T2_V_n, del_2F_del_T_del_V_n, del_2F_del_V2_T_n

    R = gas_const
    # print('vol in primary derivs:', vol)
    _f = 1 / (R * B_sum * (delta_1 - delta_2)) * math.log((vol + delta_1 * B_sum) / (vol + delta_2 * B_sum))
    g = math.log(vol - B_sum) - math.log(vol)
    F = -n_sum * g - D_sum / temp * _f

    # print('F:', F)

    g_V = B_sum / (vol * (vol - B_sum))
    g_B = -1 / (vol - B_sum)

    _f_V = -1 / (R * (vol + delta_1 * B_sum) * (vol + delta_2 * B_sum))
    _f_B = -(_f + vol * _f_V) / B_sum

    F_n = -g
    F_T = D_sum / (temp ** 2) * _f
    F_V = -n_sum * g_V - (D_sum / temp) * _f_V
    F_B = -n_sum * g_B - (D_sum / temp) * _f_B
    F_D = -_f / temp

    if wrt_composition:
        for i in valid_comps:
            del_F_del_n_T_V[i] = F_n + F_B * B[i] + F_D * D[i]

    del_F_del_T_V_n = F_T + F_D * D_T_sum
    del_F_del_V_T_n = F_V

    _f_V_V = 1 / (R * B_sum * (delta_1 - delta_2)) * (
        -1 / ((vol + delta_1 * B_sum) ** 2) + 1 / ((vol + delta_2 * B_sum) ** 2))
    _f_B_V = -(2 * _f_V + vol * _f_V_V) / B_sum
    _f_B_B = -(2 * _f_B + vol * _f_B_V) / B_sum

    g_V_V = -1 / ((vol - B_sum) ** 2) + 1 / (vol ** 2)
    g_B_V = 1 / ((vol - B_sum) ** 2)
    g_B_B = -1 / ((vol - B_sum) ** 2)

    F_n_V = -g_V
    F_n_B = -g_B
    F_T_T = -2 * F_T / temp
    F_B_T = D_sum * _f_B / (temp ** 2)
    F_D_T = _f / (temp ** 2)
    F_B_V = -n_sum * g_B_V - (D_sum / temp) * _f_B_V
    F_B_B = -n_sum * g_B_B - (D_sum / temp) * _f_B_B
    F_D_V = -_f_V / temp
    F_B_D = -_f_B / temp
    F_T_V = D_sum / (temp ** 2) * _f_V
    F_V_V = -n_sum * g_V_V - (D_sum / temp) * _f_V_V

    del_2F_del_T2_V_n = F_T_T + 2 * F_D_T * D_T_sum + F_D * D_T_T_sum
    del_2F_del_T_del_V_n = F_T_V + F_D_V * D_T_sum
    del_2F_del_V2_T_n = F_V_V

    return _f, _f_V, _f_V_V, _f_B_V, _f_B, _f_B_B, \
           F, F_n, F_n_V, F_n_B, \
           F_T, F_T_T, F_T_V, \
           F_V, F_V_V, \
           F_B, F_B_T, F_B_V, F_B_B, F_B_D, \
           F_D, F_D_V, F_D_T, \
           g, g_V, g_V_V, g_B, g_B_V, g_B_B, \
           del_F_del_n_T_V, del_F_del_T_V_n, del_F_del_V_T_n, \
           del_2F_del_T2_V_n, del_2F_del_T_del_V_n, del_2F_del_V2_T_n


@njit(cache=True)
def derivs_secondary(n_count, valid_comps,
                     B, B_prime, D, D_prime, D_T, D_T_sum,
                     _f, _f_V, _f_V_V, _f_B_V, _f_B, _f_B_B,
                     F, F_n_V, F_n_B,
                     F_B, F_B_T, F_B_V, F_B_B, F_B_D,
                     F_D, F_D_V, F_D_T,
                     second_order=False):
    del_2F_del_V_del_n_T = np.zeros(n_count)
    del_2F_del_n_del_T_V = np.zeros(n_count)
    del_2F_del_n_del_n_T_V = np.zeros((n_count, n_count))

    for i in valid_comps:
        del_2F_del_n_del_T_V[i] = (F_B_T + F_B_D * D_T_sum) * B[i] + F_D_T * D[i] + F_D * D_T[i]
        del_2F_del_V_del_n_T[i] = F_n_V + F_B_V * B[i] + F_D_V * D[i]

        if second_order:
            for j in valid_comps:
                term1 = F_n_B * (B[i] + B[j])
                term2 = F_B_D * (B[i] * D[j] + B[j] * D[i])
                term3 = F_B * B_prime[i, j] + F_B_B * B[i] * B[j] + F_D * D_prime[i, j]
                del_2F_del_n_del_n_T_V[i, j] = term1 + term2 + term3

    return del_2F_del_V_del_n_T, del_2F_del_n_del_T_V, del_2F_del_n_del_n_T_V


@njit(cache=True)
def identify_phase(eos, r, a_mix, da_dtemp_mix, b_mix, temp, vol, press):
    """
    Identify a phase given the derivative properties
    Implemented only for SRK and PR Eos

    The Poling criteria is fully described by Watson (2018), see earlier references.
    The Venkatarathnam method is used as a fall back method, since it is not consistent with the pseudo property method

    Implemented as described in "Identification of the phase of a fluid using partial derivatives of pressure, volume,
        and temperature without reference to saturation properties: Applications in phase equilibria calculations" by
        G. Venkatarathnam et. al. (2011)
    """
    # Primary method
    use_poling_method = True
    # Should be False, only used as a fallback
    use_venkatarathnam_method = False
    result = ''

    if eos == 0:
        delta_1, delta_2 = 1, 0
    elif eos == 1:
        delta_1, delta_2 = 1 + SQRT_2, 1 - SQRT_2
    else:
        return ''

    g1 = 1 / (vol - b_mix)
    g2 = 1 / (vol + delta_1 * b_mix)
    g3 = 1 / (vol + delta_2 * b_mix)
    g4 = g2 + g3
    g5 = da_dtemp_mix
    g6 = g2 * g3
    del_2_press_del_vol_del_temp = -r * (g1 ** 2) + g4 * g5 * g6
    del_press_del_temp_vol = r * g1 - g5 * g6
    del_2_press_del_vol_2_temp = 2 * r * temp * (g1 ** 3) - 2 * a_mix * (g2 ** 2 + g6 + g3 ** 2)
    del_press_del_vol_temp = -r * temp * (g1 ** 2) + a_mix * g4 * g6
    del_vol_del_press_temp = 1 / del_press_del_vol_temp
    beta = (-1 / vol) * del_vol_del_press_temp

    if use_poling_method:
        if beta < (0.005 / 101325.0):
            result = 'liq'
        elif 0.9 / press < beta < 3 / press:
            result = 'vap'
        else:
            # use_venkatarathnam_method = True
            result = 'liq'

    elif use_venkatarathnam_method:
        p1 = del_2_press_del_vol_del_temp / del_press_del_temp_vol
        p2 = del_2_press_del_vol_2_temp / del_press_del_vol_temp
        phase_id_parameter = vol * (p1 - p2)

        # temp_pi is used based on a vdw estimate by given in the paper
        # To be exactly correct, temp_pi must be searched for, but this requires iteration
        # along temperature for the eos, which can be quite intensive
        temp_pi = 2 * a_mix * ((vol - b_mix) ** 2) / (r * b_mix * (vol ** 2))
        if temp > temp_pi:
            if phase_id_parameter > 1:
                result = 'vap'
            else:
                result = 'liq'
        else:
            result = 'liq'

        if phase_id_parameter <= 1:
            result = 'vap'  # VAPOR

    return result


@njit(cache=True)
def pseudo_root_search_mathias(eos, r, temp, a_mix, b_mix,
                               desired_phase,
                               kappa,
                               search_iterations=10):
    """
    Solves the Mathias constraint given by d_press_d_rho - 0.1*r*temp == 0
    for SRK and PR equations of state. This method is technically independent of the EOS
    provided the relevant derivatives exist.

    Input parameters are the eos, gas_constant as r, temp, a_mix and b_mix which are
    from the eos mixture calculation.

    The desired phase is either 'vap' or 'liq' indicated a vapor like phase or a liquid like phase
    to be returned. This setting also determines the interval for the root search

    search iterations is how long newton search continues before terminating, default is 10.
    Search should conclude in no more than 4-5 iterations.
    """
    # 0 is SRK
    # 1 is PR
    # Bounds for the rho for a given eos

    # This is method is independent of the actual EOS,
    # But need rho_lo, rho_hi, rho_mc, temp_mc and dpress_drho and d2press_drho2 for each each of equation
    # Only SRK and PR are implemented for now.

    # Only the rho changes for these equations, so no new mixing terms need to be calculated
    # Should converge very quickly

    converged = False
    if eos == 0:
        u, w = 1, 0
        rho_lo, rho_hi = -1 / b_mix, 1 / b_mix
        # From original Mathias paper
        rho_mc = 0.25599 / b_mix
        temp_mc = 0.20268 * a_mix / (r * b_mix)
    elif eos == 1:
        u, w = 2, -1
        rho_lo, rho_hi = (1 - SQRT_2) / b_mix, 1 / b_mix
        # From Watson
        rho_mc = 0.25308 / b_mix
        temp_mc = 0.17014 * a_mix / (r * b_mix)
    else:
        return False, -1, -1, -1, -1, -1

    if desired_phase == 'vap':
        rho_interval_lo, rho_interval_hi = rho_lo, kappa * rho_mc
    elif desired_phase == 'liq':
        rho_interval_lo, rho_interval_hi = rho_mc, rho_hi
    else:
        return False, -1, -1, -1, -1, -1

    scaling = 1 / (r * temp)
    # scaling = 1

    if desired_phase == 'liq':
        # initial_estimate - given by Mathias
        rho_test = rho_hi - 0.4 * (rho_hi - rho_mc)
    else:
        rho_test = (rho_interval_lo + rho_interval_hi) * 0.5
        # rho_test = rho_hi - 0.4*(rho_hi - rho_mc)

    for j in range(search_iterations):
        # EOS in terms of rho (which 1/vol)
        # press = r*temp/(-b_mix + 1/rho) - a_mix/(w*b_mix**2 + u*b_mix/rho + rho**(-2))

        # Derivative of the EOS in terms of rho_test
        d_press_d_rho = r * temp / (rho_test ** 2 * (-b_mix + 1 / rho_test) ** 2) - (
            u * b_mix / rho_test ** 2 + 2 / rho_test ** 3) * a_mix / (
                            w * b_mix ** 2 + u * b_mix / rho_test + rho_test ** (-2)) ** 2
        f = (d_press_d_rho - 0.1 * r * temp)
        f *= scaling
        if f < 1e-6:
            converged = True
            break

        # 2nd Derivative of the EOS in terms of rho root
        d2_press_d_rho_2 = 2 * (
            -r * temp / (b_mix - 1 / rho_test) ** 2 - r * temp / (rho_test * (b_mix - 1 / rho_test) ** 3) + (
            u * b_mix + 3 / rho_test) * a_mix / (
                w * b_mix ** 2 + u * b_mix / rho_test + rho_test ** (-2)) ** 2 - (
                u * b_mix + 2 / rho_test) ** 2 * a_mix / (
                rho_test * (w * b_mix ** 2 + u * b_mix / rho_test + rho_test ** (-2)) ** 3)) / rho_test ** 3
        d2_press_d_rho_2 *= scaling

        df_drho = d2_press_d_rho_2
        rho_test_new = rho_test - f / df_drho
        if rho_test_new < rho_interval_lo:
            rho_test_new = (rho_test + rho_interval_lo) / 2
        elif rho_test_new > rho_interval_hi:
            rho_test_new = (rho_test + rho_interval_hi) / 2

        rho_test = rho_test_new

    # if not converged:
    #     print('press_rho did not converge')

    return converged, rho_mc, rho_lo, rho_hi, rho_test, temp_mc


@njit(cache=True)
def pseudo_volume_watson(eos, r, temp, press_eos, rho_eos, a_mix, b_mix, desired_phase):
    """
    Calculates a pseudo volume based on the algorithm described by Watson (2018)
    in thesis "Robust Simulation and Optimization Methods for Natural Gas Liquefaction Processes"
    Available at https://dspace.mit.edu/handle/1721.1/115702
    """
    if eos == 0:
        u, w = 1, 0
    elif eos == 1:
        u, w = 2, -1
    else:
        return '', 0, 0

    # kappa is a tuning parameter whose details are given by Watson.
    # Remains untouched for most cases
    kappa = 0.9

    solution_found, rho_mc, rho_lo, rho_hi, rho_omega, temp_mc = pseudo_root_search_mathias(eos, r,
                                                                                            temp, a_mix, b_mix,
                                                                                            desired_phase, kappa)
    if desired_phase == 'liq':
        rho_L_omega = rho_omega
        if not solution_found:
            rho_L_star = rho_mc
        else:
            rho_L_star = mid(rho_mc, rho_L_omega, rho_hi)

        rho_test = rho_L_star
        press_star = r * temp / (-b_mix + 1 / rho_test) - a_mix / (
            w * b_mix ** 2 + u * b_mix / rho_test + rho_test ** (-2))
        d_press_d_rho_star = r * temp / (rho_test ** 2 * (-b_mix + 1 / rho_test) ** 2) - (
            u * b_mix / rho_test ** 2 + 2 / rho_test ** 3) * a_mix / (
                                 w * b_mix ** 2 + u * b_mix / rho_test + rho_test ** (-2)) ** 2

        B_L = d_press_d_rho_star * (rho_L_star - 0.7 * rho_mc)
        A_L = (press_star - B_L * math.log(rho_L_star - 0.7 * rho_mc))

        rho_L_extrap = min(math.exp((press_eos - A_L) / B_L) + 0.7 * rho_mc, rho_hi)
        rho_L = mid(rho_eos, rho_L_star, rho_L_extrap)

        rho_test = rho_L
        press_calc = r * temp / (-b_mix + 1 / rho_test) - a_mix / (
            w * b_mix ** 2 + u * b_mix / rho_test + rho_test ** (-2))

        return desired_phase, 1 / rho_L, abs(press_calc)

    elif desired_phase == 'vap':
        rho_V_omega = rho_omega
        if not solution_found:
            rho_V_star = kappa * rho_mc
        else:
            rho_V_star = mid(rho_lo, rho_V_omega, kappa * rho_mc)

        rho_V_bound = mid(rho_lo, rho_V_omega, kappa * rho_mc)

        rho_test = rho_V_star
        press_star = r * temp / (-b_mix + 1 / rho_test) - a_mix / (
            w * b_mix ** 2 + u * b_mix / rho_test + rho_test ** (-2))
        # Derivative of the EOS in terms of rho_test
        d_press_d_rho_star = r * temp / (rho_test ** 2 * (-b_mix + 1 / rho_test) ** 2) - (
            u * b_mix / rho_test ** 2 + 2 / rho_test ** 3) * a_mix / (
                                 w * b_mix ** 2 + u * b_mix / rho_test + rho_test ** (-2)) ** 2

        A_V = 1 / press_star
        B_V = -d_press_d_rho_star / (press_star ** 2)
        C_V = -abs(A_V + 0.5 * B_V * (rho_mc - rho_V_star)) / ((0.5 * (rho_mc - rho_V_star)) ** 2)

        term2 = (-B_V - math.sqrt(B_V ** 2 - 4 * C_V * max(0, (A_V - 1 / press_eos)))) / (2 * C_V)

        rho_test = rho_V_omega
        d_press_d_rho_omega = r * temp / (rho_test ** 2 * (-b_mix + 1 / rho_test) ** 2) - (
            u * b_mix / rho_test ** 2 + 2 / rho_test ** 3) * a_mix / (
                                  w * b_mix ** 2 + u * b_mix / rho_test + rho_test ** (-2)) ** 2
        term3 = min(0, press_eos - press_star) / d_press_d_rho_omega + term2 + max(0, temp - temp_mc) * max(0,
                                                                                                            d_press_d_rho_star - d_press_d_rho_omega)
        rho_V_extrap = mid(0, rho_hi, rho_V_bound + term3)

        rho_V = mid(rho_eos, rho_V_star, rho_V_extrap)

        # Do we need to correct the vapor fugacity coefficients?
        # rho_test = rho_V
        # press_calc = r*temp/(-b_mix + 1/rho_test) - a_mix/(w*b_mix**2 + u*b_mix/rho_test + rho_test**(-2))

        return desired_phase, 1 / rho_V, press_eos
    else:
        return '', 0, 0


@njit(cache=True)
def calc_phase(eos, gas_const, temp, press, n, valid_comps, desired_phase, allow_pseudo,
               mw_list, crit_temp, crit_press, omega, k_ij, l_ij,
               ig_temp_ref, ig_press_ref, ig_cp_coeffs, ig_h_form, ig_s_form,
               press_comp_derivs, log_phi_temp_press_derivs, log_phi_comp_derivs):
    # is_pseudo = False
    log_press_factor = 0
    press_eos = press

    # Check if we need composition derivatives
    comp_derivs = press_comp_derivs or log_phi_comp_derivs

    # Do a quick sum, skipping invalid elements, njit will accelerate this away
    n_count = len(n)
    n_sum = 0
    for i in valid_comps:
        n_sum += n[i]

    # Get the single component parameters for a given temperature
    # This can be double computed if multiple phases are requested; not a big deal
    _, a, da_dT, d2a_dT2, b = parameters(eos, n_count, valid_comps, gas_const, crit_temp, crit_press, omega, temp)

    # Calculate the mixture parameters
    D_sum, D, D_prime, D_T_sum, D_T, D_T_T_sum, B_sum, B, B_prime = mixture(eos, n, n_count, n_sum, valid_comps,
                                                                            a, da_dT, d2a_dT2, b,
                                                                            k_ij, l_ij)

    # Now let's compute the volume corresponding to the desired phase
    success = False
    success, vol, compress_Z = volume(eos, n_sum, gas_const, temp, press_eos, desired_phase, D_sum, B_sum)

    # now figure out what that this phase is, according to Poling criteria
    a_mix = D_sum / (n_sum ** 2)
    da_dtemp_mix = D_T_sum / (n_sum ** 2)
    b_mix = B_sum / n_sum
    eventual_phase_id = identify_phase(eos, gas_const, a_mix, da_dtemp_mix, b_mix, temp, vol / n_sum, press_eos)

    # Check if the desired phase matches what we are looking for ...
    # If yes, there's nothing more to do
    if eventual_phase_id == desired_phase:
        is_pseudo = False

    # If no, then we should calculate a pseudo phase if that is allowed by the caller
    elif allow_pseudo is True:
        rho_eos = 1 / (vol / n_sum)
        eventual_phase_id, vol_calc, press_correction = pseudo_volume_watson(eos, gas_const,
                                                                             temp, press_eos, rho_eos,
                                                                             a_mix, b_mix, desired_phase)

        vol = vol_calc * n_sum
        compress_Z = (press_eos * vol) / (n_sum * gas_const * temp)
        log_press_factor = math.log(press_correction / press_eos)
        is_pseudo = True
    else:
        is_pseudo = False
        raise NotImplementedError

    _f, _f_V, _f_V_V, _f_B_V, _f_B, _f_B_B, \
    F, F_n, F_n_V, F_n_B, \
    F_T, F_T_T, F_T_V, \
    F_V, F_V_V, \
    F_B, F_B_T, F_B_V, F_B_B, F_B_D, \
    F_D, F_D_V, F_D_T, \
    g, g_V, g_V_V, g_B, g_B_V, g_B_B, \
    del_F_del_n_T_V, del_F_del_T_V_n, del_F_del_V_T_n, \
    del_2F_del_T2_V_n, del_2F_del_T_del_V_n, del_2F_del_V2_T_n = derivs_primary(eos,
                                                                                n_count, n_sum, valid_comps,
                                                                                gas_const, temp, vol,
                                                                                B, B_sum, D, D_sum, D_T_sum, D_T_T_sum,
                                                                                wrt_composition=True)

    del_2F_del_V_del_n_T, del_2F_del_n_del_T_V, del_2F_del_n_del_n_T_V = derivs_secondary(n_count, valid_comps,
                                                                                          B, B_prime, D, D_prime,
                                                                                          D_T, D_T_sum,
                                                                                          _f, _f_V, _f_V_V, _f_B_V,
                                                                                          _f_B, _f_B_B,
                                                                                          F, F_n_V, F_n_B,
                                                                                          F_B, F_B_T, F_B_V,
                                                                                          F_B_B, F_B_D,
                                                                                          F_D, F_D_V, F_D_T,
                                                                                          second_order=comp_derivs)

    press_calc, compress_Z, \
    del_press_del_vol_temp_comp, del_press_del_temp_vol_comp, \
    del_press_del_n_temp_vol, del_vol_del_n_temp_press = press_derivs(n_count, n_sum, valid_comps,
                                                                      gas_const, temp, vol,
                                                                      press, compress_Z,
                                                                      del_F_del_V_T_n,
                                                                      del_2F_del_T_del_V_n,
                                                                      del_2F_del_V2_T_n, del_2F_del_V_del_n_T,
                                                                      wrt_composition=comp_derivs)

    log_phi, \
    del_log_phi_del_temp_press_comp, del_log_phi_del_press_temp_comp, \
    del_log_phi_del_n_temp_press = log_phi_derivs(n_count, n_sum, valid_comps,
                                                  gas_const, temp, vol, compress_Z, press,
                                                  del_press_del_vol_temp_comp, del_press_del_temp_vol_comp,
                                                  del_press_del_n_temp_vol, del_vol_del_n_temp_press,
                                                  del_F_del_n_T_V, del_2F_del_n_del_T_V, del_2F_del_n_del_n_T_V,
                                                  wrt_temp_press=False, wrt_composition=False)

    # If we are a pseudo phase, we need to add a pressure correction
    # According to Mathias, this should only be added to the liquid phase
    # In the case of a pseudo vapor phase, the log_press_factor is just set to 0
    if is_pseudo:
        log_phi += log_press_factor
        del_log_phi_del_press_temp_comp -= 1 / press
        pass

    s_res_T_V_n, \
    cv_res_T_V_n, \
    cp_res, \
    a_res_T_V_n, \
    h_res_T_P_n, \
    g_res_T_P_n, s_res_T_P_n = residual_derivs(n_sum, gas_const, temp, vol, compress_Z, press, F,
                                               del_press_del_vol_temp_comp, del_press_del_temp_vol_comp,
                                               del_F_del_T_V_n, del_2F_del_T2_V_n)

    u_res_T_V_n = a_res_T_V_n + temp * s_res_T_V_n

    calc_mw, \
    calc_ig_cp, \
    calc_ig_enthalpy, \
    calc_ig_entropy, \
    calc_ig_gibbs = calc_ig_props(gas_const, temp, press, n, n_sum,
                                  valid_comps, crit_temp, mw_list,
                                  ig_cp_coeffs, ig_h_form, ig_s_form,
                                  ig_temp_ref, ig_press_ref)

    calc_ig_helmholtz = calc_ig_gibbs - n_sum * gas_const * temp
    calc_ig_int_energy = calc_ig_helmholtz + temp * calc_ig_entropy

    vals = eventual_phase_id, \
           is_pseudo, \
           temp, \
           press_eos, \
           vol / n_sum, \
           (1 / (vol / n_sum)), \
           compress_Z, \
           n, \
           n_sum, \
           calc_mw, \
           calc_ig_gibbs, \
           calc_ig_helmholtz, \
           calc_ig_int_energy, \
           calc_ig_enthalpy, \
           calc_ig_entropy, \
           calc_ig_cp - gas_const, \
           calc_ig_cp, \
           g_res_T_P_n / n_sum, \
           a_res_T_V_n / n_sum, \
           u_res_T_V_n / n_sum, \
           h_res_T_P_n / n_sum, \
           s_res_T_V_n / n_sum, \
           cv_res_T_V_n / n_sum, \
           cp_res / n_sum, \
           calc_ig_gibbs + g_res_T_P_n / n_sum, \
           calc_ig_helmholtz + a_res_T_V_n / n_sum, \
           calc_ig_int_energy + u_res_T_V_n / n_sum, \
           calc_ig_enthalpy + h_res_T_P_n / n_sum, \
           calc_ig_entropy + s_res_T_V_n / n_sum, \
           calc_ig_cp - gas_const + cv_res_T_V_n / n_sum, \
           calc_ig_cp + cp_res / n_sum, \
           log_phi, \
           del_log_phi_del_temp_press_comp, \
           del_log_phi_del_press_temp_comp, \
           del_log_phi_del_n_temp_press, \
           del_press_del_vol_temp_comp, \
           del_press_del_temp_vol_comp, \
           del_press_del_n_temp_vol, \
           del_vol_del_n_temp_press

    return vals


@njit(cache=True)
def calc_wilson_k_values(temp, press, tc_list, pc_list, omega_list):
    k_values = np.empty(len(tc_list))
    for i in range(len(k_values)):
        pc = pc_list[i]
        tc = tc_list[i]
        omega = omega_list[i]
        k_values[i] = (pc / press) * math.exp(5.37 * (1 + omega) * (1 - tc / temp))

    return k_values


@njit(cache=True)
def estimate_nbp_value(feed_comp, tc_list, valid):
    temp = 0
    for i in valid:
        temp += feed_comp[i] * tc_list[i]
    return 0.7 * temp


class CubicEos(Provider):
    def __init__(self, eos, components=None, k_ij=None, l_ij=None):
        super().__init__()
        self._components = None
        self.all_comps = None
        self._id_list = None
        self._mw_list = None
        self._tc_list = None
        self._pc_list = None
        self._omega_list = None
        self._ig_temp_ref = None
        self._ig_press_ref = None
        self._ig_cp_coeffs = None
        self._ig_h_form = h = None
        self._ig_g_form = g = None
        self._vap_visc = None
        self._liq_visc = None
        self._ig_s_form = None
        self._std_liq_vol = None
        self._source_k_ij = None
        self._source_l_ij = None
        self._k_ij = None
        self._l_ij = None
        self._eos = eos
        if components:
            c = [i for i in components]
            self.setup_components(eos, c, k_ij, l_ij)

    @property
    def components(self):
        return self._components

    @property
    def all_valid_components(self):
        return self.all_comps

    @property
    def supported_flashes(self):
        return [{'temp', 'press'}]

    def setup_components(self, eos, components, k_ij, l_ij):
        if eos == 'srk':
            self._eos_code = 0
        elif eos == 'pr':
            self._eos_code = 1
        else:
            raise NotImplementedError

        self._components = components
        self.all_comps = np.arange(0, len(components))
        self._id_list = [c.identifier for c in components]
        self._mw_list = np.array([c.mw for c in components])
        self._tc_list = np.array([c.crit_temp for c in components])
        self._pc_list = np.array([c.crit_press for c in components])
        self._omega_list = np.array([c.acen_fact for c in components])
        self._ig_temp_ref = np.array([c.ig_temp_ref for c in components])
        self._ig_press_ref = np.array([c.ig_press_ref for c in components])
        self._ig_cp_coeffs = np.array([c.ig_cp_mole_coeffs for c in components])
        self._ig_h_form = h = np.array([c.ig_enthalpy_form_mole for c in components])
        self._ig_g_form = g = np.array([c.ig_gibbs_form_mole for c in components])
        self._vap_visc = [c.vap_visc for c in components]
        self._liq_visc = [c.liq_visc for c in components]
        self._ig_s_form = (g - h) / -298.15
        self._std_liq_vol = np.array([c.std_liq_vol_mole for c in components])
        self._source_k_ij = k_ij
        self._source_l_ij = l_ij
        self._k_ij = np.zeros((len(components), len(components)))
        self._l_ij = np.zeros((len(components), len(components)))
        self.update_interactions(k_ij, l_ij)

    @property
    def mw(self):
        return self._mw_list

    @property
    def std_liq_vol_mole(self):
        return self._std_liq_vol

    def vap_visc(self, temp, comp_mole):
        return np.dot(comp_mole, [comp_visc(temp) for comp_visc in self._vap_visc])

    def liq_visc(self, temp, comp_mole):
        return np.dot(comp_mole, [comp_visc(temp) for comp_visc in self._liq_visc])

    def update_interactions(self, k_ij, l_ij):
        if k_ij is not None:
            for (i_identifier, j_identifier), value in k_ij:
                i, j = self._id_list.index(i_identifier), self._id_list.index(j_identifier)
                self._k_ij[i, j] = self._k_ij[j, i] = value

        if l_ij is not None:
            for (i_identifier, j_identifier), value in l_ij:
                i, j = self._id_list.index(i_identifier), self._id_list.index(j_identifier)
                self._l_ij[i, j] = self._l_ij[j, i] = value

    def scaling(self, prop_type):
        if prop_type in ('enthalpy_mole', 'ig_enthalpy_mole', 'res_enthalpy_mole',
                         'gibbs_mole', 'ig_gibbs_mole', 'res_gibbs_mole',
                         'int_energy_mole', 'ig_int_energy_mole', 'res_int_energy_mole',
                         'helmholtz_mole', 'ig_helmholtz_mole', 'res_helmholtz_mole'):
            return GAS_CONSTANT * 298.15

        elif prop_type in ('entropy_mole', 'ig_entropy_mole', 'res_entropy_mole'):
            return GAS_CONSTANT
        else:
            raise NotImplementedError

    def convert_to_mole_basis(self, flow_sum_basis, flow_sum_value, frac_basis, frac_value):
        if frac_basis == 'mole':
            frac_value_mole = frac_value
        elif frac_basis == 'mass':
            frac_value_mole = frac_value / self._mw_list
            frac_value_mole /= np.sum(frac_value_mole)
        else:
            raise NotImplementedError

        avg_mw = np.dot(frac_value_mole, self._mw_list)
        if flow_sum_basis == 'mole':
            flow_sum_value_mole = flow_sum_value
        elif flow_sum_basis == 'mass':
            flow_sum_value_mole = flow_sum_value / avg_mw
        else:
            raise NotImplementedError

        return flow_sum_value_mole, frac_value_mole

    def flash_temp_press(self, flow_sum_basis, flow_sum_value, frac_basis, frac_value, temp, press, previous, valid):
        flow_sum_value_mole, frac_value_mole = self.convert_to_mole_basis(flow_sum_basis, flow_sum_value,
                                                                          frac_basis, frac_value)
        prev_k = None
        if previous is not None and previous.contains('vap', 'liq'):
            prev_k = previous.k_values_vle

        results = basic_flash_temp_press_2phase(self, temp, press, frac_value_mole, valid, previous_k_values=prev_k)
        results.scale(flow_sum_mole=flow_sum_value_mole)
        return results

    def flash_press_prop(self, flow_sum_basis, flow_sum_value,
                         frac_basis, frac_value, press,
                         prop_name, prop_basis, prop_value, previous, valid):

        flow_sum_value_mole, frac_value_mole = self.convert_to_mole_basis(flow_sum_basis, flow_sum_value,
                                                                          frac_basis, frac_value)

        prop_flash_name = prop_name + '_' + prop_basis
        start_temp = None
        if previous is not None:
            start_temp = previous.temp

        results = flash_press_prop_2phase(self, press, prop_flash_name, prop_value,
                                          0, frac_value, valid=valid,
                                          previous=previous, start_temp=start_temp)

        results.scale(flow_sum_mole=flow_sum_value_mole)
        return results

    def flash_press_vap_frac(self, flow_sum_basis, flow_sum_value, frac_basis, frac_value, press,
                             vap_frac_basis, vap_frac_value, previous, valid):

        flow_sum_value_mole, frac_value_mole = self.convert_to_mole_basis(flow_sum_basis, flow_sum_value,
                                                                          frac_basis, frac_value)
        if vap_frac_basis != 'mole':
            raise NotImplementedError

        results = flash_press_vap_frac_2phase(self, press, vap_frac_value, frac_value_mole, valid=valid, previous=previous)
        results.scale(flow_sum_mole=flow_sum_value_mole)
        return results

    def flash_temp_vap_frac(self, flow_sum_basis, flow_sum_value, frac_basis, frac_value, temp,
                             vap_frac_basis, vap_frac_value, previous, valid):

        flow_sum_value_mole, frac_value_mole = self.convert_to_mole_basis(flow_sum_basis, flow_sum_value,
                                                                          frac_basis, frac_value)
        if vap_frac_basis != 'mole':
            raise NotImplementedError

        results = flash_temp_vap_frac_2phase(self, temp, vap_frac_value, frac_value_mole, valid=valid, previous=previous)
        results.scale(flow_sum_mole=flow_sum_value_mole)
        return results


    def phase(self, temp, press, n, desired_phase,
              allow_pseudo=True, valid=None, press_comp_derivs=False,
              log_phi_temp_press_derivs=False, log_phi_comp_derivs=False):

        if desired_phase not in ['vap', 'liq']:
            raise NotImplementedError

        if valid is None:
            valid = np.where(n > MIN_COMPOSITION)[0]

        results = calc_phase(self._eos_code, GAS_CONSTANT, temp, press, n, valid, desired_phase, allow_pseudo,
                             self._mw_list, self._tc_list, self._pc_list, self._omega_list, self._k_ij, self._l_ij,
                             self._ig_temp_ref, self._ig_press_ref,
                             self._ig_cp_coeffs, self._ig_h_form, self._ig_s_form,
                             press_comp_derivs, log_phi_temp_press_derivs, log_phi_comp_derivs)

        phase_id, pseudo, temp, press_eos, \
        vol_mole, dens_mole, compress_fact, \
        frac_mole, frac_sum_mole, \
        mw, \
        ig_gibbs_mole, ig_helmholtz_mole, ig_int_energy_mole, ig_enthalpy_mole, \
        ig_entropy_mole, ig_cv_mole, ig_cp_mole, \
        res_gibbs_mole, res_helmholtz_mole, res_int_energy_mole, res_enthalpy_mole, \
        res_entropy_mole, res_cv_mole, res_cp_mole, \
        gibbs_mole, helmholtz_mole, int_energy_mole, enthalpy_mole, \
        entropy_mole, cv_mole, cp_mole, \
        log_phi, \
        del_log_phi_del_temp, del_log_phi_del_press, del_log_phi_del_comp, \
        del_press_del_vol, del_press_del_temp, del_press_del_comp, del_vol_del_comp = results

        return PhaseByMole(provider=self,
                           identifier=phase_id,
                           pseudo=pseudo, temp=temp, press=press,
                           vol_mole=vol_mole, dens_mole=dens_mole, z_factor=compress_fact,
                           comp_mole=frac_mole, comp_sum_mole=frac_sum_mole, mw=mw,

                           ig_gibbs_mole=ig_gibbs_mole, ig_helmholtz_mole=ig_helmholtz_mole,
                           ig_int_energy_mole=ig_int_energy_mole,
                           ig_enthalpy_mole=ig_enthalpy_mole, ig_entropy_mole=ig_entropy_mole,
                           ig_cv_mole=ig_cv_mole, ig_cp_mole=ig_cp_mole,

                           res_gibbs_mole=res_gibbs_mole, res_helmholtz_mole=res_helmholtz_mole,
                           res_int_energy_mole=res_int_energy_mole,
                           res_enthalpy_mole=res_enthalpy_mole, res_entropy_mole=res_entropy_mole,
                           res_cv_mole=res_cv_mole, res_cp_mole=res_cp_mole,

                           gibbs_mole=gibbs_mole, helmholtz_mole=helmholtz_mole,
                           int_energy_mole=int_energy_mole,
                           enthalpy_mole=enthalpy_mole, entropy_mole=entropy_mole,
                           cv_mole=cv_mole, cp_mole=cp_mole,

                           log_phi=log_phi,
                           del_log_phi_del_temp=del_log_phi_del_temp,
                           del_log_phi_del_press=del_log_phi_del_press,
                           del_log_phi_del_comp=del_log_phi_del_comp,
                           del_press_del_vol=del_press_del_vol,
                           del_press_del_temp=del_press_del_temp,
                           del_press_del_comp=del_press_del_comp,
                           del_vol_del_comp=del_vol_del_comp,
                           flow_mole=frac_mole, flow_sum_mole=1)

    def phases_vle(self, temp, press, liq_comp, vap_comp,
                   allow_pseudo=True, valid=None, press_comp_derivs=False,
                   log_phi_temp_press_derivs=False, log_phi_comp_derivs=False):

        liq_ph = self.phase(temp, press, liq_comp, 'liq',
                            allow_pseudo, valid,
                            press_comp_derivs,
                            log_phi_temp_press_derivs,
                            log_phi_comp_derivs)

        vap_ph = self.phase(temp, press, vap_comp, 'vap',
                            allow_pseudo, valid,
                            press_comp_derivs,
                            log_phi_temp_press_derivs,
                            log_phi_comp_derivs)

        return liq_ph, vap_ph

    def guess_k_value_vle(self, temp, press):
        return calc_wilson_k_values(temp, press, self._tc_list, self._pc_list, self._omega_list)

    def guess_nbp(self, feed_comp, valid):
        return estimate_nbp_value(feed_comp, self._tc_list, valid)

    def ig_props(self, temp, press, feed_comp, valid=None):
        if valid is None:
            valid = np.where(feed_comp > MIN_COMPOSITION)[0]

        calc_mw, \
        calc_ig_cp, \
        calc_ig_enthalpy, \
        calc_ig_entropy, \
        calc_ig_gibbs = calc_ig_props(GAS_CONSTANT, temp, press, feed_comp, 1,
                                      valid, self._tc_list, self._mw_list,
                                      self._ig_cp_coeffs, self._ig_h_form, self._ig_s_form,
                                      self._ig_temp_ref, self._ig_press_ref)

        calc_ig_helmholtz = calc_ig_gibbs - GAS_CONSTANT * temp
        calc_ig_int_energy = calc_ig_helmholtz + temp * calc_ig_entropy
        vol = GAS_CONSTANT * temp / press
        return vol, calc_mw, calc_ig_cp, calc_ig_enthalpy, calc_ig_entropy, calc_ig_int_energy, calc_ig_gibbs, calc_ig_helmholtz

    def AddCompound(self, compound):
        # print('AddCompound:', compound)
        comp_obj = chemsep.pure(compound)
        if self._components is None:
            new_components = [comp_obj]
        else:
            new_components = self._components[:]
            new_components.append(comp_obj)

        # This is really inefficient, but it's simple
        self.setup_components(self._eos, new_components, None, None)

    def GetAvCompoundNames(self):
        return chemsep.available()

    def DeleteCompound(self, compound):
        compound = compound.upper()
        idx = self._id_list.index(compound)
        new_compounds = self._components[:]
        new_compounds.pop(idx)
        self.setup_components(self._eos, new_compounds, None, None)

    def ExchangeCompound(self, cmp1Name, cmp2Name):
        cmp1Name = cmp1Name.upper()
        cmp2Name = cmp2Name.upper()
        idx_1 = self._id_list.index(cmp1Name)
        idx_2 = self._id_list.index(cmp2Name)
        new_compounds = self._components[:]
        new_compounds[idx_1], new_compounds[idx_2] = new_compounds[idx_2], new_compounds[idx_1]
        self.setup_components(self._eos, new_compounds, None, None)

    def MoveCompound(self, cmp1Name, cmp2Name):
        cmp1Name = cmp1Name.upper()
        cmp2Name = cmp2Name.upper()
        new_compounds = self._components[:]
        item_1 = new_compounds.pop(self._id_list.index(cmp1Name))
        new_compounds.insert(self._id_list.index(cmp2Name), item_1)
        self.setup_components(self._eos, new_compounds, None, None)


class SoaveRedlichKwong(CubicEos):
    """
    Shortcut class for SoaveRedlichKwong instance
    """

    def __init__(self, components=None, k_ij=None, l_ij=None):
        CubicEos.__init__(self, 'srk', components, k_ij, l_ij)


class PengRobinson(CubicEos):
    """
    Shortcut class for PengRobinson instance
    """

    def __init__(self, components=None, k_ij=None, l_ij=None):
        CubicEos.__init__(self, 'pr', components, k_ij, l_ij)
