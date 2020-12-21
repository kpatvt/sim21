from math import log
import numpy as np
from numba import njit
from sim21.data.eqn import eval_eqn, eval_eqn_int, eval_eqn_int_over_t


@njit(cache=True)
def calc_ig_props(r, temp, press, n, n_sum, valid, crit_temp, mw, ig_cp_coeffs, ig_h_form, ig_s_form, t_ref, press_ref):
    """
    Calculate Ideal Gas properties for the component properties
    """
    calc_ig_cp = 0
    calc_ig_enthalpy = 0
    calc_ig_entropy = 0
    calc_ig_entropy_comp_sum = 0
    calc_mw = 0

    for i in valid:
        coeffs = ig_cp_coeffs[i, :]
        calc_ig_cp += n[i] * eval_eqn(coeffs, temp, crit_temp[i])

        h_ig = eval_eqn_int(coeffs, temp, t_ref[i]) + ig_h_form[i]
        calc_ig_enthalpy += n[i] * h_ig

        s_ig = eval_eqn_int_over_t(coeffs, temp, t_ref[i]) - r * log(press / press_ref[i]) + ig_s_form[i]
        calc_ig_entropy += n[i] * s_ig
        calc_ig_entropy_comp_sum += -r * n[i] * log(n[i])
        calc_mw += n[i] * mw[i]

    # Add the component summation
    calc_ig_entropy += calc_ig_entropy_comp_sum

    calc_ig_cp /= n_sum
    calc_ig_enthalpy /= n_sum
    calc_ig_entropy /= n_sum
    calc_ig_gibbs = calc_ig_enthalpy - temp * calc_ig_entropy

    return calc_mw, calc_ig_cp, calc_ig_enthalpy, calc_ig_entropy, calc_ig_gibbs


@njit(cache=True)
def press_derivs(n_count, n_sum, valid_comps,
                 gas_const, temp, vol,
                 press_calc, compress_Z,
                 del_F_del_V_T_n,
                 del_2F_del_T_del_V_n,
                 del_2F_del_V2_T_n,
                 del_2F_del_V_del_n_T,
                 wrt_composition=False):
    """
    Calculate the derivatives of the pressure given key derivatives of the helmholtz function
    """
    R = gas_const
    # press_calc = -R*temp*del_F_del_V_T_n + n_sum*R*temp/vol
    # Z = press_calc*vol/(n_sum*R*temp)

    del_press_del_vol_temp_comp = -R * temp * del_2F_del_V2_T_n - n_sum * R * temp / (vol ** 2)
    del_press_del_temp_vol_comp = -R * temp * del_2F_del_T_del_V_n + press_calc / temp

    del_press_del_n_temp_vol = np.zeros(n_count)
    del_vol_del_n_temp_press = np.zeros(n_count)

    if wrt_composition:
        for i in valid_comps:
            del_press_del_n_temp_vol[i] = -R * temp * (del_2F_del_V_del_n_T[i]) + R * temp / vol
            del_vol_del_n_temp_press[i] = -del_press_del_n_temp_vol[i] / del_press_del_vol_temp_comp

    return press_calc, compress_Z, \
           del_press_del_vol_temp_comp, del_press_del_temp_vol_comp, \
           del_press_del_n_temp_vol, del_vol_del_n_temp_press


@njit(cache=True)
def log_phi_derivs(n_count, n_sum, valid_comps,
                   gas_const, temp, vol, compress_Z, press,
                   del_press_del_vol_temp_comp, del_press_del_temp_vol_comp,
                   del_press_del_n_temp_vol, del_vol_del_n_temp_press,
                   del_F_del_n_T_V, del_2F_del_T_del_n_V, del_2F_del_n_del_n_T_V,
                   wrt_temp_press=False, wrt_composition=False):
    """
    Calculate the derivatives of the fugacity coefficients (log_phi) given key derivatives of the helmholtz function
    """

    R = gas_const
    log_phi = np.zeros(n_count)
    log_Z = log(compress_Z)
    temp_inv = 1 / temp
    gas_const_temp_inv = 1 / (R * temp)

    fact1 = del_press_del_temp_vol_comp
    vol_part = del_vol_del_n_temp_press

    del_log_phi_del_temp_press_comp = np.zeros(n_count)
    del_log_phi_del_press_temp_comp = np.zeros(n_count)
    del_log_phi_del_n_temp_press = np.zeros((n_count, n_count))

    for i in valid_comps:
        log_phi[i] = del_F_del_n_T_V[i] - log_Z
        if wrt_temp_press:
            del_log_phi_del_temp_press_comp[i] = del_2F_del_T_del_n_V[i] + 1 / temp - vol_part[i] / (R * temp) * fact1
            del_log_phi_del_press_temp_comp[i] = vol_part[i] / (R * temp) - 1 / press

        if wrt_composition:
            for j in valid_comps:
                term1 = del_2F_del_n_del_n_T_V[i, j]
                term3 = (del_press_del_n_temp_vol[j] * del_press_del_n_temp_vol[i]) / del_press_del_vol_temp_comp
                del_log_phi_del_n_temp_press[i, j] = (term1 + 1 + (n_sum / (R * temp)) * term3) / n_sum

    return log_phi, \
           del_log_phi_del_temp_press_comp, del_log_phi_del_press_temp_comp, \
           del_log_phi_del_n_temp_press


@njit(cache=True)
def residual_derivs(n_sum,
                    gas_const, temp, vol, compress_Z, press, F,
                    del_press_del_vol_temp_comp, del_press_del_temp_vol_comp,
                    del_F_del_T_V_n, del_2F_del_T2_V_n):
    """
    Calculate residual properties (entropy, enthalpy, gibbs, helmholtz, cp, cv) given derivatives of the helmholtz
    """
    R = gas_const
    s_res_T_V_n = R * (-temp * del_F_del_T_V_n - F)
    cv_res_T_V_n = R * (-(temp ** 2) * del_2F_del_T2_V_n - 2 * temp * del_F_del_T_V_n)

    cp_minus_cv_res = R * ((-temp / R) * (del_press_del_temp_vol_comp ** 2) / del_press_del_vol_temp_comp - n_sum)
    cp_res = cp_minus_cv_res + cv_res_T_V_n

    a_res_T_V_n = n_sum * R * temp * F
    h_res_T_P_n = a_res_T_V_n + temp * s_res_T_V_n + press * vol - n_sum * R * temp
    g_res_T_P_n = a_res_T_V_n + press * vol - n_sum * R * temp * (1 + log(compress_Z))
    s_res_T_P_n = (h_res_T_P_n - g_res_T_P_n) / temp

    return s_res_T_V_n, cv_res_T_V_n, cp_res, a_res_T_V_n, h_res_T_P_n, g_res_T_P_n, s_res_T_P_n
