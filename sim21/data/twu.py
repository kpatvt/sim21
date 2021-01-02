import math
from scipy.optimize import fsolve
from sim21.data.chemsep_consts import GAS_CONSTANT
from sim21.data.eqn import eval_eqn, eval_eqn_int, eval_eqn_int_over_t
import numpy as np


def fixed_properties(tb, sg, mw):
    """
    Obtain the key fixed properties using the Twu correlations
    :param tb: boiling point temperature in K
    :return: Tc (in K), Pc (in Pa), Vc (m3/kmol),  MW, Omega
    """
    tb_R = tb * 1.8

    t1 = 0.533272 + 0.191017e-3 * tb_R + 0.779681e-7 * (tb_R ** 2) - 0.284376e-10 * (tb_R ** 3)
    t1 += 0.959468e28 * (tb_R ** -13)

    tc_dot_R = tb_R * (1 / t1)
    alpha = 1 - tb_R / tc_dot_R

    sg_dot = 0.843593 - 0.128624 * alpha - 3.336159 * (alpha ** 3) - 13749.5 * (alpha ** 12)

    t2 = 0.419869 - 0.505839 * alpha - 1.5436 * (alpha ** 3) - 9481.70 * (alpha ** 14)
    vc_dot_ft3_lb_mol = (1 - t2) ** (-8)

    # Iterate for mw_dot
    mw_dot_trial = tb_R / (10.44 - 0.0052 * tb_R)
    theta_trial = math.log(mw_dot_trial)

    def tb_R_error(theta_value):
        temp1 = 5.71419 + 2.7157 * theta_value - 0.286590 * (theta_value ** 2)
        temp1 = temp1 - 39.8544 / theta_value - 0.122488 / (theta_value ** 2)
        return (math.exp(temp1) - 24.7522 * theta_value + 35.3155 * (theta_value ** 2)) - tb_R

    theta = fsolve(tb_R_error, theta_trial)[0]
    mw_dot = math.exp(theta)

    alpha = 1 - tb_R / tc_dot_R
    t1 = 3.83353 + 1.19629 * (alpha ** 0.5) + 34.8888 * alpha + 36.1952 * (alpha ** 2)
    t1 += 104.193 * (alpha ** 4)
    pc_dot_psi = t1 ** 2

    delta_sg_t = math.exp(5 * (sg_dot - sg)) - 1
    t1 = -0.362456 / (tb_R ** 0.5) + (0.0398285 - 0.948125 / (tb_R ** 0.5)) * delta_sg_t
    f_t = delta_sg_t * t1
    tc_R = tc_dot_R * (((1 + 2 * f_t) / (1 - 2 * f_t)) ** 2)

    delta_sg_v = math.exp(4 * (sg_dot ** 2 - sg ** 2)) - 1
    t1 = 0.466590 / (tb_R ** 0.5) + (-0.182421 + 3.01721 / (tb_R ** 0.5)) * delta_sg_v
    f_v = delta_sg_v * t1
    vc_ft3_lb_mol = vc_dot_ft3_lb_mol * (((1 + 2 * f_v) / (1 - 2 * f_v)) ** 2)

    delta_sg_p = math.exp(0.5 * (sg_dot - sg)) - 1
    t1 = 2.53262 - 46.1955 / (tb_R ** 0.5) - 0.00127885 * tb_R
    t2 = -11.4277 + 252.140 / (tb_R ** 0.5) + 0.00230535 * tb_R
    f_p = delta_sg_p * (t1 + t2 * delta_sg_p)
    t3 = ((1 + 2 * f_p) / (1 - 2 * f_p)) ** 2
    pc_psi = pc_dot_psi * (tc_R / tc_dot_R) * (vc_dot_ft3_lb_mol / vc_ft3_lb_mol) * t3

    delta_sg_m = math.exp(5 * (sg_dot - sg)) - 1
    x = abs(0.0123420 - 0.328086 / (tb_R ** 0.5))
    f_m = delta_sg_m * (x + (-0.0175691 + 0.193168 / (tb_R ** 0.5)) * delta_sg_m)
    ln_mw = math.log(mw_dot) * (((1 + 2 * f_m) / (1 - 2 * f_m)) ** 2)
    mw_calc = math.exp(ln_mw)

    if mw is None:
        mw = mw_calc

    tr = tb_R / tc_R

    ln_pr_0 = (-5.96346 * (1 - tr) + 1.17639 * ((1 - tr) ** 1.5) - 0.559607 * ((1 - tr) ** 3) - 1.31901 * (
        (1 - tr) ** 6)) * (1 / tr)
    # pr_0 = math.exp(ln_pr_0)

    ln_pr_1 = (-4.78522 * (1 - tr) + 0.413999 * ((1 - tr) ** 1.5) - 8.913290 * ((1 - tr) ** 3) - 4.986620 * (
        (1 - tr) ** 6)) * (1 / tr)
    # pr_1 = math.exp(ln_pr_1)

    pc_atm = pc_psi / 14.6959487755142
    omega = (-math.log(pc_atm) - ln_pr_0) / ln_pr_1

    tc_K = tb_R / 1.8
    pc_pa = pc_atm * 101325.0
    vc_m3_kmol = vc_ft3_lb_mol * 0.0283168 / (453.59237 * 0.001)

    return tc_K, pc_pa, vc_m3_kmol, mw, omega


def ig_heat_cp_coeffs(tb, sg):
    tb_R = tb * 1.8
    watson_k = (tb_R ** (1 / 3)) / sg

    c1 = -0.33886 + 0.02827 * watson_k
    c2 = -(0.9291 - 1.1543 * watson_k + 0.0368 * (watson_k ** 2)) * 1e-4
    c3 = -1.6658e-7
    c4 = -(0.26105 - 0.59332 * watson_k)
    c5 = -4.92 * 1e-4
    c6 = -(0.536 - 0.6828 * watson_k) * 1e-7
    c7 = ((12.8 - watson_k) * (10 - watson_k) / (10 * watson_k)) ** 2

    a2 = c1 + c7 * c4
    a3 = c2 + c7 * c5
    a4 = c3 + c7 * c1

    # T - R
    # Cp - Btu/lb-mol

    # H = a1 + a2*T + (a3/2)*(T**2) + (a4/3)*(T**3)
    # Cp = a2 + a3*T + a4*(T**2)

    # Btu/lb-mol-R -> (1055.0558526*1.8)/0.45359237
    a2 = a2 * (1055.0558526 * 1.8) / 0.45359237
    a3 = a3 * (1055.0558526 * 1.8) / 0.45359237 * 1.8
    a4 = a4 * (1055.0558526 * 1.8) / 0.45359237 * 1.8 * 1.8

    # no, tmin, tmax, a, b, c, d, e, f
    coeffs = 1, 0, 2000, a2, a3, a4, 0, 0, 0
    return coeffs


class TwuHypo:

    def __init__(self, identifier, tb, sg, mw=None):
        self.identifier = identifier.upper()
        tc, pc, vc, mw, omega = fixed_properties(tb, sg, mw)
        self.mw = mw
        self.acen_fact = omega
        self.crit_compress_fact = pc * vc / (GAS_CONSTANT * tc)  # Unitless
        self.crit_press = pc  # Pa
        self.crit_temp = tc  # K
        self.crit_vol_mole = vc  # m3/kmol
        coeffs = np.array(ig_heat_cp_coeffs(tb, sg))
        self.ig_cp_mole_coeffs = coeffs  # J/kmol-K
        # TODO Document the regression to get the ideal gas enthalpy
        self.ig_enthalpy_form_mole = -104680000+(-1476031.316)*(mw-44.097)  # J/kmol
        self.ig_gibbs_form_mole = -24390000+(584256.2662)*(mw-44.097)     # J/kmol
        self.ig_entropy_form_mole = (self.ig_gibbs_form_mole - self.ig_enthalpy_form_mole)/-298.15  # J/kmol-K

        # TODO Fix Specific Gravity conversion
        self.std_liq_vol_mole = 1/(sg*1000/mw)
        self.ig_temp_ref = 298.15
        self.ig_press_ref = 101325.0

    def ig_heat_cap_mole(self, temp):
        return eval_eqn(self.ig_cp_mole_coeffs, temp, self.crit_temp)

    def ig_enthalpy_mole(self, temp):
        return eval_eqn_int(self.ig_cp_mole_coeffs, temp, self.ig_temp_ref) + self.ig_enthalpy_form_mole

    def ig_entropy_mole(self, temp, press):
        p1 = eval_eqn_int_over_t(self.ig_cp_mole_coeffs, temp, self.ig_temp_ref) + self.ig_entropy_form_mole
        return p1 - GAS_CONSTANT * math.log(press / self.ig_press_ref)

    def ig_gibbs_mole(self, temp, press):
        return self.ig_enthalpy_mole(temp) - self.ig_entropy_mole(temp, press)

    def liq_visc(self, temp):
        return 0

    def vap_visc(self, temp):
        return 0

    def surf_tens(self, temp):
        return 0
