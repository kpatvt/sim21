import math
from numba import njit
from sim21.data.chemsep_consts import GAS_CONSTANT


@njit(cache=True)
def eqn_1_2_3_4_5_100(a, b, c, d, e, f, t):
    return a + b * t + c * (t ** 2) + d * (t ** 3) + e * (t ** 4)


@njit(cache=True)
def deriv_eqn_1_2_3_4_5_100(a, b, c, d, e, f, t):
    return b + 2 * c * t + 3 * d * (t ** 2) + 4 * e * (t ** 3)


@njit(cache=True)
def int_eqn_1_2_3_4_5_100(a, b, c, d, e, f, t, t_ref):
    res = a * (t - t_ref) + (b / 2) * (t ** 2 - t_ref ** 2) + (c / 3) * (t ** 3 - t_ref ** 3)
    res += (d / 4) * (t ** 4 - t_ref ** 4) + (e / 5) * (t ** 5 - t_ref ** 5)
    return res


@njit(cache=True)
def int_eqn_1_2_3_4_5_100_over_t(a, b, c, d, e, f, t, t_ref):
    res = a * math.log(t / t_ref) + b * (t - t_ref) + (c / 2) * (t ** 2 - t_ref ** 2)
    res += (d / 3) * (t ** 3 - t_ref ** 3) + (e / 4) * (t ** 4 - t_ref ** 4)
    return res


@njit(cache=True)
def eqn_6(a, b, c, d, e, f, t):
    return a + b * t + c * (t ** 2) + d * (t ** 3) + e / (t ** 2)


@njit(cache=True)
def eqn_10_207(a, b, c, d, e, f, t):
    return math.exp(a - b / (t + c))


@njit(cache=True)
def eqn_11_12_13_14_15(a, b, c, d, e, f, t):
    return math.exp(a + b * t + c * (t ** 2) + d * (t ** 3) + e * (t ** 4))


@njit(cache=True)
def eqn_16(a, b, c, d, e, f, t):
    return a + math.exp(b / t + c + d * t + e * (t ** 2))


@njit(cache=True)
def eqn_17(a, b, c, d, e, f, t):
    return a + math.exp(b + c * t + d * (t ** 2) + e * (t ** 3))


@njit(cache=True)
def eqn_45(a, b, c, d, e, f, t):
    return a * t + (b / 2) * (t ** 2) + (c / 3) * (t ** 3) + (d / 4) * (t ** 4) + (e / 5) * (t ** 5)


@njit(cache=True)
def eqn_75(a, b, c, d, e, f, t):
    return b + 2 * c * t + 3 * d * (t ** 2) + 4 * e * (t ** 3)


@njit(cache=True)
def eqn_10(a, b, c, d, e, f, t):
    return math.exp(a - b / (c + t))


@njit(cache=True)
def eqn_101(a, b, c, d, e, f, t):
    return math.exp(a + b / t + c * math.log(t) + d * (t ** e))


@njit(cache=True)
def eqn_102(a, b, c, d, e, f, t):
    return (a * (t ** b)) / (1 + c / t + d / (t ** 2))


@njit(cache=True)
def eqn_103(a, b, c, d, e, f, t):
    return a + b * math.exp(-c / (t ** d))


@njit(cache=True)
def eqn_104(a, b, c, d, e, f, t):
    return a + b / t + c / (t ** 3) + d / (t ** 8) + e / (t ** 9)


@njit(cache=True)
def eqn_105(a, b, c, d, e, f, t):
    return a / (b ** (1 + (1 - t / c) ** d))


@njit(cache=True)
def eqn_106(a, b, c, d, e, f, t, tr):
    return a * ((1.0 - tr) ** (b + c * tr + d * (tr ** 2) + e * (tr ** 3)))


@njit(cache=True)
def eqn_107(a, b, c, d, e, f, t):
    step1 = (c / t) / math.sinh(c / t)
    step2 = (e / t) / math.cosh(e / t)
    return a + b * (step1 ** 2) + d * (step2 ** 2)


@njit(cache=True)
def int_eqn_107(a, b, c, d, e, f, t, t_ref):
    part1 = a * t + b * c * (1 / math.tanh(c / t)) - d * e * math.tanh(e / t)
    part2 = a * t_ref + b * c * (1 / math.tanh(c / t_ref)) - d * e * math.tanh(e / t_ref)
    return part1 - part2


@njit(cache=True)
def int_eqn_107_over_t(a, b, c, d, e, f, t, t_ref):
    step1 = a * math.log(t) + b * ((c / t) * (1 / math.tanh(c / t)) - math.log(math.sinh(c / t)))
    step2 = d * ((e / t) * math.tanh(e / t) - math.log(math.cosh(e / t)))
    part1 = step1 - step2

    step1 = a * math.log(t_ref) + b * ((c / t_ref) * (1 / math.tanh(c / t_ref)) - math.log(math.sinh(c / t_ref)))
    step2 = d * ((e / t_ref) * math.tanh(e / t_ref) - math.log(math.cosh(e / t_ref)))
    part2 = step1 - step2

    return part1 - part2


@njit(cache=True)
def eqn_114(a, b, c, d, e, f, t):
    return a * t + b * ((t ** 2) / 2) + c * ((t ** 3) / 3) + d * ((t ** 4) / 4)


@njit(cache=True)
def eqn_117(a, b, c, d, e, f, t):
    step1 = b * (c / t) / math.tanh(c / t)
    step2 = d * (e / t) / math.tanh(e / t)
    return a * t + step1 - step2


@njit(cache=True)
def eqn_118(a, b, c, d, e, f, t):
    return math.exp(a + b / (t ** e) + c * math.log(t) + d * (t ** 2))


@njit(cache=True)
def eqn_120(a, b, c, d, e, f, t):
    return a - b / (t + c)


@njit(cache=True)
def eqn_121(a, b, c, d, e, f, t):
    return a + b / t + c * math.log(t) + d * (t ** e)


@njit(cache=True)
def eqn_200(a, b, c, d, e, f, t):
    return e * math.exp((a * t + b * (t ** 1.5) + c * (t ** 2.5) + d * (t ** 5)) / (1 - t))


@njit(cache=True)
def eval_eqn(coeffs, t, tc):
    """
    Evaluate coeffs with the given temp and tc
    """
    # no, tmin, tmax, a, b, c, d, e, f = coeffs
    eqn_no = int(coeffs[0])
    tmin = coeffs[1]
    tmax = coeffs[2]
    a = coeffs[3]
    b = coeffs[4]
    c = coeffs[5]
    d = coeffs[6]
    e = coeffs[7]
    f = coeffs[8]

    if eqn_no in (1, 2, 3, 4, 5, 100):
        return eqn_1_2_3_4_5_100(a, b, c, d, e, f, t)

    if eqn_no == 6:
        return eqn_6(a, b, c, d, e, f, t)

    if eqn_no in (10, 207):
        return eqn_10_207(a, b, c, d, e, f, t)

    if eqn_no in (11, 12, 13, 14, 15):
        return eqn_11_12_13_14_15(a, b, c, d, e, f, t)

    if eqn_no == 16:
        return eqn_16(a, b, c, d, e, f, t)

    if eqn_no == 17:
        return eqn_17(a, b, c, d, e, f, t)

    if eqn_no == 45:
        return eqn_45(a, b, c, d, e, f, t)

    if eqn_no == 75:
        return eqn_75(a, b, c, d, e, f, t)

    if eqn_no == 10:
        return eqn_10(a, b, c, d, e, f, t)

    if eqn_no == 101:
        return eqn_101(a, b, c, d, e, f, t)

    if eqn_no == 102:
        return eqn_102(a, b, c, d, e, f, t)

    if eqn_no == 103:
        return eqn_103(a, b, c, d, e, f, t)

    if eqn_no == 104:
        return eqn_104(a, b, c, d, e, f, t)

    if eqn_no == 105:
        return eqn_105(a, b, c, d, e, f, t)

    if eqn_no == 106:
        tr = t / tc
        return eqn_106(a, b, c, d, e, f, t, tr)

    if eqn_no == 107:
        return eqn_107(a, b, c, d, e, f, t)

    if eqn_no == 114:
        return eqn_114(a, b, c, d, e, f, t)

    if eqn_no == 117:
        return eqn_117(a, b, c, d, e, f, t)

    if eqn_no == 118:
        return eqn_118(a, b, c, d, e, f, t)

    if eqn_no == 120:
        return eqn_120(a, b, c, d, e, f, t)

    if eqn_no == 121:
        return eqn_121(a, b, c, d, e, f, t)

    if eqn_no == 200:
        return eqn_200(a, b, c, d, e, f, t)

    return math.nan


@njit(cache=True)
def eval_eqn_int(coeffs, t, t_ref):
    """
    Evaluate the integral of the coeffs with reference temp t_ref and reference value val_ref
    """
    val_ref = 0
    # no, tmin, tmax, a, b, c, d, e, f = coeffs
    eqn_no = coeffs[0]
    tmin = coeffs[1]
    tmax = coeffs[2]
    a = coeffs[3]
    b = coeffs[4]
    c = coeffs[5]
    d = coeffs[6]
    e = coeffs[7]
    f = coeffs[8]

    if eqn_no == 1 or eqn_no == 2 or eqn_no == 3 or eqn_no == 4 or eqn_no == 5 or eqn_no == 100:
        return int_eqn_1_2_3_4_5_100(a, b, c, d, e, f, t, t_ref) + val_ref

    if eqn_no == 107:
        return int_eqn_107(a, b, c, d, e, f, t, t_ref) + val_ref

    return math.nan


@njit(cache=True)
def eval_eqn_int_over_t(coeffs, t, t_ref):
    """
    Evaluate the integral of the coeffs divided by temp with reference temp t_ref and reference value val_ref
    """
    val_ref = 0
    # no, tmin, tmax, a, b, c, d, e, f = coeffs
    eqn_no = coeffs[0]
    tmin = coeffs[1]
    tmax = coeffs[2]
    a = coeffs[3]
    b = coeffs[4]
    c = coeffs[5]
    d = coeffs[6]
    e = coeffs[7]
    f = coeffs[8]

    if eqn_no == 1 or eqn_no == 2 or eqn_no == 3 or eqn_no == 4 or eqn_no == 5 or eqn_no == 100:
        return int_eqn_1_2_3_4_5_100_over_t(a, b, c, d, e, f, t, t_ref) + val_ref

    if eqn_no == 107:
        return int_eqn_107_over_t(a, b, c, d, e, f, t, t_ref) + val_ref

    return math.nan


@njit(cache=True)
def calc_ig_props(x, valid_comps, mw, ig_cp_mole_coeffs, t, t_ref, p, p_ref, h_ref, s_ref, tc):
    """
    Evaluate ideal gas properties by mole
    """
    r = GAS_CONSTANT
    mw_sum = 0
    ig_cp_mole_sum = 0
    ig_enthalpy_mole_sum = 0
    ig_entropy_mole_sum = 0
    ig_entropy_comp_sum = 0

    n = len(valid_comps)
    for i_index in range(n):
        i = valid_comps[i_index]
        mw_sum += x[i]*mw[i]
        coeffs = ig_cp_mole_coeffs[i, :]
        ig_cp_mole_sum += x[i]*eval_eqn(coeffs, t, tc[i])
        ig_enthalpy_mole_sum += x[i]*eval_eqn_int(coeffs, t, t_ref, h_ref[i])
        ig_entropy_mole_sum += x[i]*(eval_eqn_int_over_t(coeffs, t, t_ref, s_ref[i]) - r*math.log(p/p_ref))
        ig_entropy_comp_sum += -r * x[i] * math.log(x[i])

    ig_entropy_mole_sum = ig_entropy_mole_sum + ig_entropy_comp_sum
    ig_gibbs_mole_sum = ig_enthalpy_mole_sum - t_ref*ig_entropy_mole_sum
    return mw_sum, ig_cp_mole_sum, ig_enthalpy_mole_sum, ig_entropy_mole_sum,  ig_gibbs_mole_sum
