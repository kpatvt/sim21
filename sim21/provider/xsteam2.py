import math

import numpy as np
from numba import njit


# UPDATED BY Kiran Pashikanti <kpatvt@gmail.com>
# All changes are distributed under same terms as the original XSteam code.

# * Changes are as follows:
# * Jan 13, 2021
# * The conversions to bar and deg_C have been removed, all units are the same as the official specification
# * Equations added for metable vapor Region 2 but must be called directly
# * The coefficients for region 5 were updated to match most recent 2012 IAPWS 97 release
# * Iteration counters (max 1000 iterations) added to ensure no infinite loops when calling iterated properties
# * A self-test routine to check that calculated values match reference values in 2012 IAPWS 97 release (They do)
# * All arrays have been hoisted out of the functions so they are created in each function call
# * If numba is not available, should use the list version

# * Performance:
# * %timeit benchmark()
#   1.22 s ± 34.5 ms per loop (mean ± std. dev. of 7 runs, 1 loop each) (Without numba)
#   79.6 ms ± 1.78 ms per loop (mean ± std. dev. of 7 runs, 10 loops each) (With numba) - 15.3 x speedup

# ***********************************************************************************************************
# * Water and steam properties according to IAPWS IF-97                                                     *
# * By Magnus Holmgren, www.x-eng.com                                                                       *
# * The steam tables are free and provided as is.                                                           *
# * We take no responsibilities for any errors in the code or damage thereby.                               *
# * You are free to use, modify and distribute the code as long as authorship is properly acknowledged.     *
# * Please notify me at magnus@x-eng.com if the code is used in commercial applications                     *
# ***********************************************************************************************************
#
# XSteam provides accurate steam and water properties from 0 - 1000 bar and from 0 - 2000 deg C according to
# the standard IAPWS IF-97. For accuracy of the functions in different regions see IF-97 (www.iapws.org)
#
# *** Using XSteam *****************************************************************************************
# XSteam take 2 or 3 arguments. The first argument must always be the steam table function you want to use.
# The other arguments are the inputs to that function.
# Example: XSteam('h_pt',1,20)  Returns the enthalpy of water at 1 bar and 20 degC
# Example: XSteam('TSat_p',1)  Returns the saturation temperature of water at 1 bar.
# For a list of valid Steam Table functions se bellow or the XSteam macros for MS Excel.
#
# *** Nomenclature ******************************************************************************************
# First the wanted property then a _ then the wanted input properties.
# Example. T_ph is temperature as a function of pressure and enthalpy.
# For a list of valid functions se bellow or XSteam for MS Excel.
# T     Temperature (deg C)
# p	    Pressure    (bar)
# h	    Enthalpy    (kJ/kg)
# v	    Specific volume (m3/kg)
# rho	Density
# s	    Specific entropy
# u	    Specific internal energy
# Cp	Specific isobaric heat capacity
# Cv	Specific isochoric heat capacity
# w	    Speed of sound
# my	Viscosity
# tc	Thermal Conductivity
# st	Surface Tension
# x	    Vapour fraction
# vx	Vapour Volume Fraction
#
# *** Valid Steam table functions. ****************************************************************************
#
# Temperature
# Tsat_p	Saturation temperature
# T_ph  Temperature as a function of pressure and enthalpy
# T_ps  Temperature as a function of pressure and entropy
# T_hs  Temperature as a function of enthalpy and entropy
#
# Pressure
# psat_T Saturation pressure
# p_hs	 Pressure as a function of h and s.
# p_hrho Pressure as a function of h and rho. Very inaccurate for solid water region
#        since it's almost incompressible!
#
# Enthalpy
# hV_p	 Saturated vapour enthalpy
# hL_p	 Saturated liquid enthalpy
# hV_T	 Saturated vapour enthalpy
# hL_T	 Saturated liquid enthalpy
# h_pT	 Enthalpy as a function of pressure and temperature.
# h_ps	 Enthalpy as a function of pressure and entropy.
# h_px	 Enthalpy as a function of pressure and vapour fraction
# h_prho Enthalpy as a function of pressure and density. Observe for low temperatures
#       (liquid) this equation has 2 solutions.
# h_Tx	 Enthalpy as a function of temperature and vapour fraction
#
# Specific volume
# vV_p	Saturated vapour volume
# vL_p	Saturated liquid volume
# vV_T	Saturated vapour volume
# vL_T	Saturated liquid volume
# v_pT	Specific volume as a function of pressure and temperature.
# v_ph	Specific volume as a function of pressure and enthalpy
# v_ps	Specific volume as a function of pressure and entropy.
#
# Density
# rhoV_p	Saturated vapour density
# rhoL_p	Saturated liquid density
# rhoV_T	Saturated vapour density
# rhoL_T	Saturated liquid density
# rho_pT	Density as a function of pressure and temperature.
# rho_ph	Density as a function of pressure and enthalpy
# rho_ps	Density as a function of pressure and entropy.
#
# Specific entropy
# sV_p	Saturated vapour entropy
# sL_p	Saturated liquid entropy
# sV_T	Saturated vapour entropy
# sL_T	Saturated liquid entropy
# s_pT	Specific entropy as a function of pressure and temperature (Returns saturated vapour Enthalpy if mixture.)
# s_ph	Specific entropy as a function of pressure and enthalpy
#
# Specific internal energy
# uV_p	Saturated vapour internal energy
# uL_p	Saturated liquid internal energy
# uV_T	Saturated vapour internal energy
# uL_T	Saturated liquid internal energy
# u_pT	Specific internal energy as a function of pressure and temperature.
# u_ph	Specific internal energy as a function of pressure and enthalpy
# u_ps	Specific internal energy as a function of pressure and entropy.
#
# Specific isobaric heat capacity
# CpV_p	Saturated vapour heat capacity
# CpL_p	Saturated liquid heat capacity
# CpV_T	Saturated vapour heat capacity
# CpL_T	Saturated liquid heat capacity
# Cp_pT	Specific isobaric heat capacity as a function of pressure and temperature.
# Cp_ph	Specific isobaric heat capacity as a function of pressure and enthalpy
# Cp_ps	Specific isobaric heat capacity as a function of pressure and entropy.
#
# Specific isochoric heat capacity
# CvV_p	Saturated vapour isochoric heat capacity
# CvL_p	Saturated liquid isochoric heat capacity
# CvV_T	Saturated vapour isochoric heat capacity
# CvL_T	Saturated liquid isochoric heat capacity
# Cv_pT	Specific isochoric heat capacity as a function of pressure and temperature.
# Cv_ph	Specific isochoric heat capacity as a function of pressure and enthalpy
# Cv_ps	Specific isochoric heat capacity as a function of pressure and entropy.
#
# Speed of sound
# wV_p	Saturated vapour speed of sound
# wL_p	Saturated liquid speed of sound
# wV_T	Saturated vapour speed of sound
# wL_T	Saturated liquid speed of sound
# w_pT	Speed of sound as a function of pressure and temperature.
# w_ph	Speed of sound as a function of pressure and enthalpy
# w_ps	Speed of sound as a function of pressure and entropy.
#
# Viscosity
# Viscosity is not part of IAPWS Steam IF97. Equations from
# "Revised Release on the IAPWS Formulation 1985 for the Viscosity of Ordinary Water Substance", 2003 are used.
# Viscosity in the mixed region (4) is interpolated according to the density. This is not true since it will
# be two phases
# my_pT	Viscosity as a function of pressure and temperature.
# my_ph	Viscosity as a function of pressure and enthalpy
# my_ps	Viscosity as a function of pressure and entropy.
#
# Thermal Conductivity
# Revised release on the IAPWS Formulation 1985 for the Thermal Conductivity of ordinary water substance (IAPWS 1998)
# tcL_p	Saturated vapour thermal conductivity
# tcV_p	Saturated liquid thermal conductivity
# tcL_T	Saturated vapour thermal conductivity
# tcV_T	Saturated liquid thermal conductivity
# tc_pT	Thermal conductivity as a function of pressure and temperature.
# tc_ph	Thermal conductivity as a function of pressure and enthalpy
# tc_hs	Thermal conductivity as a function of enthalpy and entropy
#
# Surface tension
# st_T	Surface tension for two phase water/steam as a function of T
# st_p	Surface tension for two phase water/steam as a function of T
# Vapour fraction
# x_ph	Vapour fraction as a function of pressure and enthalpy
# x_ps	Vapour fraction as a function of pressure and entropy.
#
# Vapour volume fraction
# vx_ph	Vapour volume fraction as a function of pressure and enthalpy
# vx_ps	Vapour volume fraction as a function of pressure and entropy.
@njit(cache=True)
def Tsat_p(p):
    if 0.000611657 <= p <= 22.06395 + 0.001:
        fn_return_value = (T4_p(p))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def Tsat_s(s):
    s = s
    if - 0.0001545495919 < s < 9.155759395:
        fn_return_value = (T4_p(p4_s(s)))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def T_ph(p, h):
    h = h
    select_variable_0 = region_ph(p, h)
    if select_variable_0 == 1:
        fn_return_value = (T1_ph(p, h))
    elif select_variable_0 == 2:
        fn_return_value = (T2_ph(p, h))
    elif select_variable_0 == 3:
        fn_return_value = (T3_ph(p, h))
    elif select_variable_0 == 4:
        fn_return_value = (T4_p(p))
    elif select_variable_0 == 5:
        fn_return_value = (T5_ph(p, h))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def T_ps(p, s):
    s = s
    select_variable_1 = region_ps(p, s)
    if select_variable_1 == 1:
        fn_return_value = (T1_ps(p, s))
    elif select_variable_1 == 2:
        fn_return_value = (T2_ps(p, s))
    elif select_variable_1 == 3:
        fn_return_value = (T3_ps(p, s))
    elif select_variable_1 == 4:
        fn_return_value = (T4_p(p))
    elif select_variable_1 == 5:
        fn_return_value = (T5_ps(p, s))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def T_hs(h, s):
    h = h
    s = s
    select_variable_2 = Region_hs(h, s)
    if select_variable_2 == 1:
        fn_return_value = (T1_ph(p1_hs(h, s), h))
    elif select_variable_2 == 2:
        fn_return_value = (T2_ph(p2_hs(h, s), h))
    elif select_variable_2 == 3:
        fn_return_value = (T3_ph(p3_hs(h, s), h))
    elif select_variable_2 == 4:
        fn_return_value = (T4_hs(h, s))
    elif select_variable_2 == 5:
        fn_return_value = math.nan
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def psat_T(T):
    if 647.096 >= T > 273.15:
        fn_return_value = (p4_T(T))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def psat_s(s):
    s = s
    if - 0.0001545495919 < s < 9.155759395:
        fn_return_value = (p4_s(s))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def p_hs(h, s):
    h = h
    s = s
    select_variable_3 = Region_hs(h, s)
    if select_variable_3 == 1:
        fn_return_value = (p1_hs(h, s))
    elif select_variable_3 == 2:
        fn_return_value = (p2_hs(h, s))
    elif select_variable_3 == 3:
        fn_return_value = (p3_hs(h, s))
    elif select_variable_3 == 4:
        fn_return_value = (p4_T(T4_hs(h, s)))
    elif select_variable_3 == 5:
        fn_return_value = math.nan
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def p_hrho(h, rho):
    # Not valid for water or sumpercritical since water rho does not change very much with p.
    # Uses iteration to find p.
    High_Bound = 100
    Low_Bound = 0.000611657
    p = 10
    rhos = 1 / v_ph(p, h)
    iter_count = 0
    while abs(rho - rhos) > 0.0000001 and iter_count < 1000:
        rhos = 1 / v_ph(p, h)
        if rhos >= rho:
            High_Bound = p
        else:
            Low_Bound = p
        p = (Low_Bound + High_Bound) / 2
        iter_count += 1

    if iter_count >= 1000:
        p = math.nan

    fn_return_value = p
    return fn_return_value


@njit(cache=True)
def hV_p(p):
    if 0.000611657 < p < 22.06395:
        fn_return_value = (h4V_p(p))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def hL_p(p):
    if 0.000611657 < p < 22.06395:
        fn_return_value = (h4L_p(p))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def hV_T(T):
    if 273.15 < T < 647.096:
        fn_return_value = (h4V_p(p4_T(T)))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def hL_T(T):
    if 273.15 < T < 647.096:
        fn_return_value = (h4L_p(p4_T(T)))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def h_pT(p, T):
    select_variable_4 = region_pT(p, T)
    if select_variable_4 == 1:
        fn_return_value = (h1_pT(p, T))
    elif select_variable_4 == 2:
        fn_return_value = (h2_pT(p, T))
    elif select_variable_4 == 3:
        fn_return_value = (h3_pT(p, T))
    elif select_variable_4 == 4:
        fn_return_value = math.nan
    elif select_variable_4 == 5:
        fn_return_value = (h5_pT(p, T))
    else:
        fn_return_value = math.nan

    return fn_return_value


@njit(cache=True)
def h_ps(p, s):
    s = s
    select_variable_5 = region_ps(p, s)
    if select_variable_5 == 1:
        fn_return_value = (h1_pT(p, T1_ps(p, s)))
    elif select_variable_5 == 2:
        fn_return_value = (h2_pT(p, T2_ps(p, s)))
    elif select_variable_5 == 3:
        fn_return_value = (h3_rhoT(1 / v3_ps(p, s), T3_ps(p, s)))
    elif select_variable_5 == 4:
        xs = x4_ps(p, s)
        fn_return_value = (xs * h4V_p(p) + (1 - xs) * h4L_p(p))
    elif select_variable_5 == 5:
        fn_return_value = (h5_pT(p, T5_ps(p, s)))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def h_px(p, x):
    x = x
    if x > 1 or x < 0 or p >= 22.064:
        fn_return_value = math.nan
        return fn_return_value
    hL = h4L_p(p)
    hV = h4V_p(p)
    fn_return_value = (hL + x * (hV - hL))
    return fn_return_value


@njit(cache=True)
def h_Tx(T, x):
    x = x
    if x > 1 or x < 0 or T >= 647.096:
        fn_return_value = math.nan
        return fn_return_value
    p = p4_T(T)
    hL = h4L_p(p)
    hV = h4V_p(p)
    fn_return_value = (hL + x * (hV - hL))
    return fn_return_value


@njit(cache=True)
def h_prho(p, rho):
    rho = 1 / (1 / rho)
    select_variable_6 = Region_prho(p, rho)
    if select_variable_6 == 1:
        fn_return_value = (h1_pT(p, T1_prho(p, rho)))
    elif select_variable_6 == 2:
        fn_return_value = (h2_pT(p, T2_prho(p, rho)))
    elif select_variable_6 == 3:
        fn_return_value = (h3_rhoT(rho, T3_prho(p, rho)))
    elif select_variable_6 == 4:
        if p < 16.529:
            vV = v2_pT(p, T4_p(p))
            vL = v1_pT(p, T4_p(p))
        else:
            vV = v3_ph(p, h4V_p(p))
            vL = v3_ph(p, h4L_p(p))
        hV = h4V_p(p)
        hL = h4L_p(p)
        x = (1 / rho - vL) / (vV - vL)
        fn_return_value = ((1 - x) * hL + x * hV)
    elif select_variable_6 == 5:
        fn_return_value = (h5_pT(p, T5_prho(p, rho)))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def vV_p(p):
    if 0.000611657 < p < 22.06395:
        if p < 16.529:
            fn_return_value = (v2_pT(p, T4_p(p)))
        else:
            fn_return_value = (v3_ph(p, h4V_p(p)))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def vL_p(p):
    if 0.000611657 < p < 22.06395:
        if p < 16.529:
            fn_return_value = (v1_pT(p, T4_p(p)))
        else:
            fn_return_value = (v3_ph(p, h4L_p(p)))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def vV_T(T):
    if 273.15 < T < 647.096:
        if T <= 623.15:
            fn_return_value = (v2_pT(p4_T(T), T))
        else:
            fn_return_value = (v3_ph(p4_T(T), h4V_p(p4_T(T))))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def vL_T(T):
    if 273.15 < T < 647.096:
        if T <= 623.15:
            fn_return_value = (v1_pT(p4_T(T), T))
        else:
            fn_return_value = (v3_ph(p4_T(T), h4L_p(p4_T(T))))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def v_pT(p, T):
    select_variable_7 = region_pT(p, T)
    if select_variable_7 == 1:
        fn_return_value = (v1_pT(p, T))
    elif select_variable_7 == 2:
        fn_return_value = (v2_pT(p, T))
    elif select_variable_7 == 3:
        fn_return_value = (v3_ph(p, h3_pT(p, T)))
    elif select_variable_7 == 4:
        fn_return_value = math.nan
    elif select_variable_7 == 5:
        fn_return_value = (v5_pT(p, T))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def v_ph(p, h):
    h = h
    select_variable_8 = region_ph(p, h)
    if select_variable_8 == 1:
        fn_return_value = (v1_pT(p, T1_ph(p, h)))
    elif select_variable_8 == 2:
        fn_return_value = (v2_pT(p, T2_ph(p, h)))
    elif select_variable_8 == 3:
        fn_return_value = (v3_ph(p, h))
    elif select_variable_8 == 4:
        xs = x4_ph(p, h)
        if p < 16.529:
            v4V = v2_pT(p, T4_p(p))
            v4L = v1_pT(p, T4_p(p))
        else:
            v4V = v3_ph(p, h4V_p(p))
            v4L = v3_ph(p, h4L_p(p))
        fn_return_value = (xs * v4V + (1 - xs) * v4L)
    elif select_variable_8 == 5:
        fn_return_value = (v5_pT(p, T5_ph(p, h)))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def v_ps(p, s):
    s = s
    select_variable_9 = region_ps(p, s)
    if select_variable_9 == 1:
        fn_return_value = (v1_pT(p, T1_ps(p, s)))
    elif select_variable_9 == 2:
        fn_return_value = (v2_pT(p, T2_ps(p, s)))
    elif select_variable_9 == 3:
        fn_return_value = (v3_ps(p, s))
    elif select_variable_9 == 4:
        xs = x4_ps(p, s)
        if p < 16.529:
            v4V = v2_pT(p, T4_p(p))
            v4L = v1_pT(p, T4_p(p))
        else:
            v4V = v3_ph(p, h4V_p(p))
            v4L = v3_ph(p, h4L_p(p))
        fn_return_value = (xs * v4V + (1 - xs) * v4L)
    elif select_variable_9 == 5:
        fn_return_value = (v5_pT(p, T5_ps(p, s)))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def rhoV_p(p):
    fn_return_value = 1 / vV_p(p)
    return fn_return_value


@njit(cache=True)
def rhoL_p(p):
    fn_return_value = 1 / vL_p(p)
    return fn_return_value


@njit(cache=True)
def rhoL_T(T):
    fn_return_value = 1 / vL_T(T)
    return fn_return_value


@njit(cache=True)
def rhoV_T(T):
    fn_return_value = 1 / vV_T(T)
    return fn_return_value


@njit(cache=True)
def rho_pT(p, T):
    fn_return_value = 1 / v_pT(p, T)
    return fn_return_value


@njit(cache=True)
def rho_ph(p, h):
    fn_return_value = 1 / v_ph(p, h)
    return fn_return_value


@njit(cache=True)
def rho_ps(p, s):
    fn_return_value = 1 / v_ps(p, s)
    return fn_return_value


@njit(cache=True)
def sV_p(p):
    if 0.000611657 < p < 22.06395:
        if p < 16.529:
            fn_return_value = (s2_pT(p, T4_p(p)))
        else:
            fn_return_value = (s3_rhoT(1 / (v3_ph(p, h4V_p(p))), T4_p(p)))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def sL_p(p):
    if 0.000611657 < p < 22.06395:
        if p < 16.529:
            fn_return_value = (s1_pT(p, T4_p(p)))
        else:
            fn_return_value = (s3_rhoT(1 / (v3_ph(p, h4L_p(p))), T4_p(p)))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def sV_T(T):
    if 273.15 < T < 647.096:
        if T <= 623.15:
            fn_return_value = (s2_pT(p4_T(T), T))
        else:
            fn_return_value = (s3_rhoT(1 / (v3_ph(p4_T(T), h4V_p(p4_T(T)))), T))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def sL_T(T):
    if 273.15 < T < 647.096:
        if T <= 623.15:
            fn_return_value = (s1_pT(p4_T(T), T))
        else:
            fn_return_value = (s3_rhoT(1 / (v3_ph(p4_T(T), h4L_p(p4_T(T)))), T))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def s_pT(p, T):
    select_variable_10 = region_pT(p, T)
    if select_variable_10 == 1:
        fn_return_value = (s1_pT(p, T))
    elif select_variable_10 == 2:
        fn_return_value = (s2_pT(p, T))
    elif select_variable_10 == 3:
        fn_return_value = (s3_rhoT(1 / v3_ph(p, h3_pT(p, T)), T))
    elif select_variable_10 == 4:
        fn_return_value = math.nan
    elif select_variable_10 == 5:
        fn_return_value = (s5_pT(p, T))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def s_ph(p, h):
    h = h
    select_variable_11 = region_ph(p, h)
    if select_variable_11 == 1:
        fn_return_value = (s1_pT(p, T1_ph(p, h)))
    elif select_variable_11 == 2:
        fn_return_value = (s2_pT(p, T2_ph(p, h)))
    elif select_variable_11 == 3:
        fn_return_value = (s3_rhoT(1 / v3_ph(p, h), T3_ph(p, h)))
    elif select_variable_11 == 4:
        Ts = T4_p(p)
        xs = x4_ph(p, h)
        if p < 16.529:
            s4V = s2_pT(p, Ts)
            s4L = s1_pT(p, Ts)
        else:
            v4V = v3_ph(p, h4V_p(p))
            s4V = s3_rhoT(1 / v4V, Ts)
            v4L = v3_ph(p, h4L_p(p))
            s4L = s3_rhoT(1 / v4L, Ts)
        fn_return_value = (xs * s4V + (1 - xs) * s4L)
    elif select_variable_11 == 5:
        fn_return_value = (s5_pT(p, T5_ph(p, h)))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def uV_p(p):
    if 0.000611657 < p < 22.06395:
        if p < 16.529:
            fn_return_value = (u2_pT(p, T4_p(p)))
        else:
            fn_return_value = (u3_rhoT(1 / (v3_ph(p, h4V_p(p))), T4_p(p)))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def uL_p(p):
    if 0.000611657 < p < 22.06395:
        if p < 16.529:
            fn_return_value = (u1_pT(p, T4_p(p)))
        else:
            fn_return_value = (u3_rhoT(1 / (v3_ph(p, h4L_p(p))), T4_p(p)))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def uV_T(T):
    if 273.15 < T < 647.096:
        if T <= 623.15:
            fn_return_value = (u2_pT(p4_T(T), T))
        else:
            fn_return_value = (u3_rhoT(1 / (v3_ph(p4_T(T), h4V_p(p4_T(T)))), T))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def uL_T(T):
    if 273.15 < T < 647.096:
        if T <= 623.15:
            fn_return_value = (u1_pT(p4_T(T), T))
        else:
            fn_return_value = (u3_rhoT(1 / (v3_ph(p4_T(T), h4L_p(p4_T(T)))), T))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def u_pT(p, T):
    select_variable_12 = region_pT(p, T)
    if select_variable_12 == 1:
        fn_return_value = (u1_pT(p, T))
    elif select_variable_12 == 2:
        fn_return_value = (u2_pT(p, T))
    elif select_variable_12 == 3:
        fn_return_value = (u3_rhoT(1 / v3_ph(p, h3_pT(p, T)), T))
    elif select_variable_12 == 4:
        fn_return_value = math.nan
    elif select_variable_12 == 5:
        fn_return_value = (u5_pT(p, T))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def u_ph(p, h):
    h = h
    select_variable_13 = region_ph(p, h)
    if select_variable_13 == 1:
        fn_return_value = (u1_pT(p, T1_ph(p, h)))
    elif select_variable_13 == 2:
        fn_return_value = (u2_pT(p, T2_ph(p, h)))
    elif select_variable_13 == 3:
        fn_return_value = (u3_rhoT(1 / v3_ph(p, h), T3_ph(p, h)))
    elif select_variable_13 == 4:
        Ts = T4_p(p)
        xs = x4_ph(p, h)
        if p < 16.529:
            u4v = u2_pT(p, Ts)
            u4L = u1_pT(p, Ts)
        else:
            v4V = v3_ph(p, h4V_p(p))
            u4v = u3_rhoT(1 / v4V, Ts)
            v4L = v3_ph(p, h4L_p(p))
            u4L = u3_rhoT(1 / v4L, Ts)
        fn_return_value = (xs * u4v + (1 - xs) * u4L)
    elif select_variable_13 == 5:
        Ts = T5_ph(p, h)
        fn_return_value = (u5_pT(p, Ts))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def u_ps(p, s):
    s = s
    select_variable_14 = region_ps(p, s)
    if select_variable_14 == 1:
        fn_return_value = (u1_pT(p, T1_ps(p, s)))
    elif select_variable_14 == 2:
        fn_return_value = (u2_pT(p, T2_ps(p, s)))
    elif select_variable_14 == 3:
        fn_return_value = (u3_rhoT(1 / v3_ps(p, s), T3_ps(p, s)))
    elif select_variable_14 == 4:
        if p < 16.529:
            uLp = u1_pT(p, T4_p(p))
            uVp = u2_pT(p, T4_p(p))
        else:
            uLp = u3_rhoT(1 / (v3_ph(p, h4L_p(p))), T4_p(p))
            uVp = u3_rhoT(1 / (v3_ph(p, h4V_p(p))), T4_p(p))
        x = x4_ps(p, s)
        fn_return_value = (x * uVp + (1 - x) * uLp)
    elif select_variable_14 == 5:
        fn_return_value = (u5_pT(p, T5_ps(p, s)))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def CpV_p(p):
    if 0.000611657 < p < 22.06395:
        if p < 16.529:
            fn_return_value = (Cp2_pT(p, T4_p(p)))
        else:
            fn_return_value = (Cp3_rhoT(1 / (v3_ph(p, h4V_p(p))), T4_p(p)))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def CpL_p(p):
    if 0.000611657 < p < 22.06395:
        if p < 16.529:
            fn_return_value = (Cp1_pT(p, T4_p(p)))
        else:
            T = T4_p(p)
            h = h4L_p(p)
            v = v3_ph(p, h4L_p(p))
            fn_return_value = (Cp3_rhoT(1 / (v3_ph(p, h4L_p(p))), T4_p(p)))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def CpV_T(T):
    if 273.15 < T < 647.096:
        if T <= 623.15:
            fn_return_value = (Cp2_pT(p4_T(T), T))
        else:
            fn_return_value = (Cp3_rhoT(1 / (v3_ph(p4_T(T), h4V_p(p4_T(T)))), T))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def CpL_T(T):
    if 273.15 < T < 647.096:
        if T <= 623.15:
            fn_return_value = (Cp1_pT(p4_T(T), T))
        else:
            fn_return_value = (Cp3_rhoT(1 / (v3_ph(p4_T(T), h4L_p(p4_T(T)))), T))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def Cp_pT(p, T):
    select_variable_15 = region_pT(p, T)
    if select_variable_15 == 1:
        fn_return_value = (Cp1_pT(p, T))
    elif select_variable_15 == 2:
        fn_return_value = (Cp2_pT(p, T))
    elif select_variable_15 == 3:
        fn_return_value = (Cp3_rhoT(1 / v3_ph(p, h3_pT(p, T)), T))
    elif select_variable_15 == 4:
        fn_return_value = math.nan
    elif select_variable_15 == 5:
        fn_return_value = (Cp5_pT(p, T))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def Cp_ph(p, h):
    h = h
    select_variable_16 = region_ph(p, h)
    if select_variable_16 == 1:
        fn_return_value = (Cp1_pT(p, T1_ph(p, h)))
    elif select_variable_16 == 2:
        fn_return_value = (Cp2_pT(p, T2_ph(p, h)))
    elif select_variable_16 == 3:
        fn_return_value = (Cp3_rhoT(1 / v3_ph(p, h), T3_ph(p, h)))
    elif select_variable_16 == 4:
        fn_return_value = math.nan
    elif select_variable_16 == 5:
        fn_return_value = (Cp5_pT(p, T5_ph(p, h)))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def Cp_ps(p, s):
    s = s
    select_variable_17 = region_ps(p, s)
    if select_variable_17 == 1:
        fn_return_value = (Cp1_pT(p, T1_ps(p, s)))
    elif select_variable_17 == 2:
        fn_return_value = (Cp2_pT(p, T2_ps(p, s)))
    elif select_variable_17 == 3:
        fn_return_value = (Cp3_rhoT(1 / v3_ps(p, s), T3_ps(p, s)))
    elif select_variable_17 == 4:
        fn_return_value = math.nan
    elif select_variable_17 == 5:
        fn_return_value = (Cp5_pT(p, T5_ps(p, s)))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def CvV_p(p):
    if 0.000611657 < p < 22.06395:
        if p < 16.529:
            fn_return_value = (Cv2_pT(p, T4_p(p)))
        else:
            fn_return_value = (Cv3_rhoT(1 / (v3_ph(p, h4V_p(p))), T4_p(p)))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def CvL_p(p):
    if 0.000611657 < p < 22.06395:
        if p < 16.529:
            fn_return_value = (Cv1_pT(p, T4_p(p)))
        else:
            fn_return_value = (Cv3_rhoT(1 / (v3_ph(p, h4L_p(p))), T4_p(p)))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def CvV_T(T):
    if 273.15 < T < 647.096:
        if T <= 623.15:
            fn_return_value = (Cv2_pT(p4_T(T), T))
        else:
            fn_return_value = (Cv3_rhoT(1 / (v3_ph(p4_T(T), h4V_p(p4_T(T)))), T))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def CvL_T(T):
    if 273.15 < T < 647.096:
        if T <= 623.15:
            fn_return_value = (Cv1_pT(p4_T(T), T))
        else:
            fn_return_value = (Cv3_rhoT(1 / (v3_ph(p4_T(T), h4L_p(p4_T(T)))), T))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def Cv_pT(p, T):
    select_variable_18 = region_pT(p, T)
    if select_variable_18 == 1:
        fn_return_value = (Cv1_pT(p, T))
    elif select_variable_18 == 2:
        fn_return_value = (Cv2_pT(p, T))
    elif select_variable_18 == 3:
        fn_return_value = (Cv3_rhoT(1 / v3_ph(p, h3_pT(p, T)), T))
    elif select_variable_18 == 4:
        fn_return_value = math.nan
    elif select_variable_18 == 5:
        fn_return_value = (Cv5_pT(p, T))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def Cv_ph(p, h):
    h = h
    select_variable_19 = region_ph(p, h)
    if select_variable_19 == 1:
        fn_return_value = (Cv1_pT(p, T1_ph(p, h)))
    elif select_variable_19 == 2:
        fn_return_value = (Cv2_pT(p, T2_ph(p, h)))
    elif select_variable_19 == 3:
        fn_return_value = (Cv3_rhoT(1 / v3_ph(p, h), T3_ph(p, h)))
    elif select_variable_19 == 4:
        fn_return_value = math.nan
    elif select_variable_19 == 5:
        fn_return_value = (Cv5_pT(p, T5_ph(p, h)))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def Cv_ps(p, s):
    s = s
    select_variable_20 = region_ps(p, s)
    if select_variable_20 == 1:
        fn_return_value = (Cv1_pT(p, T1_ps(p, s)))
    elif select_variable_20 == 2:
        fn_return_value = (Cv2_pT(p, T2_ps(p, s)))
    elif select_variable_20 == 3:
        fn_return_value = (Cv3_rhoT(1 / v3_ps(p, s), T3_ps(p, s)))
    elif select_variable_20 == 4:
        fn_return_value = math.nan
    elif select_variable_20 == 5:
        fn_return_value = (Cv5_pT(p, T5_ps(p, s)))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def wV_p(p):
    if 0.000611657 < p < 22.06395:
        if p < 16.529:
            fn_return_value = (w2_pT(p, T4_p(p)))
        else:
            fn_return_value = (w3_rhoT(1 / (v3_ph(p, h4V_p(p))), T4_p(p)))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def wL_p(p):
    if 0.000611657 < p < 22.06395:
        if p < 16.529:
            fn_return_value = (w1_pT(p, T4_p(p)))
        else:
            fn_return_value = (w3_rhoT(1 / (v3_ph(p, h4L_p(p))), T4_p(p)))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def wV_T(T):
    if 273.15 < T < 647.096:
        if T <= 623.15:
            fn_return_value = (w2_pT(p4_T(T), T))
        else:
            fn_return_value = (w3_rhoT(1 / (v3_ph(p4_T(T), h4V_p(p4_T(T)))), T))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def wL_T(T):
    if 273.15 < T < 647.096:
        if T <= 623.15:
            fn_return_value = (w1_pT(p4_T(T), T))
        else:
            fn_return_value = (w3_rhoT(1 / (v3_ph(p4_T(T), h4L_p(p4_T(T)))), T))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def w_pT(p, T):
    select_variable_21 = region_pT(p, T)
    if select_variable_21 == 1:
        fn_return_value = (w1_pT(p, T))
    elif select_variable_21 == 2:
        fn_return_value = (w2_pT(p, T))
    elif select_variable_21 == 3:
        fn_return_value = (w3_rhoT(1 / v3_ph(p, h3_pT(p, T)), T))
    elif select_variable_21 == 4:
        fn_return_value = math.nan
    elif select_variable_21 == 5:
        fn_return_value = (w5_pT(p, T))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def w_ph(p, h):
    h = h
    select_variable_22 = region_ph(p, h)
    if select_variable_22 == 1:
        fn_return_value = (w1_pT(p, T1_ph(p, h)))
    elif select_variable_22 == 2:
        fn_return_value = (w2_pT(p, T2_ph(p, h)))
    elif select_variable_22 == 3:
        fn_return_value = (w3_rhoT(1 / v3_ph(p, h), T3_ph(p, h)))
    elif select_variable_22 == 4:
        fn_return_value = math.nan
    elif select_variable_22 == 5:
        fn_return_value = (w5_pT(p, T5_ph(p, h)))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def w_ps(p, s):
    s = s
    select_variable_23 = region_ps(p, s)
    if select_variable_23 == 1:
        fn_return_value = (w1_pT(p, T1_ps(p, s)))
    elif select_variable_23 == 2:
        fn_return_value = (w2_pT(p, T2_ps(p, s)))
    elif select_variable_23 == 3:
        fn_return_value = (w3_rhoT(1 / v3_ps(p, s), T3_ps(p, s)))
    elif select_variable_23 == 4:
        fn_return_value = math.nan
    elif select_variable_23 == 5:
        fn_return_value = (w5_pT(p, T5_ps(p, s)))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def my_pT(p, T):
    select_variable_24 = region_pT(p, T)
    if select_variable_24 == 4:
        fn_return_value = math.nan
    elif (select_variable_24 == 1) or (select_variable_24 == 2) or (select_variable_24 == 3) or (
        select_variable_24 == 5):
        fn_return_value = (my_AllRegions_pT(p, T))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def my_ph(p, h):
    h = h
    select_variable_25 = region_ph(p, h)
    if (select_variable_25 == 1) or (select_variable_25 == 2) or (select_variable_25 == 3) or (select_variable_25 == 5):
        fn_return_value = (my_AllRegions_ph(p, h))
    elif select_variable_25 == 4:
        fn_return_value = math.nan
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def my_ps(p, s):
    fn_return_value = my_ph(p, h_ps(p, s))
    return fn_return_value


@njit(cache=True)
def Pr_pT(p, T):
    Cp = (Cp_pT(p, T))
    my = (my_pT(p, T))
    tc = (tc_pT(p, T))
    fn_return_value = Cp * 1000 * my / tc
    return fn_return_value


@njit(cache=True)
def Pr_ph(p, h):
    Cp = (Cp_ph(p, h))
    my = (my_ph(p, h))
    tc = (tc_ph(p, h))
    fn_return_value = Cp * 1000 * my / tc
    return fn_return_value


@njit(cache=True)
def Kappa_pT(p, T):
    Cp = Cp_pT(p, T)
    Cv = Cv_pT(p, T)
    fn_return_value = Cp / Cv
    return fn_return_value


@njit(cache=True)
def Kappa_ph(p, h):
    Cv = Cv_ph(p, h)
    Cp = Cp_ph(p, h)
    fn_return_value = Cp / Cv
    return fn_return_value


@njit(cache=True)
def st_t(T):
    fn_return_value = (Surface_Tension_T(T))
    return fn_return_value


@njit(cache=True)
def st_p(p):
    T = Tsat_p(p)
    fn_return_value = (Surface_Tension_T(T))
    return fn_return_value


@njit(cache=True)
def tcL_p(p):
    T = Tsat_p(p)
    v = vL_p(p)
    fn_return_value = (tc_ptrho(p, T, 1 / v))
    return fn_return_value


@njit(cache=True)
def tcV_p(p):
    T = Tsat_p(p)
    v = vV_p(p)
    fn_return_value = tc_ptrho(p, T, 1 / v)
    return fn_return_value


@njit(cache=True)
def tcL_T(T):
    p = psat_T(T)
    v = vL_T(T)
    fn_return_value = tc_ptrho(p, T, 1 / v)
    return fn_return_value


@njit(cache=True)
def tcV_T(T):
    p = psat_T(T)
    v = vV_T(T)
    fn_return_value = tc_ptrho(p, T, 1 / v)
    return fn_return_value


@njit(cache=True)
def tc_pT(p, T):
    v = v_pT(p, T)
    fn_return_value = tc_ptrho(p, T, 1 / v)
    return fn_return_value


@njit(cache=True)
def tc_ph(p, h):
    v = v_ph(p, h)
    T = T_ph(p, h)
    fn_return_value = tc_ptrho(p, T, 1 / v)
    return fn_return_value


@njit(cache=True)
def tc_hs(h, s):
    p = p_hs(h, s)
    v = v_ph(p, h)
    T = T_ph(p, h)
    fn_return_value = tc_ptrho(p, T, 1 / v)
    return fn_return_value


@njit(cache=True)
def x_ph(p, h):
    if 0.000611657 < p < 22.06395:
        fn_return_value = (x4_ph(p, h))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def x_ps(p, s):
    if 0.000611657 < p < 22.06395:
        fn_return_value = (x4_ps(p, s))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def vx_ph(p, h):
    if 0.000611657 < p < 22.06395:
        if p < 16.529:
            vL = v1_pT(p, T4_p(p))
            vV = v2_pT(p, T4_p(p))
        else:
            vL = v3_ph(p, h4L_p(p))
            vV = v3_ph(p, h4V_p(p))
        xs = x4_ph(p, h)
        fn_return_value = (xs * vV / (xs * vV + (1 - xs) * vL))
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def vx_ps(p, s):
    #
    # s = (s)
    if 0.000611657 < p < 22.06395:
        if p < 16.529:
            vL = v1_pT(p, T4_p(p))
            vV = v2_pT(p, T4_p(p))
        else:
            vL = v3_ph(p, h4L_p(p))
            vV = v3_ph(p, h4V_p(p))
        xs = x4_ps(p, s)
        fn_return_value = (xs * vV / (xs * vV + (1 - xs) * vL))
    else:
        fn_return_value = math.nan
    return fn_return_value


# ***********************************************************************************************************
# *2 IAPWS IF 97 Calling functions                                                                          *
# ***********************************************************************************************************
#
# ***********************************************************************************************************
# *2.1 Functions for region 1

@njit(cache=True)
def v1_pT(p, T):
    R = 0.461526
    reg1_I1 = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0,
         3.0, 3.0, 4.0, 4.0, 4.0, 5.0, 8.0, 8.0, 21.0, 23.0, 29.0, 30.0, 31.0, 32.0])
    reg1_J1 = np.array(
        [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, -9.0, -7.0, -1.0, 0.0, 1.0, 3.0, -3.0, 0.0, 1.0, 3.0, 17.0,
         -4.0, 0.0, 6.0, -5.0, -2.0, 10.0, -8.0, -11.0, -6.0, -29.0, -31.0, -38.0, -39.0, -40.0, -41.0])
    reg1_n1 = np.array([0.14632971213167, -0.84548187169114, -3.756360367204, 3.3855169168385, -0.95791963387872,
                        0.15772038513228, -0.016616417199501, 8.1214629983568E-04, 2.8319080123804E-04,
                        -6.0706301565874E-04,
                        -0.018990068218419, -0.032529748770505, -0.021841717175414, -5.283835796993E-05,
                        -4.7184321073267E-04, -3.0001780793026E-04, 4.7661393906987E-05, -4.4141845330846E-06,
                        -7.2694996297594E-16, -3.1679644845054E-05, -2.8270797985312E-06, -8.5205128120103E-10,
                        -2.2425281908E-06, -6.5171222895601E-07, - 1.4341729937924E-13, -4.0516996860117E-07,
                        -1.2734301741641E-09, -1.7424871230634E-10, -6.8762131295531E-19, 1.4478307828521E-20,
                        2.6335781662795E-23, -1.1947622640071E-23, 1.8228094581404E-24, -9.3537087292458E-26])

    # Release on the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam,
    # September 1997
    # 5 Equations for Region 1, Section. 5.1 Basic Equation
    # Eqution 7, Table 3, Page 6
    ps = p / 16.53
    tau = 1386 / T
    g_p = 0
    for i in range(34):
        g_p = g_p - reg1_n1[i] * reg1_I1[i] * (7.1 - ps) ** (reg1_I1[i] - 1) * (tau - 1.222) ** reg1_J1[i]
    fn_return_value = R * T / p * ps * g_p / 1000
    return fn_return_value


@njit(cache=True)
def h1_pT(p, T):
    R = 0.461526
    reg1_I1 = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0,
         3.0, 3.0, 4.0, 4.0, 4.0, 5.0, 8.0, 8.0, 21.0, 23.0, 29.0, 30.0, 31.0, 32.0])
    reg1_J1 = np.array(
        [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, -9.0, -7.0, -1.0, 0.0, 1.0, 3.0, -3.0, 0.0, 1.0, 3.0, 17.0,
         -4.0, 0.0, 6.0, -5.0, -2.0, 10.0, -8.0, -11.0, -6.0, -29.0, -31.0, -38.0, -39.0, -40.0, -41.0])
    reg1_n1 = np.array([0.14632971213167, -0.84548187169114, -3.756360367204, 3.3855169168385, -0.95791963387872,
                        0.15772038513228, -0.016616417199501, 8.1214629983568E-04, 2.8319080123804E-04,
                        -6.0706301565874E-04,
                        -0.018990068218419, -0.032529748770505, -0.021841717175414, -5.283835796993E-05,
                        -4.7184321073267E-04, -3.0001780793026E-04, 4.7661393906987E-05, -4.4141845330846E-06,
                        -7.2694996297594E-16, -3.1679644845054E-05, -2.8270797985312E-06, -8.5205128120103E-10,
                        -2.2425281908E-06, -6.5171222895601E-07, - 1.4341729937924E-13, -4.0516996860117E-07,
                        -1.2734301741641E-09, -1.7424871230634E-10, -6.8762131295531E-19, 1.4478307828521E-20,
                        2.6335781662795E-23, -1.1947622640071E-23, 1.8228094581404E-24, -9.3537087292458E-26])

    # Release on the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam,
    # September 1997
    # 5 Equations for Region 1, Section. 5.1 Basic Equation
    # Eqution 7, Table 3, Page 6
    p = p / 16.53
    tau = 1386 / T
    g_t = 0
    for i in range(34):
        g_t = g_t + (reg1_n1[i] * (7.1 - p) ** reg1_I1[i] * reg1_J1[i] * (tau - 1.222) ** (reg1_J1[i] - 1))
    fn_return_value = R * T * tau * g_t
    return fn_return_value


@njit(cache=True)
def u1_pT(p, T):
    R = 0.461526
    reg1_I1 = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0,
         3.0, 3.0, 4.0, 4.0, 4.0, 5.0, 8.0, 8.0, 21.0, 23.0, 29.0, 30.0, 31.0, 32.0])
    reg1_J1 = np.array(
        [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, -9.0, -7.0, -1.0, 0.0, 1.0, 3.0, -3.0, 0.0, 1.0, 3.0, 17.0,
         -4.0, 0.0, 6.0, -5.0, -2.0, 10.0, -8.0, -11.0, -6.0, -29.0, -31.0, -38.0, -39.0, -40.0, -41.0])
    reg1_n1 = np.array([0.14632971213167, -0.84548187169114, -3.756360367204, 3.3855169168385, -0.95791963387872,
                        0.15772038513228, -0.016616417199501, 8.1214629983568E-04, 2.8319080123804E-04,
                        -6.0706301565874E-04,
                        -0.018990068218419, -0.032529748770505, -0.021841717175414, -5.283835796993E-05,
                        -4.7184321073267E-04, -3.0001780793026E-04, 4.7661393906987E-05, -4.4141845330846E-06,
                        -7.2694996297594E-16, -3.1679644845054E-05, -2.8270797985312E-06, -8.5205128120103E-10,
                        -2.2425281908E-06, -6.5171222895601E-07, - 1.4341729937924E-13, -4.0516996860117E-07,
                        -1.2734301741641E-09, -1.7424871230634E-10, -6.8762131295531E-19, 1.4478307828521E-20,
                        2.6335781662795E-23, -1.1947622640071E-23, 1.8228094581404E-24, -9.3537087292458E-26])

    # Release on the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam,
    # September 1997
    # 5 Equations for Region 1, Section. 5.1 Basic Equation
    # Eqution 7, Table 3, Page 6
    p = p / 16.53
    tau = 1386 / T
    g_t = 0
    g_p = 0
    for i in range(34):
        g_p = g_p - reg1_n1[i] * reg1_I1[i] * (7.1 - p) ** (reg1_I1[i] - 1) * (tau - 1.222) ** reg1_J1[i]
        g_t = g_t + (reg1_n1[i] * (7.1 - p) ** reg1_I1[i] * reg1_J1[i] * (tau - 1.222) ** (reg1_J1[i] - 1))

    fn_return_value = R * T * (tau * g_t - p * g_p)
    return fn_return_value


@njit(cache=True)
def s1_pT(p, T):
    R = 0.461526
    reg1_I1 = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0,
         3.0, 3.0, 4.0, 4.0, 4.0, 5.0, 8.0, 8.0, 21.0, 23.0, 29.0, 30.0, 31.0, 32.0])
    reg1_J1 = np.array(
        [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, -9.0, -7.0, -1.0, 0.0, 1.0, 3.0, -3.0, 0.0, 1.0, 3.0, 17.0,
         -4.0, 0.0, 6.0, -5.0, -2.0, 10.0, -8.0, -11.0, -6.0, -29.0, -31.0, -38.0, -39.0, -40.0, -41.0])
    reg1_n1 = np.array([0.14632971213167, -0.84548187169114, -3.756360367204, 3.3855169168385, -0.95791963387872,
                        0.15772038513228, -0.016616417199501, 8.1214629983568E-04, 2.8319080123804E-04,
                        -6.0706301565874E-04,
                        -0.018990068218419, -0.032529748770505, -0.021841717175414, -5.283835796993E-05,
                        -4.7184321073267E-04, -3.0001780793026E-04, 4.7661393906987E-05, -4.4141845330846E-06,
                        -7.2694996297594E-16, -3.1679644845054E-05, -2.8270797985312E-06, -8.5205128120103E-10,
                        -2.2425281908E-06, -6.5171222895601E-07, - 1.4341729937924E-13, -4.0516996860117E-07,
                        -1.2734301741641E-09, -1.7424871230634E-10, -6.8762131295531E-19, 1.4478307828521E-20,
                        2.6335781662795E-23, -1.1947622640071E-23, 1.8228094581404E-24, -9.3537087292458E-26])

    # Release on the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam,
    # September 1997
    # 5 Equations for Region 1, Section. 5.1 Basic Equation
    # Eqution 7, Table 3, Page 6
    p = p / 16.53
    T = 1386 / T
    g = 0
    g_t = 0
    for i in range(34):
        g_t = g_t + (reg1_n1[i] * (7.1 - p) ** reg1_I1[i] * reg1_J1[i] * (T - 1.222) ** (reg1_J1[i] - 1))
        g = g + reg1_n1[i] * (7.1 - p) ** reg1_I1[i] * (T - 1.222) ** reg1_J1[i]
    fn_return_value = R * T * g_t - R * g
    return fn_return_value


@njit(cache=True)
def Cp1_pT(p, T):
    R = 0.461526
    # Release on the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam,
    # September 1997
    # 5 Equations for Region 1, Section. 5.1 Basic Equation
    # Eqution 7, Table 3, Page 6
    reg1_I1 = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0,
         3.0, 3.0, 4.0, 4.0, 4.0, 5.0, 8.0, 8.0, 21.0, 23.0, 29.0, 30.0, 31.0, 32.0])
    reg1_J1 = np.array(
        [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, -9.0, -7.0, -1.0, 0.0, 1.0, 3.0, -3.0, 0.0, 1.0, 3.0, 17.0,
         -4.0, 0.0, 6.0, -5.0, -2.0, 10.0, -8.0, -11.0, -6.0, -29.0, -31.0, -38.0, -39.0, -40.0, -41.0])
    reg1_n1 = np.array([0.14632971213167, -0.84548187169114, -3.756360367204, 3.3855169168385, -0.95791963387872,
                        0.15772038513228, -0.016616417199501, 8.1214629983568E-04, 2.8319080123804E-04,
                        -6.0706301565874E-04,
                        -0.018990068218419, -0.032529748770505, -0.021841717175414, -5.283835796993E-05,
                        -4.7184321073267E-04, -3.0001780793026E-04, 4.7661393906987E-05, -4.4141845330846E-06,
                        -7.2694996297594E-16, -3.1679644845054E-05, -2.8270797985312E-06, -8.5205128120103E-10,
                        -2.2425281908E-06, -6.5171222895601E-07, - 1.4341729937924E-13, -4.0516996860117E-07,
                        -1.2734301741641E-09, -1.7424871230634E-10, -6.8762131295531E-19, 1.4478307828521E-20,
                        2.6335781662795E-23, -1.1947622640071E-23, 1.8228094581404E-24, -9.3537087292458E-26])

    p = p / 16.53
    T = 1386 / T
    G_tt = 0
    for i in range(34):
        G_tt = G_tt + (
            reg1_n1[i] * (7.1 - p) ** reg1_I1[i] * reg1_J1[i] * (reg1_J1[i] - 1) * (T - 1.222) ** (reg1_J1[i] - 2))
    fn_return_value = - R * T ** 2 * G_tt
    return fn_return_value


@njit(cache=True)
def Cv1_pT(p, T):
    R = 0.461526
    # Release on the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam,
    # September 1997
    # 5 Equations for Region 1, Section. 5.1 Basic Equation
    # Eqution 7, Table 3, Page 6
    p = p / 16.53
    T = 1386 / T
    g_p = 0
    g_pp = 0
    g_pt = 0
    G_tt = 0
    reg1_I1 = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0,
         3.0, 3.0, 4.0, 4.0, 4.0, 5.0, 8.0, 8.0, 21.0, 23.0, 29.0, 30.0, 31.0, 32.0])
    reg1_J1 = np.array(
        [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, -9.0, -7.0, -1.0, 0.0, 1.0, 3.0, -3.0, 0.0, 1.0, 3.0, 17.0,
         -4.0, 0.0, 6.0, -5.0, -2.0, 10.0, -8.0, -11.0, -6.0, -29.0, -31.0, -38.0, -39.0, -40.0, -41.0])
    reg1_n1 = np.array([0.14632971213167, -0.84548187169114, -3.756360367204, 3.3855169168385, -0.95791963387872,
                        0.15772038513228, -0.016616417199501, 8.1214629983568E-04, 2.8319080123804E-04,
                        -6.0706301565874E-04,
                        -0.018990068218419, -0.032529748770505, -0.021841717175414, -5.283835796993E-05,
                        -4.7184321073267E-04, -3.0001780793026E-04, 4.7661393906987E-05, -4.4141845330846E-06,
                        -7.2694996297594E-16, -3.1679644845054E-05, -2.8270797985312E-06, -8.5205128120103E-10,
                        -2.2425281908E-06, -6.5171222895601E-07, - 1.4341729937924E-13, -4.0516996860117E-07,
                        -1.2734301741641E-09, -1.7424871230634E-10, -6.8762131295531E-19, 1.4478307828521E-20,
                        2.6335781662795E-23, -1.1947622640071E-23, 1.8228094581404E-24, -9.3537087292458E-26])

    for i in range(34):
        g_p = g_p - reg1_n1[i] * reg1_I1[i] * (7.1 - p) ** (reg1_I1[i] - 1) * (T - 1.222) ** reg1_J1[i]
        g_pp = g_pp + reg1_n1[i] * reg1_I1[i] * (reg1_I1[i] - 1) * (7.1 - p) ** (reg1_I1[i] - 2) * (T - 1.222) ** \
               reg1_J1[i]
        g_pt = g_pt - reg1_n1[i] * reg1_I1[i] * (7.1 - p) ** (reg1_I1[i] - 1) * reg1_J1[i] * (T - 1.222) ** (
            reg1_J1[i] - 1)
        G_tt = G_tt + reg1_n1[i] * (7.1 - p) ** reg1_I1[i] * reg1_J1[i] * (reg1_J1[i] - 1) * (T - 1.222) ** (
            reg1_J1[i] - 2)

    fn_return_value = R * (- (T ** 2 * G_tt) + (g_p - T * g_pt) ** 2 / g_pp)
    return fn_return_value


@njit(cache=True)
def w1_pT(p, T):
    R = 0.461526
    # Release on the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam,
    # September 1997
    # 5 Equations for Region 1, Section. 5.1 Basic Equation
    # Eqution 7, Table 3, Page 6
    p = p / 16.53
    tau = 1386 / T
    g_p = 0
    g_pp = 0
    g_pt = 0
    G_tt = 0
    reg1_I1 = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0,
         3.0, 3.0, 4.0, 4.0, 4.0, 5.0, 8.0, 8.0, 21.0, 23.0, 29.0, 30.0, 31.0, 32.0])
    reg1_J1 = np.array(
        [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, -9.0, -7.0, -1.0, 0.0, 1.0, 3.0, -3.0, 0.0, 1.0, 3.0, 17.0,
         -4.0, 0.0, 6.0, -5.0, -2.0, 10.0, -8.0, -11.0, -6.0, -29.0, -31.0, -38.0, -39.0, -40.0, -41.0])
    reg1_n1 = np.array([0.14632971213167, -0.84548187169114, -3.756360367204, 3.3855169168385, -0.95791963387872,
                        0.15772038513228, -0.016616417199501, 8.1214629983568E-04, 2.8319080123804E-04,
                        -6.0706301565874E-04,
                        -0.018990068218419, -0.032529748770505, -0.021841717175414, -5.283835796993E-05,
                        -4.7184321073267E-04, -3.0001780793026E-04, 4.7661393906987E-05, -4.4141845330846E-06,
                        -7.2694996297594E-16, -3.1679644845054E-05, -2.8270797985312E-06, -8.5205128120103E-10,
                        -2.2425281908E-06, -6.5171222895601E-07, - 1.4341729937924E-13, -4.0516996860117E-07,
                        -1.2734301741641E-09, -1.7424871230634E-10, -6.8762131295531E-19, 1.4478307828521E-20,
                        2.6335781662795E-23, -1.1947622640071E-23, 1.8228094581404E-24, -9.3537087292458E-26])
    for i in range(34):
        g_p = g_p - reg1_n1[i] * reg1_I1[i] * (7.1 - p) ** (reg1_I1[i] - 1) * (tau - 1.222) ** reg1_J1[i]
        g_pp = g_pp + reg1_n1[i] * reg1_I1[i] * (reg1_I1[i] - 1) * (7.1 - p) ** (reg1_I1[i] - 2) * (tau - 1.222) ** \
               reg1_J1[i]
        g_pt = g_pt - reg1_n1[i] * reg1_I1[i] * (7.1 - p) ** (reg1_I1[i] - 1) * reg1_J1[i] * (tau - 1.222) ** (
            reg1_J1[i] - 1)
        G_tt = G_tt + reg1_n1[i] * (7.1 - p) ** reg1_I1[i] * reg1_J1[i] * (reg1_J1[i] - 1) * (tau - 1.222) ** (
            reg1_J1[i] - 2)

    fn_return_value = (1000 * R * T * g_p ** 2 / ((g_p - tau * g_pt) ** 2 / (tau ** 2 * G_tt) - g_pp)) ** 0.5
    return fn_return_value


@njit(cache=True)
def T1_ph(p, h):
    T1_ph_I1 = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0])
    T1_ph_J1 = np.array(
        [0.0, 1.0, 2.0, 6.0, 22.0, 32.0, 0.0, 1.0, 2.0, 3.0, 4.0, 10.0, 32.0, 10.0, 32.0, 10.0, 32.0, 32.0, 32.0,
         32.0])
    T1_ph_n1 = np.array([-238.72489924521, 404.21188637945, 113.49746881718, -5.8457616048039, -1.528548241314E-04,
                         -1.0866707695377E-06, -13.391744872602, 43.211039183559, -54.010067170506, 30.535892203916,
                         -6.5964749423638, 9.3965400878363E-03, 1.157364750534E-07, -2.5858641282073E-05,
                         -4.0644363084799E-09,
                         6.6456186191635E-08, 8.0670734103027E-11, - 9.3477771213947E-13, 5.8265442020601E-15,
                         -1.5020185953503E-17])
    # Release on the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam,
    # September 1997
    # 5 Equations for Region 1, Section. 5.1 Basic Equation, 5.2.1 The Backward Equation T ( p,h )
    # Eqution 11, Table 6, Page 10
    h = h / 2500
    T = 0
    for i in range(20):
        T = T + T1_ph_n1[i] * p ** T1_ph_I1[i] * (h + 1) ** T1_ph_J1[i]
    fn_return_value = T
    return fn_return_value


@njit(cache=True)
def T1_ps(p, s):
    # Release on the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and
    # Steam, September 1997
    # 5 Equations for Region 1, Section. 5.1 Basic Equation, 5.2.2 The Backward Equation T ( p, s )
    # Eqution 13, Table 8, Page 11
    T1_ps_I1 = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 4.0])
    T1_ps_J1 = np.array(
        [0.0, 1.0, 2.0, 3.0, 11.0, 31.0, 0.0, 1.0, 2.0, 3.0, 12.0, 31.0, 0.0, 1.0, 2.0, 9.0, 31.0, 10.0, 32.0, 32.0])
    T1_ps_n1 = np.array([174.78268058307, 34.806930892873, 6.5292584978455, 0.33039981775489, -1.9281382923196E-07,
                         -2.4909197244573E-23, -0.26107636489332, 0.22592965981586, -0.064256463395226,
                         7.8876289270526E-03,
                         3.5672110607366E-10, 1.7332496994895E-24, 5.6608900654837E-04, -3.2635483139717E-04,
                         4.4778286690632E-05, -5.1322156908507E-10, -4.2522657042207E-26, 2.6400441360689E-13,
                         7.8124600459723E-29, -3.0732199903668E-31])
    fn_return_value = 0
    for i in range(20):
        fn_return_value += T1_ps_n1[i] * p ** T1_ps_I1[i] * (s + 2) ** T1_ps_J1[i]
    return fn_return_value


@njit(cache=True)
def p1_hs(h, s):
    # Supplementary Release on Backward Equations for Pressure as a Function of Enthalpy and Entropy p(h,s) to
    # the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam
    # 5 Backward Equation p(h,s) for Region 1
    # Eqution 1, Table 2, Page 5
    p1_hs_I1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0])
    p1_hs_J1 = np.array(
        [0.0, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 14.0, 0.0, 1.0, 4.0, 6.0, 0.0, 1.0, 10.0, 4.0, 1.0, 4.0, 0.0])
    p1_hs_n1 = np.array([-0.691997014660582, -18.361254878756, -9.28332409297335, 65.9639569909906, -16.2060388912024,
                         450.620017338667, 854.68067822417, 6075.23214001162, 32.6487682621856, -26.9408844582931,
                         -319.9478483343, -928.35430704332, 30.3634537455249, -65.0540422444146, -4309.9131651613,
                         -747.512324096068, 730.000345529245, 1142.84032569021, -436.407041874559])

    h = h / 3400
    s = s / 7.6
    p = 0
    for i in range(19):
        p = p + p1_hs_n1[i] * (h + 0.05) ** p1_hs_I1[i] * (s + 0.05) ** p1_hs_J1[i]
    fn_return_value = p * 100
    return fn_return_value


@njit(cache=True)
def T1_prho(p, rho):
    # Solve by iteration. Observe that fo low temperatures this equation has 2 solutions.
    # Solve with half interval method
    Low_Bound = 273.15
    High_Bound = T4_p(p)
    iter_count = 0
    Ts = 0
    rhos = 0
    while abs(rho - rhos) > 0.00001 and iter_count < 1000:
        Ts = (Low_Bound + High_Bound) / 2
        rhos = 1 / v1_pT(p, Ts)
        if rhos < rho:
            High_Bound = Ts
        else:
            Low_Bound = Ts

        iter_count += 1

    if iter_count >= 1000:
        fn_return_value = math.nan
        return fn_return_value

    fn_return_value = Ts
    return fn_return_value


# ***********************************************************************************************************
# *2.2 Functions for region 2
#
@njit(cache=True)
def v2_pT(p, T):
    R = 0.461526
    reg2_J0 = np.array([0.0, 1.0, -5.0, -4.0, -3.0, -2.0, -1.0, 2.0, 3.0])
    reg2_n0 = np.array([-9.6927686500217, 10.086655968018, -0.005608791128302, 0.071452738081455, -0.40710498223928,
                        1.4240819171444, - 4.383951131945, - 0.28408632460772, 0.021268463753307])

    reg2_Ir = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0,
                        5.0, 6.0, 6.0, 6.0, 7.0, 7.0, 7.0, 8.0, 8.0, 9.0, 10.0, 10.0, 10.0, 16.0, 16.0,
                        18.0, 20.0, 20.0, 20.0, 21.0, 22.0, 23.0, 24.0, 24.0, 24.0])
    reg2_Jr = np.array([0.0, 1.0, 2.0, 3.0, 6.0, 1.0, 2.0, 4.0, 7.0, 36.0, 0.0, 1.0, 3.0, 6.0, 35.0, 1.0, 2.0,
                        3.0, 7.0, 3.0, 16.0, 35.0, 0.0, 11.0, 25.0, 8.0, 36.0, 13.0, 4.0, 10.0, 14.0,
                        29.0, 50.0, 57.0, 20.0, 35.0, 48.0, 21.0, 53.0, 39.0, 26.0, 40.0, 58.0])

    reg2_nr = np.array(
        [- 1.7731742473213E-03, - 0.017834862292358, - 0.045996013696365, - 0.057581259083432, - 0.05032527872793,
         - 3.3032641670203E-05, - 1.8948987516315E-04, - 3.9392777243355E-03, - 0.043797295650573,
         - 2.6674547914087E-05, 2.0481737692309E-08, 4.3870667284435E-07, - 3.227767723857E-05,
         - 1.5033924542148E-03, - 0.040668253562649, - 7.8847309559367E-10, 1.2790717852285E-08,
         4.8225372718507E-07, 2.2922076337661E-06, - 1.6714766451061E-11, - 2.1171472321355E-03,
         - 23.895741934104, - 5.905956432427E-18, - 1.2621808899101E-06, - 0.038946842435739, 1.1256211360459E-11,
         - 8.2311340897998, 1.9809712802088E-08, 1.0406965210174E-19, - 1.0234747095929E-13,
         - 1.0018179379511E-09, - 8.0882908646985E-11, 0.10693031879409, - 0.33662250574171, 8.9185845355421E-25,
         3.0629316876232E-13, - 4.2002467698208E-06, - 5.9056029685639E-26, 3.7826947613457E-06,
         - 1.2768608934681E-15, 7.3087610595061E-29, 5.5414715350778E-17, - 9.436970724121E-07])

    # Release on the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam,
    # September 1997
    # 6 Equations for Region 2, Section. 6.1 Basic Equation
    # Table 11 and 12, Page 14 and 15
    tau = 540 / T
    g0_pi = 1 / p
    gr_pi = 0
    for i in range(43):
        gr_pi = gr_pi + reg2_nr[i] * reg2_Ir[i] * p ** (reg2_Ir[i] - 1) * (tau - 0.5) ** reg2_Jr[i]
    fn_return_value = R * T / p * p * (g0_pi + gr_pi) / 1000
    return fn_return_value


@njit(cache=True)
def v2_meta_pT(p, T):
    reg2_meta_J0 = np.array([0.0, 1.0, -5.0, -4.0, -3.0, -2.0, -1.0, 2.0, 3.0])
    reg2_meta_n0 = np.array([-9.6937268393049, 10.087275970006, -0.005608791128302, 0.071452738081455,
                             -0.40710498223928, 1.4240819171444, - 4.383951131945, - 0.28408632460772,
                             0.021268463753307])

    reg2_meta_Ir = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0])
    reg2_meta_Jr = np.array([0.0, 2.0, 5.0, 11.0, 1.0, 7.0, 16.0, 4.0, 16.0, 7.0, 10.0, 9.0, 10.0])
    reg2_meta_nr = np.array([-7.3362260186506E-03, -0.088223831943146, -0.072334555213245, -4.0813178534455E-03,
                             2.0097803380207E-03, -0.053045921898642, -0.007619040908697, -6.3498037657313E-03,
                             -0.086043093028588, 0.007532158152277, -7.9238375446139E-03, -2.2888160778447E-04,
                             -0.002645650148281])
    R = 0.461526
    # Release on the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam,
    # September 1997
    # 6 Equations for Region 2, Section. 6.1 Basic Equation
    # Table 11 and 12, Page 14 and 15
    # Slightly different values for first 2 terms of n0
    tau = 540 / T
    g0_pi = 1 / p
    gr_pi = 0
    for i in range(13):
        gr_pi = gr_pi + reg2_meta_nr[i] * reg2_meta_Ir[i] * p ** (reg2_meta_Ir[i] - 1) * (tau - 0.5) ** reg2_meta_Jr[i]
    fn_return_value = R * T / p * p * (g0_pi + gr_pi) / 1000
    return fn_return_value


@njit(cache=True)
def h2_pT(p, T):
    R = 0.461526
    # Release on the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam,
    # September 1997
    # 6 Equations for Region 2, Section. 6.1 Basic Equation
    # Table 11 and 12, Page 14 and 15
    reg2_J0 = np.array([0.0, 1.0, -5.0, -4.0, -3.0, -2.0, -1.0, 2.0, 3.0])
    reg2_n0 = np.array([-9.6927686500217, 10.086655968018, -0.005608791128302, 0.071452738081455, -0.40710498223928,
                        1.4240819171444, - 4.383951131945, - 0.28408632460772, 0.021268463753307])

    reg2_Ir = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0,
                        5.0, 6.0, 6.0, 6.0, 7.0, 7.0, 7.0, 8.0, 8.0, 9.0, 10.0, 10.0, 10.0, 16.0, 16.0,
                        18.0, 20.0, 20.0, 20.0, 21.0, 22.0, 23.0, 24.0, 24.0, 24.0])
    reg2_Jr = np.array([0.0, 1.0, 2.0, 3.0, 6.0, 1.0, 2.0, 4.0, 7.0, 36.0, 0.0, 1.0, 3.0, 6.0, 35.0, 1.0, 2.0,
                        3.0, 7.0, 3.0, 16.0, 35.0, 0.0, 11.0, 25.0, 8.0, 36.0, 13.0, 4.0, 10.0, 14.0,
                        29.0, 50.0, 57.0, 20.0, 35.0, 48.0, 21.0, 53.0, 39.0, 26.0, 40.0, 58.0])

    reg2_nr = np.array(
        [- 1.7731742473213E-03, - 0.017834862292358, - 0.045996013696365, - 0.057581259083432, - 0.05032527872793,
         - 3.3032641670203E-05, - 1.8948987516315E-04, - 3.9392777243355E-03, - 0.043797295650573,
         - 2.6674547914087E-05, 2.0481737692309E-08, 4.3870667284435E-07, - 3.227767723857E-05,
         - 1.5033924542148E-03, - 0.040668253562649, - 7.8847309559367E-10, 1.2790717852285E-08,
         4.8225372718507E-07, 2.2922076337661E-06, - 1.6714766451061E-11, - 2.1171472321355E-03,
         - 23.895741934104, - 5.905956432427E-18, - 1.2621808899101E-06, - 0.038946842435739, 1.1256211360459E-11,
         - 8.2311340897998, 1.9809712802088E-08, 1.0406965210174E-19, - 1.0234747095929E-13,
         - 1.0018179379511E-09, - 8.0882908646985E-11, 0.10693031879409, - 0.33662250574171, 8.9185845355421E-25,
         3.0629316876232E-13, - 4.2002467698208E-06, - 5.9056029685639E-26, 3.7826947613457E-06,
         - 1.2768608934681E-15, 7.3087610595061E-29, 5.5414715350778E-17, - 9.436970724121E-07])

    tau = 540 / T
    g0_tau = 0
    for i in range(9):
        g0_tau = g0_tau + reg2_n0[i] * reg2_J0[i] * tau ** (reg2_J0[i] - 1)
    gr_tau = 0
    for i in range(43):
        gr_tau = gr_tau + reg2_nr[i] * p ** reg2_Ir[i] * reg2_Jr[i] * (tau - 0.5) ** (reg2_Jr[i] - 1)
    fn_return_value = R * T * tau * (g0_tau + gr_tau)
    return fn_return_value


@njit(cache=True)
def h2_meta_pT(p, T):
    R = 0.461526
    reg2_meta_J0 = np.array([0.0, 1.0, -5.0, -4.0, -3.0, -2.0, -1.0, 2.0, 3.0])
    reg2_meta_n0 = np.array([-9.6937268393049, 10.087275970006, -0.005608791128302, 0.071452738081455,
                             -0.40710498223928, 1.4240819171444, - 4.383951131945, - 0.28408632460772,
                             0.021268463753307])

    reg2_meta_Ir = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0])
    reg2_meta_Jr = np.array([0.0, 2.0, 5.0, 11.0, 1.0, 7.0, 16.0, 4.0, 16.0, 7.0, 10.0, 9.0, 10.0])
    reg2_meta_nr = np.array([-7.3362260186506E-03, -0.088223831943146, -0.072334555213245, -4.0813178534455E-03,
                             2.0097803380207E-03, -0.053045921898642, -0.007619040908697, -6.3498037657313E-03,
                             -0.086043093028588, 0.007532158152277, -7.9238375446139E-03, -2.2888160778447E-04,
                             -0.002645650148281])

    # Release on the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam,
    # September 1997
    # 6 Equations for Region 2, Section. 6.1 Basic Equation
    # Table 11 and 12, Page 14 and 15
    tau = 540 / T
    g0_tau = 0
    for i in range(9):
        g0_tau = g0_tau + reg2_meta_n0[i] * reg2_meta_J0[i] * tau ** (reg2_meta_J0[i] - 1)
    gr_tau = 0
    for i in range(13):
        gr_tau = gr_tau + reg2_meta_nr[i] * p ** reg2_meta_Ir[i] * reg2_meta_Jr[i] * (tau - 0.5) ** (
            reg2_meta_Jr[i] - 1)
    fn_return_value = R * T * tau * (g0_tau + gr_tau)
    return fn_return_value


@njit(cache=True)
def u2_pT(p, T):
    R = 0.461526
    # Release on the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam,
    # September 1997
    # 6 Equations for Region 2, Section. 6.1 Basic Equation
    # Table 11 and 12, Page 14 and 15
    reg2_J0 = np.array([0.0, 1.0, -5.0, -4.0, -3.0, -2.0, -1.0, 2.0, 3.0])
    reg2_n0 = np.array([-9.6927686500217, 10.086655968018, -0.005608791128302, 0.071452738081455, -0.40710498223928,
                        1.4240819171444, - 4.383951131945, - 0.28408632460772, 0.021268463753307])

    reg2_Ir = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0,
                        5.0, 6.0, 6.0, 6.0, 7.0, 7.0, 7.0, 8.0, 8.0, 9.0, 10.0, 10.0, 10.0, 16.0, 16.0,
                        18.0, 20.0, 20.0, 20.0, 21.0, 22.0, 23.0, 24.0, 24.0, 24.0])
    reg2_Jr = np.array([0.0, 1.0, 2.0, 3.0, 6.0, 1.0, 2.0, 4.0, 7.0, 36.0, 0.0, 1.0, 3.0, 6.0, 35.0, 1.0, 2.0,
                        3.0, 7.0, 3.0, 16.0, 35.0, 0.0, 11.0, 25.0, 8.0, 36.0, 13.0, 4.0, 10.0, 14.0,
                        29.0, 50.0, 57.0, 20.0, 35.0, 48.0, 21.0, 53.0, 39.0, 26.0, 40.0, 58.0])

    reg2_nr = np.array(
        [- 1.7731742473213E-03, - 0.017834862292358, - 0.045996013696365, - 0.057581259083432, - 0.05032527872793,
         - 3.3032641670203E-05, - 1.8948987516315E-04, - 3.9392777243355E-03, - 0.043797295650573,
         - 2.6674547914087E-05, 2.0481737692309E-08, 4.3870667284435E-07, - 3.227767723857E-05,
         - 1.5033924542148E-03, - 0.040668253562649, - 7.8847309559367E-10, 1.2790717852285E-08,
         4.8225372718507E-07, 2.2922076337661E-06, - 1.6714766451061E-11, - 2.1171472321355E-03,
         - 23.895741934104, - 5.905956432427E-18, - 1.2621808899101E-06, - 0.038946842435739, 1.1256211360459E-11,
         - 8.2311340897998, 1.9809712802088E-08, 1.0406965210174E-19, - 1.0234747095929E-13,
         - 1.0018179379511E-09, - 8.0882908646985E-11, 0.10693031879409, - 0.33662250574171, 8.9185845355421E-25,
         3.0629316876232E-13, - 4.2002467698208E-06, - 5.9056029685639E-26, 3.7826947613457E-06,
         - 1.2768608934681E-15, 7.3087610595061E-29, 5.5414715350778E-17, - 9.436970724121E-07])

    tau = 540 / T
    g0_pi = 1 / p
    g0_tau = 0
    for i in range(9):
        g0_tau = g0_tau + reg2_n0[i] * reg2_J0[i] * tau ** (reg2_J0[i] - 1)
    gr_pi = 0
    gr_tau = 0
    for i in range(43):
        gr_pi = gr_pi + reg2_nr[i] * reg2_Ir[i] * p ** (reg2_Ir[i] - 1) * (tau - 0.5) ** reg2_Jr[i]
        gr_tau = gr_tau + reg2_nr[i] * p ** reg2_Ir[i] * reg2_Jr[i] * (tau - 0.5) ** (reg2_Jr[i] - 1)
    fn_return_value = R * T * (tau * (g0_tau + gr_tau) - p * (g0_pi + gr_pi))
    return fn_return_value


@njit(cache=True)
def u2_meta_pT(p, T):
    R = 0.461526
    reg2_meta_J0 = np.array([0.0, 1.0, -5.0, -4.0, -3.0, -2.0, -1.0, 2.0, 3.0])
    reg2_meta_n0 = np.array([-9.6937268393049, 10.087275970006, -0.005608791128302, 0.071452738081455,
                             -0.40710498223928, 1.4240819171444, - 4.383951131945, - 0.28408632460772,
                             0.021268463753307])

    reg2_meta_Ir = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0])
    reg2_meta_Jr = np.array([0.0, 2.0, 5.0, 11.0, 1.0, 7.0, 16.0, 4.0, 16.0, 7.0, 10.0, 9.0, 10.0])
    reg2_meta_nr = np.array([-7.3362260186506E-03, -0.088223831943146, -0.072334555213245, -4.0813178534455E-03,
                             2.0097803380207E-03, -0.053045921898642, -0.007619040908697, -6.3498037657313E-03,
                             -0.086043093028588, 0.007532158152277, -7.9238375446139E-03, -2.2888160778447E-04,
                             -0.002645650148281])

    # Release on the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam,
    # September 1997
    # 6 Equations for Region 2, Section. 6.1 Basic Equation
    # Table 11 and 12, Page 14 and 15
    tau = 540 / T
    g0_pi = 1 / p
    g0_tau = 0
    for i in range(9):
        g0_tau = g0_tau + reg2_meta_n0[i] * reg2_meta_J0[i] * tau ** (reg2_meta_J0[i] - 1)
    gr_pi = 0
    gr_tau = 0
    for i in range(13):
        gr_pi = gr_pi + reg2_meta_nr[i] * reg2_meta_Ir[i] * p ** (reg2_meta_Ir[i] - 1) * (tau - 0.5) ** reg2_meta_Jr[i]
        gr_tau = gr_tau + reg2_meta_nr[i] * p ** reg2_meta_Ir[i] * reg2_meta_Jr[i] * (tau - 0.5) ** (
            reg2_meta_Jr[i] - 1)
    fn_return_value = R * T * (tau * (g0_tau + gr_tau) - p * (g0_pi + gr_pi))
    return fn_return_value


@njit(cache=True)
def s2_pT(p, T):
    R = 0.461526
    # Release on the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam,
    # September 1997
    # 6 Equations for Region 2, Section. 6.1 Basic Equation
    # Table 11 and 12, Page 14 and 15
    reg2_J0 = np.array([0.0, 1.0, -5.0, -4.0, -3.0, -2.0, -1.0, 2.0, 3.0])
    reg2_n0 = np.array([-9.6927686500217, 10.086655968018, -0.005608791128302, 0.071452738081455, -0.40710498223928,
                        1.4240819171444, - 4.383951131945, - 0.28408632460772, 0.021268463753307])

    reg2_Ir = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0,
                        5.0, 6.0, 6.0, 6.0, 7.0, 7.0, 7.0, 8.0, 8.0, 9.0, 10.0, 10.0, 10.0, 16.0, 16.0,
                        18.0, 20.0, 20.0, 20.0, 21.0, 22.0, 23.0, 24.0, 24.0, 24.0])
    reg2_Jr = np.array([0.0, 1.0, 2.0, 3.0, 6.0, 1.0, 2.0, 4.0, 7.0, 36.0, 0.0, 1.0, 3.0, 6.0, 35.0, 1.0, 2.0,
                        3.0, 7.0, 3.0, 16.0, 35.0, 0.0, 11.0, 25.0, 8.0, 36.0, 13.0, 4.0, 10.0, 14.0,
                        29.0, 50.0, 57.0, 20.0, 35.0, 48.0, 21.0, 53.0, 39.0, 26.0, 40.0, 58.0])

    reg2_nr = np.array(
        [- 1.7731742473213E-03, - 0.017834862292358, - 0.045996013696365, - 0.057581259083432, - 0.05032527872793,
         - 3.3032641670203E-05, - 1.8948987516315E-04, - 3.9392777243355E-03, - 0.043797295650573,
         - 2.6674547914087E-05, 2.0481737692309E-08, 4.3870667284435E-07, - 3.227767723857E-05,
         - 1.5033924542148E-03, - 0.040668253562649, - 7.8847309559367E-10, 1.2790717852285E-08,
         4.8225372718507E-07, 2.2922076337661E-06, - 1.6714766451061E-11, - 2.1171472321355E-03,
         - 23.895741934104, - 5.905956432427E-18, - 1.2621808899101E-06, - 0.038946842435739, 1.1256211360459E-11,
         - 8.2311340897998, 1.9809712802088E-08, 1.0406965210174E-19, - 1.0234747095929E-13,
         - 1.0018179379511E-09, - 8.0882908646985E-11, 0.10693031879409, - 0.33662250574171, 8.9185845355421E-25,
         3.0629316876232E-13, - 4.2002467698208E-06, - 5.9056029685639E-26, 3.7826947613457E-06,
         - 1.2768608934681E-15, 7.3087610595061E-29, 5.5414715350778E-17, - 9.436970724121E-07])

    tau = 540 / T
    g0 = math.log(p)
    g0_tau = 0
    for i in range(9):
        g0 = g0 + reg2_n0[i] * tau ** reg2_J0[i]
        g0_tau = g0_tau + reg2_n0[i] * reg2_J0[i] * tau ** (reg2_J0[i] - 1)
    gr = 0
    gr_tau = 0
    for i in range(43):
        gr = gr + reg2_nr[i] * p ** reg2_Ir[i] * (tau - 0.5) ** reg2_Jr[i]
        gr_tau = gr_tau + reg2_nr[i] * p ** reg2_Ir[i] * reg2_Jr[i] * (tau - 0.5) ** (reg2_Jr[i] - 1)
    fn_return_value = R * (tau * (g0_tau + gr_tau) - (g0 + gr))
    return fn_return_value


@njit(cache=True)
def s2_meta_pT(p, T):
    R = 0.461526
    # Release on the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam,
    # September 1997
    # 6 Equations for Region 2, Section. 6.1 Basic Equation
    # Table 11 and 12, Page 14 and 15
    reg2_meta_J0 = np.array([0.0, 1.0, -5.0, -4.0, -3.0, -2.0, -1.0, 2.0, 3.0])
    reg2_meta_n0 = np.array([-9.6937268393049, 10.087275970006, -0.005608791128302, 0.071452738081455,
                             -0.40710498223928, 1.4240819171444, - 4.383951131945, - 0.28408632460772,
                             0.021268463753307])

    reg2_meta_Ir = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0])
    reg2_meta_Jr = np.array([0.0, 2.0, 5.0, 11.0, 1.0, 7.0, 16.0, 4.0, 16.0, 7.0, 10.0, 9.0, 10.0])
    reg2_meta_nr = np.array([-7.3362260186506E-03, -0.088223831943146, -0.072334555213245, -4.0813178534455E-03,
                             2.0097803380207E-03, -0.053045921898642, -0.007619040908697, -6.3498037657313E-03,
                             -0.086043093028588, 0.007532158152277, -7.9238375446139E-03, -2.2888160778447E-04,
                             -0.002645650148281])

    tau = 540 / T
    g0 = math.log(p)
    g0_tau = 0

    for i in range(9):
        g0 = g0 + reg2_meta_n0[i] * tau ** reg2_meta_J0[i]
        g0_tau = g0_tau + reg2_meta_n0[i] * reg2_meta_J0[i] * tau ** (reg2_meta_J0[i] - 1)

    gr = 0
    gr_tau = 0
    for i in range(13):
        gr = gr + reg2_meta_nr[i] * p ** reg2_meta_Ir[i] * (tau - 0.5) ** reg2_meta_Jr[i]
        gr_tau = gr_tau + reg2_meta_nr[i] * p ** reg2_meta_Ir[i] * reg2_meta_Jr[i] * (tau - 0.5) ** (
            reg2_meta_Jr[i] - 1)

    fn_return_value = R * (tau * (g0_tau + gr_tau) - (g0 + gr))
    return fn_return_value


@njit(cache=True)
def Cp2_pT(p, T):
    R = 0.461526
    # Release on the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam,
    # September 1997
    # 6 Equations for Region 2, Section. 6.1 Basic Equation
    # Table 11 and 12, Page 14 and 15
    reg2_J0 = np.array([0.0, 1.0, -5.0, -4.0, -3.0, -2.0, -1.0, 2.0, 3.0])
    reg2_n0 = np.array([-9.6927686500217, 10.086655968018, -0.005608791128302, 0.071452738081455, -0.40710498223928,
                        1.4240819171444, - 4.383951131945, - 0.28408632460772, 0.021268463753307])

    reg2_Ir = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0,
                        5.0, 6.0, 6.0, 6.0, 7.0, 7.0, 7.0, 8.0, 8.0, 9.0, 10.0, 10.0, 10.0, 16.0, 16.0,
                        18.0, 20.0, 20.0, 20.0, 21.0, 22.0, 23.0, 24.0, 24.0, 24.0])
    reg2_Jr = np.array([0.0, 1.0, 2.0, 3.0, 6.0, 1.0, 2.0, 4.0, 7.0, 36.0, 0.0, 1.0, 3.0, 6.0, 35.0, 1.0, 2.0,
                        3.0, 7.0, 3.0, 16.0, 35.0, 0.0, 11.0, 25.0, 8.0, 36.0, 13.0, 4.0, 10.0, 14.0,
                        29.0, 50.0, 57.0, 20.0, 35.0, 48.0, 21.0, 53.0, 39.0, 26.0, 40.0, 58.0])

    reg2_nr = np.array(
        [- 1.7731742473213E-03, - 0.017834862292358, - 0.045996013696365, - 0.057581259083432, - 0.05032527872793,
         - 3.3032641670203E-05, - 1.8948987516315E-04, - 3.9392777243355E-03, - 0.043797295650573,
         - 2.6674547914087E-05, 2.0481737692309E-08, 4.3870667284435E-07, - 3.227767723857E-05,
         - 1.5033924542148E-03, - 0.040668253562649, - 7.8847309559367E-10, 1.2790717852285E-08,
         4.8225372718507E-07, 2.2922076337661E-06, - 1.6714766451061E-11, - 2.1171472321355E-03,
         - 23.895741934104, - 5.905956432427E-18, - 1.2621808899101E-06, - 0.038946842435739, 1.1256211360459E-11,
         - 8.2311340897998, 1.9809712802088E-08, 1.0406965210174E-19, - 1.0234747095929E-13,
         - 1.0018179379511E-09, - 8.0882908646985E-11, 0.10693031879409, - 0.33662250574171, 8.9185845355421E-25,
         3.0629316876232E-13, - 4.2002467698208E-06, - 5.9056029685639E-26, 3.7826947613457E-06,
         - 1.2768608934681E-15, 7.3087610595061E-29, 5.5414715350778E-17, - 9.436970724121E-07])

    tau = 540 / T
    g0_tautau = 0
    for i in range(9):
        g0_tautau = g0_tautau + reg2_n0[i] * reg2_J0[i] * (reg2_J0[i] - 1) * tau ** (reg2_J0[i] - 2)

    gr_tautau = 0
    for i in range(43):
        gr_tautau = gr_tautau + reg2_nr[i] * p ** reg2_Ir[i] * reg2_Jr[i] * (reg2_Jr[i] - 1) * (tau - 0.5) ** (
            reg2_Jr[i] - 2)

    fn_return_value = - R * tau ** 2 * (g0_tautau + gr_tautau)
    return fn_return_value


@njit(cache=True)
def Cp2_meta_pT(p, T):
    R = 0.461526
    reg2_meta_J0 = np.array([0.0, 1.0, -5.0, -4.0, -3.0, -2.0, -1.0, 2.0, 3.0])
    reg2_meta_n0 = np.array([-9.6937268393049, 10.087275970006, -0.005608791128302, 0.071452738081455,
                             -0.40710498223928, 1.4240819171444, - 4.383951131945, - 0.28408632460772,
                             0.021268463753307])

    reg2_meta_Ir = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0])
    reg2_meta_Jr = np.array([0.0, 2.0, 5.0, 11.0, 1.0, 7.0, 16.0, 4.0, 16.0, 7.0, 10.0, 9.0, 10.0])
    reg2_meta_nr = np.array([-7.3362260186506E-03, -0.088223831943146, -0.072334555213245, -4.0813178534455E-03,
                             2.0097803380207E-03, -0.053045921898642, -0.007619040908697, -6.3498037657313E-03,
                             -0.086043093028588, 0.007532158152277, -7.9238375446139E-03, -2.2888160778447E-04,
                             -0.002645650148281])
    # Release on the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam,
    # September 1997
    # 6 Equations for Region 2, Section. 6.1 Basic Equation
    # Table 11 and 12, Page 14 and 15
    tau = 540 / T
    g0_tautau = 0
    for i in range(9):
        g0_tautau = g0_tautau + reg2_meta_n0[i] * reg2_meta_J0[i] * (reg2_meta_J0[i] - 1) * tau ** (reg2_meta_J0[i] - 2)
    gr_tautau = 0
    for i in range(13):
        gr_tautau = gr_tautau + reg2_meta_nr[i] * p ** reg2_meta_Ir[i] * reg2_meta_Jr[i] * (reg2_meta_Jr[i] - 1) * (
            tau - 0.5) ** (reg2_meta_Jr[i] - 2)
    fn_return_value = - R * tau ** 2 * (g0_tautau + gr_tautau)
    return fn_return_value


@njit(cache=True)
def Cv2_pT(p, T):
    R = 0.461526
    reg2_J0 = np.array([0.0, 1.0, -5.0, -4.0, -3.0, -2.0, -1.0, 2.0, 3.0])
    reg2_n0 = np.array([-9.6927686500217, 10.086655968018, -0.005608791128302, 0.071452738081455, -0.40710498223928,
                        1.4240819171444, - 4.383951131945, - 0.28408632460772, 0.021268463753307])

    reg2_Ir = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0,
                        5.0, 6.0, 6.0, 6.0, 7.0, 7.0, 7.0, 8.0, 8.0, 9.0, 10.0, 10.0, 10.0, 16.0, 16.0,
                        18.0, 20.0, 20.0, 20.0, 21.0, 22.0, 23.0, 24.0, 24.0, 24.0])
    reg2_Jr = np.array([0.0, 1.0, 2.0, 3.0, 6.0, 1.0, 2.0, 4.0, 7.0, 36.0, 0.0, 1.0, 3.0, 6.0, 35.0, 1.0, 2.0,
                        3.0, 7.0, 3.0, 16.0, 35.0, 0.0, 11.0, 25.0, 8.0, 36.0, 13.0, 4.0, 10.0, 14.0,
                        29.0, 50.0, 57.0, 20.0, 35.0, 48.0, 21.0, 53.0, 39.0, 26.0, 40.0, 58.0])

    reg2_nr = np.array(
        [- 1.7731742473213E-03, - 0.017834862292358, - 0.045996013696365, - 0.057581259083432, - 0.05032527872793,
         - 3.3032641670203E-05, - 1.8948987516315E-04, - 3.9392777243355E-03, - 0.043797295650573,
         - 2.6674547914087E-05, 2.0481737692309E-08, 4.3870667284435E-07, - 3.227767723857E-05,
         - 1.5033924542148E-03, - 0.040668253562649, - 7.8847309559367E-10, 1.2790717852285E-08,
         4.8225372718507E-07, 2.2922076337661E-06, - 1.6714766451061E-11, - 2.1171472321355E-03,
         - 23.895741934104, - 5.905956432427E-18, - 1.2621808899101E-06, - 0.038946842435739, 1.1256211360459E-11,
         - 8.2311340897998, 1.9809712802088E-08, 1.0406965210174E-19, - 1.0234747095929E-13,
         - 1.0018179379511E-09, - 8.0882908646985E-11, 0.10693031879409, - 0.33662250574171, 8.9185845355421E-25,
         3.0629316876232E-13, - 4.2002467698208E-06, - 5.9056029685639E-26, 3.7826947613457E-06,
         - 1.2768608934681E-15, 7.3087610595061E-29, 5.5414715350778E-17, - 9.436970724121E-07])

    tau = 540 / T
    g0_tautau = 0
    for i in range(9):
        g0_tautau = g0_tautau + reg2_n0[i] * reg2_J0[i] * (reg2_J0[i] - 1) * tau ** (reg2_J0[i] - 2)
    gr_pi = 0
    gr_pitau = 0
    gr_pipi = 0
    gr_tautau = 0
    for i in range(43):
        gr_pi = gr_pi + reg2_nr[i] * reg2_Ir[i] * p ** (reg2_Ir[i] - 1) * (tau - 0.5) ** reg2_Jr[i]
        gr_pipi = gr_pipi + reg2_nr[i] * reg2_Ir[i] * (reg2_Ir[i] - 1) * p ** (reg2_Ir[i] - 2) * (tau - 0.5) ** reg2_Jr[
            i]
        gr_pitau = gr_pitau + reg2_nr[i] * reg2_Ir[i] * p ** (reg2_Ir[i] - 1) * reg2_Jr[i] * (tau - 0.5) ** (
            reg2_Jr[i] - 1)
        gr_tautau = gr_tautau + reg2_nr[i] * p ** reg2_Ir[i] * reg2_Jr[i] * (reg2_Jr[i] - 1) * (tau - 0.5) ** (
            reg2_Jr[i] - 2)

    fn_return_value = R * (- (tau ** 2 * (g0_tautau + gr_tautau)) - ((1 + p * gr_pi - tau * p * gr_pitau) ** 2) / (
        1 - p ** 2 * gr_pipi))
    return fn_return_value


@njit(cache=True)
def Cv2_meta_pT(p, T):
    R = 0.461526
    reg2_meta_J0 = np.array([0.0, 1.0, -5.0, -4.0, -3.0, -2.0, -1.0, 2.0, 3.0])
    reg2_meta_n0 = np.array([-9.6937268393049, 10.087275970006, -0.005608791128302, 0.071452738081455,
                             -0.40710498223928, 1.4240819171444, - 4.383951131945, - 0.28408632460772,
                             0.021268463753307])

    reg2_meta_Ir = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0])
    reg2_meta_Jr = np.array([0.0, 2.0, 5.0, 11.0, 1.0, 7.0, 16.0, 4.0, 16.0, 7.0, 10.0, 9.0, 10.0])
    reg2_meta_nr = np.array([-7.3362260186506E-03, -0.088223831943146, -0.072334555213245, -4.0813178534455E-03,
                             2.0097803380207E-03, -0.053045921898642, -0.007619040908697, -6.3498037657313E-03,
                             -0.086043093028588, 0.007532158152277, -7.9238375446139E-03, -2.2888160778447E-04,
                             -0.002645650148281])

    tau = 540 / T
    g0_tautau = 0
    for i in range(9):
        g0_tautau = g0_tautau + reg2_meta_n0[i] * reg2_meta_J0[i] * (reg2_meta_J0[i] - 1) * tau ** (reg2_meta_J0[i] - 2)
    gr_pi = 0
    gr_pitau = 0
    gr_pipi = 0
    gr_tautau = 0
    for i in range(13):
        gr_pi = gr_pi + reg2_meta_nr[i] * reg2_meta_Ir[i] * p ** (reg2_meta_Ir[i] - 1) * (tau - 0.5) ** reg2_meta_Jr[i]
        gr_pipi = gr_pipi + reg2_meta_nr[i] * reg2_meta_Ir[i] * (reg2_meta_Ir[i] - 1) * p ** (reg2_meta_Ir[i] - 2) * (
            tau - 0.5) ** reg2_meta_Jr[i]
        gr_pitau = gr_pitau + reg2_meta_nr[i] * reg2_meta_Ir[i] * p ** (reg2_meta_Ir[i] - 1) * reg2_meta_Jr[i] * (
            tau - 0.5) ** (reg2_meta_Jr[i] - 1)
        gr_tautau = gr_tautau + reg2_meta_nr[i] * p ** reg2_meta_Ir[i] * reg2_meta_Jr[i] * (reg2_meta_Jr[i] - 1) * (
            tau - 0.5) ** (reg2_meta_Jr[i] - 2)

    fn_return_value = R * (- (tau ** 2 * (g0_tautau + gr_tautau)) - ((1 + p * gr_pi - tau * p * gr_pitau) ** 2) / (
        1 - p ** 2 * gr_pipi))
    return fn_return_value


@njit(cache=True)
def w2_pT(p, T):
    R = 0.461526
    # Release on the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam,
    # September 1997
    # 6 Equations for Region 2, Section. 6.1 Basic Equation
    # Table 11 and 12, Page 14 and 15
    reg2_J0 = np.array([0.0, 1.0, -5.0, -4.0, -3.0, -2.0, -1.0, 2.0, 3.0])
    reg2_n0 = np.array([-9.6927686500217, 10.086655968018, -0.005608791128302, 0.071452738081455, -0.40710498223928,
                        1.4240819171444, - 4.383951131945, - 0.28408632460772, 0.021268463753307])

    reg2_Ir = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0,
                        5.0, 6.0, 6.0, 6.0, 7.0, 7.0, 7.0, 8.0, 8.0, 9.0, 10.0, 10.0, 10.0, 16.0, 16.0,
                        18.0, 20.0, 20.0, 20.0, 21.0, 22.0, 23.0, 24.0, 24.0, 24.0])
    reg2_Jr = np.array([0.0, 1.0, 2.0, 3.0, 6.0, 1.0, 2.0, 4.0, 7.0, 36.0, 0.0, 1.0, 3.0, 6.0, 35.0, 1.0, 2.0,
                        3.0, 7.0, 3.0, 16.0, 35.0, 0.0, 11.0, 25.0, 8.0, 36.0, 13.0, 4.0, 10.0, 14.0,
                        29.0, 50.0, 57.0, 20.0, 35.0, 48.0, 21.0, 53.0, 39.0, 26.0, 40.0, 58.0])

    reg2_nr = np.array(
        [- 1.7731742473213E-03, - 0.017834862292358, - 0.045996013696365, - 0.057581259083432, - 0.05032527872793,
         - 3.3032641670203E-05, - 1.8948987516315E-04, - 3.9392777243355E-03, - 0.043797295650573,
         - 2.6674547914087E-05, 2.0481737692309E-08, 4.3870667284435E-07, - 3.227767723857E-05,
         - 1.5033924542148E-03, - 0.040668253562649, - 7.8847309559367E-10, 1.2790717852285E-08,
         4.8225372718507E-07, 2.2922076337661E-06, - 1.6714766451061E-11, - 2.1171472321355E-03,
         - 23.895741934104, - 5.905956432427E-18, - 1.2621808899101E-06, - 0.038946842435739, 1.1256211360459E-11,
         - 8.2311340897998, 1.9809712802088E-08, 1.0406965210174E-19, - 1.0234747095929E-13,
         - 1.0018179379511E-09, - 8.0882908646985E-11, 0.10693031879409, - 0.33662250574171, 8.9185845355421E-25,
         3.0629316876232E-13, - 4.2002467698208E-06, - 5.9056029685639E-26, 3.7826947613457E-06,
         - 1.2768608934681E-15, 7.3087610595061E-29, 5.5414715350778E-17, - 9.436970724121E-07])

    tau = 540 / T
    g0_tautau = 0
    for i in range(9):
        g0_tautau = g0_tautau + reg2_n0[i] * reg2_J0[i] * (reg2_J0[i] - 1) * tau ** (reg2_J0[i] - 2)
    gr_pi = 0
    gr_pitau = 0
    gr_pipi = 0
    gr_tautau = 0

    for i in range(43):
        gr_pi = gr_pi + reg2_nr[i] * reg2_Ir[i] * p ** (reg2_Ir[i] - 1) * (tau - 0.5) ** reg2_Jr[i]
        gr_pipi = gr_pipi + reg2_nr[i] * reg2_Ir[i] * (reg2_Ir[i] - 1) * p ** (reg2_Ir[i] - 2) * (tau - 0.5) ** reg2_Jr[
            i]
        gr_pitau = gr_pitau + reg2_nr[i] * reg2_Ir[i] * p ** (reg2_Ir[i] - 1) * reg2_Jr[i] * (tau - 0.5) ** (
            reg2_Jr[i] - 1)
        gr_tautau = gr_tautau + reg2_nr[i] * p ** reg2_Ir[i] * reg2_Jr[i] * (reg2_Jr[i] - 1) * (tau - 0.5) ** (
            reg2_Jr[i] - 2)

    fn_return_value = (1000 * R * T * (1 + 2 * p * gr_pi + p ** 2 * gr_pi ** 2) / (
        (1 - p ** 2 * gr_pipi) + (1 + p * gr_pi - tau * p * gr_pitau) ** 2 / (
        tau ** 2 * (g0_tautau + gr_tautau)))) ** 0.5
    return fn_return_value


@njit(cache=True)
def w2_meta_pT(p, T):
    R = 0.461526
    reg2_meta_J0 = np.array([0.0, 1.0, -5.0, -4.0, -3.0, -2.0, -1.0, 2.0, 3.0])
    reg2_meta_n0 = np.array([-9.6937268393049, 10.087275970006, -0.005608791128302, 0.071452738081455,
                             -0.40710498223928, 1.4240819171444, - 4.383951131945, - 0.28408632460772,
                             0.021268463753307])

    reg2_meta_Ir = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0])
    reg2_meta_Jr = np.array([0.0, 2.0, 5.0, 11.0, 1.0, 7.0, 16.0, 4.0, 16.0, 7.0, 10.0, 9.0, 10.0])
    reg2_meta_nr = np.array([-7.3362260186506E-03, -0.088223831943146, -0.072334555213245, -4.0813178534455E-03,
                             2.0097803380207E-03, -0.053045921898642, -0.007619040908697, -6.3498037657313E-03,
                             -0.086043093028588, 0.007532158152277, -7.9238375446139E-03, -2.2888160778447E-04,
                             -0.002645650148281])

    # Release on the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam,
    # September 1997
    # 6 Equations for Region 2, Section. 6.1 Basic Equation
    # Table 11 and 12, Page 14 and 15
    tau = 540 / T
    g0_tautau = 0
    for i in range(9):
        g0_tautau = g0_tautau + reg2_meta_n0[i] * reg2_meta_J0[i] * (reg2_meta_J0[i] - 1) * tau ** (reg2_meta_J0[i] - 2)

    gr_pi = 0
    gr_pitau = 0
    gr_pipi = 0
    gr_tautau = 0
    for i in range(13):
        gr_pi = gr_pi + reg2_meta_nr[i] * reg2_meta_Ir[i] * p ** (reg2_meta_Ir[i] - 1) * (tau - 0.5) ** reg2_meta_Jr[i]
        gr_pipi = gr_pipi + reg2_meta_nr[i] * reg2_meta_Ir[i] * (reg2_meta_Ir[i] - 1) * p ** (reg2_meta_Ir[i] - 2) * (
            tau - 0.5) ** reg2_meta_Jr[i]
        gr_pitau = gr_pitau + reg2_meta_nr[i] * reg2_meta_Ir[i] * p ** (reg2_meta_Ir[i] - 1) * reg2_meta_Jr[i] * (
            tau - 0.5) ** (reg2_meta_Jr[i] - 1)
        gr_tautau = gr_tautau + reg2_meta_nr[i] * p ** reg2_meta_Ir[i] * reg2_meta_Jr[i] * (reg2_meta_Jr[i] - 1) * (
            tau - 0.5) ** (reg2_meta_Jr[i] - 2)

    fn_return_value = (1000 * R * T * (1 + 2 * p * gr_pi + p ** 2 * gr_pi ** 2) / (
        (1 - p ** 2 * gr_pipi) + (1 + p * gr_pi - tau * p * gr_pitau) ** 2 / (
        tau ** 2 * (g0_tautau + gr_tautau)))) ** 0.5
    return fn_return_value


@njit(cache=True)
def T2_ph_part1(p, h):
    T2_ph_part1_Ji = np.array([0.0, 1.0, 2.0, 3.0, 7.0, 20.0, 0.0, 1.0, 2.0, 3.0, 7.0, 9.0, 11.0, 18.0,
                               44.0, 0.0, 2.0, 7.0, 36.0, 38.0, 40.0, 42.0, 44.0, 24.0, 44.0, 12.0,
                               32.0, 44.0, 32.0, 36.0, 42.0, 34.0, 44.0, 28.0])
    T2_ph_part1_Ii = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                               2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 4.0, 5.0, 5.0,
                               5.0, 6.0, 6.0, 7.0])
    T2_ph_part1_ni = np.array([1089.8952318288, 849.51654495535, - 107.81748091826, 33.153654801263, - 7.4232016790248,
                               11.765048724356, 1.844574935579, - 4.1792700549624, 6.2478196935812, - 17.344563108114,
                               - 200.58176862096, 271.96065473796, - 455.11318285818, 3091.9688604755, 252266.40357872,
                               - 6.1707422868339E-03, - 0.31078046629583, 11.670873077107, 128127984.04046,
                               - 985549096.23276,
                               2822454697.3002, - 3594897141.0703, 1722734991.3197, - 13551.334240775, 12848734.66465,
                               1.3865724283226, 235988.32556514, - 13105236.545054, 7399.9835474766, - 551966.9703006,
                               3715408.5996233, 19127.72923966, - 415351.64835634, - 62.459855192507])
    Ts = 0
    hs = h / 2000
    for i in range(34):
        Ts = Ts + T2_ph_part1_ni[i] * p ** (T2_ph_part1_Ii[i]) * (hs - 2.1) ** T2_ph_part1_Ji[i]

    fn_return_value = Ts
    return fn_return_value


@njit(cache=True)
def T2_ph_part2(p, h):
    # Subregion B
    # Table 21, Eq 23, page 23
    T2_ph_part2_Ji = np.array([0.0, 1.0, 2.0, 12.0, 18.0, 24.0, 28.0, 40.0, 0.0, 2.0, 6.0, 12.0, 18.0,
                               24.0, 28.0, 40.0, 2.0, 8.0, 18.0, 40.0, 1.0, 2.0, 12.0, 24.0, 2.0, 12.0,
                               18.0, 24.0, 28.0, 40.0, 18.0, 24.0, 40.0, 28.0, 2.0, 28.0, 1.0, 40.0])
    T2_ph_part2_Ii = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                               1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                               5.0, 5.0, 5.0, 6.0, 7.0, 7.0, 9.0, 9.0])
    T2_ph_part2_ni = np.array([1489.5041079516, 743.07798314034, - 97.708318797837, 2.4742464705674, - 0.63281320016026,
                               1.1385952129658, - 0.47811863648625, 8.5208123431544E-03, 0.93747147377932,
                               3.3593118604916,
                               3.3809355601454, 0.16844539671904, 0.73875745236695, - 0.47128737436186,
                               0.15020273139707,
                               - 0.002176411421975, - 0.021810755324761, - 0.10829784403677, - 0.046333324635812,
                               7.1280351959551E-05, 1.1032831789999E-04, 1.8955248387902E-04, 3.0891541160537E-03,
                               1.3555504554949E-03, 2.8640237477456E-07, - 1.0779857357512E-05, - 7.6462712454814E-05,
                               1.4052392818316E-05, - 3.1083814331434E-05, - 1.0302738212103E-06, 2.821728163504E-07,
                               1.2704902271945E-06, 7.3803353468292E-08, - 1.1030139238909E-08, - 8.1456365207833E-14,
                               - 2.5180545682962E-11, - 1.7565233969407E-18, 8.6934156344163E-15])
    Ts = 0
    hs = h / 2000
    for i in range(38):
        Ts = Ts + T2_ph_part2_ni[i] * (p - 2) ** (T2_ph_part2_Ii[i]) * (hs - 2.6) ** T2_ph_part2_Ji[i]
    fn_return_value = Ts
    return fn_return_value


@njit(cache=True)
def T2_ph_part3(p, h):
    # Subregion C
    # Table 22, Eq 24, page 24
    T2_ph_part3_Ji = np.array([0.0, 4.0, 0.0, 2.0, 0.0, 2.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0,
                               4.0, 8.0, 4.0, 0.0, 1.0, 4.0, 10.0, 12.0, 16.0, 20.0, 22.0])
    T2_ph_part3_Ii = np.array([- 7.0, - 7.0, - 6.0, - 6.0, - 5.0, - 5.0, - 2.0, - 2.0, - 1.0,
                               - 1.0, 0.0, 0.0, 1.0, 1.0, 2.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0])
    T2_ph_part3_ni = np.array([- 3236839855524.2, 7326335090218.1, 358250899454.47, - 583401318515.9, - 10783068217.47,
                               20825544563.171, 610747.83564516, 859777.2253558, - 25745.72360417, 31081.088422714,
                               1208.2315865936,
                               482.19755109255, 3.7966001272486, - 10.842984880077, - 0.04536417267666,
                               1.4559115658698E-13,
                               1.126159740723E-12, - 1.7804982240686E-11, 1.2324579690832E-07, - 1.1606921130984E-06,
                               2.7846367088554E-05, - 5.9270038474176E-04, 1.2918582991878E-03])
    Ts = 0
    hs = h / 2000
    for i in range(23):
        Ts = Ts + T2_ph_part3_ni[i] * (p + 25) ** (T2_ph_part3_Ii[i]) * (hs - 1.8) ** T2_ph_part3_Ji[i]
    fn_return_value = Ts
    return fn_return_value


@njit(cache=True)
def T2_ph(p, h):
    # Release on the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam,
    # September 1997
    # 6 Equations for Region 2,6.3.1 The Backward Equations T( p, h ) for Subregions 2a, 2b, and 2c
    if p < 4:
        sub_reg = 1
    else:
        if p < (905.84278514723 - 0.67955786399241 * h + 1.2809002730136E-04 * h ** 2):
            sub_reg = 2
        else:
            sub_reg = 3
    select_variable_0 = sub_reg
    if select_variable_0 == 1:
        # Subregion A
        # Table 20, Eq 22, page 22
        return T2_ph_part1(p, h)
    elif select_variable_0 == 2:
        return T2_ph_part2(p, h)
    else:
        return T2_ph_part3(p, h)

    # return fn_return_value


@njit(cache=True)
def T2_ps_part1(p, s):
    # Subregion A
    # Table 25, Eq 25, page 26
    T2_ps_part1_Ii = np.array(
        [- 1.5, - 1.5, - 1.5, - 1.5, - 1.5, - 1.5, - 1.25, - 1.25, - 1.25, - 1, - 1, - 1, - 1, - 1, - 1,
         - 0.75, - 0.75, - 0.5, - 0.5, - 0.5, - 0.5, - 0.25, - 0.25, - 0.25, - 0.25, 0.25, 0.25, 0.25, 0.25,
         0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 0.75, 1, 1, 1.25, 1.25, 1.5, 1.5])
    T2_ps_part1_Ji = np.array([- 24.0, - 23.0, - 19.0, - 13.0, - 11.0, - 10.0, - 19.0, - 15.0, - 6.0, - 26.0, - 21.0,
                               - 17.0, - 16.0, - 9.0, - 8.0, - 15.0, - 14.0,
                               - 26.0, - 13.0, - 9.0, - 7.0, - 27.0, - 25.0, - 11.0, - 6.0,
                               1.0, 4.0, 8.0, 11.0, 0.0, 1.0,
                               5.0, 6.0, 10.0, 14.0, 16.0, 0.0, 4.0, 9.0, 17.0, 7.0, 18.0, 3.0, 15.0, 5.0, 18.0])

    T2_ps_part1_ni = np.array([- 392359.83861984, 515265.7382727, 40482.443161048, - 321.93790923902, 96.961424218694,
                               - 22.867846371773, - 449429.14124357, - 5011.8336020166, 0.35684463560015,
                               44235.33584819,
                               - 13673.388811708, 421632.60207864, 22516.925837475, 474.42144865646, - 149.31130797647,
                               - 197811.26320452, - 23554.39947076, - 19070.616302076, 55375.669883164, 3829.3691437363,
                               - 603.91860580567, 1936.3102620331, 4266.064369861, - 5978.0638872718, - 704.01463926862,
                               338.36784107553, 20.862786635187, 0.033834172656196, - 4.3124428414893E-05,
                               166.53791356412,
                               - 139.86292055898, - 0.78849547999872, 0.072132411753872, - 5.9754839398283E-03,
                               - 1.2141358953904E-05, 2.3227096733871E-07, - 10.538463566194, 2.0718925496502,
                               - 0.072193155260427,
                               2.074988708112E-07, - 0.018340657911379, 2.9036272348696E-07, 0.21037527893619,
                               2.5681239729999E-04,
                               - 0.012799002933781, - 8.2198102652018E-06])
    sigma = s / 2
    teta = 0
    for i in range(46):
        teta = teta + T2_ps_part1_ni[i] * p ** T2_ps_part1_Ii[i] * (sigma - 2) ** T2_ps_part1_Ji[i]
    fn_return_value = teta
    return fn_return_value


@njit(cache=True)
def T2_ps_part2(p, s):
    # Subregion B
    # Table 26, Eq 26, page 27
    T2_ps_part2_Ii = np.array(
        [- 6.0, - 6.0, - 5.0, - 5.0, - 4.0, - 4.0, - 4.0, - 3.0, - 3.0, - 3.0, - 3.0, - 2.0, - 2.0, - 2.0,
         - 2.0, - 1.0, - 1.0, - 1.0, - 1.0, - 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 5.0])
    T2_ps_part2_Ji = np.array(
        [0.0, 11.0, 0.0, 11.0, 0.0, 1.0, 11.0, 0.0, 1.0, 11.0, 12.0, 0.0, 1.0, 6.0, 10.0, 0.0, 1.0,
         5.0, 8.0, 9.0, 0.0, 1.0, 2.0, 4.0, 5.0, 6.0, 9.0, 0.0, 1.0, 2.0, 3.0, 7.0, 8.0, 0.0, 1.0,
         5.0,
         0.0, 1.0, 3.0, 0.0, 1.0, 0.0, 1.0, 2.0])
    T2_ps_part2_ni = np.array([316876.65083497, 20.864175881858, - 398593.99803599, - 21.816058518877, 223697.85194242,
                               - 2784.1703445817, 9.920743607148, - 75197.512299157, 2970.8605951158, - 3.4406878548526,
                               0.38815564249115, 17511.29508575, - 1423.7112854449, 1.0943803364167, 0.89971619308495,
                               - 3375.9740098958, 471.62885818355, - 1.9188241993679, 0.41078580492196,
                               - 0.33465378172097,
                               1387.0034777505, - 406.63326195838, 41.72734715961, 2.1932549434532, - 1.0320050009077,
                               0.35882943516703, 5.2511453726066E-03, 12.838916450705, - 2.8642437219381,
                               0.56912683664855,
                               - 0.099962954584931, - 3.2632037778459E-03, 2.3320922576723E-04, - 0.1533480985745,
                               0.029072288239902, 3.7534702741167E-04, 1.7296691702411E-03, - 3.8556050844504E-04,
                               - 3.5017712292608E-05, - 1.4566393631492E-05, 5.6420857267269E-06, 4.1286150074605E-08,
                               - 2.0684671118824E-08, 1.6409393674725E-09])

    sigma = s / 0.7853
    teta = 0
    for i in range(44):
        teta = teta + T2_ps_part2_ni[i] * p ** T2_ps_part2_Ii[i] * (10 - sigma) ** T2_ps_part2_Ji[i]
    fn_return_value = teta
    return fn_return_value


@njit(cache=True)
def T2_ps_part3(p, s):
    # Subregion C
    # Table 27, Eq 27, page 28
    T2_ps_part3_Ii = np.array([- 2.0, - 2.0, - 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
                               2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0,
                               6.0, 6.0, 7.0, 7.0, 7.0, 7.0, 7.0])
    T2_ps_part3_Ji = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 3.0, 4.0, 0.0,
                               1.0, 2.0, 0.0, 1.0, 5.0, 0.0, 1.0, 4.0, 0.0, 1.0, 2.0, 0.0,
                               1.0, 0.0, 1.0, 3.0, 4.0, 5.0])
    T2_ps_part3_ni = np.array([909.68501005365, 2404.566708842, - 591.6232638713, 541.45404128074, - 270.98308411192,
                               979.76525097926, - 469.66772959435, 14.399274604723, - 19.104204230429, 5.3299167111971,
                               - 21.252975375934, - 0.3114733441376, 0.60334840894623, - 0.042764839702509,
                               5.8185597255259E-03,
                               - 0.014597008284753, 5.6631175631027E-03, - 7.6155864584577E-05, 2.2440342919332E-04,
                               - 1.2561095013413E-05, 6.3323132660934E-07, - 2.0541989675375E-06, 3.6405370390082E-08,
                               - 2.9759897789215E-09, 1.0136618529763E-08, 5.9925719692351E-12, - 2.0677870105164E-11,
                               - 2.0874278181886E-11, 1.0162166825089E-10, - 1.6429828281347E-10])
    sigma = s / 2.9251
    teta = 0
    for i in range(30):
        teta = teta + T2_ps_part3_ni[i] * p ** T2_ps_part3_Ii[i] * (2 - sigma) ** T2_ps_part3_Ji[i]
    fn_return_value = teta
    return fn_return_value


@njit(cache=True)
def T2_ps(p, s):
    # Release on the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam,
    # September 1997
    # 6 Equations for Region 2,6.3.2 The Backward Equations T( p, s ) for Subregions 2a, 2b, and 2c
    # Page 26
    if p < 4:
        sub_reg = 1
    else:
        if s < 5.85:
            sub_reg = 3
        else:
            sub_reg = 2
    select_variable_1 = sub_reg
    if select_variable_1 == 1:
        return T2_ps_part1(p, s)
    elif select_variable_1 == 2:
        return T2_ps_part2(p, s)
    else:
        return T2_ps_part3(p, s)

    # return fn_return_value


@njit(cache=True)
def p2_hs_part1(h, s):
    p2_hs_part1_Ii = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0,
         3.0, 3.0, 3.0, 3.0, 4.0, 5.0, 5.0, 6.0, 7.0])
    p2_hs_part1_Ji = np.array([1.0, 3.0, 6.0, 16.0, 20.0, 22.0, 0.0, 1.0, 2.0, 3.0, 5.0, 6.0,
                               10.0, 16.0, 20.0, 22.0, 3.0, 16.0, 20.0, 0.0, 2.0, 3.0, 6.0, 16.0, 16.0, 3.0, 16.0, 3.0,
                               1.0])
    p2_hs_part1_ni = np.array(
        [- 1.82575361923032E-02, - 0.125229548799536, 0.592290437320145, 6.04769706185122, 238.624965444474,
         - 298.639090222922, 0.051225081304075, - 0.437266515606486, 0.413336902999504, - 5.16468254574773,
         - 5.57014838445711, 12.8555037824478, 11.414410895329, - 119.504225652714, - 2847.7798596156,
         4317.57846408006, 1.1289404080265, 1974.09186206319, 1516.12444706087, 1.41324451421235E-02,
         0.585501282219601, - 2.97258075863012, 5.94567314847319, - 6236.56565798905, 9659.86235133332,
         6.81500934948134, - 6332.07286824489, - 5.5891922446576, 4.00645798472063E-02])

    eta = h / 4200
    sigma = s / 12
    p = 0
    for i in range(29):
        p = p + p2_hs_part1_ni[i] * (eta - 0.5) ** p2_hs_part1_Ii[i] * (sigma - 1.2) ** p2_hs_part1_Ji[i]
    fn_return_value = p ** 4 * 4
    return fn_return_value


@njit(cache=True)
def p2_hs_part2(h, s):
    p2_hs_part2_Ii = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0,
                               3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 8.0, 8.0,
                               12.0, 14.0])
    p2_hs_part2_Ji = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 0.0, 1.0, 2.0, 3.0, 5.0, 12.0, 1.0, 6.0, 18.0, 0.0, 1.0,
                               7.0, 12.0, 1.0, 16.0, 1.0, 12.0, 1.0, 8.0, 18.0, 1.0, 16.0, 1.0, 3.0,
                               14.0, 18.0, 10.0, 16.0])
    p2_hs_part2_ni = np.array(
        [8.01496989929495E-02, - 0.543862807146111, 0.337455597421283, 8.9055545115745, 313.840736431485,
         0.797367065977789, - 1.2161697355624, 8.72803386937477, - 16.9769781757602, - 186.552827328416,
         95115.9274344237, - 18.9168510120494, - 4334.0703719484, 543212633.012715, 0.144793408386013,
         128.024559637516, - 67230.9534071268, 33697238.0095287, - 586.63419676272, - 22140322476.9889,
         1716.06668708389, - 570817595.806302, - 3121.09693178482, - 2078413.8463301, 3056059461577.86,
         3221.57004314333, 326810259797.295, - 1441.04158934487, 410.694867802691, 109077066873.024,
         -24796465425889.3, 1888019068.65134, - 123651009018773])

    eta = h / 4100
    sigma = s / 7.9
    p = 0
    for i in range(33):
        p = p + p2_hs_part2_ni[i] * (eta - 0.6) ** p2_hs_part2_Ii[i] * (sigma - 1.01) ** p2_hs_part2_Ji[i]
    fn_return_value = p ** 4 * 100
    return fn_return_value


@njit(cache=True)
def p2_hs_part3(h, s):
    p2_hs_part3_Ii = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0,
                               2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 10.0, 12.0, 16.0])

    p2_hs_part3_Ji = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 8.0, 0.0, 2.0, 5.0, 8.0, 14.0, 2.0, 3.0, 7.0, 10.0, 18.0,
                               0.0, 5.0, 8.0, 16.0, 18.0, 18.0, 1.0, 4.0, 6.0, 14.0, 8.0, 18.0, 7.0, 7.0, 10.0])

    p2_hs_part3_ni = np.array(
        [0.112225607199012, - 3.39005953606712, - 32.0503911730094, - 197.5973051049, - 407.693861553446,
         13294.3775222331, 1.70846839774007, 37.3694198142245, 3581.44365815434, 423014.446424664,
         - 751071025.760063, 52.3446127607898, - 228.351290812417, - 960652.417056937, - 80705929.2526074,
         1626980172256.69, 0.772465073604171, 46392.9973837746, - 13731788.5134128, 1704703926305.12,
         - 25110462818730.8, 31774883083552, 53.8685623675312, - 55308.9094625169, - 1028615.22421405,
         2042494187562.34, 273918446.626977, - 2.63963146312685E+15, - 1078908541.08088, - 29649262098.0124,
         - 1.11754907323424E+15])

    eta = h / 3500
    sigma = s / 5.9
    p = 0
    for i in range(31):
        p = p + p2_hs_part3_ni[i] * (eta - 0.7) ** p2_hs_part3_Ii[i] * (sigma - 1.1) ** p2_hs_part3_Ji[i]

    fn_return_value = p ** 4 * 100
    return fn_return_value


@njit(cache=True)
def p2_hs(h, s):
    # Supplementary Release on Backward Equations for Pressure as a Function of Enthalpy and Entropy p(h,s)
    # to the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam
    # Chapter 6:Backward Equations p(h,s) for Region 2
    if h < (- 3498.98083432139 + 2575.60716905876 * s - 421.073558227969 * s ** 2 + 27.6349063799944 * s ** 3):
        sub_reg = 1
    else:
        if s < 5.85:
            sub_reg = 3
        else:
            sub_reg = 2
    select_variable_2 = sub_reg
    if select_variable_2 == 1:
        # Subregion A
        # Table 6, Eq 3, page 8
        return p2_hs_part1(h, s)
    elif select_variable_2 == 2:
        # Subregion B
        # Table 7, Eq 4, page 9
        return p2_hs_part2(h, s)
    else:
        # Subregion C
        # Table 8, Eq 5, page 10
        return p2_hs_part3(h, s)

    # return fn_return_value


@njit(cache=True)
def T2_prho(p, rho):
    Ts = 0
    # Solve by iteration. Observe that fo low temperatures this equation has 2 solutions.
    # Solve with half interval method
    if p < 16.5292:
        Low_Bound = T4_p(p)
    else:
        Low_Bound = B23T_p(p)
    High_Bound = 1073.15
    iter_count = 0
    rhos = 0
    while abs(rho - rhos) > 0.000001 and iter_count < 1000:
        Ts = (Low_Bound + High_Bound) / 2
        rhos = 1 / v2_pT(p, Ts)
        if rhos < rho:
            High_Bound = Ts
        else:
            Low_Bound = Ts

        iter_count += 1

    if iter_count >= 1000:
        return math.nan

    fn_return_value = Ts
    return fn_return_value


# ***********************************************************************************************************
# *2.3 Functions for region 3
# ***********************************************************************************************************
# *2.4 Functions for region 4
# ***********************************************************************************************************
# *2.5 Functions for region 5
# ***********************************************************************************************************
@njit(cache=True)
def p3_rhoT(rho, T):
    reg3_Ii = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0,
                        3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 7.0, 8.0, 9.0, 9.0, 10.0,
                        10.0, 11.0])
    reg3_Ji = np.array([0.0, 0.0, 1.0, 2.0, 7.0, 10.0, 12.0, 23.0, 2.0, 6.0, 15.0, 17.0, 0.0, 2.0, 6.0, 7.0, 22.0,
                        26.0, 0.0, 2.0, 4.0, 16.0, 26.0, 0.0, 2.0, 4.0, 26.0, 1.0, 3.0, 26.0, 0.0,
                        2.0, 26.0, 2.0, 26.0, 2.0, 26.0, 0.0, 1.0, 26.0])
    reg3_ni = np.array([1.0658070028513, - 15.732845290239, 20.944396974307, - 7.6867707878716, 2.6185947787954,
                        - 2.808078114862, 1.2053369696517, - 8.4566812812502E-03, - 1.2654315477714, - 1.1524407806681,
                        0.88521043984318, - 0.64207765181607, 0.38493460186671, - 0.85214708824206, 4.8972281541877,
                        - 3.0502617256965, 0.039420536879154, 0.12558408424308, - 0.2799932969871, 1.389979956946,
                        - 2.018991502357, - 8.2147637173963E-03, - 0.47596035734923, 0.0439840744735,
                        - 0.44476435428739,
                        0.90572070719733, 0.70522450087967, 0.10770512626332, - 0.32913623258954, - 0.50871062041158,
                        - 0.022175400873096, 0.094260751665092, 0.16436278447961, - 0.013503372241348,
                        - 0.014834345352472,
                        5.7922953628084E-04, 3.2308904703711E-03, 8.0964802996215E-05, - 1.6557679795037E-04,
                        - 4.4923899061815E-05])
    R = 0.461526
    tc = 647.096
    pc = 22.064
    rhoc = 322
    # Release on the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam,
    # September 1997
    # 7 Basic Equation for Region 3, Section. 6.1 Basic Equation
    # Table 30 and 31, Page 30 and 31
    delta = rho / rhoc
    tau = tc / T
    fidelta = 0
    for i in range(1, 40):
        fidelta = fidelta + reg3_ni[i] * reg3_Ii[i] * delta ** (reg3_Ii[i] - 1) * tau ** reg3_Ji[i]
    fidelta = fidelta + reg3_ni[0] / delta
    fn_return_value = rho * R * T * delta * fidelta / 1000
    return fn_return_value


@njit(cache=True)
def u3_rhoT(rho, T):
    reg3_Ii = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0,
                        3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 7.0, 8.0, 9.0, 9.0, 10.0,
                        10.0, 11.0])
    reg3_Ji = np.array([0.0, 0.0, 1.0, 2.0, 7.0, 10.0, 12.0, 23.0, 2.0, 6.0, 15.0, 17.0, 0.0, 2.0, 6.0, 7.0, 22.0,
                        26.0, 0.0, 2.0, 4.0, 16.0, 26.0, 0.0, 2.0, 4.0, 26.0, 1.0, 3.0, 26.0, 0.0,
                        2.0, 26.0, 2.0, 26.0, 2.0, 26.0, 0.0, 1.0, 26.0])
    reg3_ni = np.array([1.0658070028513, - 15.732845290239, 20.944396974307, - 7.6867707878716, 2.6185947787954,
                        - 2.808078114862, 1.2053369696517, - 8.4566812812502E-03, - 1.2654315477714, - 1.1524407806681,
                        0.88521043984318, - 0.64207765181607, 0.38493460186671, - 0.85214708824206, 4.8972281541877,
                        - 3.0502617256965, 0.039420536879154, 0.12558408424308, - 0.2799932969871, 1.389979956946,
                        - 2.018991502357, - 8.2147637173963E-03, - 0.47596035734923, 0.0439840744735,
                        - 0.44476435428739,
                        0.90572070719733, 0.70522450087967, 0.10770512626332, - 0.32913623258954, - 0.50871062041158,
                        - 0.022175400873096, 0.094260751665092, 0.16436278447961, - 0.013503372241348,
                        - 0.014834345352472,
                        5.7922953628084E-04, 3.2308904703711E-03, 8.0964802996215E-05, - 1.6557679795037E-04,
                        - 4.4923899061815E-05])
    R = 0.461526
    tc = 647.096
    pc = 22.064
    rhoc = 322
    # Release on the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam,
    # September 1997
    # 7 Basic Equation for Region 3, Section. 6.1 Basic Equation
    # Table 30 and 31, Page 30 and 31
    delta = rho / rhoc
    tau = tc / T
    fitau = 0
    for i in range(1, 40):
        fitau = fitau + reg3_ni[i] * delta ** reg3_Ii[i] * reg3_Ji[i] * tau ** (reg3_Ji[i] - 1)

    fn_return_value = R * T * (tau * fitau)
    return fn_return_value


@njit(cache=True)
def h3_rhoT(rho, T):
    reg3_Ii = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0,
                        3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 7.0, 8.0, 9.0, 9.0, 10.0,
                        10.0, 11.0])
    reg3_Ji = np.array([0.0, 0.0, 1.0, 2.0, 7.0, 10.0, 12.0, 23.0, 2.0, 6.0, 15.0, 17.0, 0.0, 2.0, 6.0, 7.0, 22.0,
                        26.0, 0.0, 2.0, 4.0, 16.0, 26.0, 0.0, 2.0, 4.0, 26.0, 1.0, 3.0, 26.0, 0.0,
                        2.0, 26.0, 2.0, 26.0, 2.0, 26.0, 0.0, 1.0, 26.0])
    reg3_ni = np.array([1.0658070028513, - 15.732845290239, 20.944396974307, - 7.6867707878716, 2.6185947787954,
                        - 2.808078114862, 1.2053369696517, - 8.4566812812502E-03, - 1.2654315477714, - 1.1524407806681,
                        0.88521043984318, - 0.64207765181607, 0.38493460186671, - 0.85214708824206, 4.8972281541877,
                        - 3.0502617256965, 0.039420536879154, 0.12558408424308, - 0.2799932969871, 1.389979956946,
                        - 2.018991502357, - 8.2147637173963E-03, - 0.47596035734923, 0.0439840744735,
                        - 0.44476435428739,
                        0.90572070719733, 0.70522450087967, 0.10770512626332, - 0.32913623258954, - 0.50871062041158,
                        - 0.022175400873096, 0.094260751665092, 0.16436278447961, - 0.013503372241348,
                        - 0.014834345352472,
                        5.7922953628084E-04, 3.2308904703711E-03, 8.0964802996215E-05, - 1.6557679795037E-04,
                        - 4.4923899061815E-05])
    R = 0.461526
    tc = 647.096
    pc = 22.064
    rhoc = 322

    # Release on the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam,
    # September 1997
    # 7 Basic Equation for Region 3, Section. 6.1 Basic Equation
    # Table 30 and 31, Page 30 and 31
    delta = rho / rhoc
    tau = tc / T
    fidelta = 0
    fitau = 0
    for i in range(1, 40):
        fidelta = fidelta + reg3_ni[i] * reg3_Ii[i] * delta ** (reg3_Ii[i] - 1) * tau ** reg3_Ji[i]
        fitau = fitau + reg3_ni[i] * delta ** reg3_Ii[i] * reg3_Ji[i] * tau ** (reg3_Ji[i] - 1)

    fidelta = fidelta + reg3_ni[0] / delta
    fn_return_value = R * T * (tau * fitau + delta * fidelta)
    return fn_return_value


@njit(cache=True)
def s3_rhoT(rho, T):
    reg3_Ii = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0,
                        3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 7.0, 8.0, 9.0, 9.0, 10.0,
                        10.0, 11.0])
    reg3_Ji = np.array([0.0, 0.0, 1.0, 2.0, 7.0, 10.0, 12.0, 23.0, 2.0, 6.0, 15.0, 17.0, 0.0, 2.0, 6.0, 7.0, 22.0,
                        26.0, 0.0, 2.0, 4.0, 16.0, 26.0, 0.0, 2.0, 4.0, 26.0, 1.0, 3.0, 26.0, 0.0,
                        2.0, 26.0, 2.0, 26.0, 2.0, 26.0, 0.0, 1.0, 26.0])
    reg3_ni = np.array([1.0658070028513, - 15.732845290239, 20.944396974307, - 7.6867707878716, 2.6185947787954,
                        - 2.808078114862, 1.2053369696517, - 8.4566812812502E-03, - 1.2654315477714, - 1.1524407806681,
                        0.88521043984318, - 0.64207765181607, 0.38493460186671, - 0.85214708824206, 4.8972281541877,
                        - 3.0502617256965, 0.039420536879154, 0.12558408424308, - 0.2799932969871, 1.389979956946,
                        - 2.018991502357, - 8.2147637173963E-03, - 0.47596035734923, 0.0439840744735,
                        - 0.44476435428739,
                        0.90572070719733, 0.70522450087967, 0.10770512626332, - 0.32913623258954, - 0.50871062041158,
                        - 0.022175400873096, 0.094260751665092, 0.16436278447961, - 0.013503372241348,
                        - 0.014834345352472,
                        5.7922953628084E-04, 3.2308904703711E-03, 8.0964802996215E-05, - 1.6557679795037E-04,
                        - 4.4923899061815E-05])

    R = 0.461526
    tc = 647.096
    pc = 22.064
    rhoc = 322
    # Release on the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam,
    # September 1997
    # 7 Basic Equation for Region 3, Section. 6.1 Basic Equation
    # Table 30 and 31, Page 30 and 31
    delta = rho / rhoc
    tau = tc / T
    fi = 0
    fitau = 0
    for i in range(1, 40):
        fi = fi + reg3_ni[i] * delta ** reg3_Ii[i] * tau ** reg3_Ji[i]
        fitau = fitau + reg3_ni[i] * delta ** reg3_Ii[i] * reg3_Ji[i] * tau ** (reg3_Ji[i] - 1)

    fi = fi + reg3_ni[0] * math.log(delta)
    fn_return_value = R * (tau * fitau - fi)
    return fn_return_value


@njit(cache=True)
def Cp3_rhoT(rho, T):
    reg3_Ii = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0,
                        3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 7.0, 8.0, 9.0, 9.0, 10.0,
                        10.0, 11.0])
    reg3_Ji = np.array([0.0, 0.0, 1.0, 2.0, 7.0, 10.0, 12.0, 23.0, 2.0, 6.0, 15.0, 17.0, 0.0, 2.0, 6.0, 7.0, 22.0,
                        26.0, 0.0, 2.0, 4.0, 16.0, 26.0, 0.0, 2.0, 4.0, 26.0, 1.0, 3.0, 26.0, 0.0,
                        2.0, 26.0, 2.0, 26.0, 2.0, 26.0, 0.0, 1.0, 26.0])
    reg3_ni = np.array([1.0658070028513, - 15.732845290239, 20.944396974307, - 7.6867707878716, 2.6185947787954,
                        - 2.808078114862, 1.2053369696517, - 8.4566812812502E-03, - 1.2654315477714, - 1.1524407806681,
                        0.88521043984318, - 0.64207765181607, 0.38493460186671, - 0.85214708824206, 4.8972281541877,
                        - 3.0502617256965, 0.039420536879154, 0.12558408424308, - 0.2799932969871, 1.389979956946,
                        - 2.018991502357, - 8.2147637173963E-03, - 0.47596035734923, 0.0439840744735,
                        - 0.44476435428739,
                        0.90572070719733, 0.70522450087967, 0.10770512626332, - 0.32913623258954, - 0.50871062041158,
                        - 0.022175400873096, 0.094260751665092, 0.16436278447961, - 0.013503372241348,
                        - 0.014834345352472,
                        5.7922953628084E-04, 3.2308904703711E-03, 8.0964802996215E-05, - 1.6557679795037E-04,
                        - 4.4923899061815E-05])

    R = 0.461526
    tc = 647.096
    pc = 22.064
    rhoc = 322
    # Release on the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam,
    # September 1997
    # 7 Basic Equation for Region 3, Section. 6.1 Basic Equation
    # Table 30 and 31, Page 30 and 31
    delta = rho / rhoc
    tau = tc / T
    fitautau = 0
    fidelta = 0
    fideltatau = 0
    fideltadelta = 0
    for i in range(1, 40):
        fitautau = fitautau + reg3_ni[i] * delta ** reg3_Ii[i] * reg3_Ji[i] * (reg3_Ji[i] - 1) * tau ** (reg3_Ji[i] - 2)
        fidelta = fidelta + reg3_ni[i] * reg3_Ii[i] * delta ** (reg3_Ii[i] - 1) * tau ** reg3_Ji[i]
        fideltatau = fideltatau + reg3_ni[i] * reg3_Ii[i] * delta ** (reg3_Ii[i] - 1) * reg3_Ji[i] * tau ** (
            reg3_Ji[i] - 1)
        fideltadelta = fideltadelta + reg3_ni[i] * reg3_Ii[i] * (reg3_Ii[i] - 1) * delta ** (reg3_Ii[i] - 2) * tau ** \
                       reg3_Ji[i]

    fidelta = fidelta + reg3_ni[0] / delta
    fideltadelta = fideltadelta - reg3_ni[0] / (delta ** 2)
    fn_return_value = R * (- (tau ** 2 * fitautau) + (delta * fidelta - delta * tau * fideltatau) ** 2 / (
        2 * delta * fidelta + delta ** 2 * fideltadelta))
    return fn_return_value


@njit(cache=True)
def Cv3_rhoT(rho, T):
    reg3_Ii = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0,
                        3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 7.0, 8.0, 9.0, 9.0, 10.0,
                        10.0, 11.0])
    reg3_Ji = np.array([0.0, 0.0, 1.0, 2.0, 7.0, 10.0, 12.0, 23.0, 2.0, 6.0, 15.0, 17.0, 0.0, 2.0, 6.0, 7.0, 22.0,
                        26.0, 0.0, 2.0, 4.0, 16.0, 26.0, 0.0, 2.0, 4.0, 26.0, 1.0, 3.0, 26.0, 0.0,
                        2.0, 26.0, 2.0, 26.0, 2.0, 26.0, 0.0, 1.0, 26.0])
    reg3_ni = np.array([1.0658070028513, - 15.732845290239, 20.944396974307, - 7.6867707878716, 2.6185947787954,
                        - 2.808078114862, 1.2053369696517, - 8.4566812812502E-03, - 1.2654315477714, - 1.1524407806681,
                        0.88521043984318, - 0.64207765181607, 0.38493460186671, - 0.85214708824206, 4.8972281541877,
                        - 3.0502617256965, 0.039420536879154, 0.12558408424308, - 0.2799932969871, 1.389979956946,
                        - 2.018991502357, - 8.2147637173963E-03, - 0.47596035734923, 0.0439840744735,
                        - 0.44476435428739,
                        0.90572070719733, 0.70522450087967, 0.10770512626332, - 0.32913623258954, - 0.50871062041158,
                        - 0.022175400873096, 0.094260751665092, 0.16436278447961, - 0.013503372241348,
                        - 0.014834345352472,
                        5.7922953628084E-04, 3.2308904703711E-03, 8.0964802996215E-05, - 1.6557679795037E-04,
                        - 4.4923899061815E-05])

    R = 0.461526
    tc = 647.096
    pc = 22.064
    rhoc = 322
    # Release on the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam,
    # September 1997
    # 7 Basic Equation for Region 3, Section. 6.1 Basic Equation
    # Table 30 and 31, Page 30 and 31
    delta = rho / rhoc
    tau = tc / T
    fitautau = 0
    for i in range(1, 40):
        fitautau = fitautau + reg3_ni[i] * delta ** reg3_Ii[i] * reg3_Ji[i] * (reg3_Ji[i] - 1) * tau ** (reg3_Ji[i] - 2)
    fn_return_value = - R * (tau * tau * fitautau)
    return fn_return_value


@njit(cache=True)
def w3_rhoT(rho, T):
    reg3_Ii = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0,
                        3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 7.0, 8.0, 9.0, 9.0, 10.0,
                        10.0, 11.0])
    reg3_Ji = np.array([0.0, 0.0, 1.0, 2.0, 7.0, 10.0, 12.0, 23.0, 2.0, 6.0, 15.0, 17.0, 0.0, 2.0, 6.0, 7.0, 22.0,
                        26.0, 0.0, 2.0, 4.0, 16.0, 26.0, 0.0, 2.0, 4.0, 26.0, 1.0, 3.0, 26.0, 0.0,
                        2.0, 26.0, 2.0, 26.0, 2.0, 26.0, 0.0, 1.0, 26.0])
    reg3_ni = np.array([1.0658070028513, - 15.732845290239, 20.944396974307, - 7.6867707878716, 2.6185947787954,
                        - 2.808078114862, 1.2053369696517, - 8.4566812812502E-03, - 1.2654315477714, - 1.1524407806681,
                        0.88521043984318, - 0.64207765181607, 0.38493460186671, - 0.85214708824206, 4.8972281541877,
                        - 3.0502617256965, 0.039420536879154, 0.12558408424308, - 0.2799932969871, 1.389979956946,
                        - 2.018991502357, - 8.2147637173963E-03, - 0.47596035734923, 0.0439840744735,
                        - 0.44476435428739,
                        0.90572070719733, 0.70522450087967, 0.10770512626332, - 0.32913623258954, - 0.50871062041158,
                        - 0.022175400873096, 0.094260751665092, 0.16436278447961, - 0.013503372241348,
                        - 0.014834345352472,
                        5.7922953628084E-04, 3.2308904703711E-03, 8.0964802996215E-05, - 1.6557679795037E-04,
                        - 4.4923899061815E-05])
    R = 0.461526
    tc = 647.096
    pc = 22.064
    rhoc = 322
    # Release on the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam,
    # September 1997
    # 7 Basic Equation for Region 3, Section. 6.1 Basic Equation
    # Table 30 and 31, Page 30 and 31
    delta = rho / rhoc
    tau = tc / T
    fitautau = 0
    fidelta = 0
    fideltatau = 0
    fideltadelta = 0
    for i in range(1, 40):
        fitautau = fitautau + reg3_ni[i] * delta ** reg3_Ii[i] * reg3_Ji[i] * (reg3_Ji[i] - 1) * tau ** (reg3_Ji[i] - 2)
        fidelta = fidelta + reg3_ni[i] * reg3_Ii[i] * delta ** (reg3_Ii[i] - 1) * tau ** reg3_Ji[i]
        fideltatau = fideltatau + reg3_ni[i] * reg3_Ii[i] * delta ** (reg3_Ii[i] - 1) * reg3_Ji[i] * tau ** (
            reg3_Ji[i] - 1)
        fideltadelta = fideltadelta + reg3_ni[i] * reg3_Ii[i] * (reg3_Ii[i] - 1) * delta ** (reg3_Ii[i] - 2) * tau ** \
                       reg3_Ji[i]

    fidelta = fidelta + reg3_ni[0] / delta
    fideltadelta = fideltadelta - reg3_ni[0] / (delta ** 2)
    fn_return_value = (1000 * R * T * (
        2 * delta * fidelta + delta ** 2 * fideltadelta - (delta * fidelta - delta * tau * fideltatau) ** 2 / (
        tau ** 2 * fitautau))) ** 0.5
    return fn_return_value


@njit(cache=True)
def T3_ph_part1(p, h):
    # Subregion 3a
    # Eq 2, Table 3, Page 7
    T3_ph_part1_Ii = np.array([- 12.0, - 12.0, - 12.0, - 12.0, - 12.0, - 12.0, - 12.0, - 12.0, - 10.0, - 10.0, - 10.0,
                               - 8.0, - 8.0, - 8.0, - 8.0, - 5.0, - 3.0, - 2.0, - 2.0, - 2.0, - 1.0, - 1.0, 0.0, 0.0,
                               1.0,
                               3.0,
                               3.0, 4.0, 4.0, 10.0, 12.0])
    T3_ph_part1_Ji = np.array(
        [0.0, 1.0, 2.0, 6.0, 14.0, 16.0, 20.0, 22.0, 1.0, 5.0, 12.0, 0.0, 2.0, 4.0, 10.0, 2.0, 0.0, 1.0,
         3.0, 4.0, 0.0, 2.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 3.0, 4.0, 5.0])
    T3_ph_part1_ni = np.array(
        [- 1.33645667811215E-07, 4.55912656802978E-06, - 1.46294640700979E-05, 6.3934131297008E-03,
         372.783927268847, - 7186.54377460447, 573494.7521034, - 2675693.29111439,
         - 3.34066283302614E-05,
         - 2.45479214069597E-02, 47.8087847764996, 7.64664131818904E-06, 1.28350627676972E-03,
         1.71219081377331E-02, - 8.51007304583213, - 1.36513461629781E-02, - 3.84460997596657E-06,
         3.37423807911655E-03, - 0.551624873066791, 0.72920227710747, - 9.92522757376041E-03,
         - 0.119308831407288, 0.793929190615421, 0.454270731799386, 0.20999859125991,
         - 6.42109823904738E-03,
         - 0.023515586860454, 2.52233108341612E-03, - 7.64885133368119E-03, 1.36176427574291E-02,
         - 1.33027883575669E-02])

    ps = p / 100
    hs = h / 2300
    Ts = 0
    for i in range(31):
        Ts = Ts + T3_ph_part1_ni[i] * (ps + 0.24) ** T3_ph_part1_Ii[i] * (hs - 0.615) ** T3_ph_part1_Ji[i]
    fn_return_value = Ts * 760
    return fn_return_value


@njit(cache=True)
def T3_ph_part2(p, h):
    # Subregion 3b
    # Eq 3, Table 4, Page 7,8
    T3_ph_part2_Ii = np.array([- 12.0, - 12.0, - 10.0, - 10.0, - 10.0, - 10.0, - 10.0, - 8.0, - 8.0, - 8.0,
                               - 8.0, - 8.0, - 6.0, - 6.0, - 6.0, - 4.0, - 4.0, - 3.0, - 2.0,
                               - 2.0, - 1.0, - 1.0, - 1.0, - 1.0, - 1.0, - 1.0, 0.0, 0.0, 1.0,
                               3.0, 5.0, 6.0, 8.0])
    T3_ph_part2_Ji = np.array([0.0, 1.0, 0.0, 1.0, 5.0, 10.0, 12.0, 0.0, 1.0, 2.0, 4.0, 10.0, 0.0, 1.0, 2.0, 0.0, 1.0,
                               5.0, 0.0, 4.0, 2.0, 4.0, 6.0, 10.0, 14.0, 16.0, 0.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    T3_ph_part2_ni = np.array(
        [3.2325457364492E-05, - 1.27575556587181E-04, - 4.75851877356068E-04, 1.56183014181602E-03,
         0.105724860113781, - 85.8514221132534, 724.140095480911, 2.96475810273257E-03,
         - 5.92721983365988E-03, - 1.26305422818666E-02, - 0.115716196364853, 84.9000969739595,
         - 1.08602260086615E-02, 1.54304475328851E-02, 7.50455441524466E-02, 2.52520973612982E-02,
         - 6.02507901232996E-02, - 3.07622221350501, - 5.74011959864879E-02, 5.03471360939849,
         - 0.925081888584834, 3.91733882917546, - 77.314600713019, 9493.08762098587,
         - 1410437.19679409,
         8491662.30819026, 0.861095729446704, 0.32334644281172, 0.873281936020439,
         - 0.436653048526683,
         0.286596714529479, - 0.131778331276228, 6.76682064330275E-03])

    hs = h / 2800
    ps = p / 100
    Ts = 0
    for i in range(33):
        Ts = Ts + T3_ph_part2_ni[i] * (ps + 0.298) ** T3_ph_part2_Ii[i] * (hs - 0.72) ** T3_ph_part2_Ji[i]
    fn_return_value = Ts * 860
    return fn_return_value


@njit(cache=True)
def T3_ph(p, h):
    R = 0.461526
    tc = 647.096
    pc = 22.064
    rhoc = 322
    # Revised Supplementary Release on Backward Equations for the Functions T(p,h), v(p,h) and T(p,s), v(p,s) for
    # Region 3 of the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam
    # 2004
    # Section 3.3 Backward Equations T(p,h) and v(p,h) for Subregions 3a and 3b
    # Boundary equation, Eq 1 Page 5
    h3ab = (2014.64004206875 + 3.74696550136983 * p - 2.19921901054187E-02 * p ** 2 + 8.7513168600995E-05 * p ** 3)
    if h < h3ab:
        return T3_ph_part1(p, h)
    else:
        return T3_ph_part2(p, h)

    # return fn_return_value


@njit(cache=True)
def v3_ph_part1(p, h):
    # Subregion 3a
    # Eq 4, Table 6, Page 9
    v3_ph_part1_Ii = np.array(
        [- 12.0, - 12.0, - 12.0, - 12.0, - 10.0, - 10.0, - 10.0, - 8.0, - 8.0, - 6.0, - 6.0, - 6.0,
         - 4.0, - 4.0, - 3.0, - 2.0, - 2.0, - 1.0, - 1.0, - 1.0, - 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0,
         2.0, 3.0, 4.0, 5.0, 8.0])
    v3_ph_part1_Ji = np.array([6.0, 8.0, 12.0, 18.0, 4.0, 7.0, 10.0, 5.0, 12.0, 3.0, 4.0, 22.0, 2.0, 3.0, 7.0, 3.0,
                               16.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 2.0, 2.0])

    v3_ph_part1_ni = np.array([5.29944062966028E-03, - 0.170099690234461, 11.1323814312927, - 2178.98123145125,
                               - 5.06061827980875E-04, 0.556495239685324, - 9.43672726094016, - 0.297856807561527,
                               93.9353943717186,
                               1.92944939465981E-02, 0.421740664704763, - 3689141.2628233, - 7.37566847600639E-03,
                               - 0.354753242424366, - 1.99768169338727, 1.15456297059049, 5683.6687581596,
                               8.08169540124668E-03,
                               0.172416341519307, 1.04270175292927, - 0.297691372792847, 0.560394465163593,
                               0.275234661176914,
                               - 0.148347894866012, - 6.51142513478515E-02, - 2.92468715386302, 6.64876096952665E-02,
                               3.52335014263844, - 1.46340792313332E-02, - 2.24503486668184, 1.10533464706142,
                               - 4.08757344495612E-02])
    ps = p / 100
    hs = h / 2100
    vs = 0
    for i in range(32):
        vs = vs + v3_ph_part1_ni[i] * (ps + 0.128) ** v3_ph_part1_Ii[i] * (hs - 0.727) ** v3_ph_part1_Ji[i]
    fn_return_value = vs * 0.0028
    return fn_return_value


@njit(cache=True)
def v3_ph_part2(p, h):
    v3_ph_part2_Ii = np.array(
        [- 12.0, - 12.0, - 8.0, - 8.0, - 8.0, - 8.0, - 8.0, - 8.0, - 6.0, - 6.0, - 6.0, - 6.0, - 6.0,
         - 6.0, - 4.0, - 4.0, - 4.0, - 3.0, - 3.0, - 2.0, - 2.0, - 1.0, - 1.0, - 1.0, - 1.0, 0.0, 1.0,
         1.0,
         2.0, 2.0])
    v3_ph_part2_Ji = np.array([0.0, 1.0, 0.0, 1.0, 3.0, 6.0, 7.0, 8.0, 0.0, 1.0, 2.0, 5.0, 6.0, 10.0, 3.0, 6.0,
                               10.0, 0.0, 2.0, 1.0, 2.0, 0.0, 1.0, 4.0, 5.0, 0.0, 0.0, 1.0, 2.0, 6.0])
    v3_ph_part2_ni = np.array(
        [- 2.25196934336318E-09, 1.40674363313486E-08, 2.3378408528056E-06, - 3.31833715229001E-05,
         1.07956778514318E-03, - 0.271382067378863, 1.07202262490333, - 0.853821329075382,
         - 2.15214194340526E-05, 7.6965608822273E-04, - 4.31136580433864E-03, 0.453342167309331,
         - 0.507749535873652, - 100.475154528389, - 0.219201924648793, - 3.21087965668917,
         607.567815637771, 5.57686450685932E-04, 0.18749904002955, 9.05368030448107E-03,
         0.285417173048685, 3.29924030996098E-02, 0.239897419685483, 4.82754995951394,
         - 11.8035753702231, 0.169490044091791, - 1.79967222507787E-02, 3.71810116332674E-02,
         - 5.36288335065096E-02, 1.6069710109252])
    ps = p / 100
    hs = h / 2800
    vs = 0
    for i in range(30):
        vs = vs + v3_ph_part2_ni[i] * (ps + 0.0661) ** v3_ph_part2_Ii[i] * (hs - 0.72) ** v3_ph_part2_Ji[i]
    fn_return_value = vs * 0.0088
    return fn_return_value


@njit(cache=True)
def v3_ph(p, h):
    R = 0.461526
    tc = 647.096
    pc = 22.064
    rhoc = 322
    # Revised Supplementary Release on Backward Equations for the Functions T(p,h), v(p,h) and T(p,s), v(p,s) for
    # Region 3 of the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam
    # 2004
    # Section 3.3 Backward Equations T(p,h) and v(p,h) for Subregions 3a and 3b
    # Boundary equation, Eq 1 Page 5
    h3ab = (2014.64004206875 + 3.74696550136983 * p - 2.19921901054187E-02 * p ** 2 + 8.7513168600995E-05 * p ** 3)
    if h < h3ab:
        return v3_ph_part1(p, h)
    else:
        # Subregion 3b
        # Eq 5, Table 7, Page 9
        return v3_ph_part2(p, h)

    # return fn_return_value


@njit(cache=True)
def T3_ps_part1(p, s):
    # Subregion 3a
    # Eq 6, Table 10, Page 11
    T3_ps_part1_Ii = np.array([- 12.0, - 12.0, - 10.0, - 10.0, - 10.0, - 10.0, - 8.0, - 8.0, - 8.0, - 8.0,
                               - 6.0, - 6.0, - 6.0, - 5.0, - 5.0, - 5.0, - 4.0, - 4.0, - 4.0,
                               - 2.0, - 2.0, - 1.0, - 1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 3.0, 8.0, 8.0, 10.0])
    T3_ps_part1_Ji = np.array([28.0, 32.0, 4.0, 10.0, 12.0, 14.0, 5.0, 7.0, 8.0, 28.0, 2.0, 6.0, 32.0, 0.0, 14.0,
                               32.0, 6.0, 10.0, 36.0, 1.0, 4.0, 1.0, 6.0, 0.0, 1.0, 4.0, 0.0, 0.0, 3.0, 2.0, 0.0, 1.0,
                               2.0])
    T3_ps_part1_ni = np.array(
        [1500420082.63875, - 159397258480.424, 5.02181140217975E-04, - 67.2057767855466, 1450.58545404456,
         - 8238.8953488889, - 0.154852214233853, 11.2305046746695, - 29.7000213482822, 43856513263.5495,
         1.37837838635464E-03, - 2.97478527157462, 9717779473494.13, - 5.71527767052398E-05, 28830.794977842,
         - 74442828926270.3, 12.8017324848921, - 368.275545889071, 6.64768904779177E+15, 0.044935925195888,
         - 4.22897836099655, - 0.240614376434179, - 4.74341365254924, 0.72409399912611, 0.923874349695897,
         3.99043655281015, 3.84066651868009E-02, - 3.59344365571848E-03, - 0.735196448821653,
         0.188367048396131, 1.41064266818704E-04, - 2.57418501496337E-03, 1.23220024851555E-03])

    sigma = s / 4.4
    ps = p / 100
    teta = 0
    for i in range(33):
        teta = teta + T3_ps_part1_ni[i] * (ps + 0.24) ** T3_ps_part1_Ii[i] * (sigma - 0.703) ** T3_ps_part1_Ji[i]
    fn_return_value = teta * 760
    return fn_return_value


@njit(cache=True)
def T3_ps_part2(p, s):
    # Subregion 3b
    # Eq 7, Table 11, Page 11
    T3_ps_part2_Ii = np.array([- 12.0, - 12.0, - 12.0, - 12.0, - 8.0, - 8.0, - 8.0, - 6.0, - 6.0, - 6.0,
                               - 5.0, - 5.0, - 5.0, - 5.0, - 5.0, - 4.0, - 3.0, - 3.0, - 2.0, 0.0, 2.0, 3.0, 4.0,
                               5.0, 6.0, 8.0, 12.0, 14.0])
    T3_ps_part2_Ji = np.array(
        [1.0, 3.0, 4.0, 7.0, 0.0, 1.0, 3.0, 0.0, 2.0, 4.0, 0.0, 1.0, 2.0, 4.0, 6.0, 12.0, 1.0, 6.0,
         2.0, 0.0, 1.0, 1.0, 0.0, 24.0, 0.0, 3.0, 1.0, 2.0])
    T3_ps_part2_ni = np.array(
        [0.52711170160166, - 40.1317830052742, 153.020073134484, - 2247.99398218827, - 0.193993484669048,
         - 1.40467557893768, 42.6799878114024, 0.752810643416743, 22.6657238616417, - 622.873556909932,
         - 0.660823667935396, 0.841267087271658, - 25.3717501764397, 485.708963532948, 880.531517490555,
         2650155.92794626, - 0.359287150025783, - 656.991567673753, 2.41768149185367, 0.856873461222588,
         0.655143675313458, - 0.213535213206406, 5.62974957606348E-03, - 316955725450471,
         -6.99997000152457E-04, 1.19845803210767E-02, 1.93848122022095E-05, - 2.15095749182309E-05])

    sigma = s / 5.3
    ps = p / 100
    teta = 0
    for i in range(28):
        teta = teta + T3_ps_part2_ni[i] * (ps + 0.76) ** T3_ps_part2_Ii[i] * (sigma - 0.818) ** T3_ps_part2_Ji[i]
    fn_return_value = teta * 860
    return fn_return_value


@njit(cache=True)
def T3_ps(p, s):
    R = 0.461526
    tc = 647.096
    pc = 22.064
    rhoc = 322
    # Revised Supplementary Release on Backward Equations for the Functions T(p,h), v(p,h) and T(p,s), v(p,s) for
    # Region 3 of the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam
    # 2004
    # 3.4 Backward Equations T(p,s) and v(p,s) for Subregions 3a and 3b
    # Boundary equation, Eq 6 Page 11
    if s <= 4.41202148223476:
        return T3_ps_part1(p, s)
    else:
        return T3_ps_part2(p, s)


@njit(cache=True)
def v3_ps_part1(p, s):
    # Subregion 3a
    # Eq 8, Table 13, Page 14
    v3_ps_part1_Ii = np.array([- 12.0, - 12.0, - 12.0, - 10.0, - 10.0, - 10.0, - 10.0, - 8.0, - 8.0, - 8.0,
                               - 8.0, - 6.0, - 5.0, - 4.0, - 3.0, - 3.0, - 2.0, - 2.0, - 1.0,
                               - 1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 4.0, 5.0, 6.0])
    v3_ps_part1_Ji = np.array([10.0, 12.0, 14.0, 4.0, 8.0, 10.0, 20.0, 5.0, 6.0, 14.0, 16.0, 28.0,
                               1.0, 5.0, 2.0, 4.0, 3.0, 8.0, 1.0, 2.0, 0.0, 1.0, 3.0, 0.0, 0.0, 2.0, 2.0, 0.0])
    v3_ps_part1_ni = np.array(
        [79.5544074093975, - 2382.6124298459, 17681.3100617787, - 1.10524727080379E-03, - 15.3213833655326,
         297.544599376982, - 35031520.6871242, 0.277513761062119, - 0.523964271036888, - 148011.182995403,
         1600148.99374266, 1708023226634.27, 2.46866996006494E-04, 1.6532608479798, - 0.118008384666987,
         2.537986423559, 0.965127704669424, - 28.2172420532826, 0.203224612353823, 1.10648186063513,
         0.52612794845128, 0.277000018736321, 1.08153340501132, - 7.44127885357893E-02, 1.64094443541384E-02,
         - 6.80468275301065E-02, 0.025798857610164, - 1.45749861944416E-04])

    ps = p / 100
    sigma = s / 4.4
    omega = 0
    for i in range(28):
        omega = omega + v3_ps_part1_ni[i] * (ps + 0.187) ** v3_ps_part1_Ii[i] * (sigma - 0.755) ** v3_ps_part1_Ji[i]
    fn_return_value = omega * 0.0028
    return fn_return_value


@njit(cache=True)
def v3_ps_part2(p, s):
    v3_ps_part2_Ii = np.array([- 12.0, - 12.0, - 12.0, - 12.0, - 12.0, - 12.0, - 10.0, - 10.0, - 10.0, - 10.0, - 8.0,
                               - 5.0, - 5.0, - 5.0, - 4.0, - 4.0, - 4.0, - 4.0, - 3.0, - 2.0, - 2.0, - 2.0, - 2.0,
                               - 2.0,
                               - 2.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0])
    v3_ps_part2_Ji = np.array([0.0, 1.0, 2.0, 3.0, 5.0, 6.0, 0.0, 1.0, 2.0, 4.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0,
                               3.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 12.0, 0.0, 1.0, 2.0, 0.0, 2.0, 2.0])
    v3_ps_part2_ni = np.array([5.91599780322238E-05, - 1.85465997137856E-03, 1.04190510480013E-02, 5.9864730203859E-03,
                               - 0.771391189901699, 1.72549765557036, - 4.67076079846526E-04, 1.34533823384439E-02,
                               - 8.08094336805495E-02, 0.508139374365767, 1.28584643361683E-03, - 1.63899353915435,
                               5.86938199318063, - 2.92466667918613, - 6.14076301499537E-03, 5.76199014049172,
                               - 12.1613320606788,
                               1.67637540957944, - 7.44135838773463, 3.78168091437659E-02, 4.01432203027688,
                               16.0279837479185,
                               3.17848779347728, - 3.58362310304853, - 1159952.60446827, 0.199256573577909,
                               - 0.122270624794624,
                               - 19.1449143716586, - 1.50448002905284E-02, 14.6407900162154, - 3.2747778718823])

    ps = p / 100
    sigma = s / 5.3
    omega = 0
    for i in range(31):
        omega = omega + v3_ps_part2_ni[i] * (ps + 0.298) ** v3_ps_part2_Ii[i] * (sigma - 0.816) ** v3_ps_part2_Ji[i]
    fn_return_value = omega * 0.0088
    return fn_return_value


@njit(cache=True)
def v3_ps(p, s):
    R = 0.461526
    tc = 647.096
    pc = 22.064
    rhoc = 322
    # Revised Supplementary Release on Backward Equations for the Functions T(p,h), v(p,h) and T(p,s), v(p,s) for
    # Region 3 of the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam
    # 2004
    # 3.4 Backward Equations T(p,s) and v(p,s) for Subregions 3a and 3b
    # Boundary equation, Eq 6 Page 11
    if s <= 4.41202148223476:
        return v3_ps_part1(p, s)
    else:
        return v3_ps_part2(p, s)
        # Subregion 3b
        # Eq 9, Table 14, Page 14
    # return fn_return_value


@njit(cache=True)
def p3_hs_part1(h, s):
    # Subregion 3a
    # Eq 1, Table 3, Page 8
    p3_hs_part1_Ii = np.array(
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 5.0, 6.0,
         7.0, 8.0, 10.0, 10.0, 14.0, 18.0, 20.0, 22.0, 22.0, 24.0, 28.0, 28.0, 32.0, 32.0])
    p3_hs_part1_Ji = np.array([0.0, 1.0, 5.0, 0.0, 3.0, 4.0, 8.0, 14.0, 6.0, 16.0, 0.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0,
                               28.0, 28.0, 24.0, 1.0, 32.0, 36.0, 22.0, 28.0, 36.0, 16.0, 28.0, 36.0, 16.0, 36.0, 10.0,
                               28.0])
    p3_hs_part1_ni = np.array(
        [7.70889828326934, - 26.0835009128688, 267.416218930389, 17.2221089496844, - 293.54233214597,
         614.135601882478, - 61056.2757725674, - 65127225.1118219, 73591.9313521937,
         - 11664650591.4191,
         35.5267086434461, - 596.144543825955, - 475.842430145708, 69.6781965359503, 335.674250377312,
         25052.6809130882, 146997.380630766, 5.38069315091534E+19, 1.43619827291346E+21,
         3.64985866165994E+19,
         - 2547.41561156775, 2.40120197096563E+27, - 3.93847464679496E+29, 1.47073407024852E+24,
         - 4.26391250432059E+31, 1.94509340621077E+38, 6.66212132114896E+23, 7.06777016552858E+33,
         1.75563621975576E+41, 1.08408607429124E+28, 7.30872705175151E+43, 1.5914584739887E+24,
         3.77121605943324E+40])

    sigma = s / 4.4
    eta = h / 2300
    ps = 0
    for i in range(33):
        ps = ps + p3_hs_part1_ni[i] * (eta - 1.01) ** p3_hs_part1_Ii[i] * (sigma - 0.75) ** p3_hs_part1_Ji[i]
    fn_return_value = ps * 99
    return fn_return_value


@njit(cache=True)
def p3_hs_part2(h, s):
    # Subregion 3b
    # Eq 2, Table 4, Page 8
    p3_hs_part2_Ii = np.array(
        [- 12.0, - 12.0, - 12.0, - 12.0, - 12.0, - 10.0, - 10.0, - 10.0, - 10.0, - 8.0, - 8.0, - 6.0,
         - 6.0, - 6.0, - 6.0, - 5.0, - 4.0, - 4.0, - 4.0, - 3.0, - 3.0, - 3.0, - 3.0, - 2.0, - 2.0,
         - 1.0,
         0.0, 2.0, 2.0, 5.0, 6.0, 8.0, 10.0, 14.0, 14.0])
    p3_hs_part2_Ji = np.array(
        [2.0, 10.0, 12.0, 14.0, 20.0, 2.0, 10.0, 14.0, 18.0, 2.0, 8.0, 2.0, 6.0, 7.0, 8.0, 10.0, 4.0,
         5.0, 8.0, 1.0, 3.0, 5.0, 6.0, 0.0, 1.0, 0.0, 3.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 3.0, 7.0])
    p3_hs_part2_ni = np.array(
        [1.25244360717979E-13, - 1.26599322553713E-02, 5.06878030140626, 31.7847171154202, - 391041.161399932,
         - 9.75733406392044E-11, - 18.6312419488279, 510.973543414101, 373847.005822362, 2.99804024666572E-08,
         20.0544393820342, - 4.98030487662829E-06, - 10.230180636003, 55.2819126990325, - 206.211367510878,
         - 7940.12232324823, 7.82248472028153, - 58.6544326902468, 3550.73647696481, - 1.15303107290162E-04,
         - 1.75092403171802, 257.98168774816, - 727.048374179467, 1.21644822609198E-04, 3.93137871762692E-02,
         7.04181005909296E-03, - 82.910820069811, - 0.26517881813125, 13.7531682453991, - 52.2394090753046,
         2405.56298941048, - 22736.1631268929, 89074.6343932567, - 23923456.5822486, 5687958081.29714])

    sigma = s / 5.3
    eta = h / 2800
    ps = 0
    for i in range(35):
        ps = ps + p3_hs_part2_ni[i] * (eta - 0.681) ** p3_hs_part2_Ii[i] * (sigma - 0.792) ** p3_hs_part2_Ji[i]
    fn_return_value = 16.6 / ps
    return fn_return_value


@njit(cache=True)
def p3_hs(h, s):
    R = 0.461526
    tc = 647.096
    pc = 22.064
    rhoc = 322
    # Supplementary Release on Backward Equations ( ) , p h s for Region 3,
    # Equations as a Function of h and s for the Region Boundaries, and an Equation
    # ( ) sat , T hs for Region 4 of the IAPWS Industrial Formulation 1997 for the
    # Thermodynamic Properties of Water and Steam
    # 2004
    # Section 3 Backward Functions p(h,s), T(h,s), and v(h,s) for Region 3
    if s < 4.41202148223476:
        return p3_hs_part1(h, s)
    else:
        return p3_hs_part2(h, s)

    # return fn_return_value


@njit(cache=True)
def h3_pT(p, T):
    # Not avalible with IF 97
    # Solve function T3_ph-T=0 with half interval method.
    # ver2.6 Start corrected bug
    Ts = 0
    if p < 22.06395:
        Ts = T4_p(p)
        if T <= Ts:
            High_Bound = h4L_p(p)
            Low_Bound = h1_pT(p, 623.15)
        else:
            Low_Bound = h4V_p(p)
            High_Bound = h2_pT(p, B23T_p(p))
    else:
        Low_Bound = h1_pT(p, 623.15)
        High_Bound = h2_pT(p, B23T_p(p))

    # ver2.6 End corrected bug
    hs = 0
    Ts = T + 1
    iter_count = 0
    while abs(T - Ts) > 0.000001 and iter_count < 1000:
        hs = (Low_Bound + High_Bound) / 2
        Ts = T3_ph(p, hs)
        if Ts > T:
            High_Bound = hs
        else:
            Low_Bound = hs

        iter_count += 1

    if iter_count >= 1000:
        return math.nan

    fn_return_value = hs
    return fn_return_value


@njit(cache=True)
def T3_prho(p, rho):
    # Solve by iteration. Observe that fo low temperatures this equation has 2 solutions.
    # Solve with half interval method
    Low_Bound = 623.15
    High_Bound = 1073.15
    iter_count = 0
    Ts = 0
    ps = 0
    while abs(p - ps) > 0.00000001 and iter_count < 1000:
        Ts = (Low_Bound + High_Bound) / 2
        ps = p3_rhoT(rho, Ts)
        if ps > p:
            High_Bound = Ts
        else:
            Low_Bound = Ts

        iter_count += 1

    if iter_count >= 1000:
        return math.nan

    fn_return_value = Ts
    return fn_return_value


@njit(cache=True)
def p4_T(T):
    # Section 8.1 The Saturation-Pressure Equation
    # Eq 30, Page 33
    teta = T - 0.23855557567849 / (T - 650.17534844798)
    a = teta ** 2 + 1167.0521452767 * teta - 724213.16703206
    b = - 17.073846940092 * teta ** 2 + 12020.82470247 * teta - 3232555.0322333
    c = 14.91510861353 * teta ** 2 - 4823.2657361591 * teta + 405113.40542057
    fn_return_value = (2 * c / (- b + (b ** 2 - 4 * a * c) ** 0.5)) ** 4
    return fn_return_value


@njit(cache=True)
def T4_p(p):
    # Release on the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam,
    # September 1997
    # Section 8.2 The Saturation-Temperature Equation
    # Eq 31, Page 34
    beta = p ** 0.25
    e = beta ** 2 - 17.073846940092 * beta + 14.91510861353
    f = 1167.0521452767 * beta ** 2 + 12020.82470247 * beta - 4823.2657361591
    g = - 724213.16703206 * beta ** 2 - 3232555.0322333 * beta + 405113.40542057
    d = 2 * g / (- f - (f ** 2 - 4 * e * g) ** 0.5)
    fn_return_value = (650.17534844798 + d - (
        (650.17534844798 + d) ** 2 - 4 * (- 0.23855557567849 + 650.17534844798 * d)) ** 0.5) / 2
    return fn_return_value


@njit(cache=True)
def h4_s_part1(s):
    h4_s_part1_Ii = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 8.0,
                              12.0, 12.0, 14.0, 14.0, 16.0, 20.0, 20.0, 22.0, 24.0, 28.0, 32.0, 32.0])
    h4_s_part1_Ji = np.array([14.0, 36.0, 3.0, 16.0, 0.0, 5.0, 4.0, 36.0, 4.0, 16.0, 24.0, 18.0, 24.0,
                              1.0, 4.0, 2.0, 4.0, 1.0, 22.0, 10.0, 12.0, 28.0, 8.0, 3.0, 0.0, 6.0, 8.0])
    h4_s_part1_ni = np.array([0.332171191705237, 6.11217706323496E-04, - 8.82092478906822, - 0.45562819254325,
                              - 2.63483840850452E-05, - 22.3949661148062, - 4.28398660164013, - 0.616679338856916,
                              - 14.682303110404, 284.523138727299, - 113.398503195444, 1156.71380760859,
                              395.551267359325,
                              - 1.54891257229285, 19.4486637751291, - 3.57915139457043, - 3.35369414148819,
                              - 0.66442679633246,
                              32332.1885383934, 3317.66744667084, - 22350.1257931087, 5739538.75852936,
                              173.226193407919,
                              - 3.63968822121321E-02, 8.34596332878346E-07, 5.03611916682674, 65.5444787064505])

    sigma = s / 3.8
    eta = 0
    for i in range(27):
        eta = eta + h4_s_part1_ni[i] * (sigma - 1.09) ** h4_s_part1_Ii[i] * (sigma + 0.0000366) ** h4_s_part1_Ji[i]
    fn_return_value = eta * 1700
    return fn_return_value


@njit(cache=True)
def h4_s_part2(s):
    h4_s_part2_Ii = np.array(
        [0.0, 0.0, 0.0, 0.0, 2.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 7.0, 7.0, 7.0, 10.0, 10.0, 10.0, 32.0, 32.0])
    h4_s_part2_Ji = np.array(
        [1.0, 4.0, 10.0, 16.0, 1.0, 36.0, 3.0, 16.0, 20.0, 36.0, 4.0, 2.0, 28.0, 32.0, 14.0, 32.0, 36.0, 0.0,
         6.0])
    h4_s_part2_ni = np.array([0.822673364673336, 0.181977213534479, - 0.011200026031362, - 7.46778287048033E-04,
                              -0.179046263257381, 4.24220110836657E-02, - 0.341355823438768, - 2.09881740853565,
                              -8.22477343323596, - 4.99684082076008, 0.191413958471069, 5.81062241093136E-02,
                              - 1655.05498701029,
                              1588.70443421201, - 85.0623535172818, - 31771.4386511207, - 94589.0406632871,
                              - 1.3927384708869E-06,
                              0.63105253224098])

    sigma = s / 3.8
    eta = 0
    for i in range(19):
        eta = eta + h4_s_part2_ni[i] * (sigma - 1.09) ** h4_s_part2_Ii[i] * (sigma + 0.0000366) ** h4_s_part2_Ji[i]
    fn_return_value = eta * 1700
    return fn_return_value


@njit(cache=True)
def h4_s_part3(s):
    h4_s_part3_Ii = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 5.0, 6.0, 7.0, 8.0, 8.0, 12.0, 16.0, 22.0, 22.0, 24.0, 36.0])
    h4_s_part3_Ji = np.array([0.0, 3.0, 4.0, 0.0, 12.0, 36.0, 12.0, 16.0, 2.0, 20.0, 32.0, 36.0, 2.0, 32.0, 7.0, 20.0])
    h4_s_part3_ni = np.array(
        [1.04351280732769, - 2.27807912708513, 1.80535256723202, 0.420440834792042, - 105721.24483466,
         4.36911607493884E+24, - 328032702839.753, - 6.7868676080427E+15, 7439.57464645363,
         - 3.56896445355761E+19, 1.67590585186801E+31, - 3.55028625419105E+37, 396611982166.538,
         - 4.14716268484468E+40, 3.59080103867382E+18, - 1.16994334851995E+40])

    sigma = s / 5.9
    eta = 0
    for i in range(16):
        eta = eta + h4_s_part3_ni[i] * (sigma - 1.02) ** h4_s_part3_Ii[i] * (sigma - 0.726) ** h4_s_part3_Ji[i]
    fn_return_value = eta ** 4 * 2800
    return fn_return_value


@njit(cache=True)
def h4_s_part4(s):
    h4_s_part4_Ii = np.array([1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 7.0, 8.0, 8.0, 10.0, 12.0, 12.0, 18.0,
                              20.0, 24.0, 28.0, 28.0, 28.0, 28.0, 28.0, 32.0, 32.0, 32.0, 32.0, 32.0,
                              36.0, 36.0, 36.0, 36.0, 36.0])
    h4_s_part4_Ji = np.array([8.0, 24.0, 4.0, 32.0, 1.0, 2.0, 7.0, 5.0, 12.0, 1.0, 0.0, 7.0, 10.0, 12.0,
                              32.0, 8.0, 12.0, 20.0, 22.0, 24.0, 2.0, 7.0, 12.0, 14.0, 24.0, 10.0, 12.0, 20.0, 22.0,
                              28.0])
    h4_s_part4_ni = np.array(
        [- 524.581170928788, - 9269472.18142218, - 237.385107491666, 21077015581.2776, - 23.9494562010986,
         221.802480294197, - 5104725.33393438, 1249813.96109147, 2000084369.96201, - 815.158509791035,
         - 157.612685637523, - 11420042233.2791, 6.62364680776872E+15, - 2.27622818296144E+18,
         - 1.71048081348406E+31, 6.60788766938091E+15, 1.66320055886021E+22, - 2.18003784381501E+29,
         - 7.87276140295618E+29, 1.51062329700346E+31, 7957321.70300541, 1.31957647355347E+15,
         - 3.2509706829914E+23, - 4.18600611419248E+25, 2.97478906557467E+34, - 9.53588761745473E+19,
         1.66957699620939E+24, - 1.75407764869978E+32, 3.47581490626396E+34, - 7.10971318427851E+38])
    sigma1 = s / 5.21
    sigma2 = s / 9.2
    eta = 0
    for i in range(30):
        eta = eta + h4_s_part4_ni[i] * (1 / sigma1 - 0.513) ** h4_s_part4_Ii[i] * (sigma2 - 0.524) ** h4_s_part4_Ji[i]
    fn_return_value = math.exp(eta) * 2800
    return fn_return_value


@njit(cache=True)
def h4_s(s):
    # Supplementary Release on Backward Equations ( ) , p h s for Region 3,Equations as a Function of h and s for the
    # Region Boundaries, and an Equation( ) sat , T hs for Region 4 of the IAPWS Industrial Formulation 1997 for the
    # Thermodynamic Properties of Water and Steam
    # 4 Equations for Region Boundaries Given Enthalpy and Entropy
    # Se picture page 14
    if - 0.0001545495919 < s <= 3.77828134:
        # hL1_s
        # Eq 3,Table 9,Page 16
        return h4_s_part1(s)
    elif 3.77828134 < s <= 4.41202148223476:
        # hL3_s
        # Eq 4,Table 10,Page 16
        return h4_s_part2(s)
    elif 4.41202148223476 < s <= 5.85:
        # Section 4.4 Equations ( ) 2ab " h s and ( ) 2c3b "h s for the Saturated Vapor Line
        # Page 19, Eq 5
        # hV2c3b_s(s)
        return h4_s_part3(s)
    elif 5.85 < s < 9.155759395:
        # Section 4.4 Equations ( ) 2ab " h s and ( ) 2c3b "h s for the Saturated Vapor Line
        # Page 20, Eq 6
        return h4_s_part4(s)
    else:
        fn_return_value = math.nan

    return fn_return_value


@njit(cache=True)
def p4_s(s):
    hsat = h4_s(s)
    if - 0.0001545495919 < s <= 3.77828134:
        fn_return_value = p1_hs(hsat, s)
    elif 3.77828134 < s <= 5.210887663:
        fn_return_value = p3_hs(hsat, s)
    elif 5.210887663 < s < 9.155759395:
        fn_return_value = p2_hs(hsat, s)
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def h4L_p(p):
    Ts = 0
    ps = 0
    if 0.000611657 < p < 22.06395:
        Ts = T4_p(p)
        if p < 16.529:
            fn_return_value = h1_pT(p, Ts)
        else:
            # Iterate to find the the backward solution of p3sat_h
            Low_Bound = 1670.858218
            High_Bound = 2087.23500164864
            iter_count = 0
            hs = 0
            while abs(p - ps) > 0.00001 and iter_count < 1000:
                hs = (Low_Bound + High_Bound) / 2
                ps = p3sat_h(hs)
                if ps > p:
                    High_Bound = hs
                else:
                    Low_Bound = hs
                iter_count += 1

            if iter_count >= 1000:
                return math.nan

            fn_return_value = hs
    else:
        fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def h4V_p(p):
    if 0.000611657 < p < 22.06395:
        Ts = T4_p(p)
        if p < 16.529:
            fn_return_value = h2_pT(p, Ts)
        else:
            # Iterate to find the the backward solution of p3sat_h
            Low_Bound = 2087.23500164864
            High_Bound = 2563.592004 + 5
            ps = 0
            iter_count = 0
            hs = 0
            while abs(p - ps) > 0.000001 and iter_count < 1000:
                hs = (Low_Bound + High_Bound) / 2
                ps = p3sat_h(hs)
                if ps < p:
                    High_Bound = hs
                else:
                    Low_Bound = hs

                iter_count += 1

            if iter_count >= 1000:
                return math.nan

            fn_return_value = hs
    else:
        fn_return_value = math.nan

    return fn_return_value


@njit(cache=True)
def x4_ph(p, h):
    # Calculate vapour fraction from hL and hV for given p
    h4v = h4V_p(p)
    h4l = h4L_p(p)
    if h > h4v:
        fn_return_value = 1
    elif h < h4l:
        fn_return_value = 0
    else:
        fn_return_value = (h - h4l) / (h4v - h4l)
    return fn_return_value


@njit(cache=True)
def x4_ps(p, s):
    if p < 16.529:
        ssV = s2_pT(p, T4_p(p))
        ssL = s1_pT(p, T4_p(p))
    else:
        ssV = s3_rhoT(1 / (v3_ph(p, h4V_p(p))), T4_p(p))
        ssL = s3_rhoT(1 / (v3_ph(p, h4L_p(p))), T4_p(p))
    if s < ssL:
        fn_return_value = 0
    elif s > ssV:
        fn_return_value = 1
    else:
        fn_return_value = (s - ssL) / (ssV - ssL)
    return fn_return_value


@njit(cache=True)
def T4_hs(h, s):
    # Supplementary Release on Backward Equations ( ) , p h s for Region 3,
    # Chapter 5.3 page 30.
    # The if 97 function is only valid for part of region4. Use iteration outsida.
    T4_hs_Ii = np.array(
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0,
         6.0, 6.0, 6.0, 8.0, 10.0, 10.0, 12.0, 14.0, 14.0, 16.0, 16.0, 18.0, 18.0, 18.0, 20.0, 28.0])
    T4_hs_Ji = np.array([0.0, 3.0, 12.0, 0.0, 1.0, 2.0, 5.0, 0.0, 5.0, 8.0, 0.0, 2.0, 3.0, 4.0, 0.0, 1.0, 1.0, 2.0, 4.0,
                         16.0, 6.0, 8.0, 22.0, 1.0, 20.0, 36.0, 24.0, 1.0, 28.0, 12.0, 32.0, 14.0,
                         22.0, 36.0, 24.0, 36.0])
    T4_hs_ni = np.array([0.179882673606601, - 0.267507455199603, 1.162767226126, 0.147545428713616, - 0.512871635973248,
                         0.421333567697984, 0.56374952218987, 0.429274443819153, - 3.3570455214214, 10.8890916499278,
                         - 0.248483390456012, 0.30415322190639, - 0.494819763939905, 1.07551674933261,
                         7.33888415457688E-02,
                         1.40170545411085E-02, - 0.106110975998808, 1.68324361811875E-02, 1.25028363714877,
                         1013.16840309509,
                         - 1.51791558000712, 52.4277865990866, 23049.5545563912, 2.49459806365456E-02, 2107964.67412137,
                         366836848.613065, - 144814105.365163, - 1.7927637300359E-03, 4899556021.00459,
                         471.262212070518,
                         - 82929439019.8652, - 1715.45662263191, 3557776.82973575, 586062760258.436, - 12988763.5078195,
                         31724744937.1057])

    if 5.210887825 < s < 9.15546555571324:
        sigma = s / 9.2
        eta = h / 2800
        teta = 0
        for i in range(36):
            teta = teta + T4_hs_ni[i] * (eta - 0.119) ** T4_hs_Ii[i] * (sigma - 1.07) ** T4_hs_Ji[i]
        fn_return_value = teta * 550
    else:
        # Function psat_h
        PL = 0
        if - 0.0001545495919 < s <= 3.77828134:
            Low_Bound = 0.000611
            High_Bound = 165.291642526045
            iter_count = 0
            hL = 0
            while abs(hL - h) > 0.00001 and abs(High_Bound - Low_Bound) > 0.0001 and iter_count < 1000:
                PL = (Low_Bound + High_Bound) / 2
                Ts = T4_p(PL)
                hL = h1_pT(PL, Ts)
                if hL > h:
                    High_Bound = PL
                else:
                    Low_Bound = PL

                iter_count += 1

            if iter_count >= 1000:
                return math.nan

        elif 3.77828134 < s <= 4.41202148223476:
            PL = p3sat_h(h)
        elif 4.41202148223476 < s <= 5.210887663:
            PL = p3sat_h(h)

        Low_Bound = 0.000611
        High_Bound = PL
        iter_count = 0
        p = 0
        ss = 0
        while abs(s - ss) > 0.000001 and abs(High_Bound - Low_Bound) > 0.0000001 and iter_count < 1000:
            p = (Low_Bound + High_Bound) / 2
            # Calculate s4_ph
            Ts = T4_p(p)
            xs = x4_ph(p, h)
            if p < 16.529:
                s4V = s2_pT(p, Ts)
                s4L = s1_pT(p, Ts)
            else:
                v4V = v3_ph(p, h4V_p(p))
                s4V = s3_rhoT(1 / v4V, Ts)
                v4L = v3_ph(p, h4L_p(p))
                s4L = s3_rhoT(1 / v4L, Ts)
            ss = (xs * s4V + (1 - xs) * s4L)
            if ss < s:
                High_Bound = p
            else:
                Low_Bound = p

            iter_count += 1

        if iter_count >= 1000:
            return math.nan

        fn_return_value = T4_p(p)

    return fn_return_value


@njit(cache=True)
def h5_pT(p, T):
    R = 0.461526
    # Release on the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam,
    # September 1997
    # Basic Equation for Region 5
    # Eq 32,33, Page 36, Tables 37-41
    reg5_Ji0 = np.array([0.0, 1.0, - 3.0, -2.0, -1.0, 2.0])
    reg5_ni0 = np.array([-13.179983674201, 6.8540841634434, -0.024805148933466, 0.36901534980333, -3.1161318213925,
                         -0.32961626538917])

    # UPDATED FROM IAPWS R7-97(2012)
    # REGION 5 COEFFICIENTS UPDATED
    reg5_Iir = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0])
    reg5_Jir = np.array([1.0, 2.0, 3.0, 3.0, 9.0, 7.0])
    reg5_nir = np.array(
        [1.5736404855259E-03, 9.0153761673944E-04, -5.0270077677648E-03, 2.2440037409485E-06, -4.1163275453471E-06,
         3.7919454822955E-08])

    tau = 1000 / T
    gamma0_tau = 0
    for i in range(6):
        gamma0_tau = gamma0_tau + reg5_ni0[i] * reg5_Ji0[i] * tau ** (reg5_Ji0[i] - 1)

    gammar_tau = 0
    # UPDATED FROM IAPWS R7-97(2012)
    # INDEX GOES UP TO 5
    for i in range(6):
        gammar_tau = gammar_tau + reg5_nir[i] * p ** reg5_Iir[i] * reg5_Jir[i] * tau ** (reg5_Jir[i] - 1)

    fn_return_value = R * T * tau * (gamma0_tau + gammar_tau)
    return fn_return_value


@njit(cache=True)
def v5_pT(p, T):
    R = 0.461526
    # Release on the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam,
    # September 1997
    # Basic Equation for Region 5
    # Eq 32,33, Page 36, Tables 37-41
    reg5_Ji0 = np.array([0.0, 1.0, - 3.0, -2.0, -1.0, 2.0])
    reg5_ni0 = np.array([-13.179983674201, 6.8540841634434, -0.024805148933466, 0.36901534980333, -3.1161318213925,
                         -0.32961626538917])

    # UPDATED FROM IAPWS R7-97(2012)
    # REGION 5 COEFFICIENTS UPDATED
    reg5_Iir = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0])
    reg5_Jir = np.array([1.0, 2.0, 3.0, 3.0, 9.0, 7.0])
    reg5_nir = np.array(
        [1.5736404855259E-03, 9.0153761673944E-04, -5.0270077677648E-03, 2.2440037409485E-06, -4.1163275453471E-06,
         3.7919454822955E-08])

    tau = 1000 / T
    gamma0_pi = 1 / p
    gammar_pi = 0
    # UPDATED FROM IAPWS R7-97(2012)
    # INDEX GOES UP TO 5
    for i in range(6):
        gammar_pi = gammar_pi + reg5_nir[i] * reg5_Iir[i] * p ** (reg5_Iir[i] - 1) * tau ** reg5_Jir[i]
    fn_return_value = R * T / p * p * (gamma0_pi + gammar_pi) / 1000
    return fn_return_value


@njit(cache=True)
def u5_pT(p, T):
    R = 0.461526
    # Release on the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam,
    # September 1997
    # Basic Equation for Region 5
    # Eq 32,33, Page 36, Tables 37-41
    reg5_Ji0 = np.array([0.0, 1.0, - 3.0, -2.0, -1.0, 2.0])
    reg5_ni0 = np.array([-13.179983674201, 6.8540841634434, -0.024805148933466, 0.36901534980333, -3.1161318213925,
                         -0.32961626538917])

    # UPDATED FROM IAPWS R7-97(2012)
    # REGION 5 COEFFICIENTS UPDATED
    reg5_Iir = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0])
    reg5_Jir = np.array([1.0, 2.0, 3.0, 3.0, 9.0, 7.0])
    reg5_nir = np.array(
        [1.5736404855259E-03, 9.0153761673944E-04, -5.0270077677648E-03, 2.2440037409485E-06, -4.1163275453471E-06,
         3.7919454822955E-08])

    tau = 1000 / T
    gamma0_pi = 1 / p
    gamma0_tau = 0
    for i in range(6):
        gamma0_tau = gamma0_tau + reg5_ni0[i] * reg5_Ji0[i] * tau ** (reg5_Ji0[i] - 1)
    gammar_pi = 0
    gammar_tau = 0

    # UPDATED FROM IAPWS R7-97(2012)
    # INDEX GOES UP TO 5
    for i in range(6):
        gammar_pi = gammar_pi + reg5_nir[i] * reg5_Iir[i] * p ** (reg5_Iir[i] - 1) * tau ** reg5_Jir[i]
        gammar_tau = gammar_tau + reg5_nir[i] * p ** reg5_Iir[i] * reg5_Jir[i] * tau ** (reg5_Jir[i] - 1)

    fn_return_value = R * T * (tau * (gamma0_tau + gammar_tau) - p * (gamma0_pi + gammar_pi))
    return fn_return_value


@njit(cache=True)
def Cp5_pT(p, T):
    R = 0.461526
    # Release on the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam,
    # September 1997
    # Basic Equation for Region 5
    # Eq 32,33, Page 36, Tables 37-41
    reg5_Ji0 = np.array([0.0, 1.0, - 3.0, -2.0, -1.0, 2.0])
    reg5_ni0 = np.array([-13.179983674201, 6.8540841634434, -0.024805148933466, 0.36901534980333, -3.1161318213925,
                         -0.32961626538917])

    # UPDATED FROM IAPWS R7-97(2012)
    # REGION 5 COEFFICIENTS UPDATED
    reg5_Iir = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0])
    reg5_Jir = np.array([1.0, 2.0, 3.0, 3.0, 9.0, 7.0])
    reg5_nir = np.array(
        [1.5736404855259E-03, 9.0153761673944E-04, -5.0270077677648E-03, 2.2440037409485E-06, -4.1163275453471E-06,
         3.7919454822955E-08])

    tau = 1000 / T
    gamma0_tautau = 0
    for i in range(6):
        gamma0_tautau = gamma0_tautau + reg5_ni0[i] * reg5_Ji0[i] * (reg5_Ji0[i] - 1) * tau ** (reg5_Ji0[i] - 2)

    gammar_tautau = 0
    for i in range(6):
        gammar_tautau = gammar_tautau + reg5_nir[i] * p ** reg5_Iir[i] * reg5_Jir[i] * (reg5_Jir[i] - 1) * tau ** (
            reg5_Jir[i] - 2)

    fn_return_value = - R * tau ** 2 * (gamma0_tautau + gammar_tautau)
    return fn_return_value


@njit(cache=True)
def s5_pT(p, T):
    R = 0.461526
    # Release on the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam,
    # September 1997
    # Basic Equation for Region 5
    # Eq 32,33, Page 36, Tables 37-41
    reg5_Ji0 = np.array([0.0, 1.0, - 3.0, -2.0, -1.0, 2.0])
    reg5_ni0 = np.array([-13.179983674201, 6.8540841634434, -0.024805148933466, 0.36901534980333, -3.1161318213925,
                         -0.32961626538917])

    # UPDATED FROM IAPWS R7-97(2012)
    # REGION 5 COEFFICIENTS UPDATED
    reg5_Iir = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0])
    reg5_Jir = np.array([1.0, 2.0, 3.0, 3.0, 9.0, 7.0])
    reg5_nir = np.array(
        [1.5736404855259E-03, 9.0153761673944E-04, -5.0270077677648E-03, 2.2440037409485E-06, -4.1163275453471E-06,
         3.7919454822955E-08])

    tau = 1000 / T
    gamma0 = math.log(p)
    gamma0_tau = 0
    for i in range(6):
        gamma0_tau = gamma0_tau + reg5_ni0[i] * reg5_Ji0[i] * tau ** (reg5_Ji0[i] - 1)
        gamma0 = gamma0 + reg5_ni0[i] * tau ** reg5_Ji0[i]

    gammar = 0
    gammar_tau = 0
    for i in range(6):
        gammar = gammar + reg5_nir[i] * p ** reg5_Iir[i] * tau ** reg5_Jir[i]
        gammar_tau = gammar_tau + reg5_nir[i] * p ** reg5_Iir[i] * reg5_Jir[i] * tau ** (reg5_Jir[i] - 1)

    fn_return_value = R * (tau * (gamma0_tau + gammar_tau) - (gamma0 + gammar))
    return fn_return_value


@njit(cache=True)
def Cv5_pT(p, T):
    R = 0.461526
    # Release on the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam,
    # September 1997
    # Basic Equation for Region 5
    # Eq 32,33, Page 36, Tables 37-41
    reg5_Ji0 = np.array([0.0, 1.0, - 3.0, -2.0, -1.0, 2.0])
    reg5_ni0 = np.array([-13.179983674201, 6.8540841634434, -0.024805148933466, 0.36901534980333, -3.1161318213925,
                         -0.32961626538917])

    # UPDATED FROM IAPWS R7-97(2012)
    # REGION 5 COEFFICIENTS UPDATED
    reg5_Iir = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0])
    reg5_Jir = np.array([1.0, 2.0, 3.0, 3.0, 9.0, 7.0])
    reg5_nir = np.array(
        [1.5736404855259E-03, 9.0153761673944E-04, -5.0270077677648E-03, 2.2440037409485E-06, -4.1163275453471E-06,
         3.7919454822955E-08])

    tau = 1000 / T
    gamma0_tautau = 0
    for i in range(6):
        gamma0_tautau = gamma0_tautau + reg5_ni0[i] * (reg5_Ji0[i] - 1) * reg5_Ji0[i] * tau ** (reg5_Ji0[i] - 2)

    gammar_pi = 0
    gammar_pitau = 0
    gammar_pipi = 0
    gammar_tautau = 0
    for i in range(6):
        gammar_pi = gammar_pi + reg5_nir[i] * reg5_Iir[i] * p ** (reg5_Iir[i] - 1) * tau ** reg5_Jir[i]
        gammar_pitau = gammar_pitau + reg5_nir[i] * reg5_Iir[i] * p ** (reg5_Iir[i] - 1) * reg5_Jir[i] * tau ** (
            reg5_Jir[i] - 1)
        gammar_pipi = gammar_pipi + reg5_nir[i] * reg5_Iir[i] * (reg5_Iir[i] - 1) * p ** (reg5_Iir[i] - 2) * tau ** \
                      reg5_Jir[i]
        gammar_tautau = gammar_tautau + reg5_nir[i] * p ** reg5_Iir[i] * reg5_Jir[i] * (reg5_Jir[i] - 1) * tau ** (
            reg5_Jir[i] - 2)

    fn_return_value = R * (
        - (tau ** 2 * (gamma0_tautau + gammar_tautau)) - (1 + p * gammar_pi - tau * p * gammar_pitau) ** 2 / (
        1 - p ** 2 * gammar_pipi))
    return fn_return_value


@njit(cache=True)
def w5_pT(p, T):
    R = 0.461526
    # Release on the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam,
    # September 1997
    # Basic Equation for Region 5
    # Eq 32,33, Page 36, Tables 37-41
    reg5_Ji0 = np.array([0.0, 1.0, - 3.0, -2.0, -1.0, 2.0])
    reg5_ni0 = np.array([-13.179983674201, 6.8540841634434, -0.024805148933466, 0.36901534980333, -3.1161318213925,
                         -0.32961626538917])

    # UPDATED FROM IAPWS R7-97(2012)
    # REGION 5 COEFFICIENTS UPDATED
    reg5_Iir = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0])
    reg5_Jir = np.array([1.0, 2.0, 3.0, 3.0, 9.0, 7.0])
    reg5_nir = np.array(
        [1.5736404855259E-03, 9.0153761673944E-04, -5.0270077677648E-03, 2.2440037409485E-06, -4.1163275453471E-06,
         3.7919454822955E-08])

    tau = 1000 / T
    gamma0_tautau = 0
    for i in range(6):
        gamma0_tautau = gamma0_tautau + reg5_ni0[i] * (reg5_Ji0[i] - 1) * reg5_Ji0[i] * tau ** (reg5_Ji0[i] - 2)
    gammar_pi = 0
    gammar_pitau = 0
    gammar_pipi = 0
    gammar_tautau = 0

    for i in range(6):
        gammar_pi = gammar_pi + reg5_nir[i] * reg5_Iir[i] * p ** (reg5_Iir[i] - 1) * tau ** reg5_Jir[i]
        gammar_pitau = gammar_pitau + reg5_nir[i] * reg5_Iir[i] * p ** (reg5_Iir[i] - 1) * reg5_Jir[i] * tau ** (
            reg5_Jir[i] - 1)
        gammar_pipi = gammar_pipi + reg5_nir[i] * reg5_Iir[i] * (reg5_Iir[i] - 1) * p ** (reg5_Iir[i] - 2) * tau ** \
                      reg5_Jir[i]
        gammar_tautau = gammar_tautau + reg5_nir[i] * p ** reg5_Iir[i] * reg5_Jir[i] * (reg5_Jir[i] - 1) * tau ** (
            reg5_Jir[i] - 2)

    fn_return_value = (1000 * R * T * (1 + 2 * p * gammar_pi + p ** 2 * gammar_pi ** 2) / (
        (1 - p ** 2 * gammar_pipi) + (1 + p * gammar_pi - tau * p * gammar_pitau) ** 2 / (
        tau ** 2 * (gamma0_tautau + gammar_tautau)))) ** 0.5
    return fn_return_value


@njit(cache=True)
def T5_ph(p, h):
    hs = 0
    Ts = 0
    # Solve with half interval method
    Low_Bound = 1073.15
    High_Bound = 2273.15
    iter_count = 0
    while abs(h - hs) > 0.00001 and iter_count < 1000:
        Ts = (Low_Bound + High_Bound) / 2
        hs = h5_pT(p, Ts)
        if hs > h:
            High_Bound = Ts
        else:
            Low_Bound = Ts

        iter_count += 1

    if iter_count >= 1000:
        return math.nan

    fn_return_value = Ts
    return fn_return_value


@njit(cache=True)
def T5_ps(p, s):
    Ts = 0
    ss = 0
    # Solve with half interval method
    Low_Bound = 1073.15
    High_Bound = 2273.15
    iter_count = 0
    while abs(s - ss) > 0.00001 and iter_count < 1000:
        Ts = (Low_Bound + High_Bound) / 2
        ss = s5_pT(p, Ts)
        if ss > s:
            High_Bound = Ts
        else:
            Low_Bound = Ts

        iter_count += 1

    if iter_count >= 1000:
        return math.nan

    fn_return_value = Ts
    return fn_return_value


@njit(cache=True)
def T5_prho(p, rho):
    Ts = 0
    rhos = 0
    # Solve by iteration. Observe that fo low temperatures this equation has 2 solutions.
    # Solve with half interval method
    Low_Bound = 1073.15
    High_Bound = 2073.15
    iter_count = 0
    while abs(rho - rhos) > 0.000001 and iter_count < 1000:
        Ts = (Low_Bound + High_Bound) / 2
        rhos = 1 / v2_pT(p, Ts)
        if rhos < rho:
            High_Bound = Ts
        else:
            Low_Bound = Ts
        iter_count += 1

    if iter_count >= 1000:
        return math.nan

    fn_return_value = Ts
    return fn_return_value


@njit(cache=True)
def region_pT(p, T):
    ps = 0
    if 1073.15 < T < 2273.15 and 10 > p > 0.000611:
        fn_return_value = 5
    elif 1073.15 >= T > 273.15 and 100 >= p > 0.000611:
        if T > 623.15:
            if p > B23p_T(T):
                fn_return_value = 3
                if T < 647.096:
                    ps = p4_T(T)
                    if abs(p - ps) < 0.00001:
                        fn_return_value = 4
            else:
                fn_return_value = 2
        else:
            ps = p4_T(T)
            if abs(p - ps) < 0.00001:
                fn_return_value = 4
            elif p > ps:
                fn_return_value = 1
            else:
                fn_return_value = 2
    else:
        fn_return_value = 0

    return fn_return_value


@njit(cache=True)
def region_ph(p, h):
    # Check if outside pressure limits
    if p < 0.000611657 or p > 100:
        fn_return_value = 0
        return fn_return_value
    # Check if outside low h.
    if h < 0.963 * p + 2.2:
        if h < h1_pT(p, 273.15):
            fn_return_value = 0
            return fn_return_value
    if p < 16.5292:
        # Check Region 1
        Ts = T4_p(p)
        hL = 109.6635 * math.log(p) + 40.3481 * p + 734.58
        if abs(h - hL) < 100:
            hL = h1_pT(p, Ts)
        if h <= hL:
            fn_return_value = 1
            return fn_return_value
        # Check Region 4
        hV = 45.1768 * math.log(p) - 20.158 * p + 2804.4
        if abs(h - hV) < 50:
            hV = h2_pT(p, Ts)
        if h < hV:
            fn_return_value = 4
            return fn_return_value
        # Check upper limit of region 2 Quick Test
        if h < 4000:
            fn_return_value = 2
            return fn_return_value
        # Check region 2 (Real value)
        h_45 = h2_pT(p, 1073.15)
        if h <= h_45:
            fn_return_value = 2
            return fn_return_value
        # Check region 5
        if p > 10:
            fn_return_value = 0
            return fn_return_value
        h_5u = h5_pT(p, 2273.15)
        if h < h_5u:
            fn_return_value = 5
            return fn_return_value
        fn_return_value = 0
        return fn_return_value
    else:
        # Check if in region1
        if h < h1_pT(p, 623.15):
            fn_return_value = 1
            return fn_return_value
        # Check if in region 3 or 4 (Bellow Reg 2)
        if h < h2_pT(p, B23T_p(p)):
            # Region 3 or 4
            if p > p3sat_h(h):
                fn_return_value = 3
                return fn_return_value
            else:
                fn_return_value = 4
                return fn_return_value
        # Check if region 2
        if h < h2_pT(p, 1073.15):
            fn_return_value = 2
            return fn_return_value
    fn_return_value = 0
    return fn_return_value


@njit(cache=True)
def region_ps(p, s):
    ss = 0
    if p < 0.000611657 or p > 100 or s < 0 or s > s5_pT(p, 2273.15):
        fn_return_value = 0
        return fn_return_value

    # Check region 5
    if s > s2_pT(p, 1073.15):
        if p <= 10:
            fn_return_value = 5
            return fn_return_value
        else:
            fn_return_value = 0
            return fn_return_value

    # Check region 2
    if p > 16.529:
        ss = s2_pT(p, B23T_p(p))
    else:
        ss = s2_pT(p, T4_p(p))
    if s > ss:
        fn_return_value = 2
        return fn_return_value

    # Check region 3
    ss = s1_pT(p, 623.15)
    if p > 16.529 and s > ss:
        if p > p3sat_s(s):
            fn_return_value = 3
            return fn_return_value
        else:
            fn_return_value = 4
            return fn_return_value

    # Check region 4 (Not inside region 3)
    if p < 16.529 and s > s1_pT(p, T4_p(p)):
        fn_return_value = 4
        return fn_return_value

    # Check region 1
    if p > 0.000611657 and s > s1_pT(p, 273.15):
        fn_return_value = 1
        return fn_return_value

    fn_return_value = 1
    return fn_return_value


@njit(cache=True)
def Region_hs(h, s):
    if s < - 0.0001545495919:
        fn_return_value = 0
        return fn_return_value
    # Check linear adaption to p=0.000611. If bellow region 4.
    hMin = (((- 0.0415878 - 2500.89262) / (- 0.00015455 - 9.155759)) * s)
    if s < 9.155759395 and h < hMin:
        fn_return_value = 0
        return fn_return_value

    # ******Kolla 1 eller 4. (+liten bit över B13)
    if - 0.0001545495919 <= s <= 3.77828134:
        if h < h4_s(s):
            fn_return_value = 4
            return fn_return_value
        elif s < 3.397782955:
            TMax = T1_ps(100, s)
            hMax = h1_pT(100, TMax)
            if h < hMax:
                fn_return_value = 1
                return fn_return_value
            else:
                fn_return_value = 0
                return fn_return_value
        else:
            hB = hB13_s(s)
            if h < hB:
                fn_return_value = 1
                return fn_return_value
            TMax = T3_ps(100, s)
            vmax = v3_ps(100, s)
            hMax = h3_rhoT(1 / vmax, TMax)
            if h < hMax:
                fn_return_value = 3
                return fn_return_value
            else:
                fn_return_value = 0
                return fn_return_value

    # ******Kolla region 2 eller 4. (Övre delen av område b23-> max)
    if 5.260578707 <= s <= 11.9212156897728:
        if s > 9.155759395:
            Tmin = T2_ps(0.000611, s)
            hMin = h2_pT(0.000611, Tmin)
            # Function adapted to h(1073.15,s)
            hMax = - 0.07554022 * s ** 4 + 3.341571 * s ** 3 - 55.42151 * s ** 2 + 408.515 * s + 3031.338
            if hMin < h < hMax:
                fn_return_value = 2
                return fn_return_value
            else:
                fn_return_value = 0
                return fn_return_value
        hV = h4_s(s)
        if h < hV:
            fn_return_value = 4
            return fn_return_value
        if s < 6.04048367171238:
            TMax = T2_ps(100, s)
            hMax = h2_pT(100, TMax)
        else:
            # Function adapted to h(1073.15,s)
            hMax = - 2.988734 * s ** 4 + 121.4015 * s ** 3 - 1805.15 * s ** 2 + 11720.16 * s - 23998.33
        if h < hMax:
            fn_return_value = 2
            return fn_return_value
        else:
            fn_return_value = 0
            return fn_return_value

    # Kolla region 3 eller 4. Under kritiska punkten.
    if 3.77828134 <= s <= 4.41202148223476:
        hL = h4_s(s)
        if h < hL:
            fn_return_value = 4
            return fn_return_value
        TMax = T3_ps(100, s)
        vmax = v3_ps(100, s)
        hMax = h3_rhoT(1 / vmax, TMax)
        if h < hMax:
            fn_return_value = 3
            return fn_return_value
        else:
            fn_return_value = 0
            return fn_return_value

    # Kolla region 3 eller 4 från kritiska punkten till övre delen av b23
    if 4.41202148223476 <= s <= 5.260578707:
        hV = h4_s(s)
        if h < hV:
            fn_return_value = 4
            return fn_return_value
        # Kolla om vi är under b23 giltighetsområde.
        if s <= 5.048096828:
            TMax = T3_ps(100, s)
            vmax = v3_ps(100, s)
            hMax = h3_rhoT(1 / vmax, TMax)
            if h < hMax:
                fn_return_value = 3
                return fn_return_value
            else:
                fn_return_value = 0
                return fn_return_value
        else:
            if h > 2812.942061:
                if s > 5.09796573397125:
                    TMax = T2_ps(100, s)
                    hMax = h2_pT(100, TMax)
                    if h < hMax:
                        fn_return_value = 2
                        return fn_return_value
                    else:
                        fn_return_value = 0
                        return fn_return_value
                else:
                    fn_return_value = 0
                    return fn_return_value
            if h < 2563.592004:
                fn_return_value = 3
                return fn_return_value
            # Vi är inom b23 området i både s och h led.
            if p2_hs(h, s) > B23p_T(TB23_hs(h, s)):
                fn_return_value = 3
                return fn_return_value
            else:
                fn_return_value = 2
                return fn_return_value
    fn_return_value = math.nan
    return fn_return_value


@njit(cache=True)
def Region_prho(p, rho):
    v = 1 / rho
    if p < 0.000611657 or p > 100:
        fn_return_value = 0
        return fn_return_value
    if p < 16.5292:
        if v < v1_pT(p, 273.15):
            fn_return_value = 0
            return fn_return_value
        if v <= v1_pT(p, T4_p(p)):
            fn_return_value = 1
            return fn_return_value
        if v < v2_pT(p, T4_p(p)):
            fn_return_value = 4
            return fn_return_value
        if v <= v2_pT(p, 1073.15):
            fn_return_value = 2
            return fn_return_value
        if p > 10:
            fn_return_value = 0
            return fn_return_value
        if v <= v5_pT(p, 2073.15):
            fn_return_value = 5
            return fn_return_value
    else:
        if v < v1_pT(p, 273.15):
            fn_return_value = 0
            return fn_return_value
        if v < v1_pT(p, 623.15):
            fn_return_value = 1
            return fn_return_value
        # Check if in region 3 or 4 (Bellow Reg 2)
        if v < v2_pT(p, B23T_p(p)):
            # Region 3 or 4
            if p > 22.064:
                fn_return_value = 3
                return fn_return_value
            if v < v3_ph(p, h4L_p(p)) or v > v3_ph(p, h4V_p(p)):
                fn_return_value = 3
                return fn_return_value
            else:
                fn_return_value = 4
                return fn_return_value
        # Check if region 2
        if v < v2_pT(p, 1073.15):
            fn_return_value = 2
            return fn_return_value

    fn_return_value = 0
    return fn_return_value


@njit(cache=True)
def B23p_T(T):
    # Release on the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam
    # 1997
    # Section 4 Auxiliary Equation for the Boundary between Regions 2 and 3
    # Eq 5, Page 5
    fn_return_value = 348.05185628969 - 1.1671859879975 * T + 1.0192970039326E-03 * T ** 2
    return fn_return_value


@njit(cache=True)
def B23T_p(p):
    # Release on the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam
    # 1997
    # Section 4 Auxiliary Equation for the Boundary between Regions 2 and 3
    # Eq 6, Page 6
    fn_return_value = 572.54459862746 + ((p - 13.91883977887) / 1.0192970039326E-03) ** 0.5
    return fn_return_value


@njit(cache=True)
def p3sat_h(h):
    # Revised Supplementary Release on Backward Equations for the Functions T(p,h), v(p,h) and T(p,s), v(p,s) for
    # Region 3 of the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam
    # 2004
    # Section 4 Boundary Equations psat(h) and psat(s) for the Saturation Lines of Region 3
    # Se pictures Page 17, Eq 10, Table 17, Page 18
    p3sat_h_Ii = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 5.0, 7.0, 8.0, 14.0, 20.0, 22.0, 24.0, 28.0, 36.0])
    p3sat_h_Ji = np.array([0.0, 1.0, 3.0, 4.0, 36.0, 3.0, 0.0, 24.0, 16.0, 16.0, 3.0, 18.0, 8.0, 24.0])
    p3sat_h_ni = np.array(
        [0.600073641753024, - 9.36203654849857, 24.6590798594147, - 107.014222858224, -91582131580576.8,
         -8623.32011700662, - 23.5837344740032, 2.52304969384128E+17, - 3.89718771997719E+18,
         -3.33775713645296E+22, 35649946963.6328, - 1.48547544720641E+26, 3.30611514838798E+18,
         8.13641294467829E+37])
    h = h / 2600
    ps = 0
    for i in range(14):
        ps = ps + p3sat_h_ni[i] * (h - 1.02) ** p3sat_h_Ii[i] * (h - 0.608) ** p3sat_h_Ji[i]
    fn_return_value = ps * 22
    return fn_return_value


@njit(cache=True)
def p3sat_s(s):
    p3sat_s_Ii = np.array([0.0, 1.0, 1.0, 4.0, 12.0, 12.0, 16.0, 24.0, 28.0, 32.0])
    p3sat_s_Ji = np.array([0.0, 1.0, 32.0, 7.0, 4.0, 14.0, 36.0, 10.0, 0.0, 18.0])
    p3sat_s_ni = np.array(
        [0.639767553612785, - 12.9727445396014, - 2.24595125848403E+15, 1774667.41801846, 7170793495.71538,
         - 3.78829107169011E+17, - 9.55586736431328E+34, 1.87269814676188E+23, 119254746466.473,
         1.10649277244882E+36])

    sigma = s / 5.2
    p = 0
    for i in range(10):
        p = p + p3sat_s_ni[i] * (sigma - 1.03) ** p3sat_s_Ii[i] * (sigma - 0.699) ** p3sat_s_Ji[i]

    fn_return_value = p * 22
    return fn_return_value


@njit(cache=True)
def hB13_s(s):
    # Supplementary Release on Backward Equations ( ) , p h s for Region 3,
    # Chapter 4.5 page 23.
    hB13_s_Ii = np.array([0.0, 1.0, 1.0, 3.0, 5.0, 6.0])
    hB13_s_Ji = np.array([0.0, - 2.0, 2.0, - 12.0, - 4.0, - 3.0])
    hB13_s_ni = np.array([0.913965547600543, - 4.30944856041991E-05, 60.3235694765419, 1.17518273082168E-18,
                          0.220000904781292, - 69.0815545851641])
    sigma = s / 3.8
    eta = 0
    for i in range(6):
        eta = eta + hB13_s_ni[i] * (sigma - 0.884) ** hB13_s_Ii[i] * (sigma - 0.864) ** hB13_s_Ji[i]
    fn_return_value = eta * 1700
    return fn_return_value


@njit(cache=True)
def TB23_hs(h, s):
    # Supplementary Release on Backward Equations ( ) , p h s for Region 3,
    # Chapter 4.6 page 25.
    TB23_hs_Ii = np.array(
        [- 12.0, - 10.0, - 8.0, - 4.0, - 3.0, - 2.0, - 2.0, - 2.0, - 2.0, 0.0, 1.0, 1.0, 1.0, 3.0, 3.0, 5.0, 6.0,
         6.0, 8.0, 8.0, 8.0, 12.0, 12.0, 14.0, 14.0])
    TB23_hs_Ji = np.array(
        [10.0, 8.0, 3.0, 4.0, 3.0, - 6.0, 2.0, 3.0, 4.0, 0.0, - 3.0, - 2.0, 10.0, - 2.0, - 1.0, - 5.0, - 6.0,
         - 3.0, - 8.0, - 2.0, - 1.0, - 12.0, - 1.0, - 12.0, 1.0])
    TB23_hs_ni = np.array(
        [6.2909626082981E-04, - 8.23453502583165E-04, 5.15446951519474E-08, - 1.17565945784945, 3.48519684726192,
         -5.07837382408313E-12, - 2.84637670005479, - 2.36092263939673, 6.01492324973779, 1.48039650824546,
         3.60075182221907E-04, - 1.26700045009952E-02, - 1221843.32521413, 0.149276502463272, 0.698733471798484,
         -2.52207040114321E-02, 1.47151930985213E-02, - 1.08618917681849, - 9.36875039816322E-04,
         81.9877897570217, - 182.041861521835, 2.61907376402688E-06, - 29162.6417025961, 1.40660774926165E-05,
         7832370.62349385])

    sigma = s / 5.3
    eta = h / 3000
    teta = 0
    for i in range(25):
        teta = teta + TB23_hs_ni[i] * (eta - 0.727) ** TB23_hs_Ii[i] * (sigma - 0.864) ** TB23_hs_Ji[i]
    fn_return_value = teta * 900
    return fn_return_value


@njit(cache=True)
def my_AllRegions_pT(p, T):
    my_pT_h0 = np.array([0.5132047, 0.3205656, 0.0, 0.0, - 0.7782567, 0.1885447])
    my_pT_h1 = np.array([0.2151778, 0.7317883, 1.241044, 1.476783, 0.0, 0.0])
    my_pT_h2 = np.array([- 0.2818107, - 1.070786, - 1.263184, 0.0, 0.0, 0.0])
    my_pT_h3 = np.array([0.1778064, 0.460504, 0.2340379, - 0.4924179, 0.0, 0.0])
    my_pT_h4 = np.array([- 0.0417661, 0.0, 0.0, 0.1600435, 0.0, 0.0])
    my_pT_h5 = np.array([0.0, - 0.01578386, 0.0, 0.0, 0.0, 0.0])
    my_pT_h6 = np.array([0.0, 0.0, 0.0, - 0.003629481, 0.0, 0.0])

    # Calcualte density.
    select_variable_0 = region_pT(p, T)
    if select_variable_0 == 1:
        rho = 1 / v1_pT(p, T)
    elif select_variable_0 == 2:
        rho = 1 / v2_pT(p, T)
    elif select_variable_0 == 3:
        rho = 1 / v3_ph(p, h3_pT(p, T))
    elif select_variable_0 == 4:
        rho = math.nan
    elif select_variable_0 == 5:
        rho = 1 / v5_pT(p, T)
    else:
        fn_return_value = math.nan
        return fn_return_value
    rhos = rho / 317.763
    Ts = T / 647.226
    ps = p / 22.115
    # Check valid area
    if T > 900 + 273.15 or (T > 600 + 273.15 and p > 300) or (T > 150 + 273.15 and p > 350) or p > 500:
        fn_return_value = math.nan
        return fn_return_value
    my0 = Ts ** 0.5 / (1 + 0.978197 / Ts + 0.579829 / (Ts ** 2) - 0.202354 / (Ts ** 3))
    sum_x = 0
    for i in range(6):
        sum_x = sum_x + my_pT_h0[i] * (1 / Ts - 1) ** i + my_pT_h1[i] * (1 / Ts - 1) ** i * (rhos - 1) ** 1 + my_pT_h2[
            i] * (
                    1 / Ts - 1) ** i * (rhos - 1) ** 2 + my_pT_h3[i] * (1 / Ts - 1) ** i * (rhos - 1) ** 3 + my_pT_h4[
                    i] * (
                    1 / Ts - 1) ** i * (rhos - 1) ** 4 + my_pT_h5[i] * (1 / Ts - 1) ** i * (rhos - 1) ** 5 + my_pT_h6[
                    i] * (
                    1 / Ts - 1) ** i * (rhos - 1) ** 6

    my1 = math.exp(rhos * sum_x)
    fn_return_value = my0 * my1 * 0.000055071
    return fn_return_value


@njit(cache=True)
def my_AllRegions_ph(p, h):
    # Calcualte density
    my_ph_h0 = np.array([0.5132047, 0.3205656, 0.0, 0.0, - 0.7782567, 0.1885447])
    my_ph_h1 = np.array([0.2151778, 0.7317883, 1.241044, 1.476783, 0.0, 0.0])
    my_ph_h2 = np.array([- 0.2818107, - 1.070786, - 1.263184, 0.0, 0.0, 0.0])
    my_ph_h3 = np.array([0.1778064, 0.460504, 0.2340379, - 0.4924179, 0.0, 0.0])
    my_ph_h4 = np.array([- 0.0417661, 0.0, 0.0, 0.1600435, 0.0, 0.0])
    my_ph_h5 = np.array([0.0, - 0.01578386, 0.0, 0.0, 0.0, 0.0])
    my_ph_h6 = np.array([0.0, 0.0, 0.0, - 0.003629481, 0.0, 0.0])

    select_variable_1 = region_ph(p, h)
    if select_variable_1 == 1:
        Ts = T1_ph(p, h)
        T = Ts
        rho = 1 / v1_pT(p, Ts)
    elif select_variable_1 == 2:
        Ts = T2_ph(p, h)
        T = Ts
        rho = 1 / v2_pT(p, Ts)
    elif select_variable_1 == 3:
        rho = 1 / v3_ph(p, h)
        T = T3_ph(p, h)
    elif select_variable_1 == 4:
        xs = x4_ph(p, h)
        if p < 16.529:
            v4V = v2_pT(p, T4_p(p))
            v4L = v1_pT(p, T4_p(p))
        else:
            v4V = v3_ph(p, h4V_p(p))
            v4L = v3_ph(p, h4L_p(p))
        rho = 1 / (xs * v4V + (1 - xs) * v4L)
        T = T4_p(p)
    elif select_variable_1 == 5:
        Ts = T5_ph(p, h)
        T = Ts
        rho = 1 / v5_pT(p, Ts)
    else:
        fn_return_value = math.nan
        return fn_return_value
    rhos = rho / 317.763
    Ts = T / 647.226
    ps = p / 22.115
    # Check valid area
    if T > 900 + 273.15 or (T > 600 + 273.15 and p > 300) or (T > 150 + 273.15 and p > 350) or p > 500:
        fn_return_value = math.nan
        return fn_return_value
    my0 = Ts ** 0.5 / (1 + 0.978197 / Ts + 0.579829 / (Ts ** 2) - 0.202354 / (Ts ** 3))
    sum_x = 0
    for i in range(6):
        sum_x = sum_x + my_ph_h0[i] * (1 / Ts - 1) ** i + my_ph_h1[i] * (1 / Ts - 1) ** i * (rhos - 1) ** 1 + my_ph_h2[
            i] * (
                    1 / Ts - 1) ** i * (rhos - 1) ** 2 + my_ph_h3[i] * (1 / Ts - 1) ** i * (rhos - 1) ** 3 + my_ph_h4[
                    i] * (
                    1 / Ts - 1) ** i * (rhos - 1) ** 4 + my_ph_h5[i] * (1 / Ts - 1) ** i * (rhos - 1) ** 5 + my_ph_h6[
                    i] * (
                    1 / Ts - 1) ** i * (rhos - 1) ** 6

    my1 = math.exp(rhos * sum_x)
    fn_return_value = my0 * my1 * 0.000055071
    return fn_return_value


@njit(cache=True)
def tc_ptrho(p, T, rho):
    # Revised release on the IAPS Formulation 1985 for the Thermal Conductivity of ordinary water
    # IAPWS September 1998
    # Page 8
    # ver2.6 Start corrected bug
    if T < 273.15:
        fn_return_value = math.nan
        return fn_return_value
    elif T < 500 + 273.15:
        if p > 100:
            fn_return_value = math.nan
            return fn_return_value
    elif T <= 650 + 273.15:
        if p > 70:
            fn_return_value = math.nan
            return fn_return_value
    elif T <= 800 + 273.15:
        if p > 40:
            fn_return_value = math.nan
            return fn_return_value
    # ver2.6 End corrected bug
    T = T / 647.26
    rho = rho / 317.7
    tc0 = T ** 0.5 * (0.0102811 + 0.0299621 * T + 0.0156146 * T ** 2 - 0.00422464 * T ** 3)
    tc1 = - 0.39707 + 0.400302 * rho + 1.06 * math.exp(- 0.171587 * (rho + 2.39219) ** 2)
    dT = abs(T - 1) + 0.00308976
    Q = 2 + 0.0822994 / dT ** (3 / 5)
    if T >= 1:
        s = 1 / dT
    else:
        s = 10.0932 / dT ** (3 / 5)
    tc2 = (0.0701309 / T ** 10 + 0.011852) * rho ** (9 / 5) * math.exp(
        0.642857 * (1 - rho ** (14 / 5))) + 0.00169937 * s * rho ** Q * math.exp(
        (Q / (1 + Q)) * (1 - rho ** (1 + Q))) - 1.02 * math.exp(- 4.11717 * T ** (3 / 2) - 6.17937 / rho ** 5)
    fn_return_value = tc0 + tc1 + tc2
    return fn_return_value


@njit(cache=True)
def Surface_Tension_T(T):
    tc = 647.096
    b = 0.2358
    bb = - 0.625
    my = 1.256
    # IAPWS Release on Surface Tension of Ordinary Water Substance,
    # September 1994
    if T < 0.01 or T > tc:
        fn_return_value = math.nan
        return fn_return_value

    tau = 1 - T / tc
    fn_return_value = b * tau ** my * (1 + bb * tau)
    return fn_return_value


def self_test(show=False, repeat=1):
    def dummy(*arg, **kwargs):
        pass

    if show:
        repeat = 1
    if repeat <= 1:
        repeat = 1
    elif repeat > 1:
        show = False

    echo = dummy
    if show:
        echo = print

    for i in range(repeat):
        echo('RESULTS SHOULD MATCH MOST RECENT IFC-2012 RELEASE, UPDATED FOR REGION 5')
        echo('METABLE EQNS ARE AVALIABLE FOR REGION 2, BUT MUST CALLED DIRECTLY')

        echo('REGION 1')
        echo('TABLE 5')
        echo(v1_pT(3, 300), h1_pT(3, 300), u1_pT(3, 300), s1_pT(3, 300), Cp1_pT(3, 300), w1_pT(3, 300))
        echo(v1_pT(80, 300), h1_pT(80, 300), u1_pT(80, 300), s1_pT(80, 300), Cp1_pT(80, 300), w1_pT(80, 300))
        echo(v1_pT(3, 500), h1_pT(3, 500), u1_pT(3, 500), s1_pT(3, 500), Cp1_pT(3, 500), w1_pT(3, 500))

        echo('TABLE 7')
        echo(T1_ph(3, 500))
        echo(T1_ph(80, 500))
        echo(T1_ph(80, 1500))

        echo('TABLE 9')
        echo(T1_ps(3, 0.5))
        echo(T1_ps(80, 0.5))
        echo(T1_ps(80, 3.0))

        echo('TABLE 3')
        echo(p1_hs(0.001, 0.0))
        echo(p1_hs(90, 0.0))
        echo(p1_hs(1500, 3.4))

        echo('REGION 2')
        echo('TABLE 15')
        echo(v2_pT(0.0035, 300), h2_pT(0.0035, 300), u2_pT(0.0035, 300), s2_pT(0.0035, 300), Cp2_pT(0.0035, 300),
             w2_pT(0.0035, 300))
        echo(v2_pT(0.0035, 700), h2_pT(0.0035, 700), u2_pT(0.0035, 700), s2_pT(0.0035, 700), Cp2_pT(0.0035, 700),
             w2_pT(0.0035, 700))
        echo(v2_pT(30, 700), h2_pT(30, 700), u2_pT(30, 700), s2_pT(30, 700), Cp2_pT(30, 700), w2_pT(30, 700))

        echo('TABLE 18-METASTABLE')
        echo(v2_meta_pT(1, 450), h2_meta_pT(1, 450), u2_meta_pT(1, 450), s2_meta_pT(1, 450), Cp2_meta_pT(1, 450),
             w2_meta_pT(1, 450))
        echo(v2_meta_pT(1, 440), h2_meta_pT(1, 440), u2_meta_pT(1, 440), s2_meta_pT(1, 440), Cp2_meta_pT(1, 440),
             w2_meta_pT(1, 440))
        echo(v2_meta_pT(1.5, 450), h2_meta_pT(1.5, 450), u2_meta_pT(1.5, 450), s2_meta_pT(1.5, 450),
             Cp2_meta_pT(1.5, 450), w2_meta_pT(1.5, 450))

        echo('TABLE 24')
        echo(T2_ph(0.001, 3000))
        echo(T2_ph(3.0, 3000))
        echo(T2_ph(3.0, 4000))
        echo(T2_ph(5.0, 3500))
        echo(T2_ph(5.0, 4000))
        echo(T2_ph(25.0, 3500))
        echo(T2_ph(40.0, 2700))
        echo(T2_ph(60.0, 2700))
        echo(T2_ph(60.0, 3200))

        echo('TABLE 29')
        echo(T2_ps(0.1, 7.5))
        echo(T2_ps(0.1, 8))
        echo(T2_ps(2.5, 8))
        echo(T2_ps(8.0, 6))
        echo(T2_ps(8.0, 7.5))
        echo(T2_ps(90.0, 6))
        echo(T2_ps(20.0, 5.75))
        echo(T2_ps(80.0, 5.25))
        echo(T2_ps(80.0, 5.75))

        echo('TABLE 3')
        echo(p2_hs(2800, 6.5))
        echo(p2_hs(2800, 9.5))
        echo(p2_hs(4100, 9.5))
        echo(p2_hs(2800, 6))
        echo(p2_hs(3600, 6))
        echo(p2_hs(3600, 7))
        echo(p2_hs(2800, 5.1))
        echo(p2_hs(2800, 5.8))
        echo(p2_hs(3400, 5.8))

        echo('REGION 3')
        echo('TABLE 33')
        echo(p3_rhoT(500, 650), h3_rhoT(500, 650), u3_rhoT(500, 650), s3_rhoT(500, 650), Cp3_rhoT(500, 650),
             w3_rhoT(500, 650))
        echo(p3_rhoT(200, 650), h3_rhoT(200, 650), u3_rhoT(200, 650), s3_rhoT(200, 650), Cp3_rhoT(200, 650),
             w3_rhoT(200, 650))
        echo(p3_rhoT(500, 750), h3_rhoT(500, 750), u3_rhoT(500, 750), s3_rhoT(500, 750), Cp3_rhoT(500, 750),
             w3_rhoT(500, 750))

        echo('t3_ph')
        echo(T3_ph(20.0, 1700))
        echo(T3_ph(50.0, 2000))
        echo(T3_ph(100.0, 2100))
        echo(T3_ph(20.0, 2500))
        echo(T3_ph(50.0, 2400))
        echo(T3_ph(100.0, 2700))

        echo('v3_ph')
        echo(v3_ph(20.0, 1700))
        echo(v3_ph(50.0, 2000))
        echo(v3_ph(100.0, 2100))
        echo(v3_ph(20.0, 2500))
        echo(v3_ph(50.0, 2400))
        echo(v3_ph(100.0, 2700))

        echo('t3_ps')
        echo(T3_ps(20.0, 3.7))
        echo(T3_ps(50.0, 3.5))
        echo(T3_ps(100.0, 4))
        echo(T3_ps(20.0, 5))
        echo(T3_ps(50.0, 4.5))
        echo(T3_ps(100.0, 5))

        echo('v3_ps')
        echo(v3_ps(20.0, 3.7))
        echo(v3_ps(50.0, 3.5))
        echo(v3_ps(100.0, 4))
        echo(v3_ps(20.0, 5))
        echo(v3_ps(50.0, 4.5))
        echo(v3_ps(100.0, 5))

        echo('p3_hs')
        echo(p3_hs(1700, 3.8))
        echo(p3_hs(2000, 4.2))
        echo(p3_hs(2100, 4.3))
        echo(p3_hs(2500, 5.1))
        echo(p3_hs(2400, 4.7))
        echo(p3_hs(2700, 5))

        echo('h3_pt')
        echo(h3_pT(25.583702, 650))
        echo(h3_pT(22.293064, 650))
        echo(h3_pT(78.309564, 750))

        echo('REGION 4')
        echo('TABLE 35')
        echo(p4_T(300))
        echo(p4_T(500))
        echo(p4_T(600))

        echo('TABLE 36')
        echo(T4_p(0.1))
        echo(T4_p(1.0))
        echo(T4_p(10.0))

        echo('h4_s')
        echo(h4_s(1))
        echo(h4_s(2))
        echo(h4_s(3))
        echo(h4_s(3.8))
        echo(h4_s(4))
        echo(h4_s(4.2))
        echo(h4_s(7))
        echo(h4_s(8))
        echo(h4_s(9))
        echo(h4_s(5.5))
        echo(h4_s(5))
        echo(h4_s(4.5))

        echo('REGION 5')
        echo('TABLE 42')
        echo(v5_pT(0.5, 1500), h5_pT(0.5, 1500), u5_pT(0.5, 1500), s5_pT(0.5, 1500), Cp5_pT(0.5, 1500),
             w5_pT(0.5, 1500))
        echo(v5_pT(30, 1500), h5_pT(30, 1500), u5_pT(30, 1500), s5_pT(30, 1500), Cp5_pT(30, 1500), w5_pT(30, 1500))
        echo(v5_pT(30, 2000), h5_pT(30, 2000), u5_pT(30, 2000), s5_pT(30, 2000), Cp5_pT(30, 2000), w5_pT(30, 2000))

        echo('t5_ph')
        echo(T5_ph(0.5, 5219.76331549428))
        echo(T5_ph(8, 5206.09634477373))
        echo(T5_ph(8, 6583.80290533381))

        echo('t5_ps')
        echo(T5_ps(0.5, 9.65408430982588))
        echo(T5_ps(8, 8.36546724495503))
        echo(T5_ps(8, 9.15671044273249))

        echo('TESTING CALLS')
        echo(Tsat_p(0.1) - 273.15)
        echo(T_ph(0.1, 100) - 273.15)
        echo(T_ps(0.1, 1) - 273.15)
        echo(T_hs(100, 0.2) - 273.15)
        echo(psat_T(100 + 273.15) * 10)
        echo(p_hs(84, 0.296) * 10)

        echo(hV_p(1 * 0.1))
        echo(hL_p(1 * 0.1))
        echo(hV_T(100 + 273.15))
        echo(hL_T(100 + 273.15))
        echo(h_pT(1 * 0.1, 20 + 273.15))
        echo(h_ps(1 * 0.1, 1))
        echo(h_px(1 * 0.1, 0.5))
        echo(h_prho(1 * 0.1, 2))
        echo(h_Tx(100 + 273.15, 0.5))

        echo(vV_p(1 * 0.1))
        echo(vL_p(1 * 0.1))
        echo(vV_T(100 + 273.15))
        echo(vL_T(100 + 273.15))
        echo(v_pT(1 * 0.1, 100 + 273.15))
        echo(v_ph(1 * 0.1, 1000))
        echo(v_ps(1 * 0.1, 5))

        echo(rhoV_p(1 * 0.1))
        echo(rhoL_p(1 * 0.1))
        echo(rhoV_T(100 + 273.15))
        echo(rhoL_T(100 + 273.15))
        echo(rho_pT(1 * 0.1, 100 + 273.15))
        echo(rho_ph(1 * 0.1, 1000))
        echo(rho_ps(1 * 0.1, 1))

        echo(sV_p(0.006117 * 0.1))
        echo(sL_p(0.0061171 * 0.1))
        echo(sV_T(0.0001 + 273.15))
        echo(sL_T(100 + 273.15))
        echo(s_pT(1 * 0.1, 20 + 273.15))
        echo(s_ph(1 * 0.1, 84.01181117))

        echo(uV_p(1 * 0.1))
        echo(uL_p(1 * 0.1))
        echo(uV_T(100 + 273.15))
        echo(uL_T(100 + 273.15))
        echo(u_pT(1 * 0.1, 100 + 273.15))
        echo(u_ph(1 * 0.1, 1000))
        echo(u_ps(1 * 0.1, 1))

        echo(CpV_p(1 * 0.1))
        echo(CpL_p(1 * 0.1))
        echo(CpV_T(100 + 273.15))
        echo(CpL_T(100 + 273.15))
        echo(Cp_pT(1 * 0.1, 100 + 273.15))
        echo(Cp_ph(1 * 0.1, 200))
        echo(Cp_ps(1 * 0.1, 1))

        echo(CvV_p(1 * 0.1))
        echo(CvL_p(1 * 0.1))
        echo(CvV_T(100 + 273.15))
        echo(CvL_T(100 + 273.15))
        echo(Cv_pT(1 * 0.1, 100 + 273.15))
        echo(Cv_ph(1 * 0.1, 200))
        echo(Cv_ps(1 * 0.1, 1))

        echo(wV_p(1 * 0.1))
        echo(wL_p(1 * 0.1))
        echo(wV_T(100 + 273.15))
        echo(wL_T(100 + 273.15))
        echo(w_pT(1 * 0.1, 100 + 273.15))
        echo(w_ph(1 * 0.1, 200))
        echo(w_ps(1 * 0.1, 1))

        echo(my_pT(1 * 0.1, 100 + 273.15))
        echo(my_ph(1 * 0.1, 100))
        echo(my_ps(1 * 0.1, 1))

        echo(tcL_p(1 * 0.1))
        echo(tcV_p(1 * 0.1))
        echo(tcL_T(25 + 273.15))
        echo(tcV_T(25 + 273.15))
        echo(tc_pT(1 * 0.1, 25 + 273.15))
        echo(tc_ph(1 * 0.1, 100))
        echo(tc_hs(100, 0.34))

        echo(st_t(100 + 273.15))
        echo(st_p(1 * 0.1))

        echo(x_ph(1 * 0.1, 1000))
        echo(x_ps(1 * 0.1, 4))

        echo(vx_ph(1 * 0.1, 418))
        echo(vx_ps(1 * 0.1, 4))


def benchmark():
    self_test(False, repeat=100)


# self_test(show=True)
benchmark()
