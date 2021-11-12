import numpy as np

from sim21.old.provider.agg import AggregateByMole
from sim21.old.provider.base import Provider, MIN_COMPOSITION
from sim21.old.provider import xsteam2
import math
from dataclasses import dataclass
from numpy import ndarray

from sim21.old.provider.error import FlashConvergenceError
from sim21.old.support.roots import secant_method


class Steam97Component:
    def __init__(self):
        self.identifier = 'WATER'
        # These are set to match the IFC-97 specification
        self.mw = xsteam2.MOLAR_MASS
        self.crit_temp = xsteam2.CRITICAL_TEMPERATURE
        self.crit_press = xsteam2.CRITICAL_PRESSURE * 1e6

        # These are just dummy values, they are not really used in any calculations
        # We call custom flash routines for Steam97, so the values here have no effect
        self.acen_fact = 0.344
        self.ig_temp_ref = 298.15
        self.ig_press_ref = 101325.0
        self.ig_cp_mole_coeffs = [100, 298.15, 2000.0, 0, 0, 0, 0, 0, 0]
        self.ig_enthalpy_form_mole = 0
        self.ig_gibbs_form_mole = 0
        self.std_liq_vol_mole = 0.018069476

    def surf_tens(self, t):
        return xsteam2.Surface_Tension_T(t)

    def vap_visc(self, t, p):
        return xsteam2.my_pT(t, p / 1e6)

    def liq_visc(self, t, p):
        return xsteam2.my_pT(t, p / 1e6)


@dataclass
class PhaseByMoleSteam97:
    provider: Provider
    identifier: str
    pseudo: bool
    temp: float
    press: float
    vol_mole: float
    dens_mole: float
    z_factor: float
    comp_mole: ndarray
    comp_sum_mole: float
    mw: float

    ig_gibbs_mole: float
    ig_helmholtz_mole: float
    ig_int_energy_mole: float
    ig_enthalpy_mole: float
    ig_entropy_mole: float
    ig_cv_mole: float
    ig_cp_mole: float

    res_gibbs_mole: float
    res_helmholtz_mole: float
    res_int_energy_mole: float
    res_enthalpy_mole: float
    res_entropy_mole: float
    res_cv_mole: float
    res_cp_mole: float

    gibbs_mole: float
    helmholtz_mole: float
    int_energy_mole: float
    enthalpy_mole: float
    entropy_mole: float
    cv_mole: float
    cp_mole: float

    flow_mole: ndarray
    flow_sum_mole: float

    def scale(self, flow_sum_mole=None):
        if flow_sum_mole is not None:
            self.flow_sum_mole = flow_sum_mole
            self.flow_mole = self.comp_mole * flow_sum_mole

    def clone(self):
        return PhaseByMoleSteam97(provider=self.provider,
                                  identifier=self.identifier,
                                  pseudo=self.pseudo, temp=self.temp, press=self.press,
                                  vol_mole=self.vol_mole, dens_mole=self.dens_mole, z_factor=self.z_factor,
                                  comp_mole=self.comp_mole, comp_sum_mole=self.comp_sum_mole, mw=self.mw,

                                  ig_gibbs_mole=self.ig_gibbs_mole, ig_helmholtz_mole=self.ig_helmholtz_mole,
                                  ig_int_energy_mole=self.ig_int_energy_mole,
                                  ig_enthalpy_mole=self.ig_enthalpy_mole, ig_entropy_mole=self.ig_entropy_mole,
                                  ig_cv_mole=self.ig_cv_mole, ig_cp_mole=self.ig_cp_mole,

                                  res_gibbs_mole=self.res_gibbs_mole, res_helmholtz_mole=self.res_helmholtz_mole,
                                  res_int_energy_mole=self.res_int_energy_mole,
                                  res_enthalpy_mole=self.res_enthalpy_mole, res_entropy_mole=self.res_entropy_mole,
                                  res_cv_mole=self.res_cv_mole, res_cp_mole=self.res_cp_mole,

                                  gibbs_mole=self.gibbs_mole, helmholtz_mole=self.helmholtz_mole,
                                  int_energy_mole=self.int_energy_mole,
                                  enthalpy_mole=self.enthalpy_mole, entropy_mole=self.entropy_mole,
                                  cv_mole=self.cv_mole, cp_mole=self.cp_mole,
                                  flow_mole=self.flow_mole, flow_sum_mole=self.flow_sum_mole)

    @property
    def vap_frac_mole(self):
        if self.identifier == 'vap':
            return 1.0
        else:
            return 0.0

    @property
    def std_liq_vol_mole(self):
        # m3/kmol
        return 0.018069476

    @property
    def visc(self):
        return xsteam2.my_pT(self.press / 1e6, self.temp)

    @property
    def dens_mass(self):
        return self.dens_mole * self.mw

    @property
    def surf_tens(self):
        assert self.identifier == 'liq'
        return xsteam2.Surface_Tension_T(self.temp)

    @property
    def ig_enthalpy_form_mole(self):
        return 0


def generate_phase(provider, temp, press, act_phase, x):
    press_mpa = press/1e6
    comp_mole = np.array([1.0])
    comp_sum_mole = 1.0
    mw = xsteam2.MOLAR_MASS
    gc = mw * xsteam2.GAS_CONSTANT_MASS * 1e3

    if act_phase in ['vap', 'liq']:
        vol_mass = xsteam2.v_pT(press_mpa, temp)     # m3/kg
        vol_mole = vol_mass * mw
        dens_mass = 1/vol_mass  # kg/m3
        dens_mole = dens_mass / xsteam2.MOLAR_MASS  # kg/m3 * kmol/MW kg
        vol_mole = 1 / dens_mole  # m3/kmol

        z_factor = press * vol_mole / (gc * temp)
        enthalpy_mole = xsteam2.h_pT(press_mpa, temp) * 1e3 * mw  # kJ/kg * 1000 = J/kg * MW kg/kmol = J/kmol
        entropy_mole = xsteam2.s_pT(press_mpa, temp) * 1e3 * mw  # kJ/kg-K * 1000 = J/kg-K * MW kg/kmol = J/kmol-K
        gibbs_mole = enthalpy_mole - temp * entropy_mole  # g = h - t*s
        int_energy_mole = xsteam2.u_pT(press_mpa, temp) * 1e3 * mw  # kJ/kg * 1000 = J/kg * MW kg/kmol = J/kmol
        helmholtz_mole = int_energy_mole - temp * entropy_mole  # a = u - t*s
        cp_mole = xsteam2.Cp_pT(press_mpa, temp) * 1e3 * mw  # J/kmol-K
        cv_mole = xsteam2.Cv_pT(press_mpa, temp) * 1e3 * mw  # J/kmol-K

    else:
        if x <= 0:
            act_phase = 'liq'
        elif x >= 1:
            act_phase = 'vap'
        else:
            raise AssertionError

        vol_mass = xsteam2.v_px(press_mpa, x)
        vol_mole = vol_mass * xsteam2.MOLAR_MASS
        dens_mass = 1 / vol_mass
        dens_mole = 1 / vol_mole
        z_factor = press * vol_mole / (gc * temp)

        enthalpy_mole = xsteam2.h_px(press_mpa, x) * 1e3 * mw
        entropy_mole = xsteam2.s_px(press_mpa, x) * 1e3 * mw
        gibbs_mole = enthalpy_mole - temp * entropy_mole
        int_energy_mole = xsteam2.u_px(press_mpa, x) * 1e3 * mw
        helmholtz_mole = int_energy_mole - temp * entropy_mole
        cp_mole = xsteam2.Cp_px(press_mpa, x) * 1e3 * mw
        cv_mole = xsteam2.Cv_px(press_mpa, x) * 1e3 * mw

    h, s, g, u, a, cv, cp = xsteam2.ig_props(press_mpa, temp)
    ig_enthalpy_mole = h * 1e3 * mw
    ig_entropy_mole = s * 1e3 * mw
    ig_gibbs_mole = ig_enthalpy_mole - temp * ig_entropy_mole
    ig_int_energy_mole = u * 1e3 * mw
    ig_helmholtz_mole = ig_int_energy_mole - temp * ig_entropy_mole
    ig_cp_mole = cp * 1e3 * mw
    ig_cv_mole = cv * 1e3 * mw

    res_enthalpy_mole = enthalpy_mole - ig_enthalpy_mole
    res_entropy_mole = entropy_mole - ig_entropy_mole
    res_gibbs_mole = gibbs_mole - ig_gibbs_mole
    res_int_energy_mole = int_energy_mole - ig_int_energy_mole
    res_helmholtz_mole = helmholtz_mole - ig_helmholtz_mole
    res_cp_mole = cp_mole - ig_cp_mole
    res_cv_mole = cv_mole - ig_cv_mole

    return PhaseByMoleSteam97(provider=provider,
                              identifier=act_phase,
                              pseudo=False,
                              temp=temp,
                              press=press,
                              vol_mole=vol_mole,
                              dens_mole=dens_mole,
                              z_factor=z_factor,
                              comp_mole=comp_mole,
                              comp_sum_mole=comp_sum_mole,
                              mw=mw,

                              ig_gibbs_mole=ig_gibbs_mole,
                              ig_helmholtz_mole=ig_helmholtz_mole,
                              ig_int_energy_mole=ig_int_energy_mole,
                              ig_enthalpy_mole=ig_enthalpy_mole,
                              ig_entropy_mole=ig_entropy_mole,
                              ig_cv_mole=ig_cv_mole,
                              ig_cp_mole=ig_cp_mole,

                              res_gibbs_mole=res_gibbs_mole,
                              res_helmholtz_mole=res_helmholtz_mole,
                              res_int_energy_mole=res_int_energy_mole,
                              res_enthalpy_mole=res_enthalpy_mole,
                              res_entropy_mole=res_entropy_mole,
                              res_cv_mole=res_cv_mole,
                              res_cp_mole=res_cp_mole,

                              gibbs_mole=gibbs_mole,
                              helmholtz_mole=helmholtz_mole,
                              int_energy_mole=int_energy_mole,
                              enthalpy_mole=enthalpy_mole,
                              entropy_mole=entropy_mole,
                              cv_mole=cv_mole,
                              cp_mole=cp_mole,

                              flow_mole=np.array([1.0]),
                              flow_sum_mole=1.0)


def determine_phase(temp, press):
    # When supercritical, just assume 'vap'
    # Doesn't affect any calculated values, just the phase identifier
    if temp > xsteam2.CRITICAL_TEMPERATURE:
        return 'vap'

    press_mpa = press/1e6

    # TODO At some point we have to figure out how to handle supercritical
    #      results
    if press_mpa > xsteam2.CRITICAL_PRESSURE:
        return 'vap'

    sat_temp = xsteam2.Tsat_p(press_mpa)
    if math.isclose(temp, sat_temp, abs_tol=1e-6):
        act_phase = 'vap_liq'
    elif temp > sat_temp:
        act_phase = 'vap'
    else:
        act_phase = 'liq'

    return act_phase


class Steam97(Provider):
    def __init__(self, components=None):
        super().__init__()
        self._components = []
        self.all_comps = []
        self._id_list = []
        self._mw_list = []
        self._std_liq_vol_mole = []

    @property
    def components(self):
        return self._components

    @property
    def all_valid_components(self):
        return self.all_comps

    def setup_components(self, components, **kwargs):
        cleanup = False
        if len(components) == 0:
            cleanup = True

        if not cleanup:
            components = [Steam97Component()]
        else:
            self._components = components = []

        super().setup_components(components, **kwargs)

    def phase(self, temp, press, n, desired_phase,
              allow_pseudo=True, valid=None, press_comp_derivs=False,
              log_phi_temp_press_derivs=False, log_phi_comp_derivs=False):

        press_mpa = press/1e6
        if desired_phase not in ['vap', 'liq']:
            raise NotImplementedError

        if valid is None:
            valid = np.where(n > MIN_COMPOSITION)[0]

        x = 0
        act_phase = determine_phase(temp, press)
        if act_phase == 'vap_liq':
            if desired_phase == 'vap':
                x = 1
            else:
                x = 0

        return generate_phase(self, temp, press, act_phase, x)

    def flash_temp_press(self, flow_sum_basis, flow_sum_value, frac_basis, frac_value, temp, press, previous, valid):
        flow_sum_value_mole, frac_value_mole = self.convert_to_mole_basis(flow_sum_basis, flow_sum_value,
                                                                          frac_basis, frac_value)
        act_phase = determine_phase(temp, press)
        # This is tricky - but we are close to the boiling point return the two phase with a vap-fraction 1
        if act_phase == 'vap_liq':
            vap_ph = generate_phase(self, temp, press, act_phase, 1)
            liq_ph = generate_phase(self, temp, press, act_phase, 0)
            results = AggregateByMole(self, [vap_ph, liq_ph], [1, 0])
        else:
            ph = generate_phase(self, temp, press, act_phase, 0)
            results = AggregateByMole(self, [ph], [1])

        results.scale(flow_sum_mole=flow_sum_value_mole)
        return results

    def flash_press_prop(self, flow_sum_basis, flow_sum_value,
                         frac_basis, frac_value, press,
                         prop_name, prop_basis, prop_value, previous, valid):

        flow_sum_value_mole, frac_value_mole = self.convert_to_mole_basis(flow_sum_basis, flow_sum_value,
                                                                          frac_basis, frac_value)
        mw = xsteam2.MOLAR_MASS
        press_mpa = press/1e6
        h_given = prop_value / (1e3 * mw)
        prop_name = prop_name + '_' + prop_basis
        supported_props = {'enthalpy_mole': (xsteam2.T_ph, xsteam2.hL_p, xsteam2.hV_p),
                           'entropy_mole': (xsteam2.T_ps, xsteam2.sL_p, xsteam2.sV_p)}

        if prop_name in supported_props:
            # Get the functions to calculate the relevant properties
            temp_calculator, liq_prop, vap_prop = supported_props[prop_name]
            # Get the temp
            temp_value = temp_calculator(press_mpa, h_given)
            # if press is greater than crit. pressure, there's not much to do
            if press_mpa > xsteam2.CRITICAL_PRESSURE:
                ph = generate_phase(self, temp_value, press, 'vap', 1)
                return AggregateByMole(self, [ph], [1])
            else:
                # If not check if we are a vapor, liquid or mixed phase
                hL = liq_prop(press_mpa)
                hV = vap_prop(press_mpa)
                x = (h_given - hL)/(hV - hL)
                # It's all vapor
                if x > 1:
                    ph = generate_phase(self, temp_value, press, 'vap', 1)
                    return AggregateByMole(self, [ph], [1])
                elif x < 0:
                    ph = generate_phase(self, temp_value, press, 'liq', 0)
                    return AggregateByMole(self, [ph], [1])
                else:
                    # Return a mixture
                    vap_ph = generate_phase(self, temp_value, press, 'vap_liq', 1)
                    liq_ph = generate_phase(self, temp_value, press, 'vap_liq', 0)
                    return AggregateByMole(self, [vap_ph, liq_ph], [x, 1-x])
        else:
            raise NotImplementedError

    def flash_temp_prop(self, flow_sum_basis, flow_sum_value,
                         frac_basis, frac_value, temp,
                         prop_name, prop_basis, prop_value, previous, valid):

        flow_sum_value_mole, frac_value_mole = self.convert_to_mole_basis(flow_sum_basis, flow_sum_value,
                                                                          frac_basis, frac_value)
        mw = xsteam2.MOLAR_MASS
        h_given = prop_value / (1e3 * mw)

        supported_props = {'enthalpy_mole': (xsteam2.h_pT, xsteam2.hL_T, xsteam2.hV_T),
                           'entropy_mole': (xsteam2.s_pT, xsteam2.sL_T, xsteam2.sV_T)}

        prop_given_pT, sat_liq_prop, sat_vap_prop = supported_props[prop_name]

        def search_for_press(press_guess_mpa):
            return prop_given_pT(press_guess_mpa, temp) - h_given

        if prop_name in supported_props:
            if temp < xsteam2.CRITICAL_TEMPERATURE:
                hL = sat_liq_prop(temp)
                hV = sat_vap_prop(temp)
                x = (h_given - hL)/(hV - hL)
                p_sat_mpa = xsteam2.psat_T(temp)
                # It's all vapor
                if x > 1:
                    new_press_mpa = secant_method(search_for_press, p_sat_mpa, p_sat_mpa - 0.01, xtol=1e-6)
                    press = new_press_mpa * 1e6
                    ph = generate_phase(self, temp, press, 'vap', 1)
                    return AggregateByMole(self, [ph], [1])
                elif x < 0:
                    new_press_mpa = secant_method(search_for_press, p_sat_mpa, p_sat_mpa - 0.01, xtol=1e-6)
                    press = new_press_mpa * 1e6
                    ph = generate_phase(self, temp, press, 'liq', 0)
                    return AggregateByMole(self, [ph], [1])
                else:
                    # Return a mixture
                    press = xsteam2.psat_T(temp) * 1e6
                    vap_ph = generate_phase(self, temp, press, 'vap_liq', 1)
                    liq_ph = generate_phase(self, temp, press, 'vap_liq', 0)
                    return AggregateByMole(self, [vap_ph, liq_ph], [x, 1-x])
            else:
                # Arbitrary point guess
                p_start = 10.0
                new_press_mpa = secant_method(search_for_press, p_start, p_start - 0.01, xtol=1e-6)
                press = new_press_mpa * 1e6
                ph = generate_phase(self, temp, press, 'vap', 1)
        else:
            raise NotImplementedError

    def flash_press_vap_frac(self, flow_sum_basis, flow_sum_value, frac_basis, frac_value, press,
                             vap_frac_basis, vap_frac_value, previous, valid):

        flow_sum_value_mole, frac_value_mole = self.convert_to_mole_basis(flow_sum_basis, flow_sum_value,
                                                                          frac_basis, frac_value)
        if vap_frac_basis != 'mole':
            raise NotImplementedError

        # This should always return two phases
        x = vap_frac_value
        press_mpa = press/1e6
        # If we're greater than crit pressure, then there are no two phase results.
        if press_mpa > xsteam2.CRITICAL_PRESSURE or press_mpa <= 0:
            raise FlashConvergenceError

        # Let's determine the correspond temp
        temp = xsteam2.Tsat_p(press_mpa)
        if math.isnan(temp) or temp < 0:
            raise FlashConvergenceError

        vap_ph = generate_phase(self, temp, press, 'vap_liq', 1)
        liq_ph = generate_phase(self, temp, press, 'vap_liq', 0)
        results = AggregateByMole(self, [vap_ph, liq_ph], [x, 1 - x])
        return results

    def flash_temp_vap_frac(self, flow_sum_basis, flow_sum_value, frac_basis, frac_value, temp,
                             vap_frac_basis, vap_frac_value, previous, valid):

        flow_sum_value_mole, frac_value_mole = self.convert_to_mole_basis(flow_sum_basis, flow_sum_value,
                                                                          frac_basis, frac_value)
        if vap_frac_basis != 'mole':
            raise NotImplementedError

        # This should always return two phases
        x = vap_frac_value
        # If we're greater than crit pressure, then there are no two phase results.
        if temp > xsteam2.CRITICAL_TEMPERATURE:
            raise FlashConvergenceError

        # Let's determine the correspond temp
        press_mpa = xsteam2.psat_T(temp)
        if math.isnan(press_mpa) or press_mpa < 0:
            raise FlashConvergenceError

        press = press_mpa*1e6
        vap_ph = generate_phase(self, temp, press, 'vap', 1)
        liq_ph = generate_phase(self, temp, press, 'liq', 0)
        results = AggregateByMole(self, [vap_ph, liq_ph], [x, 1 - x])
        return results

    def AddCompound(self, compound_by_name, compound_obj=None):
        if compound_by_name != 'WATER' or (compound_obj is not None and compound_obj.identifier != 'WATER'):
            raise NotImplementedError

        super().AddCompound(compound_by_name, compound_obj)

    def GetAvCompoundNames(self):
        return ['WATER']

    def ExchangeCompound(self, cmp1Name, cmp2Name):
        raise NotImplementedError

    def MoveCompound(self, cmp1Name, cmp2Name):
        raise NotImplementedError

