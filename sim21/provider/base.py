import math
import sys
import numpy as np
from numpy import ndarray

from sim21.data import chemsep
from sim21.data.chemsep_consts import GAS_CONSTANT
from numba import njit

from sim21.provider.generic import calc_ig_props
from sim21.provider.flash.basic import basic_flash_temp_press_2phase
from sim21.provider.flash.io import flash_press_prop_2phase, flash_press_vap_frac_2phase, flash_temp_vap_frac_2phase, \
    flash_temp_prop_2phase

MIN_COMPOSITION = math.sqrt(sys.float_info.epsilon)


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


class Provider:
    def __init__(self, components=None):
        self.observers = set()
        self.flash_basis = 'mole'
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
        self._surf_tens = None
        self._ig_s_form = None
        self._std_liq_vol = None

    def add_observer(self, new_obs):
        self.observers.add(new_obs)

    @property
    def components(self):
        return self._components

    @property
    def all_valid_components(self):
        return self.all_comps

    def setup_components(self, components, **kwargs):
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
        self._surf_tens = [c.surf_tens for c in components]
        self._ig_s_form = (g - h) / -298.15
        self._std_liq_vol = np.array([c.std_liq_vol_mole for c in components])

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

    def surf_tens(self, temp, comp_mole):
        return np.dot(comp_mole, [comp_surf_tens(temp) for comp_surf_tens in self._surf_tens])

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

    def flash(self,
              flow_sum_basis=None, flow_sum_value=None,
              frac_basis=None, frac_value=None,
              temp=None, press=None,
              vol_basis=None, vol_value=None,
              vap_frac_value=None, vap_frac_basis=None,
              deg_subcool=None, deg_supheat=None,
              enthalpy_basis=None, enthalpy_value=None,
              entropy_basis=None, entropy_value=None,
              int_energy_basis=None, int_energy_value=None,
              previous=None):

        assert None not in (flow_sum_basis, flow_sum_value)
        assert frac_basis in ('mole', 'mass')
        assert frac_value is not None and isinstance(frac_value, ndarray)
        valid = np.where(frac_value > MIN_COMPOSITION)[0]

        if temp is not None:
            if press is not None:
                # flash_temp_press
                return self.flash_temp_press(flow_sum_basis, flow_sum_value, frac_basis, frac_value, temp, press,
                                             previous, valid)
            elif vap_frac_value is not None:
                # flash_temp_vap_frac
                return self.flash_temp_vap_frac(flow_sum_basis, flow_sum_value, frac_basis, frac_value, temp,
                                                vap_frac_basis, vap_frac_value, previous, valid)
            elif deg_subcool is not None:
                # flash_temp_deg_subcool
                return self.flash_temp_subcool(flow_sum_basis, flow_sum_value, frac_basis, frac_value, temp,
                                               deg_subcool, previous, valid)
            elif deg_supheat is not None:
                # flash_temp_deg_supheat
                return self.flash_temp_subcool(flow_sum_basis, flow_sum_value, frac_basis, frac_value, temp,
                                               deg_supheat, previous, valid)
            elif enthalpy_value is not None:
                # flash_temp_enthalpy
                return self.flash_temp_prop(flow_sum_basis, flow_sum_value, frac_basis, frac_value, temp, 'enthalpy',
                                            enthalpy_basis, enthalpy_value, previous, valid)
            elif entropy_value is not None:
                # flash_temp_entropy
                return self.flash_temp_prop(flow_sum_basis, flow_sum_value, frac_basis, frac_value, temp, 'entropy',
                                            entropy_basis, entropy_value, previous, valid)
            elif int_energy_value is not None:
                # flash_temp_int_energy
                return self.flash_temp_prop(flow_sum_basis, flow_sum_value, frac_basis, frac_value, temp, 'int_energy',
                                            int_energy_basis, int_energy_value, previous, valid)
            elif vol_value is not None:
                # flash_temp_vol
                return self.flash_temp_vol(flow_sum_basis, flow_sum_value, frac_basis, frac_value, temp, vol_basis,
                                           vol_value, previous, valid)
            else:
                raise NotImplementedError

        elif press is not None:
            if vap_frac_value is not None:
                # flash_press_vap_frac
                return self.flash_press_vap_frac(flow_sum_basis, flow_sum_value, frac_basis, frac_value, press,
                                                 vap_frac_basis, vap_frac_value, previous, valid)
            elif deg_subcool is not None:
                # flash_press_deg_subcool
                return self.flash_press_subcool(flow_sum_basis, flow_sum_value, frac_basis, frac_value, press,
                                                deg_subcool, previous, valid)
            elif deg_supheat is not None:
                # flash_press_deg_supheat
                return self.flash_press_subcool(flow_sum_basis, flow_sum_value, frac_basis, frac_value, press,
                                                deg_supheat, previous, valid)
            elif enthalpy_value is not None:
                # flash_press_enthalpy
                return self.flash_press_prop(flow_sum_basis, flow_sum_value, frac_basis, frac_value, press, 'enthalpy',
                                             enthalpy_basis, enthalpy_value, previous, valid)
            elif entropy_value is not None:
                # flash_press_entropy
                return self.flash_press_prop(flow_sum_basis, flow_sum_value, frac_basis, frac_value, press, 'entropy',
                                             entropy_basis, entropy_value, previous, valid)
            elif int_energy_value is not None:
                # flash_press_int_energy
                return self.flash_press_prop(flow_sum_basis, flow_sum_value, frac_basis, frac_value, press,
                                             'int_energy', int_energy_basis, int_energy_value, previous, valid)
            elif vol_value is not None:
                # flash_press_vol
                return self.flash_press_vol(flow_sum_basis, flow_sum_value, frac_basis, frac_value, press, vol_basis,
                                            vol_value, previous, valid)
            else:
                raise NotImplementedError

        elif None not in (enthalpy_value, entropy_value, int_energy_value):
            prop_basis, prop_value = None, None
            if enthalpy_value is not None:
                prop_name, prop_basis, prop_value = 'enthalpy', enthalpy_basis, enthalpy_value
            elif entropy_value is not None:
                prop_name, prop_basis, prop_value = 'entropy', entropy_basis, entropy_value
            elif int_energy_value is not None:
                prop_name, prop_basis, prop_value = 'int_energy', int_energy_basis, int_energy_value
            else:
                raise NotImplementedError

            if vap_frac_value is not None:
                # flash_prop_vap_frac
                return self.flash_prop_vap_frac(flow_sum_basis, flow_sum_value, frac_basis, frac_value,
                                                prop_name, prop_basis, prop_value, vap_frac_basis, vap_frac_value,
                                                previous, valid)
            elif vol_value is not None:
                return self.flash_prop_vol(flow_sum_basis, flow_sum_value, frac_basis, frac_value,
                                           prop_name, prop_basis, prop_value,
                                           vol_basis, vol_value, previous, valid)
            else:
                raise NotImplementedError

        else:
            raise NotImplementedError

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

    def flash_temp_prop(self, flow_sum_basis, flow_sum_value,
                         frac_basis, frac_value, temp,
                         prop_name, prop_basis, prop_value, previous, valid):

        flow_sum_value_mole, frac_value_mole = self.convert_to_mole_basis(flow_sum_basis, flow_sum_value,
                                                                          frac_basis, frac_value)

        prop_flash_name = prop_name + '_' + prop_basis
        start_press = None
        if previous is not None:
            start_press = previous.press

        results = flash_temp_prop_2phase(self, temp, prop_flash_name, prop_value, 0,
                                         frac_value_mole, valid=valid, previous=previous,
                                         start_press=start_press)

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



    def AddCompound(self, compound_by_name, compound_obj=None):
        # print('AddCompound:', compound)
        if compound_obj is None:
            compound_obj = chemsep.pure(compound_by_name)
        else:
            pass

        if self._components is None:
            new_components = [compound_obj]
        else:
            new_components = self._components[:]
            new_components.append(compound_obj)

        # This is really inefficient, but it's simple
        self.setup_components(new_components)

    def GetAvCompoundNames(self):
        return chemsep.available()

    def DeleteCompound(self, compound):
        compound = compound.upper()
        idx = self._id_list.index(compound)
        new_compounds = self._components[:]
        new_compounds.pop(idx)
        self.setup_components(new_compounds)

    def ExchangeCompound(self, cmp1Name, cmp2Name):
        cmp1Name = cmp1Name.upper()
        cmp2Name = cmp2Name.upper()
        idx_1 = self._id_list.index(cmp1Name)
        idx_2 = self._id_list.index(cmp2Name)
        new_compounds = self._components[:]
        new_compounds[idx_1], new_compounds[idx_2] = new_compounds[idx_2], new_compounds[idx_1]
        self.setup_components(new_compounds)

    def MoveCompound(self, cmp1Name, cmp2Name):
        cmp1Name = cmp1Name.upper()
        cmp2Name = cmp2Name.upper()
        new_compounds = self._components[:]
        item_1 = new_compounds.pop(self._id_list.index(cmp1Name))
        new_compounds.insert(self._id_list.index(cmp2Name), item_1)
        self.setup_components(new_compounds)
