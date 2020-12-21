import math
import sys
import numpy as np
from numpy import ndarray

MIN_COMPOSITION = math.sqrt(sys.float_info.epsilon)


class Provider:
    def __init__(self):
        self.observers = set()
        self.flash_basis = 'mole'

    def add_observer(self, new_obs):
        self.observers.add(new_obs)

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
