import math
import numpy as np
from sim21.old.provider.phase import PhaseByMole
from sim21.old.provider.base import Provider, MIN_COMPOSITION, GAS_CONSTANT


class IdealVapLiq(Provider):
    def __init__(self, components=None):
        super().__init__(components)
        if components:
            c = [i for i in components]
            self.setup_components(c)

    def setup_components(self, components, **kwargs):
        super().setup_components(components, **kwargs)
        pass

    def log_gamma(self, temp, press, n, valid):
        return np.zeros_like(n)

    def log_phi_pure_liq(self, temp, press, n, valid):
        calc = np.zeros_like(n)
        comps = self.components
        for i in valid:
            calc[i] = math.log(comps[i].vap_press(temp) / press)

        return calc

    def phase(self, temp, press, n, desired_phase,
              allow_pseudo=True, valid=None, press_comp_derivs=False,
              log_phi_temp_press_derivs=False, log_phi_comp_derivs=False):

        comps = self.components
        if desired_phase not in ['vap', 'liq']:
            raise NotImplementedError

        if True in (press_comp_derivs, log_phi_temp_press_derivs, log_phi_comp_derivs):
            raise NotImplementedError

        if valid is None:
            valid = np.where(n > MIN_COMPOSITION)[0]

        calc_ig_vol, \
        calc_mw, \
        calc_ig_cp, \
        calc_ig_enthalpy, \
        calc_ig_entropy, \
        calc_ig_int_energy, \
        calc_ig_gibbs, \
        calc_ig_helmholtz = self.ig_props(temp, press, n, valid=valid)

        phase_id = desired_phase
        pseudo = False
        temp = temp
        mw = calc_mw
        frac_mole = n
        frac_sum_mole = 1.0

        ig_gibbs_mole = calc_ig_gibbs
        ig_helmholtz_mole = 0
        ig_int_energy_mole = 0
        ig_enthalpy_mole = calc_ig_enthalpy
        ig_entropy_mole = calc_ig_entropy
        ig_cv_mole = calc_ig_cp - GAS_CONSTANT
        ig_cp_mole = calc_ig_cp

        if desired_phase == 'vap':
            vol_mole = calc_ig_vol
            dens_mole = 1 / vol_mole
            compress_fact = 1.0

            res_gibbs_mole = 0
            res_helmholtz_mole = 0
            res_int_energy_mole = 0
            res_enthalpy_mole = 0
            res_entropy_mole = 0
            res_cv_mole = 0
            res_cp_mole = 0
            log_phi = np.zeros(len(n))

            gibbs_mole = ig_gibbs_mole
            helmholtz_mole = ig_helmholtz_mole
            int_energy_mole = ig_int_energy_mole
            enthalpy_mole = ig_enthalpy_mole
            entropy_mole = ig_entropy_mole
            cv_mole = ig_cv_mole
            cp_mole = ig_cp_mole

        elif desired_phase == 'liq':
            # Replace for other properties
            vol_mole = 0
            enthalpy_mole = 0
            int_energy_mole = 0
            entropy_mole = 0
            entropy_sum_mole = 0
            cp_mole = 0
            log_phi_pure_liq = self.log_phi_pure_liq(temp, press, n, valid)
            for i in valid:
                vol_mole += n[i]*comps[i].liq_vol_mole(temp)
                calc_x = comps[i].liq_enthalpy_mole(temp)
                enthalpy_mole += n[i]*calc_x
                int_energy_mole += n[i]*(calc_x + GAS_CONSTANT*temp)
                entropy_mole += n[i]*comps[i].liq_entropy_mole(temp, press)
                entropy_sum_mole = -GAS_CONSTANT * n[i] * math.log(n[i])
                cp_mole += n[i]*comps[i].liq_cp_mole(temp)

            dens_mole = 1/vol_mole
            compress_fact = press*vol_mole/(GAS_CONSTANT * temp)
            entropy_mole = entropy_mole + entropy_sum_mole
            gibbs_mole = enthalpy_mole - temp*entropy_mole
            helmholtz_mole = int_energy_mole - temp*entropy_mole
            # This is not correct but nobody should be relying on this property anyway
            cv_mole = cp_mole - GAS_CONSTANT

            res_gibbs_mole = gibbs_mole - ig_gibbs_mole
            res_helmholtz_mole = helmholtz_mole - ig_helmholtz_mole
            res_int_energy_mole = int_energy_mole - ig_int_energy_mole
            res_enthalpy_mole = enthalpy_mole - ig_enthalpy_mole
            res_entropy_mole = entropy_mole - ig_entropy_mole
            res_cp_mole = cp_mole - ig_cp_mole
            res_cv_mole = cv_mole - ig_cv_mole

            # This is the activity coefficient; zero for this case
            log_act_coeff = self.log_gamma(temp, press, n, valid)
            log_phi = log_act_coeff + log_phi_pure_liq
        else:
            raise NotImplementedError

        del_log_phi_del_temp = None
        del_log_phi_del_press = None
        del_log_phi_del_comp = None
        del_press_del_vol = None
        del_press_del_temp = None
        del_press_del_comp = None
        del_vol_del_comp = None

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

