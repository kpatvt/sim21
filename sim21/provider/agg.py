import numpy as np
from sim21.data.chemsep_consts import GAS_CONSTANT


def collect_attribute_value(phase_list, phase_mole_fractions, attr_name):
    results = 0
    for p, f in zip(phase_list, phase_mole_fractions):
        a = getattr(p, attr_name)
        results += a * f

    return results


class AggregateByMole:
    """
    Aggregate is a collection of phase results with short cuts to access some key properties.
    """

    def __init__(self, provider, phase_list, phase_mole_fractions):
        self.provider = provider
        assert len(phase_list) == len(phase_mole_fractions)

        self._phases = {}
        self._phases_frac_mole = {}
        for ph, f in zip(phase_list, phase_mole_fractions):
            self._phases[ph.identifier] = ph
            self._phases_frac_mole[ph.identifier] = f

        # Get the common properties
        self.temp = phase_list[0].temp
        self.press = phase_list[0].press
        self.vap = self._phases.get('vap', None)
        self.liq = self._phases.get('liq', None)
        self.liq_1 = self.liq

        phase_attrs = ['vol_mole',
                       'ig_gibbs_mole', 'ig_helmholtz_mole', 'ig_int_energy_mole',
                       'ig_enthalpy_mole', 'ig_entropy_mole', 'ig_cv_mole', 'ig_cp_mole',
                       'res_gibbs_mole', 'res_helmholtz_mole', 'res_int_energy_mole',
                       'res_enthalpy_mole', 'res_entropy_mole', 'res_cv_mole', 'res_cp_mole',
                       'gibbs_mole', 'helmholtz_mole', 'int_energy_mole',
                       'enthalpy_mole', 'entropy_mole', 'cv_mole', 'cp_mole', 'comp_mole', 'mw']

        calc_values = [collect_attribute_value(phase_list, phase_mole_fractions, a) for a in phase_attrs]

        self.vol_mole, self.ig_gibbs_mole, self.ig_helmholtz_mole, self.ig_int_energy_mole, \
        self.ig_enthalpy_mole, self.ig_entropy_mol, self.ig_cv_mole, self.ig_cp_mole, \
        self.res_gibbs_mole, self.res_helmholtz_mole, self.res_int_energy_mole, \
        self.res_enthalpy_mole, self.res_entropy_mole, self.res_cv_mole, self.res_cp_mole, \
        self.gibbs_mole, self.helmholtz_mole, self.int_energy_mole, \
        self.enthalpy_mole, self.entropy_mole, self.cv_mole, self.cp_mole, self.comp_mole, self.mw = calc_values

        # self.comp_mole = collect_attribute_value(phase_list, phase_mole_fractions, 'comp_mole')
        self.flow_sum_mole = 1
        self.flow_mole = self.comp_mole
        self._scaled = False

    @property
    def phases(self):
        return set(self._phases.keys())

    def contains(self, *possible_phases):
        have_all_phases = False
        my_phases = self.phases
        if my_phases.intersection(possible_phases) == set(possible_phases):
            have_all_phases = True

        return have_all_phases

    def __contains__(self, item):
        return self.contains(item)

    def __getitem__(self, item):
        return self._phases[item]

    def frac_mole(self, item):
        return self._phases_frac_mole[item]

    @property
    def k_values_vle(self):
        vap, liq = self.vap, self.liq
        vap_mole_comp = vap.comp_mole
        liq_mole_comp = liq.comp_mole
        return vap_mole_comp / liq_mole_comp

    @property
    def vap_frac_mole(self):
        if self.vap is None:
            return 0

        return self._phases_frac_mole['vap']

    def scale(self, flow_sum_mole=None):
        if flow_sum_mole is not None:
            self.flow_sum_mole = flow_sum_mole
            self.flow_mole = flow_sum_mole * self.comp_mole
            for phase_name, ph in self._phases.items():
                f = self._phases_frac_mole[phase_name]
                ph.scale(f * flow_sum_mole)

            self._scaled = True

        return self

    def clone(self):
        phase_list = []
        phase_mole_fractions = []
        for ph_name, ph in self._phases.items():
            phase_list.append(ph)
            phase_mole_fractions.append(self._phases_frac_mole[ph_name])

        return AggregateByMole(self.provider, phase_list, phase_mole_fractions)

    def extract(self, ph):
        # This assumes the local agg has been scaled already...
        assert self._scaled
        new_agg = None
        given_ph = self._phases.get(ph, None)
        if given_ph is not None:
            ph_flow_sum_mole = given_ph.flow_sum_mole
            new_agg = AggregateByMole(self.provider, [given_ph], [1.0])
            new_agg.scale(flow_sum_mole=ph_flow_sum_mole)

        return new_agg

    def comp(self, basis):
        if basis == 'mole':
            return self.comp_mole
        elif basis == 'mass':
            m = self.comp_mole * self.provider.mw
            m /= np.sum(m)
            return m

    def get_prop(self, name, basis):
        if basis == 'mole':
            return getattr(self, name + '_mole')
        elif basis == 'mass':
            return getattr(self, name + '_mole') / self.mw
        else:
            raise NotImplementedError

    def flow_sum(self, basis):
        return self.get_prop('flow_sum', basis)

    def enthalpy(self, basis):
        return self.get_prop('enthalpy', basis)

    def flow(self, basis):
        if basis == 'mole':
            return self.flow_mole
        elif basis == 'mass':
            return self.flow_mole * self.provider.mw
        else:
            raise NotImplementedError

    @property
    def z_factor(self):
        return self.press*self.vol_mole/(GAS_CONSTANT*self.temp)

    @property
    def std_liq_vol_mole(self):
        return np.dot(self.provider.std_liq_vol_mole, self.comp_mole)

