from dataclasses import dataclass
from numpy import ndarray
# from ypsim.thermo.provider import Provider


@dataclass
class PhaseByMole:
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

    log_phi: ndarray
    del_log_phi_del_temp: ndarray
    del_log_phi_del_press: ndarray
    del_log_phi_del_comp: ndarray
    del_press_del_vol: ndarray
    del_press_del_temp: ndarray
    del_press_del_comp: ndarray
    del_vol_del_comp: ndarray

    flow_mole: ndarray
    flow_sum_mole: float

    def scale(self, flow_sum_mole=None):
        if flow_sum_mole is not None:
            self.flow_sum_mole = flow_sum_mole
            self.flow_mole = self.comp_mole * flow_sum_mole

    def clone(self):
        return PhaseByMole(provider=self.provider,
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

                           log_phi=self.log_phi,
                           del_log_phi_del_temp=self.del_log_phi_del_temp,
                           del_log_phi_del_press=self.del_log_phi_del_press,
                           del_log_phi_del_comp=self.del_log_phi_del_comp,
                           del_press_del_vol=self.del_press_del_vol,
                           del_press_del_temp=self.del_press_del_temp,
                           del_press_del_comp=self.del_press_del_comp,
                           del_vol_del_comp=self.del_vol_del_comp,
                           flow_mole=self.flow_mole, flow_sum_mole=self.flow_sum_mole)
