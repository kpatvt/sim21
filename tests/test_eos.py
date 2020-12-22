import numpy as np
import matplotlib.pyplot as plt
from sim21.data.chemsep import pure
from sim21.data.chemsep_consts import GAS_CONSTANT
from sim21.provider.cubic import PengRobinson
from sim21.provider.flash.basic import basic_flash_temp_press_2phase
from sim21.provider.flash.io import flash_press_vap_frac_2phase, flash_temp_vap_frac_2phase, flash_press_prop_2phase, \
    flash_temp_prop_2phase, flash_prop_vap_frac_2phase


def press_vs_rho():
    components = pure(['ETHANE', 'N-HEPTANE'])
    prov = PengRobinson(components)
    n = np.array([0.5, 0.5])

    press_range = np.arange(0.01, 10, 0.01)
    temp_range = [420, 500]

    for temp in temp_range:
        rho_vapor, rho_liquid = [], []
        norm_hres_vapor, norm_hres_liquid = [], []
        log_phi_vapor_C2, log_phi_liquid_C2 = [], []
        log_phi_vapor_C7, log_phi_liquid_C7 = [], []

        for press in press_range:
            ph_vap = prov.phase(temp, press*1e6, n, 'vap')
            ph_liq = prov.phase(temp, press*1e6, n, 'liq')

            rho_vapor.append(ph_vap.dens_mole)
            rho_liquid.append(ph_liq.dens_mole)

            norm_hres_vapor.append(-ph_vap.res_enthalpy_mole/(GAS_CONSTANT*temp))
            norm_hres_liquid.append(-ph_liq.res_enthalpy_mole/(GAS_CONSTANT*temp))

            log_phi_vapor_C2.append(ph_vap.log_phi[0])
            log_phi_vapor_C7.append(ph_vap.log_phi[1])

            log_phi_liquid_C2.append(ph_liq.log_phi[0])
            log_phi_liquid_C7.append(ph_liq.log_phi[1])

        plt.plot(rho_vapor, press_range)
        plt.plot(rho_liquid, press_range)
        plt.legend(['rho_vapor', 'rho_liquid'], loc='upper right')
        plt.title('Equimolar ethane-n_heptane mixture @ '+str(temp)+' K')
        plt.xlabel('Density (kg/m3)')
        plt.ylabel('Press (MPa)')
        plt.show()
        plt.close()

        plt.plot(press_range, norm_hres_vapor)
        plt.plot(press_range, norm_hres_liquid)
        plt.legend(['norm_hres_vapor', 'norm_hres_liquid'], loc='upper right')
        plt.title('Equimolar ethane-n_heptane mixture @ '+str(temp)+' K')
        plt.xlabel('Press (MPa)')
        plt.ylabel('-DH_res/RT')
        plt.show()
        plt.close()

        plt.plot(press_range, log_phi_vapor_C2)
        plt.plot(press_range, log_phi_vapor_C7)
        plt.plot(press_range, log_phi_liquid_C2)
        plt.plot(press_range, log_phi_liquid_C7)
        plt.legend(['log_phi_vapor_C2', 'log_phi_vapor_C7', 'log_phi_liquid_C2', 'log_phi_liquid_C7'], loc='upper right')
        plt.title('Equimolar ethane-n_heptane mixture @ ' + str(temp) + ' K')
        plt.xlabel('Press (MPa)')
        plt.ylabel('log_phi')
        plt.show()
        plt.close()


def rhos_vs_temp():
    components = pure(['ETHANE', 'N-HEPTANE'])
    prov = PengRobinson(components)
    comp = np.array([0.5, 0.5])
    temp_list = np.arange(300, 500)
    rho_vapor, rho_liquid = [], []
    press = 2.0*1e6

    for temp in temp_list:
        ph_vap = prov.phase(temp, press, comp, 'vap')
        ph_liq = prov.phase(temp, press, comp, 'liq')
        rho_vapor.append(ph_vap.dens_mole)
        rho_liquid.append(ph_liq.dens_mole)

    plt.plot(temp_list, rho_vapor)
    plt.plot(temp_list, rho_liquid)
    plt.legend(['rho_vapor', 'rho_liquid'], loc='upper right')
    plt.title('Equimolar ethane-n_heptane mixture')
    plt.xlabel('Temp. (K)')
    plt.ylabel('Density (kg/m3)')
    plt.show()
    plt.close()


def test_flash():
    components = pure(['ETHANE', 'N-BUTANE', 'N-HEPTANE'])
    prov = PengRobinson(components)
    temp, press = 300, 101325 * 5
    feed_comp = np.array([0.2, 0.3, 0.5])

    # Repeat flash
    temp = 406.315
    results = basic_flash_temp_press_2phase(prov, temp, press, feed_comp, prov.all_valid_components)
    print('temp:', temp, 'V/F:', results.phases, results.vap_frac_mole, results.vap.enthalpy_mole, results.enthalpy_mole)

    temp = 280.61
    results = basic_flash_temp_press_2phase(prov, temp, press, feed_comp, prov.all_valid_components)
    print('temp:', temp, 'V/F:', results.phases, results.vap_frac_mole)

    return results


def test_cavett_feed():
    components = pure(['NITROGEN', 'CARBON DIOXIDE', 'HYDROGEN SULFIDE',
                       'METHANE', 'ETHANE', 'PROPANE', 'N-BUTANE', 'ISOBUTANE',
                       'N-PENTANE', 'ISOPENTANE', 'N-HEXANE', 'N-HEPTANE',
                       'N-OCTANE', 'N-NONANE', 'N-DECANE',
                       'N-UNDECANE'])

    prov = PengRobinson(components)
    temp, press = 322.04, 0.8*1e6
    feed_comp = np.array([0.0131, 0.1816, 0.0124, 0.1096, 0.0876,
                          0.0838, 0.0563, 0.0221, 0.0413, 0.0289,
                          0.0645, 0.0953, 0.0675, 0.0610, 0.0304,
                          0.0444])

    result = None
    for temp in range(150, 500):
        results = basic_flash_temp_press_2phase(prov, temp, press, feed_comp, prov.all_valid_components)
        print('temp:', temp, 'V/F:', results.phases, results.vap_frac_mole)

    return results


def test_flash_press_vap_frac():
    components = pure(['ETHANE', 'N-BUTANE', 'N-HEPTANE'])
    prov = PengRobinson(components)
    temp, press = 300, 101325 * 12
    feed_comp = np.array([0.2, 0.3, 0.5])
    for vf in np.arange(0, 1.01, 0.01):
        r = flash_press_vap_frac_2phase(prov, press, vf, feed_comp)
        print('temp:', r.temp, 'press:', r.press, 'vap_frac', r.vap_frac_mole)

    components = pure(['NITROGEN', 'CARBON DIOXIDE', 'HYDROGEN SULFIDE',
                       'METHANE', 'ETHANE', 'PROPANE', 'N-BUTANE', 'ISOBUTANE',
                       'N-PENTANE', 'ISOPENTANE', 'N-HEXANE', 'N-HEPTANE',
                       'N-OCTANE', 'N-NONANE', 'N-DECANE',
                       'N-UNDECANE'])

    prov = PengRobinson(components)
    temp, press = 322.04, 0.5*1e6
    feed_comp = np.array([0.0131, 0.1816, 0.0124, 0.1096, 0.0876,
                          0.0838, 0.0563, 0.0221, 0.0413, 0.0289,
                          0.0645, 0.0953, 0.0675, 0.0610, 0.0304,
                          0.0444])
    for vf in np.arange(0, 1.01, 0.01):
        r = flash_press_vap_frac_2phase(prov, press, vf, feed_comp)
        print('temp:', r.temp, 'press:', r.press, 'vap_frac', r.vap_frac_mole)


def test_flash_temp_vap_frac():
    components = pure(['NITROGEN', 'CARBON DIOXIDE', 'HYDROGEN SULFIDE',
                       'METHANE', 'ETHANE', 'PROPANE', 'N-BUTANE', 'ISOBUTANE',
                       'N-PENTANE', 'ISOPENTANE', 'N-HEXANE', 'N-HEPTANE',
                       'N-OCTANE', 'N-NONANE', 'N-DECANE',
                       'N-UNDECANE'])

    prov = PengRobinson(components)
    temp, press = 316.5395439844929, 0.5*1e6
    feed_comp = np.array([0.0131, 0.1816, 0.0124, 0.1096, 0.0876,
                          0.0838, 0.0563, 0.0221, 0.0413, 0.0289,
                          0.0645, 0.0953, 0.0675, 0.0610, 0.0304,
                          0.0444])

    vf = 0.5
    # This flash can be a little more buggy ... we will fix later
    r = flash_temp_vap_frac_2phase(prov, temp, vf, feed_comp)
    print('temp:', r.temp, 'press:', r.press, 'vap_frac', r.vap_frac_mole)


def test_flash_press_prop():
    components = pure(['ETHANE', 'N-BUTANE', 'N-HEPTANE'])
    prov = PengRobinson(components)
    temp, press = 300, 101325 * 5
    feed_comp = np.array([0.2, 0.3, 0.5])

    # Repeat flash
    for temp in [50, 100, 150, 200, 300, 320, 330, 380, 400, 480, 500, 600, 800, 1000]:
        results = basic_flash_temp_press_2phase(prov, temp, press, feed_comp, prov.all_valid_components)
        print('temp:', temp, 'V/F:', results.vap_frac_mole)

        results = flash_press_prop_2phase(prov, press, 'entropy_mole', results.entropy_mole, 0, feed_comp, valid=None, previous=None)
        print('temp:', results.temp, 'V/F:', results.vap_frac_mole)
        print('---')

    # Repeat flash
    for temp in [50, 100, 150, 200, 300, 320, 330, 380, 400, 480, 500, 600, 800, 1000]:
        results = basic_flash_temp_press_2phase(prov, temp, press, feed_comp, prov.all_valid_components)
        print('temp:', temp, 'V/F:', results.vap_frac_mole)

        results = flash_press_prop_2phase(prov, press, 'enthalpy_mole', results.enthalpy_mole, 0, feed_comp, valid=None, previous=None)
        print('temp:', results.temp, 'V/F:', results.vap_frac_mole)
        print('---')


def test_flash_temp_prop():
    components = pure(['ETHANE', 'N-BUTANE', 'N-HEPTANE'])
    prov = PengRobinson(components)
    temp, press = 400, 101325 * 5
    feed_comp = np.array([0.2, 0.3, 0.5])

    results = basic_flash_temp_press_2phase(prov, temp, press, feed_comp, prov.all_valid_components)
    print('press:', press, 'V/F:', results.vap_frac_mole)

    results = flash_temp_prop_2phase(prov, temp, 'enthalpy_mole', results.enthalpy_mole, 0, feed_comp, valid=None, previous=None)
    print('press:', results.press, 'V/F:', results.vap_frac_mole)


def test_flash_prop_vap_frac():
    components = pure(['ETHANE', 'N-BUTANE', 'N-HEPTANE'])
    prov = PengRobinson(components)
    temp, press = 400, 101325 * 10
    feed_comp = np.array([0.2, 0.3, 0.5])

    results = basic_flash_temp_press_2phase(prov, temp, press, feed_comp, prov.all_valid_components)
    print('press:', press, 'temp:', temp, 'V/F:', results.vap_frac_mole)

    results = flash_prop_vap_frac_2phase(prov, 'enthalpy_mole', results.enthalpy_mole, 0, results.vap_frac_mole, feed_comp)
    print('press:', results.press, 'temp:', results.temp, 'V/F:', results.vap_frac_mole)


def test_all():
    press_vs_rho()
    rhos_vs_temp()
    test_flash()
    test_cavett_feed()
    test_flash_press_vap_frac()
    test_flash_temp_vap_frac()
    test_flash_press_prop()
    test_flash_temp_prop()
    test_flash_prop_vap_frac()
    pass

test_all()
