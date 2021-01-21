import numpy as np
import matplotlib.pyplot as plt
from sim21.data.chemsep import pure
from sim21.provider.cubic import PengRobinson


def test_1():
    components = pure(['PROPANE', 'OXYGEN', 'NITROGEN', 'CARBON DIOXIDE', 'WATER'])
    n = np.array([0, 0.03125, 0.75, 0.09375, 0.125])
    prov = PengRobinson(components)
    press = 101325.0
    ig_h = []
    act_h = []
    temp = [i for i in range(300, 2500, 100)]
    ig_cp = []
    for t in temp:
        ph_vap = prov.phase(t, press, n, 'vap')
        ig_h.append(ph_vap.ig_enthalpy_mole*1e-3)
        act_h.append(ph_vap.enthalpy_mole*1e-3)
        ig_cp.append(ph_vap.ig_cp_mole*1e-3)

    plt.plot(temp, ig_h)
    plt.plot(temp, act_h)
    plt.legend(['ig_h', 'act_h'], loc='lower right')
    plt.xlabel('temp')
    plt.ylabel('enth')
    plt.show()

    plt.plot(temp, ig_cp)
    plt.show()


test_1()
