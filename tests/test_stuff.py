import math

# import numpy as np
# import matplotlib.pyplot as plt
# from sim21.data.chemsep import pure
# from sim21.provider.cubic import PengRobinson
#
#
# def test_1():
#     components = pure(['PROPANE', 'OXYGEN', 'NITROGEN', 'CARBON DIOXIDE', 'WATER'])
#     n = np.array([0, 0.03125, 0.75, 0.09375, 0.125])
#     prov = PengRobinson(components)
#     press = 101325.0
#     ig_h = []
#     act_h = []
#     temp = [i for i in range(300, 2500, 100)]
#     ig_cp = []
#     for t in temp:
#         ph_vap = prov.phase(t, press, n, 'vap')
#         ig_h.append(ph_vap.ig_enthalpy_mole*1e-3)
#         act_h.append(ph_vap.enthalpy_mole*1e-3)
#         ig_cp.append(ph_vap.ig_cp_mole*1e-3)
#
#     plt.plot(temp, ig_h)
#     plt.plot(temp, act_h)
#     plt.legend(['ig_h', 'act_h'], loc='lower right')
#     plt.xlabel('temp')
#     plt.ylabel('enth')
#     plt.show()
#
#     plt.plot(temp, ig_cp)
#     plt.show()
#

# test_1()


# Corrected coefficients from the 2009 update
IG_n = [0, -8.3204464837497, 6.6832105275932, 3.00632, 0.012436, 0.97315, 1.27950, 0.96956, 0.24873]
IG_gamma = [0, 0, 0, 0, 1.28728967, 3.53734222, 7.74073708, 9.24437796, 27.5075105]


def ig_props(press, temp):
    # Assumes press is in MPa
    gc = 18.015268 * 0.46151805 * 1e3  # m3-Pa/kmol-K
    vol_mole = gc*temp/(press*1e6)     # m3/kmol
    rho_mole = 1/vol_mole              # kmol/m3
    rho_mass = rho_mole * 18.015268    # kmol/m3 * MW kg/kmol
    delta = rho_mass/322.0
    tau = 647.096/temp
    t1_ig_phi = 0.0
    t1_ig_phi_tau = 0.0
    t1_ig_phi_tau_tau = 0.0

    for i in range(4, 9):
        t1_ig_phi += IG_n[i] * math.log(1.0 - math.exp(-IG_gamma[i] * tau))
        t1_ig_phi_tau += IG_n[i] * IG_gamma[i] * (pow(1.0 - math.exp(-IG_gamma[i] * tau), -1.0) - 1.0)
        t1_ig_phi_tau_tau += IG_n[i] * (IG_gamma[i] ** 2) * math.exp(-IG_gamma[i] * tau) * pow(1.0 - math.exp(-IG_gamma[i] * tau), -2.0)

    ig_phi = math.log(delta) + IG_n[1] + IG_n[2] * tau + IG_n[3] * math.log(tau) + t1_ig_phi
    ig_phi_delta = 1.0 / delta
    ig_phi_delta_delta = -1.0 / (delta * delta)
    ig_phi_tau = IG_n[2] + IG_n[3] / tau + t1_ig_phi_tau
    ig_phi_tau_tau = -IG_n[3] / (tau * tau) - t1_ig_phi_tau_tau
    ig_phi_delta_tau = 0.0

    rt = 0.46151805*temp
    h = rt*(1 + tau*ig_phi_tau)
    s = (-0.46151805)*((tau*ig_phi_tau) - ig_phi)
    g = h - temp*s
    u = rt*(tau*ig_phi_tau)
    a = u - temp*s
    cv = 0.46151805*(-(tau**2)*(ig_phi_tau_tau))
    cp = cv + 0.46151805

    # print('h:', h)
    # print('s:', s)
    # print('g:', g)
    # print('u:', u)
    # print('a:', a)
    # print('cv:', cv)
    # print('cp:', cp)
    return h, s, g, u, a, cv, cp


ig_props(0.1, 500)
