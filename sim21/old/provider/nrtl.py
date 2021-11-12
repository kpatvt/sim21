import numpy as np
import math
from sim21.old.data import nrtl
from sim21.old.provider.ideal import IdealVapLiq


def nrtl_model_log_gamma(given_temp, liq_vol_mole, x, ip_g, ip_alpha, valid=None):
    n = len(x)
    log_gamma = np.zeros(n)
    if valid is None:
        valid = range(n)

    # If there is only one component, we return nothing....
    if len(valid) == 1:
        return log_gamma

    tau = np.zeros((n, n))
    G = np.zeros((n, n))
    RT = 1.9858775 * given_temp
    S = np.zeros(n)
    C = np.zeros(n)

    for i in valid:
        for j in valid:
            tau[i, j] = (ip_g[i, j])/RT
            G[i, j] = math.exp(-ip_alpha[i, j]*tau[i, j])

    for i in valid:
        for j in valid:
            S[i] += x[j]*G[j, i]
            C[i] += x[j]*G[j, i]*tau[j, i]

    for i in valid:
        log_gamma[i] = C[i]/S[i]
        for k in range(n):
            log_gamma[i] += x[k]*G[i, k]*(tau[i, k] - C[k]/S[k])/S[k]

    return log_gamma


class IdealVapLiqNRTL(IdealVapLiq):
    def __init__(self, components=None):
        super().__init__(components)
        if components:
            c = [i for i in components]
            self.setup_components(c)

        self._ip_g = None
        self._ip_alpha = None

    def setup_components(self, components, **kwargs):
        super().setup_components(components, **kwargs)
        try:
            self._ip_g, self._ip_alpha = nrtl.generate_ip([c.casn for c in components])
        except KeyError:
            raise NotImplementedError

    def log_gamma(self, temp, press, n, valid):
        return nrtl_model_log_gamma(temp, np.zeros_like(n), n, self._ip_g, self._ip_alpha, valid)


