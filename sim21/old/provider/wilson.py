import numpy as np
import math
from sim21.old.provider.ideal import IdealVapLiq
from sim21.old.data import wilson


def wilson_model_log_gamma(given_temp, liq_vol_mole, x, ip, valid=None):
    n = len(x)
    log_gamma = np.zeros(n)
    if valid is None:
        valid = range(n)

    if len(valid) == 1:
        return log_gamma

    A = np.zeros((n, n))
    RT = 1.9858775 * given_temp
    for i in range(n):
        for j in range(n):
            A[i, j] = (liq_vol_mole[j]/liq_vol_mole[i])*math.exp(-ip[i, j] / RT)

    S = np.zeros(n)
    for i in range(n):
        S[i] = 0
        for j in range(n):
            S[i] += x[j]*A[i, j]

    for i in range(n):
        term = 0
        for k in range(n):
            term += x[k]*A[k, i]/S[k]

        log_gamma[i] = 1 - math.log(S[i]) - term

    return log_gamma


class IdealVapLiqWilson(IdealVapLiq):
    def __init__(self, components=None):
        super().__init__(components)
        if components:
            c = [i for i in components]
            self.setup_components(c)
        self._ip_u = None

    def setup_components(self, components, **kwargs):
        super().setup_components(components, **kwargs)
        try:
            self._ip_u = wilson.generate_ip([c.casn for c in components])
        except KeyError:
            raise NotImplementedError

    def log_gamma(self, temp, press, n, valid):
        liq_vol_mole = [c.liq_vol_mole(temp) for c in self.components]
        return wilson_model_log_gamma(temp, liq_vol_mole, n, self._ip_u, valid)


