import numpy as np
from CoolProp.CoolProp import PropsSI
from scipy.optimize import minimize


def nrtl(alpha, tau, t, x):
    '''
    Calculate activity coefficients using NRTL equation.

    Parameters
    ----------
    alpha : ndarray(n,n)
        Array of NRTL nonrandomness parameters. n = the number of
        components in the system.
    tau : ndarray(n,n)
        Array of NRTL tau parameters. tau[i,i] should be set to 0.
    t : float
        Temperature (K)
    x : ndarray(n,)
        Mole fraction of each component

    Returns
    -------
    gamma : ndarray(n,)
        Activity coefficient of each component
    '''
    G = np.exp(-alpha * tau)
    ncomp = x.shape[0]
    gamma = np.zeros_like(x)
    summ = 0

    for i in range(ncomp):
        summ = 0
        for j in range(ncomp):
            summ += x[j] * G[i, j] / np.sum(G[:, j] * x) * (tau[i, j] -
                                                            (np.sum(x * tau[:, j] * G[:, j]) / np.sum(G[:, j] * x)))
        gamma[i] = np.sum(tau[:, i] * G[:, i] * x) / np.sum(G[:, i] * x) + summ

    return np.exp(gamma)


def bubblePfit_nrtl(p_guess, xv_guess, Psat, alpha, tau, t, x):
    '''
    Minimize this function to calculate the bubble point pressure of a
    mixture using NRTL.
    '''
    gamma_l = nrtl(alpha, tau, t, x)

    itr = 0
    dif = 10000.
    xv = np.copy(xv_guess)
    xv_old = np.zeros_like(xv)
    while (dif > 1e-9) and (itr < 100):
        xv_old[:] = xv
        xv = gamma_l * x * Psat / p_guess
        xv = xv / np.sum(xv)
        dif = np.sum(abs(xv - xv_old))
        itr += 1

    error = (np.sum((gamma_l * x * Psat - p_guess * xv) ** 2) / float(x.shape[0])) * 100
    return error


def bubbleP_nrtl(p_guess, xv_guess, Psat, alpha, tau, t, x):
    ''' Calculate the bubble point of a mixture using NRTL. '''
    result = minimize(bubblePfit_nrtl, p_guess, args=(xv_guess, Psat, alpha, tau, t, x), tol=1e-10,
                      method='Nelder-Mead', options={'maxiter': 100})
    bubP = result.x

    # Calculated the vapor composition
    gamma_l = nrtl(alpha, tau, t, x)

    itr = 0
    dif = 10000.
    xv = np.copy(xv_guess)
    xv_old = np.zeros_like(xv)
    while (dif > 1e-9) and (itr < 100):
        xv_old[:] = xv
        xv = gamma_l * x * Psat / bubP
        xv = xv / np.sum(xv)
        dif = np.sum(abs(xv - xv_old))
        itr += 1

    return [bubP, xv]


def dippr_acid(T):
    '''
    Calculate the vapor pressure of acetic acid using DIPPR correlation

    Parameters
    ----------
    T : float
        Temperature (K)

    Returns
    -------
    P : float
        Vapor pressure of acetic acid (Pa)
    '''
    c = np.asarray([53.27, -6304.5, -4.2985, 8.8865e-018, 6])

    P = np.exp(c[0] + c[1] / T + c[2] * np.log(T) + c[3] * T ** c[4])
    return P


def test_nrtl():
    ''' Test whether the NRTL function returns correct results. '''
    # Binary mixture: water-acetic acid
    print('\n##########  Test with water-acetic acid mixture  ##########')
    # 0 = water, 1 = acetic acid
    alpha = np.asarray([[0, 0.3],
                        [0.3, 0]])
    taubase = np.asarray([[0, 3.3293],
                          [-1.9763, 0]])
    tauT = np.asarray([[0, -723.888],
                       [609.889, 0]])

    xl = np.asarray([0.9898662364, 0.0101337636])
    t = 403.574
    p_ref = 273722.  # source: Othmer, D. F.; Silvis, S. J.; Spiel, A. Ind. Eng. Chem., 1952, 44, 1864-72 Composition of vapors from boiling binary solutions pressure equilibrium still for studying water - acetic acid system
    xv_ref = np.asarray([0.9923666645, 0.0076333355])
    Psat = np.zeros_like(xl)
    Psat[0] = PropsSI('P', 'T', t, 'Q', 0, 'Water')
    Psat[1] = dippr_acid(t)
    tau = taubase + tauT / t
    result = bubbleP_nrtl(p_ref, xv_ref, Psat, alpha, tau, t, xl)
    calc = result[0]
    xv = result[1]
    print('----- Bubble point pressure at %s K -----' % t)
    print('    Liquid composition:', xl, '\n')
    print('    p_reference pressure:', p_ref, 'Pa')
    print('    NRTL pressure:', calc, 'Pa')
    print('    NRTL pressure (from Aspen):', 274325, 'Pa')
    print('    Relative deviation:', (calc - p_ref) / p_ref * 100, '%\n')
    print('    Vapor composition (p_reference):', xv_ref)
    print('    Vapor composition (NRTL):', xv)
    print('    Vapor composition (NRTL from Aspen):', [0.985809, 0.0141907], '\n')

    xl = np.asarray([0.2691800943, 0.7308199057])
    t = 372.774
    p_ref = 74463.  # source: Freeman, J. R.; Wilson, G. M. AIChE Symp. Ser., 1985, 81, 14-25 High temperature vapor-liquid equilibrium measurements on acetic acid/water mixtures
    xv_ref = np.asarray([0.3878269411, 0.6121730589])
    Psat[0] = PropsSI('P', 'T', t, 'Q', 0, 'Water')
    Psat[1] = dippr_acid(t)
    tau = taubase + tauT / t
    result = bubbleP_nrtl(p_ref, xv_ref, Psat, alpha, tau, t, xl)
    calc = result[0]
    xv = result[1]
    print('----- Bubble point pressure at %s K -----' % t)
    print('    Liquid composition:', xl, '\n')
    print('    p_reference pressure:', p_ref, 'Pa')
    print('    NRTL pressure:', calc, 'Pa')
    print('    NRTL pressure (from Aspen):', 82178.2, 'Pa')
    print('    Relative deviation:', (calc - p_ref) / p_ref * 100, '%\n')
    print('    Vapor composition (p_reference):', xv_ref)
    print('    Vapor composition (NRTL):', xv)
    print('    Vapor composition (NRTL from Aspen):', [0.488574, 0.511427], '\n')

    return None

test_nrtl()
