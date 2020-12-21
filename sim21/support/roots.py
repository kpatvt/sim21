from numba import njit
import numpy as np
from math import acos, sqrt, cos, pi as PI


@njit(cache=True)
def solve_tridiagonal(a, b, c, d):
    """
    Solve a tri-diagonal matrix in the form: [ABC]X = D
    A is the sub diagonal
    B is the diagonal
    C is the super diagonal
    D is the solution vector

    A, B, C, D are expected to contain the same number of elements
    A[0] and C[n-1] are ignored.
    """
    n = len(b)
    p = np.empty(n)
    q = np.empty(n)
    x = np.empty(n)

    p[0] = c[0] / b[0]
    q[0] = d[0] / b[0]
    for j in range(1, n-1):
        den = (b[j] - a[j] * p[j - 1])
        p[j] = c[j] / den

    for j in range(1, n):
        den = (b[j] - a[j] * p[j - 1])
        q[j] = (d[j] - a[j] * q[j - 1]) / den

    x[n-1] = q[n-1]
    for j in range(n-2, -1, -1):
        x[j] = q[j] - p[j]*x[j+1]

    return x


@njit(cache=True)
def mid(a, b, c):
    """
    Returns the value that is the mid of a three element sequence a, b, c
    """
    return max(min(a, b), min(max(a, b), c))


@njit(cache=True)
def solve_cubic_reals(a, b, c, d):
    """
    Returns tuple containing real roots of quadratic in the form
    a*x^3 + b*x^2 + c*x + d = 0 in increasing order
    """
    a1 = (b / a)
    a2 = (c / a)
    a3 = (d / a)
    q = (a1 * a1 - 3.0 * a2) / 9.0
    r = (2.0 * a1 * a1 * a1 - 9.0 * a1 * a2 + 27.0 * a3) / 54.0
    disc = q ** 3 - r * r

    if disc >= 0:
        theta = acos(r / sqrt(q ** 3))
        sqrtq = sqrt(q)
        x_0 = -2 * sqrtq * cos(theta / 3.0) - (a1 / 3.0)
        x_1 = -2 * sqrtq * cos((theta + 2.0 * PI) / 3.0) - (a1 / 3.0)
        x_2 = -2 * sqrtq * cos((theta + 4.0 * PI) / 3.0) - (a1 / 3.0)

        return np.array([min(x_0, x_1, x_2), mid(x_0, x_1, x_2), max(x_0, x_1, x_2)])
    else:
        e = pow(sqrt(-disc) + abs(r), 1.0 / 3.0)
        if r > 0:
            e = -e
        x = (e + q / e) - a1 / 3.0
        # if polish:
        #     for i in range(3):
        #         x = x - (a*(x**3) + b*(x**2) + c*x + d)/(3*a*(x**2) + 2*b*x + c)

        return np.array([x])
