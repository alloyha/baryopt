from __future__ import annotations

from numpy import exp, zeros
from numpy.random import normal

DEFAULT_NU = 3
DEFAULT_LAMBDA = 1
DEFAULT_SIGMA = 0.5
DEFAULT_ZETA = 0
DEFAULT_ITERANTION_COUNT = 1000


def drecexpbary(
    oracle,
    x0,
    nu=DEFAULT_NU,
    sigma=DEFAULT_SIGMA,
    zeta=DEFAULT_ZETA,
    lambda_=DEFAULT_LAMBDA,
    iterations=DEFAULT_ITERANTION_COUNT,
):
    """
    Recursive barycenter algorithm for direct optimization

    In:
      - oracle     [function]  : Oracle map e.g. lambda x: numpy.norm(x, 2)
      - x0         [np.array]  : Initial query values
      - nu         [double]    : positive value (Caution due overflow)
      - sigma      [double]    : Std deviation of normal distribution
      - zeta       [double]    : scaler for mean of normal distribution
      - lambda     [double]    : Forgetting factor between 0 and 1
      - iterations [integer]   : Maximum number of iterations

    Out:
       - xhat      [np.array]  : barycenter position
    """

    def bexp_fun(x):
        return exp(-nu * oracle(x))

    # Initialization
    xhat_1 = x0
    m_1 = 0
    card_x = len(x0)

    deltax_1 = zeros((card_x, 1))
    solution_is_found = False

    # Optimization loop
    i = 1
    while not solution_is_found:
        z = normal(zeta * deltax_1, sigma).T

        x = xhat_1 + z
        e_i = bexp_fun(x)
        m = lambda_ * m_1 + e_i
        xhat = (1 / m) * (lambda_ * m_1 * xhat_1 + x * e_i)

        solution_is_found = i >= iterations

        # Update previous variables
        m_1 = m
        xhat_1 = xhat

        i = i + 1

    return xhat

