"""Main module."""
from __future__ import annotations

from numpy import exp, zeros
from functools import reduce

DEFAULT_NU = 3
DEFAULT_LAMBDA = 1
DEFAULT_SIGMA = 0.5
DEFAULT_ZETA = 0
DEFAULT_ITERANTION_COUNT = 1000

def dbatchbary(
    oracle,
    xs,
    nu=DEFAULT_NU,
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

    n = len(xs[0])
    size_x = (n, 1)

    def bexp_fun(x):
        return exp(-nu * oracle(x))

    def prod_func(elems):
        return elems[0] * elems[1]

    def sum_func(acc, a):
        return acc + a

    coord_value_iter = zip(map(bexp_fun, xs), xs)
    num = reduce(sum_func, map(prod_func, coord_value_iter), zeros(size_x).T)

    den = reduce(sum_func, map(bexp_fun, xs), 0)

    return num / den