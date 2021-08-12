import numpy as np
import sympy as sp
import functools
import operator


def map(x, ds):
    # Henon map
    return sp.Matrix([x[1] + 1 - ds.params[0]*x[0]*x[0],
                     ds.params[1]*x[0]])


def store_state(v, ds):
    ds.xk[0] = v
    x = sp.MatrixSymbol('x', ds.xdim, 1)

    T = map(x, ds)
    for i in range(ds.period):
        ds.xk[i+1] = T.subs(x, ds.xk[i])

    dTdx = sp.derive_by_array([T[i] for i in range(ds.xdim)], [
                              x[i] for i in range(ds.xdim)])
    for i in range(ds.period):
        ds.dTdx[i] = sp.Matrix(dTdx.subs(x, ds.xk[i]))


def newton_func(ds):
    return ds.xk[ds.period] - ds.xk[0]


def newton_jac(ds):
    return functools.partial(functools.reduce, operator.mul)(list(reversed(ds.dTdx))) - sp.eye(ds.xdim)
