import numpy as np
import sympy as sp


def store_state(v, ds):
    ds.xk[0] = sp.Matrix(v)
    ds.xk[1] = ds.T.subs([(ds.sym_x, ds.xk[0]), (ds.sym_p, ds.params)])
    ds.dTldx = sp.Matrix(ds.dTdx.subs(
        [(ds.sym_x, ds.xk[0]), (ds.sym_p, ds.params)]))
    ds.dTldx = sp.matrix2numpy(ds.dTldx).astype(np.float64)


def newton_func(ds):
    return sp.matrix2numpy(ds.xk[1] - ds.xk[0]).astype(np.float64)


def newton_jac(ds):
    return ds.dTldx - np.eye(ds.xdim)
