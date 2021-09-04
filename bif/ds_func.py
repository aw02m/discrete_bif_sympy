import numpy as np
import sympy as sp


def store_state(v, ds):
    ds.xk[0] = sp.Matrix(v[0:ds.xdim, 0])
    # x = sp.MatrixSymbol('x', ds.xdim, 1)
    # p = sp.MatrixSymbol('p', sp.shape(ds.params)[0], 1)

    ds.xk[1] = ds.T.subs([(ds.sym_x, ds.xk[0]), (ds.sym_p, ds.params)])

    ds.dTldx = sp.Matrix(ds.dTdx.subs(
        [(ds.sym_x, ds.xk[0]), (ds.sym_p, ds.params)]))
    ds.dTldx = sp.matrix2numpy(ds.dTldx).astype(np.float64)

    ds.dTldlambda = sp.Matrix(ds.dTdlambda.subs(
        [(ds.sym_x, ds.xk[0]), (ds.sym_p, ds.params)]))
    ds.dTldlambda = sp.matrix2numpy(ds.dTldlambda).astype(np.float64)

    for i in range(ds.xdim):
        ds.dTldxdx[i] = sp.Matrix(ds.dTdxdx[i].subs(
            [(ds.sym_x, ds.xk[0]), (ds.sym_p, ds.params)]))
        ds.dTldxdx[i] = sp.matrix2numpy(ds.dTldxdx[i]).astype(np.float64)
        # print(ds.dTldxdx[0])
        # exit()

    ds.dTldxdlambda = sp.Matrix(
        ds.dTdxdlambda.subs([(ds.sym_x, ds.xk[0]), (ds.sym_p, ds.params)]))
    ds.dTldxdlambda = sp.matrix2numpy(ds.dTldxdlambda).astype(np.float64)

    ds.chara = ds.dTldx + np.eye(ds.xdim)


def newton_func(ds):
    ret = np.zeros((ds.xdim + 1, 1))
    ret[0:ds.xdim, 0] = sp.matrix2numpy(
        ds.xk[1] - ds.xk[0]).astype(np.float64).flatten()
    ret[ds.xdim, 0] = np.linalg.det(ds.chara)
    return ret


def newton_jac(ds):
    ret = np.zeros((ds.xdim + 1, ds.xdim + 1))
    ret[0:ds.xdim, 0:ds.xdim] = ds.dTldx - np.eye(ds.xdim)
    ret[0:ds.xdim, ds.xdim] = ds.dTldlambda.flatten()
    for i in range(ds.xdim):
        ret[ds.xdim, i] = det_derivative(ds.chara, ds.dTldxdx[i], ds)
    ret[ds.xdim, ds.xdim] = det_derivative(ds.chara, ds.dTldxdlambda, ds)
    return ret


def det_derivative(A, dA, ds):
    temp = np.zeros((ds.xdim, ds.xdim))
    ret = 0
    for i in range(0, ds.xdim):
        temp = A
        temp[0:ds.xdim, i] = dA[0:ds.xdim, i]
        ret += np.linalg.det(temp)
    return ret
