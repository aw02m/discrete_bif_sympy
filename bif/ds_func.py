import sympy as sp


def store_state(v, ds):
    ds.xk[0] = sp.Matrix(v[0:ds.xdim, 0])
    ds.xk[1] = ds.T.subs([(ds.sym_x, ds.xk[0]), (ds.sym_p, ds.params)])

    ds.dTldx = sp.Matrix(ds.dTdx.subs(
        [(ds.sym_x, ds.xk[0]), (ds.sym_p, ds.params)]))
    ds.dTldlambda = sp.Matrix(ds.dTdlambda.subs(
        [(ds.sym_x, ds.xk[0]), (ds.sym_p, ds.params)]))
    for i in range(ds.xdim):
        ds.dTldxdx[i] = sp.Matrix(ds.dTdxdx[i].subs(
            [(ds.sym_x, ds.xk[0]), (ds.sym_p, ds.params)]))
    ds.dTldxdlambda = sp.Matrix(
        ds.dTdxdlambda.subs([(ds.sym_x, ds.xk[0]), (ds.sym_p, ds.params)]))

    ds.chara = ds.dTldx + sp.eye(ds.xdim)


def newton_func(ds):
    return (ds.xk[1] - ds.xk[0]).col_join(sp.Matrix([ds.chara.det()]))


def newton_jac(ds):
    chi = []
    for i in range(ds.xdim):
        chi.append(det_derivative(ds.chara, ds.dTldxdx[i], ds))
    chi.append(det_derivative(ds.chara, ds.dTldxdlambda, ds))
    chi = sp.Matrix(chi).transpose()
    return (ds.dTldx - sp.eye(ds.xdim)).row_join(ds.dTldlambda).col_join(chi)


def det_derivative(A, dA, ds):
    temp = sp.zeros(ds.xdim, ds.xdim)
    ret = 0
    for i in range(0, ds.xdim):
        temp = A
        temp[0:ds.xdim, i] = dA.col(i)
        ret += temp.det()
    return ret
