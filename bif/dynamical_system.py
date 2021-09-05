import sympy as sp


def map(x, p):
    ret = sp.Matrix([(p[0] * x[0] + p[1] * x[1]) /
                     (1 + sp.exp(-p[4] *
                                 (p[0] * x[0] + p[1] * x[1]))),
                     (p[2] * x[0] + p[3] * x[1]) /
                     (1 + sp.exp(-p[4] *
                                 (p[2] * x[0] + p[3] * x[1])))
                     ])
    return ret


class DynamicalSystem:
    x0 = []
    params = []
    period = 0
    xdim = 0
    inc_param = 0
    var_param = 0
    delta_inc = 0.0
    inc_iter = 0
    max_iter = 0
    dif_strip = 0.0
    eps = 0.0
    explode = 0.0

    vk = []
    xk = []
    dTdx = []
    dTdlambda = []
    dTdxdx = []
    dTdxdlambda = []
    dTldx = []
    dTldlambda = []
    dTldxdx = []
    dTldxdlambda = []

    sym_x = 0
    sym_p = 0
    T = 0
    chara = 0

    def __init__(self, json):
        self.x0 = sp.Matrix(json['x0'])
        self.params = sp.Matrix(json['params'])
        self.period = json['period']
        self.xdim = len(self.x0)
        self.inc_param = json['inc_param']
        self.var_param = json['var_param']
        self.delta_inc = json['delta_inc']
        self.inc_iter = json['inc_iter']
        self.max_iter = json['max_iter']
        self.dif_strip = json['dif_strip']
        self.eps = json['eps']
        self.explode = json['explode']
        self.vk = [sp.zeros(self.xdim + 1, 1) for i in range(self.period + 1)]
        self.xk = [sp.zeros(self.xdim, 1) for i in range(self.period + 1)]
        self.xk[0] = self.x0
        self.dTdx = [sp.zeros(self.xdim, self.xdim)
                     for i in range(self.period)]
        self.dTdlambda = [sp.zeros(self.xdim, 1)
                          for i in range(self.period)]
        self.dTdxdx = [[sp.zeros(self.xdim, self.xdim) for j in range(
            self.xdim)] for i in range(self.period)]
        self.dTdxdlambda = [sp.zeros(self.xdim, self.xdim)
                            for i in range(self.period)]
        self.dTldxdx = [
            sp.zeros(
                self.xdim,
                self.xdim) for i in range(
                self.xdim)]

        self.sym_x = sp.MatrixSymbol('x', self.xdim, 1)
        self.sym_p = sp.MatrixSymbol('p', sp.shape(self.params)[0], 1)
        self.T = map(map(self.sym_x, self.sym_p), self.sym_p)
        self.dTdx = sp.derive_by_array([self.T[i] for i in range(self.xdim)], [
            self.sym_x[i] for i in range(self.xdim)]).transpose()
        self.dTdlambda = sp.diff(self.T, self.sym_p[self.var_param])
        for i in range(self.xdim):
            self.dTdxdx[i] = sp.diff(self.dTdx, self.sym_x[i])
        self.dTdxdlambda = sp.diff(self.dTdx, self.sym_p[self.var_param])
