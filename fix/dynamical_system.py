import numpy as np
import sympy as sp


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

    xk = []
    dTdx = []

    def __init__(self, json):
        self.x0 = sp.Matrix(json['fixed'])
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
        self.xk = [sp.zeros(self.xdim, 1) for i in range(self.period+1)]
        self.xk[0] = self.x0
        self.dTdx = [sp.zeros(self.xdim, self.xdim)
                     for i in range(self.period)]
