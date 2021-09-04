import sys
import numpy as np
import sympy as sp
import json
import dynamical_system
import ds_func


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} filename")
        sys.exit(0)
    fd = open(sys.argv[1], 'r')
    json_data = json.load(fd)
    ds = dynamical_system.DynamicalSystem(json_data)

    vp = sp.matrix2numpy(ds.x0).astype(np.float64)
    vp = np.vstack((vp, float(ds.params[ds.var_param])))

    for p in range(ds.inc_iter):
        for i in range(ds.max_iter):
            ds_func.store_state(vp, ds)
            F = ds_func.newton_func(ds)
            J = ds_func.newton_jac(ds)
            vn = np.linalg.solve(J, -F) + vp
            norm = np.linalg.norm(vn - vp)
            if (norm < ds.eps):
                print("converged")
                print(vn)
                vp = vn
                ds.params[ds.var_param] = vn[ds.xdim]
                break
            elif (norm > ds.explode):
                print("explode")
                sys.exit()
            vp = vn
            ds.params[ds.var_param] = vn[ds.xdim]
        else:
            print("iter over")
            print(F)
            exit()
        ds.params[ds.inc_param] += ds.delta_inc

if __name__ == '__main__':
    main()
