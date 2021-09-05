import sys
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

    vp = ds.x0
    vp = vp.col_join(sp.Matrix([ds.params[ds.var_param]]))

    f = open('out', 'w')

    for p in range(ds.inc_iter):
        for i in range(ds.max_iter):
            ds_func.store_state(vp, ds)
            F = ds_func.newton_func(ds)
            J = ds_func.newton_jac(ds)
            delta = sp.Matrix(list(sp.linsolve((J, -F)))).transpose()
            vn = delta + vp
            norm = (vn - vp).norm()
            if (norm < ds.eps):
                print("************************************************")
                print(str(p) + ":converged : (iter = " + str(i + 1) + ")")
                print(ds.params[0:sp.shape(ds.params)[0]])
                print(vn[0:ds.xdim])
                print(ds.dTldx.eigenvals())
                print("************************************************")
                f.write(str(ds.params[0]) + " " + str(ds.params[1]) + "\n")
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
    f.close()


if __name__ == '__main__':
    main()
