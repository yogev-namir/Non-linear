import numpy as np
from pytictoc import TicToc


def hybrid_newton(f, gf, hf, lsearch, x0, eps):
    clock = TicToc()
    clock.tic()
    xk = x0
    fs = [x0]  # f(x0)?
    ts = [0]
    gs = [gf(x0)]
    newton = []
    while np.linalg.norm(gf(xk)) >= eps:
        print("iteration started")
        if is_poitive_definite(hf(xk)):
            dk = (np.linalg.solve(hf(xk), -gf(xk)), 'newton')
            newton.append(1)
        else:
            dk = (-gf(xk), 'grad')
            newton.append(0)
        grad_xk = gf(xk)
        tk = lsearch(xk, grad_xk, dk)
        xk = xk + tk * dk[0]
        fs.append(f(xk))
        gs.append(np.linalg.norm(gf(xk)))
        ts.append(clock.tocvalue())
    return xk, fs, gs, ts, newton


def calc_t(f, xk, gk, alpha, beta, s):
    t = s
    # print(b - (A.dot(xk - t * dk)))
    while f(xk - t * gk) >= (f(xk) - (alpha * t * (np.linalg.norm(gk) ** 2))):
        t = beta * t
    return t


def is_poitive_definite(A):
    try:
        np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        return False
    return True


def hybrid_back(f, alpha, beta, s):
    return lambda xk, gk, direction: calc_t(f, xk, gk, alpha, beta, s) if direction[1] == 'grad' else 1


def main():
    f = lambda x: x[0] ** 4 + x[1] ** 4 - 36 * x[0] * x[1]
    gf = lambda x: np.array([4 * (x[0] ** 3) - 36 * x[1], 4 * (x[1] ** 3) - 36 * x[0]])
    hf = lambda x: np.array([[12 * x[0] ** 2 - 36, -36], [-36, 12 * x[1] ** 2 - 36]])
    lsearch = hybrid_back(f, alpha=0.25, beta=0.5, s=1)
    x0 = np.array([200, 0])
    eps = pow(10, -6)
    x, fs, gs, ts, newton = hybrid_newton(f, gf, hf, lsearch, x0, eps)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/

if __name__ == '__main__':
    main()
