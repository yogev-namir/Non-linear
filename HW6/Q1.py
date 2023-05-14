import numpy as np

def calc_t(f, xk, gk, dk, alpha, beta, s):
    t = s
    print(dk.shape)
    while f(xk-t*dk) >= (f(xk) + (alpha*t*(gk(xk).T.dot(dk)))):
        t = beta*t
    return t

def analytic_center(A,b,x0):
    try:
        if np.any((A.dot(x0)-b)>0):
            raise ValueError("x0 isn't in the interior of P")
        f = lambda x: -sum(np.log(b-(A.dot(x))))
        gf = lambda x: np.matmul(A.T, 1/(b-A.dot(x)))
        hf = lambda x: np.matmul(A.T, np.matmul(np.diag(1/((b-A.dot(x))**2)), A))
        xk = x0
        xk_hist = [x0]
        fs = [f(x0)]
        dk = None

        while np.linalg.norm(gf(xk))>=10**(-6):
            dk = np.linalg.solve(hf(xk),-gf(xk))
            tk = calc_t(f,xk,gf,dk,alpha=0.25,beta=0.5,s=2)
            print('test')
            xk = xk + tk*dk
            xk_hist.append(xk)
            fs.append(f(xk))

        return xk,fs

    except ValueError as err:
        print(err)


A = np.array([[2,10],
             [1,0],
             [-1,3],
             [-1,-1]])
b = np.array([1,0,2,2])
x0 = np.array([-1.99,0])

xk,fs = analytic_center(A,b,x0)