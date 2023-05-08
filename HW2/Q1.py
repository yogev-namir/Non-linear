import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return pow((x - 1), 3) + pow(1 - (x ** 2), -1)
def df(x):
    return 3*pow((x - 1), 2) + (2*x)*pow(1 - (x ** 2), -2)
def ddf(x):
    return (2 + 6*x*pow(1 - x**2, 3) + 6*(x**2) - 6*pow(1 - (x ** 2), 3)) / pow(1-x**2, 3)
def generic_bisect(f, df, l, u, eps, k):
    x = 0
    count = 1
    fv = [f(u)]
    while np.abs(u - l) >= eps and count <= k:
        x = (l + u) / 2
        if df(x) * df(u) > 0:
            u = x
        else:
            l = x
        count += 1
        fv.append(f(x))
    return x, fv


def generic_newton(f, df, ddf, x0, k):
    count = 1
    x = x0
    fv = [f(u)]
    while count <= k:
        x = x - df(x) / ddf(x)
        fv.append(f(x))
        count += 1
    return x, fv


def generic_hybrid(f, df, ddf, l, u, x0, eps, k):
    count = 1
    x = x0
    fv = [f(u)]
    while np.abs(u - l) >= eps and count <= k and np.abs(df(x)) > eps:
        x_newt = x - df(x) / ddf(x)
        if l < x_newt < u and np.abs(df(x_newt)) < 0.99 * np.abs(df(x)):
            x = x_newt
        else:
            x = (u + l) / 2
        if df(u)*df(x)>0:
            u=x
        else:
            l=x
        fv.append(f(x))
        count += 1
    return x,fv

def generic_gs(f, l, u, eps, k):
    count = 1
    fv = [f(u)]
    tao = (3-np.sqrt(5)) / 2
    x2 = l + tao*(u-l)
    x3 = l + (1-tao)*(u-l)
    while np.abs(u - l) >= eps and count<=k:
        if f(x2)<f(x3):
            u=x3
            x3=x2
            x2 = l + tao*(u-l)
        else:
            l=x2
            x2 = x3
            x3 = l + (1-tao)*(u-l)
        fv.append(f(u))
        count += 1
    return l, fv

def y_noise(fv):
    fv_noised = [v+3.08891254695156 for v in fv ]
    return fv_noised

eps = pow(10.0, -6)
k = 50
l = -0.999
u = 0.9

x1, fv1 = generic_bisect(f, df, l, u, eps, k)
x2, fv2 = generic_newton(f, df, ddf, u, k)
x3, fv3 = generic_hybrid(f, df, ddf, l, u, u, eps, k)
x4, fv4 = generic_gs(f, l, u, eps, k)

plt.semilogy(np.arange(len(fv1)), y_noise(fv1), color='red', label='bisect')
plt.semilogy(np.arange(len(fv2)), y_noise(fv2), color='blue', label='newton')
plt.semilogy(np.arange(len(fv3)), y_noise(fv3), color='green', label='hybrid')
plt.semilogy(np.arange(len(fv4)), y_noise(fv4), color='orange', label='golden-section')
plt.legend()
plt.show()