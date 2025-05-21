import numpy as np
from copy import copy

'''
    Из 4.1
    -------------------------------------------------------------------------------------------------------
'''

def splitting(x0, xk, h):
    xs = []
    x = x0
    while x < xk:
        xs.append(x)
        x += h
    xs.append(xk)
    return xs


def tridiagonalMatrixAlgorithm(A, b):
    n = len(b)

    # Вычисляем прогоночные коэффициенты
    P = np.empty((n))
    P[0] = -A[0][2] / A[0][1]
    Q = np.empty((n))
    Q[0] = b[0] / A[0][1]
    for i in range(n):
        P[i] = (-A[i][2]) / (A[i][1] + A[i][0] * P[i - 1])
        Q[i] = (b[i] - A[i][0] * Q[i - 1]) / (A[i][1] + A[i][0] * P[i - 1])

    # Обратный ход
    x = np.empty((n))
    x[n - 1] = Q[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = P[i] * x[i + 1] + Q[i]

    return x

def RungeError(ys: np.ndarray, ys2: np.ndarray, p):
    k = 2
    error = 0
    for i in range(ys.shape[0]):
        error = max(error, abs(ys2[i * 2] - ys[i]) / (k ** p - 1))
    return error

'''
    Из 4.1
    -------------------------------------------------------------------------------------------------------
'''


def FiniteDifference(N, xs, h):
    A = np.zeros((N-1, 3))
    b = np.empty(N-1)

    A[0][1]   = -2 * h**2 * q(xs[1])
    A[0][2]   = 1 + (h * p(xs[1])) / 2
    b[0]      = h**2 * f(xs[1]) - (1 - (p(xs[1]) * h) / 2) * ya

    A[-1][0] = 1 - p(xs[N-1]) * h / 2
    A[-1][1] = -2 + h**2 * q(xs[N-1])
    b[-1]    = h**2 * f(xs[N-1]) - (1 + (p(xs[N-1]) * h) / 2) * yb

    for k in range(2, N-2 + 1):
        A[k-1][0] = 1 - p(xs[k]) * h / 2
        A[k-1][1] = -2 + h**2 * q(xs[k])
        A[k-1][2] = 1 + p(xs[k]) * h / 2
        b[k-1]    = h**2 * f(xs[k])

    # print(f'tridiag: {tridiagonalMatrixAlgorithm(A, b)}')
    ys = tridiagonalMatrixAlgorithm(A, b)
    ys = np.concatenate((np.array([ya]), ys, np.array([yb])))
    # print(f'Returning {ys}')
    return ys

def p(x):
    return 2.0 / x
def q(x):
    return -1
def f(x):
    return 0

def getTrueY(x):
    return np.e ** (-x) / x

a = 1
b = 2
N = 8
h = abs(b - a) / N
ya = 1.0 / np.e
yb = 1.0 / (2 * np.e**2)

print(f"N = {N}\n")
print(f"Шаг: {h}\n")

xs = splitting(a, b, h)
ys = FiniteDifference(N, xs, h)

for i in range(len(xs)):
    y = getTrueY(xs[i])

    print(f"xk = {np.round(xs[i], 5)}, y(xk) = {np.round(y, 5)}")

    error = abs(ys[i] - y)

    print(f"yk = {np.round(ys[i], 5)}, e = {np.round(error, 16)}\n")


# Считаем для шага в два раза короче, чтобы применить оценку Рунге
h2 = h / 2

xs2 = splitting(a, b, h2)
ys2 = FiniteDifference(len(xs2), xs2, h2)
print("===================================================================")
print(f"Апостериорная оценка погрешности по Рунге: {RungeError(ys, ys2, 1)}")