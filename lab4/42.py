import numpy as np


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


def FiniteDifference(n, xs, h, A_b1, A_c1, A_an, A_bn, b1, bn):
    A = np.zeros((n, 3))
    b = np.empty(n)
    A[0][1] = A_b1
    A[0][2] = A_c1
    A[n - 1][0] = A_an
    A[n - 1][1] = A_bn
    b[0] = b1
    b[n - 1] = bn
    for k in range(1, n - 1):
        A[k][0] = 1 - p(xs[k]) * h / 2
        A[k][1] = -2 + h ** 2 * q(xs[k])
        A[k][2] = 1 + p(xs[k]) * h / 2
        b[k] = h ** 2 * f(xs[k])

    ys = tridiagonalMatrixAlgorithm(A, b)
    return ys

# def p(x):
#     return (x - 3) / (x ** 2 - 1)
# def q(x):
#     return -1 / (x ** 2 - 1)
# def f(x):
#     return 0


def RungeError(ys: np.ndarray, ys2: np.ndarray, p):
    k = 2
    error = 0
    for i in range(ys.shape[0]):
        error = max(error, abs(ys2[i * 2] - ys[i]) / (k ** p - 1))
    return error


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
# h = 0.03125
h = 0.125
ya = 1.0 / np.e
yb = 1.0 / (2 * np.e**2)

print(f"Шаг: {h}\n")

xs = splitting(a, b, h)

ys = FiniteDifference(len(xs), xs, h, -1/h, 1/h, -1/h, (h + 1)/h, 0, -0.75)

for i in range(len(xs)):
    y = getTrueY(xs[i])

    print(f"xk = {np.round(xs[i], 5)}, y(xk) = {np.round(y, 5)}")

    error = abs(ys[i] - y)

    print(f"yk = {np.round(ys[i], 5)}, e = {np.round(error, 16)}\n")


# Считаем для шага в два раза короче, чтобы применить оценку Рунге
h2 = h / 2

xs2 = splitting(a, b, h2)
ys2 = FiniteDifference(int(b / h2) + 1, xs2, h2, -1/h2, 1/h2, -1/h2, (h2 + 1)/h2, 0, -0.75)
print("===================================================================")
print(f"Апостериорная оценка погрешности по Рунге: {RungeError(ys, ys2, 1)}")