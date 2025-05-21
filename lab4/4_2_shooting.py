import numpy as np
from random import randint

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


# Рунге-Кутт 4го порядка
p = 4
as_ = [0, 0.5, 0.5, 1]
bs = [[0.5], [0, 0.5], [0, 0, 1]]
cs = [1/6, 1/3, 1/3, 1/6]

def getKs(x: float, y: np.ndarray, h):
    dim = y.shape[0]
    Ks = np.empty((p, dim))

    for i in range(p):
        newX = x + as_[i] * h
        newY = np.copy(y)
        for j in range(i):
            newY += bs[i - 1][j] * Ks[j]

        K = h * f(newX, newY)
        Ks[i] = K

    return Ks

def getDeltaY(x: float, y: np.ndarray, h):
    Ks = getKs(x, y, h)
    dim = Ks.shape[1]
    sum_ = np.zeros(dim)
    for i in range(p):
        sum_ += cs[i] * Ks[i]
    return sum_

def RungeKutta(xs: list, y0: np.ndarray, h):
    N = len(xs) - 1
    dim = y0.shape[0]
    ys = np.empty((N + 1, dim))
    ys[0] = y0

    for k in range(1, N + 1):
        ys[k] = ys[k - 1] + getDeltaY(xs[k - 1], ys[k - 1], h)

    return ys

def RungeError(ys: np.ndarray, ys2: np.ndarray, p):
    k = 2
    error = 0
    for i in range(ys.shape[0]):
        error = max(error, abs(ys2[i * 2][0] - ys[i][0]) / (k ** p - 1))
    return error

'''
    Из 4.1
    -------------------------------------------------------------------------------------------------------
'''


'''
    Численные методы решение краевой задачи для ОДУ

    Пример:
    y'' = f(x, y, y')
    И заданы условия на концах отрезка:

    y(a) = y_0
    y(b) = y_1

    Следует найти такое решение  на этом отрезке, которое  принимает на концах
    отрезка значения y_0, y_1. Если функция f(x, y, y') линейна по аргументам y, y', то задача 
     - линейная краевая задача, в противном случае – нелинейная.


    Метод стрельбы:
    Пусть надо решить краевую задачу на отрезке [a, b].
    Вместо исходной задачи формулируется задача Коши с уравнением 
    y'' = f(x, y, y') и с начальными условиями
    y(a) = y_0
    y'(b) = eta

    Где eta - некоторое значение тангенса угла наклона касательной
    к решению в точке x=a
'''

def Shooting(xs, y0, h, eps):
    '''
        Если просто:
        Суть этого метода в том, что мы пытаемся
        угадать недостающее условие y'(a) = eta.

        Для каждой догадки мы сверяемся с условием
        y(b) = CONST. И ищем такое eta, что ошибка между
        y(b) = CONST, которое дано и тем, которое получилось
        пр y'(a) = eta будет минимальным.

        каким-то образом решаем Задачу Коши с угаданным y'(a)
    '''
    eta0 = randint(-2166, 2166)
    eta1 = randint(-2166, 2166)

    ys0 = RungeKutta(xs, np.array([y0, eta0]), h)
    ys1 = RungeKutta(xs, np.array([y0, eta1]), h)
    # F - разность между "угаданным" значением на конце и действительным
    F0 = ys0[-1][0] - y1
    F1 = ys1[-1][0] - y1

    iter = 1
    while True:
        # Значения для текущего шага

        # какой метод????
        eta = eta1 - (eta1 - eta0) / (F1 - F0) * F1
        ys = RungeKutta(xs, np.array([y0, eta]), h)

        F0 = F1
        F1 = ys[-1][0] - y1

        if abs(F1) < eps:
            return ys, iter, eta
        
        iter += 1
        eta0 = eta1
        eta1 = eta


def f(x: float, y: np.ndarray):
    return np.array([
        y[1],
        (x * y[0] - 2*y[1]) / x
    ])

def getTrueY(x):
    return np.e ** (-x) / x

a = 1
b = 2
y0 = 1.0 / np.e
y1 = 1.0 / (2 * np.e**2)
h = 0.125
eps = 1e-9

print(f"Шаг: {h}")
print(f"Точность: {eps}")

xs = splitting(a, b, h)                # -0.73576 - значение производной f в точке 1 
ysRungeKutta = RungeKutta(xs, np.array([y0, -0.73576]), h)
ysShooting, iterShooting, eta = Shooting(xs, y0, h, eps)

print(f"Итераций в стрельбе: {iterShooting}, Вычисленная y'(0) = {eta}")

for i in range(len(xs)):
    y = getTrueY(xs[i])

    print(f"xk = {np.round(xs[i], 5)}, y(xk) = {np.round(y, 5)}")

    errorRungeKutta = abs(ysRungeKutta[i][0] - y)
    errorShooting = abs(ysShooting[i][0] - y)

    print(f"\tРунге-Кутт: yk = {np.round(ysRungeKutta[i][0], 5)}, e = {np.round(errorRungeKutta, 16)}")
    print(f"\tСтрельба:   yk = {np.round(ysShooting[i][0], 5)}, e = {np.round(errorShooting, 16)}\n")


# Считаем для шага в два раза короче, чтобы применить оценку Рунге
h2 = h / 2

xs2 = splitting(a, b, h2)
ysShooting2, iterShooting, eta = Shooting(xs2, y0, h2, eps)
print("===================================================================")
print(f"Апостериорная оценка погрешности по Рунге: {RungeError(ysShooting, ysShooting2, 4)}")
