import numpy as np

debug = False

'''
    https://www.youtube.com/watch?v=nnPVA48gIAE


    ОДУ n-го порядка - уравнение, у который самая старшая производная имеет порядок n

    Задача Коши - решение ОДУ с заданными начальными условиями.
    Количество начальных условий = порядок уравнения.

-------------------------------------------------------------------------------------

    Задача Коши для ОДУ порядка >1 будет выглядеть так:
    y^{(n)}   = f(x, y', y'', ..., y^{(n-1)})
    y(x_0)    = y_0
    y'(x_0)   = y_{01}
    y''(x_0)  = y_{02}
    ...
    y^{(n-1)} = y_{0(n-1)}

    
    (Немного иначе с индексами, в сравнении с методичкой)
    Для решения такой задачи Коши введем замену:
    z_1       = y
    z_2       = y'
    ...
    z_n       = y^{(n-1)}
    
    Тогда исходную задачу можно переписать в виде системы из n ОДУ первого порядка:
    z_1'      = z_2
    z_2'      = z_3
    ...
    z_n'      = g(x, z_1, z_2, ..., z_3)

    Или в общем виде:
    z_1'      = g_1(x, z_1, z_2, ..., z_3)
    z_2'      = g_2(x, z_1, z_2, ..., z_3)
    ...
    z_n'      = g_n(x, z_1, z_2, ..., z_3)
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
    Ks = np.empty((p, dim))   # dim штук, т.к. dim уравнений в системе

    for i in range(p):
        newX = x + as_[i] * h
        newY = np.copy(y)
        for j in range(i):
            newY += h * bs[i - 1][j] * Ks[j]

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
    '''
        y_{k+1}    = y_k + delta y_k
        delta y_k  = sum(i=1, p) (c_i*K^k_i)

        K^k_i      = hf(x_k + a_i*h,
                        y_k + h * sum(j=1, i-1) (b_{ij}K^k_j))

        i = 2,3,...,p
    '''
    N = len(xs) - 1             # Количество точек
    dim = y0.shape[0]
    ys = np.empty((N + 1, dim))
    ys[0] = y0

    for k in range(1, N + 1):
        ys[k] = ys[k - 1] + getDeltaY(xs[k - 1], ys[k - 1], h)

    return ys


# y0 - начальные условия.
# Причем, сначала начальное условие для y(x_0), а потом y'(x_0)
def Euler(xs: list, y0: np.ndarray, h):
    '''
        Мы знаем все для начальной точки. Поэтому, можем вычислить y'(x_0)

        Далее мы проводим касательную.
        И за значение y'(x_1) принимаем значение касательной в точке x_1

        y_{k+1} = y_k + h*f(x_k, y_k)
    '''
    N = len(xs) - 1
    dim = y0.shape[0]                  # Получение размерности = порядок ОДУ
    ys = np.empty((N + 1, dim))
    ys[0] = y0

    for k in range(N):
        ys[k + 1] = ys[k] + h * f(xs[k], ys[k])

    return ys


def Adams(xs: list, y0s: np.ndarray, h):
    '''
        В этом методе нужно получить сначала первые
        4 пары (x_i, y_i).
        Для этого получаем y0s - результат работы Рунге-Кутты
    '''
    N = len(xs) - 1
    dim = y0s.shape[1]
    ys = np.empty((N + 1, dim))
    
    fs = np.empty((N + 1, dim))
    for i in range(4):
        ys[i] = np.copy(y0s[i])
        fs[i] = f(xs[i], ys[i])

    for k in range(4, N + 1):
        ys[k] = ys[k - 1] + h/24 * (55 * fs[k - 1] - 59 * fs[k - 2] + 37 * fs[k - 3] - 9 * fs[k - 4])
        fs[k] = f(xs[k], ys[k])

    return ys


def RungeError(ys: np.ndarray, ys2: np.ndarray, p):
    '''
        Для вычисления ошибки здесь нужено уменьшить шаг вдвое

        ys  - значения в исходных точках
        ys2 - значения в точках, который в 2 раза чаще
    '''
    k = 2
    error = 0
    for i in range(ys.shape[0]):
        # Ищем максимум среди всех ошибок по модулю
        # т.е. не для каждой точки, а для всего набора
        error = max(error, abs(ys2[i * 2][0] - ys[i][0]) / (k ** p - 1))
    return error


# Функции
def f(x: float, y: np.ndarray):
    '''
        y'' + y - 2cosx = 0
        Замена:
        z_1 = y
        z_2 = y'

        Получим систему:
        z_1' = z_2
        z_2' = 2cosx - z_1
    '''
    return np.array([
        y[1],
        2 * np.cos(x) - y[0]
    ])

def calcRealValue(x):
    return x * np.sin(x) + np.cos(x)

# Интервал
a = 0.00000001
b = 1
# Начальные условия
y0 = np.array([1, 0])
# Считаем для шага из варианта
h = 0.1

xs = splitting(a, b, h)
ysEuler = Euler(xs, y0, h)
ysRungeKutta = RungeKutta(xs, y0, h)
ysAdams = Adams(xs, ysRungeKutta, h)

print(f"Шаг: {h}")
for i in range(len(xs)):
    y = calcRealValue(xs[i])

    print(f"xk = {np.round(xs[i], 5)}, y(xk) = {np.round(y, 5)}")

    errorEuler = abs(ysEuler[i][0] - y)
    errorRungeKutta = abs(ysRungeKutta[i][0] - y)
    errorAdams = abs(ysAdams[i][0] - y)

    print(f"\tЭйлер:      yk = {np.round(ysEuler[i][0], 5)}, e = {np.round(errorEuler, 8)}")
    print(f"\tРунге-Кутт: yk = {np.round(ysRungeKutta[i][0], 5)}, e = {np.round(errorRungeKutta, 8)}")
    print(f"\tАдамс:      yk = {np.round(ysAdams[i][0], 5)}, e = {np.round(errorAdams, 8)}")


# Считаем для шага в два раза короче, чтобы применить оценку Рунге
h2 = h / 2

xs2 = splitting(a, b, h2)
ysEuler2 = Euler(xs2, y0, h2)
ysRungeKutta2 = RungeKutta(xs2, y0, h2)
ysAdams2 = Adams(xs2, ysRungeKutta2, h2)

print("===================================================================")
print("Апостериорные оценки погрешности по Рунге:")
print(f"\tЭйлер:      {RungeError(ysEuler, ysEuler2, 1)}")
print(f"\tРунге-Кутт: {RungeError(ysRungeKutta, ysRungeKutta2, 4)}")
print(f"\tАдамс:      {RungeError(ysAdams, ysAdams2, 3)}")