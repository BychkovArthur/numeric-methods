import math
from copy import copy


'''
    Вариант №2
    Первое и второе задания - интерполяция
    Третье - приближение
    Четвертое - численное дифференцирование
'''
def f(x):
    return math.cos(x)

def M(x):  # Для рассчета ошибки
    return -math.sin(x)

def lagrange(xs, x):
    '''
        l_{i}(x) = ( (x - x_1)(x - x_2)...(x - x_{i-1})(x - x_{i+1})...(x - x_n) ) /
        ( (x_i - x_1)(x_i - x_2)...(x_i - x_{i-1})(x_i - x_{i+1})...(x - x_n) )
        Эта функция равна 1 только при x = x_i и 0 иначе.

        В это случае, функция выглядит как:
        sum(i=1, n) (f_i * l_i), где f_i - значение в i-й точке (которое дано нам)
    '''
    n = len(xs)
    res = 0

    for i in range(n):
        resCur = f(xs[i])
        for j in range(n):
            if (i == j):
                continue
            resCur *= (x - xs[j]) / (xs[i] - xs[j])
        res += resCur

    return res

def newton(xs, x):
    '''
        Если даны n точек:
        x_0, x_1, ..., x_{n-1}


        Разделенные разности:
        0 порядка: f(x_i) = f_i, где f_i - значение известной точки
        1 порядка: f(x_i, x_j) = (f(x_i) - f(x_j)) / (x_i - x_j)
        2 порядка: f(x_i, x_j, x_k) = ( f(x_i, x_j) - f(x_j, x_k) ) / (x_i - x_k)
        
        
        P_{n-1} = f(x_0) + (x-x_0)f(x_0, x_1) + ... + (x-x_0)(x-x_1)...(x-x_{n-2})f(x_0, x_1, ..., x_{n-1})
    '''
    n = len(xs)
    res = f(xs[0])                     # Сразу положил первое слагаемое

    polynom = 1                        # Здесь аккумулируются значения (x-x_0)*(x-x_1)....
    diffsPrev = [f(x) for x in xs]     # Начальные значения f_i
    for i in range(2, n + 1):          # Cколько аргументов у разделенной разности
        diffsCur = []
        polynom *= x - xs[i - 2]
        for j in range(n - i + 1):     # C какого икса начинаем считать f(x_j, x_{j+1}, ...)
            diffsCur.append((diffsPrev[j] - diffsPrev[j + 1]) / (xs[j] - xs[j + i - 1]))
        res += polynom * diffsCur[0]
        diffsPrev = copy(diffsCur)

    return res

def error(xs, x):
    n = len(xs)
    res = M(xs[-1]) / math.factorial(n)
    for i in range(n):
        res *= x - xs[i]
    return abs(res)

xs_a = [0, math.pi / 6, math.pi / 3, math.pi / 2]
xs_b = [0, math.pi / 6, 5* math.pi / 12, math.pi / 2]
xs = [xs_a, xs_b]
x = math.pi / 4

print("Истинное значение: ", f(x), "\n\n\n")

for xs_i in xs:
    print("xs =", xs_i)
    y = lagrange(xs_i, x)
    print("Лагранж: ", y)

    y = newton(xs_i, x)
    print("Ньютон: ", y)

    print("Погрешность: ", error(xs_i, x), "\n\n\n\n")
