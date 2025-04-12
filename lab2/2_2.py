import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from lab1.task_1_1 import solve_slu
'''
    Вариант 2

    (x1^2 + 3^2)x2 - 3^3 = 0
    (x1 - 3/2)^2 + (x2 - 3/2)^2 - 3^2 = 0

    --------------------------------------------------------------------
    Систему нелинейных уравнений можно записать в виде:
    f_{1}(x_{1}, ..., x_{n}) = 0
    f_{2}(x_{1}, ..., x_{n}) = 0
    ...
    f_{n}(x_{1}, ..., x_{n}) = 0
    --------------------------------------------------------------------
    f - Вектор функций
    x^{k} - Вектор переменных на k-м шаге
    d x^{k} - Вектор приращений на k-м шаге
    
    --------------------------------------------------------------------
    Изначальное приближение x^{0} = x_{1}^{0}, ..., x_{n}^{0} - задано. 
    
    x_{1}^{k+1} = x_{1}^{k} + d x_{1}^{k}
    x_{2}^{k+1} = x_{2}^{k} + d x_{2}^{k}
    ...
    x_{n}^{k+1} = x_{n}^{k} + d x_{n}^{k}
    
    --------------------------------------------------------------------
    Дельты вычисляются из решения СЛАУ:
    f(x^{k}) + J(x^{k}) * d x^{k} = 0
    
    Где J(x^{k}) - Матрица Якоби.
    В столбцах меняется то, по чему берем частную производную.
    x_{1}, x_{2}, ..., x_{n}
    В строках меняется то, от какой функции берем частную производную.
    f_{1}, f_{2}, ..., f_{n}
    -------------------------------------------------------------------
    Из формулы можно выразить итерационный процесс решения:
    
    x^{k+1} = x^{k} - JJ(x^{k}) * f(x^{k}), где JJ - обратная к J.

'''

def inf_norm_diff(xCur, xPrev):
    return np.max(np.abs(np.array(xCur) - np.array(xPrev)))


'''
    (x1^2 + 3^2)x2 - 3^3 = 0
    (x1 - 3/2)^2 + (x2 - 3/2)^2 - 3^2 = 0
    
    Решение должно быть:
    (4.44695, 0.9383)
    
'''
def f(x):
    return np.array([
        (x[0]**2 + 9) * x[1] - 27,
        (x[0] - 1.5)**2 + (x[1] - 1.5)**2 - 9
    ])

def J(x):
    return np.array([
        [2 * x[0] * x[1], x[0]**2 + 9],
        [2 * x[0] - 3, 2 * x[1] - 3]
    ], dtype='float64')

def newton(x0, eps):
    xPrev = x0
    iter = 0
    while (True):
        iter += 1
        xDelta = solve_slu(J(xPrev), -f(xPrev))
        xCur = xPrev + xDelta
        if inf_norm_diff(xCur ,xPrev) < eps:
            break
        xPrev = xCur
    return xCur, iter


def phi(x):
    return np.array([
        np.sqrt(9 - (x[1] - 1.5)**2) + 1.5,
        27 / (x[0]**2 + 9)
    ])
def simpleIterations(x0, q, eps):
    xPrev = x0
    iter = 0
    while (True):
        iter += 1
        xCur = phi(xPrev)
        error = q / (1 - q) * inf_norm_diff(xCur ,xPrev)
        if error < eps:
            break
        xPrev = xCur
    return xCur, iter


eps = float(input("Точность: "))

x0 = np.array([4.4, 1])

newtonAns, iter = newton(x0, eps)
print("Метод Ньютона")
print("\tРешение: ", newtonAns)
print("\tКоличество итераций: ", iter)

q = max(
    abs((-54 * 4.3) / (4.3**4 + 18 * 4.3**2 + 81)),
    abs((-2 * 0.9 + 3) / (np.sqrt(-4 * 0.9**2 + 12 * 0.9 + 27)))
)
simpleIterationsAns, iter = simpleIterations(x0, q, eps)
print("Метода простых итераций")
print("\tРешение: ", simpleIterationsAns)
print("\tКоличество итераций: ", iter)