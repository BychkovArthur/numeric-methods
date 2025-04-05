# f(x) = ln(x + 2) - x^2
# x^2 = ln(x + 2)
# x = sqrt(ln(x+2)), здесь именно со знаком +, т.к. это решение даст положительный корень
# [1, 1.5]

import math

def f(x):
    return math.log(x + 2) - x**2

def fDer(x):
    return (1 / (x + 2)) - 2 * x

def fDer2(x):
    return (-1 / (x + 2)**2) - 2

def newton(x0, eps):
    if (f(x0) * fDer2(x0) <= 0):
        raise Exception(f'Последовательность с x_0={x0} не будет сходится к корню')

    xPrev = x0
    iter = 0
    while (True):
        iter += 1
        xCur = xPrev - f(xPrev) / fDer(xPrev)
        if abs(xCur - xPrev) < eps:
            break
        xPrev = xCur

    return xCur, iter


def phi(x):
    '''
        1) phi(x) принадлежит [1, 1.5]
           для любого x из [1, 1.5]
           См simple_iter.png
           
        2) phi'(x) = -1 / (2 * ln(x + 2) * (x + 2))
            Подходит q=0.5. Для него
            abs(phi'(x)) <= q < 1 Для любого x из (1, 1.5)
            См simple_iter_q.png
    '''
    return math.sqrt(math.log(x + 2))

def simpleIterations(x0, q, eps):
    xPrev = x0
    iter = 0
    while (True):
        iter += 1
        xCur = phi(xPrev)
        error = q / (1 - q) * abs(xCur - xPrev)
        if error < eps:
            break
        xPrev = xCur
    return xCur, iter


eps = float(input("Точность: "))

x0 = 1.5

newtonAns, iter = newton(x0, eps)
print("Метод Ньютона")
print("\tКорень: ", newtonAns)
print("\tКоличество итераций: ", iter)

q = 0.5
simpleIterationsAns, iter = simpleIterations(x0, q, eps)
print("Метода простых итераций")
print("\tКорень: ", simpleIterationsAns)
print("\tКоличество итераций: ", iter)