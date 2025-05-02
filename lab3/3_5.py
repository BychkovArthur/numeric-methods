import math

'''
    Формулы численного интегрирования используются в тех случаях, когда 
    вычислить  аналитически  определенный  интеграл не удается.
'''

def f(x):
    return x / (3*x + 4)**3

def F(x):
    return -1 * (3*x + 2) / (81*x**2 + 216*x + 144)

def splitting(x0, xk, h):
    '''
        Получить точки разбиения
    '''
    xs = []
    x = x0
    while x < xk:
        xs.append(x)
        x += h
    xs.append(xk)
    return xs


def rectangles(x0, xk, h):
    '''
        Проходит через середину
    '''
    integral = 0
    xs = splitting(x0, xk, h)
    for i in range(1, len(xs)):
        integral += h * f((xs[i - 1] + xs[i]) / 2)
    return integral


def trapezoids(x0, xk, h):
    '''
        Соединяются прямой f(x_i) и f(x_{i+1})
    '''
    integral = 0
    xs = splitting(x0, xk, h)
    for i in range(1, len(xs)):
        integral += 0.5 * h * (f(xs[i - 1]) + f(xs[i]))
    return integral


def simpson(x0, xk, h):
    '''
        Каждый отрезок [x_i, x_{i+1}] интерполируем параболой по точкам
        x_i, (x_i + x_{i+1}) / 2, x_{i+1}
    '''
    integral = 0
    xs = splitting(x0, xk, h)
    for i in range(1, len(xs)):  # h/2, т.к. разделили каждый [x_i, x_{i+1}] пополам
        integral += 1/3 * h/2 * (f(xs[i - 1]) + 4 * f((xs[i - 1] + xs[i]) / 2) + f(xs[i]))
    return integral


'''
    Метод  Рунге-Ромберга-Ричардсона  позволяет  получать  более  высокий  порядок точности вычисления
'''
def runge(values, hs, p):
    k = hs[0] / hs[1]
    return values[1] + (values[1] - values[0]) / (k**p - 1)


'''
    Порядки точности
'''
orderRectangles = 2
orderTrapezoids = 2
orderSimpson = 4


x0 = 0
xk = 4
hs = [1, 0.5]
integral = F(4) - F(0)

integralsRectangles = []
errorsRectangles = []
integralsTrapezoids = []
errorsTrapezoids = []
integralsSimpson = []
errorsSimpson = []

print(f"Истинное значение: {integral}")

for h in hs:
    print(f"Шаг {h}")
    integralRectangles = rectangles(x0, xk, h)
    integralsRectangles.append(integralRectangles)
    errorRectangles = abs(integral - integralRectangles)
    errorsRectangles.append(errorRectangles)
    print(f"Метод прямоугольников")
    print(f"\tЗначение: {integralRectangles}")
    print(f"\tАбсолютная погрешность: {errorRectangles}")

    integralTrapezoids = trapezoids(x0, xk, h)
    integralsTrapezoids.append(integralTrapezoids)
    errorTrapezoids = abs(integral - integralTrapezoids)
    errorsTrapezoids.append(errorTrapezoids)
    print(f"Метод трапеций")
    print(f"\tЗначение: {integralTrapezoids}")
    print(f"\tАбсолютная погрешность: {errorTrapezoids}")

    integralSimpson = simpson(x0, xk, h)
    integralsSimpson.append(integralSimpson)
    errorSimpson = abs(integral - integralSimpson)
    errorsSimpson.append(errorSimpson)
    print(f"Метод Симпсона")
    print(f"\tЗначение: {integralSimpson}")
    print(f"\tАбсолютная погрешность: {errorSimpson}")
    print("==================================================")


print("Уточненные значения")
integralRectangles = runge(integralsRectangles, hs, orderRectangles)
errorRectangles = abs(integral - integralRectangles)
print(f"Метод прямоугольников")
print(f"\tЗначение: {integralRectangles}")
print(f"\tАбсолютная погрешность: {errorRectangles}")

integralTrapezoids = runge(integralsTrapezoids, hs, orderTrapezoids)
errorTrapezoids = abs(integral - integralTrapezoids)
print(f"Метод трапеций")
print(f"\tЗначение: {integralTrapezoids}")
print(f"\tАбсолютная погрешность: {errorTrapezoids}")

integralSimpson = runge(integralsSimpson, hs, orderSimpson)
errorSimpson = abs(integral - integralSimpson)
print(f"Метод Симпсона")
print(f"\tЗначение: {integralSimpson}")
print(f"\tАбсолютная погрешность: {errorSimpson}")
