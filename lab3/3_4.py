def getIndex(x, xs):
    '''
        Определяем i, такое что:
        x в [x_i, x_{i+1}]
    '''
    for i in range(1, len(xs)):
        if x <= xs[i]:
            return i - 1

'''
    Используем интерполяционный многочлен 2-й степени, т.к.
    хотим найти вторую производную.
    (По идее, здесь Ньютоновская интерполяция)
'''
def firstDerivative(x, xs, ys):
    i = getIndex(x, xs)
    return (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]) + ((ys[i + 2] - ys[i + 1]) / (xs[i + 2] - xs[i + 1]) - (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i])) / (xs[i + 2] - xs[i]) * (2 * x - xs[i] - xs[i + 1])

def secondDerivative(x, xs, ys):
    i = getIndex(x, xs)
    return 2 * ((ys[i + 2] - ys[i + 1]) / (xs[i + 2] - xs[i + 1]) - (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i])) / (xs[i + 2] - xs[i])

x = 1.0
xs = [-1.0, 0.0, 1.0, 2.0, 3.0]
ys = [-0.5, 0.0, 0.5, 0.86603, 1.0]

print(f"Первая производная: {firstDerivative(x, xs, ys)}")
print(f"Вторая производная: {secondDerivative(x, xs, ys)}")