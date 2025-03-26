import numpy as np

'''
    Вариант №2
'''

def progonka(a, b, c, d):
    n = len(d)

    P = np.zeros(n - 1)
    Q = np.zeros(n)  

    # Первый шаг
    P[0] = -c[0] / b[0]
    Q[0] = d[0] / b[0]

    # Середина
    for i in range(1, n - 1):
        P[i] = -c[i] / (b[i] + a[i] * P[i - 1])
        Q[i] = (d[i] - a[i] * Q[i - 1]) / (b[i] + a[i] * P[i - 1])

    # Конец
    Q[n - 1] = (d[n - 1] - a[n - 1] * Q[n - 2]) / (b[n - 1] + a[n - 1] * P[n - 2])

    x = np.zeros(n)
    x[n - 1] = Q[n - 1]

    # Обратный ход
    for i in range(n - 2, -1, -1):
        x[i] = Q[i] + P[i] * x[i + 1]

    return x

a = np.array([0, 3, 2, 5, -8])
b = np.array([10, 10, -9, 16, 16])     
c = np.array([5, -2, -5, -4])
d = np.array([120, -91, 5, -74, -56])

x = progonka(a, b, c, d)

print("Решение системы:", x)