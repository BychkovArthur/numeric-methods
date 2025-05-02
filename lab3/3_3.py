import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from lab1.task_1_1 import solve_slu

def getValue(as_: list, x):
    return sum(ai * x**i for i, ai in enumerate(as_))

n = int(input("Степень многочлена: ")) + 1
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
ys = np.array([0.86603, 1.0, 0.86603, 0.50, 0.0, -0.50])

N = len(xs)

A = np.zeros((n, n))
b = np.zeros(n)
for k in range(n):
    for j in range(N):
        b[k] += ys[j] * xs[j] ** k

    for i in range(n):
        for j in range(N):
            A[k][i] += xs[j]**(k+i)

as_ = solve_slu(A, b)

print("Приближающий многочлен:")
polynom = []
for i in range(len(as_)):
    polynom.append(f"{np.round(as_[i], 4)} * x^{i}")
print(" + ".join(polynom))

fs = [getValue(as_, x) for x in xs]

error = sum([(fs[i] - ys[i]) ** 2 for i in range(N)])
print(f"Сумма квадратов ошибок: {np.round(error, 4)}")

plt.plot(xs, ys, linestyle='-', color=(1, 0, 0), label=f"Функция")
plt.plot(xs, fs, linestyle='-', color=(0, 0, 1), label=f"Приближение")
plt.legend()
plt.grid(True)
plt.show()