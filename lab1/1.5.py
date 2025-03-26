import numpy as np

def sign(x):
    if x == 0: return 0
    return -1 if x < 0 else 1

def qr_decomposition(A):
    """
    Выполняет QR-разложение матрицы A методом Грамма-Шмидта.
    Возвращает ортогональную матрицу Q и верхнетреугольную матрицу R.
    """
    m, n = A.shape
    A = np.asarray(A, dtype=float)
    Q = np.eye(n, dtype=float)
    
    for i in range(n - 1):
        v = A[:, i].copy()
        v[:i] = 0
        v[i] += sign(A[i, i]) * np.linalg.norm(A[:, i])
        
        I = np.eye(n, dtype=float)
                    # Аналог простого умножения, но чтобы получилась матрица
        H = I - 2 * np.outer(v, v) / (v.T @ v)
        
        A = H @ A
        Q = Q @ H

    return Q, A

A = np.array([[-6., -4., 0.], [-7., 6., -7.], [-2., -6., -7.]])
# A = np.array([[1., 3., 1.], [1., 1., 4.], [4., 3., 1.]])
Q, R = qr_decomposition(A)
print("Q:")
print(Q)
print("R:")
print(R)

print("QR:")
print(Q @ R)


def alrorithmQR(A, eps=0.001):
    n = len(A)
    A = np.copy(A)
    iter = 0
    lambdas = np.empty((n, 2))
    while (True):
        iter += 1
        Q, R = qr_decomposition(A)
        A = np.dot(R, Q)

        flg = True
        skip = False
        # print(f"iter #{iter}")
        for i in range(n):
            if skip:
                skip = False
                continue

            if i < n - 1:
                D = A[i][i] ** 2 + A[i + 1][i + 1] ** 2 - 2 * A[i][i] * A[i + 1][i + 1] + 4 * A[i][i + 1] * A[i + 1][i]
                if D < 0:
                    re = (A[i][i] + A[i + 1][i + 1]) / 2
                    im = np.sqrt(-D) / 2
                    
                    # Критерий остановки для пары комплексно-сопряженных
                    lambda_ = np.sqrt(re ** 2 + im ** 2)
                    lambdaPrev = np.sqrt(lambdas[i][0] ** 2 + lambdas[i][1] ** 2)
                    # print(f"coord #{i}: abs(lambda_ - lambdaPrev) = {abs(lambda_ - lambdaPrev)}")
                    if iter > 1 and abs(lambda_ - lambdaPrev) > eps:
                        flg = False

                    lambdas[i][0] = re
                    lambdas[i][1] = im
                    lambdas[i + 1][0] = re
                    lambdas[i + 1][1] = -im

                    skip = True
                    continue

            lambdas[i][0] = A[i][i]
            lambdas[i][1] = 0
            # Критерий остановки для действительного значения
            sum_ = np.sqrt(sum([A[j][i] ** 2 for j in range(i + 1, n)]))
            # print(f"coord #{i}: sum = {sum_}")
            if sum_ > eps:
                flg = False

        if flg:
            break

    return lambdas, iter


lam, iter = alrorithmQR(A)
print("lam = \n", lam)
print("iterations: ", iter)