import numpy as np

def sign(x):
    if x == 0: return 0
    return -1 if x < 0 else 1

'''

    Преобразование Хаусхолдера

    Q - Ортогональная матрица
    R - Верхняя треугольная
'''
def qr_decomposition(A):
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
        
        '''
            A = QR
            
            R = H_{n-1} ... H_2 * H_1 * A
        
            => Q = (H_{n-1} ... H_2 * H_1)^{-1} = 
            =      (H_{n-1} ... H_2 * H_1)^T
            =      H_1 * .... * H_{n-1}
        '''
        A = H @ A
        Q = Q @ H

    return Q, A

A = np.array([[-6., -4., 0.], [-7., 6., -7.], [-2., -6., -7.]])
# A = np.array([[1., 3., 1.], [1., 1., 4.], [4., 3., 1.]])
# alpha = np.pi / 4
# A = np.array([[np.cos(alpha), -np.sin(alpha)],
#               [np.sin(alpha), np.cos(alpha)]])
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
        for i in range(n):
            if skip:
                skip = False
                continue

            if i < n - 1:
                D = A[i][i]**2 + A[i + 1][i + 1]**2 - 2 * A[i][i] * A[i + 1][i + 1] + 4 * A[i][i + 1] * A[i + 1][i]
                if D < 0:
                    re = (A[i][i] + A[i + 1][i + 1]) / 2
                    im = np.sqrt(-D) / 2

                    # Критерий остановки для пары комплексно-сопряженных
                    diff_re = re - lambdas[i][0]
                    diff_im = im - lambdas[i][1]
                    norma = np.sqrt(diff_re**2 + diff_im**2)
                    if iter > 1 and abs(norma) > eps:
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
            sum_ = np.sqrt(sum([A[j][i]**2 for j in range(i + 1, n)]))
            if sum_ > eps:
                flg = False

        if flg:
            break

    return lambdas, iter


lam, iter = alrorithmQR(A)
print("СЗ: \n", lam)
print("Итерации: ", iter)