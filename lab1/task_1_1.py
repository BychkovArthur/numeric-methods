import numpy as np


'''
    Вариант №2
    
    LU = PA
    L - нижнетреугольная матрица. На главной диагонали 1.
    U - то, что получилось в Гауссе.
    P - перестановка строк.
    A - исходная матрица
    
    p - сколько раз меняли строки
'''

p = 0

def lu_decomposition(A):
    global p
    n = len(A)
    
    # Единичные матрицы
    P = np.eye(n)
    L = np.eye(n) 
    
    U = A.copy() 

    for i in range(n):
        # U[i:, i] - взять строки, начиная с i; столбец i
        # Получаем в итоге номер строки с максимальным элементом в i столбце
        max_row = np.argmax(np.abs(U[i:, i])) + i
        if max_row != i:
            U[[i, max_row]] = U[[max_row, i]]
            P[[i, max_row]] = P[[max_row, i]]
            L[[i, max_row], :i] = L[[max_row, i], :i]
            p += 1


        for j in range(i+1, n):
            L[j, i] = U[j, i] / U[i, i]
            U[j, i:] -= L[j, i] * U[i, i:]

    return P, L, U

def solve_slu(A, b):
    n = len(b)
    P, L, U = lu_decomposition(A)
    b_permuted = P @ b
    x = np.zeros(n)
    y = np.zeros(n)
    
    '''
        У меня есть LU = PA
        
        Изначально надо было решить
        Ax = b
        
        Рассмотрим
        PAx = Pb --> LUx = Pb
        
        Решим две системы
        1) Ux = y
        2) Ly = Pb
    '''
    
    for i in range(n):
        y[i] = b_permuted[i] - np.dot(L[i, :i], y[:i])
    for i in range(n - 1, -1, -1):                        # Т.к. на главной диагонали не только единицы
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i] 
    return x

def determinant(A):
    '''
        LU = PA
        det(LU) = det(PA)
        
        1) det(L) = 1
        2) det(P) = (-1)^p
        3) det(U) = Произведение элементов на диагонали
        
        => det(A) = det(LU) / det(P)
        => det(A) = 1 * det(U) / (-1)^p
        => det(A) = det(U) * (-1)^p
    '''
    plu = lu_decomposition(A)
    return np.prod(np.diag(plu[2])) * (-1)**p

def inverse_matrix(A):
    '''
        Решаем n систем
    '''
    n = len(A)
    inv_A = np.zeros((n, n))
    
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1
        inv_A[:, i] = solve_slu(A, e)
    
    return inv_A


if __name__ == "main":
    A = np.array([[2, 7, -8, 6],
                [4, 4, 0, -7],
                [-1, -3, 6, 3],
                [9, -7, -2, -8]], dtype=float)
    b = np.array([-39, 41, 4, 113], dtype=float)

    P, L, U = lu_decomposition(A)

    print("P = \n", P, end='\n\n\n')
    print("L = \n", L, end='\n\n\n')
    print("U = \n", U, end='\n\n\n')

    print("x = ", solve_slu(A, b), end='\n\n\n')

    print("det from numpy = ", np.linalg.det(A))
    print("det            = ", determinant(A), end='\n\n\n')

    print("inverse A = \n", inverse_matrix(A))