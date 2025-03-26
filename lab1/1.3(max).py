import numpy as np

'''
    Вариант 2
'''


with open('1.3.txt') as file:
    n = int(file.readline())
    A_B = np.array(list((list(map(float, file.readline().strip().split())) for _ in range(n))))
    eps = float(file.readline())

def get_norma_matrix(matrix):
    max_ = -1
    for i in range(n):
        sum = 0
        for j in range(n):
            sum += abs(matrix[i][j])
        if sum > max_:
            max_ = sum
    return max_

def get_norma_vector(vector):
    return np.max(np.abs(vector))

def simple_iterations(alpha, beta, eps):
    currEps = float('inf')

    if get_norma_matrix(alpha) < 1: # проверяю выполнение достаточного условия
        x_prev = beta
        x_next = beta + alpha @ beta
        currEps = (get_norma_matrix(alpha) / (1 - get_norma_matrix(alpha))) * get_norma_vector(x_next - x_prev)
        while(currEps > eps):
            x_prev = x_next
            x_next = beta + alpha @ x_prev
            currEps = (get_norma_matrix(alpha) / (1 - get_norma_matrix(alpha))) * get_norma_vector(x_next - x_prev)
    else: # если не выполнилось, нахожу по другому
        x_prev = beta
        x_next = beta + alpha @ beta
        currEps = get_norma_vector(x_next - x_prev)
        while(currEps > eps):
            x_prev = x_next
            x_next = beta + alpha @ x_prev
            currEps = get_norma_vector(x_next - x_prev)
    return x_next

def Seidel(alpha, beta, eps):
    B = np.zeros((n, n))
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i < j:
                B[i][j] = alpha[i][j]
            elif i > j:
                C[i][j] = alpha[i][j]

    return simple_iterations(np.linalg.inv((np.eye(n, n) - B)) @ C, np.linalg.inv((np.eye(n, n) - B)) @ beta, eps)

alpha = list([] for _ in range(n))
beta = list()

for i in range(n): #перебираю начальную матрицу, переношу всё вправо, кроме диагональных коэфициентов
    if(A_B[i][i] == 0):
        for j in range(i+1, n):
            if(A_B[j][i] != 0):
                A_B[[i, j]] = A_B[[j, i]]
    
               # Это b_i
    beta.append(A_B[i][n] / A_B[i][i])
    for j in range(n):
        if(j == i):
            alpha[i].append(0)
            continue
        alpha[i].append(-1 * A_B[i][j] / A_B[i][i])



print(simple_iterations(np.array(alpha).copy(), np.array(beta).copy(), eps))
print(Seidel(np.array(alpha).copy(), np.array(beta).copy(), eps))