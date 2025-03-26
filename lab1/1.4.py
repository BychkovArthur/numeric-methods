import numpy as np
import matplotlib.pyplot as plt

'''
    Т.к. матрица симетричная, ее СВ ортогональны:
    А = JT*S*J, где J - ортогональная матрица.
    S - диагональная матрица из СЗ
    
    Lam = JT * S * J
'''

def jacobi_eigenvalue(A, tol=1e-20):
    n = A.shape[0]
    V = np.eye(n)  
    error_history = []

    while True:
        off_diag_sum = np.sum(np.square(A - np.diag(np.diag(A))))
        error_history.append(off_diag_sum)

        if off_diag_sum < tol:
            break

        # Это матрица с нулями на диагонали, чтоб в ней найти максимальный внедиагональный 
        without_diag = A - np.diag(np.diag(A))
        # Линейный индекс максимального элемента
        max_abs_of_non_diag = np.argmax(np.abs(without_diag))
        p, q = np.unravel_index(max_abs_of_non_diag, A.shape) # Перевод одночисленного индекса в обычный
        
        print(p, q)

        # Угол, на который надо сделать поворот
        theta = 0.5 * np.arctan(2 * A[p, q] / (A[p, p] - A[q, q])) if A[p, p] != A[q, q] else np.pi / 4

        # Создаем матрицу поворота
        J = np.eye(n)
        J[p, p] = np.cos(theta)
        J[q, q] = np.cos(theta)
        J[p, q] = -np.sin(theta)
        J[q, p] = np.sin(theta)

        # Поэтапно восстанавливаю диагональную матрицу.
        # Домножаю слева на обратное, и справа на обратное к обратному (т.е. просто на J)
        A = J.T @ A @ J

        # Матрица V будет матрицей из собственных векторов,
        # т.к это по сути будет матрица перехода в базис из собственных векторов
        V = V @ J

    eigenvalues = np.diag(A)
    eigenvectors = V

    return eigenvalues, eigenvectors, error_history

A = np.array([[-9, 7, 5],
              [7, 8, 9],
              [5, 9, 8]])

eigenvalues, eigenvectors, error_history = jacobi_eigenvalue(A, tol=1e-25)

print("Собственные значения:")
print(eigenvalues)
print("\nСобственные векторы:")
print(eigenvectors)

plt.plot(error_history)
plt.xlabel('Итерации')
plt.ylabel('Сумма квадратов внедиагональных элементов')
plt.title('Зависимость суммы квадратов внедиагональных элементов от числа итераций')
plt.savefig('error_vs_iterations.png') 

plt.show()