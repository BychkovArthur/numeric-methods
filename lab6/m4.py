import numpy as np
import matplotlib.pyplot as plt

# Настройка размера графиков
plt.rcParams['figure.figsize'] = [10, 10]

# Функция точного решения волнового уравнения
def U(x, t, a):
    return np.sin(x - a * t) + np.cos(x + a * t)

# Начальное условие u(x, 0) = psi(x)
def psi(x):
    return np.sin(x) + np.cos(x)

# --------------------------------------------------------------------------------
# Второе начальное условие: u_t(x, 0)
# Истинная производная по времени: phi(x) = -a * (sin(x) + cos(x))
# --------------------------------------------------------------------------------

# Аппроксимация 2-го порядка (Ряд Тейлора): u^1 = u^0 + tau*phi + tau^2/2 * u_tt
def d_psi_2nd(x, a, tau):
    return (1 - a * tau - a ** 2 * tau ** 2 / 2) * (np.sin(x) + np.cos(x))

# Аппроксимация 1-го порядка (Явный Эйлер): (u^1 - u^0)/tau = phi  => u^1 = u^0 + tau*phi
def d_psi_1st(x, a, tau):
    # u^1 = (sin(x)+cos(x)) + tau * (-a*(sin(x)+cos(x)))
    return (1 - a * tau) * (np.sin(x) + np.cos(x))

# --------------------------------------------------------------------------------

# Граничные условия
u_0 = 0
u_l = 0

# Проверка диагонального преобладания матрицы (необходима для метода прогонки)
def check(A):
    if np.shape(A)[0] != np.shape(A)[1]:
        return False
    n = np.shape(A)[0]
    for i in range(n):
        sum_row = 0
        for j in range(n):
            if i != j:
                sum_row += abs(A[i][j])
        if abs(A[i][i]) < sum_row:
            return False
    return True

# Метод прогонки для решения СЛАУ (трёхдиагональная матрица)
def solve(a, b):
    if check(a):
        p = np.zeros(len(b))
        q = np.zeros(len(b))
        # Прямой ход
        p[0] = -a[0][1] / a[0][0]
        q[0] = b[0] / a[0][0]
        for i in range(1, len(p) - 1):
            p[i] = -a[i][i + 1] / (a[i][i] + a[i][i - 1] * p[i - 1])
            q[i] = (b[i] - a[i][i - 1] * q[i - 1]) / (a[i][i] + a[i][i - 1] * p[i - 1])
        # Последняя точка
        p[-1] = 0
        q[-1] = (b[-1] - a[-1][-2] * q[-2]) / (a[-1][-1] + a[-1][-2] * p[-2])
        # Обратный ход
        x = np.zeros(len(b))
        x[-1] = q[-1]
        for i in reversed(range(len(b) - 1)):
            x[i] = p[i] * x[i + 1] + q[i]
        return x

# Функция для подсчёта погрешности схем
def error(sigma, l, a, T, U, type, init_order):
    N_array = [10, 20, 40]  # Разные числа разбиений по x
    size = np.size(N_array)
    h_array = np.zeros(size)
    tau_array = np.zeros(size)
    K_array = np.zeros(size)
    errors1 = np.zeros(size)  # Ошибки явной схемы
    errors2 = np.zeros(size)  # Ошибки неявной схемы

    for i in range(0, size):
        h_array[i] = l / N_array[i]  # шаг по пространству
        tau_array[i] = np.sqrt(sigma * h_array[i] ** 2 / a)  # шаг по времени
        K_array[i] = int(round(T / tau_array[i]))  # число шагов по времени
        x_array = np.arange(0, l + h_array[i], h_array[i])
        
        u1 = Explicit_Method(N_array[i], int(K_array[i]), sigma, tau_array[i], a, h_array[i], type, init_order)
        u2 = Implicit_Method(N_array[i], int(K_array[i]), sigma, tau_array[i], a, h_array[i], init_order)
        
        t = tau_array[i] * K_array[i] / 2  # средний момент времени
        if (np.size(x_array) != N_array[i] + 1):
            x_array = x_array[:N_array[i] + 1]
        u_correct = U(x_array, t, a)
        u1_calculated = u1[int(K_array[i] / 2)]
        u2_calculated = u2[int(K_array[i] / 2)]
        errors1[i] = np.amax(np.abs(u_correct - u1_calculated))
        errors2[i] = np.amax(np.abs(u_correct - u2_calculated))
    return N_array, errors1, errors2

# Функция для отображения решения
def show_solution(h, tau, K, l, u, U, a):
    x_array = np.arange(0, l + h, h)
    fig, ax = plt.subplots()
    t_indices = [int(K * 0.05), int(K * 0.1), int(K * 0.25)]  # моменты времени
    colors = ['blue', 'green', 'red']
    for i in range(len(t_indices)):
        idx = t_indices[i]
        if idx >= len(u): idx = len(u) - 1
        
        u_correct = U(x_array, idx * tau, a)
        u_calculated = u[idx]
        plt.plot(x_array, u_correct, color=colors[i], label=f't={round(idx * tau, 2)} (Точное)')
        plt.plot(x_array, u_calculated, color=colors[i], linestyle='--', label=f't={round(idx * tau, 2)} (Числ.)')
    ax.set_xlabel('x')
    ax.set_ylabel('U(t, x)')
    plt.grid()
    ax.legend()
    plt.show()

# Функция для отображения ошибок
def show_errors(sigma, l, a, T, U, approx_type, init_order):
    N_array, errors1, errors2 = error(sigma, l, a, T, U, approx_type, init_order)
    colors = ['blue', 'green']
    deltaX = np.zeros(np.size(N_array))
    for i in range(np.size(N_array)):
        deltaX[i] = l / N_array[i]
    fig, ax = plt.subplots()
    plt.plot(deltaX, errors1, color=colors[0], label='Явный метод')
    plt.plot(deltaX, errors2, color=colors[1], label='Неявный метод')
    ax.set_title(f'Зависимость ошибки от шага (Нач. условие {init_order}-го порядка)')
    ax.set_xlabel('delta X')
    ax.set_ylabel('Max Error')
    plt.grid()
    ax.legend()
    plt.show()

# Функция для выбора формулы начального условия
def get_second_layer(mode, x, a, tau):
    if mode == 1:
        return d_psi_1st(x, a, tau) # Первый порядок
    else:
        return d_psi_2nd(x, a, tau) # Второй порядок (Тейлор)

# Явная схема
def Explicit_Method(N, K, sigma, tau, a, h, approx_type, init_order):
    if sigma > 1:
        raise Exception("Схема неустойчива (sigma > 1)")

    u = np.zeros((K + 1, N + 1))

    # Начальные условия
    for i in range(N + 1):
        u[0][i] = psi(i * h)
        # Выбор аппроксимации второго начального условия
        u[1][i] = get_second_layer(init_order, i * h, a, tau)

    # Основной цикл по времени
    for k in range(2, K + 1):
        for j in range(1, N):
            u[k][j] = sigma * u[k-1][j-1] + 2 * (1 - sigma) * u[k-1][j] \
                     + sigma * u[k-1][j+1] - u[k-2][j]
        
        # Простейшая аппроксимация граничных условий (как в исходном коде)
        u[k][0] = u[k][1] / (h + 1)
        u[k][N] = u[k][N - 1] / (1 - h)
        
        # Здесь можно добавить switch для approx_type, если нужно изменить ГУ
        
    return u

# Неявная схема
def Implicit_Method(N, K, sigma, tau, a, h, init_order):
    a_j = sigma
    b_j = -(1 + 2 * sigma)
    c_j = sigma
    u = np.zeros((K + 1, N + 1))

    # Начальные условия
    for i in range(0, N + 1):
        u[0][i] = psi(i * h)
        # Выбор аппроксимации второго начального условия
        u[1][i] = get_second_layer(init_order, i * h, a, tau)

    # Основной цикл по времени
    for k in range(2, K + 1):
        matrix = np.zeros((N - 1, N - 1))  # матрица для прогонки
        d = np.zeros(N - 1)                # правая часть
        
        # Первая строка
        matrix[0][0] = b_j + sigma / (h + 1)
        matrix[0][1] = c_j
        d[0] = -2 * u[k - 1][1] + u[k - 2][1]
        
        # Средние строки
        for j in range(1, N - 2):
            matrix[j][j - 1] = a_j
            matrix[j][j] = b_j
            matrix[j][j + 1] = c_j
            d[j] = -2 * u[k - 1][j + 1] + u[k - 2][j + 1]
            
        # Последняя строка
        matrix[N - 2][N - 3] = a_j
        matrix[N - 2][N - 2] = b_j + sigma / (1 - h)
        d[N - 2] = -2 * u[k - 1][N - 1] + u[k - 2][N - 1]
        
        # Решаем СЛАУ методом прогонки
        ans = solve(matrix, d)
        
        if ans is not None:
            u[k][1:N] = ans
        else:
            # На случай если solve вернет None из-за нарушения диаг. преобл.
            u[k][1:N] = u[k-1][1:N] 
            
        u[k][0] = u[k][1] / (h + 1)
        u[k][N] = u[k][N - 1] / (1 - h)
        
    return u

# Главная функция
def main():
    a = 1  # скорость волны
    T = 1  # время моделирования
    N = 50  # число разбиений по x
    K = 50  # Число разбиений по t
    sigma = 0.5  # параметр схемы
    l = np.pi  # длина области
    h = l / N  # шаг по пространству
    tau = T / K

    print("-" * 40)
    # Выбор аппроксимации граничных условий
    choice_bound = int(input(
        "Введите номер аппроксимации ГРАНИЧНЫХ условий:\n"
        "1 - двухточечная первого порядка\n"
        "2 - трехточечная второго порядка\n"
        "3 - двухточечная второго порядка\n"
        "> "
    ))

    match choice_bound:
        case 1:
            approx_type = "two_point_first"
        case 2:
            approx_type = "three_point_second"
        case 3:
            approx_type = "two_point_second"
        case _:
            print("Ошибка выбора ГУ! Используется двухточечная 1-го порядка.")
            approx_type = "two_point_first"

    print("-" * 40)
    # Выбор аппроксимации начального условия
    choice_init = int(input(
        "Введите порядок аппроксимации НАЧАЛЬНОГО условия (u_t):\n"
        "1 - Первый порядок O(tau)\n"
        "2 - Второй порядок O(tau^2) (ряд Тейлора)\n"
        "> "
    ))
    
    if choice_init not in [1, 2]:
        print("Ошибка выбора! Используется 2-й порядок.")
        choice_init = 2

    # tau = np.sqrt(sigma * h ** 2 / a)  # шаг по времени
    # K = int(round(T / tau))  # число шагов по времени
    
    print(f"Параметры: N={N}, K={K}, h={h:.4f}, tau={tau:.4f}")

    # Решение явной схемой
    print("Расчет явной схемой...")
    u1 = Explicit_Method(N, K, sigma, tau, a, h, approx_type, choice_init)
    print("Построение графика явной схемы...")
    show_solution(h, tau, K, l, u1, U, a)

    # Решение неявной схемой
    print("Расчет неявной схемой...")
    u2 = Implicit_Method(N, K, sigma, tau, a, h, choice_init)
    print("Построение графика неявной схемы...")
    show_solution(h, tau, K, l, u2, U, a)

    # Построение графиков погрешностей
    print("Расчет и построение погрешностей...")
    show_errors(sigma, l, a, T, U, approx_type, choice_init)

if __name__ == "__main__":
    main()
