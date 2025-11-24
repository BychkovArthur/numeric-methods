import numpy as np
import matplotlib.pyplot as plt

# Настройка размера графиков
plt.rcParams['figure.figsize'] = [10, 8]

# ==============================================================================
# ФИЗИКА И ТОЧНОЕ РЕШЕНИЕ
# ==============================================================================

# Точное решение U(x,t)
def U(x, t, a):
    return np.sin(x - a * t) + np.cos(x + a * t)

# Начальное условие u(x, 0)
def psi(x):
    return np.sin(x) + np.cos(x)

# Второе начальное условие u_t(x, 0) - производная
# u^1 = u^0 + tau * u_t + ...
def get_second_layer(mode, x, a, tau):
    # Истинная производная по времени phi(x) при t=0:
    # U_t = -a*cos(x-at) - a*sin(x+at) -> при t=0 -> -a(cos(x) + sin(x))
    phi = -a * (np.sin(x) + np.cos(x))
    
    if mode == 1:
        # 1-й порядок: u1 = u0 + tau * phi
        return psi(x) + tau * phi
    else:
        # 2-й порядок (Тейлор): u1 = u0 + tau*phi + (tau^2/2)*u_tt
        # Из уравнения волны: u_tt = a^2 * u_xx
        # u_xx = -sin(x) - cos(x) = -psi(x)
        u_tt = (a**2) * (-psi(x))
        return psi(x) + tau * phi + (tau**2 / 2) * u_tt

# ==============================================================================
# ЧИСЛЕННЫЕ МЕТОДЫ
# ==============================================================================

def solve_tridiagonal(a, b):
    """Метод прогонки (без проверок для скорости)"""
    n = len(b)
    p = np.zeros(n)
    q = np.zeros(n)
    
    # Прямой ход
    p[0] = -a[0][1] / a[0][0]
    q[0] = b[0] / a[0][0]
    for i in range(1, n - 1):
        denom = a[i][i] + a[i][i - 1] * p[i - 1]
        p[i] = -a[i][i + 1] / denom
        q[i] = (b[i] - a[i][i - 1] * q[i - 1]) / denom
        
    # Последний элемент
    denom_last = a[n - 1][n - 1] + a[n - 1][n - 2] * p[n - 2]
    q[n - 1] = (b[n - 1] - a[n - 1][n - 2] * q[n - 2]) / denom_last
    
    # Обратный ход
    x = np.zeros(n)
    x[-1] = q[-1]
    for i in range(n - 2, -1, -1):
        x[i] = p[i] * x[i + 1] + q[i]
    return x

def Explicit_Method(N, K, sigma, tau, a, h, init_order):
    """Явная схема"""
    if sigma > 1.000001:
        print(f"ВНИМАНИЕ: Явная схема неустойчива! (Sigma = {sigma:.4f} > 1)")

    u = np.zeros((K + 1, N + 1))

    # 1. Начальные условия
    x_vals = np.linspace(0, N*h, N+1)
    u[0] = psi(x_vals)
    u[1] = get_second_layer(init_order, x_vals, a, tau)

    # 2. Расчет по времени
    for k in range(1, K):
        # Внутренние точки (j от 1 до N-1)
        # Формула: u^{k+1} = 2(1-sigma)u^k + sigma(u^k_{j-1} + u^k_{j+1}) - u^{k-1}
        u[k+1][1:N] = 2 * (1 - sigma) * u[k][1:N] + \
                      sigma * (u[k][0:N-1] + u[k][2:N+1]) - \
                      u[k-1][1:N]
        
        # 3. Граничные условия (Дирихле) - берем точное значение на границах
        t_next = (k + 1) * tau
        u[k+1][0] = U(0, t_next, a)
        u[k+1][N] = U(N*h, t_next, a)
        
    return u

def Implicit_Method(N, K, sigma, tau, a, h, init_order):
    """Неявная схема"""
    u = np.zeros((K + 1, N + 1))

    # Коэффициенты матрицы (для уравнения: sigma*u_{j-1} - (1+2sigma)*u_j + sigma*u_{j+1} = RHS)
    # A_coeff = sigma
    # B_coeff = -(1 + 2 * sigma)
    # C_coeff = sigma
    
    # 1. Начальные условия
    x_vals = np.linspace(0, N*h, N+1)
    u[0] = psi(x_vals)
    u[1] = get_second_layer(init_order, x_vals, a, tau)

    # Матрица остаётся постоянной (за исключением правой части)
    matrix = np.zeros((N - 1, N - 1))
    
    # Заполнение диагоналей матрицы
    # Главная диагональ
    np.fill_diagonal(matrix, -(1 + 2 * sigma))
    # Побочные диагонали
    np.fill_diagonal(matrix[1:], sigma)   # под главной
    np.fill_diagonal(matrix[:, 1:], sigma) # над главной

    # 2. Расчет по времени
    for k in range(1, K):
        t_next = (k + 1) * tau
        
        # Правая часть СЛАУ: d = -2*u^k + u^{k-1}
        d = -2 * u[k][1:N] + u[k-1][1:N]
        
        # Учет граничных условий в правой части (перенос известных слагаемых)
        # Для первой строки (j=1): u_{j-1} это u_0 - известно
        u_bound_left = U(0, t_next, a)
        d[0] -= sigma * u_bound_left 
        
        # Для последней строки (j=N-1): u_{j+1} это u_N - известно
        u_bound_right = U(N*h, t_next, a)
        d[-1] -= sigma * u_bound_right
        
        # Решение СЛАУ
        u_inner = solve_tridiagonal(matrix, d)
        
        # Запись решения
        u[k+1][1:N] = u_inner
        u[k+1][0] = u_bound_left
        u[k+1][N] = u_bound_right
        
    return u

# ==============================================================================
# АНАЛИЗ И ГРАФИКИ
# ==============================================================================

def error(l, a, T, U, init_order):
    # Для графика ошибок мы создаем НОВЫЕ сетки, чтобы показать сходимость
    # Мы фиксируем число Куранта (sigma), чтобы тест был корректным
    base_N = 10
    sigma_target = 0.5 # Целевое число Куранта для теста
    
    N_array = [10, 20, 40, 80]
    errors1 = []
    errors2 = []
    
    for cur_N in N_array:
        cur_h = l / cur_N
        
        # Рассчитываем tau так, чтобы sigma была константой (0.5)
        # sigma = (a * tau / h)^2  => sqrt(sigma) = a * tau / h  => tau = h * sqrt(sigma) / a
        cur_tau = cur_h * np.sqrt(sigma_target) / a
        cur_K = int(round(T / cur_tau))
        
        # Пересчитываем точное sigma
        cur_real_sigma = (a * cur_tau / cur_h) ** 2
        
        u1 = Explicit_Method(cur_N, cur_K, cur_real_sigma, cur_tau, a, cur_h, init_order)
        u2 = Implicit_Method(cur_N, cur_K, cur_real_sigma, cur_tau, a, cur_h, init_order)
        
        # Сравнение в момент времени T/2
        target_k = int(cur_K / 2)
        t_val = target_k * cur_tau
        
        x_arr = np.linspace(0, l, cur_N + 1)
        u_exact = U(x_arr, t_val, a)
        
        errors1.append(np.max(np.abs(u_exact - u1[target_k])))
        errors2.append(np.max(np.abs(u_exact - u2[target_k])))
        
    return N_array, errors1, errors2

def show_solution(h, tau, K, l, u, U, a, title):
    x_array = np.linspace(0, l, u.shape[1])
    fig, ax = plt.subplots()
    
    # Рисуем 3 момента времени
    steps = [int(K*0.2), int(K*0.5), int(K*0.8)]
    colors = ['blue', 'green', 'red']
    
    for i, k in enumerate(steps):
        if k >= K: k = K - 1
        t = k * tau
        
        u_exact = U(x_array, t, a)
        u_calc = u[k]
        
        ax.plot(x_array, u_exact, color=colors[i], alpha=0.4, linewidth=3, label=f'Точн t={t:.2f}')
        ax.plot(x_array, u_calc, color=colors[i], linestyle='--', label=f'Числ t={t:.2f}')
        
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('U(x,t)')
    ax.grid(True)
    ax.legend()
    plt.show()

def show_errors(l, a, T, U, init_order):
    N_array, errors1, errors2 = error(l, a, T, U, init_order)
    h_vals = [l/n for n in N_array] # шаг h
    
    fig, ax = plt.subplots()
    # Логарифмический масштаб лучше показывает порядок ошибки
    ax.loglog(h_vals, errors1, 'o-', label='Явный метод')
    ax.loglog(h_vals, errors2, 's-', label='Неявный метод')
    
    ax.set_xlabel('Шаг по пространству h (log)')
    ax.set_ylabel('Максимальная ошибка (log)')
    ax.set_title(f'Сходимость методов (Нач. усл. порядок {init_order})')
    ax.grid(True, which="both", ls="-")
    ax.legend()
    plt.show()

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    a = 1.0       # скорость волны
    l = np.pi     # длина области
    T = 4.0       # время моделирования

    print("="*50)
    print("МОДЕЛИРОВАНИЕ ВОЛНОВОГО УРАВНЕНИЯ")
    print("="*50)

    # Ввод параметров пользователя
    try:
        N_in = int(input("Введите число разбиений по X (например, 50): "))
        K_in = int(input("Введите число разбиений по T (например, 200): "))
    except ValueError:
        print("Ошибка ввода! Используются значения по умолчанию (50, 200).")
        N_in, K_in = 50, 200

    h = l / N_in
    tau = T / K_in
    
    # Главный параметр устойчивости (квадрат числа Куранта)
    sigma = (a * tau / h) ** 2
    
    print(f"\nПараметры:")
    print(f"  Шаг h   = {h:.4f}")
    print(f"  Шаг tau = {tau:.4f}")
    print(f"  Sigma (C^2) = {sigma:.4f}")
    
    if sigma > 1:
        print("\n!!! ВНИМАНИЕ: Sigma > 1. Явная схема развалится.")
        print("    Увеличьте K или уменьшите N.")

    # Выбор порядка начального условия
    print("-" * 40)
    try:
        choice_init = int(input("Порядок второго начального условия (1 или 2): "))
    except ValueError:
        choice_init = 2
    if choice_init not in [1, 2]: choice_init = 2

    # Решение явной схемой
    print("\n1. Явная схема...")
    u1 = Explicit_Method(N_in, K_in, sigma, tau, a, h, choice_init)
    show_solution(h, tau, K_in, l, u1, U, a, "Явная схема")

    # Решение неявной схемой
    print("\n2. Неявная схема...")
    u2 = Implicit_Method(N_in, K_in, sigma, tau, a, h, choice_init)
    show_solution(h, tau, K_in, l, u2, U, a, "Неявная схема")

    # Ошибки
    print("\n3. Анализ сходимости...")
    show_errors(l, a, T, U, choice_init)

if __name__ == "__main__":
    main()
