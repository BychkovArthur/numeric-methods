import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [10, 8]

# ==============================================================================
# АНАЛИТИЧЕСКОЕ РЕШЕНИЕ И НАЧАЛЬНЫЕ ДАННЫЕ
# ==============================================================================

def U(x, t, a):
    return np.sin(x - a * t) + np.cos(x + a * t)

def psi(x):
    return np.sin(x) + np.cos(x)

def get_second_layer(mode, x, a, tau):
    phi = -a * (np.sin(x) + np.cos(x))
    if mode == 1:
        return psi(x) + tau * phi
    else:
        u_tt = (a ** 2) * (-psi(x))
        return psi(x) + tau * phi + (tau ** 2 / 2) * u_tt

def bar(a):
    if a == 3:
        return 1
    return a

# ==============================================================================
# ГРАНИЧНЫЕ УСЛОВИЯ (u_x - u = 0 или Дирихле)
# ==============================================================================

def apply_boundary(u_layer, h, a, l, t, bc_type):
    N = len(u_layer) - 1

    if bc_type == 1:
        u_layer[0] = u_layer[1] / (1 + h)
        u_layer[N] = u_layer[N-1] / (1 - h)

    elif bc_type == 2:
        u_layer[0] = (4*u_layer[1] - u_layer[2]) / (3 + 2*h)
        u_layer[N] = (4*u_layer[N-1] - u_layer[N-2]) / (3 - 2*h)

    elif bc_type == 3:
        u_layer[0] = u_layer[1] / (1 + 2*h)
        u_layer[N] = u_layer[N-1] / (1 - 2*h)

    elif bc_type == 4:
        u_layer[0] = U(0, t, a)
        u_layer[N] = U(l, t, a)
    return u_layer


# ==============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==============================================================================

def solve_tridiagonal(a, b):
    n = len(b)
    p = np.zeros(n)
    q = np.zeros(n)
    p[0] = -a[0][1] / a[0][0]
    q[0] = b[0] / a[0][0]
    for i in range(1, n - 1):
        denom = a[i][i] + a[i][i - 1] * p[i - 1]
        p[i] = -a[i][i + 1] / denom
        q[i] = (b[i] - a[i][i - 1] * q[i - 1]) / denom
    denom_last = a[n - 1][n - 1] + a[n - 1][n - 2] * p[n - 2]
    q[n - 1] = (b[n - 1] - a[n - 1][n - 2] * q[n - 2]) / denom_last
    x = np.zeros(n)
    x[-1] = q[-1]
    for i in range(n - 2, -1, -1):
        x[i] = p[i] * x[i + 1] + q[i]
    return x


def build_implicit_matrix_robin(N, sigma, h, bc_type):
    A = np.zeros((N - 1, N - 1))
    np.fill_diagonal(A, -(1 + 2 * sigma))
    np.fill_diagonal(A[1:], sigma)
    np.fill_diagonal(A[:, 1:], sigma)

    if bc_type == 1:
        A[0, 0] = sigma / (1 + h) - (1 + 2 * sigma)
        A[-1, -1] = sigma / (1 - h) - (1 + 2 * sigma)

    elif bc_type == 2:
        A[0, 0] = 4 * sigma / (3 + 2*h) - (1 + 2 * sigma)
        A[0, 1] = sigma - sigma / (3 + 2*h)
        A[-1, -2] = sigma - sigma / (3 - 2*h)
        A[-1, -1] = 4 * sigma / (3 - 2*h) - (1 + 2 * sigma)

    elif bc_type == 3:
        A[0, 0] = sigma / (1 + 2*h) - (1 + 2 * sigma)
        A[-1, -1] = sigma / (1 - 2*h) - (1 + 2 * sigma)

    return A


# ==============================================================================
# ЯВНАЯ СХЕМА
# ==============================================================================

def Explicit_Method(N, K, sigma, tau, a, h, l, init_order, bc_type):
    if sigma > 1.000001:
        print(f"ВНИМАНИЕ: Явная схема неустойчива! (Sigma = {sigma:.4f} > 1)")

    u = np.zeros((K + 1, N + 1))
    x_vals = np.linspace(0, l, N+1)
    u[0] = psi(x_vals)
    u[1] = get_second_layer(init_order, x_vals, a, tau)

    for k in range(1, K):
        t_next = (k + 1) * tau

        if bc_type == 4:
            u[k+1][1:N] = (
                2*(1 - sigma)*u[k][1:N]
                + sigma*(u[k][0:N-1] + u[k][2:N+1])
                - u[k-1][1:N]
            )
            u[k+1][0] = U(0, t_next, a)
            u[k+1][N] = U(l, t_next, a)
        else:
            u[k] = apply_boundary(u[k], h, a, l, t_next, bc_type)
            u[k+1][1:N] = (
                2*(1 - sigma)*u[k][1:N]
                + sigma*(u[k][0:N-1] + u[k][2:N+1])
                - u[k-1][1:N]
            )
            u[k+1] = apply_boundary(u[k+1], h, a, l, t_next, bc_type)
    return u


# ==============================================================================
# НЕЯВНАЯ СХЕМА
# ==============================================================================

def Implicit_Method(N, K, sigma, tau, a, h, l, init_order, bc_type):
    u = np.zeros((K + 1, N + 1))
    x_vals = np.linspace(0, l, N+1)
    u[0] = psi(x_vals)
    u[1] = get_second_layer(init_order, x_vals, a, tau)

    if bc_type == 4:
        matrix = np.zeros((N - 1, N - 1))
        np.fill_diagonal(matrix, -(1 + 2 * sigma))
        np.fill_diagonal(matrix[1:], sigma)
        np.fill_diagonal(matrix[:, 1:], sigma)

        for k in range(1, K):
            t_next = (k + 1) * tau
            d = -2 * u[k][1:N] + u[k-1][1:N]
            u_bound_left = U(0, t_next, a)
            d[0] -= sigma * u_bound_left
            u_bound_right = U(l, t_next, a)
            d[-1] -= sigma * u_bound_right
            u_inner = solve_tridiagonal(matrix, d)
            u[k+1][1:N] = u_inner
            u[k+1][0] = u_bound_left
            u[k+1][N] = u_bound_right
    else:
        matrix = build_implicit_matrix_robin(N, sigma, h, bc_type)

        for k in range(1, K):
            t_next = (k + 1) * tau
            d = -2 * u[k][1:N] + u[k-1][1:N]
            u_inner = solve_tridiagonal(matrix, d)
            u[k+1][1:N] = u_inner
            
            if bc_type == 1:
                u[k+1][0] = u[k+1][1] / (1 + h)
                u[k+1][N] = u[k+1][N-1] / (1 - h)
            elif bc_type == 2:
                u[k+1][0] = (4*u[k+1][1] - u[k+1][2]) / (3 + 2*h)
                u[k+1][N] = (4*u[k+1][N-1] - u[k+1][N-2]) / (3 - 2*h)
            elif bc_type == 3:
                u[k+1][0] = u[k+1][1] / (1 + 2*h)
                u[k+1][N] = u[k+1][N-1] / (1 - 2*h)

    return u


# ==============================================================================
# АНАЛИЗ ОШИБОК
# ==============================================================================

def compute_errors(l, a, T, init_order, bc_type):
    """
    Вычисляет ошибки для разных размеров сетки.
    Фиксируем sigma = 0.5 для корректного сравнения.
    """
    sigma_target = 0.5
    N_array = [10, 20, 40, 80]
    
    errors_explicit = []
    errors_implicit = []
    h_vals = []
    
    for cur_N in N_array:
        cur_h = l / cur_N
        h_vals.append(cur_h)
        
        # Рассчитываем tau так, чтобы sigma была постоянной
        cur_tau = cur_h * np.sqrt(sigma_target) / a
        cur_K = int(round(T / cur_tau))
        
        # Пересчитываем реальное sigma
        cur_sigma = (a * cur_tau / cur_h) ** 2
        
        print(f"  Сетка N={cur_N}, K={cur_K}, h={cur_h:.4f}, tau={cur_tau:.4f}, sigma={cur_sigma:.4f}")
        
        # Решаем явным и неявным методами
        u_exp = Explicit_Method(cur_N, cur_K, cur_sigma, cur_tau, a, cur_h, l, init_order, bc_type)
        u_imp = Implicit_Method(cur_N, cur_K, cur_sigma, cur_tau, a, cur_h, l, init_order, bc_type)
        
        # Вычисляем ошибку в момент времени T/2
        target_k = int(cur_K / 2)
        t_val = target_k * cur_tau
        
        x_arr = np.linspace(0, l, cur_N + 1)
        u_exact = U(x_arr, t_val, a)
        
        # Максимальная ошибка
        err_exp = np.max(np.abs(u_exact - u_exp[target_k]))
        err_imp = np.max(np.abs(u_exact - u_imp[target_k]))
        
        errors_explicit.append(err_exp)
        errors_implicit.append(err_imp)
    
    return N_array, h_vals, errors_explicit, errors_implicit


def show_errors(l, a, T, init_order, bc_type):
    """Строит график зависимости ошибки от шага сетки"""
    print("\n" + "="*60)
    print("АНАЛИЗ СХОДИМОСТИ")
    print("="*60)
    
    N_array, h_vals, errors_exp, errors_imp = compute_errors(l, a, T, init_order, bc_type)
    
    fig, ax = plt.subplots()
    ax.loglog(h_vals, errors_exp, 'o-', color='red', label='Явный метод', linewidth=2, markersize=8)
    ax.loglog(h_vals, errors_imp, 's-', color='blue', label='Неявный метод', linewidth=2, markersize=8)
    
    # Добавляем линии эталонного порядка для сравнения
    h_ref = np.array(h_vals)
    if init_order == 1:
        ax.loglog(h_ref, errors_exp[0] * (h_ref/h_ref[0])**1, '--', 
                  color='gray', alpha=0.5, label='O(h)')
    else:
        ax.loglog(h_ref, errors_exp[0] * (h_ref/h_ref[0])**2, '--', 
                  color='gray', alpha=0.5, label='O(h²)')
    
    ax.set_xlabel('Шаг по пространству h (log)', fontsize=12)
    ax.set_ylabel('Максимальная ошибка (log)', fontsize=12)
    ax.set_title(f'Сходимость методов\n(нач. усл. порядок {init_order}, гран. усл. тип {bc_type})', 
                 fontsize=13)
    ax.grid(True, which="both", ls="-", alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.show()
    
    # Выводим таблицу ошибок
    print("\nТаблица ошибок:")
    print(f"{'N':>6} {'h':>10} {'Явный':>15} {'Неявный':>15}")
    print("-" * 50)
    for i, N in enumerate(N_array):
        print(f"{N:>6} {h_vals[i]:>10.6f} {errors_exp[i]:>15.6e} {errors_imp[i]:>15.6e}")


# ==============================================================================
# ВИЗУАЛИЗАЦИЯ
# ==============================================================================

def show_all_methods(h, tau, K, l, u_explicit, u_implicit, U, a, title):
    x_array = np.linspace(0, l, u_explicit.shape[1])
    steps = [int(K*0.3), int(K*0.5), int(K*0.8)]
    for k in steps:
        if k >= K:
            k = K - 1
        t = k * tau
        u_exact = U(x_array, t, a)
        fig, ax = plt.subplots()
        ax.plot(x_array, u_exact, 'k-', linewidth=3, label=f"Аналитическое t={t:.2f}")
        ax.plot(x_array, u_explicit[k], 'r--', label=f"Явное t={t:.2f}")
        ax.plot(x_array, u_implicit[k], 'b-.', label=f"Неявное t={t:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("u(x,t)")
        ax.set_title(f"Сравнение методов, t={t:.2f}")
        ax.legend()
        ax.grid(True)
        plt.show()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    a = 1.0
    l = np.pi
    T = 4.0

    print("=" * 60)
    print("Решение волнового уравнения u_tt = a^2 u_xx")
    print("Граничные условия: u_x - u = 0, или Дирихле (по выбору)")
    print("=" * 60)

    try:
        N_in = int(input("Число разбиений по X (например, 50): "))
        K_in = int(input("Число шагов по T (например, 200): "))
    except ValueError:
        N_in, K_in = 50, 200

    h = l / N_in
    tau = T / K_in
    sigma = (a * tau / h) ** 2

    print(f"\nПараметры:")
    print(f"  h = {h:.4f}")
    print(f"  tau = {tau:.4f}")
    print(f"  sigma = {sigma:.4f}\n")

    if sigma > 1.0:
        print("⚠️  Для явной схемы желательно sigma ≤ 1")

    try:
        choice_init = int(input("Порядок второго начального условия (1 или 2): "))
    except ValueError:
        choice_init = 2

    print(f'{choice_init=}')

    print("\nТипы граничных условий:")
    print(" 1 — двухточечная, 1-го порядка (Robin: u_x - u = 0)")
    print(" 2 — трёхточечная, 2-го порядка (Robin: u_x - u = 0)")
    print(" 3 — двухточечная, 2-го порядка (центральная)")
    print(" 4 — Дирихле (u = U)")
    try:
        bc_type = bar(int(input("Выберите вариант аппроксимации (1,2,3,4): ")))
    except ValueError:
        bc_type = 4

    print("\n1. Явная схема...")
    u_exp = Explicit_Method(N_in, K_in, sigma, tau, a, h, l, choice_init, bc_type)

    print("\n2. Неявная схема...")
    u_imp = Implicit_Method(N_in, K_in, sigma, tau, a, h, l, choice_init, bc_type)

    show_all_methods(h, tau, K_in, l, u_exp, u_imp, U, a, "Сравнение методов")

    print("\n3. Анализ сходимости...")
    show_errors(l, a, T, choice_init, bc_type)


if __name__ == "__main__":
    main()
