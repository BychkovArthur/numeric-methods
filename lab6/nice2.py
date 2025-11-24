import numpy as np
import matplotlib.pyplot as plt

# Настройка размера графиков
plt.rcParams['figure.figsize'] = [10, 8]

# ==============================================================================
# ФИЗИКА И ТОЧНОЕ РЕШЕНИЕ
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
        u_tt = (a**2) * (-psi(x))
        return psi(x) + tau * phi + (tau**2 / 2) * u_tt

# ==============================================================================
# ЧИСЛЕННЫЕ МЕТОДЫ
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

def Explicit_Method(N, K, sigma, tau, a, h, init_order):
    if sigma > 1.000001:
        print(f"ВНИМАНИЕ: Явная схема неустойчива! (Sigma = {sigma:.4f} > 1)")
    u = np.zeros((K + 1, N + 1))
    x_vals = np.linspace(0, N*h, N+1)
    u[0] = psi(x_vals)
    u[1] = get_second_layer(init_order, x_vals, a, tau)
    for k in range(1, K):
        u[k+1][1:N] = 2*(1 - sigma)*u[k][1:N] + sigma*(u[k][0:N-1] + u[k][2:N+1]) - u[k-1][1:N]
        t_next = (k + 1) * tau
        u[k+1][0] = U(0, t_next, a)
        u[k+1][N] = U(N*h, t_next, a)
    return u

def Implicit_Method(N, K, sigma, tau, a, h, init_order):
    u = np.zeros((K + 1, N + 1))
    x_vals = np.linspace(0, N*h, N+1)
    u[0] = psi(x_vals)
    u[1] = get_second_layer(init_order, x_vals, a, tau)
    matrix = np.zeros((N - 1, N - 1))
    np.fill_diagonal(matrix, -(1 + 2 * sigma))
    np.fill_diagonal(matrix[1:], sigma)
    np.fill_diagonal(matrix[:, 1:], sigma)
    for k in range(1, K):
        t_next = (k + 1) * tau
        d = -2 * u[k][1:N] + u[k-1][1:N]
        u_bound_left = U(0, t_next, a)
        d[0] -= sigma * u_bound_left
        u_bound_right = U(N*h, t_next, a)
        d[-1] -= sigma * u_bound_right
        u_inner = solve_tridiagonal(matrix, d)
        u[k+1][1:N] = u_inner
        u[k+1][0] = u_bound_left
        u[k+1][N] = u_bound_right
    return u

# ==============================================================================
# АНАЛИЗ И ВИЗУАЛИЗАЦИЯ
# ==============================================================================

def error(l, a, T, U, init_order):
    base_N = 10
    sigma_target = 0.5 
    N_array = [10, 20, 40, 80]
    errors1 = []
    errors2 = []
    for cur_N in N_array:
        cur_h = l / cur_N
        cur_tau = cur_h * np.sqrt(sigma_target) / a
        cur_K = int(round(T / cur_tau))
        cur_real_sigma = (a * cur_tau / cur_h) ** 2
        u1 = Explicit_Method(cur_N, cur_K, cur_real_sigma, cur_tau, a, cur_h, init_order)
        u2 = Implicit_Method(cur_N, cur_K, cur_real_sigma, cur_tau, a, cur_h, init_order)
        target_k = int(cur_K / 2)
        t_val = target_k * cur_tau
        x_arr = np.linspace(0, l, cur_N + 1)
        u_exact = U(x_arr, t_val, a)
        errors1.append(np.max(np.abs(u_exact - u1[target_k])))
        errors2.append(np.max(np.abs(u_exact - u2[target_k])))
    return N_array, errors1, errors2

def show_all_methods(h, tau, K, l, u_explicit, u_implicit, U, a, title):
    """
    Для каждого выбранного момента времени рисуем отдельную картинку:
    на одной — явный, неявный и аналитический методы.
    """
    x_array = np.linspace(0, l, u_explicit.shape[1])
    steps = [int(K*0.3), int(K*0.5), int(K*0.8)]
    for k in steps:
        if k >= K:
            k = K - 1
        t = k * tau
        u_exact = U(x_array, t, a)
        u_expl = u_explicit[k]
        u_impl = u_implicit[k]
        fig, ax = plt.subplots()
        ax.plot(x_array, u_exact, 'k-', linewidth=3, label=f"Аналитическое, t={t:.2f}")
        ax.plot(x_array, u_expl, 'r--', label=f"Явное, t={t:.2f}")
        ax.plot(x_array, u_impl, 'b-.', label=f"Неявное, t={t:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("u(x,t)")
        ax.set_title(f"Сравнение методов при t = {t:.2f}")
        ax.grid(True)
        ax.legend()
        plt.show()

def show_errors(l, a, T, U, init_order):
    N_array, errors1, errors2 = error(l, a, T, U, init_order)
    h_vals = [l/n for n in N_array]
    fig, ax = plt.subplots()
    ax.loglog(h_vals, errors1, 'o-', label='Явный метод')
    ax.loglog(h_vals, errors2, 's-', label='Неявный метод')
    ax.set_xlabel('Шаг по пространству h (log)')
    ax.set_ylabel('Макс. ошибка (log)')
    ax.set_title(f'Сходимость методов (нач. усл. порядок {init_order})')
    ax.grid(True, which="both", ls="-")
    ax.legend()
    plt.show()

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    a = 1.0
    l = np.pi
    T = 4.0

    print("="*50)
    print("МОДЕЛИРОВАНИЕ ВОЛНОВОГО УРАВНЕНИЯ")
    print("="*50)

    try:
        N_in = int(input("Введите число разбиений по X (например, 50): "))
        K_in = int(input("Введите число разбиений по T (например, 200): "))
    except ValueError:
        print("Ошибка ввода! Используются значения по умолчанию (50, 200).")
        N_in, K_in = 50, 200

    h = l / N_in
    tau = T / K_in
    sigma = (a * tau / h) ** 2

    print(f"\nПараметры:")
    print(f"  Шаг h   = {h:.4f}")
    print(f"  Шаг tau = {tau:.4f}")
    print(f"  Sigma (C^2) = {sigma:.4f}")

    if sigma > 1:
        print("\n!!! ВНИМАНИЕ: Sigma > 1. Явная схема развалится.")
        print("    Увеличьте K или уменьшите N.")

    print("-" * 40)
    try:
        choice_init = int(input("Порядок второго начального условия (1 или 2): "))
    except ValueError:
        choice_init = 2
    if choice_init not in [1, 2]: 
        choice_init = 2

    print("\n1. Явная схема...")
    u1 = Explicit_Method(N_in, K_in, sigma, tau, a, h, choice_init)

    print("\n2. Неявная схема...")
    u2 = Implicit_Method(N_in, K_in, sigma, tau, a, h, choice_init)

    show_all_methods(h, tau, K_in, l, u1, u2, U, a, "Сравнение методов")

    print("\n3. Анализ сходимости...")
    show_errors(l, a, T, U, choice_init)

if __name__ == "__main__":
    main()
