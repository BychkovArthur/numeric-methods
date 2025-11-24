import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [10, 8]

# ==============================================================================
# АНАЛИТИЧЕСКОЕ РЕШЕНИЕ И НАЧАЛЬНЫЕ ДАННЫЕ
# ==============================================================================

def U(x, t, a):
    # Аналитическое решение
    return np.sin(x - a * t) + np.cos(x + a * t)

def psi(x):
    return np.sin(x) + np.cos(x)

def get_second_layer(mode, x, a, tau):
    # phi = u_t(x,0)
    phi = -a * (np.sin(x) + np.cos(x))
    if mode == 1:
        return psi(x) + tau * phi
    else:
        # u_tt = a^2 * u_xx = -a^2*(sin+cos)
        u_tt = (a ** 2) * (-psi(x))
        return psi(x) + tau * phi + (tau ** 2 / 2) * u_tt


# ==============================================================================
# ГРАНИЧНЫЕ УСЛОВИЯ (u_x - u = 0 или Дирихле)
# ==============================================================================

def apply_boundary(u_layer, h, a, l, t, bc_type):
    """
    Реализация граничных условий:
      1 — двухточечная, 1-го порядка   (u_x - u = 0)
      2 — трёхточечная, 2-го порядка   (u_x - u = 0)
      3 — двухточечная, 2-го порядка (условная, центральная)
      4 — Дирихле: u(0,t)=U(0,t), u(l,t)=U(l,t)
    """
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


def Explicit_Method(N, K, sigma, tau, a, h, l, init_order, bc_type):
    """Явная схема крест"""
    if sigma > 1.000001:
        print(f"ВНИМАНИЕ: Явная схема неустойчива! (Sigma = {sigma:.4f} > 1)")

    u = np.zeros((K + 1, N + 1))
    x_vals = np.linspace(0, l, N+1)
    u[0] = psi(x_vals)
    u[1] = get_second_layer(init_order, x_vals, a, tau)

    for k in range(1, K):
        t_next = (k + 1) * tau

        if bc_type == 4:
            # ДИРИХЛЕ — точно как в первом коде
            u[k+1][1:N] = (
                2*(1 - sigma)*u[k][1:N]
                + sigma*(u[k][0:N-1] + u[k][2:N+1])
                - u[k-1][1:N]
            )
            u[k+1][0] = U(0, t_next, a)
            u[k+1][N] = U(l, t_next, a)
        else:
            # Остальные типы через аппроксимации
            u[k] = apply_boundary(u[k], h, a, l, t_next, bc_type)
            u[k+1][1:N] = (
                2*(1 - sigma)*u[k][1:N]
                + sigma*(u[k][0:N-1] + u[k][2:N+1])
                - u[k-1][1:N]
            )
            u[k+1] = apply_boundary(u[k+1], h, a, l, t_next, bc_type)
    return u


def Implicit_Method(N, K, sigma, tau, a, h, l, init_order, bc_type):
    """Неявная схема"""
    u = np.zeros((K + 1, N + 1))
    x_vals = np.linspace(0, l, N+1)
    u[0] = psi(x_vals)
    u[1] = get_second_layer(init_order, x_vals, a, tau)

    matrix = np.zeros((N - 1, N - 1))
    np.fill_diagonal(matrix, -(1 + 2 * sigma))
    np.fill_diagonal(matrix[1:], sigma)
    np.fill_diagonal(matrix[:, 1:], sigma)

    for k in range(1, K):
        t_next = (k + 1) * tau

        if bc_type == 4:
            # ДИРИХЛЕ — как в первом коде
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
            u[k] = apply_boundary(u[k], h, a, l, t_next, bc_type)
            d = -2 * u[k][1:N] + u[k-1][1:N]
            u_inner = solve_tridiagonal(matrix, d)
            u[k+1][1:N] = u_inner
            u[k+1] = apply_boundary(u[k+1], h, a, l, t_next, bc_type)
    return u


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

    try:
        choice_init = int(input("Порядок второго начального условия (1 или 2): "))
    except ValueError:
        choice_init = 2

    print("\nТипы граничных условий:")
    print(" 1 — двухточечная, 1-го порядка (Neumann/Robin: u_x - u = 0)")
    print(" 2 — трёхточечная, 2-го порядка (Neumann/Robin: u_x - u = 0)")
    print(" 3 — двухточечная, 2-го порядка (центральная)")
    print(" 4 — Дирихле (u = U)")
    try:
        bc_type = int(input("Выберите вариант аппроксимации (1,2,3,4): "))
    except ValueError:
        bc_type = 4

    print("\n1. Явная схема...")
    u_exp = Explicit_Method(N_in, K_in, sigma, tau, a, h, l, choice_init, bc_type)

    print("\n2. Неявная схема...")
    u_imp = Implicit_Method(N_in, K_in, sigma, tau, a, h, l, choice_init, bc_type)

    show_all_methods(h, tau, K_in, l, u_exp, u_imp, U, a, "Сравнение методов")


if __name__ == "__main__":
    main()
