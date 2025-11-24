import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

def progonka(a, b, c, d):
    n = len(d)
    P = np.zeros(n - 1)
    Q = np.zeros(n)  

    # Первый шаг
    P[0] = -c[0] / b[0]
    Q[0] = d[0] / b[0]

    # Середина
    for i in range(1, n - 1):
        P[i] = -c[i] / (b[i] + a[i] * P[i - 1])
        Q[i] = (d[i] - a[i] * Q[i - 1]) / (b[i] + a[i] * P[i - 1])

    # Конец
    Q[n - 1] = (d[n - 1] - a[n - 1] * Q[n - 2]) / (b[n - 1] + a[n - 1] * P[n - 2])

    x = np.zeros(n)
    x[n - 1] = Q[n - 1]

    # Обратный ход
    for i in range(n - 2, -1, -1):
        x[i] = Q[i] + P[i] * x[i + 1]

    return x

a = 1.0
L = np.pi
T = 1.0

def analytical_solution(x, t):
    """Аналитическое решение U(x,t) = sin(x - at) + cos(x + at)"""
    return np.sin(x - a * t) + np.cos(x + a * t)

def initial_condition_u(x):
    """Начальное условие для u(x,0)"""
    return np.sin(x) + np.cos(x)

def initial_condition_ut(x):
    """Начальное условие для u_t(x,0)"""
    return -a * (np.sin(x) + np.cos(x))

# Варианты аппроксимации граничных условий
def solve_left_boundary_condition(u, h, boundary_type, scheme_type):
    """Решение граничного условия на левой границе: u_x(0,t) - u(0,t) = 0"""
    if scheme_type == "first_order":
        # (u1 - u0)/h - u0 = 0 => u0 = u1/(1 + h)
        return u[1] / (1 + h)
    elif scheme_type == "three_point":
        # (-3*u0 + 4*u1 - u2)/(2h) - u0 = 0 => u0 = (4*u1 - u2)/(3 + 2h)
        return (4 * u[1] - u[2]) / (3 + 2 * h)
    elif scheme_type == "two_point_second_order":
        # (u1 - u0)/h + (h/2)*u_xx - u0 = 0
        # Используем u_xx ≈ (u1 - 2*u0 + u(-1))/h², но u(-1) неизвестно
        # Упрощенный вариант: u0 = u1/(1 + h)
        return u[1] / (1 + h)

def solve_right_boundary_condition(u, h, boundary_type, scheme_type):
    """Решение граничного условия на правой границе: u_x(π,t) - u(π,t) = 0"""
    n = len(u)
    if scheme_type == "first_order":
        # (u_n - u_{n-1})/h - u_n = 0 => u_n = u_{n-1}/(1 - h)
        return u[-2] / (1 - h)
    elif scheme_type == "three_point":
        # (3*u_n - 4*u_{n-1} + u_{n-2})/(2h) - u_n = 0 => u_n = (4*u_{n-1} - u_{n-2})/(3 - 2h)
        return (4 * u[-2] - u[-3]) / (3 - 2 * h)
    elif scheme_type == "two_point_second_order":
        # (u_n - u_{n-1})/h - (h/2)*u_xx - u_n = 0
        # Упрощенный вариант: u_n = u_{n-1}/(1 - h)
        return u[-2] / (1 - h)

def cross_scheme(Nx, Nt, initial_approx_order=1):
    """
        Явная схема крест для волнового уравнения
        initial_approx_order: порядок аппроксимации второго начального условия (1 или 2)
    """
    x = np.linspace(0, L, Nx)
    t = np.linspace(0, T, Nt)

    h = x[1] - x[0]
    tau = t[1] - t[0]

    sigma = (a * tau / h)**2  # σ = (aτ/h)²
    if sigma > 1:
        print(f"  ОШИБКА: σ = {sigma:.3f} > 1")
        print(f"  Схема НЕУСТОЙЧИВА! Условие устойчивости: (aτ/h)² ≤ 1")
        print(f"  Требуется: tau ≤ {h/a:.6f} (сейчас tau = {tau:.6f})")
        return x, t, None

    print(f"  Устойчивость: σ = {sigma:.3f} ≤ 1 - схема устойчива")

    u = np.zeros((Nt, Nx))

    # Первое начальное условие
    u[0, :] = initial_condition_u(x)

    # Второе начальное условие (аппроксимация u_t(x,0))
    if initial_approx_order == 1:
        # Первый порядок: u_t ≈ (u¹ - u⁰)/τ
        u[1, :] = u[0, :] + tau * initial_condition_ut(x)
    else:
        # Второй порядок: используем разложение в ряд Тейлора
        u_xx = np.zeros(Nx)
        for j in range(1, Nx-1):
            u_xx[j] = (u[0, j+1] - 2*u[0, j] + u[0, j-1]) / h**2
        # Граничные точки для u_xx (упрощенная аппроксимация)
        u_xx[0] = (u[0, 1] - 2*u[0, 0] + u[0, 0]) / h**2
        u_xx[-1] = (u[0, -1] - 2*u[0, -1] + u[0, -2]) / h**2
        
        u[1, :] = u[0, :] + tau * initial_condition_ut(x) + 0.5 * (a * tau)**2 * u_xx
    
    # Основной цикл
    for k in range(1, Nt-1):
        # Внутренние точки
        for j in range(1, Nx-1):
            u[k+1, j] = 2*u[k, j] - u[k-1, j] + sigma * (u[k, j+1] - 2*u[k, j] + u[k, j-1])
        
        # Граничные условия
        u[k+1, 0] = solve_left_boundary_condition(u[k+1], h, "robin", "first_order")
        u[k+1, -1] = solve_right_boundary_condition(u[k+1], h, "robin", "first_order")
    
    return x, t, u

def implicit_scheme(Nx, Nt, initial_approx_order=1):
    """
    Неявная схема для волнового уравнения
    """
    x = np.linspace(0, L, Nx)
    t = np.linspace(0, T, Nt)
    
    h = x[1] - x[0]
    tau = t[1] - t[0]
    
    u = np.zeros((Nt, Nx))
    
    # Первое начальное условие
    u[0, :] = initial_condition_u(x)
    
    # Второе начальное условие
    if initial_approx_order == 1:
        u[1, :] = u[0, :] + tau * initial_condition_ut(x)
    else:
        u_xx = np.zeros(Nx)
        for j in range(1, Nx-1):
            u_xx[j] = (u[0, j+1] - 2*u[0, j] + u[0, j-1]) / h**2
        u_xx[0] = (u[0, 1] - 2*u[0, 0] + u[0, 0]) / h**2
        u_xx[-1] = (u[0, -1] - 2*u[0, -1] + u[0, -2]) / h**2
        
        u[1, :] = u[0, :] + tau * initial_condition_ut(x) + 0.5 * (a * tau)**2 * u_xx
    
    sigma = (a * tau / h)**2
    
    # Матрица для неявной схемы
    for k in range(1, Nt-1):
        n = Nx
        A = np.zeros(n)
        B = np.zeros(n)
        C = np.zeros(n)
        D = np.zeros(n)
        
        # Внутренние точки
        for j in range(1, n-1):
            A[j] = -sigma
            B[j] = 2 + 2*sigma
            C[j] = -sigma
            D[j] = 4*u[k, j] - 2*u[k-1, j] - sigma*(u[k, j+1] - 2*u[k, j] + u[k, j-1])
        
        # Граничные условия (левая)
        B[0] = 1
        C[0] = -1/(1 + h)  # для first_order
        D[0] = 0
        
        # Граничные условия (правая)
        A[-1] = -1/(1 - h)  # для first_order
        B[-1] = 1
        D[-1] = 0
        
        # Решение системы методом прогонки
        u[k+1, :] = progonka(A, B, C, D)
    
    return x, t, u

def calculate_errors(u_numeric, u_analytic):
    """Вычисление абсолютной погрешности"""
    return np.abs(u_numeric - u_analytic)

def plot_solutions(x, t, u_cross, u_imp, u_anal, initial_order):
    """Построение графиков решений"""
    times_to_plot = [0, len(t)//4, len(t)//2, -1]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, time_idx in enumerate(times_to_plot):
        ax = axes[idx]
        ax.plot(x, u_anal[time_idx, :], 'k-', linewidth=2, label='Аналитическое')
        ax.plot(x, u_cross[time_idx, :], 'r--', label='Схема крест')
        ax.plot(x, u_imp[time_idx, :], 'b-.', label='Неявная схема')
        
        ax.set_xlabel('Координата x')
        ax.set_ylabel('u(x,t)')
        ax.set_title(f'Решение при t = {t[time_idx]:.2f}')
        ax.legend()
        ax.grid(True)
    
    plt.suptitle(f'РЕШЕНИЕ ВОЛНОВОГО УРАВНЕНИЯ (порядок аппроксимации: {initial_order})\n'
                'U(x,t) = sin(x - at) + cos(x + at)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_errors_comparison(x, t, u_cross, u_imp, u_anal, initial_order):
    """Построение графиков погрешностей"""
    times_to_plot = [len(t)//4, len(t)//2, -1]
    
    plt.figure(figsize=(15, 5))
    
    for idx, time_idx in enumerate(times_to_plot):
        plt.subplot(1, 3, idx+1)
        
        error_cross = calculate_errors(u_cross[time_idx, :], u_anal[time_idx, :])
        error_imp = calculate_errors(u_imp[time_idx, :], u_anal[time_idx, :])
        
        plt.plot(x, error_cross, 'r-', label='Схема крест', alpha=0.7)
        plt.plot(x, error_imp, 'b-', label='Неявная схема', alpha=0.7)
        
        plt.xlabel('Координата x')
        plt.ylabel('Абсолютная погрешность')
        plt.title(f'Погрешности при t = {t[time_idx]:.2f}')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')
    
    plt.suptitle(f'ПОГРЕШНОСТИ ЧИСЛЕННЫХ МЕТОДОВ (порядок аппроксимации: {initial_order})', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def convergence_study(initial_order=1):
    """Исследование сходимости"""
    Nx_list = [20, 40, 80, 160, 200]
    Nt_list = [200, 400, 800, 1600, 3200]
    
    errors_cross = []
    errors_imp = []
    
    print(f"\nИССЛЕДОВАНИЕ СХОДИМОСТИ (порядок аппроксимации: {initial_order}):")
    print("Nx\tNt\tСхема крест\tНеявная схема")
    print("-" * 50)
    
    for Nx, Nt in zip(Nx_list, Nt_list):
        x = np.linspace(0, L, Nx)
        t = np.linspace(0, T, Nt)
        X, T_grid = np.meshgrid(x, t)
        u_anal = analytical_solution(X, T_grid)
        
        x_cross, t_cross, u_cross = cross_scheme(Nx, Nt, initial_order)
        x_imp, t_imp, u_imp = implicit_scheme(Nx, Nt, initial_order)
        
        error_cross = np.max(calculate_errors(u_cross, u_anal))
        error_imp = np.max(calculate_errors(u_imp, u_anal))
        
        errors_cross.append(error_cross)
        errors_imp.append(error_imp)
        
        print(f"{Nx}\t{Nt}\t{error_cross:.2e}\t{error_imp:.2e}")
    
    h_list = [L/(Nx-1) for Nx in Nx_list]
    
    plt.figure(figsize=(12, 8))
    plt.loglog(h_list, errors_cross, 'ro-', label='Схема крест', linewidth=2, markersize=8)
    plt.loglog(h_list, errors_imp, 'bs-', label='Неявная схема', linewidth=2, markersize=8)
    
    h_ref = np.array(h_list)
    plt.loglog(h_ref, 0.1*h_ref**2, 'k--', label='Теоретическая сходимость O(h²)', linewidth=2)
    
    plt.xlabel('Шаг по пространству h', fontsize=12)
    plt.ylabel('Максимальная погрешность ε_max', fontsize=12)
    plt.title(f'СХОДИМОСТЬ МЕТОДОВ (порядок аппроксимации: {initial_order})', 
             fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    print("РЕШЕНИЕ ВОЛНОВОГО УРАВНЕНИЯ")
    print("=" * 50)
    print(f"Параметры: a = {a}, L = π, T = {T}")
    print(f"Аналитическое решение: U(x,t) = sin(x - {a}t) + cos(x + {a}t)")
    print(f"Граничные условия: u_x(0,t) - u(0,t) = 0, u_x(π,t) - u(π,t) = 0")

    Nx = 100
    Nt = 1000

    # Исследование для разных порядков аппроксимации начального условия
    for initial_order in [1, 2]:
        print(f"\n" + "="*60)
        print(f"РАСЧЕТ С ПОРЯДКОМ АППРОКСИМАЦИИ: {initial_order}")
        print("="*60)

        print(f"Сетка: Nx = {Nx}, Nt = {Nt}")

        x_analytical = np.linspace(0, L, Nx)
        t_analytical = np.linspace(0, T, Nt)
        X, T_grid = np.meshgrid(x_analytical, t_analytical)
        u_analytical = analytical_solution(X, T_grid)

        print("\n1. Схема крест...")
        x_cross, t_cross, u_cross = cross_scheme(Nx, Nt, initial_order)

        print("2. Неявная схема...")
        x_imp, t_imp, u_imp = implicit_scheme(Nx, Nt, initial_order)

        print("\nПОГРЕШНОСТИ В РАЗЛИЧНЫЕ МОМЕНТЫ ВРЕМЕНИ:")
        time_indices = [0, len(t_analytical)//4, len(t_analytical)//2, -1]
        
        print("Метод\t\tt=0.00\tt=0.25\tt=0.50\tt=1.00")
        print("-" * 50)
        
        for method_name, u_num in [("Схема крест", u_cross), ("Неявная", u_imp)]:
            errors = []
            for time_idx in time_indices:
                max_error = np.max(calculate_errors(u_num[time_idx, :], u_analytical[time_idx, :]))
                errors.append(f"{max_error:.2e}")
            print(f"{method_name}\t{errors[0]}\t{errors[1]}\t{errors[2]}\t{errors[3]}")
        
        plot_solutions(x_cross, t_cross, u_cross, u_imp, u_analytical, initial_order)
        plot_errors_comparison(x_cross, t_cross, u_cross, u_imp, u_analytical, initial_order)
        
        convergence_study(initial_order)

if __name__ == "__main__":
    main()