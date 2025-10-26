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


a = 1.0  # Положительная константа
L = np.pi
T = 1.0

def analytical_solution(x, t):
    return np.exp(-a * t) * np.cos(x)

# Начальное решениие (t=0)
def initial_condition(x):
    return np.cos(x)

# Варианты граничных условий
def left_boundary(t, boundary_type="dirichlet", scheme_type="standard"):
    """Граничное условие на левой границе"""
    if boundary_type == "dirichlet":
        return np.exp(-a * t)
    elif boundary_type == "neumann":
        # du/dx = 0 на левой границе
        if scheme_type == "first_order":
            return 0.0
        elif scheme_type == "three_point":
            return 0.0
        elif scheme_type == "two_point_second_order":
            return 0.0
        else:
            return 0.0
    elif boundary_type == "robin":
        # u + 0.5*du/dx = exp(-a*t)
        if scheme_type == "first_order":
            return np.exp(-a * t)
        elif scheme_type == "three_point":
            return np.exp(-a * t)
        elif scheme_type == "two_point_second_order":
            return np.exp(-a * t)
        else:
            return np.exp(-a * t)
    else:  # mixed
        # 2*u + du/dx = 2*exp(-a*t)
        if scheme_type == "first_order":
            return 2 * np.exp(-a * t)
        elif scheme_type == "three_point":
            return 2 * np.exp(-a * t)
        elif scheme_type == "two_point_second_order":
            return 2 * np.exp(-a * t)
        else:
            return 2 * np.exp(-a * t)

def right_boundary(t, boundary_type="dirichlet", scheme_type="standard"):
    """Граничное условие на правой границе"""
    if boundary_type == "dirichlet":
        return -np.exp(-a * t)
    elif boundary_type == "neumann":
        # du/dx = 0 на правой границе
        if scheme_type == "first_order":
            return 0.0
        elif scheme_type == "three_point":
            return 0.0
        elif scheme_type == "two_point_second_order":
            return 0.0
        else:
            return 0.0
    elif boundary_type == "robin":
        # u - 0.3*du/dx = -exp(-a*t)
        if scheme_type == "first_order":
            return -np.exp(-a * t)
        elif scheme_type == "three_point":
            return -np.exp(-a * t)
        elif scheme_type == "two_point_second_order":
            return -np.exp(-a * t)
        else:
            return -np.exp(-a * t)
    else:  # mixed
        # 2*u - du/dx = -2*exp(-a*t)
        if scheme_type == "first_order":
            return -2 * np.exp(-a * t)
        elif scheme_type == "three_point":
            return -2 * np.exp(-a * t)
        elif scheme_type == "two_point_second_order":
            return -2 * np.exp(-a * t)
        else:
            return -2 * np.exp(-a * t)

def solve_left_boundary_condition(u, u_next, h, t, boundary_type, scheme_type):
    """Решение граничного условия на левой границе с различными аппроксимациями"""
    if boundary_type == "dirichlet":
        return left_boundary(t, boundary_type, scheme_type)
    
    elif boundary_type == "neumann":
        if scheme_type == "first_order":
            # du/dx = 0 => (u1 - u0)/h = 0 => u0 = u1
            return u_next[1]
        elif scheme_type == "three_point":
            # du/dx = 0 => (-3*u0 + 4*u1 - u2)/(2h) = 0 => u0 = (4*u1 - u2)/3
            return (4 * u_next[1] - u_next[2]) / 3
        elif scheme_type == "two_point_second_order":
            # du/dx = 0 с учетом второй производной
            d2u = (u[1] - 2*u[0] + left_boundary(t, "dirichlet")) / h**2
            return u_next[1] - (h**2 / 2) * d2u
    
    elif boundary_type == "robin":
        # u + 0.5*du/dx = exp(-a*t)
        gamma = np.exp(-a * t)
        if scheme_type == "first_order":
            # u0 + 0.5*(u1 - u0)/h = gamma
            return (gamma - 0.5 * u_next[1] / h) / (1 - 0.5 / h)
        elif scheme_type == "three_point":
            # u0 + 0.5*(-3*u0 + 4*u1 - u2)/(2h) = gamma
            return (gamma - 0.5 * (4*u_next[1] - u_next[2]) / (2*h)) / (1 - 0.5 * (-3) / (2*h))
        elif scheme_type == "two_point_second_order":
            # u0 + 0.5*(u1 - u0)/h + (0.5*h/2)*d²u/dx² = gamma
            d2u = (u[1] - 2*u[0] + left_boundary(t, "dirichlet")) / h**2
            return (gamma - 0.5 * u_next[1] / h - 0.5 * h * d2u / 2) / (1 - 0.5 / h)
    
    else:  # mixed
        # 2*u + du/dx = 2*exp(-a*t)
        gamma = 2 * np.exp(-a * t)
        if scheme_type == "first_order":
            # 2*u0 + (u1 - u0)/h = gamma
            return (gamma - u_next[1] / h) / (2 - 1/h)
        elif scheme_type == "three_point":
            # 2*u0 + (-3*u0 + 4*u1 - u2)/(2h) = gamma
            return (gamma - (4*u_next[1] - u_next[2]) / (2*h)) / (2 - 3/(2*h))
        elif scheme_type == "two_point_second_order":
            # 2*u0 + (u1 - u0)/h + (h/2)*d²u/dx² = gamma
            d2u = (u[1] - 2*u[0] + left_boundary(t, "dirichlet")) / h**2
            return (gamma - u_next[1] / h - h * d2u / 2) / (2 - 1/h)

def solve_right_boundary_condition(u, u_next, h, t, boundary_type, scheme_type):
    """Решение граничного условия на правой границе с различными аппроксимациями"""
    n = len(u_next)
    
    if boundary_type == "dirichlet":
        return right_boundary(t, boundary_type, scheme_type)
    
    elif boundary_type == "neumann":
        if scheme_type == "first_order":
            # du/dx = 0 => (u_n - u_{n-1})/h = 0 => u_n = u_{n-1}
            return u_next[-2]
        elif scheme_type == "three_point":
            # du/dx = 0 => (3*u_n - 4*u_{n-1} + u_{n-2})/(2h) = 0 => u_n = (4*u_{n-1} - u_{n-2})/3
            return (4 * u_next[-2] - u_next[-3]) / 3
        elif scheme_type == "two_point_second_order":
            # du/dx = 0 с учетом второй производной
            d2u = (u[-2] - 2*u[-1] + right_boundary(t, "dirichlet")) / h**2
            return u_next[-2] - (h**2 / 2) * d2u
    
    elif boundary_type == "robin":
        # u - 0.3*du/dx = -exp(-a*t)
        gamma = -np.exp(-a * t)
        if scheme_type == "first_order":
            # u_n - 0.3*(u_n - u_{n-1})/h = gamma
            return (gamma + 0.3 * u_next[-2] / h) / (1 + 0.3 / h)
        elif scheme_type == "three_point":
            # u_n - 0.3*(3*u_n - 4*u_{n-1} + u_{n-2})/(2h) = gamma
            return (gamma + 0.3 * (4*u_next[-2] - u_next[-3]) / (2*h)) / (1 + 0.3 * 3 / (2*h))
        elif scheme_type == "two_point_second_order":
            # u_n - 0.3*(u_n - u_{n-1})/h - (0.3*h/2)*d²u/dx² = gamma
            d2u = (u[-2] - 2*u[-1] + right_boundary(t, "dirichlet")) / h**2
            return (gamma + 0.3 * u_next[-2] / h - 0.3 * h * d2u / 2) / (1 + 0.3 / h)
    
    else:  # mixed
        # 2*u - du/dx = -2*exp(-a*t)
        gamma = -2 * np.exp(-a * t)
        if scheme_type == "first_order":
            # 2*u_n - (u_n - u_{n-1})/h = gamma
            return (gamma + u_next[-2] / h) / (2 + 1/h)
        elif scheme_type == "three_point":
            # 2*u_n - (3*u_n - 4*u_{n-1} + u_{n-2})/(2h) = gamma
            return (gamma + (4*u_next[-2] - u_next[-3]) / (2*h)) / (2 + 3/(2*h))
        elif scheme_type == "two_point_second_order":
            # 2*u_n - (u_n - u_{n-1})/h - (h/2)*d²u/dx² = gamma
            d2u = (u[-2] - 2*u[-1] + right_boundary(t, "dirichlet")) / h**2
            return (gamma + u_next[-2] / h - h * d2u / 2) / (2 + 1/h)

def explicit_scheme(Nx, Nt, boundary_type="dirichlet", scheme_type="standard"):
    '''
        Nx: количество шагов по X
        Nt: количество шагов по T
        boundary_type: тип граничных условий ("dirichlet", "neumann", "robin", "mixed")
        scheme_type: тип аппроксимации ("standard", "first_order", "three_point", "two_point_second_order")
    '''
    x = np.linspace(0, L, Nx)
    t = np.linspace(0, T, Nt)

    h = x[1] - x[0]
    tau = t[1] - t[0]

    sigma = a * tau / h**2
    if sigma > 0.5:
        print(f"  Явная схема: sigma = {sigma:.3f} > 0.5 - может быть неустойчива")

    u = np.zeros((Nt, Nx))

    u[0, :] = initial_condition(x)

    for k in range(0, Nt-1):
        # Внутренние точки
        for j in range(1, Nx-1):
            u[k+1, j] = sigma * u[k, j+1] + (1 - 2*sigma)*u[k, j] + sigma * u[k, j-1]
        
        # Граничные условия
        if scheme_type == "standard":
            u[k+1, 0] = left_boundary(t[k+1], boundary_type, scheme_type)
            u[k+1, -1] = right_boundary(t[k+1], boundary_type, scheme_type)
        else:
            u[k+1, 0] = solve_left_boundary_condition(u[k], u[k+1], h, t[k+1], boundary_type, scheme_type)
            u[k+1, -1] = solve_right_boundary_condition(u[k], u[k+1], h, t[k+1], boundary_type, scheme_type)

    return x, t, u

def implicit_scheme(Nx, Nt, boundary_type="dirichlet", scheme_type="standard"):
    '''
        Nx: количество шагов по X
        Nt: количество шагов по T
        boundary_type: тип граничных условий ("dirichlet", "neumann", "robin", "mixed")
        scheme_type: тип аппроксимации ("standard", "first_order", "three_point", "two_point_second_order")
    '''
    x = np.linspace(0, L, Nx)
    t = np.linspace(0, T, Nt)

    h = x[1] - x[0]
    tau = t[1] - t[0]

    sigma = a * tau / h**2

    u = np.zeros((Nt, Nx))

    u[0, :] = initial_condition(x)

    As = np.ones(Nx-2) * (sigma)
    As[0] = 0
    Bs = np.ones(Nx-2) * (-(1 + 2*sigma))
    Cs = np.ones(Nx-3) * (sigma)

    for k in range(0, Nt-1):
        Ds = u[k, 1:-1].copy()

        # Граничные условия
        if scheme_type == "standard":
            Ds[0] += sigma * left_boundary(t[k+1], boundary_type, scheme_type)
            Ds[-1] += sigma * right_boundary(t[k+1], boundary_type, scheme_type)
        else:
            left_bc = solve_left_boundary_condition(u[k], u[k+1], h, t[k+1], boundary_type, scheme_type)
            right_bc = solve_right_boundary_condition(u[k], u[k+1], h, t[k+1], boundary_type, scheme_type)
            Ds[0] += sigma * left_bc
            Ds[-1] += sigma * right_bc

        Ds = -Ds
        u[k+1, 1:-1] = progonka(As, Bs, Cs, Ds)

        if scheme_type == "standard":
            u[k+1, 0] = left_boundary(t[k+1], boundary_type, scheme_type)
            u[k+1, -1] = right_boundary(t[k+1], boundary_type, scheme_type)
        else:
            u[k+1, 0] = solve_left_boundary_condition(u[k], u[k+1], h, t[k+1], boundary_type, scheme_type)
            u[k+1, -1] = solve_right_boundary_condition(u[k], u[k+1], h, t[k+1], boundary_type, scheme_type)

    return x, t, u

def crank_nicolson_scheme(Nx, Nt, boundary_type="dirichlet", scheme_type="standard"):
    '''
        Nx: количество шагов по X
        Nt: количество шагов по T
        boundary_type: тип граничных условий ("dirichlet", "neumann", "robin", "mixed")
        scheme_type: тип аппроксимации ("standard", "first_order", "three_point", "two_point_second_order")
    '''
    x = np.linspace(0, L, Nx)
    t = np.linspace(0, T, Nt)

    h = x[1] - x[0]
    tau = t[1] - t[0]

    sigma = a * tau / (2 * h**2)

    u = np.zeros((Nt, Nx))

    u[0, :] = initial_condition(x)

    As = np.ones(Nx-2) * (-sigma)
    As[0] = 0
    Bs = np.ones(Nx-2) * (1 + 2*sigma)
    Cs = np.ones(Nx-3) * (-sigma)

    main_diag_B = np.ones(Nx-2) * (1 - 2*sigma)
    off_diag_B = np.ones(Nx-3) * sigma
    B = diags([off_diag_B, main_diag_B, off_diag_B], [-1, 0, 1], format='csc')

    for k in range(0, Nt-1):
        Ds = B.dot(u[k, 1:-1])

        # Граничные условия
        if scheme_type == "standard":
            Ds[0] += sigma * (left_boundary(t[k+1], boundary_type, scheme_type) + left_boundary(t[k], boundary_type, scheme_type))
            Ds[-1] += sigma * (right_boundary(t[k+1], boundary_type, scheme_type) + right_boundary(t[k], boundary_type, scheme_type))
        else:
            left_bc_k1 = solve_left_boundary_condition(u[k], u[k+1], h, t[k+1], boundary_type, scheme_type)
            right_bc_k1 = solve_right_boundary_condition(u[k], u[k+1], h, t[k+1], boundary_type, scheme_type)
            left_bc_k = solve_left_boundary_condition(u[k-1] if k > 0 else u[k], u[k], h, t[k], boundary_type, scheme_type)
            right_bc_k = solve_right_boundary_condition(u[k-1] if k > 0 else u[k], u[k], h, t[k], boundary_type, scheme_type)
            
            Ds[0] += sigma * (left_bc_k1 + left_bc_k)
            Ds[-1] += sigma * (right_bc_k1 + right_bc_k)

        u[k+1, 1:-1] = progonka(As, Bs, Cs, Ds)

        if scheme_type == "standard":
            u[k+1, 0] = left_boundary(t[k+1], boundary_type, scheme_type)
            u[k+1, -1] = right_boundary(t[k+1], boundary_type, scheme_type)
        else:
            u[k+1, 0] = solve_left_boundary_condition(u[k], u[k+1], h, t[k+1], boundary_type, scheme_type)
            u[k+1, -1] = solve_right_boundary_condition(u[k], u[k+1], h, t[k+1], boundary_type, scheme_type)

    return x, t, u

def calculate_errors(u_numeric, u_analytic):
    return np.abs(u_numeric - u_analytic)

def plot_solutions(x, t, u_exp, u_imp, u_cn, u_anal):
    times_to_plot = [0, len(t)//4, len(t)//2, -1]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for idx, time_idx in enumerate(times_to_plot):
        ax = axes[idx]
        ax.plot(x, u_anal[time_idx, :], 'k-', linewidth=2, label='Аналитическое')
        ax.plot(x, u_exp[time_idx, :], 'r--', label='Явная')
        ax.plot(x, u_imp[time_idx, :], 'b-.', label='Неявная')
        ax.plot(x, u_cn[time_idx, :], 'g:', label='Кранк-Николсон')

        ax.set_xlabel('Координата x')
        ax.set_ylabel('Температура u(x,t)')
        ax.set_title(f'Распределение температуры вдоль стержня при t = {t[time_idx]:.2f}')
        ax.legend()
        ax.grid(True)

    plt.suptitle('СРАВНЕНИЕ ЧИСЛЕННЫХ РЕШЕНИЙ С АНАЛИТИЧЕСКИМ\n'
                'Зависимость: температура u(x) в различные моменты времени t', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_errors_comparison(x, t, u_exp, u_imp, u_cn, u_anal):
    times_to_plot = [len(t)//4, len(t)//2, -1]

    plt.figure(figsize=(15, 5))

    for idx, time_idx in enumerate(times_to_plot):
        plt.subplot(1, 3, idx+1)

        error_exp = calculate_errors(u_exp[time_idx, :], u_anal[time_idx, :])
        error_imp = calculate_errors(u_imp[time_idx, :], u_anal[time_idx, :])
        error_cn = calculate_errors(u_cn[time_idx, :], u_anal[time_idx, :])

        plt.plot(x, error_exp, 'r-', label='Явная', alpha=0.7)
        plt.plot(x, error_imp, 'b-', label='Неявная', alpha=0.7)
        plt.plot(x, error_cn, 'g-', label='Кранк-Николсон', alpha=0.7)

        plt.xlabel('Координата x')
        plt.ylabel('Абсолютная погрешность')
        plt.title(f'Погрешности методов при t = {t[time_idx]:.2f}')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')

    plt.suptitle('СРАВНЕНИЕ ПОГРЕШНОСТЕЙ ЧИСЛЕННЫХ МЕТОДОВ\n'
                'Зависимость: погрешность ε(x) в различные моменты времени t', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def convergence_study():
    Nx_list = [20, 40, 80, 160, 200]
    Nt_list = [200, 800, 3200, 12800, 20000]

    errors_exp = []
    errors_imp = []
    errors_cn = []

    print("\nИССЛЕДОВАНИЕ СХОДИМОСТИ:")
    print("Nx\tNt\tЯвная\t\tНеявная\t\tКранк-Николсон")
    print("-" * 60)

    for Nx, Nt in zip(Nx_list, Nt_list):
        x = np.linspace(0, L, Nx)
        t = np.linspace(0, T, Nt)
        X, T_grid = np.meshgrid(x, t)
        u_anal = analytical_solution(X, T_grid)

        x_exp, t_exp, u_exp = explicit_scheme(Nx, Nt)
        x_imp, t_imp, u_imp = implicit_scheme(Nx, Nt)
        x_cn, t_cn, u_cn = crank_nicolson_scheme(Nx, Nt)

        if u_exp is not None:
            error_exp = np.max(calculate_errors(u_exp, u_anal))
        else:
            error_exp = np.nan
 
        error_imp = np.max(calculate_errors(u_imp, u_anal))
        error_cn = np.max(calculate_errors(u_cn, u_anal))

        errors_exp.append(error_exp)
        errors_imp.append(error_imp)
        errors_cn.append(error_cn)

        print(f"{Nx}\t{Nt}\t{error_exp:.2e}\t{error_imp:.2e}\t{error_cn:.2e}")

    h_list = [L/(Nx-1) for Nx in Nx_list]

    plt.figure(figsize=(12, 8))
    plt.loglog(h_list, errors_exp, 'ro-', label='Явная схема', linewidth=2, markersize=8)
    plt.loglog(h_list, errors_imp, 'bs-', label='Неявная схема', linewidth=2, markersize=8)
    plt.loglog(h_list, errors_cn, 'g^-', label='Кранк-Николсон', linewidth=2, markersize=8)

    h_ref = np.array(h_list)
    plt.loglog(h_ref, 0.1*h_ref**2, 'k--', label='Теоретическая сходимость O(h²)', linewidth=2)

    plt.xlabel('Шаг по пространству h', fontsize=12)
    plt.ylabel('Максимальная погрешность ε_max', fontsize=12)
    plt.title('ИССЛЕДОВАНИЕ СХОДИМОСТИ ЧИСЛЕННЫХ СХЕМ\n'
             'Зависимость: максимальная погрешность ε_max от шага сетки h', 
             fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    for i, h in enumerate(h_list):
        if i > 0:

            if errors_exp[i-1] > 0 and errors_exp[i] > 0:
                order_exp = np.log(errors_exp[i-1]/errors_exp[i]) / np.log(h_list[i-1]/h_list[i])

    plt.tight_layout()
    plt.show()

def main():
    print("РЕШЕНИЕ УРАВНЕНИЯ ТЕПЛОПРОВОДНОСТИ")
    print("=" * 50)
    print(f"Параметры: a = {a}, L = π, T = {T}")
    print(f"Аналитическое решение: U(x,t) = exp(-{a}*t) * cos(x)")

    Nx = 100
    Nt = 10000

    print(f"\nОСНОВНОЙ РАСЧЕТ:")
    print(f"Сетка: Nx = {Nx}, Nt = {Nt}")

    x_analytical = np.linspace(0, L, Nx)
    t_analytical = np.linspace(0, T, Nt)
    X, T_grid = np.meshgrid(x_analytical, t_analytical)
    u_analytical = analytical_solution(X, T_grid)

    print("\n1. Явная схема...")
    x_exp, t_exp, u_exp = explicit_scheme(Nx, Nt)

    print("2. Неявная схема...")
    x_imp, t_imp, u_imp = implicit_scheme(Nx, Nt)

    print("3. Схема Кранка-Николсона...")
    x_cn, t_cn, u_cn = crank_nicolson_scheme(Nx, Nt)

    print("\nПОГРЕШНОСТИ В РАЗЛИЧНЫЕ МОМЕНТЫ ВРЕМЕНИ:")
    time_indices = [0, len(t_analytical)//4, len(t_analytical)//2, -1]

    print("Метод\t\tt=0.00\tt=0.25\tt=0.50\tt=1.00")
    print("-" * 50)

    for method_name, u_num in [("Явная", u_exp), ("Неявная", u_imp), ("Кранк-Николсон", u_cn)]:
        if u_num is not None:
            errors = []
            for time_idx in time_indices:
                max_error = np.max(calculate_errors(u_num[time_idx, :], u_analytical[time_idx, :]))
                errors.append(f"{max_error:.2e}")
            print(f"{method_name}\t\t{errors[0]}\t{errors[1]}\t{errors[2]}\t{errors[3]}")

    if u_exp is not None:
        plot_solutions(x_exp, t_exp, u_exp, u_imp, u_cn, u_analytical)
        plot_errors_comparison(x_exp, t_exp, u_exp, u_imp, u_cn, u_analytical)

    convergence_study()


if __name__ == "__main__":
    main()