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

# Граничное решение x=0
def left_boundary(t):
    return np.exp(-a * t)

# Граничное решение x=L
def right_boundary(t):
    return -np.exp(-a * t)

def explicit_scheme(Nx, Nt):
    '''
        Nx: количество шагов по X
        Nt: количество шагов по T
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
        u[k+1, 0] = left_boundary(t[k+1])
        u[k+1, -1] = right_boundary(t[k+1])

        for j in range(1, Nx-1):
            u[k+1, j] = sigma * u[k, j+1] + (1 - 2*sigma)*u[k, j] + sigma * u[k, j-1]

    return x, t, u

def implicit_scheme(Nx, Nt):
    '''
        Nx: количество шагов по X
        Nt: количество шагов по T
    '''
    x = np.linspace(0, L, Nx)
    t = np.linspace(0, T, Nt)

    h = x[1] - x[0]
    tau = t[1] - t[0]

    sigma = a * tau / h**2

    u = np.zeros((Nt, Nx))

    u[0, :] = initial_condition(x)

    '''
        a_1=0       b_1 c_1  0   0  ... 0          | d_1      j = 1
                    a_2 b_2 c_2  0  ... 0          | d_2
                     0  a_3 b_3 c_3 ... 0          | d_3      j = 2...{n-2}
                    ...                            | ...
                    ...                            | ...
        c_{n-1}=0    0   0  ... a_{n-1} b_{n-1}    | d_{n-1}  j = n-1
    
    Всего N-1 уравнение. Это позволит найти переменные u_j^{k+1} | j = 1...n-1
    d_1     = -( u_1^k     + sigma * f0(t^{k+1}) )
    d_j     = -u_j^k                                             | j = 2...n-2
    d_{n-1} = -( u_{n-1}^k + sigma * f1(t^{k+1}) )
    
    На главной диагонали стоят   b_j = -(1+2*sigma)
    Под главной диагональю стоят a_j = sigma
    Над главной диагональю стоят c_j = sigma
    '''

    As = np.ones(Nx-2) * (sigma)
    As[0] = 0  # У меня кривой метод прогонки. Нужно добавлять ноль в начало
    Bs = np.ones(Nx-2) * (-(1 + 2*sigma))
    Cs = np.ones(Nx-3) * (sigma)

    for k in range(0, Nt-1):
        # Базово D[j] = u[j]
        Ds = u[k, 1:-1].copy()

        # Но к D[0] и D[-1] надо прибавить еще граничные условия
        Ds[0] += sigma * left_boundary(t[k+1])
        Ds[-1] += sigma * right_boundary(t[k+1])

        Ds = -Ds
        u[k+1, 1:-1] = progonka(As, Bs, Cs, Ds)

        u[k+1, 0] = left_boundary(t[k+1])
        u[k+1, -1] = right_boundary(t[k+1])

    return x, t, u

def crank_nicolson_scheme(Nx, Nt):
    '''
        Nx: количество шагов по X
        Nt: количество шагов по T
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

        Ds[0] += sigma * (left_boundary(t[k+1]) + left_boundary(t[k]))
        Ds[-1] += sigma * (right_boundary(t[k+1]) + right_boundary(t[k]))

        u[k+1, 1:-1] = progonka(As, Bs, Cs, Ds)

        u[k+1, 0] = left_boundary(t[k+1])
        u[k+1, -1] = right_boundary(t[k+1])

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