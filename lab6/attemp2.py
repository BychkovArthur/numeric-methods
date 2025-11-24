import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

a = 1.0
L = np.pi
T = 2.0

def exact_solution(x, t):
    return np.sin(x - a*t) + np.cos(x + a*t)

def initial_u(x):
    return np.sin(x) + np.cos(x)

def initial_ut(x):
    return -a * (np.sin(x) + np.cos(x))

def exact_du_dx(x, t):
    return np.cos(x - a*t) - np.sin(x + a*t)

# --- метод прогонки ---
def progonka(a_coef, b, c, d):
    n = len(d)
    P = np.zeros(n-1)
    Q = np.zeros(n)
    P[0] = -c[0] / b[0]
    Q[0] = d[0] / b[0]
    for i in range(1, n-1):
        P[i] = -c[i] / (b[i] + a_coef[i-1] * P[i-1])
        Q[i] = (d[i] - a_coef[i-1] * Q[i-1]) / (b[i] + a_coef[i-1] * P[i-1])
    Q[n-1] = (d[n-1] - a_coef[n-2] * Q[n-2]) / (b[n-1] + a_coef[n-2] * P[n-2])
    x = np.zeros(n)
    x[n-1] = Q[n-1]
    for i in range(n-2, -1, -1):
        x[i] = Q[i] + P[i] * x[i+1]
    return x

# Вычисление ошибок
def compute_errors(u_num, x, t):
    u_ex = exact_solution(x, t)
    err_max = np.max(np.abs(u_num - u_ex))
    err_L2 = np.sqrt(np.sum((u_num - u_ex)**2) * (x[1]-x[0]))
    return err_max, err_L2

# Применение граничных условий: u_x(0,t) - u(0,t) = 0 и u_x(pi,t) - u(pi,t) = 0
def apply_boundary_conditions(u, dx, t, bc_type='two_point_first'):
    """
    bc_type:
    - 'two_point_first': двухточечная аппроксимация первого порядка
    - 'three_point_second': трехточечная аппроксимация второго порядка
    - 'two_point_second': двухточечная аппроксимация второго порядка (с фиктивным узлом)
    """
    if bc_type == 'two_point_first':
        # Левая граница: (u[1] - u[0])/dx - u[0] = 0
        # => u[0] = u[1]/(1 + dx)
        u[0] = u[1] / (1 + dx)
        
        # Правая граница: (u[-1] - u[-2])/dx - u[-1] = 0
        # => u[-1] = u[-2]/(1 + dx)
        u[-1] = u[-2] / (1 + dx)
        
    elif bc_type == 'three_point_second':
        # Левая граница: (-3*u[0] + 4*u[1] - u[2])/(2*dx) - u[0] = 0
        # => u[0] = (4*u[1] - u[2])/(3 + 2*dx)
        u[0] = (4*u[1] - u[2]) / (3 + 2*dx)
        
        # Правая граница: (3*u[-1] - 4*u[-2] + u[-3])/(2*dx) - u[-1] = 0
        # => u[-1] = (4*u[-2] - u[-3])/(3 + 2*dx)
        u[-1] = (4*u[-2] - u[-3]) / (3 + 2*dx)
        
    elif bc_type == 'two_point_second':
        # Двухточечная аппроксимация второго порядка с использованием фиктивного узла
        # Для левой границы: используем центральную разность
        # (u[1] - u[-1])/(2*dx) - u[0] = 0, где u[-1] - фиктивный узел
        # Из уравнения: u_tt = a^2 * u_xx
        # На границе: u[-1] = u[1] - 2*dx*u[0]
        # Подставляем это в схему
        u[0] = u[1] / (1 + dx)
        u[-1] = u[-2] / (1 + dx)
        
    return u

def compute_u1_first_order(u0, v0, dx, dt, bc_type):
    """
    Аппроксимация первого порядка для u^1:
    u^1 = u^0 + dt * u_t^0
    """
    u1 = u0.copy()
    u1 = u0 + dt * v0
    u1 = apply_boundary_conditions(u1, dx, dt, bc_type)
    return u1

def compute_u1_second_order(u0, v0, dx, dt, bc_type):
    """
    Аппроксимация второго порядка для u^1:
    u^1 = u^0 + dt * u_t^0 + (dt^2/2) * u_tt^0
    где u_tt^0 = a^2 * u_xx^0 (из уравнения)
    """
    u1 = u0.copy()
    
    # Вычисляем u_xx для внутренних точек
    u_xx_0 = np.zeros_like(u0)
    u_xx_0[1:-1] = (u0[2:] - 2*u0[1:-1] + u0[:-2]) / dx**2
    
    # Для граничных точек используем односторонние разности
    # Левая граница
    if bc_type == 'two_point_first':
        u_xx_0[0] = (u0[2] - 2*u0[1] + u0[0]) / dx**2
    elif bc_type == 'three_point_second':
        u_xx_0[0] = (2*u0[0] - 5*u0[1] + 4*u0[2] - u0[3]) / dx**2
    else:
        u_xx_0[0] = (u0[2] - 2*u0[1] + u0[0]) / dx**2
    
    # Правая граница
    if bc_type == 'two_point_first':
        u_xx_0[-1] = (u0[-1] - 2*u0[-2] + u0[-3]) / dx**2
    elif bc_type == 'three_point_second':
        u_xx_0[-1] = (2*u0[-1] - 5*u0[-2] + 4*u0[-3] - u0[-4]) / dx**2
    else:
        u_xx_0[-1] = (u0[-1] - 2*u0[-2] + u0[-3]) / dx**2
    
    # u^1 = u^0 + dt * v^0 + (dt^2/2) * a^2 * u_xx^0
    u1 = u0 + dt * v0 + 0.5 * (a * dt)**2 * u_xx_0
    
    # Применяем граничные условия
    u1 = apply_boundary_conditions(u1, dx, dt, bc_type)
    
    return u1

def explicit_cross_scheme(Nx, Nt, T, initial_order=2, bc_type='two_point_first'):
    dx = L / Nx
    dt = T / Nt
    s = (a * dt / dx)**2
    
    if s > 1.0:
        print(f"Warning: explicit cross scheme may be unstable if (a*dt/dx)^2 > 1; current s = {s:.4f}")

    x = np.linspace(0, L, Nx+1)
    t_grid = np.linspace(0, T, Nt+1)

    # начальный слой
    u0 = initial_u(x)
    v0 = initial_ut(x)
    
    # Применяем граничные условия к начальному слою
    u0 = apply_boundary_conditions(u0, dx, 0.0, bc_type)

    # массив для хранения
    U = np.zeros((Nt+1, Nx+1))
    U[0, :] = u0.copy()

    # Аппроксимация u^1 (первый или второй порядок)
    if initial_order == 1:
        print(f"  Используется аппроксимация 1-го порядка для второго начального условия")
        u1 = compute_u1_first_order(u0, v0, dx, dt, bc_type)
    else:  # initial_order == 2
        print(f"  Используется аппроксимация 2-го порядка для второго начального условия")
        u1 = compute_u1_second_order(u0, v0, dx, dt, bc_type)

    U[1, :] = u1.copy()

    # основной цикл
    u_nm1 = u0.copy()
    u_n = u1.copy()
    for n in range(1, Nt):
        t_np1 = t_grid[n+1]
        u_np1 = u_n.copy()
        
        # Схема крест для внутренних узлов:
        # (u^{n+1} - 2u^n + u^{n-1})/dt^2 = a^2*(u^n_{i+1} - 2u^n_i + u^n_{i-1})/dx^2
        u_np1[1:-1] = (2*u_n[1:-1] - u_nm1[1:-1] 
                       + s * (u_n[2:] - 2*u_n[1:-1] + u_n[:-2]))
        
        # граничные условия
        u_np1 = apply_boundary_conditions(u_np1, dx, t_np1, bc_type)

        U[n+1, :] = u_np1.copy()
        u_nm1, u_n = u_n, u_np1

    return x, t_grid, U

def implicit_scheme(Nx, Nt, T, initial_order=2, bc_type='two_point_first'):
    dx = L / Nx
    dt = T / Nt
    s = (a * dt / dx)**2

    x = np.linspace(0, L, Nx+1)
    t_grid = np.linspace(0, T, Nt+1)

    # начальные слои
    u0 = initial_u(x)
    v0 = initial_ut(x)
    
    # Применяем граничные условия к начальному слою
    u0 = apply_boundary_conditions(u0, dx, 0.0, bc_type)

    U = np.zeros((Nt+1, Nx+1))
    U[0, :] = u0.copy()

    # Аппроксимация u^1 (первый или второй порядок)
    if initial_order == 1:
        print(f"  Используется аппроксимация 1-го порядка для второго начального условия")
        u1 = compute_u1_first_order(u0, v0, dx, dt, bc_type)
    else:  # initial_order == 2
        print(f"  Используется аппроксимация 2-го порядка для второго начального условия")
        u1 = compute_u1_second_order(u0, v0, dx, dt, bc_type)

    U[1, :] = u1.copy()

    # Для неявной схемы с граничными условиями третьего рода
    n_all = Nx + 1
    
    u_nm1 = u0.copy()
    u_n = u1.copy()
    
    for n in range(1, Nt):
        t_np1 = t_grid[n+1]
        
        # Формируем трехдиагональную систему для всех узлов
        a_coef = np.zeros(n_all - 1)
        b_diag = np.zeros(n_all)
        c_coef = np.zeros(n_all - 1)
        d_vec = np.zeros(n_all)
        
        # Левая граница (i=0)
        if bc_type == 'two_point_first':
            # (u[1] - u[0])/dx - u[0] = 0
            # => -u[0]*(1 + dx) + u[1] = 0
            b_diag[0] = -(1 + dx)
            c_coef[0] = 1
            d_vec[0] = 0
        elif bc_type == 'three_point_second':
            # (-3*u[0] + 4*u[1] - u[2])/(2*dx) - u[0] = 0
            # => -(3 + 2*dx)*u[0] + 4*u[1] - u[2] = 0
            b_diag[0] = -(3 + 2*dx)
            c_coef[0] = 4
            # Для u[2] нужно модифицировать систему
            d_vec[0] = 0
        else:  # two_point_second
            b_diag[0] = -(1 + dx)
            c_coef[0] = 1
            d_vec[0] = 0
        
        # Внутренние узлы (i=1..Nx-1)
        for i in range(1, Nx):
            a_coef[i-1] = -s
            b_diag[i] = 1 + 2*s
            if i < Nx:
                c_coef[i] = -s
            d_vec[i] = 2*u_n[i] - u_nm1[i]
        
        # Коррекция для трехточечной аппроксимации на левой границе
        if bc_type == 'three_point_second' and Nx >= 2:
            # Учитываем вклад u[2] в уравнение для u[0]
            d_vec[1] += s * u_n[0]  # компенсируем влияние
        
        # Правая граница (i=Nx)
        if bc_type == 'two_point_first':
            a_coef[Nx-1] = 1
            b_diag[Nx] = -(1 + dx)
            d_vec[Nx] = 0
        elif bc_type == 'three_point_second':
            a_coef[Nx-1] = 4
            b_diag[Nx] = -(3 + 2*dx)
            d_vec[Nx] = 0
        else:  # two_point_second
            a_coef[Nx-1] = 1
            b_diag[Nx] = -(1 + dx)
            d_vec[Nx] = 0
        
        # Решаем систему
        u_np1 = progonka(a_coef, b_diag, c_coef, d_vec)
        
        U[n+1, :] = u_np1.copy()
        u_nm1, u_n = u_n, u_np1

    return x, t_grid, U

if __name__ == "__main__":
    # параметры сетки
    Nx = 40
    Nt = 80

    dx = L / Nx
    dt = T / Nt
    s = (a * dt / dx)**2

    current_dir = Path(".")

    # очищаем старые изображения
    for png_file in current_dir.glob("*.png"):
        png_file.unlink()

    print(f"Grid: Nx={Nx}, Nt={Nt}, dx={dx:.4f}, dt={dt:.4f}, s={(a*dt/dx)**2:.4f}")
    
    # Типы граничных условий для тестирования
    bc_types = ['two_point_first', 'three_point_second', 'two_point_second']
    bc_names = ['2-точечная 1-го порядка', '3-точечная 2-го порядка', '2-точечная 2-го порядка']
    
    # Порядки аппроксимации начального условия
    initial_orders = [1, 2]
    initial_names = ['1-го порядка', '2-го порядка']
    
    # Тестируем все комбинации
    for bc_type, bc_name in zip(bc_types, bc_names):
        for init_order, init_name in zip(initial_orders, initial_names):
            print(f"\n{'='*70}")
            print(f"Граничные условия: {bc_name}")
            print(f"Аппроксимация второго начального условия: {init_name}")
            print(f"{'='*70}")
            
            # Запуск схем
            print("Запуск явной схемы...")
            x_e, t_e, U_e = explicit_cross_scheme(Nx, Nt, T, initial_order=init_order, bc_type=bc_type)
            
            print("Запуск неявной схемы...")
            x_i, t_i, U_i = implicit_scheme(Nx, Nt, T, initial_order=init_order, bc_type=bc_type)

            # Вычисление ошибок на нескольких временных слоях
            test_times = [0, Nt//4, Nt//2, 3*Nt//4, Nt]
            
            for idx in test_times:
                t_curr = t_e[idx]
                
                layer_e = U_e[idx]
                layer_i = U_i[idx]
                
                u_exact = exact_solution(x_e, t_curr)
                
                err_e_max, err_e_L2 = compute_errors(layer_e, x_e, t_curr)
                err_i_max, err_i_L2 = compute_errors(layer_i, x_i, t_curr)
                
                print(f"\nt = {t_curr:.3f}:")
                print(f"  Явная схема:   max_err={err_e_max:.3e}, L2_err={err_e_L2:.3e}")
                print(f"  Неявная схема: max_err={err_i_max:.3e}, L2_err={err_i_L2:.3e}")
                
                # График
                plt.figure(figsize=(10, 6))
                plt.plot(x_e, u_exact, 'k-', linewidth=2, label='Точное решение')
                plt.plot(x_e, layer_e, 'b--', linewidth=1.5, label='Явная схема')
                plt.plot(x_i, layer_i, 'r:', linewidth=1.5, label='Неявная схема')
                plt.legend(fontsize=12)
                plt.xlabel('x', fontsize=12)
                plt.ylabel('u(x, t)', fontsize=12)
                plt.title(f'{bc_name}, аппр. нач. усл. {init_name}\nt = {t_curr:.3f}, s = {s:.4f}', fontsize=11)
                plt.grid(True, alpha=0.3)
                
                safe_bc = bc_type.replace('_', '-')
                safe_t = f"{t_curr:.3f}".replace('.', 'p')
                plt.savefig(f"./images/comparison_{safe_bc}_init{init_order}_step_{idx:03d}_t_{safe_t}.png", dpi=100)
                plt.close()

    # Исследование зависимости погрешности от параметров сетки
    print(f"\n{'='*70}")
    print("Исследование зависимости погрешности от параметров сетки")
    print(f"{'='*70}")
    
    N_values = [10, 20, 40, 80]
    bc_type = 'two_point_first'
    
    # Сравним оба порядка аппроксимации начальных условий
    for init_order, init_name in zip(initial_orders, initial_names):
        print(f"\n--- Аппроксимация начального условия: {init_name} ---")
        
        errors_explicit = []
        errors_implicit = []
        
        for N in N_values:
            Nt_test = 4 * N
            x_test, t_test, U_test_e = explicit_cross_scheme(N, Nt_test, T, initial_order=init_order, bc_type=bc_type)
            x_test, t_test, U_test_i = implicit_scheme(N, Nt_test, T, initial_order=init_order, bc_type=bc_type)
            
            err_e_max, err_e_L2 = compute_errors(U_test_e[-1], x_test, T)
            err_i_max, err_i_L2 = compute_errors(U_test_i[-1], x_test, T)
            
            errors_explicit.append((N, L/N, err_e_max, err_e_L2))
            errors_implicit.append((N, L/N, err_i_max, err_i_L2))
            
            print(f"N={N:3d}, dx={L/N:.4f}: Явная max={err_e_max:.3e}, L2={err_e_L2:.3e} | "
                  f"Неявная max={err_i_max:.3e}, L2={err_i_L2:.3e}")
        
        # График зависимости погрешности от dx для данного порядка
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        dx_vals = [e[1] for e in errors_explicit]
        err_e_max_vals = [e[2] for e in errors_explicit]
        err_i_max_vals = [e[2] for e in errors_implicit]
        plt.loglog(dx_vals, err_e_max_vals, 'bo-', label='Явная схема', markersize=8)
        plt.loglog(dx_vals, err_i_max_vals, 'rs-', label='Неявная схема', markersize=8)
        plt.xlabel('dx', fontsize=12)
        plt.ylabel('Max ошибка', fontsize=12)
        plt.title(f'Зависимость max-ошибки от dx\n(аппр. нач. усл. {init_name})', fontsize=11)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3, which='both')
        
        plt.subplot(1, 2, 2)
        err_e_L2_vals = [e[3] for e in errors_explicit]
        err_i_L2_vals = [e[3] for e in errors_implicit]
        plt.loglog(dx_vals, err_e_L2_vals, 'bo-', label='Явная схема', markersize=8)
        plt.loglog(dx_vals, err_i_L2_vals, 'rs-', label='Неявная схема', markersize=8)
        plt.xlabel('dx', fontsize=12)
        plt.ylabel('L2 ошибка', fontsize=12)
        plt.title(f'Зависимость L2-ошибки от dx\n(аппр. нач. усл. {init_name})', fontsize=11)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        plt.savefig(f'./images/error_analysis_init_order_{init_order}.png', dpi=100)
        plt.close()
    
    print("\n" + "="*70)
    print("Все графики сохранены!")
    print("="*70)
