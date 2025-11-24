import numpy as np
import matplotlib.pyplot as plt

# Настройка размера графиков
plt.rcParams['figure.figsize'] = [10, 8]

# ==============================================================================
# ФИЗИКА И ТОЧНОЕ РЕШЕНИЕ
# ==============================================================================

# Точное решение U(x,t) = sin(x-at) + cos(x+at)
def U(x, t, a):
    return np.sin(x - a * t) + np.cos(x + a * t)

# Начальное условие u(x, 0)
def psi(x):
    return np.sin(x) + np.cos(x)

# Второе начальное условие u_t(x, 0)
def get_second_layer(mode, x, a, tau):
    # u_t(x,0) = -a(sin x + cos x)
    phi = -a * (np.sin(x) + np.cos(x)) 
    
    if mode == 1:
        # 1-й порядок: u1 = u0 + tau * phi
        return psi(x) + tau * phi
    else:
        # 2-й порядок (Тейлор): u1 = u0 + tau*phi + (tau^2/2)*u_tt
        # Из уравнения u_tt = a^2 * u_xx
        # u_xx(x,0) = -sin(x) - cos(x) = -psi(x)
        u_tt = (a**2) * (-psi(x))
        return psi(x) + tau * phi + (tau**2 / 2) * u_tt

# ==============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==============================================================================

def solve_tridiagonal(a, b):
    """Метод прогонки для решения Ax=b"""
    n = len(b)
    p = np.zeros(n)
    q = np.zeros(n)
    
    # Защита от деления на ноль
    if a[0][0] == 0: a[0][0] = 1e-20

    p[0] = -a[0][1] / a[0][0]
    q[0] = b[0] / a[0][0]
    
    for i in range(1, n - 1):
        denom = a[i][i] + a[i][i - 1] * p[i - 1]
        if denom == 0: denom = 1e-20
        p[i] = -a[i][i + 1] / denom
        q[i] = (b[i] - a[i][i - 1] * q[i - 1]) / denom
        
    denom_last = a[n - 1][n - 1] + a[n - 1][n - 2] * p[n - 2]
    if denom_last == 0: denom_last = 1e-20
    q[n - 1] = (b[n - 1] - a[n - 1][n - 2] * q[n - 2]) / denom_last
    
    x = np.zeros(n)
    x[-1] = q[-1]
    for i in range(n - 2, -1, -1):
        x[i] = p[i] * x[i + 1] + q[i]
    return x

# ==============================================================================
# ЯВНАЯ СХЕМА
# ==============================================================================

def Explicit_Method(N, K, sigma, tau, a, h, bc_type, init_order):
    if sigma > 1.00001:
        print("ВНИМАНИЕ: Явная схема неустойчива!")

    u = np.zeros((K + 1, N + 1))
    x_vals = np.linspace(0, N*h, N+1)
    
    # Начальные условия
    u[0] = psi(x_vals)
    u[1] = get_second_layer(init_order, x_vals, a, tau)

    for k in range(1, K):
        # 1. Сначала считаем внутренние точки
        u[k+1][1:N] = 2 * (1 - sigma) * u[k][1:N] + \
                      sigma * (u[k][0:N-1] + u[k][2:N+1]) - \
                      u[k-1][1:N]
        
        # 2. Аппроксимация граничных условий: u_x - u = 0 => u_x = u
        
        if bc_type == 1: # Двухточечная 1-го порядка
            # (u1 - u0)/h = u0 => u0 = u1 / (1+h)
            u[k+1][0] = u[k+1][1] / (1 + h)
            # (uN - uN-1)/h = uN => uN = uN-1 / (1-h)
            u[k+1][N] = u[k+1][N-1] / (1 - h)

        elif bc_type == 2: # Трехточечная 2-го порядка
            # (-3u0 + 4u1 - u2)/2h = u0 => u0(3+2h) = 4u1 - u2
            u[k+1][0] = (4 * u[k+1][1] - u[k+1][2]) / (3 + 2 * h)
            # (3uN - 4uN-1 + uN-2)/2h = uN => uN(3-2h) = 4uN-1 - uN-2
            u[k+1][N] = (4 * u[k+1][N-1] - u[k+1][N-2]) / (3 - 2 * h)

        elif bc_type == 3: # Двухточечная 2-го порядка (фиктивная точка)
            # Левая: (u1 - u_-1)/2h = u0 => u_-1 = u1 - 2h*u0
            # Подставляем в ур. волны для узла 0:
            # u^{k+1}_0 = 2(1-sigma)u^k_0 + sigma(u_1 + u_-1) - u^{k-1}_0
            # u^{k+1}_0 = 2(1-sigma)u^k_0 + sigma(2u_1 - 2h*u^k_0) - u^{k-1}_0
            u[k+1][0] = 2*(1-sigma)*u[k][0] + sigma*(2*u[k][1] - 2*h*u[k][0]) - u[k-1][0]
            
            # Правая: (u_N+1 - u_N-1)/2h = uN => u_N+1 = u_N-1 + 2h*uN
            # u^{k+1}_N = 2(1-sigma)u^k_N + sigma(u_N-1 + u_N+1) - u^{k-1}_N
            # u^{k+1}_N = 2(1-sigma)u^k_N + sigma(2u_N-1 + 2h*u^k_N) - u^{k-1}_N
            u[k+1][N] = 2*(1-sigma)*u[k][N] + sigma*(2*u[k][N-1] + 2*h*u[k][N]) - u[k-1][N]

    return u

# ==============================================================================
# НЕЯВНАЯ СХЕМА
# ==============================================================================

def Implicit_Method(N, K, sigma, tau, a, h, bc_type, init_order):
    u = np.zeros((K + 1, N + 1))
    x_vals = np.linspace(0, N*h, N+1)
    
    u[0] = psi(x_vals)
    u[1] = get_second_layer(init_order, x_vals, a, tau)

    # Коэффициенты стандартного уравнения: sigma*u_{j-1} - (1+2sigma)*u_j + sigma*u_{j+1} = RHS
    A_coef, B_coef, C_coef = sigma, -(1 + 2 * sigma), sigma

    for k in range(1, K):
        matrix = np.zeros((N + 1, N + 1))
        d = np.zeros(N + 1)

        # Заполняем внутренние узлы
        for j in range(1, N):
            matrix[j][j-1] = A_coef
            matrix[j][j]   = B_coef
            matrix[j][j+1] = C_coef
            d[j] = -2 * u[k][j] + u[k-1][j]

        # --- АППРОКСИМАЦИЯ ГРАНИЦ В МАТРИЦЕ (u_x = u) ---

        if bc_type == 1: # Двухточечная 1-го порядка
            # Левая: u[0](1+h) - u[1] = 0
            matrix[0][0] = 1 + h
            matrix[0][1] = -1
            d[0] = 0
            
            # Правая: -u[N-1] + u[N](1-h) = 0
            matrix[N][N-1] = -1
            matrix[N][N]   = 1 - h
            d[N] = 0

        elif bc_type == 3: # Двухточечная 2-го порядка (Фиктивная точка)
            # Левая: Из фиктивной точки u_-1 = u_1 - 2h*u_0
            # Ур-ние для узла 0: sigma*u_-1 + B*u_0 + sigma*u_1 = d[0]
            # sigma(u_1 - 2h*u_0) + B*u_0 + sigma*u_1 = d[0]
            # u_0(B - 2*h*sigma) + u_1(2*sigma) = d[0]
            matrix[0][0] = B_coef - 2 * h * sigma
            matrix[0][1] = 2 * sigma
            d[0] = -2*u[k][0] + u[k-1][0]
            
            # Правая: Из фиктивной точки u_N+1 = u_N-1 + 2h*u_N
            # sigma*u_N-1 + B*u_N + sigma*u_N+1 = d[N]
            # sigma*u_N-1 + B*u_N + sigma(u_N-1 + 2h*u_N) = d[N]
            # u_N-1(2*sigma) + u_N(B + 2*h*sigma) = d[N]
            matrix[N][N-1] = 2 * sigma
            matrix[N][N]   = B_coef + 2 * h * sigma
            d[N] = -2*u[k][N] + u[k-1][N]

        elif bc_type == 2: # Трехточечная 2-го порядка
            # Левая: (3+2h)u_0 - 4u_1 + u_2 = 0  => u_0 = (4u_1 - u_2)/(3+2h)
            # Подставляем u_0 в стандартное уравнение для 1-го узла (чтобы не нарушать трехдиагональность)
            # sigma*u_0 + B*u_1 + sigma*u_2 = d[1]
            # sigma*( (4u_1 - u_2)/(3+2h) ) + B*u_1 + sigma*u_2 = d[1]
            denom_L = 3 + 2*h
            coeff_u1 = sigma * 4 / denom_L + B_coef
            coeff_u2 = sigma * (-1) / denom_L + C_coef # фактически это новый C
            
            matrix[1][0] = 0 # u_0 исключили
            matrix[1][1] = coeff_u1
            matrix[1][2] = coeff_u2
            
            # Аналогично справа:
            # u_N = (4u_N-1 - u_N-2)/(3-2h)
            # Стандартное для N-1: sigma*u_N-2 + B*u_N-1 + sigma*u_N = d[N-1]
            # sigma*u_N-2 + B*u_N-1 + sigma*((4u_N-1 - u_N-2)/(3-2h)) = d[N-1]
            denom_R = 3 - 2*h
            coeff_uN1 = B_coef + sigma * 4 / denom_R
            coeff_uN2 = sigma + sigma * (-1) / denom_R # коэф при u_N-2
            
            matrix[N-1][N] = 0 # u_N исключили
            matrix[N-1][N-1] = coeff_uN1
            matrix[N-1][N-2] = coeff_uN2
            
            # "Заглушки" для крайних строк, чтобы матрица не была вырожденной
            matrix[0][0], matrix[N][N] = 1, 1
            
            # Решаем
            ans = solve_tridiagonal(matrix, d)
            u[k+1] = ans
             
            # Явно досчитываем u_0 и u_N после прогонки
            u[k+1][0] = (4 * ans[1] - ans[2]) / denom_L
            u[k+1][N] = (4 * ans[N-1] - ans[N-2]) / denom_R
            continue

        # Решение СЛАУ
        u[k+1] = solve_tridiagonal(matrix, d)

    return u

# ==============================================================================
# АНАЛИЗ И ГРАФИКИ
# ==============================================================================

def show_solution(h, tau, K, l, u, U, a, title):
    x_array = np.linspace(0, l, u.shape[1])
    fig, ax = plt.subplots()
    # Выберем несколько моментов времени для отображения
    steps = [int(K*0.3), int(K*0.6), int(K*0.9)]
    colors = ['blue', 'green', 'red']
    
    for i, k in enumerate(steps):
        if k > K: k = K
        t = k * tau
        u_exact = U(x_array, t, a)
        u_calc = u[k]
        ax.plot(x_array, u_exact, color=colors[i], alpha=0.4, linewidth=4, label=f'Точн t={t:.2f}')
        ax.plot(x_array, u_calc, color=colors[i], linestyle='--', linewidth=1.5, label=f'Числ t={t:.2f}')
        
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('U')
    ax.grid(True)
    ax.legend()
    plt.show()

def error(l, a, T, U, bc_type, init_order):
    sigma_target = 0.5
    N_array = [20, 40, 80]
    errors1 = []
    errors2 = []
    
    print("\nРасчет погрешностей (L2 norm)...")
    for cur_N in N_array:
        cur_h = l / cur_N
        cur_tau = cur_h * np.sqrt(sigma_target) / a
        cur_K = int(round(T / cur_tau))
        cur_real_sigma = (a * cur_tau / cur_h) ** 2
        
        u1 = Explicit_Method(cur_N, cur_K, cur_real_sigma, cur_tau, a, cur_h, bc_type, init_order)
        u2 = Implicit_Method(cur_N, cur_K, cur_real_sigma, cur_tau, a, cur_h, bc_type, init_order)
        
        target_k = int(cur_K / 2)
        t_val = target_k * cur_tau
        x_arr = np.linspace(0, l, cur_N + 1)
        u_exact = U(x_arr, t_val, a)
        
        errors1.append(np.max(np.abs(u_exact - u1[target_k])))
        errors2.append(np.max(np.abs(u_exact - u2[target_k])))
        
    return N_array, errors1, errors2

def show_errors(l, a, T, U, bc_type, init_order):
    N_array, errors1, errors2 = error(l, a, T, U, bc_type, init_order)
    h_vals = [l/n for n in N_array]
    
    fig, ax = plt.subplots()
    ax.loglog(h_vals, errors1, 'o-', label='Явный метод')
    ax.loglog(h_vals, errors2, 's-', label='Неявный метод')
    ax.set_xlabel('Шаг h (log)')
    ax.set_ylabel('Макс ошибка (log)')
    ax.set_title(f'Сходимость (ГУ тип {bc_type}, НУ тип {init_order})')
    ax.grid(True, which="both", ls="-")
    ax.legend()
    plt.show()

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    a = 1.0
    l = np.pi
    T = 2.0 

    print("="*60)
    print("ВОЛНОВОЕ УРАВНЕНИЕ: u_x - u = 0")
    print("="*60)

    try:
        N = int(input("Число разбиений по X (например 50): ") or 50)
        K = int(input("Число разбиений по T (например 200): ") or 200)
    except ValueError:
        N, K = 50, 200

    h = l / N
    tau = T / K
    sigma = (a * tau / h) ** 2
    
    print(f"\nПараметры: h={h:.4f}, tau={tau:.4f}, Sigma={sigma:.4f}")
    if sigma > 1: print("!!! Sigma > 1. Явная схема будет неустойчивой !!!")

    print("-" * 40)
    print("Выберите аппроксимацию ГУ (u_x - u = 0):")
    print("1 - Двухточечная 1-го порядка")
    print("2 - Трехточечная 2-го порядка")
    print("3 - Двухточечная 2-го порядка (фиктивная точка)")
    try: bc_choice = int(input("> ") or 1)
    except: bc_choice = 1

    print("-" * 40)
    print("Порядок начального условия u_t:")
    print("1 - Первый порядок")
    print("2 - Второй порядок")
    try: init_choice = int(input("> ") or 2)
    except: init_choice = 2

    print("\nЗапуск расчета...")
    
    u1 = Explicit_Method(N, K, sigma, tau, a, h, bc_choice, init_choice)
    show_solution(h, tau, K, l, u1, U, a, f"Явная схема (ГУ тип {bc_choice})")

    u2 = Implicit_Method(N, K, sigma, tau, a, h, bc_choice, init_choice)
    show_solution(h, tau, K, l, u2, U, a, f"Неявная схема (ГУ тип {bc_choice})")

    show_errors(l, a, T, U, bc_choice, init_choice)

if __name__ == "__main__":
    main()
