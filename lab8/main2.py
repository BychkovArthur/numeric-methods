import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import pi, cos, exp
import json
import os

# --- Параметры задачи (физика) ---
a_coeff = 1.0  # Коэффициент температуропроводности

# Файл конфигурации по умолчанию
DEFAULT_CONFIG_FILE = "config.json"

# ------------------------------------------------------------------------------
# 1. Работа с JSON конфигурацией
# ------------------------------------------------------------------------------
def load_configuration():
    # Настройки по умолчанию, если файла нет
    default_config = {
        "Nx": 40,           # Разбиения по X
        "Ny": 40,           # Разбиения по Y
        "Nt_steps": 200,    # Количество шагов по времени (разбиение по T)
        "T_end": 0.1,       # Конечное время
        "plot_times": [0.02, 0.05, 0.1] # Моменты времени для графиков
    }
    
    if not os.path.exists(DEFAULT_CONFIG_FILE):
        print(f"Файл {DEFAULT_CONFIG_FILE} не найден. Создаю файл по умолчанию.")
        with open(DEFAULT_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=4)
        return default_config
    else:
        print(f"Читаю настройки из {DEFAULT_CONFIG_FILE}...")
        with open(DEFAULT_CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)

# ------------------------------------------------------------------------------
# 2. Математические функции условий задачи
# ------------------------------------------------------------------------------
def initial_condition(x, y, mu1, mu2):
    return np.cos(mu1 * x) * np.cos(mu2 * y)

def boundary_x0(y, t, mu1, mu2, a):
    return np.cos(mu2 * y) * np.exp(-(mu1**2 + mu2**2) * a * t)

def boundary_y0(x, t, mu1, mu2, a):
    return np.cos(mu1 * x) * np.exp(-(mu1**2 + mu2**2) * a * t)

def exact_solution(x, y, t, mu1, mu2, a):
    return np.cos(mu1 * x) * np.cos(mu2 * y) * np.exp(-(mu1**2 + mu2**2) * a * t)

# ------------------------------------------------------------------------------
# 3. Метод прогонки
# ------------------------------------------------------------------------------
def progonka(a, b, c, d):
    n = len(d)
    c_prime = np.zeros(n-1)
    d_prime = np.zeros(n)
    
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]
    
    for i in range(1, n-1):
        temp = b[i] - a[i-1] * c_prime[i-1]
        c_prime[i] = c[i] / temp
        d_prime[i] = (d[i] - a[i-1] * d_prime[i-1]) / temp
        
    d_prime[n-1] = (d[n-1] - a[n-2] * d_prime[n-2]) / (b[n-1] - a[n-2] * c_prime[n-2])
    
    x = np.zeros(n)
    x[n-1] = d_prime[n-1]
    for i in range(n-2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]
        
    return x

# ------------------------------------------------------------------------------
# 4. Схема переменных направлений (МПН)
# ------------------------------------------------------------------------------
def solve_adi_history(config, mu1, mu2):
    # --- БЛОК 1: Распаковка параметров из конфига ---
    Nx = config["Nx"]
    Ny = config["Ny"]
    Nt = config["Nt_steps"]
    T_end = config["T_end"]
    plot_times = np.array(config["plot_times"])
    
    # Расчет шагов
    Lx = pi / (2 * mu1)
    Ly = pi / (2 * mu2)
    hx = Lx / Nx
    hy = Ly / Ny
    tau = T_end / Nt  # Шаг по времени из количества разбиений
    
    # Сетка
    x = np.linspace(0, Lx, Nx+1)
    y = np.linspace(0, Ly, Ny+1)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # НУ
    u = initial_condition(X, Y, mu1, mu2)
    t = 0
    
    # Словарь для сохранения результатов: {время: матрица_решения}
    history = {} 
    
    # Числа Куранта (для полушага tau/2)
    rx = a_coeff * tau / (2 * hx**2)
    ry = a_coeff * tau / (2 * hy**2)
    
    # --- БЛОК 2: Цикл по времени ---
    for n_step in range(Nt):
        u_half = np.zeros_like(u)
        t_half = t + tau / 2
        t_next = t + tau
        
        # --- ЭТАП 1: Неявный по X (строки) ---
        # ГУ слева/справа на t_half
        for j in range(Ny + 1):
            u_half[0, j] = boundary_x0(y[j], t_half, mu1, mu2, a_coeff)
            u_half[Nx, j] = 0.0
            
        adj_a = -rx * np.ones(Nx-1)
        adj_b = (1 + 2*rx) * np.ones(Nx-1)
        adj_c = -rx * np.ones(Nx-1)
        
        for j in range(1, Ny):
            Lambda_yy = (u[1:Nx, j-1] - 2*u[1:Nx, j] + u[1:Nx, j+1])
            RHS = u[1:Nx, j] + ry * Lambda_yy
            RHS[0] += rx * u_half[0, j]
            RHS[-1] += rx * u_half[Nx, j]
            u_half[1:Nx, j] = progonka(adj_a[1:], adj_b, adj_c[:-1], RHS)

        # Границы по Y для корректности массива
        for i in range(Nx+1):
            u_half[i, 0] = boundary_y0(x[i], t_half, mu1, mu2, a_coeff)
            u_half[i, Ny] = 0.0

        # --- ЭТАП 2: Неявный по Y (столбцы) ---
        u_next = np.zeros_like(u)
        
        # ГУ по X (лево/право) на t_next
        for j in range(Ny + 1):
            u_next[0, j] = boundary_x0(y[j], t_next, mu1, mu2, a_coeff)
            u_next[Nx, j] = 0.0

        # ГУ по Y (низ/верх) на t_next
        for i in range(Nx + 1):
            u_next[i, 0] = boundary_y0(x[i], t_next, mu1, mu2, a_coeff)
            u_next[i, Ny] = 0.0
            
        adj_a = -ry * np.ones(Ny-1)
        adj_b = (1 + 2*ry) * np.ones(Ny-1)
        adj_c = -ry * np.ones(Ny-1)
        
        for i in range(1, Nx):
            Lambda_xx = (u_half[i-1, 1:Ny] - 2*u_half[i, 1:Ny] + u_half[i+1, 1:Ny])
            RHS = u_half[i, 1:Ny] + rx * Lambda_xx
            RHS[0] += ry * u_next[i, 0]
            RHS[-1] += ry * u_next[i, Ny]
            u_next[i, 1:Ny] = progonka(adj_a[1:], adj_b, adj_c[:-1], RHS)
            
        u = u_next
        t = t_next
        
        # --- БЛОК 3: Сохранение результата, если подошло время графика ---
        # Проверяем, близко ли текущее время к одному из целевых
        # (используем погрешность в полшага)
        for target_t in plot_times:
            if abs(t - target_t) < tau / 1.99:
                history[target_t] = np.copy(u)
    
    return X, Y, history

# ------------------------------------------------------------------------------
# 5. Схема дробных шагов (МДШ)
# ------------------------------------------------------------------------------
def solve_splitting_history(config, mu1, mu2):
    Nx = config["Nx"]
    Ny = config["Ny"]
    Nt = config["Nt_steps"]
    T_end = config["T_end"]
    plot_times = np.array(config["plot_times"])
    
    Lx = pi / (2 * mu1)
    Ly = pi / (2 * mu2)
    hx = Lx / Nx
    hy = Ly / Ny
    tau = T_end / Nt
    
    x = np.linspace(0, Lx, Nx+1)
    y = np.linspace(0, Ly, Ny+1)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    u = initial_condition(X, Y, mu1, mu2)
    t = 0
    history = {}
    
    rx = a_coeff * tau / (hx**2)
    ry = a_coeff * tau / (hy**2)
    
    for n_step in range(Nt):
        t_next = t + tau
        u_star = np.copy(u)
        
        # --- ШАГ 1: По X ---
        for j in range(Ny + 1):
            u_star[0, j] = boundary_x0(y[j], t_next, mu1, mu2, a_coeff)
            u_star[Nx, j] = 0.0
        
        adj_a = -rx * np.ones(Nx-1)
        adj_b = (1 + 2*rx) * np.ones(Nx-1)
        adj_c = -rx * np.ones(Nx-1)
        
        for j in range(1, Ny): # 0 и Ny фиксированы ГУ по Y (см. ниже)
            RHS = u[1:Nx, j]
            RHS[0] += rx * u_star[0, j]
            RHS[-1] += rx * u_star[Nx, j]
            u_star[1:Nx, j] = progonka(adj_a[1:], adj_b, adj_c[:-1], RHS)
            
        # Заливка границ Y для u_star (важно для шага 2)
        # На краях по Y u_star = u_result = граничному условию
        for i in range(Nx+1):
           u_star[i, 0] = boundary_y0(x[i], t_next, mu1, mu2, a_coeff)
           u_star[i, Ny] = 0.0
            
        # --- ШАГ 2: По Y ---
        u_next = np.copy(u_star)
        
        for i in range(Nx + 1):
            u_next[i, 0] = boundary_y0(x[i], t_next, mu1, mu2, a_coeff)
            u_next[i, Ny] = 0.0
            
        adj_a = -ry * np.ones(Ny-1)
        adj_b = (1 + 2*ry) * np.ones(Ny-1)
        adj_c = -ry * np.ones(Ny-1)
        
        for i in range(1, Nx):
            RHS = u_star[i, 1:Ny]
            RHS[0] += ry * u_next[i, 0]
            RHS[-1] += ry * u_next[i, Ny]
            u_next[i, 1:Ny] = progonka(adj_a[1:], adj_b, adj_c[:-1], RHS)
            
        u = u_next
        t = t_next
        
        # --- Сохранение ---
        for target_t in plot_times:
            if abs(t - target_t) < tau / 1.99:
                history[target_t] = np.copy(u)
        
    return X, Y, history

# ------------------------------------------------------------------------------
# 6. Основной цикл анализа и отображения
# ------------------------------------------------------------------------------
def run_analysis():
    # Читаем конфигурацию
    config = load_configuration()
    
    # 3 варианта параметров из задания (физику JSON не меняет, только сетку)
    cases = [
        {'mu1': 1, 'mu2': 1, 'label': 'Вариант 1 (μ1=1, μ2=1)'},
        {'mu1': 2, 'mu2': 1, 'label': 'Вариант 2 (μ1=2, μ2=1)'},
        {'mu1': 1, 'mu2': 2, 'label': 'Вариант 3 (μ1=1, μ2=2)'}
    ]
    
    print(f"Параметры сетки: Nx={config['Nx']}, Ny={config['Ny']}, steps={config['Nt_steps']}")
    print(f"Графики будут построены для t = {config['plot_times']}")
    print("-" * 60)

    for case in cases:
        mu1 = case['mu1']
        mu2 = case['mu2']
        
        # Запускаем расчет сразу на все времена
        X, Y, hist_adi = solve_adi_history(config, mu1, mu2)
        _, _, hist_split = solve_splitting_history(config, mu1, mu2)
        
        # Получаем список сохраненных времен (отсортированный)
        times_found = sorted(hist_adi.keys())
        
        if not times_found:
            print(f"ВНИМАНИЕ: Для {case['label']} не найдено совпадений по времени!")
            print("Возможно, шаг по времени слишком велик или плохие plot_times.")
            continue

        # Строим графики для КАЖДОГО найденного момента времени
        for t_val in times_found:
            u_adi = hist_adi[t_val]
            u_split = hist_split[t_val]
            u_ex = exact_solution(X, Y, t_val, mu1, mu2, a_coeff)
            
            err_adi = np.max(np.abs(u_adi - u_ex))
            err_split = np.max(np.abs(u_split - u_ex))
            
            print(f"{case['label']} | T={t_val:.3f} | Err МПН: {err_adi:.2e} | Err Split: {err_split:.2e}")

            # РИСОВАНИЕ
            fig = plt.figure(figsize=(15, 8))
            fig.suptitle(f"{case['label']} при t = {t_val:.3f}", fontsize=14)
            
            # --- МПН (Верхний ряд) ---
            ax1 = fig.add_subplot(2, 3, 1, projection='3d')
            ax1.plot_surface(X, Y, u_ex, cmap='viridis')
            ax1.set_title(f'Аналитика (Max={np.max(u_ex):.2f})')
            
            ax2 = fig.add_subplot(2, 3, 2, projection='3d')
            ax2.plot_surface(X, Y, u_adi, cmap='inferno')
            ax2.set_title('МПН Решение')
            
            ax3 = fig.add_subplot(2, 3, 3, projection='3d')
            surf3 = ax3.plot_surface(X, Y, np.abs(u_adi - u_ex), cmap='Reds')
            ax3.set_title(f'МПН Ошибка (Max {err_adi:.1e})')
            fig.colorbar(surf3, ax=ax3, fraction=0.046, pad=0.04)
            
            # --- МДШ (Нижний ряд) ---
            ax4 = fig.add_subplot(2, 3, 4, projection='3d')
            ax4.plot_surface(X, Y, u_ex, cmap='viridis')
            ax4.set_title('Аналитика')
            ax4.set_xlabel('X')
            ax4.set_ylabel('Y')
            
            ax5 = fig.add_subplot(2, 3, 5, projection='3d')
            ax5.plot_surface(X, Y, u_split, cmap='inferno')
            ax5.set_title('МДШ Решение')
            ax5.set_xlabel('X')
            ax5.set_ylabel('Y')
            
            ax6 = fig.add_subplot(2, 3, 6, projection='3d')
            surf6 = ax6.plot_surface(X, Y, np.abs(u_split - u_ex), cmap='Reds')
            ax6.set_title(f'МДШ Ошибка (Max {err_split:.1e})')
            ax6.set_xlabel('X')
            ax6.set_ylabel('Y')
            fig.colorbar(surf6, ax=ax6, fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    run_analysis()
