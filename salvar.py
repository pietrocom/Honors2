import numpy as np
import matplotlib.pyplot as plt
import time
import trajetoria_dinamica as td
import busca_linear as bl
import diferencas_finitas as df


def gd(f, x0, grad=None, eps=1e-5, alpha=0.1, itmax=10000, fd=False, h=1e-7, plot=False, search=False, tau=0.5):
    x = x0.copy()
    k = 0
    trajectory = [x.copy()]

    while k < itmax:
        if fd:
            g = df.fin_diff(f, x, 1, h)
        elif grad is not None:
            g = grad(x)
        else:
            raise ValueError("O gradiente analítico não foi fornecido e `fd` está desativado.")

        if np.linalg.norm(g) < eps:
            break

        d = -g

        if search:
            alpha = bl.linesearch(f, x, d, g, tau=tau, gama=gama)
        else:
            pass

        x = x + alpha * d
        trajectory.append(x.copy())
        k += 1

    if plot and len(x0) == 2:
        td.td(f, trajectory)

    return x, k






# Função de exemplo e gradiente fornecidos no enunciado
def f(x):
    return x[0]**4 - 2*x[0]**2 + x[0] - x[0]*x[1] + x[1]**2
def grad(x):
    return np.array([4*x[0]**3 - 4*x[0] + 1 - x[1], -x[0] + 2*x[1]])


# Exemplo 1: Passo fixo
x, k = gd(f, np.array([0, 0]), grad=grad, alpha=0.1, fd=False, plot=True, search=False)
print(f"Passo fixo: x = {x}, iterações = {k}")

# Exemplo 2: Diferenças finitas
x, k = gd(f,np.array([3,3]), grad=grad, alpha=1e-2, eps=1e-8, fd=True, plot=True, search=False)
print(f"Diferenças finitas: x = {x}, iterações = {k}")

# Exemplo 3: Busca linear
x, k = gd(f, np.array([0, 0]), grad=grad, fd=True, search=True, plot=True)
print(f"Busca linear: x = {x}, iterações = {k}")























# ---------- Plotagem e aplicacao ----------

# Exemplo 1: Função Quadrática
# Passo fixo
x1, k1, x_vals1 = gd(f1, np.array([2, 3]), grad=grad1, alpha=0.1, fd=False, search=False)
print(f"Passo fixo: x = {x1}, iterações = {k1}")
plot_curvas_nivel(x_vals1)

# Diferenças finitas
x2, k2, x_vals2 = gd(f1, np.array([2, 3]), grad=grad1, alpha=0.01, eps=1e-8, fd=True, search=False)
print(f"Diferenças finitas: x = {x2}, iterações = {k2}")
plot_curvas_nivel(x_vals2)

# Busca linear
x3, k3, x_vals3 = gd(f1, np.array([2, 3]), grad=grad1, fd=True, search=True)
print(f"Busca linear: x = {x3}, iterações = {k3}")
plot_curvas_nivel(x_vals3)

# Exemplo 2: Função de Rosenbrock
# Passo fixo
x4, k4, x_vals4 = gd(f2, np.array([0, 0]), grad=grad2, alpha=0.0001, fd=False, search=False)
print(f"Passo fixo: x = {x4}, iterações = {k4}")
plot_curvas_nivel(x_vals4)

# Diferenças finitas
x5, k5, x_vals5 = gd(f2, np.array([0, 0]), grad=grad2, alpha=0.00001, eps=1e-8, fd=True, search=False)
print(f"Diferenças finitas: x = {x5}, iterações = {k5}")
plot_curvas_nivel(x_vals5)

# Busca linear
x6, k6, x_vals6 = gd(f2, np.array([0, 0]), grad=grad2, fd=True, search=True)
print(f"Busca linear: x = {x6}, iterações = {k6}")
plot_curvas_nivel(x_vals6)

# Exemplo 3: Função de Rastrigin
# Passo fixo
x7, k7, x_vals7 = gd(f3, np.array([2, 2]), grad=grad3, alpha=0.01, fd=False, search=False)
print(f"Passo fixo: x = {x7}, iterações = {k7}")
plot_curvas_nivel(x_vals7)

# Diferenças finitas
x8, k8, x_vals8 = gd(f3, np.array([2, 2]), grad=grad3, alpha=0.001, eps=1e-8, fd=True, search=False)
print(f"Diferenças finitas: x = {x8}, iterações = {k8}")
plot_curvas_nivel(x_vals8)

# Busca linear
x9, k9, x_vals9 = gd(f3, np.array([2, 2]), grad=grad3, fd=True, search=True)
print(f"Busca linear: x = {x9}, iterações = {k9}")
plot_curvas_nivel(x_vals9)