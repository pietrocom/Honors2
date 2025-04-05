# Nome: Pietro Comin
# GRR:  20241955
# Todos os achievements serao realizados


import numpy as np
import matplotlib.pyplot as plt
import time


def fin_diff(f, x, degree, h=1e-7):
    n = len(x)
    if degree == 1:
        grad = np.zeros(n)
        for i in range(n):
            x1, x2 = np.copy(x), np.copy(x)
            x1[i] += h
            x2[i] -= h
            grad[i] = (f(x1) - f(x2)) / (2 * h)
        return grad
    elif degree == 2:
        hess = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                x1, x2, x3, x4 = [np.copy(x) for _ in range(4)]
                x1[i] += h
                x1[j] += h
                x2[i] += h
                x2[j] -= h
                x3[i] -= h
                x3[j] += h
                x4[i] -= h
                x4[j] -= h
                hess[i, j] = (f(x1) - f(x2) - f(x3) + f(x4)) / (4 * h**2)
        return hess


def calcula_grad(grad, fd, f, x, h=1e-7):
    return fin_diff(f, x, degree=1, h=h) if fd else grad(x)


def bl(f, x, d, grad_x, tau=0.5, alpha_init=1.0, gama=0.5):
    alpha = alpha_init
    while f(x + alpha * d) > f(x) + alpha * tau * np.dot(grad_x, d):
        alpha *= gama
    return alpha


def gera_z(X, Y, f):
    Z = np.zeros_like(X)
    for i in range(len(X)):
        for j in range(len(X[i])):
            Z[i, j] = f([X[i, j], Y[i, j]])
    return Z


def plot_curvas_nivel(f, x_vals):
    x_vals = np.array(x_vals)
    x_min, x_max = x_vals[:, 0].min() - 2, x_vals[:, 0].max() + 2
    y_min, y_max = x_vals[:, 1].min() - 2, x_vals[:, 1].max() + 2

    x_range = np.linspace(x_min, x_max, 400)
    y_range = np.linspace(y_min, y_max, 400)
    X, Y = np.meshgrid(x_range, y_range)

    Z = gera_z(X, Y, f)

    plt.figure(figsize=(12, 8))
    plt.contour(X, Y, Z, levels=50, cmap='jet')
    plt.plot(x_vals[:, 0], x_vals[:, 1], '-o', color='red', markersize=4, label='Caminho')
    plt.scatter(x_vals[0, 0], x_vals[0, 1], color='green', label='Início')
    plt.scatter(x_vals[-1, 0], x_vals[-1, 1], color='blue', label='Fim')
    plt.title('Curvas de Nível e Trajetória do Método do Gradiente', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.legend()
    plt.show()


def gd(f, x0, grad, eps=1e-5, alpha=0.1, itmax=10000, fd=False, h=1e-7, plot=False, search=False):
    x = np.array(x0, dtype=np.float64)
    k = 0
    trajectory = [x.copy()]
    
    while k < itmax:
        grad_x = calcula_grad(grad, fd, f, x, h=h)

        # Critério de parada
        if np.linalg.norm(grad_x) <= eps:
            break

        d = -grad_x

        # Normalização do gradiente para evitar explosões
        if not search:
            d /= np.linalg.norm(d)

        # Atualização
        x += alpha * d
        trajectory.append(x.copy())

        # Diagnóstico para detectar divergência
        if np.any(np.abs(x) > 1e12):
            print(f"Divergência detectada na iteração {k}: x = {x}")
            break

        k += 1

    if plot:
        plot_curvas_nivel(f, trajectory)

    return x, k


def newton(f, x0, grad, hess, eps=1e-5, alpha=0.1, itmax=10000, fd=False, h=1e-7, plot=False):
    x = x0.astype(np.float64)  
    k = 0
    x_vals = [x.copy()]

    gradient = grad(x) if not fd else fin_diff(f, x, 1, h=h)

    while np.linalg.norm(gradient) > eps and k < itmax:
        k += 1

        if fd:
            hessian = fin_diff(f, x, 2, h=h)
            gradient = fin_diff(f, x, 1, h=h)
        else:
            hessian = hess(x)
            gradient = grad(x)

        try:
            d = np.linalg.solve(hessian, -gradient)
        except np.linalg.LinAlgError as e:
            print(f"Erro ao resolver o sistema linear na iteração {k}: {e}")
            return x, k

        x += alpha * d
        x_vals.append(x.copy())

        gradient = grad(x) if not fd else fin_diff(f, x, 1, h=h)

    if plot:
        plot_curvas_nivel(f, np.array(x_vals))

    return x, k



def is_singular(matrix):
    det = np.linalg.det(matrix)
    return np.isclose(det, 0)



def newton_salvaguardas(f, x0, grad, hess, eps=1e-5, alpha=0.1, itmax=10000, fd=False, h=1e-7, plot=False, search=False, tau=0.5, alpha_init=1.0, gama=0.1):
    x = x0.astype(np.float64)  
    k = 0
    gradient = grad(x) if not fd else fin_diff(f, x, 1, h=h)
    x_vals = [x.copy()]

    while np.linalg.norm(gradient) > eps and k < itmax:
        k += 1
        
        if fd:
            hessian = fin_diff(f, x, 2, h=h)
        else:
            hessian = hess(x)
        
        # Regularização da Hessiana
        if is_singular(hessian):
            hessian = 0.9 * hessian + 0.1 * np.eye(len(x))

        try:
            d = np.linalg.solve(hessian, -gradient)
        except np.linalg.LinAlgError:
            print(f"Erro na iteração {k}: Hessiana não invertível.")
            return x, k

        # Salvaguarda: condição de descida
        max_inner_iter = 50   # Regula quantas iteracoes internas podem ocorrer
        inner_iter = 0
        while d.T @ gradient > -1e-3 * np.linalg.norm(gradient) * np.linalg.norm(d):
            inner_iter += 1
            if inner_iter >= max_inner_iter:
                print(f"Loop interno não convergiu na iteração {k}.")
                return x, k
            hessian = 0.9 * hessian + 0.1 * np.eye(len(x))
            d = np.linalg.solve(hessian, -gradient)

        if search:
            alpha = bl(f, x, -gradient, gradient, tau=tau, alpha_init=alpha_init, gama=gama)

        x += alpha * d
        gradient = grad(x) if not fd else fin_diff(f, x, 1, h=h)
        x_vals.append(x.copy())

    if plot:
        plot_curvas_nivel(f, np.array(x_vals))

    return x, k



def atualiza_hessiana(H, s, y):
    s = s.reshape(-1, 1)  
    y = y.reshape(-1, 1) 
    
    # Termo 1: (s^T y + y^T H y)(s s^T) / (s^T y)^2
    sTy = s.T @ y 
    yTHy = y.T @ H @ y  
    termo1 = ((sTy + yTHy) * (s @ s.T)) / (sTy ** 2)

    # Termo 2: (H y s^T + s y^T H) / (s^T y)
    termo2 = (H @ y @ s.T + s @ y.T @ H) / sTy

    H_new = H + termo1 - termo2

    return H_new



def bfgs(f, x0, grad, hess, eps=1e-5, itmax=10000, fd=False, h=1e-7, plot=False, tau=0.5, gama=0.1):
    x = x0.astype(np.float64)  
    k = 0
    gradient = grad(x) if not fd else fin_diff(f, x, 1, h=h)
    x_vals = [x.copy()]
    hessian = np.eye(len(x))  # Inicializa Hessiana como identidade

    while np.linalg.norm(gradient) > eps and k < itmax:
        k += 1

        s = x.copy()
        y = gradient.copy()
        d = np.linalg.solve(hessian, -gradient)

        # Salvaguarda: garantir a condição de descida
        while d.T @ gradient > -1e-3 * np.linalg.norm(gradient) * np.linalg.norm(d):
            hessian = 0.9 * hessian + 0.1 * np.eye(len(x))
            d = np.linalg.solve(hessian, -gradient)
        
        alpha = bl(f, x, d, gradient, tau, alpha_init=1.0, gama=0.1)

        x += alpha * d
        gradient = grad(x) if not fd else fin_diff(f, x, 1, h=h)

        s = x - s  
        y = gradient - y  

        hessian = atualiza_hessiana(hessian, s, y)

        x_vals.append(x.copy())

    if plot:
            plot_curvas_nivel(f, np.array(x_vals))

    return x, k




# ---------- Exemplo (PRINCIPAL) ----------
def f_prin(x):
    return x[0]**4 - 2*x[0]**2 + x[0] - x[0]*x[1] + x[1]**2
def grad_prin(x):
    return np.array([4*x[0]**3 - 4*x[0] + 1 - x[1], -x[0] + 2*x[1]])
#x, k = gd(f_prin, np.array([5,5]), grad_prin, fd=True, plot=True, search=False)
#print(f"Gradiente sem BL: x = {x}, iterações = {k}")

# ---------- Exemplo (BUSCA LINEAR) ----------
def f_linear(x):
    return x[0]**4-2*x[0]**2+x[0]-x[0]*x[1]+x[1]**2
def grad_linear(x):
    return np.array([4*x[0]**3-4*x[0]+1-x[1],-x[0]+2*x[1]])
#x9, k9 = gd(f_linear, np.array([10, 10]), grad_linear, search=True)
#print(f"Busca linear: x = {x9}, iterações = {k9}")

# ---------- Exemplo (NEWTON) ----------
def f_newton(x):
    return x[0]**4-2*x[0]**2+x[0]-x[0]*x[1]+x[1]**2
def grad_newton(x):
    return np.array([4*x[0]**3-4*x[0]+1-x[1],-x[0]+2*x[1]])
def hess_newton(x):
    return np.array([[12*x[0]**2-4,-1],[-1,2]])
#x, k = newton(f_newton,np.array([5,5]),grad_newton,hess_newton,alpha=1e0,eps = 1e-6,plot=True)
#print(f"Newton sem Sal. Guar.: x = {x}, iterações = {k}")
#x, k = newton_salvaguardas(f_newton,np.array([5,5]),grad_newton,hess_newton, search=True ,alpha=1e0,eps = 1e-6,plot=True)
#print(f"Newton com BL: x = {x}, iterações = {k}")
#x, k = newton_salvaguardas(f_newton,np.array([5,5]),grad_newton,hess_newton, search=False ,alpha=1e0,eps = 1e-6,plot=True)
#print(f"Newton sem BL: x = {x}, iterações = {k}")
#x, k = bfgs(f_newton, np.array([5,5]), grad_newton, hess_newton, eps = 1e-6, fd=True, plot=True)
#print(f"BFGS: x = {x}, iterações = {k}")

# ---------- Exemplos de Funcoes do Aluno ----------
# Quadratica classica
def f1(x):
    """
    Função Quadrática: f(x,y) = x^2 + y^2 + 2x + 3y
    Mínimo global: Em torno de (-1, -1.5)
    """
    return x[0]**2 + x[1]**2 + 2*x[0] + 3*x[1]

def grad1(x):
    """Gradiente da função quadrática f1"""
    return np.array([2*x[0] + 2, 2*x[1] + 3])

def hess1(x):
    x0, x1 = x[0], x[1]
    return np.array([[2 - 400*x1 + 1200*x0**2, -400*x0],
                     [-400*x0, 200]])

# Função f2
def f2(x):
    return np.exp(x[0]) + x[1]**2

def grad2(x):
    return np.array([np.exp(x[0]), 2*x[1]])

def hess2(x):
    return np.array([[np.exp(x[0]), 0], [0, 2]])

# Função f3
def f3(x):
    return x[0]**3 - 3*x[0]*x[1]**2

def grad3(x):
    return np.array([3*x[0]**2 - 3*x[1]**2, -6*x[0]*x[1]])

def hess3(x):
    return np.array([[6*x[0], -6*x[1]], [-6*x[1], -6*x[0]]])


# ---------- Plotagem e aplicacao ----------
# Gradiente Descendente para f1 
x1, k1 = gd(f1, np.array([1, 1]), grad=grad1, alpha=0.1, fd=False, plot=True, search=True)
print(f"Passo fixo: x = {x1}, iterações = {k1}")

# Diferenças finitas para f2 
x5, k5 = gd(f2, np.array([0, 0]), grad=grad2, alpha=0.1, eps=1e-8, fd=True, plot=True, search=False)
print(f"Diferenças finitas: x = {x5}, iterações = {k5}")

# Busca linear para f3 
x9, k9 = gd(f3, np.array([1, 0]), grad=grad3, fd=True, plot=True, search=True)
print(f"Busca linear: x = {x9}, iterações = {k9}")

# Newton
x, k = newton_salvaguardas(f2, np.array([2, 2]), grad2, hess2, eps=1e-5, alpha=0.1, itmax=10000, fd=False, h=1e-7, plot=True, search=True, tau=0.5, alpha_init=1.0, gama=0.1)
print(f"Newton SG: x = {x}, iterações = {k}")