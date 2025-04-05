import numpy as np

def bl(f, x, d, grad_x, tau=0.5, alpha_init=1.0, gama=0.5):
    """
    Implementa a busca linear para encontrar o melhor tamanho de passo.
    """
    alpha = alpha_init
    fx = f(x)
    while f(x + alpha * d) > fx + alpha * tau * np.dot(grad_x, d):
        alpha *= gama
    return alpha

def linesearch(f, x, g, d, eps, itmax):
    k = 0
    while np.linalg.norm(g) < eps and k < itmax:
        k += 1
        alpha = bl(f, x, d, g, tau=tau, gama=gama)
        x -= alpha * np.linalg(g)
    return x, k