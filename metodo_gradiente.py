import numpy as np
import matplotlib.pyplot as plt
import time

def metodo_gradiente(f, x0, grad, alpha=0.1, eps=1e-5, itmax=10000):
    x = x0.copy()
    k = 0
    trajectory = [x.copy()]

    while k < itmax and np.linalg.norm(g) < eps:
        g = grad(x)
        k += 1
        d = -g
        x = x + alpha * d
        trajectory.append(x.copy())

    if plot and len(x0) == 2:
        td.td(f, trajectory)

    return x, k