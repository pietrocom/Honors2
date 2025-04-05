import numpy as np
import matplotlib.pyplot as plt
import time

def td(f, trajectory):
    """
    Exibe a trajetória do gradiente descendente dinamicamente.
    """
    # Definindo os valores de x e y para o grid de contorno
    x_vals = np.linspace(-2, 2, 100)
    y_vals = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.array([f(np.array([x, y])) for x, y in zip(X.ravel(), Y.ravel())]).reshape(X.shape)

    plt.figure() 
    ax = plt.gca()
    ax.contour(X, Y, Z, levels=50, cmap='viridis')

    trajectory = np.array(trajectory)

    scatter, = ax.plot([], [], marker='o', color='red', label='Trajetória')

    total_points = len(trajectory)
    delay = max(0.05, 0.5 / total_points)  

    for i in range(total_points):
        scatter.set_data(trajectory[:i+1, 0], trajectory[:i+1, 1])
        plt.pause(delay)
        plt.draw()  

    plt.legend()
    plt.title('Trajetória do Método do Gradiente (Dinâmica)')
    plt.show()  