import numpy as np

def fin_diff(f, x, degree, h=1e-7):
    """
    Calcula o gradiente ou a matriz Hessiana usando diferenças finitas centradas.
    
    Parâmetros:
        x: numpy array com o ponto no qual calcular as derivadas.
        degree: Grau da derivada (1 para gradiente, 2 para Hessiana).
        h: Passo de diferença finita.
        
    Retorna:
        gradiente (vetor) se degree=1, ou a matriz Hessiana se degree=2.
    """
    n = len(x)  # Número de variáveis
    if degree == 1:
        # Gradiente (derivadas de primeira ordem)
        grad = np.zeros(n)
        for i in range(n):
            x1 = np.copy(x)
            x2 = np.copy(x)
            x1[i] += h
            x2[i] -= h
            grad[i] = (f(x1) - f(x2)) / (2 * h)
        return grad

    elif degree == 2:
        # Matriz Hessiana (derivadas de segunda ordem)
        hess = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                x1 = np.copy(x)
                x2 = np.copy(x)
                x3 = np.copy(x)
                x4 = np.copy(x)
                # Modificar as variáveis em relação a i e j
                x1[i] += h
                x1[j] += h
                x2[i] += h
                x2[j] -= h
                x3[i] -= h
                x3[j] += h
                x4[i] -= h
                x4[j] -= h
                # Calcula a segunda derivada usando a fórmula central
                hess[i, j] = (f(x1) - f(x2) - f(x3) + f(x4)) / (4 * h**2)
        return hess
