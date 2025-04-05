import numpy as np
import matplotlib.pyplot as plt
import math

def fd_error(f, df, x0, h0, hn, n):
    # Gerar valores de h em escala logarítmica
    h = np.logspace(np.log10(h0), np.log10(hn), n + 1)
    lista_erro = []

    # Calcular o erro para cada valor de h
    for i in range(n + 1):
        d_aprox = (f(x0 + h[i]) - f(x0)) / h[i]
        erro = abs(d_aprox - df(x0))
        lista_erro.append(erro)

    # Plotar o gráfico em escala log-log
    plt.loglog(h, lista_erro, label='Erro da Aproximação')
    plt.xlabel('h')
    plt.ylabel('Erro')
    plt.title('Erro vs. h para a Aproximação da Derivada')
    plt.legend()
    plt.gca().invert_xaxis()
    plt.show()

def ode_solver(f, x0, y0, xn, n, plot):
    i = 0
    h = (xn - x0) / n
    x, y = [], []
    valx = x0
    valy = y0
    for i in range(n + 1):
        x.append(valx)
        y.append(valy)
        valy += h * f(valx, valy)
        valx += h

    if plot:
        plt.plot(x, y, label="Solução Numérica", color="b")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Solução Numérica da EDO')
        plt.legend()
        plt.show()

    return np.array(x), np.array(y)



fd_error(lambda x: math.atan(x),lambda x:1/(1+x**2),1,1e-15,1e-1,100)

ode_solver(lambda x,y: np.exp(-x**2),-3,-0.8862073482595,3,500,True)

ode_solver(lambda x,y: 10*np.sqrt(y)*np.sin(x)+x,0,0,100,500,True)

x,y = ode_solver(lambda x,y: np.cos(x**2),0.5,2,5,4,True)
print(np.array([x,y]).transpose())