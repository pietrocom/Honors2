import math   # Para o uso dos recursos da bilbioteca "math", consultar manual disponivel em:
              # <https://docs.python.org/pt-br/3/library/math.html>

#1 Metodo dos Trapezios
def trapezio(f, a, b, n):
    h = (b - a) / n
    soma = (f(a) + f(b)) / 2.0
    for i in range(1, n):
        soma += f(a + i * h)
    return soma * h

#2 Metodo de Simpson
def simpson(f, a, b, n):
    if n % 2 != 0:                                                               # Testa se n eh par
        raise ValueError("Número de subintervalos deve ser par para Simpson.")   # Se nao for, aparece uma mensagem de erro que encerra
                                                                                 # o codigo
    h = (b - a) / n
    soma = f(a) + f(b)
    for i in range(1, n):
        x = a + i * h
        if i % 2 == 0:
            soma += 2 * f(x)
        else:
            soma += 4 * f(x)
    return soma * h / 3.0

def f(x):
    f = x**5 
    return f


a = 0         # Optei por deixar os parametros fora da funcao por fins praticos
b = 1         # Caso seja do interesse colocar limites ou subintervalos diferentes,
n = 90        # basta faze-lo manualmente diretamente nas funcoes que preferir

print(trapezio(f, a, b, n)) # No caso de inserir valores manualmente, devem ser colocados nos parametros das funcoes na ordem descrita abaixo:
print(simpson(f, a, b, n))  # (lambda x: funcao de x, limite inferior, limite superior, numero de subintervalos)

