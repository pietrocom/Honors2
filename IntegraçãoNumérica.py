import math

# Definir a função que desejamos integrar
def func(x):
    return x * x  # Exemplo: f(x) = x^2

# Método dos Trapézios
def trapezios(a, b, n):
    h = (b - a) / n
    soma = (func(a) + func(b)) / 2.0

    for i in range(1, n):
        soma += func(a + i * h)

    return soma * h

# Método de Simpson
def simpson(a, b, n):
    if n % 2 != 0:
        print("Número de subintervalos deve ser par para Simpson.")
        return 0

    h = (b - a) / n
    soma = func(a) + func(b)

    for i in range(1, n):
        x = a + i * h
        if i % 2 == 0:
            soma += 2 * func(x)
        else:
            soma += 4 * func(x)

    return soma * h / 3.0

# Função principal
def main():
    # Entrada do usuário para o intervalo e o número de subintervalos
    a = float(input("Digite o limite inferior do intervalo: "))
    b = float(input("Digite o limite superior do intervalo: "))
    n = int(input("Digite o número de subintervalos (par para Simpson): "))

    # Calculando usando os dois métodos
    resultado_trapezios = trapezios(a, b, n)
    resultado_simpson = simpson(a, b, n)

    # Exibindo os resultados
    print(f"Resultado pelo método dos Trapézios: {resultado_trapezios}")
    print(f"Resultado pelo método de Simpson: {resultado_simpson}")

if __name__ == "__main__":
    main()
