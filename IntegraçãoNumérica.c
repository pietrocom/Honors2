#include <stdio.h>
#include <math.h>

// Definir a função que desejamos integrar
double func(double x) {
    return x * x;  // Exemplo: f(x) = x^2
}

// Método dos Trapézios
double trapezios(double a, double b, int n) {
    double h = (b - a) / n;
    double soma = (func(a) + func(b)) / 2.0;

    for (int i = 1; i < n; i++) {
        soma += func(a + i * h);
    }

    return soma * h;
}

// Método de Simpson
double simpson(double a, double b, int n) {
    if (n % 2 != 0) {
        printf("Número de subintervalos deve ser par para Simpson.\n");
        return 0;
    }

    double h = (b - a) / n;
    double soma = func(a) + func(b);

    for (int i = 1; i < n; i++) {
        double x = a + i * h;
        if (i % 2 == 0) {
            soma += 2 * func(x);
        } else {
            soma += 4 * func(x);
        }
    }

    return soma * h / 3.0;
}

int main() {
    double a, b;
    int n;

    // Entrada do usuário para o intervalo e o número de subintervalos
    printf("Digite o limite inferior do intervalo: ");
    scanf("%lf", &a);
    printf("Digite o limite superior do intervalo: ");
    scanf("%lf", &b);
    printf("Digite o número de subintervalos (par para Simpson): ");
    scanf("%d", &n);

    // Calculando usando os dois métodos
    double resultado_trapezios = trapezios(a, b, n);
    double resultado_simpson = simpson(a, b, n);

    // Exibindo os resultados
    printf("Resultado pelo método dos Trapézios: %lf\n", resultado_trapezios);
    printf("Resultado pelo método de Simpson: %lf\n", resultado_simpson);

    return 0;
}

