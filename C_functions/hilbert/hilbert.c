#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>

#define PI 3.14159265358979323846

// 1. Función para calcular la FFT (Requiere que N sea potencia de 2)
void fft(double complex *X, int N) {
    if (N <= 1) return;

    // Dividir en pares e impares
    double complex *even = malloc(N / 2 * sizeof(double complex));
    double complex *odd = malloc(N / 2 * sizeof(double complex));

    for (int i = 0; i < N / 2; i++) {
        even[i] = X[i * 2];
        odd[i] = X[i * 2 + 1];
    }

    // Recursión
    fft(even, N / 2);
    fft(odd, N / 2);

    // Combinar resultados
    for (int k = 0; k < N / 2; k++) {
        double complex t = cexp(-I * 2 * PI * k / N) * odd[k];
        X[k] = even[k] + t;
        X[k + N / 2] = even[k] - t;
    }

    free(even);
    free(odd);
}

// 2. Función para calcular la IFFT (Transformada Inversa)
void ifft(double complex *X, int N) {
    // Tomar el conjugado
    for (int i = 0; i < N; i++) X[i] = conj(X[i]);
    
    // Aplicar FFT normal
    fft(X, N);
    
    // Tomar el conjugado nuevamente y normalizar
    for (int i = 0; i < N; i++) {
        X[i] = conj(X[i]) / N;
    }
}

// 3. El clon exacto de hilbert() de MATLAB
void hilbert_matlab(double *x, double complex *z, int N) {
    // A. Copiar la señal real al arreglo de números complejos
    for (int i = 0; i < N; i++) {
        z[i] = x[i] + 0.0 * I;
    }

    // B. Transformada al dominio de la frecuencia
    fft(z, N);

    // C. Crear y aplicar el vector multiplicador 'h'
    double *h = malloc(N * sizeof(double));
    h[0] = 1.0;
    
    if (N % 2 == 0) { // Si N es par
        for (int i = 1; i < N / 2; i++) h[i] = 2.0;
        h[N / 2] = 1.0;
        for (int i = N / 2 + 1; i < N; i++) h[i] = 0.0;
    } else {          // Si N es impar
        for (int i = 1; i <= (N - 1) / 2; i++) h[i] = 2.0;
        for (int i = (N - 1) / 2 + 1; i < N; i++) h[i] = 0.0;
    }

    // Multiplicar elemento a elemento
    for (int i = 0; i < N; i++) {
        z[i] *= h[i];
    }
    free(h);

    // D. Transformada Inversa de vuelta al dominio del tiempo
    ifft(z, N);
}

// --- Pruebas ---
int main() {
    int N = 1024;
    
    // 1. Reservar memoria dinámicamente para los arreglos
    // Usar malloc es crucial para arreglos grandes y no desbordar el stack
    double *senal_real = malloc(N * sizeof(double));
    double complex *resultado = malloc(N * sizeof(double complex));

    if (senal_real == NULL || resultado == NULL) {
        printf("Error: No se pudo asignar memoria.\n");
        return 1;
    }

    // 2. Leer el archivo de entrada (triangulo.txt)
    FILE *archivo_entrada = fopen("triangulo.txt", "r");
    if (archivo_entrada == NULL) {
        printf("Error: No se pudo abrir triangulo.txt. Asegurate de que este en la misma carpeta.\n");
        free(senal_real);
        free(resultado);
        return 1;
    }

    for (int i = 0; i < N; i++) {
        // Leer cada línea como un double (%lf)
        if (fscanf(archivo_entrada, "%lf", &senal_real[i]) != 1) {
            printf("Advertencia: Error de lectura en la linea %d. Revisa tu archivo de texto.\n", i + 1);
            break; 
        }
    }
    fclose(archivo_entrada);
    printf("Lectura de triangulo.txt completada.\n");

    // 3. Ejecutar la Transformada de Hilbert
    hilbert_matlab(senal_real, resultado, N);
    printf("Calculo de la Senal Analitica finalizado.\n");

    // 4. Guardar el archivo de salida (hilb_c.txt)
    FILE *archivo_salida = fopen("hilb_c.txt", "w");
    if (archivo_salida == NULL) {
        printf("Error: No se pudo crear el archivo hilb_c.txt.\n");
        free(senal_real);
        free(resultado);
        return 1;
    }

    // Escribir los resultados en dos columnas separadas por un espacio: [ParteReal] [ParteImaginaria]
    // Usamos %.6f para simular la precision de un float de 32 bits
    for (int i = 0; i < N; i++) {
        fprintf(archivo_salida, "%.6f %.6f\n", creal(resultado[i]), cimag(resultado[i]));
    }
    
    fclose(archivo_salida);
    printf("Resultados guardados exitosamente en hilb_c.txt.\n");

    // 5. Liberar la memoria
    free(senal_real);
    free(resultado);
    
    return 0;
}