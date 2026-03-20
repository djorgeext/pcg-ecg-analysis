N = 1024;

% Opción A: Ventana de Bartlett (Garantiza que el primer y último valor
% sean exactamente 0)
triangulo_bartlett = bartlett(N);

% Opción B: Ventana Triangular estándar (Los extremos no tocan el 0
% absoluto, útil para evitar solapamientos en FFT de ventanas contiguas)
triangulo_triang = triang(N);

hilb = hilbert(triangulo_triang);

% 1. Abrir (o crear) el archivo de texto en modo escritura ('w')
fileID = fopen('triangulo.txt', 'w');

% 2. Escribir los datos en el archivo
% '%1.6f\n' indica: formato float (f) con 6 decimales, seguido de un salto de línea (\n)
fprintf(fileID, '%1.6f\n', triangulo_triang);

% 3. Cerrar el archivo (muy importante para que se guarden los cambios)
fclose(fileID);
hilb_c = load('hilb_c.txt');
hilb_c = hilb_c(1:end,1) + 1i*hilb_c(1:end,2);

disp('Archivo triangulo.txt guardado exitosamente.');
error_real = abs(real(hilb) - real(hilb_c));
error_imag = abs(imag(hilb) - imag(hilb_c));
figure(1),plot(error_real);
figure(2),plot(error_imag)