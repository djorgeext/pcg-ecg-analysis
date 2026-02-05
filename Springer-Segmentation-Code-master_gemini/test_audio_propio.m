%% 1. Configuración Inicial
clear all; close all; clc;

Fs_original = 1000; % Tu frecuencia de muestreo actual
springer_options = default_Springer_HSMM_options;
Fs_features = springer_options.audio_segmentation_Fs; % Por defecto es 50 Hz

% Cargar datos (Asegurando que sean double y vectores columna)
% NOTA: Asumo que los .txt son solo números crudos.
audio_train = load('/home/david/Documents/pcg-ecg-analysis/mediciones_simultaneas/DAVID171225_2-JORGE-20051201021025/pcg_cut_matlab.txt');
r_peaks_train = load('/home/david/Documents/pcg-ecg-analysis/mediciones_simultaneas/DAVID171225_2-JORGE-20051201021025/R_marks_adjusted.txt');
t_points_train = load('/home/david/Documents/pcg-ecg-analysis/mediciones_simultaneas/DAVID171225_2-JORGE-20051201021025/T_marks_adjusted.txt');

audio_test = load('/home/david/Documents/pcg-ecg-analysis/mediciones_simultaneas/DAVID171225_3-JORGE-20051201023210/pcg_cut_matlab.txt');

% Forzar a vector columna si vienen como fila
audio_train = audio_train(:);
r_peaks_train = r_peaks_train(:);
t_points_train = t_points_train(:);
audio_test = audio_test(:);

%% 2. Preparación de Datos de ENTRENAMIENTO
% Convertir anotaciones de 1000 Hz a 50 Hz
r_peaks_train_50hz = round(double(r_peaks_train) * (Fs_features / Fs_original));
t_points_train_50hz = round(double(t_points_train) * (Fs_features / Fs_original));

% Asegurarse de que no haya índices < 1
r_peaks_train_50hz(r_peaks_train_50hz < 1) = 1;
t_points_train_50hz(t_points_train_50hz < 1) = 1;

% IMPORTANTE: Filtro de seguridad
% Si por el redondeo algún índice excede la longitud esperada de las características,
% el entrenamiento fallará. Calculamos la longitud esperada aproximada.
len_features_aprox = ceil(length(audio_train) * (Fs_features / Fs_original));
r_peaks_train_50hz(r_peaks_train_50hz > len_features_aprox) = len_features_aprox;
t_points_train_50hz(t_points_train_50hz > len_features_aprox) = len_features_aprox;


% Crear las estructuras de entrada (Cell Arrays)
train_recordings = {audio_train}; 
train_annotations = cell(1, 2);
train_annotations{1, 1} = r_peaks_train_50hz;
train_annotations{1, 2} = t_points_train_50hz;

%% 3. Entrenar el Algoritmo
disp('Entrenando el algoritmo...');
% ¡ASEGÚRATE DE HABER APLICADO EL CAMBIO EN getSpringerPCGFeatures.m ANTES DE EJECUTAR ESTO!
[B_matrix, pi_vector, total_obs_distribution] = trainSpringerSegmentationAlgorithm(...
    train_recordings, ...
    train_annotations, ...
    Fs_original, ...
    false);

disp('Entrenamiento completado.');

%% 4. Probar con Datos de TEST
disp('Ejecutando en datos de test...');

% Ejecutar el algoritmo
assigned_states = runSpringerSegmentationAlgorithm(...
    audio_test, ...
    Fs_original, ...
    B_matrix, ...
    pi_vector, ...
    total_obs_distribution, ...
    true);

%% 5. (Opcional) Evaluar precisión si tienes las anotaciones de test
% Para comparar, necesitarías convertir los 'assigned_states' (que están a 1000Hz)
% a eventos discretos y compararlos con r_peaks_test (1000Hz).