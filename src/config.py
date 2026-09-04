# -*- coding: utf-8 -*-

import argparse


# ============================================================
# CONFIGURACIÓN DEL ANALIZADOR DE ARGUMENTOS
# ============================================================

parser = argparse.ArgumentParser(
    description='Configuración para el entrenamiento y la predicción '
                'de modelos ECG con PhysioNet/CinC Challenge 2020.'
)


def add_argument_group(name):
    """
    Función auxiliar para crear grupos de argumentos.
    """
    return parser.add_argument_group(name)


# ============================================================
# ARGUMENTOS MISCELÁNEOS
# ============================================================

misc_arg = add_argument_group('misc')

misc_arg.add_argument(
    '--split',
    action='store_true',
    default=True,
    help='Utilizar los conjuntos de entrenamiento y validación.'
)

misc_arg.add_argument(
    '--input_length',
    type=int,
    default=5000,
    help='Número de muestras de cada ECG. '
         'CinC 2020 se procesa a 500 Hz durante 10 segundos = 5000 muestras.'
)

misc_arg.add_argument(
    '--num_leads',
    type=int,
    default=12,
    help='Número de derivaciones ECG utilizadas como entrada.'
)

misc_arg.add_argument(
    '--num_classes',
    type=int,
    default=27,
    help='Número de clases diagnósticas puntuadas de CinC 2020.'
)

misc_arg.add_argument(
    '--seed',
    type=int,
    default=42,
    help='Semilla aleatoria para reproducibilidad.'
)


# ============================================================
# ARGUMENTOS DE DATOS
# ============================================================

data_arg = add_argument_group('data')

data_arg.add_argument(
    '--train_file',
    type=str,
    default='data/train.h5',
    help='Archivo HDF5 con los datos de entrenamiento.'
)

data_arg.add_argument(
    '--val_file',
    type=str,
    default='data/val.h5',
    help='Archivo HDF5 con los datos de validación.'
)

data_arg.add_argument(
    '--test_file',
    type=str,
    default='data/test.h5',
    help='Archivo HDF5 con los datos de prueba.'
)

data_arg.add_argument(
    '--num_workers',
    type=int,
    default=2,
    help='Número de procesos utilizados por el DataLoader.'
)

data_arg.add_argument(
    '--pin_memory',
    action='store_true',
    default=True,
    help='Utilizar memoria fijada para acelerar la transferencia CPU-GPU.'
)


# ============================================================
# ARGUMENTOS DE ARQUITECTURA DEL MODELO
# ============================================================

graph_arg = add_argument_group('graph')

graph_arg.add_argument(
    '--filter_length',
    type=int,
    default=32,
    help='Número inicial de filtros de las capas convolucionales.'
)

graph_arg.add_argument(
    '--kernel_size',
    type=int,
    default=7,
    help='Tamaño del kernel de las convoluciones 1D.'
)

graph_arg.add_argument(
    '--drop_rate',
    type=float,
    default=0.2,
    help='Tasa de dropout utilizada para regularización.'
)


# ============================================================
# ARGUMENTOS DE ENTRENAMIENTO
# ============================================================

train_arg = add_argument_group('train')

train_arg.add_argument(
    '--epochs',
    type=int,
    default=80,
    help='Número máximo de épocas de entrenamiento.'
)

train_arg.add_argument(
    '--batch',
    type=int,
    default=8,
    help='Tamaño del batch. Se recomienda comenzar con 8 debido a los 4 GB de VRAM.'
)

train_arg.add_argument(
    '--lr',
    type=float,
    default=0.001,
    help='Tasa de aprendizaje inicial.'
)

train_arg.add_argument(
    '--min_lr',
    type=float,
    default=0.00005,
    help='Tasa de aprendizaje mínima.'
)

train_arg.add_argument(
    '--weight_decay',
    type=float,
    default=1e-4,
    help='Factor de regularización L2 del optimizador.'
)

train_arg.add_argument(
    '--patience',
    type=int,
    default=10,
    help='Número de épocas sin mejora antes de activar Early Stopping.'
)

train_arg.add_argument(
    '--checkpoint_path',
    type=str,
    default=None,
    help='Ruta del checkpoint desde el cual reanudar el entrenamiento.'
)

train_arg.add_argument(
    '--resume_epoch',
    type=int,
    default=None,
    help='Época desde la cual se reanuda el entrenamiento.'
)

train_arg.add_argument(
    '--trained_model',
    type=str,
    default=None,
    help='Ruta del modelo entrenado utilizado para inferencia.'
)


# ============================================================
# ARGUMENTOS DE HARDWARE
# ============================================================

device_arg = add_argument_group('device')

device_arg.add_argument(
    '--device',
    type=str,
    default='cuda',
    choices=['cuda', 'cpu'],
    help='Dispositivo utilizado para el entrenamiento.'
)

device_arg.add_argument(
    '--amp',
    action='store_true',
    default=True,
    help='Utilizar Automatic Mixed Precision para reducir el consumo de VRAM.'
)


# ============================================================
# ARGUMENTOS DE PREDICCIÓN
# ============================================================

predict_arg = add_argument_group('predict')

predict_arg.add_argument(
    '--num',
    type=int,
    default=None,
    help='Índice del registro a utilizar durante la predicción.'
)

predict_arg.add_argument(
    '--upload',
    action='store_true',
    default=False,
    help='Utilizar un ECG externo en lugar de un registro del conjunto de datos.'
)

predict_arg.add_argument(
    '--sample_rate',
    type=int,
    default=500,
    help='Frecuencia de muestreo esperada. El preprocessing de CinC 2020 '
         'normaliza las señales a 500 Hz.'
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

def get_config():
    """
    Obtiene la configuración parseada desde los argumentos
    de línea de comandos.
    """
    config, unparsed = parser.parse_known_args()

    return config