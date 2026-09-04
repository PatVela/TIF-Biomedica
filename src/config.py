#-*- coding: utf-8 -*-
import argparse

# Configuración del analizador de argumentos
parser = argparse.ArgumentParser(description='Configuración para el entrenamiento y la predicción de modelos ECG.')

def add_argument_group(name):
    """
    Función auxiliar para crear grupos de argumentos en el analizador.
    Esto ayuda a organizar los argumentos relacionados en la ayuda de la línea de comandos.
    """
    arg = parser.add_argument_group(name)
    return arg


# --- Argumentos Misceláneos (misc_arg) ---
# Argumentos generales que controlan el comportamiento del flujo de trabajo.
misc_arg = add_argument_group('misc')
misc_arg.add_argument('--split', type=bool, default = True,
                      help='Indica si los datos deben dividirse en conjuntos de entrenamiento y prueba (True) o usar todo el conjunto como objetivo (False).')
misc_arg.add_argument('--input_size', type=int, default = 256,
                      help='Tamaño de entrada para el modelo, en múltiplos de 256, debido a la arquitectura del modelo.')
misc_arg.add_argument('--use_network', type=bool, default = False,
                      help='Controla si se utiliza una red preexistente (True) o se entrena una nueva (False).')

# --- Argumentos de Datos (data_arg) ---
# Argumentos relacionados con la descarga y preparación de los datos.
data_arg = add_argument_group('data')
data_arg.add_argument('--downloading', type=bool, default = False,
                      help='Indica si los datos del conjunto de PhysioNet deben descargarse automáticamente (True).')

# --- Argumentos de Grafo/Arquitectura del Modelo (graph_arg) ---
# Argumentos que definen la estructura y parámetros del modelo de red neuronal.
graph_arg = add_argument_group('graph')
graph_arg.add_argument('--filter_length', type=int, default = 32,
                       help='Número inicial de filtros en las capas convolucionales del modelo.')
graph_arg.add_argument('--kernel_size', type=int, default = 16,
                       help='Tamaño del kernel para las operaciones de convolución en el modelo.')
graph_arg.add_argument('--drop_rate', type=float, default = 0.2,
                       help='Tasa de abandono (dropout) para la regularización en las capas del modelo.')

# --- Argumentos de Entrenamiento (train_arg) ---
# Argumentos que controlan el proceso de entrenamiento del modelo.
train_arg = add_argument_group('train')
train_arg.add_argument('--feature', type=str, default = "MLII",
                       help='La característica o derivación del ECG a utilizar para el entrenamiento (ej. MLII, V1, V2, V4, V5). Se recomienda MLII o V1.')
train_arg.add_argument('--epochs', type=int, default = 80,
                       help='Número total de épocas para entrenar el modelo.')
train_arg.add_argument('--batch', type=int, default = 256,
                       help='Tamaño del batch para el entrenamiento.')
train_arg.add_argument('--patience', type=int, default = 10,
                       help='Número de épocas sin mejora en la validación antes de detener el entrenamiento (Early Stopping).')
train_arg.add_argument('--min_lr', type=float, default = 0.00005,
                       help='Tasa de aprendizaje mínima para el ajuste de la tasa de aprendizaje durante el entrenamiento.')
train_arg.add_argument('--checkpoint_path', type=str, default = None,
                       help='Ruta para cargar un checkpoint del modelo si se desea reanudar el entrenamiento.')
train_arg.add_argument('--resume_epoch', type=int,
                       help='Época desde la cual reanudar el entrenamiento si se carga un checkpoint.')
train_arg.add_argument('--ensemble', type=bool, default = False,
                       help='Indica si se utiliza un enfoque de ensemble (combinación de modelos) para la predicción.')
train_arg.add_argument('--trained_model', type=str, default = None,
                       help='Directorio y nombre de archivo del modelo entrenado a usar para la inferencia.')

# --- Argumentos de Predicción (predict_arg) ---
# Argumentos específicos para el modo de predicción.
predict_arg = add_argument_group('predict')
predict_arg.add_argument('--num', type=int, default = None,
                       help='Número de registro del conjunto de datos de prueba a cargar para la predicción. Si es None, se selecciona uno aleatoriamente.')
predict_arg.add_argument('--upload', type=bool, default = False,
                       help='Indica si los datos de entrada provienen de una carga de archivo (True) o del conjunto de datos CINC (False).')
predict_arg.add_argument('--sample_rate', type=int, default = None,
                      help='Frecuencia de muestreo de los datos de entrada. Si es None, se asume 300 Hz (frecuencia de CINC).')
predict_arg.add_argument('--cinc_download', type=bool, default = False,
                      help='Indica si los datos del desafío CINC 2017 deben descargarse (True).')

def get_config():
    """
    Función para obtener la configuración parseada de los argumentos de línea de comandos.
    """
    config, unparsed = parser.parse_known_args()

    return config