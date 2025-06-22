from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, LearningRateScheduler
from keras import models
from graph import ECG_model
from config import get_config
from utils import *
import contextlib # Importar contextlib para redirección de stderr
import os # Importar os para os.devnull

# Configuración de variables de entorno para suprimir mensajes de log de TensorFlow y OneDNN
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def train(config, X, y, Xval=None, yval=None):
    """
    Función principal para entrenar el modelo de ECG.

    Args:
        config (object): Objeto de configuración con parámetros de entrenamiento.
        X (np.array): Datos de entrenamiento.
        y (np.array): Etiquetas de entrenamiento.
        Xval (np.array, optional): Datos de validación. Por defecto, None.
        yval (np.array, optional): Etiquetas de validación. Por defecto, None.
    """
    
    # Definición de las clases de latidos cardíacos que el modelo clasificará.
    # Algunas clases se excluyen si son muy pocas o no están en el conjunto de entrenamiento.
    classes = ['N','V','/','A','F','~'] # N: Normal, V: Ventricular, /: Paced, A: Auricular, F: Fusión, ~: Ruido

    # Impresión de las formas (shapes) iniciales de los datos para depuración.
    print("Formas iniciales - X:", X.shape, "y:", y.shape)
    print("Formas de validación inicial - Xval:", Xval.shape if Xval is not None else None, "yval:", yval.shape if yval is not None else None)
    # Verificación de valores NaN (Not a Number) en los datos iniciales.
    print("Cualquier NaN en inicial X:", np.any(np.isnan(X)), "y:", np.any(np.isnan(y)))
    
    # Expande las dimensiones de los datos de entrenamiento para que coincidan con la entrada del modelo Keras (samples, timesteps, features).
    # Aquí se asume que cada señal es una 'muestra' con 'input_size' timesteps y 1 'feature'.
    Xe = np.expand_dims(X, axis=2)

    # Lógica para manejar la división de datos (si no se han dividido previamente).
    if not config.split:
        # Si config.split es False, se realiza una división del 20% para validación.
        from sklearn.model_selection import train_test_split
        Xe, Xvale, y, yval = train_test_split(Xe, y, test_size=0.2, random_state=1)
    else:
        # Si config.split es True, se asume que Xval y yval ya existen y se expanden sus dimensiones.
        Xvale = np.expand_dims(Xval, axis=2)

    # Impresión de las formas de los datos después de la preparación para el entrenamiento.
    print("Formas de datos antes del entrenamiento - Xe:", Xe.shape, "y:", y.shape)
    print("Formas Val antes del entrenamiento - Xvale:", Xvale.shape, "yval:", yval.shape)
    
    print("Formas finales - Xe:", Xe.shape, "y:", y.shape)
    print("Formas Val finales - Xvale:", Xvale.shape, "yval:", yval.shape)

    # Carga de un modelo pre-entrenado o creación de uno nuevo.
    if config.checkpoint_path is not None:
        # Si se especifica una ruta de checkpoint, se carga el modelo desde allí.
        model = models.load_model(config.checkpoint_path)
        initial_epoch = config.resume_epoch # Se establece la época desde la cual reanudar el entrenamiento.
    else:
        # Si no hay checkpoint, se crea un nuevo modelo utilizando la función ECG_model.
        model = ECG_model(config)
        initial_epoch = 0 # El entrenamiento comienza desde la época 0.

    # Asegura que el directorio 'modelos' exista para guardar los modelos.
    mkdir_recursive('modelos')
    
    # Validar que los datos de entrada no contengan valores NaN, lo que podría causar problemas durante el entrenamiento.
    if np.any(np.isnan(Xe)) or np.any(np.isnan(y)):
        raise ValueError("Los datos de entrada contienen valores None/NaN")
    if np.any(np.isnan(Xvale)) or np.any(np.isnan(yval)):
        raise ValueError("Los datos de validación contienen valores None/NaN")

    # Definición de callbacks para el entrenamiento del modelo.
    callbacks = [
            # EarlyStopping: Detiene el entrenamiento si la métrica de monitoreo no mejora después de un número de épocas (patience).
            EarlyStopping(patience = config.patience, verbose=1),
            # ReduceLROnPlateau: Reduce la tasa de aprendizaje si la métrica de monitoreo deja de mejorar.
            ReduceLROnPlateau(factor = 0.5, patience = 3, min_lr = 0.01, verbose=1),
            # TensorBoard: Permite visualizar el progreso del entrenamiento en TensorBoard.
            TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True),
            # ModelCheckpoint: Guarda el modelo (o solo sus pesos) en puntos específicos durante el entrenamiento.
            # Guarda el modelo con el mejor 'val_loss' cada 10 batches.
            ModelCheckpoint('modelos/{}-latest.keras'.format(config.feature), monitor='val_loss', save_best_only=False, verbose=1, save_freq=10)
            # , lr_decay_callback # Callback de ajuste de tasa de aprendizaje por batch (comentado).
    ]

    # Entrenamiento del modelo.
    model.fit(Xe, y,
            validation_data=(Xvale, yval), # Datos para la evaluación de validación.
            epochs=config.epochs, # Número total de épocas.
            batch_size=config.batch, # Tamaño del batch.
            callbacks=callbacks, # Lista de callbacks a usar.
            initial_epoch=initial_epoch) # Época inicial para reanudar el entrenamiento.

    # Imprime los resultados del entrenamiento y la evaluación.
    print_results(config, model, Xvale, yval, classes)

    # El modelo no se devuelve ya que se guarda mediante ModelCheckpoint.
    #return model

def main(config):
    """
    Función principal para la ejecución del script de entrenamiento.

    Args:
        config (object): Objeto de configuración.
    """
    print('feature:', config.feature) # Muestra la característica (derivación de ECG) que se está utilizando.
    #np.random.seed(0) # Semilla para reproducibilidad si es necesario (comentado).
    
    # Carga los datos de entrenamiento y validación.
    (X,y, Xval, yval) = loaddata(config.input_size, config.feature)
    print(X, y) # Imprime una representación de los datos cargados (puede ser muy grande).
    
    # Llama a la función de entrenamiento con los datos cargados.
    train(config, X, y, Xval, yval)

if __name__=="__main__":
    # Bloque para ejecutar el script directamente.
    # Redirige stderr a /dev/null para suprimir los mensajes de log de TensorFlow/CUDA.
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stderr(fnull):
            config = get_config() # Obtiene la configuración de los argumentos de línea de comandos.
            main(config) # Llama a la función principal.