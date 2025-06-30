from __future__ import division, print_function
from keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, f1_score, classification_report, accuracy_score
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suprime los mensajes de log de TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Deshabilita optimizaciones de OneDNN
import h5py # Para trabajar con archivos HDF5 (formato .keras)
import csv # Importa el módulo csv para leer y escribir archivos CSV
import pandas as pd # Importado para manejar datos y exportar a CSV
import hashlib # Importar para calcular hashes de los datos

def mkdir_recursive(path):
  """
  Crea un directorio de forma recursiva si no existe.

  Args:
      path (str): La ruta del directorio a crear.
  """
  if path == "":
    return
  sub_path = os.path.dirname(path) # Obtiene el directorio padre.
  if not os.path.exists(sub_path):
    mkdir_recursive(sub_path) # Llama recursivamente para crear directorios padres.
  if not os.path.exists(path):
    print("Creando directorio " + path)
    os.mkdir(path) # Crea el directorio actual.

def loaddata(input_size, feature):
    """
    Carga los datos de entrenamiento y validación desde archivos .keras.

    Args:
        input_size (int): Tamaño de la entrada del modelo (número de muestras por segmento).
        feature (str): La característica o derivación del ECG a cargar (ej. MLII).

    Returns:
        tuple: Una tupla que contiene (X_train, y_train, X_val, y_val) como arrays de NumPy.
    """
    mkdir_recursive('dataset') # Asegura que el directorio 'dataset' exista.
    print("Cargando datos de entrenamiento...")
    with h5py.File('dataset/train.keras', 'r') as f:
        # Carga todos los datasets del archivo HDF5 en un diccionario.
        trainData = {key: f[key][...] for key in f.keys()}

    print("Cargando etiquetas de entrenamiento...")
    with h5py.File('dataset/trainlabel.keras', 'r') as f:
        testlabelData = {key: f[key][...] for key in f.keys()}

    print("Características disponibles en los datos de entrenamiento:", list(trainData.keys()))
    print("Funciones disponibles en los datos de la etiqueta:", list(testlabelData.keys()))

    # Extrae los datos y etiquetas para la característica especificada.
    X = np.float32(trainData[feature])
    y = np.float32(testlabelData[feature])
    print("Training shapes before shuffle - X:", X.shape, "y:", y.shape)
    print("Any NaN in X:", np.any(np.isnan(X)), "y:", np.any(np.isnan(y)))

    # Concatena los datos y las etiquetas para mezclarlos juntos y mantener la correspondencia.
    att = np.concatenate((X,y), axis=1)
    np.random.shuffle(att) # Mezcla las filas aleatoriamente.
    # Divide de nuevo los datos y etiquetas después de mezclar.
    X, y = att[:,:input_size], att[:, input_size:]
    print("Training shapes after shuffle - X:", np.any(np.isnan(X)), "y:", np.any(np.isnan(y)))

    print("Cargando datos de validación...")
    with h5py.File('dataset/test.keras', 'r') as f:
        valData = {key: f[key][...] for key in f.keys()}

    print("Cargando etiquetas de validación...")
    with h5py.File('dataset/testlabel.keras', 'r') as f:
        vallabelData = {key: f[key][...] for key in f.keys()}

    # Extrae los datos y etiquetas de validación.
    Xval = np.float32(valData[feature])
    yval = np.float32(vallabelData[feature])
    print("Validation shapes - Xval:", Xval.shape, "yval:", yval.shape)
    print("Any NaN in validation - Xval:", np.any(np.isnan(Xval)), "yval:", np.any(np.isnan(yval)))

    return (X, y, Xval, yval) # Retorna los cuatro conjuntos de datos.

class LearningRateSchedulerPerBatch(LearningRateScheduler):
    """
    Clase de callback para modificar el programador de tasa de aprendizaje por defecto para operar cada batch.
    Código de https://towardsdatascience.com/resuming-a-training-process-con-keras-3e93152ee11a
    """
    def __init__(self, schedule, verbose=0):
        super(LearningRateSchedulerPerBatch, self).__init__(schedule, verbose)
        self.count = 0  # Índice de batch global.

    def on_epoch_begin(self, epoch, logs=None):
        pass # No hace nada al inicio de la época.

    def on_epoch_end(self, epoch, logs=None):
        pass # No hace nada al final de la época.

    def on_batch_begin(self, batch, logs=None):
        # Llama al método on_epoch_begin del padre, pero con el contador global de batches.
        super(LearningRateSchedulerPerBatch, self).on_epoch_begin(self.count, logs)

    def on_batch_end(self, batch, logs=None):
        # Llama al método on_epoch_end del padre, pero con el contador global de batches.
        super(LearningRateSchedulerPerBatch, self).on_epoch_end(self.count, logs)
        self.count += 1 # Incrementa el contador global de batches.


def plot_confusion_matrix(y_true, y_pred, classes, feature,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    Traza y/o imprime la matriz de confusión.
    Modificación del código de https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    Args:
        y_true (array): Etiquetas verdaderas.
        y_pred (array): Etiquetas predichas por el modelo.
        classes (list): Lista de nombres de clases.
        feature (str): La característica (derivación de ECG) usada, para el nombre del archivo.
        normalize (bool, optional): Si es True, normaliza la matriz. Por defecto, False.
        title (str, optional): Título del gráfico. Por defecto, None.
        cmap (matplotlib.colors.Colormap, optional): Mapa de colores para el gráfico. Por defecto, plt.cm.Blues.

    Returns:
        matplotlib.axes.Axes: Los ejes de la figura de la matriz de confusión.
    """
    if not title:
        if normalize:
            title = 'Matriz de confusión normalizada'
        else:
            title = 'Matriz de confusión, sin normalización'

    # Calcula la matriz de confusión.
    cm = confusion_matrix(y_true, y_pred)
    # classes = classes[unique_labels(y_true, y_pred)] # Comentado: Ajuste para clases únicas si es necesario.

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # Normaliza la matriz.
        print("Matriz de confusión normalizada")
    else:
        print('Matriz de confusión, sin normalización')

    print(cm) # Imprime la matriz de confusión en la consola.
    fig, ax = plt.subplots() # Crea una nueva figura y ejes.
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap) # Muestra la matriz como una imagen.
    ax.figure.colorbar(im, ax=ax) # Añade una barra de color.
    # Configura los ticks y etiquetas de los ejes.
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Etiqueta Verdadera',
           xlabel='Etiqueta Predicha')

    # Ajusta las etiquetas del eje x para que giren y se alineen a la derecha.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd' # Formato para mostrar los valores en la matriz.
    thresh = cm.max() / 2. # Umbral para el color del texto.
    # Añade los valores numéricos a las celdas de la matriz.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout() # Ajusta el layout de la figura.

    # Asegura que el directorio 'static/asset' exista para guardar la imagen.
    # La ruta es relativa al directorio raíz de la aplicación Flask (donde se encuentra app.py).
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'asset')
    mkdir_recursive(output_dir)

    # Guarda la matriz de confusión como una imagen PNG.
    png_path = os.path.join(output_dir, 'confusion_matrix.png')
    fig.savefig(png_path, format="png", dpi=300, bbox_inches='tight')

    # La línea para guardar como .eps puede mantenerse si es necesaria para otros propósitos.
    # fig.savefig('resultados/confusionMatrix-'+feature+'.eps', format='eps', dpi=1000)

    return ax # Retorna los ejes.


# Curvas de precisión-recuperación y curvas ROC para cada clase
def PR_ROC_curves(ytrue, ypred, classes, ypred_mat):
    """
    Genera y guarda las curvas Precision-Recall (PR) y Receiver Operating Characteristic (ROC)
    para cada clase.

    Args:
        ytrue (array): Etiquetas verdaderas (índices de clase).
        ypred (array): Etiquetas predichas (índices de clase).
        classes (list): Nombres de las clases.
        ypred_mat (array): Matriz de probabilidades predichas (scores) para cada clase.
    """
    ybool = ypred == ytrue # Booleano que indica si la predicción es correcta.
    f, ax = plt.subplots(3,4,figsize=(10, 10)) # Crea una figura con subplots.
    ax = [a for i in ax for a in i] # Aplanar la matriz de ejes a una lista.

    e = -1 # Contador para los ejes de los subplots.
    for c_idx, c in enumerate(classes): # Itera sobre cada clase.
        # Obtiene los índices de las muestras que pertenecen a la clase actual (verdaderas o predichas).
        idx1 = [n for n,x in enumerate(ytrue) if classes[x]==c]
        idx2 = [n for n,x in enumerate(ypred) if classes[x]==c]
        idx = list(set(idx1 + idx2)) # Combina y elimina duplicados.

        if not idx: # Si no hay muestras para esta clase, salta.
            continue

        # Selecciona las etiquetas verdaderas y probabilidades predichas para las muestras de la clase actual.
        bi_ytrue = (ytrue[idx] == c_idx).astype(int) # Convierte a 0 o 1 para la clase actual.
        bi_prob = ypred_mat[idx, c_idx] # Probabilidades para la clase actual.

        try:
            # Calcula y imprime el AUC (Area Under the Curve) para la curva ROC.
            auc_score = roc_auc_score(bi_ytrue, bi_prob)
            print(f"AUC para {c}: {auc_score}")
            e+=1 # Incrementa el contador de ejes.
        except ValueError:
            print(f"Advertencia: No se pudo calcular AUC para la clase {c}. Posiblemente solo una clase presente.")
            continue # Si hay un error (ej. solo una clase presente), salta.

        # Curva Precision-Recall
        ppvs, senss, _ = precision_recall_curve(bi_ytrue, bi_prob)
        cax = ax[2*e] # Selecciona el eje actual para PR.
        cax.plot(senss, ppvs, lw=2, label="Modelo") # Dibuja la curva PR.
        cax.set_xlim(-0.008, 1.05) # Establece los límites del eje x.
        cax.set_ylim(0.0, 1.05) # Establece los límites del eje y.
        cax.set_title(f"Clase {c}") # Título del subplot.
        cax.set_xlabel('Sensibilidad (Recall)')
        cax.set_ylabel('PPV (Precisión)')
        cax.legend(loc=3) # Posición de la leyenda.

        # Curva ROC (Receiver Operating Characteristic)
        fpr, tpr, _ = roc_curve(bi_ytrue, bi_prob)
        cax2 = ax[2*e+1] # Selecciona el eje actual para ROC.
        cax2.plot(fpr, tpr, lw=2, label="Modelo") # Dibuja la curva ROC.
        cax2.set_xlim(-0.1, 1.) # Establece los límites del eje x.
        cax2.set_ylim(0.0, 1.05) # Establece los límites del eje y.
        cax2.set_title(f"Clase {c}") # Título del subplot.
        cax2.set_xlabel('1 - Especificidad')
        cax2.set_ylabel('Sensibilidad')
        cax2.legend(loc=4) # Posición de la leyenda.

    mkdir_recursive("resultados") # Asegura que el directorio 'resultados' exista.
    plt.savefig("resultados/model_prec_recall_and_roc.eps",
        dpi=400,
        format='eps',
        bbox_inches='tight') # Guarda la figura.
    plt.close() # Cierra la figura para liberar memoria.

def print_results(config, model, Xval, yval, classes):
    """
    Imprime los resultados de la evaluación del modelo, incluyendo
    el reporte de clasificación, la matriz de confusión y las curvas PR/ROC.
    Ahora devuelve el accuracy y el f1-score.

    Args:
        config (object): Objeto de configuración.
        model (keras.Model): El modelo Keras entrenado.
        Xval (array): Datos de validación.
        yval (array): Etiquetas de validación (one-hot encoded).
        classes (list): Nombres de las clases.

    Returns:
        tuple: (accuracy_val, f1_score_val)
    """
    model2 = model # Copia del modelo para posible uso en ensemble (comentado).
    if config.trained_model:
        model.load_weights(config.trained_model) # Carga los pesos de un modelo pre-entrenado si se especifica.
    else:
        model.load_weights('modelos/{}-latest.keras'.format(config.feature)) # Carga los pesos del último modelo guardado.

    # Si se usa un enfoque de ensemble, las predicciones de múltiples modelos se promedian.
    if config.ensemble:
        model2.load_weight('modelos/weights-V1.keras') # Carga pesos de otro modelo (comentado).
        ypred_mat = (model.predict(Xval) + model2.predict(Xval))/2 # Promedia las predicciones.
    else:
        ypred_mat = model.predict(Xval) # Predice las probabilidades para los datos de validación.

    print("yval.shape", yval.shape) # Imprime la forma de las etiquetas de validación.

    ytrue = np.argmax(yval,axis=1) # Convierte las etiquetas one-hot a índices de clase.
    # Obtiene las probabilidades del modelo para las clases verdaderas.
    yscore = np.array([ypred_mat[x][ytrue[x]] for x in range(len(yval))])
    ypred = np.argmax(ypred_mat, axis=1) # Obtiene la clase predicha con mayor probabilidad.

    # Calcula el accuracy
    accuracy_val = accuracy_score(ytrue, ypred)

    # Calcula el F1-score (promedio ponderado)
    f1_score_val = f1_score(ytrue, ypred, average='weighted')

    print(classification_report(ytrue, ypred, target_names=classes)) # Imprime el reporte de clasificación.
    plot_confusion_matrix(ytrue, ypred, classes, feature=config.feature, normalize=False) # Traza la matriz de confusión no normalizada.
    print("F1 score:", f1_score(ytrue, ypred, average=None)) # Imprime el F1-score por clase.
    PR_ROC_curves(ytrue, ypred, classes, ypred_mat) # Genera y guarda las curvas PR y ROC.

    return accuracy_val, f1_score_val # Retorna los valores calculados

def add_noise(config):
    """
    Carga y procesa segmentos de datos clasificados como 'Ruido' del dataset CINC 2017
    para añadirlos como ejemplos negativos al conjunto de entrenamiento y prueba.

    Args:
        config (object): Objeto de configuración.

    Returns:
        dict: Un diccionario con listas de segmentos de ruido para entrenamiento y prueba.
    """
    noises = dict()
    noises["trainset"] = list()
    noises["testset"] = list()
    import csv
    try:
        # Intenta leer el archivo REFERENCE.csv para obtener las etiquetas de ruido.
        testlabel = list(csv.reader(open('training2017/REFERENCE.csv')))
    except:
        # Si el archivo no existe, lo descarga y descomprime el dataset CINC 2017.
        cmd = "curl -O https://archive.physionet.org/challenge/2017/training2017.zip"
        os.system(cmd)
        os.system("unzip training2017.zip")
        testlabel = list(csv.reader(open('training2017/REFERENCE.csv'))) # Vuelve a intentar leer.

    # Itera sobre las etiquetas del dataset.
    for i, label in enumerate(testlabel):
      if label[1] == '~': # Busca registros clasificados como 'Ruido' (~).
        filename = 'training2017/'+ label[0] + '.mat' # Construye el nombre del archivo .mat.
        from scipy.io import loadmat # Importa loadmat para leer archivos .mat.
        noise = loadmat(filename) # Carga los datos de ruido.
        noise = noise['val'] # Asume que los datos están en la clave 'val'.
        _, size = noise.shape
        noise = noise.reshape(size,) # Remodela los datos a un array 1D.
        noise = np.nan_to_num(noise) # Elimina NaNs e Infs.
        from scipy.signal import resample # Importa resample para el remuestreo.
        # Remuestrea el ruido para que coincida con la frecuencia de muestreo de los datos (360 Hz).
        noise= resample(noise, int(len(noise) * 360 / 300) )
        from sklearn import preprocessing # Importa preprocessing para escalado.
        noise = preprocessing.scale(noise) # Escala los datos de ruido.
        noise = noise/1000*6 # Normalización adicional (aproximada).
        from scipy.signal import find_peaks # Importa find_peaks.
        peaks, _ = find_peaks(noise, distance=150) # Encuentra picos en el ruido (aunque sea ruido).

        choices = 10 # Número de segmentos de ruido a extraer por registro.
        if len(peaks) > choices:
            # Selecciona aleatoriamente 'choices' picos para extraer segmentos de ruido.
            picked_peaks = np.random.choice(peaks, choices, replace=False)
        else:
            picked_peaks = peaks # Si no hay suficientes picos, usa todos los disponibles.

        # Extrae segmentos de ruido alrededor de los picos seleccionados.
        for j, peak in enumerate(picked_peaks):
          # Asegura que el segmento esté dentro de los límites del array.
          if peak > config.input_size//2 and peak < len(noise) - config.input_size//2:
              start,end  = peak-config.input_size//2, peak+config.input_size//2
              # Divide los segmentos de ruido entre entrenamiento y prueba.
              if i > len(testlabel)/6: # Aproximadamente 1/6 de los registros para prueba.
                noises["trainset"].append(noise[start:end].tolist())
              else:
                noises["testset"].append(noise[start:end].tolist())
    return noises # Retorna los conjuntos de ruido.

def preprocess(data, config):
    """
    Preprocesa los datos de la señal de ECG: remuestreo, escalado y expansión de dimensiones.
    También detecta los picos QRS en la señal.

    Args:
        data (np.array): Datos de la señal de ECG (originalmente 1D o 2D).
        config (object): Objeto de configuración con atributos como 'sample_rate' e 'input_size'.

    Returns:
        tuple: Una tupla que contiene (datos_procesados, picos_detectados).
               datos_procesados (np.array): Datos de la señal preprocesados y con dimensiones expandidas.
               picos_detectados (np.array): Índices de los picos QRS detectados.
    """
    # Obtiene la frecuencia de muestreo de la configuración; si no está definida, usa 300 Hz por defecto.
    sr = config.sample_rate
    if sr is None:
      sr = 300

    # Asegura que `data` sea un array 1D y numérico.
    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=np.float32)

    if data.ndim > 1:
        data = data.flatten() # Aplan el array si es multidimensional.

    data = np.nan_to_num(data) # Elimina NaNs e Infs en los datos.

    from scipy.signal import resample # Importa la función de remuestreo.
    # Remuestrea los datos a 360 Hz (frecuencia estándar de MIT-BIH Arrhythmia).
    data = resample(data, int(len(data) * 360 / sr) )

    from sklearn import preprocessing # Importa el módulo de preprocesamiento.
    data = preprocessing.scale(data) # Escala los datos para que tengan media 0 y varianza 1.

    # Calcular y imprimir el hash de los datos después de preprocesamiento
    # para verificar si son idénticos antes de la detección de picos.
    data_bytes = data.tobytes()
    data_hash = hashlib.sha256(data_bytes).hexdigest()
    print(f"DEBUG: Hash de los datos preprocesados antes de find_peaks: {data_hash}")


    from scipy.signal import find_peaks # Importa la función para encontrar picos.
    peaks, _ = find_peaks(data, distance=150)

    data = data.reshape(1, len(data)) # Remodela los datos a (1, longitud_señal)
    data = np.expand_dims(data, axis=2) # Expande las dimensiones a (1, longitud_señal, 1) para Keras.
    return data, peaks # Retorna los datos procesados y los picos.

def _extract_sampling_rate_from_comment(comment_line):
    """
    Extrae el sampling rate de una línea de comentario si está presente.
    Args:
        comment_line (str): Línea de comentario que puede contener el sampling rate.
    Returns:
        float or None: El sampling rate si se encuentra, de lo contrario None.
    """
    if 'Sampling Rate:' in comment_line:
        try:
            sr_part = comment_line.split('Sampling Rate:')[1].strip().split(' ')[0]
            return float(sr_part)
        except (ValueError, IndexError):
            return None
    return None


def uploadedData(filename, csvbool = True):
    """
    Lee datos de un archivo subido (CSV o potencialmente otros formatos en el futuro).

    Args:
        filename (str): La ruta del archivo subido.
        csvbool (bool, optional): Si es True, intenta leer el archivo como CSV. Por defecto, True.

    Returns:
        tuple: Una tupla que contiene (sampling_rate, signal_data_array).
               sampling_rate (float or None): Frecuencia de muestreo leída de la primera línea del CSV.
               signal_data_array (np.array): Datos de la señal como un array 1D o 2D (muestras, derivaciones).
    """
    if csvbool:
        sampling_rate = None
        with open(filename, 'r') as f:
            first_line = f.readline().strip()
            if first_line.startswith('#'):
                sampling_rate = _extract_sampling_rate_from_comment(first_line)
                try:
                    if sampling_rate is not None:
                        df = pd.read_csv(filename, skiprows=1)
                    else:
                        df = pd.read_csv(filename)
                except Exception:
                    df = pd.read_csv(filename)
            else:
                df = pd.read_csv(filename)
        signal_data_array = df.values.squeeze()
        return sampling_rate, signal_data_array
    else:
        raise NotImplementedError("uploadedData con csvbool=False no está completamente implementada para este caso.")