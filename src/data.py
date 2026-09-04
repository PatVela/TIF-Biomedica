"""
Los datos provienen de
https://physionet.org/physiobank/database/html/mitdbdir/mitdbdir.htm

Los registros se digitalizaron a 360 muestras por segundo y canal con una resolución de 11 bits en un intervalo de 10 mV.
Dos o más cardiólogos anotaron independientemente cada registro; los desacuerdos se resolvieron para obtener las anotaciones de referencia legibles por ordenador
para cada latido (aproximadamente 110.000 anotaciones en total) incluidas en la base de datos.

 Código Descripción
 N Latido normal (mostrado como . por el PhysioBank ATM, LightWAVE, pschart, y psfd)
 L Latido de bloqueo de rama izquierda del haz de His
 R Latido de bloqueo de rama derecha del haz de His
 B Latido de bloqueo de rama (no especificado)
 A Latido auricular prematuro
 a Latido auricular prematuro aberrado
 J Latido prematuro nodal (juncional)
 S Latido supraventricular prematuro o ectópico (auricular o nodal)
 V Contracción ventricular prematura
 r Contracción ventricular prematura R- on-T contracción ventricular prematura
 F Fusión de latido ventricular y normal
 e Latido auricular de escape
 j Latido nodal (juncional) de escape
 n Latido supraventricular de escape (auricular o nodal)
 E Latido ventricular de escape
 / Latido estimulado
 f Fusión de latido estimulado y normal
 Q Latido inclasificable
 ? Latido no clasificado durante el aprendizaje
"""

from __future__ import division, print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suprime los mensajes de log de TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Deshabilita optimizaciones de OneDNN
from tqdm import tqdm # Para barras de progreso
import numpy as np
import random
import h5py # Para trabajar con archivos HDF5 (formato .keras)
from utils import * # Importa todas las funciones de utils.py (como mkdir_recursive)
from config import get_config # Importa la función para obtener la configuración

def preprocess( split ):
    """
    Preprocesa los datos de ECG del dataset MIT-BIH Arrhythmia.
    Carga los datos de PhysioNet, extrae segmentos alrededor de los picos QRS,
    y guarda los datos preprocesados en formato .keras.

    Args:
        split (bool): Si es True, divide los datos en conjuntos de entrenamiento y prueba.
                      Si es False, utiliza todo el conjunto como datos objetivo.
    """
    # Lista de números de registro del dataset MIT-BIH Arrhythmia a procesar.
    nums = ['100','101','102','103','104','105','106','107','108','109','111','112','113','114','115','116','117','118','119','121','122','123','124','200','201','202','203','205','207','208','209','210','212','213','214','215','217','219','220','221','222','223','228','230','231','232','233','234']
    # Derivaciones de ECG a extraer.
    features = ['MLII', 'V1', 'V2', 'V4', 'V5']

    if split :
        # Define los registros que se utilizarán para el conjunto de prueba.
        testset = ['101', '105','114','118', '124', '201', '210' , '217']
        # Los registros restantes se utilizan para el conjunto de entrenamiento.
        trainset = [x for x in nums if x not in testset]

    def dataSaver(dataSet, datasetname, labelsname):
        """
        Guarda los datos preprocesados y sus etiquetas en archivos HDF5 (.keras).

        Args:
            dataSet (list): Lista de números de registro a procesar.
            datasetname (str): Nombre del archivo donde se guardarán los datos.
            labelsname (str): Nombre del archivo donde se guardarán las etiquetas.
        """
        # Clases de latidos que el modelo va a clasificar.
        classes = ['N','V','/','A','F','~']
        Nclass = len(classes) # Número total de clases.
        datadict, datalabel= dict(), dict() # Diccionarios para almacenar los datos y las etiquetas.

        # Inicializa listas vacías para cada derivación en los diccionarios.
        for feature in features:
            datadict[feature] = list()
            datalabel[feature] = list()

        def dataprocess():
          """
          Procesa cada registro del dataset, extrayendo segmentos de señal y sus anotaciones.
          """
          config = get_config() # Obtiene la configuración para acceder a input_size.
          input_size = config.input_size # Tamaño del segmento de señal (número de muestras).

          for num in tqdm(dataSet): # Itera sobre cada número de registro con barra de progreso.
            from wfdb import rdrecord, rdann # Importa funciones de WFDB para leer registros y anotaciones.
            # Lee el registro de ECG, incluyendo las señales y sus metadatos.
            record = rdrecord('dataset/'+ num, smooth_frames= True)
            from sklearn import preprocessing # Importa preprocessing para escalado de datos.
            # Escala las señales de las dos primeras derivaciones (0 y 1) y convierte a lista.
            signals0 = preprocessing.scale(np.nan_to_num(record.p_signal[:,0])).tolist()
            signals1 = preprocessing.scale(np.nan_to_num(record.p_signal[:,1])).tolist()
            from scipy.signal import find_peaks # Importa find_peaks para detectar picos QRS.
            # Encuentra los picos en la primera señal, con una distancia mínima de 150 muestras.
            peaks, _ = find_peaks(signals0, distance=150)

            # Obtiene los nombres de las derivaciones (asume que las dos primeras son MLII y V1 o similares).
            feature0, feature1 = record.sig_name[0], record.sig_name[1]

            global lappened0, lappend1, dappend0, dappend1 # Declara variables globales para optimización.
            # Asigna métodos append de las listas para acceso más rápido.
            lappend0 = datalabel[feature0].append
            lappend1 = datalabel[feature1].append
            dappend0 = datadict[feature0].append
            dappend1 = datadict[feature1].append

            # Omite el primer y último pico para asegurar suficiente alcance de la muestra.
            for peak in tqdm(peaks[1:-1]):
              # Define el inicio y fin del segmento centrado en el pico.
              start, end =  peak-input_size//2 , peak+input_size//2
              # Lee las anotaciones dentro del segmento actual.
              ann = rdann('dataset/'+ num, extension='atr', sampfrom = start, sampto = end, return_label_elements=['symbol'])
              
              def to_dict(chosenSym):
                """
                Convierte un símbolo de anotación a un vector one-hot y añade el segmento
                de señal y la etiqueta al diccionario.
                """
                y = [0]*Nclass # Inicializa un vector one-hot de ceros.
                y[classes.index(chosenSym)] = 1 # Establece un 1 en la posición de la clase elegida.
                lappend0(y) # Añade la etiqueta a la lista de la primera derivación.
                lappend1(y) # Añade la etiqueta a la lista de la segunda derivación.
                dappend0(signals0[start:end]) # Añade el segmento de señal de la primera derivación.
                dappend1(signals1[start:end]) # Añade el segmento de señal de la segunda derivación.

              annSymbol = ann.symbol # Obtiene el símbolo de anotación.
              # Procesa solo si hay una única anotación y si está en las clases definidas,
              # y aplica un muestreo aleatorio para la clase "N" para equilibrar el dataset.
              if len(annSymbol) == 1 and (annSymbol[0] in classes) and (annSymbol[0] != "N" or np.random.random()<0.15):
                to_dict(annSymbol[0]) # Llama a to_dict para añadir los datos.

        print("Procesando datos...")
        dataprocess() # Inicia el procesamiento de los datos.

        noises = add_noise(config) # Añade segmentos de ruido al dataset (función de utils.py).
        
        # Concatena los datos y etiquetas de ruido a los datos existentes para las derivaciones MLII y V1.
        for feature in ["MLII", "V1"]:
            d = np.array(datadict[feature]) # Convierte los datos existentes a array numpy.
            if len(d) > 15*10**3: # Si el dataset es grande, usa el conjunto de entrenamiento de ruido.
                n = np.array(noises["trainset"])
            else: # De lo contrario, usa el conjunto de prueba de ruido.
                n = np.array(noises["testset"])
            datadict[feature]=np.concatenate((d,n)) # Concatena los datos de señal con el ruido.
            
            size, _  = n.shape # Obtiene el número de muestras de ruido.
            l = np.array(datalabel[feature]) # Convierte las etiquetas existentes a array numpy.
            noise_label = [0]*Nclass # Crea un vector one-hot para la etiqueta de ruido.
            noise_label[-1] = 1 # Asigna 1 a la última clase (Ruido).
            
            noise_label = np.array([noise_label] * size) # Crea un array de etiquetas de ruido del tamaño adecuado.
            datalabel[feature] = np.concatenate((l, noise_label)) # Concatena las etiquetas con las etiquetas de ruido.

        # Guarda los datos procesados y las etiquetas en archivos HDF5.
        with h5py.File(datasetname, 'w') as f:
            for key, data in datadict.items():
                f.create_dataset(key, data=data) # Crea un dataset para cada derivación.
        with h5py.File(labelsname, 'w') as f:
            for key, data in datalabel.items():
                f.create_dataset(key, data=data) # Crea un dataset para las etiquetas de cada derivación.

    # Llama a dataSaver con los conjuntos de datos apropiados según el valor de 'split'.
    if split:
        dataSaver(trainset, 'dataset/train.keras', 'dataset/trainlabel.keras') # Guarda datos de entrenamiento y sus etiquetas.
        dataSaver(testset, 'dataset/test.keras', 'dataset/testlabel.keras') # Guarda datos de prueba y sus etiquetas.
    else:
        dataSaver(nums, 'dataset/targetdata.keras', 'dataset/labeldata.keras') # Guarda todos los datos como objetivo.

def main(config):
    """
    Función principal para la ejecución del script de procesamiento de datos.

    Args:
        config (object): Objeto de configuración que incluye 'downloading' y 'split'.
    """
    def Downloadmitdb():
        """
        Descarga los archivos del dataset MIT-BIH Arrhythmia de PhysioNet.
        """
        ext = ['dat', 'hea', 'atr'] # Extensiones de los archivos a descargar.
        # Números de registro del dataset.
        nums = ['100','101','102','103','104','105','106','107','108','109','111','112','113','114','115','116','117','118','119','121','122','123','124','200','201','202','203','205','207','208','209','210','212','213','214','215','217','219','220','221','222','223','228','230','231','232','233','234']
        for num in tqdm(nums): # Itera sobre cada registro.
            for e in ext: # Itera sobre cada extensión.
                url = "https://physionet.org/physiobank/database/mitdb/"
                url = url + num +"."+e # Construye la URL completa del archivo.
                mkdir_recursive('dataset') # Asegura que el directorio 'dataset' exista.
                cmd = "cd dataset && curl -O "+url # Comando curl para descargar el archivo.
                os.system(cmd) # Ejecuta el comando del sistema.

    if config.downloading:
        Downloadmitdb() # Si la configuración indica descargar, llama a la función de descarga.
        #print("no descargar") # Comentado, se usaba para depuración.
    return preprocess(config.split) # Llama a la función de preprocesamiento con la configuración de división.

if __name__=="__main__":
    # Este bloque se ejecuta cuando el script se corre directamente.
    config = get_config() # Obtiene la configuración de los argumentos de línea de comandos.
    main(config) # Llama a la función principal.