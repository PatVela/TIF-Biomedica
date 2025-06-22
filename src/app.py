# coding=utf-8
from __future__ import division, print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suprime mensajes de log de TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Deshabilita optimizaciones de OneDNN

import numpy as np
import re # Importar el módulo re para expresiones regulares
import shutil # Importado para crear archivos ZIP
import pandas as pd # Importado para manejar datos y exportar a CSV
import wfdb # Importado para leer archivos WFDB
from scipy.io import loadmat # Importado para leer archivos .mat directamente

# Importa jsonify y send_from_directory desde Flask
from flask import Flask, redirect, url_for, request, render_template, jsonify, send_from_directory, send_file
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Importaciones locales desde otros módulos
from predict import predict_and_summarize # Importa predict_and_summarize
from utils import uploadedData, preprocess, mkdir_recursive, print_results, loaddata # Importa las funciones necesarias de utils
from config import get_config # Importa get_config para acceder a la configuración
from graph import ECG_model # Importa ECG_model para inicializar el modelo si no hay checkpoint

# --- Configuración para Determinismo/Reproducibilidad ---
# Es crucial para obtener resultados consistentes en algunas operaciones
SEED = 42
np.random.seed(SEED)
try:
    import tensorflow as tf
    tf.random.set_seed(SEED)
    # Algunas operaciones de GPU pueden seguir siendo no deterministas,
    # pero esto ayuda mucho en la mayoría de los casos.
    # tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
    # tf.config.experimental.enable_tensor_float_32_execution(False) # Puede afectar el rendimiento pero aumenta la precisión
except ImportError:
    print("TensorFlow no está instalado, no se pueden configurar las semillas de TF.")


app = Flask(__name__)
# Configura el tamaño máximo del contenido para la carga de archivos
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 # Aumentado a 10 MB para permitir múltiples archivos WFDB

# Configura una ruta para servir archivos estáticos (los gráficos generados)
app.config['RESULTS_FOLDER'] = 'resultados'
app.config['UPLOAD_FOLDER'] = 'uploads' # Carpeta para subidas temporales
app.config['ASSET_FOLDER'] = 'asset' # Carpeta para assets estáticos, incluyendo la matriz de confusión

# Asegúrate de que las carpetas existan al inicio de la aplicación
mkdir_recursive(app.config['RESULTS_FOLDER'])
mkdir_recursive(os.path.join(os.path.dirname(__file__), 'static', app.config['ASSET_FOLDER'])) # Para 'static/asset'
mkdir_recursive(app.config['UPLOAD_FOLDER'])


app.add_url_rule(
    f'/{app.config["RESULTS_FOLDER"]}/<path:filename>',
    endpoint='results_static',
    view_func=lambda filename: send_from_directory(app.config['RESULTS_FOLDER'], filename)
)

print('Abrir http://127.0.0.1:5002/')

def model_predict(img_path):
    """
    Realiza la predicción sobre un archivo de imagen subido.
    Genera el gráfico completo,
    y devuelve sus rutas y datos de predicción estructurados.

    Args:
        img_path (str): La ruta del archivo de imagen subido.

    Returns:
        dict: Un diccionario con los detalles de la predicción,
              incluyendo las rutas a los gráficos generados y
              las probabilidades estructuradas.
    """
    # Cargar datos desde el archivo de imagen
    # uploadedData ahora devuelve sampling_rate y los datos de la señal como un array 1D
    sampling_rate, data_signal_only = uploadedData(img_path, csvbool=True)

    # Asignar el sampling_rate a la configuración antes de preprocess
    config = get_config()
    config.sample_rate = sampling_rate # Asigna el sampling rate leído del CSV

    size = len(data_signal_only)
    if size > 9001: # El tamaño se basa ahora en la longitud del array 1D
        size = 9001
        data_signal_only = data_signal_only[:size]

    # Preprocesar los datos y encontrar los picos
    # preprocess ahora espera el sampling rate dentro del objeto config
    data_processed, peaks = preprocess(data_signal_only, config)

    # Extraer el nombre base del archivo subido para usarlo como nombre de carpeta del registro
    record_name_base = os.path.splitext(os.path.basename(img_path))[0]

    # Llamar a predict_and_summarize que ahora devuelve un diccionario completo de resumen.
    # Para uploadedData, no tenemos una 'label' real de antemano, pasamos None.
    prediction_summary_data = predict_and_summarize(data_processed, None, peaks, config, record_name_base)

    return prediction_summary_data

@app.route('/', methods=['GET'])
def index():
    """
    Renderiza la página principal de la aplicación y pasa la información del modelo.
    """
    config = get_config()

    # --- INICIO: Valores de métricas predefinidos o cargados una única vez ---
    accuracy_value = "0.925" # Valor de ejemplo, reemplazar con valor real si se carga de archivo
    f1_score_value = "0.887" # Valor de ejemplo, reemplazar con valor real si se carga de archivo
    # --- FIN: Valores de métricas predefinidos o cargados una única vez ---

    # Información estática del modelo extraída de los códigos
    model_info = {
        'architecture_type': 'Una **Red Neuronal Convolucional (CNN) profunda tipo ResNet** diseñada específicamente para la clasificación de latidos cardíacos. Se compone de 3 bloques principales: un bloque de entrada, un bloque de bucle (loop-block) ResNet y un bloque de salida.',
        'architecture_details': [
            '**Bloque de Entrada (First Conv Block):** Consiste en una capa convolucional 1D inicial que expande el número de filtros de 1 a 32. Esto permite al modelo aprender características iniciales de la señal de ECG.',
            '**Bloque de Bucle (Main Loop Block - 15 Bloques Residuales):** Esta es la parte central del modelo, donde se aplican 15 bloques residuales. Cada bloque contiene capas convolucionales (`Conv1D`), normalización por lotes (`BatchNormalization`), activación `ReLU` y una capa de abandono (`Dropout`).',
            '**Reducción de Dimensión y Expansión de Filtros en el Loop-Block:** Una de las capas convolucionales 1D dentro del `loop-block` reduce a la mitad el tamaño de la capa por cada dos bucles. Esto permite que la entrada se reduzca a 1/256 de su tamaño original durante las 15 iteraciones. Al mismo tiempo, el número de filtros se duplica cada 4 bucles (utilizando la función `zeropad` para igualar las dimensiones en las conexiones de atajo), pasando de 32 a 256 filtros.',
            '**Conexiones de Atajo (Skip Connections):** Inspirado en la arquitectura ResNet, el modelo utiliza conexiones de atajo. Estas conexiones permiten que la información "salte" algunas capas, lo que facilita el entrenamiento de redes muy profundas y ayuda a mitigar problemas como el gradiente desvanecedor.',
            '**Capa de Salida (Output Block):** Después del `loop-block`, la señal se aplanada (`Flatten`) y se pasa a una capa densa (`Dense`) con activación `softmax`. Esta capa final clasifica el latido en una de las seis categorías de salida.'
        ],
        'optimizer': '**Optimizador:** Adam (con una tasa de aprendizaje inicial de 0.1, con beta_1=0.9, beta_2=0.999, epsilon=1e-7 y amsgrad=False)',
        'loss_function': '**Función de Pérdida:** Categorical Crossentropy (para clasificación multiclase one-hot encoded)',
        'metrics_monitored': '**Métrica de Entrenamiento:** Accuracy',
        'dataset_source': 'MIT-BIH Arrhythmia Database (PhysioNet) complementado con datos de ruido del PhysioNet/Computing in Cardiology Challenge 2017 (CINC2017)',
        'dataset_total_records': '47 señales largas (MIT-BIH) con aproximadamente 650,000 muestras cada una, resultando en un total de 22,766 segmentos de 256 muestras después del procesamiento y balanceo.',
        'dataset_sampling_rate': '360 muestras por segundo (después del remuestreo de todas las fuentes)',
        'dataset_annotations': 'Múltiples etiquetas por señal en MIT-BIH, permitiendo la extracción de picos y el rebanado de muestras.',
        'dataset_lead': 'Principalmente MLII y V1 (V2, V4, V5 también disponibles pero con menos datos).',
        'dataset_balancing': 'Reducción aleatoria del 85% de las señales normales (N-labeled) del MIT-BIH y adición de ruido (~-labeled) del dataset CINC2017 para balancear las clases. Se eliminaron categorías minoritarias y las etiquetas L/R (bloqueo de rama) para enfocarse en 6 clases principales.',
        'model_classes': [
            {'code': 'N', 'description': 'Latido normal (9460 muestras)'},
            {'code': 'V', 'description': 'Contracción ventricular prematura (5951 muestras)'},
            {'code': '/', 'description': 'Latido a ritmo (marcapasos) (2074 muestras)'},
            {'code': 'A', 'description': 'Latido auricular prematuro (2092 muestras)'},
            {'code': 'F', 'description': 'Fusión de latido ventricular y normal (761 muestras)'},
            {'code': '~', 'description': 'Ruido (sin latido) (2428 muestras - provenientes de CINC2017)'}
        ],
        'metrics_accuracy': accuracy_value, # Se actualizará con el valor calculado
        'metrics_f1_score': f1_score_value, # Se actualizará con el valor calculado
        'performance_notes': [
            'Se lograron muy buenos F1-scores para las clases normal (N), ventricular (V), a ritmo (/) y ruido (~).',
            'Las métricas podrían mejorarse aún más al aumentar el número de datos para las etiquetas menos representadas y al realizar una sintonización de hiperparámetros más exhaustiva.',
            'El modelo puede discernir clasificaciones finas entre diferentes tipos de latidos, incluso con un tamaño de dataset más pequeño que otras investigaciones (aproximadamente 1/4 del tamaño del grupo de Stanford).'
        ],
        'preprocessing_notes': [
            'Los datos de ECG son remuestreados a 360 Hz (si es necesario) y escalados para tener media cero y varianza uno.',
            'La detección de picos QRS se realiza para segmentar la señal.',
            'A diferencia de otros enfoques, no se utilizan filtros de paso de banda o paso bajo explícitos, ya que las capas convolucionales del modelo están diseñadas para aprender automáticamente las configuraciones de filtro óptimas.'
        ],
        'prediction_notes': [
            'El modelo predice las probabilidades para cada segmento de 256 muestras.',
            'Aunque un solo segmento puede clasificarse como anómalo, la evaluación de la salud cardíaca puede requerir considerar el promedio de las probabilidades o la segunda etiqueta más probable si la probabilidad de "Normal" no es muy alta (ej. no supera el 85%).',
            'Se observó que la clase "O" (Otros) del dataset CINC2017 no se predice bien, posiblemente debido a la eliminación de categorías con pocos datos en el entrenamiento del modelo. Se evita mezclar directamente datos de diferentes fuentes (MIT-BIH, CINC2017, irhythm) debido a las variaciones en los dispositivos y la calidad de la señal, a menos que se realice un análisis de señal más detallado.'
        ]
    }

    # Procesar el texto del tipo de arquitectura
    model_info['architecture_type'] = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', model_info['architecture_type'])

    # Procesar las negritas en 'architecture_details'
    processed_architecture_details = []
    for detail_text in model_info['architecture_details']:
        # Utiliza una expresión regular para encontrar y reemplazar todos los **text** con <strong>text</strong>
        processed_text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', detail_text)
        processed_architecture_details.append(processed_text)
    model_info['architecture_details'] = processed_architecture_details

    # Procesar las negritas en el optimizador, función de pérdida y métrica de entrenamiento
    model_info['optimizer'] = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', model_info['optimizer'])
    model_info['loss_function'] = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', model_info['loss_function'])
    model_info['metrics_monitored'] = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', model_info['metrics_monitored'])


    return render_template('index.html', model_info=model_info)

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    """
    Maneja las solicitudes de subida de archivos para la predicción.
    """
    if request.method == 'POST':
        # Obtener el archivo de la solicitud POST
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        f = request.files['file']
        if f.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if f:
            # Guardar el archivo en ./uploads de forma segura
            basepath = os.path.dirname(__file__)
            upload_dir = os.path.join(basepath, app.config['UPLOAD_FOLDER']) # Usar la constante UPLOAD_FOLDER

            mkdir_recursive(upload_dir) # Asegurarse de que el directorio 'uploads' exista

            file_path = os.path.join(upload_dir, secure_filename(f.filename))
            f.save(file_path)

            # Realizar la predicción utilizando la función model_predict
            prediction_summary_data = model_predict(file_path)

            # Preparar los datos para la respuesta JSON
            predictions_for_json = []
            segment_plot_urls = []

            # Helper function to remove the base results folder prefix
            def get_relative_path_for_url_for(full_path, base_folder_name):
                if full_path and full_path.startswith(base_folder_name + os.sep):
                    return full_path[len(base_folder_name + os.sep):]
                return full_path

            # Las rutas de los plots ya son relativas a 'resultados/'
            # directamente desde predict.py. No necesitamos os.path.basename(os.path.dirname(...))
            # para construir la URL, solo usamos la ruta tal cual se devuelve.

            full_ecg_plot_url = url_for('results_static', filename=get_relative_path_for_url_for(prediction_summary_data['full_ecg_plot_path'], app.config['RESULTS_FOLDER'])) \
                                        if prediction_summary_data.get('full_ecg_plot_path') else None

            # Obtener la URL para el gráfico ECG_Lectura
            ecg_lectura_plot_url = url_for('results_static', filename=get_relative_path_for_url_for(prediction_summary_data['ecg_lectura_plot_path'], app.config['RESULTS_FOLDER'])) \
                                       if prediction_summary_data.get('ecg_lectura_plot_path') else None

            # Obtener la URL para el gráfico ECG_Lectura_Reducido (NUEVA LÍNEA)
            ecg_lectura_reducido_plot_url = url_for('results_static', filename=get_relative_path_for_url_for(prediction_summary_data['ecg_lectura_reducido_plot_path'], app.config['RESULTS_FOLDER'])) \
                                                 if prediction_summary_data.get('ecg_lectura_reducido_plot_path') else None

            # URLs para los segmentos y formateo de probabilidades
            for segment_data in prediction_summary_data['predictions_detailed']:
                plot_path_relative_to_results = segment_data.get('plot_path') # Esto ya es relativo a 'resultados/'
                if plot_path_relative_to_results:
                    segment_url = url_for('results_static', filename=get_relative_path_for_url_for(plot_path_relative_to_results, app.config['RESULTS_FOLDER']))
                    segment_plot_urls.append(segment_url)
                else:
                    segment_plot_urls.append(None) # O manejar como un error

                # Formatear el array completo de probabilidades a un string legible.
                # Se mostrarán como porcentajes con 2 decimales para mayor legibilidad.
                all_probs_formatted = [f"{p*100:.2f}%" for p in segment_data['all_probs_array']]

                predictions_for_json.append({
                    'class': segment_data['class'],
                    'probability': segment_data['probability'], # Probabilidad de la clase predicha (ej: "74.2%")
                    'all_probs_string': " | ".join(all_probs_formatted) # String de todas las probabilidades formateadas
                })

            # Crear el objeto JSON de respuesta
            response_data = {
                'total_parts': len(predictions_for_json),
                'predictions': predictions_for_json, # Lista de diccionarios con clase, prob de la clase, y todas las probs en string
                # 'summary' ahora viene directamente de prediction_summary_data
                'summary': {
                    'N': prediction_summary_data['summary']['N'] if 'N' in prediction_summary_data['summary'] else 0,
                    'Ventricular': prediction_summary_data['summary']['Ventricular'] if 'Ventricular' in prediction_summary_data['summary'] else 0,
                    'Paced': prediction_summary_data['summary']['Paced'] if 'Paced' in prediction_summary_data['summary'] else 0,
                    'A': prediction_summary_data['summary']['A'] if 'A' in prediction_summary_data['summary'] else 0,
                    'F': prediction_summary_data['summary']['F'] if 'F' in prediction_summary_data['summary'] else 0,
                    'Noise': prediction_summary_data['summary']['Noise'] if 'Noise' in prediction_summary_data['summary'] else 0,
                },
                'most_probable_class': prediction_summary_data.get('most_probable_class'),
                'most_probable_certainty': prediction_summary_data.get('most_probable_certainty'),
                'second_probable_class': prediction_summary_data.get('second_probable_class'),
                'second_probable_certainty': prediction_summary_data.get('second_probable_certainty'),
                'third_probable_class': prediction_summary_data.get('third_probable_class'),
                'third_probable_certainty': prediction_summary_data.get('third_probable_certainty'),
                'original_label': prediction_summary_data.get('original_label'),
                'average_probabilities': prediction_summary_data.get('average_probabilities'), # Array de probabilidades promedio
                'full_ecg_plot_url': full_ecg_plot_url,
                'ecg_lectura_plot_url': ecg_lectura_plot_url, # Añadido
                'ecg_lectura_reducido_plot_url': ecg_lectura_reducido_plot_url, # NUEVA LÍNEA: Añadido al JSON
                'segment_plot_urls': segment_plot_urls
            }

            # Eliminar el archivo después del análisis para limpiar el directorio de subidas
            try:
                os.remove(file_path)
            except OSError as e:
                print(f"Error al eliminar el archivo {file_path}: {e}")

            # Enviar la respuesta como JSON
            return jsonify(response_data)

    return None # Si el método no es POST o no se sube ningún archivo válido

@app.route('/download_ecg_results/<record_name_base>', methods=['GET'])
def download_ecg_results(record_name_base):
    """
    Comprime la carpeta de resultados de un registro específico en un archivo ZIP
    y lo envía para su descarga.
    """
    # Ruta a la carpeta de resultados específica para este registro
    record_results_folder_full_path = os.path.join(app.config['RESULTS_FOLDER'], record_name_base)

    if not os.path.exists(record_results_folder_full_path):
        return "Carpeta de resultados no encontrada.", 404

    # Nombre base para el archivo ZIP (sin extensión)
    zip_base_name = os.path.join(app.config['RESULTS_FOLDER'], f"{record_name_base}_Results")

    # Crear un archivo ZIP de la carpeta
    # shutil.make_archive(nombre_base_zip, 'formato_zip', directorio_fuente)
    # El archivo ZIP se creará como 'resultados/Muestras_ECG_ECG_Results.zip'
    zip_path = shutil.make_archive(zip_base_name, 'zip', record_results_folder_full_path)

    # Enviar el archivo ZIP para descarga
    try:
        return send_file(zip_path, as_attachment=True, download_name=f'{record_name_base}_Results.zip')
    except Exception as e:
        print(f"Error al enviar el archivo ZIP: {e}")
        return "Error al descargar el archivo.", 500
    finally:
        # Opcional: Eliminar el archivo ZIP temporal después de enviarlo
        # Ten cuidado con esto en entornos de producción, ya que puede haber problemas
        # si la descarga es muy grande o si el servidor necesita más tiempo.
        # Para desarrollo, es útil para limpiar.
        try:
            if os.path.exists(zip_path):
                os.remove(zip_path)
        except OSError as e:
            print(f"Error al eliminar archivos temporales: {e}")


@app.route('/format_ecg', methods=['POST'])
def format_ecg_to_csv():
    """
    Recibe archivos WFDB (.mat, .dat y .hea), los lee, extrae la derivación MLII (o la segunda),
    y convierte los datos a formato CSV. Luego envía el archivo resultante para su descarga.
    """
    uploaded_files = request.files.getlist('ecg_file_to_format')
    if not uploaded_files:
        return jsonify({'error': 'No se encontraron archivos en la solicitud.'}), 400

    basepath = os.path.dirname(__file__)
    upload_dir = os.path.join(basepath, app.config['UPLOAD_FOLDER'])
    mkdir_recursive(upload_dir)

    temp_dir_name = f"temp_wfdb_{os.urandom(8).hex()}"
    temp_wfdb_dir = os.path.join(upload_dir, temp_dir_name)
    mkdir_recursive(temp_wfdb_dir)

    file_base_name = None
    hea_file_path = None
    mat_file_path = None
    dat_file_path = None

    for uploaded_file in uploaded_files:
        if uploaded_file.filename == '':
            continue

        original_filename = secure_filename(uploaded_file.filename)
        file_extension = os.path.splitext(original_filename)[1].lower()
        current_file_base_name = os.path.splitext(original_filename)[0]

        if file_base_name is None:
            file_base_name = current_file_base_name
        elif file_base_name != current_file_base_name:
            shutil.rmtree(temp_wfdb_dir)
            return jsonify({'error': 'Todos los archivos WFDB deben tener el mismo nombre base (ej. record.mat y record.hea, o record.dat y record.hea).'}), 400

        temp_file_path = os.path.join(temp_wfdb_dir, original_filename)
        uploaded_file.save(temp_file_path)

        if file_extension == '.hea':
            hea_file_path = temp_file_path
        elif file_extension == '.mat':
            mat_file_path = temp_file_path
        elif file_extension == '.dat':
            dat_file_path = temp_file_path

    signal_data = None
    sampling_rate = None
    record_name_only = os.path.splitext(file_base_name)[0] # Base name without extension

    try:
        if hea_file_path and mat_file_path:
            # Lógica para archivos .mat y .hea
            data_mat = loadmat(mat_file_path)
            if 'val' in data_mat:
                signal_data = data_mat['val']
            else:
                for key in data_mat:
                    if not key.startswith('__') and isinstance(data_mat[key], np.ndarray):
                        signal_data = data_mat[key]
                        break
                if signal_data is None:
                    raise ValueError("No se encontraron datos de señal válidos en el archivo .mat (ni 'val' ni otras claves de datos).")

            # Intentar leer el encabezado para el sampling rate
            try:
                record_header = wfdb.rdheader(os.path.join(temp_wfdb_dir, record_name_only))
                sampling_rate = record_header.fs if hasattr(record_header, 'fs') else None
            except Exception as e_header:
                print(f"Advertencia: No se pudo leer el archivo .hea para el sampling rate (mat/hea): {e_header}. Usando valor por defecto.")

        elif hea_file_path and dat_file_path:
            # Lógica para archivos .dat y .hea usando wfdb.rdrecord
            record = wfdb.rdrecord(os.path.join(temp_wfdb_dir, record_name_only))
            signal_data = record.p_signal # Obtener la señal principal
            sampling_rate = record.fs if hasattr(record, 'fs') else None

            # Si signal_data es 2D (múltiples derivaciones), seleccionar MLII o la segunda
            if signal_data.ndim == 2:
                lead_index = 1 # Por defecto, la segunda derivación
                if hasattr(record, 'sig_name'):
                    signal_names = record.sig_name
                    try:
                        lead_index = [name.upper() for name in signal_names].index('MLII')
                        print(f"Derivación 'MLII' encontrada en el índice {lead_index}.")
                    except ValueError:
                        print("Advertencia: No se encontró la derivación 'MLII'. Usando la segunda derivación (índice 1) por defecto.")

                # Asegurarse de que el índice de la derivación sea válido
                if lead_index >= signal_data.shape[1]: # Comprobar contra el número de columnas (derivaciones)
                    print(f"Error: El índice de la derivación ({lead_index}) está fuera de los límites. Usando la primera derivación (índice 0).")
                    lead_index = 0
                selected_lead_data = signal_data[:, lead_index] # Seleccionar la derivación
            else:
                selected_lead_data = signal_data # Si ya es 1D, usar directamente

            signal_data = selected_lead_data # Usar la derivación seleccionada como signal_data

        else:
            shutil.rmtree(temp_wfdb_dir)
            return jsonify({'error': 'Se requieren un par de archivos WFDB (.hea con .mat o .hea con .dat) para el formateo.'}), 400

        if signal_data is None or signal_data.size == 0:
            raise ValueError("Los datos de señal no pudieron ser leídos o están vacíos.")

        # Aplanar los datos para asegurar que sea una sola columna
        signal_data_flat = signal_data.flatten()

        # Definir la ruta para el archivo CSV de salida
        csv_filename = f"{file_base_name}.csv"
        csv_file_path = os.path.join(upload_dir, csv_filename)

        # Escribir los datos en el archivo CSV, un valor por línea
        with open(csv_file_path, 'w', newline='') as f:
            if sampling_rate is not None:
                f.write(str(sampling_rate) + '\n')
            # Usar pandas para escribir la única columna sin encabezado ni índice
            pd.DataFrame(signal_data_flat).to_csv(f, index=False, header=False)

        # Enviar el archivo CSV para descarga
        return send_file(csv_file_path, as_attachment=True, download_name=csv_filename)

    except Exception as e:
        print(f"Error al formatear archivo WFDB a CSV: {e}")
        if os.path.exists(temp_wfdb_dir):
            shutil.rmtree(temp_wfdb_dir)
        return jsonify({'error': f'Error al procesar los archivos WFDB: {str(e)}. Asegúrate de que los archivos sean válidos y correspondan a un mismo registro.'}), 500
    finally:
        try:
            if os.path.exists(temp_wfdb_dir):
                shutil.rmtree(temp_wfdb_dir)
        except OSError as e:
            print(f"Error al eliminar archivos temporales: {e}")


if __name__ == '__main__':
    config = get_config()
    # Servir la aplicación con gevent
    http_server = WSGIServer(('0.0.0.0', 5002), app)
    http_server.serve_forever()