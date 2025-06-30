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
import json # Importar json para manejar archivos de metadatos

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
# Configura el tamaño máximo del contenido para la carga de archivos (en bytes)
# Aumentado a 100 MB, puedes ajustarlo según tus necesidades y recursos del servidor.
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024 # Permite archivos hasta 100 MB

# Define el número máximo de puntos de datos de la señal ECG que se procesarán.
# Este valor DEBE coincidir con lo que tu modelo puede manejar sin agotar la memoria
# o causar otros problemas de rendimiento. Ajusta este valor con precaución.
MAX_ECG_SIGNAL_POINTS = 18002
# Si el modelo puede manejar señales más largas, es posible aumentar este valor.
# Por ejemplo, para 30,000 puntos: MAX_ECG_SIGNAL_POINTS = 30000


# Configura una ruta para servir archivos estáticos (los gráficos generados)
app.config['RESULTS_FOLDER'] = 'resultados'
# Define UPLOAD_FOLDER relative to the current script's directory.
# This assumes app.py is located in the 'src' directory, and 'uploads' should be inside 'src'.
# So, os.path.join(os.path.dirname(__file__), 'uploads') will correctly point to 'ECG-Biomedica/src/uploads'.
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
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

print('Abrir http://localhost:5002/')

def model_predict(img_path, original_label=None, patient_name="Desconocido", patient_age="Desconocido", patient_sex="Desconocido", target_lead_index=1): # Añadidos parámetros de paciente
    """
    Realiza la predicción sobre un archivo de imagen subido.
    Genera el gráfico completo,
    y devuelve sus rutas y datos de predicción estructurados.

    Args:
        img_path (str): La ruta del archivo de imagen subido.
        original_label (str, optional): La etiqueta original del registro, si está disponible.
                                        Se pasa a predict_and_summarize. Por defecto, None.
        patient_name (str, optional): Nombre del paciente. Por defecto "Desconocido".
        patient_age (str, optional): Edad del paciente. Por defecto "Desconocido".
        patient_sex (str, optional): Sexo del paciente. Por defecto "Desconocido".
        target_lead_index (int, optional): Índice de la derivación a seleccionar para la predicción
                                            cuando hay múltiples derivaciones. Por defecto, 1 (segunda derivación).

    Returns:
        dict: Un diccionario con los detalles de la predicción,
              incluyendo las rutas a los gráficos generados y
              las probabilidades estructuradas.
    """
    # Cargar datos desde el archivo de imagen (en este contexto, un CSV)
    sampling_rate, data_signal_only = uploadedData(img_path, csvbool=True)

    print(f"DEBUG: img_path: {img_path}")
    print(f"DEBUG: data_signal_only shape: {data_signal_only.shape}")
    
    # Determinar qué derivación seleccionar
    selected_signal = None
    selected_lead_name = "Desconocida" # Para depuración
    if data_signal_only.ndim == 2:
        # Si es un DataFrame de pandas con columnas, intentar usar el nombre de columna 'MLII' o 'II'
        if hasattr(data_signal_only, 'columns'):
            print(f"DEBUG: Nombres de columnas en CSV completo: {data_signal_only.columns.tolist()}")
            if 'MLII' in data_signal_only.columns:
                selected_signal = data_signal_only['MLII'].values
                selected_lead_name = 'MLII'
                print(f"Información: Se detectaron múltiples derivaciones. Seleccionando la derivación 'MLII'.")
            elif 'II' in data_signal_only.columns:
                selected_signal = data_signal_only['II'].values
                selected_lead_name = 'II'
                print(f"Información: Se detectaron múltiples derivaciones. Seleccionando la derivación 'II'.")
            else:
                # Si no se encontraron MLII/II por nombre de columna, usar por índice
                if target_lead_index < data_signal_only.shape[1]:
                    selected_signal = data_signal_only[:, target_lead_index]
                    selected_lead_name = f'Índice {target_lead_index}'
                    print(f"Información: Se detectaron múltiples derivaciones. No se encontró 'MLII' o 'II' por nombre. Seleccionando la derivación en el índice {target_lead_index}.")
                else:
                    selected_signal = data_signal_only[:, 0]
                    selected_lead_name = f'Índice 0 (por defecto, índice objetivo fuera de límites)'
                    print(f"Advertencia: El índice de derivación objetivo ({target_lead_index}) está fuera de los límites. Seleccionando la primera derivación (índice 0) por defecto.")
        else: # Si es un numpy array 2D sin nombres de columna
            if target_lead_index < data_signal_only.shape[1]:
                selected_signal = data_signal_only[:, target_lead_index]
                selected_lead_name = f'Índice {target_lead_index}'
                print(f"Información: Se detectaron múltiples derivaciones (array NumPy). Seleccionando la derivación en el índice {target_lead_index}.")
            else:
                selected_signal = data_signal_only[:, 0]
                selected_lead_name = f'Índice 0 (por defecto, índice objetivo fuera de límites)'
                print(f"Advertencia: El índice de derivación objetivo ({target_lead_index}) está fuera de los límites. Seleccionando la primera derivación (índice 0) por defecto.")

    elif data_signal_only.ndim == 1:
        # Si ya es un array 1D (una sola derivación)
        selected_signal = data_signal_only
        selected_lead_name = 'Única derivación disponible'
        print(f"Información: Se detectó una sola derivación en el CSV. Usando la única derivación disponible.")
    else:
        # Manejar dimensiones de datos inesperadas (ej. 3D o vacío)
        print(f"Error: Dimensiones de la señal inesperadas: {data_signal_only.shape}. Intentando aplanar a 1D.")
        selected_signal = data_signal_only.flatten() # Asegurar que sea 1D para preprocess
        selected_lead_name = 'Aplanado (dimensión inesperada)'

    if selected_signal is None or selected_signal.size == 0:
        print("Error: No se pudo seleccionar una señal válida para preprocesamiento.")
        return {} # Retorna un diccionario vacío o maneja el error apropiadamente

    print(f"DEBUG: Derivación seleccionada para preprocesamiento: {selected_lead_name}")
    print(f"DEBUG: Shape final de selected_signal para preprocesamiento: {selected_signal.shape}")
    print(f"DEBUG: Primeros 10 puntos de selected_signal: {selected_signal[:10]}") # Imprimir primeros puntos para comparar

    # Asignar el sampling_rate a la configuración antes de preprocess
    config = get_config()
    config.sample_rate = sampling_rate # Asigna el sampling rate leído del CSV

    size = len(selected_signal) # Usar selected_signal para el tamaño y truncamiento
    if size > MAX_ECG_SIGNAL_POINTS: # Usar la constante definida globalmente
        print(f"Advertencia: La señal de ECG tiene {size} puntos, se truncará a {MAX_ECG_SIGNAL_POINTS} puntos para la predicción.")
        size = MAX_ECG_SIGNAL_POINTS
        selected_signal = selected_signal[:size]

    # Preprocesar los datos y encontrar los picos
    # preprocess ahora espera el sampling rate dentro del objeto config
    data_processed, peaks = preprocess(selected_signal, config) # Pasar selected_signal

    # Extraer el nombre base del archivo subido para usarlo como nombre de carpeta del registro
    record_name_base = os.path.splitext(os.path.basename(img_path))[0]

    # Llamar a predict_and_summarize que ahora devuelve un diccionario completo de resumen.
    # Usar el original_label pasado como parámetro. Si es None, predict_and_summarize lo manejará.
    prediction_summary_data = predict_and_summarize(data_processed, original_label, peaks, config, record_name_base)

    # Añadir información del paciente a los datos de resumen de predicción
    prediction_summary_data['patient_name'] = patient_name
    prediction_summary_data['patient_age'] = patient_age
    prediction_summary_data['patient_sex'] = patient_sex

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
        'architecture_type': 'Una **Red Neural Convolucional (CNN) profunda tipo ResNet** diseñada específicamente para la clasificación de latidos cardíacos. Se compone de 3 bloques principales: un bloque de entrada, un bloque de bucle (loop-block) ResNet y un bloque de salida.',
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
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        f = request.files['file']
        if f.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if f:
            mkdir_recursive(app.config['UPLOAD_FOLDER'])
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
            f.save(file_path)
            uploaded_filename_base = os.path.splitext(os.path.basename(file_path))[0]
            cleaned_filename_base_temp = re.sub(r'\(\d+\)$|_\d+$', '', uploaded_filename_base)
            json_base_name = cleaned_filename_base_temp
            if cleaned_filename_base_temp.endswith('_Completo'):
                json_base_name = cleaned_filename_base_temp.replace('_Completo', '')
            metadata_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{json_base_name}.json")
            extracted_original_label = "Desconocida"
            extracted_patient_name = "Desconocido"
            extracted_patient_age = "Desconocido"
            extracted_patient_sex = "Desconocido"
            if os.path.exists(metadata_file_path):
                print(f"DEBUG: Found metadata JSON at: {metadata_file_path}")
                try:
                    with open(metadata_file_path, 'r') as mf:
                        metadata = json.load(mf)
                        extracted_original_label = metadata.get("original_label", "Desconocida")
                        extracted_patient_name = metadata.get("patient_name", "Desconocido")
                        extracted_patient_age = metadata.get("patient_age", "Desconocido")
                        extracted_patient_sex = metadata.get("patient_sex", "Desconocido")
                    print(f"DEBUG: Metadata loaded - Label: {extracted_original_label}, Name: {extracted_patient_name}, Age: {extracted_patient_age}, Sex: {extracted_patient_sex}")
                except json.JSONDecodeError as e:
                    print(f"Error decodificando el JSON de metadatos para {json_base_name}.json: {e}")
            else:
                print(f"DEBUG: Metadata JSON NOT found at: {metadata_file_path}")
            # Lógica para asignar nombres predeterminados si es "Desconocido"
            if extracted_patient_name == "Desconocido":
                if extracted_patient_age != "Desconocido":
                    try:
                        age_int = int(extracted_patient_age)
                        if age_int < 18:
                            if extracted_patient_sex == 'M':
                                extracted_patient_name = "Johnie Doe"
                            elif extracted_patient_sex == 'F':
                                extracted_patient_name = "Janie Doe"
                            else:
                                extracted_patient_name = "Baby Doe"
                        else:
                            if extracted_patient_sex == 'M':
                                extracted_patient_name = "John Doe"
                            elif extracted_patient_sex == 'F':
                                extracted_patient_name = "Jane Doe"
                            else:
                                extracted_patient_name = "A. N. Other"
                    except ValueError:
                        if extracted_patient_sex == 'M':
                            extracted_patient_name = "John Doe"
                        elif extracted_patient_sex == 'F':
                            extracted_patient_name = "Jane Doe"
                        else:
                            extracted_patient_name = "A. N. Other"
                else:
                    if extracted_patient_sex == 'M':
                        extracted_patient_name = "John Doe"
                    elif extracted_patient_sex == 'F':
                        extracted_patient_name = "Jane Doe"
                    else:
                        extracted_patient_name = "A. N. Other"
            print(f"DEBUG final patient info before model_predict: Name={extracted_patient_name}, Age={extracted_patient_age}, Sex={extracted_patient_sex}")
            prediction_summary_data = model_predict(
                file_path,
                original_label=extracted_original_label,
                patient_name=extracted_patient_name,
                patient_age=extracted_patient_age,
                patient_sex=extracted_patient_sex
            )
            predictions_for_json = []
            segment_plot_urls = []
            def get_relative_path_for_url_for(full_path, base_folder_name):
                if full_path and full_path.startswith(base_folder_name + os.sep):
                    return full_path[len(base_folder_name + os.sep):]
                return full_path
            full_ecg_plot_url = url_for('results_static', filename=get_relative_path_for_url_for(prediction_summary_data['full_ecg_plot_path'], app.config['RESULTS_FOLDER'])) \
                                        if prediction_summary_data.get('full_ecg_plot_path') else None
            ecg_lectura_plot_url = url_for('results_static', filename=get_relative_path_for_url_for(prediction_summary_data['ecg_lectura_plot_path'], app.config['RESULTS_FOLDER'])) \
                                       if prediction_summary_data.get('ecg_lectura_plot_path') else None
            ecg_lectura_reducido_plot_url = url_for('results_static', filename=get_relative_path_for_url_for(prediction_summary_data['ecg_lectura_reducido_plot_path'], app.config['RESULTS_FOLDER'])) \
                                                 if prediction_summary_data.get('ecg_lectura_reducido_plot_path') else None
            for segment_data in prediction_summary_data['predictions_detailed']:
                plot_path_relative_to_results = segment_data.get('plot_path')
                if plot_path_relative_to_results:
                    segment_url = url_for('results_static', filename=get_relative_path_for_url_for(plot_path_relative_to_results, app.config['RESULTS_FOLDER']))
                    segment_plot_urls.append(segment_url)
                else:
                    segment_plot_urls.append(None)
                all_probs_formatted = [f"{p*100:.2f}%" for p in segment_data['all_probs_array']]
                predictions_for_json.append({
                    'class': segment_data['class'],
                    'probability': segment_data['probability'],
                    'all_probs_string': " | ".join(all_probs_formatted)
                })
            response_data = {
                'total_parts': len(predictions_for_json),
                'predictions': predictions_for_json,
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
                'average_probabilities': prediction_summary_data.get('average_probabilities'),
                'full_ecg_plot_url': full_ecg_plot_url,
                'ecg_lectura_plot_url': ecg_lectura_plot_url,
                'ecg_lectura_reducido_plot_url': ecg_lectura_reducido_plot_url,
                'segment_plot_urls': segment_plot_urls,
                'cardiac_condition_suggestion': prediction_summary_data.get('cardiac_condition_suggestion'),
                'patient_name': prediction_summary_data.get('patient_name'),
                'patient_age': prediction_summary_data.get('patient_age'),
                'patient_sex': prediction_summary_data.get('patient_sex')
            }
            try:
                os.remove(file_path)
            except OSError as e:
                print(f"Error al eliminar el archivo {file_path}: {e}")
            return jsonify(response_data)
    return None

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
        return send_file(zip_path, as_attachment=True, download_name=f'{record_name_base}_Resultados.zip')
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
    También extrae y guarda la etiqueta original del registro en un archivo JSON.
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

    csv_file_path = None
    metadata_file_path = None

    try:
        # Validación y guardado de archivos
        file_base_name, hea_file_found, mat_file_found, dat_file_found = _validate_and_prepare_wfdb_files(uploaded_files, temp_wfdb_dir)
        if not hea_file_found or (not mat_file_found and not dat_file_found):
            shutil.rmtree(temp_wfdb_dir)
            return jsonify({'error': 'Para formatear un ECG, se requiere un par de archivos: un archivo .hea y un archivo de datos WFDB (.mat o .dat) con el mismo nombre base.'}), 400

        record_name_only = os.path.splitext(file_base_name)[0]
        record = wfdb.rdrecord(os.path.join(temp_wfdb_dir, record_name_only))
        signal_data = record.p_signal
        sampling_rate = record.fs if hasattr(record, 'fs') else None

        # Extraer metadatos del paciente
        original_label, patient_name, patient_age, patient_sex = _extract_patient_metadata(record)
        print(f"DEBUG extracted patient info: Age={patient_age}, Sex={patient_sex}, Name={patient_name}")

        selected_lead_name_for_save = "N/A (ya es 1D)"
        if signal_data.ndim == 2:
            print(f"DEBUG: Nombres de derivación en el archivo WFDB para guardar CSV simple: {record.sig_name if hasattr(record, 'sig_name') else 'No disponibles'}")
            lead_index = 1
            if hasattr(record, 'sig_name'):
                signal_names = record.sig_name
                try:
                    lead_index = [name.upper() for name in signal_names].index('MLII')
                    print(f"Derivación 'MLII' encontrada en el índice {lead_index} para guardar CSV simple.")
                except ValueError:
                    print("Advertencia: No se encontró la derivación 'MLII' para guardar CSV simple. Usando la segunda derivación (índice 1) por defecto.")
            if lead_index >= signal_data.shape[1]:
                print(f"Error: El índice de la derivación ({lead_index}) está fuera de los límites para guardar CSV simple. Usando la primera derivación (índice 0).")
                lead_index = 0
            selected_lead_data = signal_data[:, lead_index]
            selected_lead_name_for_save = record.sig_name[lead_index] if hasattr(record, 'sig_name') else f'Índice {lead_index}'
        else:
            selected_lead_data = signal_data
        signal_data = selected_lead_data
        print(f"DEBUG: Derivación seleccionada para guardar en CSV simple: {selected_lead_name_for_save}")
        print(f"DEBUG: Primeros 10 puntos de la señal guardada en CSV simple: {signal_data[:10]}")

        if signal_data is None or signal_data.size == 0:
            raise ValueError("Los datos de señal no pudieron ser leídos o están vacíos.")
        signal_data_flat = signal_data.flatten()
        csv_filename = f"{file_base_name}.csv"
        csv_file_path = os.path.join(upload_dir, csv_filename)
        with open(csv_file_path, 'w', newline='') as f:
            if sampling_rate is not None:
                f.write(f"# Sampling Rate: {sampling_rate}\n")
            pd.DataFrame(signal_data_flat).to_csv(f, index=False, header=False)
        metadata_filename = f"{file_base_name}.json"
        metadata_file_path = os.path.join(upload_dir, metadata_filename)
        _save_metadata_json(metadata_file_path, original_label, patient_name, patient_age, patient_sex)
        return send_file(csv_file_path, as_attachment=True, download_name=csv_filename)
    except Exception as e:
        print(f"Error al formatear archivo WFDB a CSV: {e}")
        _cleanup_temp_files(temp_wfdb_dir, csv_file_path)
        return jsonify({'error': f'Error al procesar los archivos WFDB: {str(e)}. Asegúrate de que los archivos sean válidos y correspondan a un mismo registro.'}), 500
    finally:
        _cleanup_temp_files(temp_wfdb_dir, csv_file_path)
@app.route('/format_ecg_full', methods=['POST'])
def format_ecg_full_to_csv():
    """
    Recibe archivos WFDB (.mat, .dat y .hea), los lee,
    y convierte TODAS las derivaciones disponibles a formato CSV,
    cada derivación en una columna separada.
    Luego envía el archivo resultante para su descarga.
    También extrae y guarda la etiqueta original del registro en un archivo JSON.
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

    csv_file_path = None
    metadata_file_path = None

    try:
        # Validación y guardado de archivos
        file_base_name, hea_file_found, mat_file_found, dat_file_found = _validate_and_prepare_wfdb_files(uploaded_files, temp_wfdb_dir)
        if not hea_file_found or (not mat_file_found and not dat_file_found):
            shutil.rmtree(temp_wfdb_dir)
            return jsonify({'error': 'Para formatear un ECG, se requiere un par de archivos: un archivo .hea y un archivo de datos WFDB (.mat o .dat) con el mismo nombre base.'}), 400

        record_name_only = os.path.splitext(file_base_name)[0]
        record = wfdb.rdrecord(os.path.join(temp_wfdb_dir, record_name_only))
        signal_data = record.p_signal
        sampling_rate = record.fs if hasattr(record, 'fs') else None
        signal_names = record.sig_name if hasattr(record, 'sig_name') else None

        # Extraer metadatos del paciente
        original_label, patient_name, patient_age, patient_sex = _extract_patient_metadata(record)
        print(f"DEBUG extracted patient info: Age={patient_age}, Sex={patient_sex}, Name={patient_name}")

        if signal_data is None:
            raise ValueError("No se encontraron datos de señal válidos en el archivo .dat/.hea.")
        df = pd.DataFrame(signal_data)
        if signal_names: # Asignar nombres de columna si se obtuvieron
            df.columns = signal_names
        else: # Si signal_names es None, asignar nombres predeterminados
            df.columns = [f'ECG_Lead_{i+1}' for i in range(df.shape[1])]


        # Escribir los datos en el archivo CSV
        csv_filename = f"{file_base_name}_Completo.csv"
        csv_file_path = os.path.join(upload_dir, csv_filename)

        with open(csv_file_path, 'w', newline='') as f:
            if sampling_rate is not None:
                # Escribir la frecuencia de muestreo como la primera línea, precedida de un '#'
                f.write(f"# Sampling Rate: {sampling_rate}\n")
            df.to_csv(f, index=False) # Escribir el DataFrame a CSV con encabezados

        # Guardar metadatos (incluyendo original_label, nombre, edad, sexo) en un archivo JSON
        metadata_filename = f"{file_base_name}.json"
        metadata_file_path = os.path.join(upload_dir, metadata_filename)
        _save_metadata_json(metadata_file_path, original_label, patient_name, patient_age, patient_sex)

        # Enviar el archivo CSV para descarga
        return send_file(csv_file_path, as_attachment=True, download_name=csv_filename)

    except Exception as e:
        print(f"Error al formatear archivo WFDB a CSV completo: {e}")
        if os.path.exists(temp_wfdb_dir):
            shutil.rmtree(temp_wfdb_dir)
        return jsonify({'error': f'Error al procesar los archivos WFDB para formateo completo: {str(e)}. Asegúrate de que los archivos sean válidos y correspondan a un mismo registro.'}), 500
    finally:
        # Usar la función de limpieza auxiliar para eliminar temporales
        _cleanup_temp_files(temp_wfdb_dir, csv_file_path)

def _extract_patient_metadata(record):
    """
    Extrae metadatos del paciente (diagnóstico, nombre, edad, sexo) de un registro WFDB.
    Devuelve: original_label, patient_name, patient_age, patient_sex
    """
    original_label = "Desconocida"
    patient_name = "Desconocido"
    patient_age = "Desconocido"
    patient_sex = "Desconocido"
    if hasattr(record, 'comments') and record.comments:
        for comment in record.comments:
            if "Diagnosis:" in comment:
                extracted_diagnosis = comment.split("Diagnosis:", 1)[1].strip()
                if extracted_diagnosis:
                    original_label = extracted_diagnosis
            age_match = re.search(r'Age:\s*(\d+)', comment)
            if age_match:
                patient_age = age_match.group(1)
            sex_match = re.search(r'Sex:\s*(.+)', comment)
            if sex_match:
                patient_sex_raw = sex_match.group(1).strip().lower()
                if patient_sex_raw == 'male':
                    patient_sex = 'M'
                elif patient_sex_raw == 'female':
                    patient_sex = 'F'
                else:
                    patient_sex = "Desconocido"
            name_match = re.search(r'Name:\s*(.+)', comment)
            if name_match:
                patient_name = name_match.group(1).strip()
            if patient_name == "Desconocido":
                if hasattr(record, 'description') and record.description and "patient" in record.description.lower():
                    name_from_desc = re.search(r'([A-Za-z\s]+\s[A-Za-z\s]+)', record.description)
                    if name_from_desc:
                        patient_name = name_from_desc.group(1).strip()
                    else:
                        patient_name = record.description.strip()
                elif comment and "patient" in comment.lower():
                    name_from_comment = re.search(r'([A-Za-z\s]+\s[A-Za-z\s]+)', comment)
                    if name_from_comment:
                        patient_name = name_from_comment.group(1).strip()
                    else:
                        patient_name = "Desconocido" # Mantener como desconocido si no se puede extraer
    # Lógica para asignar nombres predeterminados si es "Desconocido"
    if patient_name == "Desconocido":
        if patient_age != "Desconocido":
            try:
                age_int = int(patient_age)
                if age_int < 18:
                    if patient_sex == 'M':
                        patient_name = "Johnie Doe"
                    elif patient_sex == 'F':
                        patient_name = "Janie Doe"
                    else:
                        patient_name = "Baby Doe"
                else:
                    if patient_sex == 'M':
                        patient_name = "John Doe"
                    elif patient_sex == 'F':
                        patient_name = "Jane Doe"
                    else:
                        patient_name = "A. N. Other"
            except ValueError:
                if patient_sex == 'M':
                    patient_name = "John Doe"
                elif patient_sex == 'F':
                    patient_name = "Jane Doe"
                else:
                    patient_name = "A. N. Other"
        else:
            if patient_sex == 'M':
                patient_name = "John Doe"
            elif patient_sex == 'F':
                patient_name = "Jane Doe"
            else:
                patient_name = "A. N. Other"
    return original_label, patient_name, patient_age, patient_sex

def _validate_and_prepare_wfdb_files(uploaded_files, temp_wfdb_dir):
    """
    Valida y guarda archivos WFDB subidos en un directorio temporal.
    Devuelve: file_base_name, hea_file_found, mat_file_found, dat_file_found
    """
    file_base_name = None
    hea_file_found = False
    mat_file_found = False
    dat_file_found = False
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
            raise ValueError('Todos los archivos WFDB deben tener el mismo nombre base (ej. record.mat y record.hea, o record.dat y record.hea).')
        temp_file_path = os.path.join(temp_wfdb_dir, original_filename)
        uploaded_file.save(temp_file_path)
        if file_extension == '.hea':
            hea_file_found = True
        elif file_extension == '.mat':
            mat_file_found = True
        elif file_extension == '.dat':
            dat_file_found = True
    return file_base_name, hea_file_found, mat_file_found, dat_file_found

def _save_metadata_json(metadata_path, original_label, patient_name, patient_age, patient_sex):
    """
    Guarda los metadatos del paciente en un archivo JSON.
    """
    with open(metadata_path, 'w') as f_meta:
        json.dump({
            "original_label": original_label,
            "patient_name": patient_name,
            "patient_age": patient_age,
            "patient_sex": patient_sex
        }, f_meta)

def _cleanup_temp_files(temp_wfdb_dir, csv_file_path=None):
    """
    Elimina directorios y archivos temporales generados durante el formateo.
    """
    try:
        if os.path.exists(temp_wfdb_dir):
            shutil.rmtree(temp_wfdb_dir)
        if csv_file_path and os.path.exists(csv_file_path):
            os.remove(csv_file_path)
    except OSError as e:
        print(f"Error al eliminar archivos temporales: {e}")

if __name__ == '__main__':
    config = get_config()
    # Servir la aplicación con gevent
    http_server = WSGIServer(('0.0.0.0', 5002), app)
    http_server.serve_forever()