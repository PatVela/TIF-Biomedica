# coding=utf-8
from __future__ import division, print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suprime mensajes de log de TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Deshabilita optimizaciones de OneDNN

import numpy as np
import re
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
from utils import uploadedData, preprocess, mkdir_recursive # Importa las funciones necesarias de utils
from config import get_config # Importa get_config para acceder a la configuración

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

# Asegúrate de que las carpetas existan al inicio de la aplicación
mkdir_recursive(app.config['RESULTS_FOLDER'])
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
    Renderiza la página principal de la aplicación.
    """
    return render_template('index.html')

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

            # Las rutas de los plots ya son relativas a 'resultados/'
            # directamente desde predict.py. No necesitamos os.path.basename(os.path.dirname(...))
            # para construir la URL, solo usamos la ruta tal cual se devuelve.

            full_ecg_plot_url = url_for('results_static', filename=prediction_summary_data['full_ecg_plot_path']) \
                                        if prediction_summary_data.get('full_ecg_plot_path') else None


            # URLs para los segmentos y formateo de probabilidades
            for segment_data in prediction_summary_data['predictions_detailed']:
                plot_path_relative_to_results = segment_data.get('plot_path') # Esto ya es relativo a 'resultados/'
                if plot_path_relative_to_results:
                    segment_url = url_for('results_static', filename=plot_path_relative_to_results)
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
            print(f"Error al eliminar el archivo ZIP temporal {zip_path}: {e}")


@app.route('/format_ecg', methods=['POST'])
def format_ecg_to_csv():
    """
    Recibe archivos WFDB (.mat y .hea), los lee y convierte a formato CSV
    utilizando la lógica de carga de datos similar a predict.py.
    Luego envía el archivo CSV resultante para su descarga.
    """
    # Usamos request.files.getlist para obtener todos los archivos con el mismo nombre de campo
    uploaded_files = request.files.getlist('ecg_file_to_format')
    if not uploaded_files:
        return jsonify({'error': 'No se encontraron archivos en la solicitud.'}), 400

    basepath = os.path.dirname(__file__)
    upload_dir = os.path.join(basepath, app.config['UPLOAD_FOLDER'])
    mkdir_recursive(upload_dir)

    # Creamos una subcarpeta temporal para almacenar los archivos WFDB subidos
    temp_dir_name = f"temp_wfdb_{os.urandom(8).hex()}"
    temp_wfdb_dir = os.path.join(upload_dir, temp_dir_name)
    mkdir_recursive(temp_wfdb_dir)

    file_base_name = None
    hea_file_path = None
    mat_file_path = None

    # Guardar todos los archivos subidos en la carpeta temporal
    for uploaded_file in uploaded_files:
        if uploaded_file.filename == '':
            continue # Saltar archivos vacíos

        original_filename = secure_filename(uploaded_file.filename)
        file_extension = os.path.splitext(original_filename)[1].lower()
        current_file_base_name = os.path.splitext(original_filename)[0]

        # Necesitamos que todos los archivos compartan el mismo nombre base
        if file_base_name is None:
            file_base_name = current_file_base_name
        elif file_base_name != current_file_base_name:
            shutil.rmtree(temp_wfdb_dir) # Limpiar si los nombres base no coinciden
            return jsonify({'error': 'Todos los archivos WFDB deben tener el mismo nombre base (ej. record.mat y record.hea).'}), 400
        
        temp_file_path = os.path.join(temp_wfdb_dir, original_filename)
        uploaded_file.save(temp_file_path) # Guardar el archivo subido

        if file_extension == '.hea':
            hea_file_path = temp_file_path
        elif file_extension == '.mat':
            mat_file_path = temp_file_path

    if not hea_file_path or not mat_file_path:
        shutil.rmtree(temp_wfdb_dir) # Limpiar si no se encontraron ambos tipos
        return jsonify({'error': 'Se requieren tanto el archivo .hea como el .mat para el formateo WFDB.'}), 400

    csv_file_path = None
    try:
        # Cargar el archivo .mat usando scipy.io.loadmat
        data_mat = loadmat(mat_file_path)
        
        # Extraer los datos de la señal.
        signal_data = None
        if 'val' in data_mat:
            signal_data = data_mat['val']
        else:
            # Fallback: buscar otras claves que no sean metadatos
            for key in data_mat:
                if not key.startswith('__') and isinstance(data_mat[key], np.ndarray):
                    signal_data = data_mat[key]
                    break
            if signal_data is None:
                raise ValueError("No se encontraron datos de señal válidos en el archivo .mat (ni 'val' ni otras claves de datos).")

        if signal_data is None or signal_data.size == 0:
            raise ValueError("Los datos de señal no pudieron ser leídos o están vacíos en el archivo .mat.")

        # Leer el encabezado WFDB para obtener los nombres de las señales y la frecuencia de muestreo
        record_name_only = os.path.splitext(os.path.basename(hea_file_path))[0]
        record_dir_only = os.path.dirname(hea_file_path) 
        
        sampling_rate = None
        signal_names = [] # Inicializar para asegurar que siempre sea una lista
        try:
            record_header = wfdb.rdheader(os.path.join(record_dir_only, record_name_only))
            signal_names = record_header.sig_name if hasattr(record_header, 'sig_name') and record_header.sig_name else [] # Obtener nombres de señal
            sampling_rate = record_header.fs if hasattr(record_header, 'fs') else None
        except Exception as e_header:
            print(f"Advertencia: No se pudieron leer los nombres de las señales o la frecuencia de muestreo del archivo .hea. Error: {e_header}")
            # Si la lectura del encabezado falla, los nombres se inferirán más adelante.

        # --- Importante: Asegurar que signal_data tenga la forma (muestras, canales) para Pandas ---
        # Si signal_data es 2D y el número de filas coincide con el número de nombres de señal (indicando (canales, muestras))
        if signal_data.ndim == 2:
            n_rows, n_cols = signal_data.shape
            # Si el número de filas coincide con los nombres de señal Y el número de columnas es mucho mayor,
            # es probable que los datos estén en (canales, muestras)
            if len(signal_names) > 0 and len(signal_names) == n_rows and n_cols > n_rows:
                signal_data = signal_data.T # Transponer a (muestras, canales)
                # Si los nombres de las señales no se leyeron robustamente, actualizarlos si es necesario
                # (aunque la lógica de wfdb.rdheader debería ser robusta si hay nombres de señal)
                # Esto es más bien una doble verificación o un fallback.
                if not (hasattr(record_header, 'sig_name') and record_header.sig_name):
                    signal_names = [f'signal_{i+1}' for i in range(n_rows)] # Usar el conteo original de filas como conteo de canales

            # Si signal_names está vacío porque la lectura del encabezado falló, o si el conteo no coincide
            if not signal_names or len(signal_names) != signal_data.shape[1]:
                # Generar nombres de señal por defecto basados en el número de columnas actual
                signal_names = [f'signal_{i+1}' for i in range(signal_data.shape[1])]
        elif signal_data.ndim == 1:
            # Para una señal 1D, asegurarse de que sea (N, 1) y tener un nombre de señal
            signal_data = signal_data.reshape(-1, 1)
            signal_names = ['signal_1'] # Nombre por defecto para una señal única

        # Crear DataFrame
        df = pd.DataFrame(signal_data, columns=signal_names)

        # Definir la ruta para el archivo CSV de salida
        csv_filename = f"{file_base_name}_formatted.csv"
        csv_file_path = os.path.join(upload_dir, csv_filename) # Guarda el CSV en la carpeta principal de uploads

        # MANEJO PARA INCLUIR EL SAMPLING RATE COMO PRIMERA LÍNEA
        with open(csv_file_path, 'w', newline='') as f:
            if sampling_rate is not None:
                f.write(str(sampling_rate) + '\n') # Escribe el sampling rate en la primera línea
            # Escribe el DataFrame a partir de la segunda línea, incluyendo los encabezados de columna
            df.to_csv(f, index=False, header=True) # header=True para escribir los nombres de las señales como encabezados

        # Enviar el archivo CSV para descarga
        return send_file(csv_file_path, as_attachment=True, download_name=csv_filename)

    except Exception as e:
        print(f"Error al formatear archivo WFDB a CSV: {e}")
        # Asegúrate de limpiar el directorio temporal si ocurre un error
        if os.path.exists(temp_wfdb_dir):
            shutil.rmtree(temp_wfdb_dir)
        # Se envía un mensaje de error más detallado al frontend
        return jsonify({'error': f'Error al procesar los archivos WFDB: {str(e)}. Asegúrate de que ambos archivos (.hea y .mat) sean válidos y correspondan a un mismo registro.'}), 500
    finally:
        # Limpia los archivos temporales WFDB (la carpeta completa)
        try:
            if os.path.exists(temp_wfdb_dir):
                shutil.rmtree(temp_wfdb_dir) # Elimina la carpeta temporal y su contenido
            # Flask gestiona la eliminación del archivo CSV una vez que send_file completa la respuesta.
            # No es necesario eliminar csv_file_path aquí directamente.
        except OSError as e:
            print(f"Error al eliminar archivos temporales: {e}")


if __name__ == '__main__':
    config = get_config()
    # Servir la aplicación con gevent
    http_server = WSGIServer(('0.0.0.0', 5002), app)
    http_server.serve_forever()

