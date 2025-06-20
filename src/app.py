# coding=utf-8
from __future__ import division, print_function
import os
import numpy as np
import re

# Importa jsonify y send_from_directory desde Flask
from flask import Flask, redirect, url_for, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Importaciones locales desde otros módulos
from predict import predict_and_summarize # Importa predict_and_summarize
from utils import uploadedData, preprocess, mkdir_recursive # Importa las funciones necesarias de utils
from config import get_config # Importa get_config para acceder a la configuración

app = Flask(__name__)
# Configura el tamaño máximo del contenido para la carga de archivos
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 # 5 MB

# Configura una ruta para servir archivos estáticos (los gráficos generados)
app.config['RESULTS_FOLDER'] = 'resultados'
# Asegúrate de que la carpeta 'resultados' exista al inicio de la aplicación
mkdir_recursive(app.config['RESULTS_FOLDER'])

app.add_url_rule(
    f'/{app.config["RESULTS_FOLDER"]}/<path:filename>',
    endpoint='results_static',
    view_func=lambda filename: send_from_directory(app.config['RESULTS_FOLDER'], filename)
)

print('Abrir http://127.0.0.1:5002/')

def model_predict(img_path):
    """
    Realiza la predicción sobre un archivo de imagen subido.
    Genera el gráfico del ECG completo y los segmentos individuales,
    y devuelve sus rutas y datos de predicción estructurados.

    Args:
        img_path (str): La ruta del archivo de imagen subido.

    Returns:
        dict: Un diccionario con los detalles de la predicción,
              incluyendo las rutas a los gráficos generados y
              las probabilidades estructuradas.
    """
    # Cargar datos desde el archivo de imagen
    data = uploadedData(img_path, csvbool=True)
    
    # Asumimos que el primer valor puede no ser la frecuencia de muestreo, se maneja en preprocess
    if not isinstance(data[0], (int, float)):
        # Si la primera línea no es un número, la omitimos como si fuera un encabezado
        data = data[1:]
    
    size = len(data)
    if size > 9001:
        size = 9001
        data = data[:size]
    
    # Obtener la configuración del modelo
    config = get_config() 
    
    # Preprocesar los datos y encontrar los picos
    data, peaks = preprocess(data, config)

    # Extraer el nombre base del archivo subido para usarlo como nombre de carpeta del registro
    record_name_base = os.path.splitext(os.path.basename(img_path))[0]
    
    # Asegurarse de que el directorio de salida del registro exista
    # Esto se hará dentro de predict_and_summarize, pero lo mantenemos aquí para claridad si se separara
    # mkdir_recursive(os.path.join(app.config['RESULTS_FOLDER'], record_name_base))

    # Llamar a predict_and_summarize que ahora devuelve un diccionario completo de resumen
    # Para uploadedData, no tenemos una 'label' real de antemano, pasamos None.
    prediction_summary_data = predict_and_summarize(data, None, peaks, config, record_name_base)

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
            upload_dir = os.path.join(basepath, 'uploads')
            mkdir_recursive(upload_dir) # Asegurarse de que el directorio 'uploads' exista

            file_path = os.path.join(upload_dir, secure_filename(f.filename))
            f.save(file_path)

            # Realizar la predicción utilizando la función model_predict
            # model_predict ahora devuelve el diccionario completo de resumen.
            prediction_summary_data = model_predict(file_path)

            # Preparar los datos para la respuesta JSON
            predictions_for_json = []
            segment_plot_urls = []

            # Obtener el nombre de la subcarpeta del registro dentro de 'resultados'
            # Esto asume que full_ecg_plot_path contiene la ruta completa incluyendo el nombre del registro
            record_name_base_from_summary = os.path.basename(os.path.dirname(prediction_summary_data['full_ecg_plot_path']))

            # URL para el ECG completo
            full_ecg_plot_url = url_for('results_static', 
                                        filename=f"{record_name_base_from_summary}/{os.path.basename(prediction_summary_data['full_ecg_plot_path'])}") \
                                        if prediction_summary_data.get('full_ecg_plot_path') else None

            # URLs para los segmentos y formateo de probabilidades
            for segment_data in prediction_summary_data['predictions_detailed']:
                plot_path = segment_data.get('plot_path')
                if plot_path:
                    segment_url = url_for('results_static', 
                                          filename=f"{record_name_base_from_summary}/{os.path.basename(plot_path)}")
                    segment_plot_urls.append(segment_url)
                else:
                    segment_plot_urls.append(None) # O manejar como un error

                # Formatear el array completo de probabilidades a un string legible
                all_probs_formatted = [f"{p:.7e}%" for p in segment_data['all_probs_array']]
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

if __name__ == '__main__':
    config = get_config()
    # Servir la aplicación con gevent
    http_server = WSGIServer(('0.0.0.0', 5002), app)
    http_server.serve_forever()
