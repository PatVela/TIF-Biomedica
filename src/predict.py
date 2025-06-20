"""
Módulo para la predicción de datos de ECG utilizando un modelo entrenado.
Incluye funciones para la descarga de datos CINC, preprocesamiento y predicción.

Datos CINC provienen de https://physionet.org/challenge/2017/
"""

from __future__ import division, print_function
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from scipy.io import loadmat
from keras.models import load_model
import sys # Importar sys para la barra de carga
import time # Importar time para medir el tiempo por paso

# Importaciones locales desde otros módulos
from config import get_config
from utils import preprocess, uploadedData, mkdir_recursive
from graph import zeropad, zeropad_output_shape # Importa las funciones custom_objects para cargar el modelo

# Definiciones de colores ANSI para la salida de la consola
class Colors:
    HEADER = '\033[95m'  # Magenta
    BLUE = '\033[94m'    # Azul brillante
    CYAN = '\033[96m'    # Cian brillante
    GREEN = '\033[92m'   # Verde brillante
    WARNING = '\033[93m' # Amarillo brillante
    FAIL = '\033[91m'    # Rojo brillante
    ENDC = '\033[0m'     # Resetear color
    BOLD = '\033[1m'     # Negrita
    UNDERLINE = '\033[4m'# Subrayado
    WHITE = '\033[97m'   # Blanco brillante (para la barra de carga)
    DARK_GRAY = '\033[90m' # Gris oscuro para el fondo de la barra

def cincData(config):
    """
    Descarga y carga datos del desafío CINC 2017.

    Args:
        config (object): Objeto de configuración con atributos como cinc_download y num.

    Returns:
        tuple: Una tupla que contiene los datos de la señal, la etiqueta de clasificación
               y el ID del registro.
    """
    if config.cinc_download:
        print(f"{Colors.BLUE}Descargando datos del desafío CINC 2017...{Colors.ENDC}")
        cmd = "curl -O https://archive.physionet.org/challenge/2017/training2017.zip"
        os.system(cmd)
        os.system("unzip training2017.zip")
        print(f"{Colors.BLUE}Descarga y descompresión completadas.{Colors.ENDC}")

    testlabel = []
    try:
        with open('training2017/REFERENCE.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                testlabel.append([row[0], row[1]])
                line_count += 1
            print(f'{Colors.CYAN}Líneas {line_count} procesadas del archivo REFERENCE.csv.{Colors.ENDC}')
    except FileNotFoundError:
        print(f"{Colors.FAIL}Error: El archivo REFERENCE.csv no se encontró. Asegúrate de que los datos CINC estén descargados.{Colors.ENDC}")
        return None, None, None # Retorna None para el record_id también

    num = config.num
    if num is None:
        # Si no se especifica un número, elige uno aleatoriamente
        high = len(testlabel) - 1
        num = np.random.randint(1, high)

    # MODIFICACIÓN: Separar el record_id y la etiqueta real
    record_id_from_csv, actual_label = testlabel[num - 1]
    
    file_path_mat = 'training2017/' + record_id_from_csv + '.mat'

    print(f"{Colors.BLUE}Cargando el registro: {file_path_mat}{Colors.ENDC}")
    try:
        data_mat = loadmat(file_path_mat)
    except FileNotFoundError:
        print(f"{Colors.FAIL}Error: El archivo .mat {file_path_mat} no se encontró. Asegúrate de que los datos CINC estén descargados.{Colors.ENDC}")
        return None, None, None # Retorna None para el record_id también

    if not config.upload:
        # Asume que los datos están en la clave 'val' para archivos .mat de CINC
        data = data_mat['val']
        _, size = data.shape
        data = data.reshape(size,) # Remodelar a un array 1D
    else:
        # Para datos subidos, asume que ya están en el formato correcto
        data = np.array(data_mat) # Convertir a array de numpy si no lo es

    return data, actual_label, record_id_from_csv # Retorna también el record_id

def predict_and_summarize(data, label, peaks, config, record_name_base):
    """
    Realiza predicciones sobre los datos y genera un resumen detallado.
    También genera gráficos de la señal ECG completa y de cada segmento de registro.

    Args:
        data (np.array): Datos de la señal preprocesados.
        label (str): Etiqueta original del registro (si aplica).
        peaks (np.array): Picos detectados en la señal.
        config (object): Objeto de configuración.
        record_name_base (str): Nombre base del registro para la creación de directorios.

    Returns:
        dict: Un diccionario que contiene el resumen de las predicciones, incluyendo
              la lista de predicciones detalladas, la clase más probable,
              la segunda clase más probable y sus certezas, la tercera clase y certeza,
              la etiqueta original, las probabilidades promedio y las rutas a los gráficos.
    """
    # Crear el directorio para los resultados de este registro
    record_output_dir = f'resultados/{record_name_base}'
    mkdir_recursive(record_output_dir)

    # Generar y guardar el gráfico de la señal ECG completa
    full_ecg_plot_filename = f'full_ecg_signal_{record_name_base}.png'
    full_ecg_plot_path = os.path.join(record_output_dir, full_ecg_plot_filename)
    plt.figure(figsize=(15, 6))
    plt.plot(data[0, :, 0]) # Asumiendo data es (1, length, 1)
    plt.title(f"Señal ECG Completa para Registro: {record_name_base}")
    plt.xlabel("Muestras")
    plt.ylabel("Amplitud normalizada")
    plt.grid(True)
    plt.savefig(full_ecg_plot_path, format="png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{Colors.BLUE}Señal ECG completa guardada en: {full_ecg_plot_path}{Colors.ENDC}")

    # Realiza la predicción por partes y obtiene la lista de predicciones y la cadena de resultados
    # MODIFICACIÓN: predictByPart ahora devolverá también las rutas de los gráficos de segmentos
    predicted_list_raw, result_string, segment_plot_paths = predictByPart(data, peaks, record_output_dir, record_name_base)

    # Las clases mapeadas para una salida más legible
    classes_map = ['N','Ventricular','Paced','A','F','Noise'] # Mapeo de índices a nombres de clase

    # Initialize a default summary dictionary
    summary_counts = {cls: 0 for cls in classes_map}
    
    # Calculate average probabilities
    all_segment_probs = [item['all_probs_array'] for item in predicted_list_raw if 'all_probs_array' in item]
    if all_segment_probs:
        avg_predict = np.mean(all_segment_probs, axis=0)
    else:
        avg_predict = np.array([0.0] * len(classes_map)) # Default to zeros if no probabilities found

    # If predictions were made, update summary_counts from the result_string
    if predicted_list_raw:
        # Parse the result_string to get the counts for the summary
        # Assuming result_string ends with "Count1-Class1,Count2-Class2,..."
        summary_str_part = result_string[result_string.rfind(')') + 1:].strip()
        summary_parts = summary_str_part.split(',')
        for part in summary_parts:
            clean_part = part.strip()
            if '-' in clean_part:
                try:
                    count, name = clean_part.split('-', 1)
                    # Map the name back to the correct key in summary_counts
                    # The classes_map is used for keys, e.g., 'N', 'Ventricular'
                    # The CLI output string has 'N', 'Ventricular', 'Paced', etc.
                    # Ensure the key matches classes_map
                    if name.strip() == 'N': summary_counts['N'] = int(count.strip())
                    elif name.strip() == 'Ventricular': summary_counts['Ventricular'] = int(count.strip())
                    elif name.strip() == 'Paced': summary_counts['Paced'] = int(count.strip())
                    elif name.strip() == 'A': summary_counts['A'] = int(count.strip())
                    elif name.strip() == 'F': summary_counts['F'] = int(count.strip())
                    elif name.strip() == 'Noise': summary_counts['Noise'] = int(count.strip())
                except ValueError:
                    print(f"Warning: Could not parse count or name from summary part: {clean_part}")


    # Order indices of probabilities from highest to lowest
    sorted_indices = avg_predict.argsort()[::-1]

    # Most probable class
    most_probable_idx = sorted_indices[0]
    most_probable_class = classes_map[most_probable_idx]
    most_probable_certainty = 100 * avg_predict[most_probable_idx]

    # Second most probable class
    second_probable_class = None
    second_probable_certainty = 0.0
    if len(avg_predict) > 1:
        sec_idx = sorted_indices[1]
        second_probable_class = classes_map[sec_idx]
        second_probable_certainty = 100 * avg_predict[sec_idx]

    # Third most probable class
    third_probable_class = None
    third_probable_certainty = 0.0
    if len(avg_predict) > 2:
        third_idx = sorted_indices[2]
        third_probable_class = classes_map[third_idx]
        third_probable_certainty = 100 * avg_predict[third_idx]

    # Build the results dictionary
    summary_results = {
        'predictions_detailed': predicted_list_raw, # Now contains the full probability array and plot path
        'prediction_summary_string': result_string, # This is the CLI output string (maintained for compatibility)
        'most_probable_class': most_probable_class,
        'most_probable_certainty': most_probable_certainty,
        'second_probable_class': second_probable_class,
        'second_probable_certainty': second_probable_certainty,
        'third_probable_class': third_probable_class,
        'third_probable_certainty': third_probable_certainty,
        'original_label': label,
        'average_probabilities': avg_predict.tolist(), # Convert to list for serialization if necessary
        'full_ecg_plot_path': full_ecg_plot_path, # Path to the full ECG plot
        'segment_plot_paths': segment_plot_paths, # List of paths to segment plots
        'summary': summary_counts # Ensure summary is always present
    }

    return summary_results


def predictByPart(data, peaks, record_output_dir, record_name_base):
    """
    Realiza la predicción de cada parte (segmento alrededor de un pico) de los datos.
    Genera gráficos para cada segmento de registro.

    Args:
        data (np.array): Datos de la señal preprocesados.
        peaks (np.array): Picos detectados en la señal.
        record_output_dir (str): Directorio donde se guardarán los gráficos de este registro.
        record_name_base (str): Nombre base del registro (solo para el título de las imágenes).

    Returns:
        tuple: Una tupla que contiene:
               - Una lista de diccionarios, cada uno con la clase predicha, el array completo
                 de probabilidades y la ruta al gráfico del segmento.
               - Una cadena de resumen de resultados (para compatibilidad con CLI).
               - Una lista de rutas a los gráficos de segmentos individuales.
    """
    # Mapeo de clases para la salida
    classes_map = ['N','Ventricular','Paced','A','F','Noise'] # Mapeo de índices a nombres de clase
    predicted_details = [] # Stores [{'class': 'N', 'probability': 'XX.X%', 'all_probs_array': [...], 'plot_path': '...'}]
    result_string = "" # String for the results summary (for CLI)
    segment_plot_paths = [] # List to store paths to individual segment plots
    
    # FIX: Initialize 'counter' here
    counter = [0] * len(classes_map) 

    # Load the trained Keras model. It is crucial to include `custom_objects`
    # if the model uses custom layers or functions.
    try:
        model = load_model(
            'modelos/MLII-latest.keras',
            custom_objects={
                'zeropad': zeropad,
                'zeropad_output_shape': zeropad_output_shape
            }
        )
    except Exception as e:
        print(f"{Colors.FAIL}Error al cargar el modelo: {e}{Colors.ENDC}")
        print(f"{Colors.WARNING}Asegúrate de que el modelo 'MLII-latest.keras' exista en la carpeta 'modelos'.{Colors.ENDC}")
        return [], "Error al cargar el modelo.", []

    config = get_config() # Get configuration for parameters like input_size

    total_records = len(peaks) # Count all peaks
    if total_records == 0:
        print(f"{Colors.WARNING}No hay registros para procesar.{Colors.ENDC}")
        return [], "", []

    print(f"\n{Colors.BLUE}Iniciando procesamiento de registros...{Colors.ENDC}")
    
    # Loop over all peaks
    for i, peak in enumerate(peaks):
        start_time = time.time() # Start timer for this step
        current_record_num = i + 1
        
        # Define the start and end of the segment centered on the peak
        start = peak - config.input_size // 2
        end = peak + config.input_size // 2

        # Adjust for data array boundaries, ensuring the segment has the correct size
        if start < 0:
            start = 0
            end = config.input_size
        if end > data.shape[1]:
            end = data.shape[1]
            start = data.shape[1] - config.input_size

        # Ensure the segment is valid before predicting
        if (end - start) != config.input_size:
            sys.stdout.write(f"\n{Colors.WARNING}Registro-{current_record_num}/{total_records}: Salta pico en {peak} - segmento inválido (tamaño incorrecto). {Colors.ENDC}\n")
            sys.stdout.flush()
            continue # Skip this record and move to the next

        # Perform prediction for the current segment
        prob = model.predict(data[:, start:end], verbose=0) # verbose=0 to not show Keras bar
        prob = prob[0] # Probabilities for the first (and only) sample in the batch

        ann_idx = np.argmax(prob) # Index of the class with the highest probability
        counter[ann_idx] += 1 # Increment counter for the predicted class

        end_time = time.time() # End timer for this step
        ms_per_step = (end_time - start_time) * 1000

        # Progress bar and progress message for the current line
        progress = (current_record_num / total_records) * 100
        bar_length = 20 # Visual length of the bar
        filled_bar = int(bar_length * progress / 100)
        
        # The filled part of the bar is white, the rest is dark gray
        bar_fill = f"{Colors.WHITE}" + '█' * filled_bar + f"{Colors.ENDC}"
        bar_empty = f"{Colors.DARK_GRAY}" + '-' * (bar_length - filled_bar) + f"{Colors.ENDC}"
        
        # Build the progress line for this step
        progress_line = (
            f"{Colors.CYAN}{ms_per_step:.0f}ms/step - [{bar_fill}{bar_empty}] "
            f"{progress:.1f}% ({current_record_num}/{total_records}){Colors.ENDC}"
        )
        
        # Individual classification message for each record (now always printed)
        classification_msg = (
            f" {Colors.GREEN}Registro-{current_record_num}/{total_records}: "
            f"Clasificado como '{classes_map[ann_idx]}' con una certeza del {100 * prob[ann_idx]:3.1f}%.{Colors.ENDC}"
        )
        
        # Print the full line for this step, followed by a newline
        sys.stdout.write(f"{progress_line}{classification_msg}\n")
        sys.stdout.flush()


        # Build the detailed results string (for CLI)
        result_string += f"({classes_map[ann_idx]}:{round(100 * prob[ann_idx], 1)}%)"

        # Generate a plot for EACH record segment
        individual_record_plot_filename = f'segment_{current_record_num}_pred_{classes_map[ann_idx]}.png'
        individual_record_plot_path = os.path.join(record_output_dir, individual_record_plot_filename)
        plt.figure(figsize=(8, 4))
        plt.plot(data[0, start:end, 0])
        plt.title(f"Registro: {record_name_base} - Segmento {current_record_num}: {classes_map[ann_idx]} ({100 * prob[ann_idx]:.1f}%)")
        plt.xlabel("Muestras")
        plt.ylabel("Amplitud normalizada")
        plt.grid(True)
        plt.savefig(individual_record_plot_path, format="png", dpi=300, bbox_inches='tight')
        plt.close()
        segment_plot_paths.append(individual_record_plot_path) # Add path to the list

        # Store predicted class, all probabilities, and plot path
        predicted_details.append({
            'class': classes_map[ann_idx],
            'probability': f"{round(100 * prob[ann_idx], 1)}%", # Probability of the predicted class as string (e.g., "74.2%")
            'all_probs_array': prob.tolist(), # Full array of probabilities (for average calculation in predict_and_summarize)
            'plot_path': individual_record_plot_path # Path to this segment's plot
        })
    
    sys.stdout.write(f"\n{Colors.GREEN}Procesamiento de registros completado.{Colors.ENDC}\n") # Final completion message
    sys.stdout.flush()

    # Add the summary of counters to the end of the results string (for CLI)
    summary_parts = []
    for idx, count in enumerate(counter):
        summary_parts.append(f"{count}-{classes_map[idx]}")
    result_string += ",".join(summary_parts)

    return predicted_details, result_string, segment_plot_paths

def main(config):
    """
    Main function to run the prediction process.

    Args:
        config (object): Configuration object.
    """
    data = None
    label = None
    record_name_for_folder = "unknown_record" # Default value for folder name

    if config.upload:
        print(f"{Colors.BLUE}Modo de carga de archivo activado. Asegúrate de que el archivo se pase correctamente.{Colors.ENDC}")
        # In a real scenario, the base name of the uploaded file would be obtained here
        # For this example, if there is no actual file uploaded, the default value is used
        # data = uploadedData("path/to/your/uploaded_file.csv")
    else:
        # Load data from the CINC 2017 challenge
        data, label, record_id_from_cinc = cincData(config) # Capture the record_id
        if data is None:
            print(f"{Colors.FAIL}No se pudieron cargar los datos. Terminando la ejecución.{Colors.ENDC}")
            return
        
        # Use the record_id obtained from cincData for the folder name
        if record_id_from_cinc:
            record_name_for_folder = record_id_from_cinc


    # Preprocessing of signal data and peak detection
    data, peaks = preprocess(data, config)

    # Perform prediction and get the result
    if data is not None and peaks is not None:
        # Pass record_name_for_folder to predict_and_summarize
        prediction_summary = predict_and_summarize(data, label, peaks, config, record_name_for_folder)

        # Print information in the requested order and with colors (for CLI)
        print(f"\n{Colors.HEADER}--- Resumen de Predicción General ---{Colors.ENDC}")

        # Detailed predictions section in table format
        col1_width = 12 # Record #
        col2_width = 18 # Predicted Class
        # Width for probabilities: 6 probs * 12 chars/prob + 5 separators * 3 chars/separator = 72 + 15 = 87.
        col3_width = 87 # Adjusted for scientific notation and separators
        
        print(f"{Colors.BLUE}Predicciones Detalladas (por parte):{Colors.ENDC}")
        # Top table line
        print(f"{Colors.CYAN}+{'-' * (col1_width + 2)}+{'-' * (col2_width + 2)}+{'-' * (col3_width + 2)}+{Colors.ENDC}")
        # Table header
        # Format probability header to match subcolumn width
        prob_header_parts = [
            f"{'N':<12}", f"{'V':<12}", f"{'P':<12}",
            f"{'A':<12}", f"{'F':<12}", f"{'Noise':<12}"
        ]
        prob_header_str = f"({' | '.join(prob_header_parts)})"
        
        print(f"{Colors.CYAN}| {Colors.UNDERLINE}{'Registro #':<{col1_width}}{Colors.ENDC}{Colors.CYAN} | {Colors.UNDERLINE}{'Clase Predicha':<{col2_width}}{Colors.ENDC}{Colors.CYAN} | {Colors.UNDERLINE}{'Probabilidades':<{col3_width}}{Colors.ENDC}{Colors.CYAN} |{Colors.ENDC}")
        # Separator between header and data
        print(f"{Colors.CYAN}+{'-' * (col1_width + 2)}+{'-' * (col2_width + 2)}+{'-' * (col3_width + 2)}+{Colors.ENDC}")
        
        for i, pred_detail in enumerate(prediction_summary['predictions_detailed']):
            pred_class = pred_detail['class']
            probabilities = pred_detail['all_probs_array']
            # Alternate text colors for rows
            text_color = Colors.BLUE if i % 2 == 0 else Colors.CYAN
            
            # Format probabilities with scientific notation and vertical lines between them
            prob_parts_formatted = [f'{p:.7e}' for p in probabilities] # Scientific notation format
            
            prob_cell_content = f'{prob_parts_formatted[0]:<12}' # First element with its padding
            for k in range(1, len(prob_parts_formatted)):
                # Add vertical separator with color and then the next element with its color and padding
                prob_cell_content += f'{Colors.CYAN} | {text_color}{prob_parts_formatted[k]:<12}'
            
            # Print the row with colors and vertical lines
            print(f"{Colors.CYAN}| {text_color}{i + 1:<{col1_width}}{Colors.ENDC}{Colors.CYAN} | {text_color}{pred_class:<{col2_width}}{Colors.ENDC}{Colors.CYAN} | {text_color}{prob_cell_content:<{col3_width}}{Colors.ENDC}{Colors.CYAN} |{Colors.ENDC}")
            
            # Horizontal line after each row, except the last
            if i < len(prediction_summary['predictions_detailed']) - 1:
                print(f"{Colors.CYAN}+{'-' * (col1_width + 2)}+{'-' * (col2_width + 2)}+{'-' * (col3_width + 2)}+{Colors.ENDC}")
        # Final table line
        print(f"{Colors.CYAN}+{'-' * (col1_width + 2)}+{'-' * (col2_width + 2)}+{'-' * (col3_width + 2)}+{Colors.ENDC}")


        # Average prediction section in table format
        col1_width_avg = 15 # Class
        col2_width_avg = 20 # Average Probability (for header text)
        
        print(f"\n{Colors.BLUE}Media de la Predicción:{Colors.ENDC}")
        
        if prediction_summary['average_probabilities']:
            avg_classes_map = ['Normal', 'Ventricular', 'Estimulado', 'Auricular', 'Fusión', 'Ruido']
            
            # Top table line
            print(f"{Colors.CYAN}+{'-' * (col1_width_avg + 2)}+{'-' * (col2_width_avg + 2)}+{Colors.ENDC}")
            # Table header
            print(f"{Colors.CYAN}| {Colors.UNDERLINE}{'Clase':<{col1_width_avg}}{Colors.ENDC}{Colors.CYAN} | {Colors.UNDERLINE}{'Probabilidad Media':<{col2_width_avg}}{Colors.ENDC}{Colors.CYAN} |{Colors.ENDC}")
            # Separator between header and data
            print(f"{Colors.CYAN}+{'-' * (col1_width_avg + 2)}+{'-' * (col2_width_avg + 2)}+{Colors.ENDC}")
            
            for i, prob_avg in enumerate(prediction_summary['average_probabilities']):
                # Alternate text colors for rows
                text_color = Colors.BLUE if i % 2 == 0 else Colors.CYAN
                # Format average probability in scientific notation
                formatted_prob_avg = f'{prob_avg:.7e}%'
                print(f"{Colors.CYAN}| {text_color}{avg_classes_map[i]:<{col1_width_avg}}{Colors.ENDC}{Colors.CYAN} | {text_color}{formatted_prob_avg:<{col2_width_avg}}{Colors.ENDC}{Colors.CYAN} |{Colors.ENDC}")
                
                # Horizontal line after each row, except the last
                if i < len(prediction_summary['average_probabilities']) - 1:
                    print(f"{Colors.CYAN}+{'-' * (col1_width_avg + 2)}+{'-' * (col2_width_avg + 2)}+{Colors.ENDC}")
            # Final table line
            print(f"{Colors.CYAN}+{'-' * (col1_width_avg + 2)}+{'-' * (col2_width_avg + 2)}+{Colors.ENDC}")
        else:
            print(f"{Colors.WARNING}No hay probabilidades promedio disponibles.{Colors.ENDC}")


        print(f"{Colors.GREEN}La etiqueta más probable es {Colors.BOLD}{prediction_summary['most_probable_class']}{Colors.ENDC}{Colors.GREEN} con una certeza del {prediction_summary['most_probable_certainty']:3.1f}%.{Colors.ENDC}")
        if prediction_summary['second_probable_class']:
            print(f"{Colors.CYAN}La segunda etiqueta prevista es {Colors.BOLD}{prediction_summary['second_probable_class']}{Colors.ENDC}{Colors.CYAN} con una certeza del {prediction_summary['second_probable_certainty']:3.1f}%.{Colors.ENDC}")
        if prediction_summary['third_probable_class']:
            print(f"{Colors.CYAN}La tercera etiqueta prevista es {Colors.BOLD}{prediction_summary['third_probable_class']}{Colors.ENDC}{Colors.CYAN} con una certeza del {prediction_summary['third_probable_certainty']:3.1f}%.{Colors.ENDC}")
        if prediction_summary['original_label']:
            print(f"{Colors.CYAN}La etiqueta original del registro es {Colors.BOLD}{prediction_summary['original_label']}{Colors.ENDC}")

    else:
        print(f"{Colors.FAIL}El preprocesamiento de datos falló. No se pudo realizar la predicción.{Colors.ENDC}")

if __name__ == '__main__':
    # This runs only if the script is executed directly
    config_obj = get_config() # Get configuration from command line
    main(config_obj)
