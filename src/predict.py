"""
Módulo para la predicción de datos de ECG utilizando un modelo entrenado.
Incluye funciones para la descarga de datos CINC, preprocesamiento y predicción.

Datos CINC provienen de https://physionet.org/challenge/2017/
"""

from __future__ import division, print_function
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
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

def predict_and_summarize(data, label, peaks, config, record_name_base, raw_multilead_signal=None):
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
              la etiqueta original, las probabilidades promedio y las rutas a los gráficos,
              y una sugerencia sobre la afección cardíaca.
    """
    # Crear el directorio para los resultados de este registro
    record_output_dir = f'resultados/{record_name_base}'
    mkdir_recursive(record_output_dir)

    # Generar y guardar el gráfico de la señal ECG completa
    full_ecg_plot_filename = f'ECG_Completo_{record_name_base}.png'
    full_ecg_plot_path = os.path.join(record_output_dir, full_ecg_plot_filename)
    plt.figure(figsize=(15, 6))
    plt.plot(data[0, :, 0]) # Asumiendo data es (1, length, 1)
    plt.title(f"Señal ECG Completa para Registro: {record_name_base}")
    plt.xlabel("Muestras")
    plt.ylabel("Amplitud normalizada")
    plt.grid(True)
    # --- AJUSTE para padding horizontal en el gráfico completo ---
    ax = plt.gca()
    ax.set_xlim(0, len(data[0, :, 0]) - 1) # Establecer límite exacto del eje X
    ax.autoscale_view() # Asegurar que la vista de los ejes se actualice
    # --- FIN AJUSTE ---
    plt.savefig(full_ecg_plot_path, format="png", dpi=300, bbox_inches='tight', pad_inches=0) 
    plt.close()
    print(f"{Colors.BLUE}Señal ECG completa guardada en: {full_ecg_plot_path}{Colors.ENDC}")

    # --- NUEVO: Generar y guardar el gráfico de la señal ECG para lectura ---
    ecg_lectura_filename = f'ECG_Lectura_{record_name_base}.png'
    ecg_lectura_plot_path = os.path.join(record_output_dir, ecg_lectura_filename)
    
    # Calcular las dimensiones para un espaciado similar a un ECG real
    # Asumiendo 360 Hz (después del remuestreo en preprocess) y 25 mm/s de velocidad de papel
    # 360 muestras/s / 25 mm/s = 14.4 muestras/mm
    # Queremos ~10 segundos de datos visibles. 10s * 360 muestras/s = 3600 muestras.
    num_samples = len(data[0, :, 0])
    seconds = num_samples / 360
    mm_per_second = 25
    total_mm = seconds * mm_per_second
    
    # Establecer la relación de aspecto para que se parezca a un papel de ECG
    # Ancho: total_mm, Alto: ~60mm (para una derivación). Relación de aspecto ~ total_mm / 60
    fig_width = total_mm / 10 # Convertir mm a pulgadas para figsize (aprox)
    fig_height = 6 # Altura fija en pulgadas

    plt.figure(figsize=(fig_width, fig_height))
    plt.plot(data[0, :, 0], color='black', linewidth=0.75)
    plt.title(f"ECG para Lectura: {record_name_base}")
    plt.xlabel(f"Tiempo (s) - {mm_per_second} mm/s")
    
    # Configurar los ejes para que se parezcan a un papel de ECG
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#d3d3d3')
    ax.spines['left'].set_color('#d3d3d3')

    # Rejilla principal (cuadrados de 5mm)
    # 1 segundo = 25 mm = 5 cuadrados grandes. 1 cuadrado = 0.2 s
    major_ticks_x = np.arange(0, num_samples, 0.2 * 360) # Cada 0.2 segundos
    major_ticks_y = np.arange(np.floor(data.min()), np.ceil(data.max()), 0.5) # Cada 0.5 mV
    ax.set_xticks(major_ticks_x)
    ax.set_yticks(major_ticks_y)
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='red')
    
    # Rejilla secundaria (cuadrados de 1mm)
    # 1 cuadrado grande = 5 cuadrados pequeños. 0.2s / 5 = 0.04s
    minor_ticks_x = np.arange(0, num_samples, 0.04 * 360)
    minor_ticks_y = np.arange(np.floor(data.min()), np.ceil(data.max()), 0.1)
    ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(minor_ticks_y, minor=True)
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='lightcoral')

    #Establecer los límites del eje X para eliminar el padding horizontal
    ax.set_xlim(0, num_samples - 1) # Ajusta el límite derecho a `num_samples - 1` para incluir el último punto.

    # Etiquetas de los ejes
    ax.set_xticklabels([f'{tick/360:.1f}' for tick in major_ticks_x])
    plt.ylabel("Amplitud (mV)")

    #Ajustar los límites del eje Y para que se ajusten perfectamente a los datos
    # Reducir el padding vertical a 0.01 (1%) para un ajuste más ceñido
    y_min, y_max = data.min(), data.max()
    ax.set_ylim(y_min - (y_max - y_min) * 0.01, y_max + (y_max - y_min) * 0.01) # Padding vertical muy reducido
    ax.autoscale_view() # Asegurar que la vista de los ejes se actualice

    plt.savefig(ecg_lectura_plot_path, format="png", dpi=300, bbox_inches='tight', pad_inches=0) 
    plt.close()
    print(f"{Colors.BLUE}Gráfico de lectura de ECG guardado en: {ecg_lectura_plot_path}{Colors.ENDC}")
    # --- FIN NUEVO ---

    # --- NUEVO: Generar y guardar el gráfico de la señal ECG de lectura reducido ---
    ecg_lectura_reducido_filename = f'ECG_Lectura_Reducido_{record_name_base}.png'
    ecg_lectura_reducido_plot_path = os.path.join(record_output_dir, ecg_lectura_reducido_filename)

    # Reducir el número de muestras a la mitad para el gráfico reducido
    half_num_samples = num_samples // 2
    data_reduced = data[0, :half_num_samples, 0]

    plt.figure(figsize=(fig_width / 2, fig_height)) # Ancho de la figura a la mitad
    plt.plot(data_reduced, color='black', linewidth=0.75)
    plt.title(f"ECG para Lectura Reducido: {record_name_base}")
    plt.xlabel(f"Tiempo (s) - {mm_per_second} mm/s")

    # Configurar los ejes para que se parezcan a un papel de ECG (ajustado para el tamaño reducido)
    ax_reducido = plt.gca()
    ax_reducido.spines['top'].set_visible(False)
    ax_reducido.spines['right'].set_visible(False)
    ax_reducido.spines['bottom'].set_color('#d3d3d3')
    ax_reducido.spines['left'].set_color('#d3d3d3')

    # Rejilla principal (cuadrados de 5mm)
    major_ticks_x_reducido = np.arange(0, half_num_samples, 0.2 * 360)
    major_ticks_y_reducido = np.arange(np.floor(data.min()), np.ceil(data.max()), 0.5)
    ax_reducido.set_xticks(major_ticks_x_reducido)
    ax_reducido.set_yticks(major_ticks_y_reducido)
    ax_reducido.grid(which='major', linestyle='-', linewidth='0.5', color='red')

    # Rejilla secundaria (cuadrados de 1mm)
    minor_ticks_x_reducido = np.arange(0, half_num_samples, 0.04 * 360)
    minor_ticks_y_reducido = np.arange(np.floor(data.min()), np.ceil(data.max()), 0.1)
    ax_reducido.set_xticks(minor_ticks_x_reducido, minor=True)
    ax_reducido.set_yticks(minor_ticks_y_reducido, minor=True)
    ax_reducido.grid(which='minor', linestyle=':', linewidth='0.5', color='lightcoral')

    # Establecer los límites del eje X para eliminar el padding horizontal
    ax_reducido.set_xlim(0, half_num_samples - 1)

    # Etiquetas de los ejes
    ax_reducido.set_xticklabels([f'{tick/360:.1f}' for tick in major_ticks_x_reducido])
    plt.ylabel("Amplitud (mV)")

    # Ajustar los límites del eje Y para que se ajusten perfectamente a los datos
    ax_reducido.set_ylim(y_min - (y_max - y_min) * 0.01, y_max + (y_max - y_min) * 0.01)
    ax_reducido.autoscale_view()

    plt.savefig(ecg_lectura_reducido_plot_path, format="png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"{Colors.BLUE}Gráfico de lectura de ECG reducido guardado en: {ecg_lectura_reducido_plot_path}{Colors.ENDC}")
    # --- FIN NUEVO: Gráfico reducido ---

    # --- NUEVO: Gráfico de 12 derivaciones (si existen) ---
    ecg_12leads_plot_path = None
    # Usar raw_multilead_signal si está disponible y es 2D con >=2 columnas
    if raw_multilead_signal is not None and hasattr(raw_multilead_signal, 'shape') and raw_multilead_signal.ndim == 2 and raw_multilead_signal.shape[1] >= 2:
        num_leads = raw_multilead_signal.shape[1]
        leads_to_plot = min(num_leads, 12)
        # Obtener nombres de columna si es un DataFrame de pandas
        lead_names = None
        raw_multilead_np = raw_multilead_signal
        try:
            import pandas as pd
            if isinstance(raw_multilead_signal, pd.DataFrame):
                lead_names = list(raw_multilead_signal.columns)[:leads_to_plot]
                raw_multilead_np = raw_multilead_signal.values
        except ImportError:
            pass
        # Si no es DataFrame, intentar obtener de config.lead_names
        if lead_names is None:
            # Si el archivo original era un .csv leído con pandas, pero aquí llega como np.ndarray,
            # intentamos recuperar los nombres leyendo el .csv de nuevo
            if hasattr(config, 'csv_path') and os.path.exists(config.csv_path):
                try:
                    # --- NUEVO: Leer encabezado real saltando comentarios ---
                    with open(config.csv_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if not line.strip().startswith('#') and line.strip():
                                lead_names = [col.strip() for col in line.strip().split(',')][:leads_to_plot]
                                break
                    # Si no se encontraron nombres, intentar con pandas como fallback
                    if not lead_names:
                        import pandas as pd
                        df_tmp = pd.read_csv(config.csv_path, comment='#')
                        lead_names = list(df_tmp.columns)[:leads_to_plot]
                except Exception as e:
                    print(f"No se pudieron extraer los nombres de las derivaciones del CSV: {e}")
                    lead_names = None
            elif hasattr(config, 'lead_names') and config.lead_names:
                lead_names = list(config.lead_names)[:leads_to_plot]
            else:
                # Nombres estándar para 12 derivadas
                standard_12lead_names = [
                    'I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                    'V1', 'V2', 'V3', 'V4', 'V5', 'V6'
                ]
                lead_names = standard_12lead_names[:leads_to_plot]
        ecg_12leads_filename = f'ECG_12_derivadas_{record_name_base}.png'
        ecg_12leads_plot_path = os.path.join(record_output_dir, ecg_12leads_filename)
        fig, axes = plt.subplots(4, 3, figsize=(18, 10), sharex=True)
        fig.suptitle(f"ECG 12 derivaciones para: {record_name_base}", fontsize=18)
        mm_per_second = 25
        fs = getattr(config, 'sample_rate', 360)
        num_samples = raw_multilead_np.shape[0]
        for i in range(leads_to_plot):
            ax = axes[i // 3, i % 3]
            ax.plot(raw_multilead_np[:, i], color='black', linewidth=0.75)
            ax.set_title(lead_names[i], fontsize=12)
            ax.set_ylabel("mV")
            # Rejilla principal (cuadrados de 5mm)
            major_ticks_x = np.arange(0, num_samples, 0.2 * fs)
            major_ticks_y = np.arange(np.floor(raw_multilead_np[:, i].min()), np.ceil(raw_multilead_np[:, i].max()), 0.5)
            ax.set_xticks(major_ticks_x)
            ax.set_yticks(major_ticks_y)
            ax.grid(which='major', linestyle='-', linewidth='0.5', color='red')
            # Rejilla secundaria (cuadrados de 1mm)
            minor_ticks_x = np.arange(0, num_samples, 0.04 * fs)
            minor_ticks_y = np.arange(np.floor(raw_multilead_np[:, i].min()), np.ceil(raw_multilead_np[:, i].max()), 0.1)
            ax.set_xticks(minor_ticks_x, minor=True)
            ax.set_yticks(minor_ticks_y, minor=True)
            ax.grid(which='minor', linestyle=':', linewidth='0.5', color='lightcoral')
            ax.set_xlim(0, num_samples - 1)
            # Etiquetas de tiempo solo en la última fila
            if i // 3 == 3:
                ax.set_xlabel(f"Tiempo (s) - {mm_per_second} mm/s")
            else:
                ax.set_xticklabels([])
        # Eliminar subplots vacíos si hay menos de 12 derivadas
        for j in range(leads_to_plot, 12):
            fig.delaxes(axes[j // 3, j % 3])
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(ecg_12leads_plot_path, format="png", dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"{Colors.BLUE}Gráfico de 12 derivaciones guardado en: {ecg_12leads_plot_path}{Colors.ENDC}")
    # --- FIN NUEVO ---

    # Realiza la predicción por partes y obtiene la lista de predicciones y la cadena de resultados
    # predictByPart ahora devolverá también las rutas de los gráficos de segmentos
    predicted_list_raw, result_string, segment_plot_paths = predictByPart(data, peaks, record_output_dir, record_name_base)

    # Las clases mapeadas para una salida más legible
    classes_map = ['N','Ventricular','Paced','A','F','Noise'] # Mapeo de índices a nombres de clase

    # Inicializa un diccionario de resumen por defecto
    summary_counts = {cls: 0 for cls in classes_map}
    
    # Calcula las probabilidades promedio
    all_segment_probs = [item['all_probs_array'] for item in predicted_list_raw if 'all_probs_array' in item]
    if all_segment_probs:
        avg_predict = np.mean(all_segment_probs, axis=0)
    else:
        avg_predict = np.array([0.0] * len(classes_map)) # Establece ceros si no se encuentran probabilidades

    # Si se realizaron predicciones, actualiza summary_counts desde la cadena de resultados
    if predicted_list_raw:
        # Analiza la cadena de resultados para obtener los recuentos para el resumen
        # Asumiendo que la cadena de resultados termina con "Count1-Class1,Count2-Class2,..."
        summary_str_part = result_string[result_string.rfind(')') + 1:].strip()
        summary_parts = summary_str_part.split(',')
        for part in summary_parts:
            clean_part = part.strip()
            if '-' in clean_part:
                try:
                    count, name = clean_part.split('-', 1)
                    # Mapea el nombre de nuevo a la clave correcta en summary_counts
                    # El classes_map se usa para las claves, p. ej., 'N', 'Ventricular'
                    # La cadena de salida de la CLI tiene 'N', 'Ventricular', 'Paced', etc.
                    # Asegúrate de que la clave coincida con classes_map
                    if name.strip() == 'N': summary_counts['N'] = int(count.strip())
                    elif name.strip() == 'Ventricular': summary_counts['Ventricular'] = int(count.strip())
                    elif name.strip() == 'Paced': summary_counts['Paced'] = int(count.strip())
                    elif name.strip() == 'A': summary_counts['A'] = int(count.strip())
                    elif name.strip() == 'F': summary_counts['F'] = int(count.strip())
                    elif name.strip() == 'Noise': summary_counts['Noise'] = int(count.strip())
                except ValueError:
                    print(f"Warning: No se pudo analizar el recuento o el nombre de la parte del resumen: {clean_part}")


    # Ordena los índices de probabilidades de mayor a menor
    sorted_indices = avg_predict.argsort()[::-1]

    # Clase más probable
    most_probable_idx = sorted_indices[0]
    most_probable_class = classes_map[most_probable_idx]
    most_probable_certainty = 100 * avg_predict[most_probable_idx]

    # Segunda clase más probable
    second_probable_class = None
    second_probable_certainty = 0.0
    if len(avg_predict) > 1:
        sec_idx = sorted_indices[1]
        second_probable_class = classes_map[sec_idx]
        second_probable_certainty = 100 * avg_predict[sec_idx]

    # Tercera clase más probable
    third_probable_class = None
    third_probable_certainty = 0.0
    if len(avg_predict) > 2:
        third_idx = sorted_indices[2]
        third_probable_class = classes_map[third_idx]
        third_probable_certainty = 100 * avg_predict[third_idx]

    # --- Sugerencia de afección cardíaca ---
    cardiac_condition_suggestion = ""
    if most_probable_class == 'N':
        if most_probable_certainty >= 80.0: # Umbral para considerar "Normal" con alta certeza
            cardiac_condition_suggestion = "Los resultados sugieren que el paciente tiene un ritmo cardíaco normal y no presenta afecciones cardíacas significativas."
        else:
            cardiac_condition_suggestion = "Aunque la etiqueta más probable es Normal, la certeza no es muy alta. Se recomienda una revisión médica más detallada para descartar posibles afecciones sutiles."
    elif most_probable_class == 'Ventricular':
        cardiac_condition_suggestion = f"Los resultados sugieren que el paciente sufre de contracciones ventriculares prematuras ({most_probable_class}). Se recomienda una evaluación médica."
    elif most_probable_class == 'A':
        cardiac_condition_suggestion = f"Los resultados sugieren que el paciente sufre de latidos auriculares prematuros ({most_probable_class}). Se recomienda una evaluación médica."
    elif most_probable_class == 'Paced':
        cardiac_condition_suggestion = f"Los resultados sugieren que el paciente tiene un ritmo cardíaco inducido por marcapasos ({most_probable_class}). Esto es un hallazgo esperado si el paciente usa marcapasos."
    elif most_probable_class == 'F':
        cardiac_condition_suggestion = f"Los resultados sugieren que el paciente presenta indicios de latidos fusionados ({most_probable_class}), indicando una combinación de actividad ventricular y normal. Se recomienda una evaluación médica."
    elif most_probable_class == 'Noise':
        cardiac_condition_suggestion = "La señal contiene demasiado ruido (Noise), lo que impide una clasificación confiable. Se recomienda repetir el estudio o asegurar una mejor calidad de señal."
    else:
        cardiac_condition_suggestion = "No se pudo determinar una sugerencia clara de afección cardíaca debido a la naturaleza de la predicción."

    # Construye el diccionario de resultados
    summary_results = {
        'predictions_detailed': predicted_list_raw, # CORREGIDO: Usar predicted_list_raw
        'prediction_summary_string': result_string, # Esta es la cadena de salida de la CLI (mantenida para compatibilidad)
        'most_probable_class': most_probable_class,
        'most_probable_certainty': most_probable_certainty,
        'second_probable_class': second_probable_class,
        'second_probable_certainty': second_probable_certainty,
        'third_probable_class': third_probable_class,
        'third_probable_certainty': third_probable_certainty,
        'original_label': label,
        'average_probabilities': avg_predict.tolist(), # Convertir a lista para serialización si es necesario
        'full_ecg_plot_path': full_ecg_plot_path, # Ruta al gráfico completo del ECG
        'ecg_lectura_plot_path': ecg_lectura_plot_path, # NUEVO: Ruta al ECG de lectura
        'ecg_lectura_reducido_plot_path': ecg_lectura_reducido_plot_path, # NUEVO: Ruta al ECG de lectura reducido
        'ecg_12leads_plot_path': ecg_12leads_plot_path, # NUEVO: Ruta al gráfico de 12 derivaciones
        'segment_plot_paths': segment_plot_paths, # Lista de rutas a los gráficos de segmentos
        'summary': summary_counts, # Asegurarse de que el resumen siempre esté presente
        'cardiac_condition_suggestion': cardiac_condition_suggestion # NUEVO: Sugerencia de afección cardíaca
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
    predicted_details = [] # Almacena [{'class': 'N', 'probability': 'XX.X%', 'all_probs_array': [...], 'plot_path': '...'}]
    result_string = "" # Cadena para el resumen de resultados (para CLI)
    segment_plot_paths = [] # Lista para almacenar las rutas a gráficos de segmentos individuales
    
    # FIX: Inicializa 'counter' aquí
    counter = [0] * len(classes_map) 

    # Carga el modelo Keras entrenado. Es crucial incluir `custom_objects`
    # si el modelo utiliza capas o funciones personalizadas.
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

    config = get_config() # Obtiene la configuración para parámetros como input_size

    total_records = len(peaks) # Cuenta todos los picos
    if total_records == 0:
        print(f"{Colors.WARNING}No hay registros para procesar.{Colors.ENDC}")
        return [], "", []

    print(f"\n{Colors.BLUE}Iniciando procesamiento de registros...{Colors.ENDC}")
    
    # Itera sobre todos los picos
    for i, peak in enumerate(peaks):
        start_time = time.time() # Inicia el temporizador para este paso
        current_record_num = i + 1
        
        # Define el inicio y el fin del segmento centrado en el pico
        start = peak - config.input_size // 2
        end = peak + config.input_size // 2

        # Ajusta para los límites del array de datos, asegurando que el segmento tenga el tamaño correcto
        if start < 0:
            start = 0
            end = config.input_size
        if end > data.shape[1]:
            end = data.shape[1]
            start = data.shape[1] - config.input_size

        # Asegura que el segmento sea válido antes de predecir
        if (end - start) != config.input_size:
            sys.stdout.write(f"\n{Colors.WARNING}Registro-{current_record_num}/{total_records}: Salta pico en {peak} - segmento inválido (tamaño incorrecto). {Colors.ENDC}\n")
            sys.stdout.flush()
            continue # Salta este registro y pasa al siguiente

        # Realiza la predicción para el segmento actual
        prob = model.predict(data[:, start:end], verbose=0) # verbose=0 para no mostrar la barra de Keras
        prob = prob[0] # Probabilidades para la primera (y única) muestra en el lote

        ann_idx = np.argmax(prob) # Índice de la clase con la probabilidad más alta
        counter[ann_idx] += 1 # Incrementa el contador para la clase predicha

        end_time = time.time() # Termina el temporizador para este paso
        ms_per_step = (end_time - start_time) * 1000

        # Barra de progreso y mensaje de progreso para la línea actual
        progress = (current_record_num / total_records) * 100
        bar_length = 20 # Longitud visual de la barra
        filled_bar = int(bar_length * progress / 100)
        
        # La parte llena de la barra es blanca, el resto es gris oscuro
        bar_fill = f"{Colors.WHITE}" + '█' * filled_bar + f"{Colors.ENDC}"
        bar_empty = f"{Colors.DARK_GRAY}" + '-' * (bar_length - filled_bar) + f"{Colors.ENDC}"
        
        # Construye la línea de progreso para este paso
        progress_line = (
            f"{Colors.CYAN}{ms_per_step:.0f}ms/paso - [{bar_fill}{bar_empty}] "
            f"{progress:.1f}% ({current_record_num}/{total_records}){Colors.ENDC}"
        )
        
        # Mensaje de clasificación individual para cada registro (ahora siempre se imprime)
        classification_msg = (
            f" {Colors.GREEN}Registro-{current_record_num}/{total_records}: "
            f"Clasificado como '{classes_map[ann_idx]}' con una certeza del {100 * prob[ann_idx]:3.1f}%.{Colors.ENDC}"
        )
        
        # Imprime la línea completa para este paso, seguida de un salto de línea
        sys.stdout.write(f"{progress_line}{classification_msg}\n")
        sys.stdout.flush()


        # Construye la cadena de resultados detallados (para CLI)
        result_string += f"({classes_map[ann_idx]}:{round(100 * prob[ann_idx], 1)}%)"

        # Genera un gráfico para CADA segmento de registro
        individual_record_plot_filename = f'Segmento_{current_record_num}_pred_{classes_map[ann_idx]}.png'
        individual_record_plot_path = os.path.join(record_output_dir, individual_record_plot_filename)
        plt.figure(figsize=(8, 4))
        plt.plot(data[0, start:end, 0])
        plt.title(f"Registro: {record_name_base} - Segmento {current_record_num}: {classes_map[ann_idx]} ({100 * prob[ann_idx]:.1f}%)")
        plt.xlabel("Muestras")
        plt.ylabel("Amplitud normalizada")
        plt.grid(True)
        # --- AJUSTE para padding horizontal en el gráfico de segmento ---
        ax = plt.gca()
        ax.set_xlim(0, (end - start) - 1) # Establecer límite exacto del eje X para el segmento
        ax.autoscale_view() # Asegurar que la vista de los ejes se actualice
        # --- FIN AJUSTE ---
        plt.savefig(individual_record_plot_path, format="png", dpi=300, bbox_inches='tight', pad_inches=0) 
        plt.close()
        segment_plot_paths.append(individual_record_plot_path) # Añade la ruta a la lista

        # Almacena la clase predicha, todas las probabilidades y la ruta del gráfico
        predicted_details.append({
            'class': classes_map[ann_idx],
            'probability': f"{round(100 * prob[ann_idx], 1)}%", # Probabilidad de la clase predicha como cadena (ej., "74.2%")
            'all_probs_array': prob.tolist(), # Array completo de probabilidades (para cálculo promedio en predict_and_summarize)
            'plot_path': individual_record_plot_path # Ruta al gráfico de este segmento
        })
    
    sys.stdout.write(f"\n{Colors.GREEN}Procesamiento de registros completado.{Colors.ENDC}\n") # Mensaje final de completado
    sys.stdout.flush()

    # Añade el resumen de contadores al final de la cadena de resultados (para CLI)
    summary_parts = []
    for idx, count in enumerate(counter):
        summary_parts.append(f"{count}-{classes_map[idx]}")
    result_string += ",".join(summary_parts)

    return predicted_details, result_string, segment_plot_paths

def print_table(headers, rows, col_widths, colors=None):
    """
    Imprime una tabla con encabezados y filas, usando anchos de columna y colores opcionales.
    """
    # Línea superior
    print(f"{Colors.CYAN}+" + "+".join(['-' * (w + 2) for w in col_widths]) + f"+{Colors.ENDC}")
    # Encabezado
    header_line = f"{Colors.CYAN}| "
    for i, header in enumerate(headers):
        color = colors[i] if colors and i < len(colors) else Colors.UNDERLINE
        header_line += f"{color}{header:<{col_widths[i]}}{Colors.ENDC}{Colors.CYAN} | "
    print(header_line[:-1] + f"{Colors.ENDC}")
    # Separador
    print(f"{Colors.CYAN}+" + "+".join(['-' * (w + 2) for w in col_widths]) + f"+{Colors.ENDC}")
    # Filas
    for i, row in enumerate(rows):
        text_color = Colors.BLUE if i % 2 == 0 else Colors.CYAN
        row_line = f"{Colors.CYAN}| "
        for j, cell in enumerate(row):
            row_line += f"{text_color}{cell:<{col_widths[j]}}{Colors.ENDC}{Colors.CYAN} | "
        print(row_line[:-1] + f"{Colors.ENDC}")
        if i < len(rows) - 1:
            print(f"{Colors.CYAN}+" + "+".join(['-' * (w + 2) for w in col_widths]) + f"+{Colors.ENDC}")
    # Línea final
    print(f"{Colors.CYAN}+" + "+".join(['-' * (w + 2) for w in col_widths]) + f"+{Colors.ENDC}")

def print_prediction_summary(prediction_summary):
    print(f"{Colors.GREEN}La etiqueta más probable es {Colors.BOLD}{prediction_summary['most_probable_class']}{Colors.ENDC}{Colors.GREEN} con una certeza del {prediction_summary['most_probable_certainty']:3.1f}%.{Colors.ENDC}")
    if prediction_summary['second_probable_class']:
        print(f"{Colors.CYAN}La segunda etiqueta prevista es {Colors.BOLD}{prediction_summary['second_probable_class']}{Colors.ENDC}{Colors.CYAN} con una certeza del {prediction_summary['second_probable_certainty']:3.1f}%.{Colors.ENDC}")
    if prediction_summary['third_probable_class']:
        print(f"{Colors.CYAN}La tercera etiqueta prevista es {Colors.BOLD}{prediction_summary['third_probable_class']}{Colors.ENDC}{Colors.CYAN} con una certeza del {prediction_summary['third_probable_certainty']:3.1f}%.{Colors.ENDC}")
    if prediction_summary['original_label']:
        print(f"{Colors.CYAN}La etiqueta original del registro es {Colors.BOLD}{prediction_summary['original_label']}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.WARNING}{prediction_summary['cardiac_condition_suggestion']}{Colors.ENDC}")

def print_detailed_predictions(predictions_detailed):
    col1_width = 12
    col2_width = 18
    col3_width = 87
    headers = ['Registro #', 'Clase Predicha', 'Probabilidades']
    rows = []
    for i, pred_detail in enumerate(predictions_detailed):
        pred_class = pred_detail['class']
        probabilities = pred_detail['all_probs_array']
        # Multiplica por 100, limita a 5 decimales y añade el símbolo de porcentaje
        prob_parts_formatted = [f'{p*100:.5f}%' for p in probabilities]
        prob_cell_content = f'{prob_parts_formatted[0]:<12}'
        for k in range(1, len(prob_parts_formatted)):
            prob_cell_content += f'{Colors.CYAN} | {Colors.BLUE if i % 2 == 0 else Colors.CYAN}{prob_parts_formatted[k]:<12}'
        rows.append([str(i + 1), pred_class, prob_cell_content])
    print(f"{Colors.BLUE}Predicciones Detalladas (por parte):{Colors.ENDC}")
    print_table(headers, rows, [col1_width, col2_width, col3_width])

def print_average_probabilities(avg_probs):
    col1_width = 15
    col2_width = 20
    headers = ['Clase', 'Probabilidad Media']
    avg_classes_map = ['Normal', 'Ventricular', 'Estimulado', 'Auricular', 'Fusión', 'Ruido']
    rows = []
    for i, prob_avg in enumerate(avg_probs):
        # Multiplica por 100 y limita a 5 decimales
        formatted_prob_avg = f'{prob_avg*100:.5f}%'
        rows.append([avg_classes_map[i], formatted_prob_avg])
    print(f"\n{Colors.BLUE}Media de la Predicción:{Colors.ENDC}")
    print_table(headers, rows, [col1_width, col2_width])

def main(config):
    """
    Función principal para ejecutar el proceso de predicción.

    Args:
        config (object): Objeto de configuración.
    """
    data = None
    label = None
    record_name_for_folder = "registro_desconocido" # Valor por defecto para el nombre de la carpeta

    if config.upload:
        print(f"{Colors.BLUE}Modo de carga de archivo activado. Asegúrate de que el archivo se pase correctamente.{Colors.ENDC}")
        # En un escenario real, el nombre base del archivo subido se obtendría aquí
        # Para este ejemplo, si no hay un archivo real subido, se utiliza el valor por defecto
        # data = uploadedData("path/to/your/uploaded_file.csv")
    else:
        # Cargar datos del desafío CINC 2017
        data, label, record_id_from_cinc = cincData(config) # Capturar el record_id
        if data is None:
            print(f"{Colors.FAIL}No se pudieron cargar los datos. Terminando la ejecución.{Colors.ENDC}")
            return
        
        # Usar el record_id obtenido de cincData para el nombre de la carpeta
        if record_id_from_cinc:
            record_name_for_folder = record_id_from_cinc


    # Preprocesamiento de datos de señal y detección de picos
    data, peaks = preprocess(data, config)

    # Realizar la predicción y obtener el resultado
    if data is not None and peaks is not None:
        # Pasar record_name_for_folder a predict_and_summarize
        prediction_summary = predict_and_summarize(data, label, peaks, config, record_name_for_folder)

        # Imprimir información en el orden solicitado y con colores (para CLI)
        print(f"\n{Colors.HEADER}--- Resumen de Predicción General ---{Colors.ENDC}")
        print_detailed_predictions(prediction_summary['predictions_detailed'])
        if prediction_summary['average_probabilities']:
            print_average_probabilities(prediction_summary['average_probabilities'])
        else:
            print(f"{Colors.WARNING}No hay probabilidades promedio disponibles.{Colors.ENDC}")
        print_prediction_summary(prediction_summary)
    else:
        print(f"{Colors.FAIL}El preprocesamiento de datos falló. No se pudo realizar la predicción.{Colors.ENDC}")

if __name__ == '__main__':
    # Esto se ejecuta solo si el script se ejecuta directamente
    config_obj = get_config() # Obtener la configuración de la línea de comandos
    main(config_obj)