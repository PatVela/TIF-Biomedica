# -*- coding: utf-8 -*-

"""
Predicción de ECG utilizando un modelo PyTorch entrenado
con el dataset PhysioNet/CinC Challenge 2020.

El modelo trabaja con:
    - 12 derivaciones
    - 500 Hz
    - 10 segundos
    - 5000 muestras
    - 27 clases multilabel

Entrada esperada:
    (batch, 12, 5000)

Salida del modelo:
    (batch, 27) logits

Se utiliza Sigmoid para obtener las probabilidades independientes
de cada diagnóstico.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

from config import get_config
from graph import ECG_model
from utils import preprocess, uploadedData, mkdir_recursive


# ============================================================
# COLORES PARA LA TERMINAL
# ============================================================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# ============================================================
# CONFIGURACIÓN CINC2020
# ============================================================

# Debe coincidir exactamente con el orden utilizado durante
# el entrenamiento y con dx_mapping_scored.csv.
CLASSES = [
    # Estas 27 clases serán reemplazadas por el orden exacto
    # de dx_mapping_scored.csv utilizado por data2020.py.
]

LEAD_NAMES = [
    'I', 'II', 'III', 'aVR', 'aVL', 'aVF',
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6'
]

SAMPLE_RATE = 500
SIGNAL_LENGTH = 5000
NUM_LEADS = 12

# Umbral inicial para considerar una clase como positiva.
# Más adelante podemos optimizarlo utilizando el conjunto
# de validación.
DEFAULT_THRESHOLD = 0.5


# ============================================================
# DISPOSITIVO
# ============================================================

def get_device():
    """
    Selecciona automáticamente CUDA si está disponible.
    En caso contrario utiliza CPU.
    """

    if torch.cuda.is_available():
        device = torch.device('cuda')

        print(
            f"{Colors.GREEN}"
            f"CUDA disponible: {torch.cuda.get_device_name(0)}"
            f"{Colors.ENDC}"
        )

    else:
        device = torch.device('cpu')

        print(
            f"{Colors.WARNING}"
            f"CUDA no disponible. Se utilizará CPU."
            f"{Colors.ENDC}"
        )

    return device


# ============================================================
# CARGAR MODELO
# ============================================================

def load_trained_model(config, device):
    """
    Carga el modelo PyTorch entrenado.

    El checkpoint debe contener normalmente:
        {
            'model_state_dict': ...,
            ...
        }

    También permite cargar directamente un state_dict.
    """

    model_path = config.trained_model

    if model_path is None:
        model_path = 'modelos/MLII-latest.pt'

    if not os.path.exists(model_path):

        raise FileNotFoundError(
            f"No se encontró el modelo:\n{model_path}"
        )

    print(
        f"{Colors.BLUE}"
        f"Cargando modelo: {model_path}"
        f"{Colors.ENDC}"
    )

    # Crea la arquitectura.
    model = ECG_model(config)

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    # --------------------------------------------------------
    # Checkpoint completo
    # --------------------------------------------------------

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:

        model.load_state_dict(
            checkpoint['model_state_dict']
        )

        print(
            f"{Colors.GREEN}"
            "Checkpoint completo cargado correctamente."
            f"{Colors.ENDC}"
        )

    # --------------------------------------------------------
    # State dict directo
    # --------------------------------------------------------

    elif isinstance(checkpoint, dict):

        model.load_state_dict(checkpoint)

        print(
            f"{Colors.GREEN}"
            "State dict cargado correctamente."
            f"{Colors.ENDC}"
        )

    else:

        raise ValueError(
            "El archivo del modelo no contiene un checkpoint "
            "o state_dict válido."
        )

    model.to(device)
    model.eval()

    return model


# ============================================================
# PREPARAR ECG PARA PYTORCH
# ============================================================

def prepare_signal_for_model(data):
    """
    Convierte una señal ECG a la forma requerida por PyTorch.

    Formato final:
        (1, 12, 5000)

    Se aceptan formatos comunes:
        (5000, 12)
        (12, 5000)
        (1, 5000, 12)
        (1, 12, 5000)
    """

    data = np.asarray(data, dtype=np.float32)

    # --------------------------------------------------------
    # Eliminar dimensiones unitarias
    # --------------------------------------------------------

    if data.ndim == 3:

        if data.shape[0] == 1:
            data = data[0]

        elif data.shape[-1] == 1:
            data = data[:, :, 0]

    # --------------------------------------------------------
    # Comprobar que sea 2D
    # --------------------------------------------------------

    if data.ndim != 2:

        raise ValueError(
            f"Forma de ECG no válida: {data.shape}"
        )

    # --------------------------------------------------------
    # Convertir a (12, 5000)
    # --------------------------------------------------------

    if data.shape == (SIGNAL_LENGTH, NUM_LEADS):

        data = data.T

    elif data.shape == (NUM_LEADS, SIGNAL_LENGTH):

        pass

    else:

        raise ValueError(
            f"Se esperaba una señal de "
            f"(5000, 12) o (12, 5000), "
            f"pero se recibió {data.shape}"
        )

    # --------------------------------------------------------
    # Añadir dimensión batch
    # --------------------------------------------------------

    data = np.expand_dims(data, axis=0)

    return torch.from_numpy(data)


# ============================================================
# PREDICCIÓN
# ============================================================

@torch.no_grad()
def predict_ecg(model, data, device):
    """
    Realiza una predicción multilabel sobre un ECG.

    Entrada:
        data -> (1, 12, 5000)

    Salida:
        probabilities -> (27,)
    """

    data = prepare_signal_for_model(data)

    data = data.to(
        device=device,
        dtype=torch.float32
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    logits = model(data)

    # --------------------------------------------------------
    # Multilabel:
    #
    # NO usamos softmax.
    #
    # Cada diagnóstico tiene su propia probabilidad.
    # --------------------------------------------------------

    probabilities = torch.sigmoid(logits)

    probabilities = probabilities[0].cpu().numpy()

    return probabilities


# ============================================================
# GRÁFICO ECG COMPLETO
# ============================================================

def plot_ecg_12_leads(
    signal,
    record_name,
    output_dir
):
    """
    Genera un gráfico de las 12 derivaciones del ECG.
    """

    signal = np.asarray(signal)

    if signal.shape == (SIGNAL_LENGTH, NUM_LEADS):
        signal_plot = signal

    elif signal.shape == (NUM_LEADS, SIGNAL_LENGTH):
        signal_plot = signal.T

    else:
        raise ValueError(
            f"Forma inválida para gráfico: {signal.shape}"
        )

    num_samples = signal_plot.shape[0]

    time_axis = np.arange(num_samples) / SAMPLE_RATE

    fig, axes = plt.subplots(
        6,
        2,
        figsize=(18, 14),
        sharex=True
    )

    fig.suptitle(
        f"ECG de 12 derivaciones - {record_name}",
        fontsize=18
    )

    for i, ax in enumerate(axes.flat):

        if i >= NUM_LEADS:
            ax.axis('off')
            continue

        ax.plot(
            time_axis,
            signal_plot[:, i],
            linewidth=0.75
        )

        ax.set_title(
            LEAD_NAMES[i],
            fontsize=12
        )

        ax.set_ylabel("Amplitud")

        # ----------------------------------------------------
        # Rejilla principal
        # ----------------------------------------------------

        major_ticks_x = np.arange(
            0,
            time_axis[-1] + 0.2,
            0.2
        )

        ax.set_xticks(major_ticks_x)

        # ----------------------------------------------------
        # Rejilla secundaria
        # ----------------------------------------------------

        minor_ticks_x = np.arange(
            0,
            time_axis[-1] + 0.04,
            0.04
        )

        ax.set_xticks(
            minor_ticks_x,
            minor=True
        )

        ax.grid(
            which='major',
            linestyle='-',
            linewidth=0.5
        )

        ax.grid(
            which='minor',
            linestyle=':',
            linewidth=0.5
        )

        ax.set_xlim(
            0,
            time_axis[-1]
        )

    axes[-1, 0].set_xlabel(
        "Tiempo (s)"
    )

    axes[-1, 1].set_xlabel(
        "Tiempo (s)"
    )

    plt.tight_layout(
        rect=[0, 0.03, 1, 0.95]
    )

    path = os.path.join(
        output_dir,
        f'ECG_12_derivaciones_{record_name}.png'
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

    print(
        f"{Colors.BLUE}"
        f"Gráfico ECG guardado en: {path}"
        f"{Colors.ENDC}"
    )

    return path


# ============================================================
# TABLA DE RESULTADOS
# ============================================================

def print_prediction_table(
    probabilities,
    threshold=DEFAULT_THRESHOLD
):
    """
    Imprime las 27 probabilidades del modelo.
    """

    print(
        f"\n{Colors.HEADER}"
        "========== RESULTADOS CINC2020 =========="
        f"{Colors.ENDC}\n"
    )

    results = []

    for i, probability in enumerate(probabilities):

        class_name = (
            CLASSES[i]
            if i < len(CLASSES)
            else f"Clase_{i}"
        )

        results.append(
            (
                class_name,
                probability,
                probability >= threshold
            )
        )

    # Ordenar de mayor a menor probabilidad
    results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    print(
        f"{'Diagnóstico':<35}"
        f"{'Probabilidad':>15}"
        f"{'Estado':>15}"
    )

    print("-" * 65)

    for class_name, probability, positive in results:

        if positive:
            status = "DETECTADO"
        else:
            status = "No detectado"

        print(
            f"{class_name:<35}"
            f"{probability * 100:>14.2f}%"
            f"{status:>15}"
        )


# ============================================================
# DIAGNÓSTICOS DETECTADOS
# ============================================================

def get_detected_classes(
    probabilities,
    threshold=DEFAULT_THRESHOLD
):
    """
    Obtiene las clases cuya probabilidad supera el umbral.
    """

    detected = []

    for i, probability in enumerate(probabilities):

        if probability >= threshold:

            class_name = (
                CLASSES[i]
                if i < len(CLASSES)
                else f"Clase_{i}"
            )

            detected.append(
                {
                    'class': class_name,
                    'probability': float(probability)
                }
            )

    detected.sort(
        key=lambda x: x['probability'],
        reverse=True
    )

    return detected


# ============================================================
# RESUMEN
# ============================================================

def print_summary(
    probabilities,
    threshold=DEFAULT_THRESHOLD
):
    """
    Imprime un resumen clínico orientativo de las clases
    detectadas.

    IMPORTANTE:
    Las predicciones son resultados de un modelo de IA y
    no constituyen un diagnóstico médico.
    """

    detected = get_detected_classes(
        probabilities,
        threshold
    )

    print(
        f"\n{Colors.HEADER}"
        "========== RESUMEN =========="
        f"{Colors.ENDC}\n"
    )

    if len(detected) == 0:

        print(
            f"{Colors.WARNING}"
            "No se superó el umbral de detección "
            "para ninguna de las 27 clases."
            f"{Colors.ENDC}"
        )

    else:

        print(
            f"{Colors.GREEN}"
            "Diagnósticos/clases detectados:"
            f"{Colors.ENDC}"
        )

        for item in detected:

            print(
                f"  • {item['class']}: "
                f"{item['probability'] * 100:.2f}%"
            )

    print(
        f"\n{Colors.WARNING}"
        "NOTA: estos resultados corresponden a la "
        "predicción del modelo y no constituyen por sí "
        "solos un diagnóstico médico."
        f"{Colors.ENDC}"
    )


# ============================================================
# MAIN
# ============================================================

def main(config):

    print(
        f"\n{Colors.HEADER}"
        "=========================================="
        "\n        PREDICCIÓN ECG - CINC2020"
        "\n=========================================="
        f"{Colors.ENDC}"
    )

    # --------------------------------------------------------
    # Dispositivo
    # --------------------------------------------------------

    device = get_device()

    # --------------------------------------------------------
    # Modelo
    # --------------------------------------------------------

    try:

        model = load_trained_model(
            config,
            device
        )

    except Exception as e:

        print(
            f"{Colors.FAIL}"
            f"Error al cargar el modelo:\n{e}"
            f"{Colors.ENDC}"
        )

        return

    # --------------------------------------------------------
    # Datos
    # --------------------------------------------------------

    if config.upload:

        print(
            f"{Colors.BLUE}"
            "Modo de carga de ECG externo."
            f"{Colors.ENDC}"
        )

        # Esta parte dependerá del formato que definamos
        # finalmente para uploadedData().
        data = uploadedData()

        record_name = "ECG_subido"

    else:

        print(
            f"{Colors.BLUE}"
            "Modo CINC2020."
            f"{Colors.ENDC}"
        )

        # ----------------------------------------------------
        # Aquí se utilizará el cargador CINC2020 que
        # definiremos en utils.py.
        #
        # No existe ya REFERENCE.csv de CINC2017.
        # ----------------------------------------------------

        if hasattr(config, 'record'):

            record_name = config.record

        else:

            record_name = "registro_CINC2020"

        raise NotImplementedError(
            "La carga de registros CINC2020 debe "
            "implementarse en utils.py/data2020.py. "
            "No se utilizará cincData() de CINC2017."
        )

    # --------------------------------------------------------
    # Preprocesamiento
    # --------------------------------------------------------

    print(
        f"{Colors.BLUE}"
        "Preprocesando señal ECG..."
        f"{Colors.ENDC}"
    )

    processed_data, _ = preprocess(
        data,
        config
    )

    if processed_data is None:

        print(
            f"{Colors.FAIL}"
            "El preprocesamiento falló."
            f"{Colors.ENDC}"
        )

        return

    # --------------------------------------------------------
    # Directorio de resultados
    # --------------------------------------------------------

    output_dir = os.path.join(
        'resultados',
        record_name
    )

    mkdir_recursive(
        output_dir
    )

    # --------------------------------------------------------
    # Gráfico de ECG
    # --------------------------------------------------------

    try:

        plot_ecg_12_leads(
            data,
            record_name,
            output_dir
        )

    except Exception as e:

        print(
            f"{Colors.WARNING}"
            f"No se pudo generar el gráfico ECG: {e}"
            f"{Colors.ENDC}"
        )

    # --------------------------------------------------------
    # Predicción
    # --------------------------------------------------------

    print(
        f"\n{Colors.BLUE}"
        "Realizando predicción..."
        f"{Colors.ENDC}"
    )

    probabilities = predict_ecg(
        model,
        processed_data,
        device
    )

    # --------------------------------------------------------
    # Resultados
    # --------------------------------------------------------

    print_prediction_table(
        probabilities,
        DEFAULT_THRESHOLD
    )

    print_summary(
        probabilities,
        DEFAULT_THRESHOLD
    )


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == '__main__':

    config = get_config()

    main(config)

# ============================================================
# ADAPTADOR PARA FLASK (APP.PY)
# ============================================================

def predict_and_summarize(
    data_processed,
    original_label=None,
    config=None,
    record_name_base="registro_ecg",
    raw_multilead_signal=None,
    lead_names=None,
    threshold=DEFAULT_THRESHOLD
):
    """
    Función de enlace para app.py. Ejecuta la inferencia del modelo,
    genera gráficos del ECG y retorna el diccionario de resumen completo.
    """
    device = get_device()
    model = load_trained_model(config, device)
    
    # 1. Obtener probabilidades (27 clases)
    probabilities = predict_ecg(model, data_processed, device)
    
    # 2. Generar gráfico del ECG si la señal está disponible
    full_ecg_plot_path = None
    if raw_multilead_signal is not None:
        try:
            output_dir = os.path.join('resultados', record_name_base)
            mkdir_recursive(output_dir)
            full_ecg_plot_path = plot_ecg_12_leads(
                raw_multilead_signal,
                record_name_base,
                output_dir
            )
        except Exception as e:
            print(f"{Colors.WARNING}Error al generar gráfico para Flask: {e}{Colors.ENDC}")

    # 3. Formatear y ordenar diagnósticos por probabilidad descendente
    detected_list = get_detected_classes(probabilities, threshold=threshold)
    
    predictions_detailed = []
    for i, prob in enumerate(probabilities):
        class_name = CLASSES[i] if i < len(CLASSES) else f"Clase_{i}"
        predictions_detailed.append({
            "class": class_name,
            "probability": float(prob),
            "all_probs_array": probabilities
        })
    
    # Ordenar por mayor probabilidad
    sorted_classes = sorted(
        predictions_detailed,
        key=lambda x: x["probability"],
        reverse=True
    )
    
    # 4. Estructurar diccionario final para Flask
    summary_data = {
        "predictions_detailed": sorted_classes,
        "summary": {item['class']: item['probability'] for item in detected_list},
        "most_probable_class": sorted_classes[0]["class"] if sorted_classes else "Ninguna",
        "most_probable_certainty": f"{sorted_classes[0]['probability'] * 100:.2f}%" if sorted_classes else "0%",
        "second_probable_class": sorted_classes[1]["class"] if len(sorted_classes) > 1 else "N/A",
        "second_probable_certainty": f"{sorted_classes[1]['probability'] * 100:.2f}%" if len(sorted_classes) > 1 else "0%",
        "third_probable_class": sorted_classes[2]["class"] if len(sorted_classes) > 2 else "N/A",
        "third_probable_certainty": f"{sorted_classes[2]['probability'] * 100:.2f}%" if len(sorted_classes) > 2 else "0%",
        "original_label": original_label or "Desconocida",
        "average_probabilities": {item['class']: float(item['probability']) for item in sorted_classes},
        "full_ecg_plot_path": full_ecg_plot_path,
        "ecg_lectura_plot_path": full_ecg_plot_path,
        "ecg_lectura_reducido_plot_path": full_ecg_plot_path,
        "cardiac_condition_suggestion": "Sugerencia del modelo basada en threshold multilabel."
    }
    
    return summary_data