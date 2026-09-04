# coding=utf-8

from __future__ import division, print_function

import os
import re
import shutil
import json

import numpy as np
import pandas as pd
import wfdb

from flask import (
    Flask,
    request,
    render_template,
    jsonify,
    send_from_directory,
    send_file,
    url_for,
)
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

from predict import predict_and_summarize
from utils import uploadedData, preprocess, mkdir_recursive
from config import get_config


# ============================================================
# CONFIGURACIÓN
# ============================================================

SEED = 42
np.random.seed(SEED)

app = Flask(__name__)

# Tamaño máximo del archivo subido
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

# CINC2020:
# 12 derivaciones x 5000 muestras = 10 segundos a 500 Hz
CINC2020_SAMPLE_RATE = 500
CINC2020_INPUT_LENGTH = 5000
CINC2020_NUM_LEADS = 12
CINC2020_NUM_CLASSES = 27

STANDARD_LEADS = [
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]

# Umbral inicial para convertir probabilidades en predicciones
# binarias. Posteriormente puede optimizarse por clase.
DEFAULT_THRESHOLD = 0.5

# Directorios
app.config["RESULTS_FOLDER"] = "resultados"

app.config["UPLOAD_FOLDER"] = os.path.join(
    os.path.dirname(__file__),
    "uploads"
)

app.config["ASSET_FOLDER"] = "asset"


# Crear carpetas
mkdir_recursive(app.config["RESULTS_FOLDER"])

mkdir_recursive(
    os.path.join(
        os.path.dirname(__file__),
        "static",
        app.config["ASSET_FOLDER"]
    )
)

mkdir_recursive(app.config["UPLOAD_FOLDER"])


# ============================================================
# SERVIR RESULTADOS
# ============================================================

app.add_url_rule(
    f'/{app.config["RESULTS_FOLDER"]}/<path:filename>',
    endpoint="results_static",
    view_func=lambda filename: send_from_directory(
        app.config["RESULTS_FOLDER"],
        filename
    ),
)


print("Abrir http://localhost:5002/")


# ============================================================
# PREDICCIÓN DE UN ECG
# ============================================================

def model_predict(
    ecg_path,
    original_label=None,
    patient_name="Desconocido",
    patient_age="Desconocido",
    patient_sex="Desconocido",
):
    """
    Procesa un ECG para CINC2020 y ejecuta la predicción mediante PyTorch.

    Flujo:

        archivo CSV
            ↓
        12 derivaciones
            ↓
        remuestreo a 500 Hz
            ↓
        5000 muestras
            ↓
        tensor (1, 12, 5000)
            ↓
        modelo PyTorch
            ↓
        27 probabilidades
            ↓
        predicciones multilabel

    No utiliza:
        - CINC2017
        - MIT-BIH
        - MLII
        - detección de QRS
        - segmentación por latidos
        - Softmax
        - clasificación de 6 clases
    """

    print("=" * 70)
    print("INICIO DE PREDICCIÓN CINC2020")
    print("=" * 70)

    print(f"Archivo: {ecg_path}")

    # --------------------------------------------------------
    # 1. Leer CSV
    # --------------------------------------------------------

    sampling_rate, data_signal = uploadedData(
        ecg_path,
        csvbool=True,
        return_lead_names=True,
    )

    signal_data, lead_names = data_signal

    print(f"Sampling rate original: {sampling_rate} Hz")
    print(f"Shape original: {signal_data.shape}")
    print(f"Derivaciones: {lead_names}")

    if signal_data is None or signal_data.size == 0:
        raise ValueError(
            "No se encontraron datos de ECG válidos en el archivo."
        )

    if signal_data.ndim != 2:
        raise ValueError(
            f"Se esperaba una señal multiderivación 2D. "
            f"Shape recibido: {signal_data.shape}"
        )

    # --------------------------------------------------------
    # 2. Validar derivaciones
    # --------------------------------------------------------

    normalized_leads = [
        str(lead).strip().upper()
        for lead in lead_names
    ]

    expected_leads_upper = [
        lead.upper()
        for lead in STANDARD_LEADS
    ]

    missing_leads = [
        lead
        for lead in expected_leads_upper
        if lead not in normalized_leads
    ]

    if missing_leads:
        raise ValueError(
            "El ECG no contiene las 12 derivaciones requeridas por "
            f"el modelo CINC2020. Faltan: {missing_leads}"
        )

    # --------------------------------------------------------
    # 3. Configuración
    # --------------------------------------------------------

    config = get_config()

    # IMPORTANTE:
    # sample_rate aquí es el sampling rate OBJETIVO del modelo,
    # no el sampling rate original del CSV.
    config.sample_rate = CINC2020_SAMPLE_RATE

    # Nuestras señales tienen siempre 10 segundos
    config.input_length = CINC2020_INPUT_LENGTH

    # Número de canales
    config.num_leads = CINC2020_NUM_LEADS

    # Número de diagnósticos
    config.num_classes = CINC2020_NUM_CLASSES

    # --------------------------------------------------------
    # 4. Preprocesamiento CINC2020
    # --------------------------------------------------------

    data_processed = preprocess(
        signal_data,
        sampling_rate=sampling_rate,
        target_fs=CINC2020_SAMPLE_RATE,
        target_length=CINC2020_INPUT_LENGTH,
        lead_names=lead_names,
    )

    print(
        f"Shape después de preprocess: "
        f"{data_processed.shape}"
    )

    expected_shape = (
        1,
        CINC2020_NUM_LEADS,
        CINC2020_INPUT_LENGTH,
    )

    if data_processed.shape != expected_shape:
        raise ValueError(
            f"El preprocesamiento produjo una forma incorrecta. "
            f"Esperado: {expected_shape}, "
            f"obtenido: {data_processed.shape}"
        )

    # --------------------------------------------------------
    # 5. Nombre del registro
    # --------------------------------------------------------

    record_name_base = os.path.splitext(
        os.path.basename(ecg_path)
    )[0]

    # --------------------------------------------------------
    # 6. Predicción PyTorch
    # --------------------------------------------------------

    prediction_summary_data = predict_and_summarize(
        data_processed=data_processed,
        original_label=original_label,
        config=config,
        record_name_base=record_name_base,
        raw_multilead_signal=signal_data,
        lead_names=lead_names,
        threshold=DEFAULT_THRESHOLD,
    )

    # --------------------------------------------------------
    # 7. Información del paciente
    # --------------------------------------------------------

    prediction_summary_data["patient_name"] = patient_name
    prediction_summary_data["patient_age"] = patient_age
    prediction_summary_data["patient_sex"] = patient_sex

    print("=" * 70)
    print("FIN DE PREDICCIÓN CINC2020")
    print("=" * 70)

    return prediction_summary_data


# ============================================================
# PÁGINA PRINCIPAL
# ============================================================

@app.route("/", methods=["GET"])
def index():
    """
    Página principal de la aplicación.
    """

    config = get_config()

    # --------------------------------------------------------
    # Información del modelo CINC2020
    # --------------------------------------------------------

    model_info = {

        "architecture_type":
            "Red neuronal convolucional 1D (CNN) implementada "
            "en PyTorch para clasificación multilabel de ECG de "
            "12 derivaciones.",

        "architecture_details": [

            "La entrada del modelo contiene un ECG de "
            "12 derivaciones.",

            "Cada registro se procesa a una frecuencia de "
            "muestreo de 500 Hz.",

            "Cada ECG se representa mediante 5000 muestras, "
            "correspondientes a 10 segundos de señal.",

            "La red utiliza capas convolucionales 1D para "
            "extraer características morfológicas y temporales "
            "del ECG.",

            "La capa de salida contiene 27 neuronas independientes, "
            "una por cada clase diagnóstica evaluada en "
            "PhysioNet/CinC Challenge 2020.",

            "La salida utiliza probabilidades independientes "
            "mediante funciones sigmoid, ya que un ECG puede "
            "presentar más de un diagnóstico simultáneamente.",
        ],

        "optimizer":
            "Optimizador: Adam. La tasa de aprendizaje definitiva "
            "debe corresponder a la configuración utilizada durante "
            "el entrenamiento del modelo.",

        "loss_function":
            "Función de pérdida: Binary Cross Entropy with Logits "
            "(BCEWithLogitsLoss), adecuada para clasificación "
            "multilabel.",

        "metrics_monitored":
            "Métricas: F1 macro, F1 micro, AUROC, AUPRC, "
            "sensibilidad, especificidad y precisión.",

        "dataset_source":
            "PhysioNet/CinC Challenge 2020.",

        "dataset_total_records":
            "Conjunto de registros de ECG de múltiples bases de "
            "datos participantes en PhysioNet/CinC Challenge 2020. "
            "El número definitivo depende del conjunto utilizado "
            "durante el entrenamiento.",

        "dataset_sampling_rate":
            "500 Hz después del remuestreo.",

        "dataset_annotations":
            "Etiquetas diagnósticas multilabel basadas en códigos "
            "SNOMED-CT y el conjunto de clases puntuadas de "
            "CinC2020.",

        "dataset_lead":
            "ECG de 12 derivaciones: I, II, III, aVR, aVL, aVF, "
            "V1, V2, V3, V4, V5 y V6.",

        "dataset_balancing":
            "El tratamiento del desequilibrio de clases se realiza "
            "durante el entrenamiento mediante las estrategias "
            "definidas en el pipeline de entrenamiento. No se "
            "incorpora CINC2017.",

        "model_classes": [],

        "metrics_accuracy":
            "N/A",

        "metrics_f1_score":
            "N/A",

        "performance_notes": [

            "El problema se formula como clasificación multilabel: "
            "un mismo ECG puede recibir múltiples diagnósticos.",

            "Las probabilidades de las 27 clases se calculan "
            "independientemente mediante sigmoid.",

            "El umbral inicial de decisión es 0.5 y puede "
            "optimizarse posteriormente por clase.",

            "Las métricas definitivas deben cargarse desde los "
            "resultados reales del entrenamiento y evaluación."
        ],

        "preprocessing_notes": [

            "Los ECG se convierten a una frecuencia común de "
            "500 Hz.",

            "Cada registro se adapta a una duración de 10 segundos "
            "(5000 muestras).",

            "Las 12 derivaciones se reorganizan al orden estándar "
            "del ECG.",

            "No se realiza segmentación por latidos ni detección "
            "de picos QRS para la entrada principal del modelo.",

            "El preprocesamiento utilizado durante inferencia debe "
            "ser idéntico al empleado durante el entrenamiento."
        ],

        "prediction_notes": [

            "El modelo genera 27 probabilidades independientes.",

            "No se utiliza Softmax ni argmax como mecanismo principal "
            "de clasificación.",

            "Se pueden detectar simultáneamente múltiples "
            "condiciones cardíacas.",

            "Las predicciones se obtienen aplicando un umbral a "
            "cada probabilidad."
        ],
    }

    return render_template(
        "index.html",
        model_info=model_info,
    )


# ============================================================
# API DE PREDICCIÓN
# ============================================================

@app.route("/predict", methods=["GET", "POST"])
def upload():
    """
    Recibe un archivo CSV de ECG y devuelve las predicciones
    multilabel de CINC2020 en formato JSON.
    """

    if request.method == "GET":
        return jsonify({
            "status": "ok",
            "message": "Endpoint de predicción CINC2020",
        })

    # --------------------------------------------------------
    # 1. Validar archivo
    # --------------------------------------------------------

    if "file" not in request.files:
        return jsonify({
            "error": "No se recibió ningún archivo."
        }), 400

    uploaded_file = request.files["file"]

    if uploaded_file.filename == "":
        return jsonify({
            "error": "No se seleccionó ningún archivo."
        }), 400

    # --------------------------------------------------------
    # 2. Guardar archivo
    # --------------------------------------------------------

    mkdir_recursive(app.config["UPLOAD_FOLDER"])

    filename = secure_filename(uploaded_file.filename)

    if not filename:
        return jsonify({
            "error": "Nombre de archivo no válido."
        }), 400

    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    uploaded_file.save(file_path)

    # --------------------------------------------------------
    # 3. Buscar metadata
    # --------------------------------------------------------

    uploaded_filename_base = os.path.splitext(
        os.path.basename(file_path)
    )[0]

    cleaned_filename_base = re.sub(
        r"\(\d+\)$|_\d+$",
        "",
        uploaded_filename_base,
    )

    json_base_name = cleaned_filename_base

    if json_base_name.endswith("_Completo"):
        json_base_name = json_base_name[:-len("_Completo")]

    metadata_file_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        f"{json_base_name}.json"
    )

    extracted_original_label = "Desconocida"
    extracted_patient_name = "Desconocido"
    extracted_patient_age = "Desconocido"
    extracted_patient_sex = "Desconocido"

    if os.path.exists(metadata_file_path):

        try:

            with open(
                metadata_file_path,
                "r",
                encoding="utf-8"
            ) as metadata_file:

                metadata = json.load(metadata_file)

            extracted_original_label = metadata.get(
                "original_label",
                "Desconocida"
            )

            extracted_patient_name = metadata.get(
                "patient_name",
                "Desconocido"
            )

            extracted_patient_age = metadata.get(
                "patient_age",
                "Desconocido"
            )

            extracted_patient_sex = metadata.get(
                "patient_sex",
                "Desconocido"
            )

        except (json.JSONDecodeError, OSError) as error:

            print(
                f"Advertencia: no se pudo leer metadata: {error}"
            )

    # --------------------------------------------------------
    # 4. Nombres predeterminados
    # --------------------------------------------------------

    if extracted_patient_name == "Desconocido":

        try:

            age = int(extracted_patient_age)

        except (ValueError, TypeError):

            age = None

        if age is not None and age < 18:

            if extracted_patient_sex == "M":
                extracted_patient_name = "Johnie Doe"

            elif extracted_patient_sex == "F":
                extracted_patient_name = "Janie Doe"

            else:
                extracted_patient_name = "Baby Doe"

        else:

            if extracted_patient_sex == "M":
                extracted_patient_name = "John Doe"

            elif extracted_patient_sex == "F":
                extracted_patient_name = "Jane Doe"

            else:
                extracted_patient_name = "A. N. Other"

    # --------------------------------------------------------
    # 5. Predicción
    # --------------------------------------------------------

    try:

        prediction_summary_data = model_predict(
            file_path,
            original_label=extracted_original_label,
            patient_name=extracted_patient_name,
            patient_age=extracted_patient_age,
            patient_sex=extracted_patient_sex,
        )

    except Exception as error:

        print(
            f"Error durante la predicción: {error}"
        )

        try:
            os.remove(file_path)
        except OSError:
            pass

        return jsonify({
            "error": str(error)
        }), 500

    # --------------------------------------------------------
    # 6. Preparar respuesta JSON
    # --------------------------------------------------------

    predictions_for_json = []

    predictions_detailed = prediction_summary_data.get(
        "predictions_detailed",
        []
    )

    for prediction in predictions_detailed:

        probabilities = prediction.get(
            "all_probs_array",
            []
        )

        probabilities_formatted = [
            f"{float(probability) * 100:.2f}%"
            for probability in probabilities
        ]

        predictions_for_json.append({

            "class": prediction.get(
                "class"
            ),

            "probability": prediction.get(
                "probability"
            ),

            "all_probs_string":
                " | ".join(probabilities_formatted),

        })

    # --------------------------------------------------------
    # 7. URLs de gráficos
    # --------------------------------------------------------

    def get_relative_path(full_path):

        if not full_path:
            return None

        results_folder = app.config["RESULTS_FOLDER"]

        full_path = os.path.normpath(full_path)
        results_folder = os.path.normpath(results_folder)

        try:

            return os.path.relpath(
                full_path,
                results_folder
            )

        except ValueError:

            return full_path

    full_ecg_plot_url = None

    if prediction_summary_data.get(
        "full_ecg_plot_path"
    ):

        full_ecg_plot_url = url_for(
            "results_static",
            filename=get_relative_path(
                prediction_summary_data[
                    "full_ecg_plot_path"
                ]
            )
        )

    ecg_lectura_plot_url = None

    if prediction_summary_data.get(
        "ecg_lectura_plot_path"
    ):

        ecg_lectura_plot_url = url_for(
            "results_static",
            filename=get_relative_path(
                prediction_summary_data[
                    "ecg_lectura_plot_path"
                ]
            )
        )

    ecg_lectura_reducido_plot_url = None

    if prediction_summary_data.get(
        "ecg_lectura_reducido_plot_path"
    ):

        ecg_lectura_reducido_plot_url = url_for(
            "results_static",
            filename=get_relative_path(
                prediction_summary_data[
                    "ecg_lectura_reducido_plot_path"
                ]
            )
        )

    # --------------------------------------------------------
    # 8. Predicciones multilabel
    # --------------------------------------------------------

    average_probabilities = prediction_summary_data.get(
        "average_probabilities",
        {}
    )

    # --------------------------------------------------------
    # 9. Respuesta
    # --------------------------------------------------------

    response_data = {

        "total_parts":
            len(predictions_for_json),

        "predictions":
            predictions_for_json,

        "summary":
            prediction_summary_data.get(
                "summary",
                {}
            ),

        "most_probable_class":
            prediction_summary_data.get(
                "most_probable_class"
            ),

        "most_probable_certainty":
            prediction_summary_data.get(
                "most_probable_certainty"
            ),

        "second_probable_class":
            prediction_summary_data.get(
                "second_probable_class"
            ),

        "second_probable_certainty":
            prediction_summary_data.get(
                "second_probable_certainty"
            ),

        "third_probable_class":
            prediction_summary_data.get(
                "third_probable_class"
            ),

        "third_probable_certainty":
            prediction_summary_data.get(
                "third_probable_certainty"
            ),

        "original_label":
            prediction_summary_data.get(
                "original_label"
            ),

        "average_probabilities":
            average_probabilities,

        "full_ecg_plot_url":
            full_ecg_plot_url,

        "ecg_lectura_plot_url":
            ecg_lectura_plot_url,

        "ecg_lectura_reducido_plot_url":
            ecg_lectura_reducido_plot_url,

        "segment_plot_urls":
            [],

        "cardiac_condition_suggestion":
            prediction_summary_data.get(
                "cardiac_condition_suggestion"
            ),

        "patient_name":
            prediction_summary_data.get(
                "patient_name"
            ),

        "patient_age":
            prediction_summary_data.get(
                "patient_age"
            ),

        "patient_sex":
            prediction_summary_data.get(
                "patient_sex"
            ),

        "num_classes":
            CINC2020_NUM_CLASSES,

        "num_leads":
            CINC2020_NUM_LEADS,

        "sampling_rate":
            CINC2020_SAMPLE_RATE,

        "input_length":
            CINC2020_INPUT_LENGTH,
    }

    # --------------------------------------------------------
    # 10. Eliminar CSV temporal
    # --------------------------------------------------------

    try:

        if os.path.exists(file_path):
            os.remove(file_path)

    except OSError as error:

        print(
            f"Error eliminando archivo temporal: {error}"
        )

    return jsonify(response_data)


# ============================================================
# DESCARGAR RESULTADOS
# ============================================================

@app.route(
    "/download_ecg_results/<record_name_base>",
    methods=["GET"]
)
def download_ecg_results(record_name_base):
    """
    Comprime la carpeta de resultados de un registro y
    permite descargarla.
    """

    record_results_folder = os.path.join(
        app.config["RESULTS_FOLDER"],
        secure_filename(record_name_base)
    )

    if not os.path.exists(record_results_folder):
        return (
            "Carpeta de resultados no encontrada.",
            404
        )

    zip_base_name = os.path.join(
        app.config["RESULTS_FOLDER"],
        f"{secure_filename(record_name_base)}_Results"
    )

    try:

        zip_path = shutil.make_archive(
            zip_base_name,
            "zip",
            record_results_folder
        )

        return send_file(
            zip_path,
            as_attachment=True,
            download_name=(
                f"{secure_filename(record_name_base)}_Resultados.zip"
            )
        )

    except Exception as error:

        print(
            f"Error al crear/enviar ZIP: {error}"
        )

        return (
            "Error al descargar el archivo.",
            500
        )

    finally:

        try:

            if os.path.exists(
                zip_base_name + ".zip"
            ):
                os.remove(
                    zip_base_name + ".zip"
                )

        except OSError as error:

            print(
                f"Error eliminando ZIP temporal: {error}"
            )


# ============================================================
# VALIDACIÓN DE ARCHIVOS WFDB
# ============================================================

def _validate_and_prepare_wfdb_files(
    uploaded_files,
    temp_wfdb_dir
):
    """
    Valida y guarda archivos WFDB.

    Se acepta:

        record.hea + record.mat

    o:

        record.hea + record.dat
    """

    file_base_name = None

    hea_file_found = False
    mat_file_found = False
    dat_file_found = False

    for uploaded_file in uploaded_files:

        if uploaded_file.filename == "":
            continue

        original_filename = secure_filename(
            uploaded_file.filename
        )

        extension = os.path.splitext(
            original_filename
        )[1].lower()

        current_base_name = os.path.splitext(
            original_filename
        )[0]

        if file_base_name is None:

            file_base_name = current_base_name

        elif file_base_name != current_base_name:

            raise ValueError(
                "Todos los archivos WFDB deben tener "
                "el mismo nombre base."
            )

        temp_file_path = os.path.join(
            temp_wfdb_dir,
            original_filename
        )

        uploaded_file.save(temp_file_path)

        if extension == ".hea":
            hea_file_found = True

        elif extension == ".mat":
            mat_file_found = True

        elif extension == ".dat":
            dat_file_found = True

    return (
        file_base_name,
        hea_file_found,
        mat_file_found,
        dat_file_found,
    )


# ============================================================
# FORMATEAR WFDB → CSV MULTIDERIVACIÓN
# ============================================================

@app.route("/format_ecg_full", methods=["POST"])
def format_ecg_full_to_csv():
    """
    Convierte un registro WFDB a CSV conservando todas
    las derivaciones.

    Formato producido:

        # Sampling Rate: 500

        I,II,III,aVR,aVL,aVF,V1,...,V6
        ...
    """

    uploaded_files = request.files.getlist(
        "ecg_file_to_format"
    )

    if not uploaded_files:

        return jsonify({
            "error":
                "No se encontraron archivos en la solicitud."
        }), 400

    upload_dir = app.config["UPLOAD_FOLDER"]

    mkdir_recursive(upload_dir)

    temp_dir_name = (
        f"temp_wfdb_{os.urandom(8).hex()}"
    )

    temp_wfdb_dir = os.path.join(
        upload_dir,
        temp_dir_name
    )

    mkdir_recursive(temp_wfdb_dir)

    csv_file_path = None

    try:

        # ----------------------------------------------------
        # Validar archivos
        # ----------------------------------------------------

        (
            file_base_name,
            hea_file_found,
            mat_file_found,
            dat_file_found,
        ) = _validate_and_prepare_wfdb_files(
            uploaded_files,
            temp_wfdb_dir
        )

        if (
            not hea_file_found
            or not (mat_file_found or dat_file_found)
        ):

            return jsonify({
                "error":
                    "Se requiere un archivo .hea y un "
                    "archivo .mat o .dat con el mismo "
                    "nombre base."
            }), 400

        # ----------------------------------------------------
        # Leer WFDB
        # ----------------------------------------------------

        record_name = os.path.splitext(
            file_base_name
        )[0]

        record = wfdb.rdrecord(
            os.path.join(
                temp_wfdb_dir,
                record_name
            )
        )

        signal_data = record.p_signal

        if signal_data is None:

            raise ValueError(
                "No se encontraron datos de ECG."
            )

        sampling_rate = getattr(
            record,
            "fs",
            None
        )

        signal_names = getattr(
            record,
            "sig_name",
            None
        )

        if signal_data.ndim != 2:

            raise ValueError(
                "El registro WFDB no contiene una "
                "señal multiderivación válida."
            )

        # ----------------------------------------------------
        # Nombres de derivaciones
        # ----------------------------------------------------

        if signal_names:

            columns = list(signal_names)

        else:

            columns = [
                f"ECG_Lead_{i + 1}"
                for i in range(signal_data.shape[1])
            ]

        # ----------------------------------------------------
        # DataFrame
        # ----------------------------------------------------

        df = pd.DataFrame(
            signal_data,
            columns=columns
        )

        # ----------------------------------------------------
        # Nombre CSV
        # ----------------------------------------------------

        csv_filename = (
            f"{file_base_name}_Completo.csv"
        )

        csv_file_path = os.path.join(
            upload_dir,
            csv_filename
        )

        # ----------------------------------------------------
        # Guardar CSV
        # ----------------------------------------------------

        with open(
            csv_file_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as csv_file:

            if sampling_rate is not None:

                csv_file.write(
                    f"# Sampling Rate: "
                    f"{sampling_rate}\n"
                )

            df.to_csv(
                csv_file,
                index=False
            )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        (
            original_label,
            patient_name,
            patient_age,
            patient_sex,
        ) = _extract_patient_metadata(record)

        metadata_path = os.path.join(
            upload_dir,
            f"{file_base_name}.json"
        )

        _save_metadata_json(
            metadata_path,
            original_label,
            patient_name,
            patient_age,
            patient_sex,
        )

        return send_file(
            csv_file_path,
            as_attachment=True,
            download_name=csv_filename,
        )

    except Exception as error:

        print(
            f"Error al formatear WFDB: {error}"
        )

        return jsonify({
            "error":
                f"Error al procesar el ECG: {str(error)}"
        }), 500

    finally:

        _cleanup_temp_files(
            temp_wfdb_dir,
            csv_file_path
        )


# ============================================================
# COMPATIBILIDAD: /format_ecg
# ============================================================

@app.route("/format_ecg", methods=["POST"])
def format_ecg_to_csv():
    """
    Alias compatible con el endpoint antiguo.

    Para CINC2020 se recomienda utilizar /format_ecg_full,
    porque el modelo necesita las 12 derivaciones.
    """

    return format_ecg_full_to_csv()


# ============================================================
# METADATOS DEL PACIENTE
# ============================================================

def _extract_patient_metadata(record):
    """
    Extrae información disponible en los comentarios WFDB.

    Devuelve:

        original_label
        patient_name
        patient_age
        patient_sex
    """

    original_label = "Desconocida"
    patient_name = "Desconocido"
    patient_age = "Desconocido"
    patient_sex = "Desconocido"

    comments = getattr(
        record,
        "comments",
        []
    )

    for comment in comments:

        comment = str(comment)

        # ----------------------------------------------------
        # Diagnóstico
        # ----------------------------------------------------

        diagnosis_match = re.search(
            r"Diagnosis:\s*(.+)",
            comment,
            flags=re.IGNORECASE
        )

        if diagnosis_match:

            diagnosis = (
                diagnosis_match.group(1)
                .strip()
            )

            if diagnosis:
                original_label = diagnosis

        # ----------------------------------------------------
        # Edad
        # ----------------------------------------------------

        age_match = re.search(
            r"Age:\s*(\d+)",
            comment,
            flags=re.IGNORECASE
        )

        if age_match:

            patient_age = age_match.group(1)

        # ----------------------------------------------------
        # Sexo
        # ----------------------------------------------------

        sex_match = re.search(
            r"Sex:\s*(.+)",
            comment,
            flags=re.IGNORECASE
        )

        if sex_match:

            sex = (
                sex_match.group(1)
                .strip()
                .lower()
            )

            if sex in ("male", "m", "man"):
                patient_sex = "M"

            elif sex in ("female", "f", "woman"):
                patient_sex = "F"

    # --------------------------------------------------------
    # Nombre predeterminado
    # --------------------------------------------------------

    if patient_name == "Desconocido":

        try:
            age = int(patient_age)
        except (ValueError, TypeError):
            age = None

        if age is not None and age < 18:

            if patient_sex == "M":
                patient_name = "Johnie Doe"

            elif patient_sex == "F":
                patient_name = "Janie Doe"

            else:
                patient_name = "Baby Doe"

        else:

            if patient_sex == "M":
                patient_name = "John Doe"

            elif patient_sex == "F":
                patient_name = "Jane Doe"

            else:
                patient_name = "A. N. Other"

    return (
        original_label,
        patient_name,
        patient_age,
        patient_sex,
    )


# ============================================================
# GUARDAR METADATA
# ============================================================

def _save_metadata_json(
    metadata_path,
    original_label,
    patient_name,
    patient_age,
    patient_sex
):
    """
    Guarda los metadatos asociados al ECG.
    """

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as metadata_file:

        json.dump(
            {
                "original_label": original_label,
                "patient_name": patient_name,
                "patient_age": patient_age,
                "patient_sex": patient_sex,
            },
            metadata_file,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# LIMPIEZA
# ============================================================

def _cleanup_temp_files(
    temp_wfdb_dir,
    csv_file_path=None
):
    """
    Elimina archivos temporales.
    """

    try:

        if (
            temp_wfdb_dir
            and os.path.exists(temp_wfdb_dir)
        ):

            shutil.rmtree(
                temp_wfdb_dir
            )

        if (
            csv_file_path
            and os.path.exists(csv_file_path)
        ):

            os.remove(
                csv_file_path
            )

    except OSError as error:

        print(
            f"Error limpiando temporales: {error}"
        )


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":

    http_server = WSGIServer(
        ("0.0.0.0", 5002),
        app
    )

    http_server.serve_forever()