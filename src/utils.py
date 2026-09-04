from __future__ import division, print_function

import os
import hashlib
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

from scipy.signal import resample, find_peaks
from sklearn import preprocessing
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
    classification_report,
    accuracy_score,
    average_precision_score,
)


# ============================================================
# COLORES PARA CONSOLA
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
    WHITE = '\033[97m'
    DARK_GRAY = '\033[90m'


# ============================================================
# DIRECTORIOS
# ============================================================

def mkdir_recursive(path):
    """
    Crea un directorio y sus directorios padre si no existen.

    Args:
        path (str): Ruta del directorio.
    """
    if not path:
        return

    os.makedirs(path, exist_ok=True)


# ============================================================
# CARGA DE DATOS CINC2020
# ============================================================

def loaddata(input_size=None, feature=None):
    """
    Carga los datos de CINC2020 desde los archivos HDF5
    generados por data2020.py.

    Esta función reemplaza completamente la antigua carga
    de datasets CINC2017.

    Los archivos esperados son:

        dataset_cinc2020/
            train.h5
            val.h5
            test.h5

    Cada archivo contiene:

        signals
        labels
        ages
        sexes
        source_dbs
        record_names

    Args:
        input_size:
            Se mantiene por compatibilidad con código antiguo.
            CINC2020 ya tiene señales de tamaño fijo.

        feature:
            Se mantiene por compatibilidad.
            Ya no se utiliza porque CINC2020 trabaja con
            las 12 derivaciones.

    Returns:
        tuple:
            X_train, y_train, X_val, y_val
    """

    dataset_dir = 'dataset_cinc2020'

    train_path = os.path.join(dataset_dir, 'train.h5')
    val_path = os.path.join(dataset_dir, 'val.h5')

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"No se encontró el archivo de entrenamiento: {train_path}"
        )

    if not os.path.exists(val_path):
        raise FileNotFoundError(
            f"No se encontró el archivo de validación: {val_path}"
        )

    print(
        f"{Colors.BLUE}"
        f"Cargando datos CINC2020 de entrenamiento..."
        f"{Colors.ENDC}"
    )

    with h5py.File(train_path, 'r') as f:
        X_train = f['signals'][...]
        y_train = f['labels'][...]

        classes = None

        if 'classes' in f.attrs:
            classes_attr = f.attrs['classes']

            if isinstance(classes_attr, bytes):
                classes_attr = classes_attr.decode('utf-8')

            if isinstance(classes_attr, str):
                classes = classes_attr.split(',')

    print(
        f"{Colors.BLUE}"
        f"Cargando datos CINC2020 de validación..."
        f"{Colors.ENDC}"
    )

    with h5py.File(val_path, 'r') as f:
        X_val = f['signals'][...]
        y_val = f['labels'][...]

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)

    X_val = np.asarray(X_val, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.float32)

    print(
        f"{Colors.CYAN}"
        f"Training shapes - X: {X_train.shape}, "
        f"y: {y_train.shape}"
        f"{Colors.ENDC}"
    )

    print(
        f"{Colors.CYAN}"
        f"Validation shapes - X: {X_val.shape}, "
        f"y: {y_val.shape}"
        f"{Colors.ENDC}"
    )

    print(
        f"NaN en X_train: {np.any(np.isnan(X_train))}, "
        f"NaN en y_train: {np.any(np.isnan(y_train))}"
    )

    print(
        f"NaN en X_val: {np.any(np.isnan(X_val))}, "
        f"NaN en y_val: {np.any(np.isnan(y_val))}"
    )

    if classes is not None:
        print(
            f"{Colors.CYAN}"
            f"Clases CINC2020 encontradas: {len(classes)}"
            f"{Colors.ENDC}"
        )

    return X_train, y_train, X_val, y_val


# ============================================================
# PREPROCESAMIENTO DE ECG
# ============================================================

def preprocess(data, config):
    """
    Preprocesa una señal ECG para inferencia.

    Esta función ya no utiliza ninguna lógica de CINC2017.

    Para CINC2020 se trabaja con 12 derivaciones. Sin embargo,
    para mantener compatibilidad con el flujo de predicción,
    la señal se puede convertir a una representación 1D
    cuando se utiliza un ECG de una sola derivación.

    Args:
        data:
            Señal ECG como ndarray 1D o 2D.

        config:
            Configuración del proyecto.

    Returns:
        tuple:
            data_processed, peaks
    """

    sr = getattr(config, 'sample_rate', None)

    if sr is None:
        sr = 500

    if not isinstance(data, np.ndarray):
        data = np.asarray(data, dtype=np.float32)

    data = np.asarray(data, dtype=np.float32)

    data = np.nan_to_num(
        data,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # --------------------------------------------------------
    # Determinar formato de la señal
    # --------------------------------------------------------

    if data.ndim == 1:

        # ECG de una sola derivación.
        #
        # El modelo actual (ECGResNet) exige 12 derivaciones fijas
        # (in_channels=12 en la primera Conv1D) — no existe una ruta
        # válida para inferencia con 1 sola derivación. Antes esta
        # rama devolvía igual una forma (1, N, 1), que más adelante
        # terminaba fallando de forma confusa dentro de
        # prepare_signal_for_model() (ValueError de forma inesperada,
        # sin indicar la causa real). Preferimos fallar acá mismo,
        # con un mensaje que explique el motivo real.
        raise ValueError(
            "Se recibió un ECG de una sola derivación, pero el modelo "
            "requiere las 12 derivaciones estándar (I, II, III, aVR, "
            "aVL, aVF, V1-V6). Verifique el archivo de entrada."
        )

    elif data.ndim == 2:

        # ----------------------------------------------------
        # ECG multiderivación
        #
        # Se espera:
        #
        # (muestras, derivaciones)
        #
        # CINC2020:
        # (5000, 12)
        # ----------------------------------------------------

        if data.shape[0] < data.shape[1]:
            # Posible formato (derivaciones, muestras)
            # Se convierte a (muestras, derivaciones)
            data = data.T

        # Remuestreo
        if sr != 500:

            new_length = int(
                round(
                    data.shape[0] * 500 / sr
                )
            )

            data = resample(
                data,
                new_length,
                axis=0
            )

        # Escalado por derivación (en float64 para evitar el aviso de
        # precisión de sklearn con float32; se vuelve a float32 después)
        data = preprocessing.scale(
            data.astype(np.float64),
            axis=0
        )

        data = np.asarray(
            data,
            dtype=np.float32
        )

        # --------------------------------------------------------
        # Ajuste a longitud fija (5000 muestras = 10s a 500Hz)
        #
        # Sin este paso, cualquier ECG que no dure exactamente 10s
        # tras el remuestreo llega con una longitud distinta a la
        # que espera prepare_signal_for_model() en predict.py, que
        # exige (5000, 12) o (12, 5000) exactos y no acepta otra
        # cosa. Recorte centrado si es más largo, relleno con ceros
        # al final si es más corto — mismo criterio que fix_length()
        # en data2020.py, para no introducir un criterio distinto
        # entre el preprocesamiento de entrenamiento y el de inferencia.
        # --------------------------------------------------------

        target_len = 5000
        current_len = data.shape[0]

        if current_len > target_len:
            start = (current_len - target_len) // 2
            data = data[start:start + target_len, :]
        elif current_len < target_len:
            pad_width = target_len - current_len
            data = np.pad(
                data,
                ((0, pad_width), (0, 0)),
                mode='constant'
            )

        # Hash
        data_hash = hashlib.sha256(
            data.tobytes()
        ).hexdigest()

        print(
            f"DEBUG: Hash ECG multiderivación "
            f"preprocesado: {data_hash}"
        )

        # Para detección de picos utilizamos la derivación II
        if data.shape[1] >= 2:
            qrs_signal = data[:, 1]
        else:
            qrs_signal = data[:, 0]

        peaks, _ = find_peaks(
            qrs_signal,
            distance=125
        )

        # --------------------------------------------------------
        # IMPORTANTE: NO se agrega dimensión de batch acá.
        #
        # prepare_signal_for_model() (en predict.py) espera recibir
        # un array 2D limpio (5000, 12) o (12, 5000) — es esa función
        # la que agrega la dimensión de batch más adelante con
        # np.expand_dims(). Agregarla acá también producía un array
        # 3D (1, 5000, 12) que hacía fallar SIEMPRE la validación
        # `if data.ndim != 2` de prepare_signal_for_model(), sin
        # importar el tamaño del archivo subido.
        # --------------------------------------------------------

        return data, peaks

    else:

        raise ValueError(
            "La señal ECG debe ser 1D o 2D."
        )


# ============================================================
# LECTURA DE ECG SUBIDO
# ============================================================

def _extract_sampling_rate_from_comment(comment_line):
    """
    Extrae la frecuencia de muestreo de una línea de comentario.

    Ejemplo:

        # Sampling Rate: 500 Hz

    Returns:
        float or None
    """

    if 'Sampling Rate:' not in comment_line:
        return None

    try:
        sr_part = (
            comment_line
            .split('Sampling Rate:')[1]
            .strip()
            .split(' ')[0]
        )

        return float(sr_part)

    except (ValueError, IndexError):
        return None


def uploadedData(filename, csvbool=True, return_lead_names=False):
    """
    Lee un ECG subido en formato CSV.

    Compatible con señales de una o múltiples derivaciones.

    Returns:
        tuple:
            Si return_lead_names es True:
                sampling_rate, (signal_data_array, lead_names)
            Si return_lead_names es False:
                sampling_rate, signal_data_array
    """

    if not csvbool:
        raise NotImplementedError(
            "uploadedData con csvbool=False "
            "no está implementada."
        )

    sampling_rate = None

    with open(
        filename,
        'r',
        encoding='utf-8'
    ) as f:

        first_line = f.readline().strip()

    if first_line.startswith('#'):

        sampling_rate = (
            _extract_sampling_rate_from_comment(
                first_line
            )
        )

        try:

            if sampling_rate is not None:

                df = pd.read_csv(
                    filename,
                    skiprows=1
                )

            else:

                df = pd.read_csv(
                    filename
                )

        except Exception:

            df = pd.read_csv(
                filename
            )

    else:

        df = pd.read_csv(
            filename
        )

    lead_names = list(df.columns)
    signal_data_array = df.values

    if signal_data_array.shape[1] == 1:

        signal_data_array = (
            signal_data_array[:, 0]
        )

    if return_lead_names:
        return sampling_rate, (signal_data_array, lead_names)

    return sampling_rate, signal_data_array


# ============================================================
# MATRIZ DE CONFUSIÓN MULTILABEL
# ============================================================

def plot_confusion_matrix(
    y_true,
    y_pred,
    classes,
    feature=None,
    normalize=False,
    title=None,
    threshold=0.5
):
    """
    Genera matrices de confusión para un problema multilabel.

    Se genera una matriz 2x2 independiente para cada clase:

        [[TN, FP],
         [FN, TP]]

    Args:
        y_true:
            Matriz binaria (N, C).

        y_pred:
            Probabilidades (N, C) o etiquetas binarias (N, C).

        classes:
            Lista de nombres de clases.

        normalize:
            Normaliza cada matriz.

        threshold:
            Umbral para convertir probabilidades en etiquetas.

    Returns:
        list de figuras/ejes.
    """

    y_true = np.asarray(y_true)

    y_pred = np.asarray(y_pred)

    if y_pred.ndim != 2:
        raise ValueError(
            "y_pred debe tener forma (N, clases)."
        )

    if y_pred.dtype.kind in 'fc':
        y_pred_binary = (
            y_pred >= threshold
        ).astype(int)
    else:
        y_pred_binary = y_pred.astype(int)

    n_classes = len(classes)

    output_dir = os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)
        ),
        'static',
        'asset'
    )

    mkdir_recursive(output_dir)

    figures = []

    for i, class_name in enumerate(classes):

        cm = confusion_matrix(
            y_true[:, i],
            y_pred_binary[:, i],
            labels=[0, 1]
        )

        if normalize:

            row_sums = cm.sum(
                axis=1,
                keepdims=True
            )

            cm = np.divide(
                cm.astype(float),
                row_sums,
                out=np.zeros_like(
                    cm,
                    dtype=float
                ),
                where=row_sums != 0
            )

        fig, ax = plt.subplots()

        im = ax.imshow(
            cm,
            interpolation='nearest',
            cmap=plt.cm.Blues
        )

        ax.figure.colorbar(
            im,
            ax=ax
        )

        ax.set(
            xticks=np.arange(2),
            yticks=np.arange(2),
            xticklabels=['Negativo', 'Positivo'],
            yticklabels=['Negativo', 'Positivo'],
            title=(
                f'{class_name}'
                if title is None
                else f'{title} - {class_name}'
            ),
            ylabel='Etiqueta verdadera',
            xlabel='Etiqueta predicha'
        )

        thresh = cm.max() / 2.0

        fmt = '.2f' if normalize else 'd'

        for row in range(2):

            for col in range(2):

                value = cm[row, col]

                ax.text(
                    col,
                    row,
                    format(value, fmt),
                    ha='center',
                    va='center',
                    color=(
                        'white'
                        if value > thresh
                        else 'black'
                    )
                )

        fig.tight_layout()

        safe_class_name = (
            str(class_name)
            .replace('/', '_')
            .replace(' ', '_')
        )

        png_path = os.path.join(
            output_dir,
            f'confusion_matrix_{safe_class_name}.png'
        )

        fig.savefig(
            png_path,
            format='png',
            dpi=300,
            bbox_inches='tight'
        )

        figures.append(
            (class_name, fig, ax, png_path)
        )

        plt.close(fig)

    print(
        f"{Colors.GREEN}"
        f"Se generaron matrices de confusión "
        f"para {n_classes} clases."
        f"{Colors.ENDC}"
    )

    return figures


# ============================================================
# CURVAS PR Y ROC
# ============================================================

def PR_ROC_curves(
    ytrue,
    ypred,
    classes,
    ypred_mat,
    classification_report_dict=None
):
    """
    Genera curvas Precision-Recall y ROC para cada clase
    del problema multilabel CINC2020.

    Args:
        ytrue:
            Etiquetas binarias reales (N, C).

        ypred:
            Etiquetas binarias predichas (N, C).

        classes:
            Lista de clases.

        ypred_mat:
            Probabilidades/scores (N, C).

    Returns:
        dict con ROC-AUC y AP por clase.
    """

    ytrue = np.asarray(
        ytrue,
        dtype=int
    )

    ypred_mat = np.asarray(
        ypred_mat,
        dtype=float
    )

    results = {}

    n_classes = len(classes)

    # --------------------------------------------------------
    # ROC
    # --------------------------------------------------------

    fig_roc, ax_roc = plt.subplots(
        1,
        1,
        figsize=(10, 8)
    )

    # --------------------------------------------------------
    # PR
    # --------------------------------------------------------

    fig_pr, ax_pr = plt.subplots(
        1,
        1,
        figsize=(10, 8)
    )

    valid_roc_auc = []
    valid_ap = []

    for i, class_name in enumerate(classes):

        y_class = ytrue[:, i]

        scores = ypred_mat[:, i]

        result = {
            'roc_auc': np.nan,
            'average_precision': np.nan
        }

        # ----------------------------------------------------
        # ROC
        # ----------------------------------------------------

        if len(np.unique(y_class)) >= 2:

            try:

                fpr, tpr, _ = roc_curve(
                    y_class,
                    scores
                )

                auc_score = roc_auc_score(
                    y_class,
                    scores
                )

                ax_roc.plot(
                    fpr,
                    tpr,
                    lw=1.5,
                    label=f'{class_name} (AUC={auc_score:.3f})'
                )

                result['roc_auc'] = auc_score

                valid_roc_auc.append(
                    auc_score
                )

            except ValueError as e:

                print(
                    f"{Colors.WARNING}"
                    f"No se pudo calcular ROC para "
                    f"{class_name}: {e}"
                    f"{Colors.ENDC}"
                )

        else:

            print(
                f"{Colors.WARNING}"
                f"ROC omitida para {class_name}: "
                f"solo existe una clase en y_true."
                f"{Colors.ENDC}"
            )

        # ----------------------------------------------------
        # Precision-Recall
        # ----------------------------------------------------

        try:

            precision, recall, _ = (
                precision_recall_curve(
                    y_class,
                    scores
                )
            )

            ap_score = average_precision_score(
                y_class,
                scores
            )

            ax_pr.plot(
                recall,
                precision,
                lw=1.5,
                label=f'{class_name} (AP={ap_score:.3f})'
            )

            result['average_precision'] = ap_score

            valid_ap.append(
                ap_score
            )

        except ValueError as e:

            print(
                f"{Colors.WARNING}"
                f"No se pudo calcular PR para "
                f"{class_name}: {e}"
                f"{Colors.ENDC}"
            )

        results[class_name] = result

    # --------------------------------------------------------
    # Configuración ROC
    # --------------------------------------------------------

    ax_roc.plot(
        [0, 1],
        [0, 1],
        linestyle='--',
        linewidth=1
    )

    ax_roc.set_xlim(
        0,
        1
    )

    ax_roc.set_ylim(
        0,
        1.05
    )

    ax_roc.set_xlabel(
        '1 - Especificidad'
    )

    ax_roc.set_ylabel(
        'Sensibilidad'
    )

    ax_roc.set_title(
        'Curvas ROC - CINC2020'
    )

    ax_roc.legend(
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        fontsize=7
    )

    ax_roc.grid(
        True,
        alpha=0.3
    )

    fig_roc.tight_layout()

    # --------------------------------------------------------
    # Configuración PR
    # --------------------------------------------------------

    ax_pr.set_xlim(
        0,
        1
    )

    ax_pr.set_ylim(
        0,
        1.05
    )

    ax_pr.set_xlabel(
        'Recall / Sensibilidad'
    )

    ax_pr.set_ylabel(
        'Precision / PPV'
    )

    ax_pr.set_title(
        'Curvas Precision-Recall - CINC2020'
    )

    ax_pr.legend(
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        fontsize=7
    )

    ax_pr.grid(
        True,
        alpha=0.3
    )

    fig_pr.tight_layout()

    mkdir_recursive(
        'resultados'
    )

    roc_path = (
        'resultados/'
        'model_roc_curves.png'
    )

    pr_path = (
        'resultados/'
        'model_precision_recall_curves.png'
    )

    fig_roc.savefig(
        roc_path,
        dpi=300,
        bbox_inches='tight'
    )

    fig_pr.savefig(
        pr_path,
        dpi=300,
        bbox_inches='tight'
    )

    plt.close(fig_roc)
    plt.close(fig_pr)

    # --------------------------------------------------------
    # Promedios
    # --------------------------------------------------------

    results['macro_roc_auc'] = (
        float(np.mean(valid_roc_auc))
        if valid_roc_auc
        else np.nan
    )

    results['macro_average_precision'] = (
        float(np.mean(valid_ap))
        if valid_ap
        else np.nan
    )

    print(
        f"{Colors.GREEN}"
        f"ROC-AUC macro: "
        f"{results['macro_roc_auc']:.4f}"
        f"{Colors.ENDC}"
    )

    print(
        f"{Colors.GREEN}"
        f"Average Precision macro: "
        f"{results['macro_average_precision']:.4f}"
        f"{Colors.ENDC}"
    )

    return results


# ============================================================
# EVALUACIÓN DEL MODELO PYTORCH
# ============================================================

def print_results(
    config,
    model,
    Xval,
    yval,
    classes,
    device=None,
    threshold=0.5
):
    """
    Evalúa un modelo PyTorch en un conjunto multilabel.

    IMPORTANTE:

    CINC2020 NO utiliza argmax.

    Cada una de las 27 clases tiene una probabilidad
    independiente.

    Args:
        config:
            Configuración.

        model:
            Modelo PyTorch.

        Xval:
            Datos de validación.

        yval:
            Etiquetas binarias (N, 27).

        classes:
            Lista de nombres de clases.

        device:
            torch.device.

        threshold:
            Umbral de clasificación.

    Returns:
        dict con las métricas.
    """

    import torch

    model.eval()

    if device is None:

        device = torch.device(
            'cuda'
            if torch.cuda.is_available()
            else 'cpu'
        )

    model = model.to(device)

    # --------------------------------------------------------
    # Preparar X
    # --------------------------------------------------------

    Xval = np.asarray(
        Xval,
        dtype=np.float32
    )

    yval = np.asarray(
        yval,
        dtype=np.float32
    )

    # HDF5:
    #
    # (N, 5000, 12)
    #
    # PyTorch Conv1D:
    #
    # (N, 12, 5000)

    if Xval.ndim == 3:

        Xval_tensor = torch.from_numpy(
            Xval
        ).permute(
            0,
            2,
            1
        )

    else:

        Xval_tensor = torch.from_numpy(
            Xval
        )

    yval_tensor = torch.from_numpy(
        yval
    )

    # --------------------------------------------------------
    # Predicción por lotes
    # --------------------------------------------------------

    batch_size = getattr(
        config,
        'batch_size',
        8
    )

    probabilities = []

    with torch.no_grad():

        for start in range(
            0,
            len(Xval_tensor),
            batch_size
        ):

            end = min(
                start + batch_size,
                len(Xval_tensor)
            )

            batch = Xval_tensor[
                start:end
            ].to(device)

            logits = model(
                batch
            )

            # BCEWithLogitsLoss trabaja con logits.
            # Para evaluación convertimos a probabilidades.

            probs = torch.sigmoid(
                logits
            )

            probabilities.append(
                probs.cpu().numpy()
            )

    ypred_mat = np.concatenate(
        probabilities,
        axis=0
    )

    # --------------------------------------------------------
    # Etiquetas binarias
    # --------------------------------------------------------

    ypred = (
        ypred_mat >= threshold
    ).astype(int)

    ytrue = (
        yval >= 0.5
    ).astype(int)

    print(
        f"{Colors.CYAN}"
        f"ytrue.shape: {ytrue.shape}"
        f"{Colors.ENDC}"
    )

    print(
        f"{Colors.CYAN}"
        f"ypred.shape: {ypred.shape}"
        f"{Colors.ENDC}"
    )

    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    f1_macro = f1_score(
        ytrue,
        ypred,
        average='macro',
        zero_division=0
    )

    f1_micro = f1_score(
        ytrue,
        ypred,
        average='micro',
        zero_division=0
    )

    f1_weighted = f1_score(
        ytrue,
        ypred,
        average='weighted',
        zero_division=0
    )

    f1_per_class = f1_score(
        ytrue,
        ypred,
        average=None,
        zero_division=0
    )

    # --------------------------------------------------------
    # Accuracy multilabel
    # --------------------------------------------------------

    subset_accuracy = accuracy_score(
        ytrue,
        ypred
    )

    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    roc_auc_per_class = []

    for i in range(
        len(classes)
    ):

        if len(
            np.unique(
                ytrue[:, i]
            )
        ) >= 2:

            try:

                auc = roc_auc_score(
                    ytrue[:, i],
                    ypred_mat[:, i]
                )

                roc_auc_per_class.append(
                    auc
                )

            except ValueError:

                roc_auc_per_class.append(
                    np.nan
                )

        else:

            roc_auc_per_class.append(
                np.nan
            )

    macro_roc_auc = np.nanmean(
        roc_auc_per_class
    )

    # --------------------------------------------------------
    # Average Precision
    # --------------------------------------------------------

    ap_per_class = []

    for i in range(
        len(classes)
    ):

        try:

            ap = average_precision_score(
                ytrue[:, i],
                ypred_mat[:, i]
            )

            ap_per_class.append(
                ap
            )

        except ValueError:

            ap_per_class.append(
                np.nan
            )

    macro_ap = np.nanmean(
        ap_per_class
    )

    # --------------------------------------------------------
    # Reporte por clase
    # --------------------------------------------------------

    report = classification_report(
        ytrue,
        ypred,
        target_names=classes,
        output_dict=True,
        zero_division=0
    )

    # --------------------------------------------------------
    # Mostrar métricas
    # --------------------------------------------------------

    print(
        f"\n{Colors.HEADER}"
        f"===== RESULTADOS CINC2020 ====="
        f"{Colors.ENDC}"
    )

    print(
        f"F1 macro:       {f1_macro:.4f}"
    )

    print(
        f"F1 micro:       {f1_micro:.4f}"
    )

    print(
        f"F1 weighted:    {f1_weighted:.4f}"
    )

    print(
        f"Subset accuracy:{subset_accuracy:.4f}"
    )

    print(
        f"ROC-AUC macro:  {macro_roc_auc:.4f}"
    )

    print(
        f"AP macro:       {macro_ap:.4f}"
    )

    print(
        f"\n{Colors.BLUE}"
        f"Reporte por clase:"
        f"{Colors.ENDC}"
    )

    print(
        classification_report(
            ytrue,
            ypred,
            target_names=classes,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Matrices de confusión
    # --------------------------------------------------------

    plot_confusion_matrix(
        ytrue,
        ypred,
        classes,
        feature=getattr(
            config,
            'feature',
            None
        ),
        normalize=False,
        threshold=threshold
    )

    # --------------------------------------------------------
    # Curvas ROC / PR
    # --------------------------------------------------------

    curve_results = PR_ROC_curves(
        ytrue,
        ypred,
        classes,
        ypred_mat,
        report
    )

    # --------------------------------------------------------
    # Resultado final
    # --------------------------------------------------------

    results = {
        'f1_macro': float(
            f1_macro
        ),

        'f1_micro': float(
            f1_micro
        ),

        'f1_weighted': float(
            f1_weighted
        ),

        'subset_accuracy': float(
            subset_accuracy
        ),

        'roc_auc_macro': float(
            macro_roc_auc
        ),

        'average_precision_macro': float(
            macro_ap
        ),

        'f1_per_class': (
            f1_per_class.tolist()
        ),

        'roc_auc_per_class': (
            roc_auc_per_class
        ),

        'average_precision_per_class': (
            ap_per_class
        ),

        'classification_report': report,

        'probabilities': ypred_mat,

        'predictions': ypred,

        'true_labels': ytrue,

        'curve_results': curve_results
    }

    return results