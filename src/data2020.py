# -*- coding: utf-8 -*-
"""
data2020.py

Pipeline de preprocesamiento de datos para el PhysioNet/CinC Challenge 2020
(12 derivaciones, multi-etiqueta). Reemplaza la lógica de data.py (pensada
para MIT-BIH, clasificación de latido individual) por un pipeline a nivel
de registro completo, con diagnósticos SNOMED-CT.

Corresponde a la Fase 1 de pipeline_investigacion_ecg.md.

Uso esperado (ver bloque __main__ al final):
    python data2020.py --data_dir dataset2020 --output_dir dataset2020_procesado
"""

from __future__ import division, print_function
import os
import re
import glob
import numpy as np
import pandas as pd
from scipy.signal import resample
from scipy.io import loadmat


# =====================================================================
# PASO 2 — Parseo de encabezados WFDB (.hea)
# =====================================================================

def parse_header(hea_path):
    """
    Parsea un archivo .hea de WFDB y extrae metadata clínica y técnica.

    Formato esperado del .hea (ejemplo real del challenge):

        A0001 12 500 5000
        A0001.mat 16+24 1000/mV 16 0 -260 -25835 0 I
        A0001.mat 16+24 1000/mV 16 0 -1220 27189 0 II
        ...  (una línea por derivación)
        #Age: 74
        #Sex: Male
        #Dx: 426783006,164865005
        #Rx: Unknown
        #Hx: Unknown
        #Sx: Unknown

    Args:
        hea_path (str): ruta al archivo .hea.

    Returns:
        dict con:
            record_name (str)
            num_leads (int)
            sampling_freq (float): en Hz.
            num_samples (int)
            lead_names (list[str]): orden de las derivaciones tal como
                aparecen en el .mat asociado (ej. ['I','II','III',...]).
            age (float | None): NaN si es 'NaN' o falta.
            sex (str | None): 'Male' / 'Female' / None.
            dx_codes (list[str]): códigos SNOMED-CT como strings, tal
                cual aparecen en el header (pueden ser uno o varios).
    """
    with open(hea_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip() != '']

    # Primera línea: nombre_registro num_derivaciones frecuencia num_muestras
    header_line = lines[0].split()
    record_name = header_line[0]
    num_leads = int(header_line[1])
    sampling_freq = float(header_line[2])
    num_samples = int(header_line[3])

    # Siguientes num_leads líneas: una por derivación; el nombre de la
    # derivación es el último token de cada línea.
    lead_names = []
    for i in range(1, num_leads + 1):
        lead_names.append(lines[i].split()[-1])

    # Líneas de comentario (#Age, #Sex, #Dx, ...)
    age, sex, dx_codes = None, None, []
    for line in lines[num_leads + 1:]:
        if line.startswith('#Age'):
            raw = line.split(':', 1)[1].strip()
            age = np.nan if raw.lower() == 'nan' else float(raw)
        elif line.startswith('#Sex'):
            sex = line.split(':', 1)[1].strip()
        elif line.startswith('#Dx'):
            raw = line.split(':', 1)[1].strip()
            dx_codes = [c.strip() for c in raw.split(',') if c.strip() != '']

    return {
        'record_name': record_name,
        'num_leads': num_leads,
        'sampling_freq': sampling_freq,
        'num_samples': num_samples,
        'lead_names': lead_names,
        'age': age,
        'sex': sex,
        'dx_codes': dx_codes,
    }


# Orden estándar de derivaciones que vamos a exigir en todo el dataset.
# (Paso 2, verificación de consistencia entre bases)
STANDARD_LEAD_ORDER = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                        'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def reorder_leads(signal, lead_names):
    """
    Reordena las derivaciones de una señal al orden estándar
    STANDARD_LEAD_ORDER, sin importar el orden en que vinieron en el .hea.

    Args:
        signal (np.ndarray): forma (num_muestras, num_derivaciones).
        lead_names (list[str]): nombres de derivación en el orden actual
            de `signal` (tal como los devuelve parse_header).

    Returns:
        np.ndarray reordenado, forma (num_muestras, 12).

    Raises:
        ValueError: si falta alguna derivación estándar en el registro
            (esto NO debería pasar en las 6 bases del challenge, pero
            preferimos que truene con un mensaje claro a que produzca
            un dataset silenciosamente mal alineado).
    """
    try:
        idx = [lead_names.index(lead) for lead in STANDARD_LEAD_ORDER]
    except ValueError as e:
        raise ValueError(
            f"Registro con derivaciones inesperadas. "
            f"Esperadas: {STANDARD_LEAD_ORDER}, encontradas: {lead_names}. "
            f"Detalle: {e}"
        )
    return signal[:, idx]


# =====================================================================
# PASO 3 — Mapeo de códigos SNOMED-CT a las 27 clases evaluadas
# =====================================================================

def load_scored_classes(csv_path='dx_mapping_scored.csv'):
    """
    Carga el mapping oficial del challenge (dx_mapping_scored.csv) y arma
    la fusión de códigos SNOMED-CT equivalentes indicada en la columna
    'Notes' (ej. "We score 713427006 and 59118001 as the same diagnosis.").

    Args:
        csv_path (str): ruta al CSV oficial (incluido en este entrego).

    Returns:
        tuple:
            classes (list[str]): las 27 abreviaturas de clase, en el orden
                en que van a corresponder a las columnas del vector de
                etiquetas (ej. ['IAVB','AF','AFL', ...]).
            code_to_class_idx (dict[str, int]): mapea CADA código SNOMED-CT
                (incluyendo los códigos "duplicados" fusionados) al índice
                de clase correspondiente en `classes`.
    """
    df = pd.read_csv(csv_path, dtype=str)
    df['SNOMED CT Code'] = df['SNOMED CT Code'].astype(str)

    classes = df['Abbreviation'].tolist()
    code_to_class_idx = {
        code: idx for idx, code in enumerate(df['SNOMED CT Code'])
    }

    # Fusionar códigos equivalentes descritos en 'Notes' con una regex simple
    # que extrae los dos códigos SNOMED-CT numéricos de la frase.
    note_pattern = re.compile(r'(\d{6,})\s+and\s+(\d{6,})')
    for _, row in df.iterrows():
        note = row.get('Notes', '')
        if isinstance(note, str) and note.strip():
            match = note_pattern.search(note)
            if match:
                code_a, code_b = match.group(1), match.group(2)
                # Ambos códigos deben apuntar al mismo índice de clase.
                # Usamos el índice ya asignado a code_a (o code_b) como
                # canónico, según cuál ya esté en el diccionario.
                canonical_idx = code_to_class_idx.get(code_a,
                                  code_to_class_idx.get(code_b))
                code_to_class_idx[code_a] = canonical_idx
                code_to_class_idx[code_b] = canonical_idx

    return classes, code_to_class_idx


def dx_codes_to_label_vector(dx_codes, code_to_class_idx, num_classes):
    """
    Convierte la lista de códigos SNOMED-CT de un registro (parse_header
    -> 'dx_codes') a un vector multi-etiqueta binario.

    Códigos que no están entre las 27 clases evaluadas (hay 111 códigos
    distintos en total en el dataset, ver dx_mapping_unscored.csv) se
    ignoran para efectos del vector de etiquetas, pero se recomienda
    seguir contándolos aparte para reportar cobertura en el paper.

    Args:
        dx_codes (list[str]): códigos SNOMED-CT del registro.
        code_to_class_idx (dict[str, int]): de load_scored_classes().
        num_classes (int): 27, largo de `classes`.

    Returns:
        np.ndarray de forma (num_classes,), dtype float32, con 1.0 en
        cada clase presente y 0.0 en el resto. Puede tener más de un 1
        (multi-etiqueta) o ningún 1 si el registro no tiene ninguno de
        los 27 diagnósticos evaluados (registro "no puntuable" para el
        challenge, pero igual usable como negativo para todas las clases).
    """
    y = np.zeros(num_classes, dtype=np.float32)
    for code in dx_codes:
        idx = code_to_class_idx.get(code)
        if idx is not None:
            y[idx] = 1.0
    return y


# =====================================================================
# PASO 4 — Resampleo a frecuencia común
# =====================================================================

TARGET_FS = 500  # Hz. Elegida porque es la frecuencia de la mayoría de
                  # las bases (CPSC2018, CPSC-Extra, PTB-XL, Georgia).
                  # INCART (257Hz) y PTB (1000Hz) se resamplean hacia acá.


def resample_signal(signal, orig_fs, target_fs=TARGET_FS):
    """
    Resamplea una señal multicanal de orig_fs a target_fs usando FFT
    (scipy.signal.resample). Si orig_fs == target_fs, devuelve la señal
    sin modificar (evita error numérico innecesario por resampleo trivial).

    Args:
        signal (np.ndarray): forma (num_muestras, num_derivaciones).
        orig_fs (float): frecuencia original, tomada de parse_header().
        target_fs (int): frecuencia destino, default TARGET_FS.

    Returns:
        np.ndarray resampleado, forma (num_muestras_nuevo, num_derivaciones).
    """
    if orig_fs == target_fs:
        return signal
    num_samples_orig = signal.shape[0]
    num_samples_new = int(round(num_samples_orig * target_fs / orig_fs))
    return resample(signal, num_samples_new, axis=0)


# =====================================================================
# PASO 5 — Ventana de longitud fija (recorte / relleno)
# =====================================================================

WINDOW_SECONDS = 10          # ventana clínicamente razonable, alineada a
                              # la duración típica de CPSC-Extra/PTB-XL/Georgia
WINDOW_LENGTH = WINDOW_SECONDS * TARGET_FS  # 5000 muestras a 500Hz


def fix_length(signal, target_len=WINDOW_LENGTH, mode='center'):
    """
    Ajusta una señal (ya resampleada) a exactamente `target_len` muestras.

    - Si la señal es más larga: recorta según `mode`.
        'center' -> toma el segmento central (default, más estable para
                    entrenamiento/evaluación determinística).
        'random' -> toma un segmento aleatorio (usar solo en entrenamiento,
                    como data augmentation liviano).
    - Si la señal es más corta (pasa con CPSC-Extra, que tiene registros
      de 6s): rellena con ceros al final (zero-padding).

    Args:
        signal (np.ndarray): forma (num_muestras, 12).
        target_len (int): longitud objetivo en muestras.
        mode (str): 'center' o 'random'.

    Returns:
        np.ndarray de forma exacta (target_len, 12).
    """
    current_len = signal.shape[0]

    if current_len == target_len:
        return signal

    if current_len > target_len:
        if mode == 'random':
            start = np.random.randint(0, current_len - target_len + 1)
        else:  # 'center'
            start = (current_len - target_len) // 2
        return signal[start:start + target_len, :]

    # current_len < target_len: zero-padding al final
    pad_width = target_len - current_len
    return np.pad(signal, ((0, pad_width), (0, 0)), mode='constant')


def is_incart_record(source_db):
    """
    Identifica registros de INCART (30 min, Holter) para tratarlos como
    set de validación externa de generalización en vez de entrenamiento
    (ver Fase 6 del pipeline). `source_db` es el nombre de carpeta/base
    de origen que se les asigna al recorrer el dataset (Paso 7).
    """
    return source_db.lower() in ('incart', 'st-petersburg-incart', 'stpetersburg')


# =====================================================================
# PASO 6 — Split train/val/test estratificado por clase Y por base
# =====================================================================

def stratified_multilabel_split(labels, source_dbs, val_frac=0.15,
                                 test_frac=0.15, random_state=42):
    """
    Divide los índices del dataset en train/val/test, intentando preservar
    tanto la distribución de clases (multi-etiqueta) como la proporción
    de cada base de origen en cada split.

    Requiere el paquete 'iterative-stratification' para la estratificación
    real por clase (agregar a requirements.txt: iterative-stratification).
    Si no está instalado, cae a un split aleatorio simple estratificado
    solo por base de origen (menos riguroso, pero funcional) e imprime
    una advertencia — no debería usarse así para los resultados finales
    del paper.

    Args:
        labels (np.ndarray): forma (N, 27), vectores multi-etiqueta.
        source_dbs (list[str]): base de origen de cada registro, largo N.
        val_frac (float): fracción para validación.
        test_frac (float): fracción para test.
        random_state (int): semilla, para reproducibilidad (Fase 4).

    Returns:
        dict con 'train', 'val', 'test' -> cada uno un np.ndarray de
        índices (posiciones en `labels`/`source_dbs`).
    """
    n = labels.shape[0]
    indices = np.arange(n)

    try:
        from skmultilearn.model_selection import IterativeStratification
        np.random.seed(random_state)

        # Primer split: train+val vs test
        stratifier = IterativeStratification(
            n_splits=2, order=1,
            sample_distribution_per_fold=[test_frac, 1 - test_frac])
        trainval_idx, test_idx = next(stratifier.split(indices, labels))

        # Segundo split: train vs val, dentro de trainval_idx
        val_frac_of_trainval = val_frac / (1 - test_frac)
        stratifier2 = IterativeStratification(
            n_splits=2, order=1,
            sample_distribution_per_fold=[val_frac_of_trainval,
                                           1 - val_frac_of_trainval])
        train_sub_idx, val_sub_idx = next(
            stratifier2.split(indices[trainval_idx], labels[trainval_idx]))
        train_idx = indices[trainval_idx][train_sub_idx]
        val_idx = indices[trainval_idx][val_sub_idx]

    except ImportError:
        print("AVISO: paquete 'iterative-stratification' no encontrado. "
              "Usando split aleatorio simple estratificado solo por base "
              "de origen. Para resultados de paper, instalar con: "
              "pip install iterative-stratification")
        rng = np.random.RandomState(random_state)
        train_idx, val_idx, test_idx = [], [], []
        source_dbs = np.array(source_dbs)
        for db in np.unique(source_dbs):
            db_idx = indices[source_dbs == db]
            rng.shuffle(db_idx)
            n_db = len(db_idx)
            n_test = int(round(n_db * test_frac))
            n_val = int(round(n_db * val_frac))
            test_idx.extend(db_idx[:n_test])
            val_idx.extend(db_idx[n_test:n_test + n_val])
            train_idx.extend(db_idx[n_test + n_val:])
        train_idx, val_idx, test_idx = (np.array(train_idx, dtype=int),
                                         np.array(val_idx, dtype=int),
                                         np.array(test_idx, dtype=int))

    return {'train': train_idx, 'val': val_idx, 'test': test_idx}


# =====================================================================
# PASO 7 — Orquestador: recorrer las 6 bases y guardar en HDF5
# =====================================================================

def infer_source_db(hea_path, data_dir):
    """
    Infiere la base de origen de un registro a partir de su ruta relativa
    dentro de `data_dir`. Asume que cada base quedó en su propia carpeta
    de primer nivel tras la descarga (Paso 1) — ajustar esta función si
    la estructura de carpetas real difiere una vez descargado el dataset.
    """
    rel_path = os.path.relpath(hea_path, data_dir)
    return rel_path.split(os.sep)[0]


def build_dataset(data_dir, scored_classes_csv='dx_mapping_scored.csv'):
    """
    Recorre recursivamente `data_dir` buscando todos los archivos .hea,
    y para cada uno:
      1. Parsea el header (Paso 2) y verifica/reordena derivaciones.
      2. Mapea los códigos Dx a un vector multi-etiqueta (Paso 3).
      3. Carga la señal del .mat correspondiente y la resamplea (Paso 4).
      4. La ajusta a longitud fija (Paso 5).
      5. Registra la base de origen (para el split del Paso 6 y para
         poder excluir INCART del entrenamiento si así se decide).

    Args:
        data_dir (str): carpeta raíz donde quedaron las 6 bases descargadas.
        scored_classes_csv (str): ruta al CSV oficial de clases evaluadas.

    Returns:
        dict con:
            signals (np.ndarray): forma (N, WINDOW_LENGTH, 12), float32.
            labels (np.ndarray): forma (N, 27), float32.
            ages (np.ndarray): forma (N,), float32 (NaN si falta).
            sexes (list[str]): largo N.
            source_dbs (list[str]): largo N.
            record_names (list[str]): largo N, para trazabilidad.
            classes (list[str]): las 27 abreviaturas, en orden de columna.
    """
    classes, code_to_class_idx = load_scored_classes(scored_classes_csv)
    num_classes = len(classes)

    hea_files = sorted(glob.glob(os.path.join(data_dir, '**', '*.hea'),
                                  recursive=True))
    print(f"Encontrados {len(hea_files)} registros (.hea) en {data_dir}")

    signals_list, labels_list = [], []
    ages_list, sexes_list, source_dbs_list, record_names_list = [], [], [], []

    n_errores = 0
    for hea_path in hea_files:
        try:
            meta = parse_header(hea_path)
            mat_path = hea_path.replace('.hea', '.mat')
            mat_data = loadmat(mat_path)
            # La convención del challenge guarda la señal bajo la clave 'val'
            raw_signal = mat_data['val'].astype(np.float32).T  # -> (muestras, derivaciones)

            signal = reorder_leads(raw_signal, meta['lead_names'])
            signal = resample_signal(signal, meta['sampling_freq'])
            signal = fix_length(signal, mode='center')

            label_vec = dx_codes_to_label_vector(
                meta['dx_codes'], code_to_class_idx, num_classes)

            signals_list.append(signal)
            labels_list.append(label_vec)
            ages_list.append(meta['age'] if meta['age'] is not None else np.nan)
            sexes_list.append(meta['sex'])
            source_dbs_list.append(infer_source_db(hea_path, data_dir))
            record_names_list.append(meta['record_name'])

        except Exception as e:
            # Un registro problemático no debe tirar abajo todo el
            # preprocesamiento — se loguea y se sigue. Revisar este
            # conteo al final: si es alto, hay un problema sistemático
            # que vale la pena investigar (no solo ignorar).
            print(f"AVISO: error procesando {hea_path}: {e}")
            n_errores += 1

    print(f"Procesamiento completo. {len(signals_list)} registros OK, "
          f"{n_errores} con error.")

    return {
        'signals': np.stack(signals_list, axis=0),
        'labels': np.stack(labels_list, axis=0),
        'ages': np.array(ages_list, dtype=np.float32),
        'sexes': sexes_list,
        'source_dbs': source_dbs_list,
        'record_names': record_names_list,
        'classes': classes,
    }


def save_to_hdf5(dataset, split_indices, output_dir):
    """
    Guarda el dataset preprocesado en 3 archivos HDF5 (train/val/test),
    conservando metadata de origen para análisis de subgrupos posterior.

    Args:
        dataset (dict): salida de build_dataset().
        split_indices (dict): salida de stratified_multilabel_split(),
            con claves 'train', 'val', 'test'.
        output_dir (str): carpeta destino.
    """
    import h5py
    os.makedirs(output_dir, exist_ok=True)

    for split_name, idx in split_indices.items():
        out_path = os.path.join(output_dir, f'{split_name}.h5')
        with h5py.File(out_path, 'w') as f:
            f.create_dataset('signals', data=dataset['signals'][idx])
            f.create_dataset('labels', data=dataset['labels'][idx])
            f.create_dataset('ages', data=dataset['ages'][idx])
            # h5py necesita strings de largo variable en formato especial
            str_dt = h5py.special_dtype(vlen=str)
            f.create_dataset('sexes', data=np.array(
                [s if s else 'Unknown' for s in
                 np.array(dataset['sexes'])[idx]], dtype=object), dtype=str_dt)
            f.create_dataset('source_dbs', data=np.array(
                dataset['source_dbs'])[idx].astype(object), dtype=str_dt)
            f.create_dataset('record_names', data=np.array(
                dataset['record_names'])[idx].astype(object), dtype=str_dt)
            f.attrs['classes'] = dataset['classes']
        print(f"Guardado {split_name}: {len(idx)} registros -> {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Preprocesamiento PhysioNet/CinC Challenge 2020 (Fase 1).')
    parser.add_argument('--data_dir', type=str, default='dataset2020',
                         help='Carpeta con las 6 bases descargadas (Paso 1).')
    parser.add_argument('--output_dir', type=str, default='dataset2020_procesado',
                         help='Carpeta destino para los .h5 de train/val/test.')
    parser.add_argument('--scored_classes_csv', type=str,
                         default='dx_mapping_scored.csv')
    parser.add_argument('--exclude_incart_from_train', action='store_true',
                         help='Si se pasa, INCART se excluye de train/val y '
                              'se guarda aparte como set de generalización '
                              '(ver Fase 6 del pipeline).')
    args = parser.parse_args()

    dataset = build_dataset(args.data_dir, args.scored_classes_csv)

    if args.exclude_incart_from_train:
        is_incart = np.array([is_incart_record(db)
                               for db in dataset['source_dbs']])
        incart_idx = np.where(is_incart)[0]
        rest_idx = np.where(~is_incart)[0]

        print(f"Excluyendo INCART de train/val: {len(incart_idx)} registros "
              f"apartados como set de generalización externa.")

        rest_labels = dataset['labels'][rest_idx]
        rest_sources = [dataset['source_dbs'][i] for i in rest_idx]
        splits_local = stratified_multilabel_split(rest_labels, rest_sources)
        # Traducir índices locales (sobre rest_idx) a índices globales
        splits = {k: rest_idx[v] for k, v in splits_local.items()}
        splits['incart_external'] = incart_idx
    else:
        splits = stratified_multilabel_split(
            dataset['labels'], dataset['source_dbs'])

    save_to_hdf5(dataset, splits, args.output_dir)
