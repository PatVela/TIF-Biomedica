# -*- coding: utf-8 -*-
"""
data2020.py

Pipeline de preprocesamiento para PhysioNet/CinC Challenge 2020.

Estructura del proyecto:

    proyecto/
    ├── dataset2020/
    ├── data/
    ├── src/
    │   ├── data2020.py
    │   └── dx_mapping_scored.csv
    └── ...

Este script está diseñado para ejecutarse desde la raíz:

    python src/data2020.py

o:

    python src/data2020.py


Características:
    - Lee metadata desde archivos WFDB .hea.
    - Obtiene diagnósticos SNOMED-CT.
    - Mapea a las 27 clases puntuadas.
    - Exige las 12 derivaciones estándar.
    - Reordena derivaciones.
    - Resamplea a 500 Hz mediante resample_poly().
    - Ajusta a 10 segundos = 5000 muestras.
    - Split multilabel estratificado.
    - INCART opcional como conjunto externo.
    - Multiprocessing para procesamiento pesado.
    - Escritura HDF5 desde un único proceso.
    - Escritura por lotes.
"""

from __future__ import division, print_function

import os
import re
import glob
import argparse

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import freeze_support
from fractions import Fraction

import numpy as np
import pandas as pd

from sklearn import preprocessing

from scipy.io import loadmat
from scipy.signal import resample_poly


# =====================================================================
# RUTAS DEL PROYECTO
# =====================================================================

# data2020.py está dentro de:
#
# proyecto/src/data2020.py
#
# Por tanto:
#   SRC_DIR      = proyecto/src
#   PROJECT_ROOT = proyecto

SRC_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.dirname(
    SRC_DIR
)

DEFAULT_DATA_DIR = os.path.join(
    PROJECT_ROOT,
    'dataset2020'
)

DEFAULT_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    'data'
)

DEFAULT_CLASSES_CSV = os.path.join(
    SRC_DIR,
    'dx_mapping_scored.csv'
)


# =====================================================================
# CONFIGURACIÓN GENERAL
# =====================================================================

TARGET_FS = 500

WINDOW_SECONDS = 10

WINDOW_LENGTH = (
    TARGET_FS * WINDOW_SECONDS
)

STANDARD_LEAD_ORDER = [
    'I',
    'II',
    'III',
    'aVR',
    'aVL',
    'aVF',
    'V1',
    'V2',
    'V3',
    'V4',
    'V5',
    'V6'
]

NUM_LEADS = len(
    STANDARD_LEAD_ORDER
)


# =====================================================================
# NORMALIZACIÓN DE DERIVACIONES
# =====================================================================

def normalize_lead_name(name):
    """
    Normaliza nombres de derivaciones.

    Ejemplos:

        I   -> I
        i   -> I
        V1  -> V1
        v1  -> V1
        AVR -> aVR
        AVL -> aVL
        AVF -> aVF
    """

    name = str(name).strip().upper()

    mapping = {
        'I': 'I',
        'II': 'II',
        'III': 'III',
        'AVR': 'aVR',
        'AVL': 'aVL',
        'AVF': 'aVF',
        'V1': 'V1',
        'V2': 'V2',
        'V3': 'V3',
        'V4': 'V4',
        'V5': 'V5',
        'V6': 'V6'
    }

    return mapping.get(
        name,
        str(name).strip()
    )


# =====================================================================
# PASO 2 — PARSEO DE .HEA
# =====================================================================

def parse_header(hea_path):
    """
    Parsea un archivo WFDB .hea y extrae metadata técnica y clínica.

    Soporta comentarios con formato:

        # Age: 74
        # Sex: Male
        # Dx: 59118001

    y también variantes sin espacio:

        #Age: 74
        #Sex: Male
        #Dx: 59118001
    """

    with open(
        hea_path,
        'r',
        encoding='utf-8-sig'
    ) as f:

        lines = [
            line.strip()
            for line in f
            if line.strip()
        ]

    if not lines:
        raise ValueError(
            'El archivo .hea está vacío.'
        )

    # -------------------------------------------------------------
    # Primera línea
    # -------------------------------------------------------------

    header = lines[0].split()

    if len(header) < 4:
        raise ValueError(
            f'Primera línea inválida en {hea_path}: '
            f'{lines[0]}'
        )

    record_name = header[0]
    num_leads = int(header[1])
    sampling_freq = float(header[2])
    num_samples = int(header[3])

    if len(lines) < num_leads + 1:
        raise ValueError(
            f'El header no contiene las {num_leads} '
            f'líneas de derivaciones esperadas.'
        )

    # -------------------------------------------------------------
    # Derivaciones
    # -------------------------------------------------------------

    lead_names = []

    for i in range(
        1,
        num_leads + 1
    ):

        tokens = lines[i].split()

        if not tokens:
            raise ValueError(
                f'Línea de derivación vacía en '
                f'{hea_path}, línea {i + 1}.'
            )

        lead_names.append(
            normalize_lead_name(
                tokens[-1]
            )
        )

    # -------------------------------------------------------------
    # Metadata clínica
    # -------------------------------------------------------------

    age = None
    sex = None
    dx_codes = []

    for line in lines[num_leads + 1:]:

        # ---------------------------------------------------------
        # Normalizar comentarios:
        #
        # "# Age: 74" -> "Age: 74"
        # "#Age: 74"  -> "Age: 74"
        # ---------------------------------------------------------

        if line.startswith('#'):

            comment = line[1:].strip()

        else:

            continue

        # ---------------------------------------------------------
        # Age
        # ---------------------------------------------------------

        if comment.lower().startswith('age:'):

            raw = comment.split(
                ':',
                1
            )[1].strip()

            if raw.lower() == 'nan':

                age = np.nan

            else:

                raw_clean = (
                    raw
                    .replace('>', '')
                    .replace('<', '')
                    .strip()
                )

                try:

                    parsed_age = float(raw_clean)

                    # Se consideran válidas únicamente
                    # edades dentro de un rango razonable.
                    if 0 <= parsed_age <= 120:
                        age = parsed_age
                    else:
                        age = np.nan
                except ValueError:

                    age = np.nan

        # ---------------------------------------------------------
        # Sex
        # ---------------------------------------------------------

        elif comment.lower().startswith('sex:'):

            sex = comment.split(
                ':',
                1
            )[1].strip()

        # ---------------------------------------------------------
        # Diagnosis
        # ---------------------------------------------------------

        elif comment.lower().startswith('dx:'):

            raw = comment.split(
                ':',
                1
            )[1].strip()

            dx_codes = [
                code.strip()
                for code in raw.split(',')
                if code.strip()
            ]

    return {
        'record_name': record_name,
        'num_leads': num_leads,
        'sampling_freq': sampling_freq,
        'num_samples': num_samples,
        'lead_names': lead_names,
        'age': age,
        'sex': sex,
        'dx_codes': dx_codes
    }


# =====================================================================
# REORDENAMIENTO DE DERIVACIONES
# =====================================================================

def reorder_leads(
    signal,
    lead_names
):
    """
    Reordena las derivaciones a:

        I, II, III, aVR, aVL, aVF,
        V1, V2, V3, V4, V5, V6
    """

    if signal.ndim != 2:

        raise ValueError(
            f'La señal debe ser 2D. '
            f'Shape: {signal.shape}'
        )

    normalized_names = [
        normalize_lead_name(name)
        for name in lead_names
    ]

    missing = [
        lead
        for lead in STANDARD_LEAD_ORDER
        if lead not in normalized_names
    ]

    if missing:

        raise ValueError(
            'Faltan derivaciones estándar.\n'
            f'Esperadas: {STANDARD_LEAD_ORDER}\n'
            f'Encontradas: {normalized_names}\n'
            f'Faltantes: {missing}'
        )

    indices = [
        normalized_names.index(lead)
        for lead in STANDARD_LEAD_ORDER
    ]

    reordered = signal[
        :,
        indices
    ]

    if reordered.shape[1] != NUM_LEADS:

        raise ValueError(
            f'Se esperaban {NUM_LEADS} derivaciones, '
            f'se obtuvieron {reordered.shape[1]}.'
        )

    return reordered


# =====================================================================
# PASO 3 — MAPEO SNOMED → 27 CLASES
# =====================================================================

def load_scored_classes(csv_path):
    """
    Carga el mapping de las 27 clases puntuadas del Challenge 2020.

    Cada código SNOMED conserva su propia clase.
    Las equivalencias indicadas en 'Notes' se utilizarán
    posteriormente durante la evaluación/scoring, no durante
    la creación de las etiquetas de entrenamiento.
    """

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f'No se encontró el archivo de clases:\n{csv_path}'
        )

    df = pd.read_csv(
        csv_path,
        dtype=str
    )

    required_columns = [
        'SNOMED CT Code',
        'Abbreviation'
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f'Falta la columna "{column}" en {csv_path}.'
            )

    df['SNOMED CT Code'] = (
        df['SNOMED CT Code']
        .astype(str)
        .str.strip()
    )

    df['Abbreviation'] = (
        df['Abbreviation']
        .astype(str)
        .str.strip()
    )

    classes = df['Abbreviation'].tolist()

    code_to_class_idx = {
        code: idx
        for idx, code in enumerate(
            df['SNOMED CT Code']
        )
    }

    if len(classes) != 27:
        raise ValueError(
            f'Se esperaban 27 clases, '
            f'pero el CSV contiene {len(classes)}.'
        )

    return classes, code_to_class_idx


# =====================================================================
# SNOMED → VECTOR MULTILABEL
# =====================================================================

def dx_codes_to_label_vector(
    dx_codes,
    code_to_class_idx,
    num_classes
):
    """
    Convierte códigos SNOMED a un vector multilabel.
    """

    y = np.zeros(
        num_classes,
        dtype=np.float32
    )

    for code in dx_codes:

        idx = code_to_class_idx.get(
            str(code).strip()
        )

        if idx is not None:

            y[idx] = 1.0

    return y


# =====================================================================
# PASO 4 — RESAMPLEO
# =====================================================================

def resample_signal(
    signal,
    orig_fs,
    target_fs=TARGET_FS
):
    """
    Resamplea mediante resample_poly().

    Soporta frecuencias enteras y decimales.
    """

    if signal.ndim != 2:

        raise ValueError(
            f'Señal inválida: {signal.shape}'
        )

    orig_fs = float(
        orig_fs
    )

    target_fs = float(
        target_fs
    )

    if orig_fs <= 0:

        raise ValueError(
            f'Frecuencia inválida: {orig_fs}'
        )

    if np.isclose(
        orig_fs,
        target_fs
    ):

        return signal

    ratio = Fraction(
        target_fs / orig_fs
    ).limit_denominator(
        10000
    )

    up = ratio.numerator
    down = ratio.denominator

    return resample_poly(
        signal,
        up,
        down,
        axis=0
    )


# =====================================================================
# PASO 5 — LONGITUD FIJA
# =====================================================================

def fix_length(
    signal,
    target_len=WINDOW_LENGTH,
    mode='center'
):
    """
    Ajusta la señal a target_len muestras.
    """

    if signal.ndim != 2:

        raise ValueError(
            f'Señal inválida: {signal.shape}'
        )

    current_len = signal.shape[0]

    if current_len == target_len:

        return signal

    if current_len > target_len:

        if mode == 'random':

            start = np.random.randint(
                0,
                current_len - target_len + 1
            )

        elif mode == 'center':

            start = (
                current_len - target_len
            ) // 2

        else:

            raise ValueError(
                f'Modo inválido: {mode}'
            )

        return signal[
            start:start + target_len,
            :
        ]

    # -------------------------------------------------------------
    # Zero-padding
    # -------------------------------------------------------------

    pad = (
        target_len - current_len
    )

    return np.pad(
        signal,
        (
            (0, pad),
            (0, 0)
        ),
        mode='constant'
    )


# =====================================================================
# INCART
# =====================================================================

def is_incart_record(
    source_db
):
    """
    Determina si una base corresponde a INCART.
    """

    if source_db is None:

        return False

    return (
        'incart'
        in str(source_db).lower()
    )


# =====================================================================
# SPLIT MULTILABEL
# =====================================================================

def stratified_multilabel_split(
    labels,
    source_dbs,
    val_frac=0.15,
    test_frac=0.15,
    random_state=42
):
    """
    Split aproximado:

        70 % train
        15 % validation
        15 % test
    """

    n = labels.shape[0]

    if n < 2:

        raise ValueError(
            'No hay suficientes registros.'
        )

    if (
        val_frac <= 0
        or test_frac <= 0
        or val_frac + test_frac >= 1
    ):

        raise ValueError(
            'Fracciones de split inválidas.'
        )

    indices = np.arange(n)

    try:

        from iterstrat.ml_stratifiers import (
            MultilabelStratifiedShuffleSplit
        )

        # ---------------------------------------------------------
        # Test
        # ---------------------------------------------------------

        split_test = (
            MultilabelStratifiedShuffleSplit(
                n_splits=1,
                test_size=test_frac,
                random_state=random_state
            )
        )

        trainval_idx, test_idx = next(
            split_test.split(
                np.zeros(
                    (n, 1)
                ),
                labels
            )
        )

        # ---------------------------------------------------------
        # Validation
        # ---------------------------------------------------------

        local_val_frac = (
            val_frac
            / (1 - test_frac)
        )

        split_val = (
            MultilabelStratifiedShuffleSplit(
                n_splits=1,
                test_size=local_val_frac,
                random_state=random_state
            )
        )

        train_sub, val_sub = next(
            split_val.split(
                np.zeros(
                    (
                        len(trainval_idx),
                        1
                    )
                ),
                labels[
                    trainval_idx
                ]
            )
        )

        train_idx = (
            trainval_idx[
                train_sub
            ]
        )

        val_idx = (
            trainval_idx[
                val_sub
            ]
        )

        test_idx = np.asarray(
            test_idx,
            dtype=int
        )

    except ImportError:

        print(
            'AVISO: iterative-stratification '
            'no está instalado.'
        )

        print(
            'Usando fallback por base de datos.'
        )

        rng = np.random.RandomState(
            random_state
        )

        source_dbs = np.asarray(
            source_dbs
        )

        train_idx = []
        val_idx = []
        test_idx = []

        for db in np.unique(
            source_dbs
        ):

            db_idx = indices[
                source_dbs == db
            ].copy()

            rng.shuffle(
                db_idx
            )

            n_db = len(
                db_idx
            )

            n_test = int(
                round(
                    n_db * test_frac
                )
            )

            n_val = int(
                round(
                    n_db * val_frac
                )
            )

            test_idx.extend(
                db_idx[
                    :n_test
                ]
            )

            val_idx.extend(
                db_idx[
                    n_test:
                    n_test + n_val
                ]
            )

            train_idx.extend(
                db_idx[
                    n_test + n_val:
                ]
            )

        train_idx = np.asarray(
            train_idx,
            dtype=int
        )

        val_idx = np.asarray(
            val_idx,
            dtype=int
        )

        test_idx = np.asarray(
            test_idx,
            dtype=int
        )

    return {
        'train': np.asarray(
            train_idx,
            dtype=int
        ),
        'val': np.asarray(
            val_idx,
            dtype=int
        ),
        'test': np.asarray(
            test_idx,
            dtype=int
        )
    }


# =====================================================================
# IDENTIFICAR BASE
# =====================================================================

def infer_source_db(
    hea_path,
    data_dir
):
    """
    Obtiene el nombre de la base.

    Se espera preferentemente:

        dataset2020/
            training/
                CPSC2018/
                    ...

    En ese caso devuelve:

        CPSC2018
    """

    relative_path = os.path.relpath(
        hea_path,
        data_dir
    )

    parts = relative_path.split(
        os.sep
    )

    parts_lower = [
        part.lower()
        for part in parts
    ]

    # -------------------------------------------------------------
    # Caso esperado:
    # training/<database>/...
    # -------------------------------------------------------------

    if 'training' in parts_lower:

        training_idx = (
            parts_lower.index(
                'training'
            )
        )

        if (
            training_idx + 1
            < len(parts)
        ):

            return parts[
                training_idx + 1
            ]

    # -------------------------------------------------------------
    # Fallback:
    #
    # Si no existe training, usamos
    # la primera carpeta del relative path.
    # -------------------------------------------------------------

    if len(parts) > 1:

        return parts[0]

    return 'Unknown'


# =====================================================================
# PASADA 1 — SCAN DE HEADERS
# =====================================================================

def scan_records(
    data_dir,
    scored_classes_csv
):
    """
    Escanea todos los .hea sin cargar señales .mat.
    """

    if not os.path.isdir(
        data_dir
    ):

        raise FileNotFoundError(
            f'No existe el directorio:\n'
            f'{data_dir}'
        )

    classes, code_to_class_idx = (
        load_scored_classes(
            scored_classes_csv
        )
    )

    num_classes = len(
        classes
    )

    if num_classes != 27:

        print(
            f'AVISO: CSV contiene '
            f'{num_classes} clases, '
            f'no 27.'
        )

    hea_files = sorted(
        glob.glob(
            os.path.join(
                data_dir,
                '**',
                '*.hea'
            ),
            recursive=True
        )
    )

    print(
        f'Encontrados {len(hea_files)} '
        f'archivos .hea.'
    )

    from tqdm import tqdm

    records = []

    n_missing_mat = 0
    n_bad_header = 0
    n_bad_leads = 0

    for hea_path in tqdm(
        hea_files,
        desc='Escaneando headers',
        unit='reg'
    ):

        base_path = os.path.splitext(
            hea_path
        )[0]

        mat_path = (
            base_path
            + '.mat'
        )

        if not os.path.exists(
            mat_path
        ):

            n_missing_mat += 1

            continue

        try:

            meta = parse_header(
                hea_path
            )

        except Exception as e:

            tqdm.write(
                f'AVISO header inválido '
                f'{hea_path}: {e}'
            )

            n_bad_header += 1

            continue

        # ---------------------------------------------------------
        # Leads
        # ---------------------------------------------------------

        if (
            meta['num_leads']
            != NUM_LEADS
        ):

            tqdm.write(
                f'AVISO {meta["record_name"]}: '
                f'{meta["num_leads"]} leads.'
            )

            n_bad_leads += 1

            continue

        missing = [
            lead
            for lead in STANDARD_LEAD_ORDER
            if lead not in meta['lead_names']
        ]

        if missing:

            tqdm.write(
                f'AVISO {meta["record_name"]}: '
                f'faltan {missing}'
            )

            n_bad_leads += 1

            continue

        # ---------------------------------------------------------
        # Labels
        # ---------------------------------------------------------

        label = (
            dx_codes_to_label_vector(
                meta['dx_codes'],
                code_to_class_idx,
                num_classes
            )
        )

        # ---------------------------------------------------------
        # Source DB
        # ---------------------------------------------------------

        source_db = infer_source_db(
            hea_path,
            data_dir
        )

        records.append({
            'hea_path': hea_path,
            'mat_path': mat_path,
            'label': label,
            'age': (
                meta['age']
                if meta['age'] is not None
                else np.nan
            ),
            'sex': (
                meta['sex']
                if meta['sex']
                else 'Unknown'
            ),
            'source_db': source_db,
            'record_name': meta['record_name'],
            'lead_names': meta['lead_names'],
            'sampling_freq': meta['sampling_freq']
        })

    print()
    print(
        'Escaneo completo:'
    )
    print(
        f'  Válidos            : {len(records)}'
    )
    print(
        f'  .mat faltantes     : {n_missing_mat}'
    )
    print(
        f'  Headers inválidos  : {n_bad_header}'
    )
    print(
        f'  Leads inválidas    : {n_bad_leads}'
    )
    print(
        f'  Clases             : {num_classes}'
    )

    return records, classes


# =====================================================================
# WORKER
# =====================================================================

def process_single_record(
    record
):
    """
    Procesa un único registro.

    .mat
      ↓
    loadmat
      ↓
    reorder
      ↓
    resample
      ↓
    fix_length
    """

    try:

        # ---------------------------------------------------------
        # Load MAT
        # ---------------------------------------------------------

        mat_data = loadmat(
            record['mat_path']
        )

        if 'val' not in mat_data:

            raise KeyError(
                "No existe 'val' en el .mat."
            )

        raw_signal = (
            mat_data['val']
            .astype(
                np.float32
            )
            .T
        )

        if raw_signal.ndim != 2:

            raise ValueError(
                f'Shape inesperado: '
                f'{raw_signal.shape}'
            )

        # ---------------------------------------------------------
        # Leads
        # ---------------------------------------------------------

        signal = reorder_leads(
            raw_signal,
            record['lead_names']
        )

        # ---------------------------------------------------------
        # Resample
        # ---------------------------------------------------------

        signal = resample_signal(
            signal,
            record['sampling_freq'],
            TARGET_FS
        )

        # ---------------------------------------------------------
        # Normalización z-score por derivación
        #
        # IMPORTANTE: se normaliza ANTES de fix_length (recorte/
        # relleno), no después. Varias bases (CPSC-Extra, Georgia)
        # tienen registros más cortos que la ventana de 5000 muestras
        # (ej. 6s = 3000 muestras contra una ventana de 10s) — si se
        # normalizara después del relleno con ceros, esos ceros de
        # padding diluirían el cálculo real de media/desvío por
        # derivación, y además la señal seguiría en unidades crudas
        # de ADC (valores grandes) al momento de escalar, lo que
        # combinado produce pérdida de precisión numérica (de ahí los
        # avisos de sklearn sobre valores demasiado grandes / desvío
        # cercano a cero). Normalizando antes del padding, el cálculo
        # de media/desvío usa solo muestras reales de señal, y el
        # relleno posterior queda en exactamente 0 (la media esperada
        # bajo z-score), que es un valor neutro razonable. Este orden
        # además coincide con el de utils.preprocess() usado en
        # inferencia, evitando reintroducir la inconsistencia
        # entrenamiento/inferencia que se había corregido antes.
        # ---------------------------------------------------------

        signal = preprocessing.scale(signal.astype(np.float64), axis=0)
        signal = np.asarray(signal, dtype=np.float32)

        # ---------------------------------------------------------
        # Longitud
        # ---------------------------------------------------------

        signal = fix_length(
            signal,
            WINDOW_LENGTH,
            mode='center'
        )

        signal = np.asarray(
            signal,
            dtype=np.float32
        )

        expected_shape = (
            WINDOW_LENGTH,
            NUM_LEADS
        )

        if signal.shape != expected_shape:

            raise ValueError(
                f'Shape final incorrecta: '
                f'{signal.shape}; '
                f'esperada: {expected_shape}'
            )

        return {
            'ok': True,
            'signal': signal,
            'label': record['label'],
            'age': record['age'],
            'sex': record['sex'],
            'source_db': record['source_db'],
            'record_name': record['record_name'],
            'error': None
        }

    except Exception as e:

        return {
            'ok': False,
            'signal': None,
            'label': None,
            'age': None,
            'sex': None,
            'source_db': record.get(
                'source_db',
                'Unknown'
            ),
            'record_name': record.get(
                'record_name',
                'Unknown'
            ),
            'error': str(e)
        }


# =====================================================================
# PASADA 2 — HDF5 + MULTIPROCESSING
# =====================================================================

def process_split_to_hdf5(
    records,
    indices,
    output_path,
    classes,
    num_workers=6,
    chunksize=8,
    write_batch_size=32
):
    """
    Procesa un split mediante multiprocessing.

    Los workers procesan señales.

    El proceso principal escribe HDF5.
    """

    import h5py
    from tqdm import tqdm

    if num_workers < 1:

        raise ValueError(
            'num_workers debe ser >= 1.'
        )

    str_dt = h5py.special_dtype(
        vlen=str
    )

    with h5py.File(
        output_path,
        'w'
    ) as f:

        # ---------------------------------------------------------
        # signals
        # ---------------------------------------------------------

        sig_ds = f.create_dataset(
            'signals',
            shape=(
                0,
                WINDOW_LENGTH,
                NUM_LEADS
            ),
            maxshape=(
                None,
                WINDOW_LENGTH,
                NUM_LEADS
            ),
            dtype='float32',
            chunks=(
                16,
                WINDOW_LENGTH,
                NUM_LEADS
            ),
            compression='gzip',
            compression_opts=4
        )

        # ---------------------------------------------------------
        # labels
        # ---------------------------------------------------------

        lbl_ds = f.create_dataset(
            'labels',
            shape=(
                0,
                len(classes)
            ),
            maxshape=(
                None,
                len(classes)
            ),
            dtype='float32'
        )

        # ---------------------------------------------------------
        # Metadata
        # ---------------------------------------------------------

        age_ds = f.create_dataset(
            'ages',
            shape=(0,),
            maxshape=(None,),
            dtype='float32'
        )

        sex_ds = f.create_dataset(
            'sexes',
            shape=(0,),
            maxshape=(None,),
            dtype=str_dt
        )

        src_ds = f.create_dataset(
            'source_dbs',
            shape=(0,),
            maxshape=(None,),
            dtype=str_dt
        )

        name_ds = f.create_dataset(
            'record_names',
            shape=(0,),
            maxshape=(None,),
            dtype=str_dt
        )

        # ---------------------------------------------------------
        # Attributes
        # ---------------------------------------------------------

        f.attrs['classes'] = classes
        f.attrs['sampling_rate'] = TARGET_FS
        f.attrs['window_seconds'] = WINDOW_SECONDS
        f.attrs['window_length'] = WINDOW_LENGTH
        f.attrs['num_leads'] = NUM_LEADS
        f.attrs['lead_order'] = ','.join(
            STANDARD_LEAD_ORDER
        )

        # ---------------------------------------------------------
        # Buffers
        # ---------------------------------------------------------

        batch_signals = []
        batch_labels = []
        batch_ages = []
        batch_sexes = []
        batch_sources = []
        batch_names = []

        n_ok = 0
        n_error = 0

        # ---------------------------------------------------------
        # Función de escritura
        # ---------------------------------------------------------

        def flush_batch():

            nonlocal n_ok

            if not batch_signals:
                return

            signals_np = np.stack(
                batch_signals
            ).astype(
                np.float32,
                copy=False
            )

            labels_np = np.stack(
                batch_labels
            ).astype(
                np.float32,
                copy=False
            )

            ages_np = np.asarray(
                batch_ages,
                dtype=np.float32
            )

            batch_size = len(
                batch_signals
            )

            old_size = n_ok

            new_size = (
                old_size
                + batch_size
            )

            # -----------------------------------------------------
            # Resize
            # -----------------------------------------------------

            sig_ds.resize(
                new_size,
                axis=0
            )

            lbl_ds.resize(
                new_size,
                axis=0
            )

            age_ds.resize(
                new_size,
                axis=0
            )

            sex_ds.resize(
                new_size,
                axis=0
            )

            src_ds.resize(
                new_size,
                axis=0
            )

            name_ds.resize(
                new_size,
                axis=0
            )

            # -----------------------------------------------------
            # Escritura
            # -----------------------------------------------------

            sig_ds[
                old_size:new_size
            ] = signals_np

            lbl_ds[
                old_size:new_size
            ] = labels_np

            age_ds[
                old_size:new_size
            ] = ages_np

            sex_ds[
                old_size:new_size
            ] = batch_sexes

            src_ds[
                old_size:new_size
            ] = batch_sources

            name_ds[
                old_size:new_size
            ] = batch_names

            n_ok = new_size

            # -----------------------------------------------------
            # Limpiar
            # -----------------------------------------------------

            batch_signals.clear()
            batch_labels.clear()
            batch_ages.clear()
            batch_sexes.clear()
            batch_sources.clear()
            batch_names.clear()

        # ---------------------------------------------------------
        # Registros
        # ---------------------------------------------------------

        selected_records = [
            records[i]
            for i in indices
        ]

        # ---------------------------------------------------------
        # Multiprocessing
        # ---------------------------------------------------------

        with ProcessPoolExecutor(
            max_workers=num_workers
        ) as executor:

            results = executor.map(
                process_single_record,
                selected_records,
                chunksize=chunksize
            )

            for result in tqdm(
                results,
                total=len(
                    selected_records
                ),
                desc=(
                    'Procesando '
                    + os.path.basename(
                        output_path
                    )
                ),
                unit='reg'
            ):

                if not result['ok']:

                    n_error += 1

                    tqdm.write(
                        f'AVISO: '
                        f'{result["record_name"]}: '
                        f'{result["error"]}'
                    )

                    continue

                batch_signals.append(
                    result['signal']
                )

                batch_labels.append(
                    result['label']
                )

                batch_ages.append(
                    result['age']
                )

                batch_sexes.append(
                    result['sex']
                )

                batch_sources.append(
                    result['source_db']
                )

                batch_names.append(
                    result['record_name']
                )

                if (
                    len(batch_signals)
                    >= write_batch_size
                ):

                    flush_batch()

        flush_batch()

    print(
        f'Guardado {output_path}: '
        f'{n_ok} OK, '
        f'{n_error} errores.'
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            'PhysioNet/CinC Challenge 2020 '
            'preprocessing.'
        )
    )

    # -------------------------------------------------------------
    # Rutas
    # -------------------------------------------------------------

    parser.add_argument(
        '--data_dir',
        type=str,
        default=DEFAULT_DATA_DIR,
        help=(
            'Directorio de dataset2020.'
        )
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            'Directorio de salida HDF5.'
        )
    )

    parser.add_argument(
        '--scored_classes_csv',
        type=str,
        default=DEFAULT_CLASSES_CSV,
        help=(
            'CSV de clases puntuadas.'
        )
    )

    # -------------------------------------------------------------
    # INCART
    # -------------------------------------------------------------

    parser.add_argument(
        '--exclude_incart_from_train',
        action='store_true',
        help=(
            'Excluye INCART de train/val/test '
            'y lo guarda como conjunto externo.'
        )
    )

    # -------------------------------------------------------------
    # Multiprocessing
    # -------------------------------------------------------------

    parser.add_argument(
        '--workers',
        type=int,
        default=6,
        help=(
            'Número de procesos. '
            'Recomendado para i5-11400H: 5-6.'
        )
    )

    parser.add_argument(
        '--chunksize',
        type=int,
        default=8
    )

    parser.add_argument(
        '--write_batch_size',
        type=int,
        default=32
    )

    # -------------------------------------------------------------
    # Split
    # -------------------------------------------------------------

    parser.add_argument(
        '--val_frac',
        type=float,
        default=0.15
    )

    parser.add_argument(
        '--test_frac',
        type=float,
        default=0.15
    )

    parser.add_argument(
        '--random_state',
        type=int,
        default=42
    )

    args = parser.parse_args()

    # -------------------------------------------------------------
    # Validaciones
    # -------------------------------------------------------------

    if args.workers < 1:

        parser.error(
            '--workers debe ser >= 1'
        )

    if args.chunksize < 1:

        parser.error(
            '--chunksize debe ser >= 1'
        )

    if args.write_batch_size < 1:

        parser.error(
            '--write_batch_size debe ser >= 1'
        )

    if (
        args.val_frac <= 0
        or args.test_frac <= 0
        or args.val_frac + args.test_frac >= 1
    ):

        parser.error(
            'Fracciones de split inválidas.'
        )

    # -------------------------------------------------------------
    # Convertir a rutas absolutas
    # -------------------------------------------------------------

    data_dir = os.path.abspath(
        args.data_dir
    )

    output_dir = os.path.abspath(
        args.output_dir
    )

    csv_path = os.path.abspath(
        args.scored_classes_csv
    )

    # -------------------------------------------------------------
    # Mostrar configuración
    # -------------------------------------------------------------

    print()
    print('=' * 70)
    print('PREPROCESAMIENTO CINC2020')
    print('=' * 70)

    print(
        f'Proyecto      : {PROJECT_ROOT}'
    )

    print(
        f'src           : {SRC_DIR}'
    )

    print(
        f'dataset       : {data_dir}'
    )

    print(
        f'CSV clases    : {csv_path}'
    )

    print(
        f'salida        : {output_dir}'
    )

    print(
        f'workers       : {args.workers}'
    )

    print(
        f'chunksize     : {args.chunksize}'
    )

    print(
        f'batch HDF5    : {args.write_batch_size}'
    )

    print(
        f'frecuencia    : {TARGET_FS} Hz'
    )

    print(
        f'duración      : {WINDOW_SECONDS} s'
    )

    print(
        f'muestras      : {WINDOW_LENGTH}'
    )

    print(
        f'derivaciones  : {NUM_LEADS}'
    )

    print('=' * 70)
    print()

    # -------------------------------------------------------------
    # PASADA 1
    # -------------------------------------------------------------

    print(
        'PASADA 1 — ESCANEO DE HEADERS'
    )

    records, classes = scan_records(
        data_dir,
        csv_path
    )

    if not records:

        raise RuntimeError(
            'No se encontraron registros válidos.'
        )

    # -------------------------------------------------------------
    # Labels
    # -------------------------------------------------------------

    labels = np.stack([
        record['label']
        for record in records
    ])

    source_dbs = [
        record['source_db']
        for record in records
    ]

    # -------------------------------------------------------------
    # Clases
    # -------------------------------------------------------------

    print()
    print(
        f'Total de clases: {len(classes)}'
    )

    for i, class_name in enumerate(
        classes
    ):

        print(
            f'  {i:02d}: {class_name}'
        )

    # -------------------------------------------------------------
    # SPLIT
    # -------------------------------------------------------------

    if args.exclude_incart_from_train:

        print()
        print(
            'INCART será separado '
            'como conjunto externo.'
        )

        is_incart = np.array([
            is_incart_record(db)
            for db in source_dbs
        ])

        incart_idx = np.where(
            is_incart
        )[0]

        rest_idx = np.where(
            ~is_incart
        )[0]

        if len(rest_idx) == 0:

            raise RuntimeError(
                'No quedaron registros fuera de INCART.'
            )

        local_splits = (
            stratified_multilabel_split(
                labels[rest_idx],
                [
                    source_dbs[i]
                    for i in rest_idx
                ],
                val_frac=args.val_frac,
                test_frac=args.test_frac,
                random_state=args.random_state
            )
        )

        splits = {
            name: rest_idx[idx]
            for name, idx
            in local_splits.items()
        }

        splits[
            'incart_external'
        ] = incart_idx

    else:

        splits = (
            stratified_multilabel_split(
                labels,
                source_dbs,
                val_frac=args.val_frac,
                test_frac=args.test_frac,
                random_state=args.random_state
            )
        )

    # -------------------------------------------------------------
    # Mostrar tamaños
    # -------------------------------------------------------------

    print()
    print('=' * 70)
    print('SPLITS')
    print('=' * 70)

    for split_name, idx in splits.items():

        print(
            f'{split_name:20s}: '
            f'{len(idx)} registros'
        )

    # -------------------------------------------------------------
    # Crear salida
    # -------------------------------------------------------------

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # -------------------------------------------------------------
    # PASADA 2
    # -------------------------------------------------------------

    print()
    print('=' * 70)
    print('PASADA 2 — PROCESAMIENTO DE SEÑALES')
    print('=' * 70)

    for split_name, idx in splits.items():

        output_path = os.path.join(
            output_dir,
            f'{split_name}.h5'
        )

        print()
        print(
            f'Procesando: {split_name}'
        )

        print(
            f'Registros: {len(idx)}'
        )

        process_split_to_hdf5(
            records=records,
            indices=idx,
            output_path=output_path,
            classes=classes,
            num_workers=args.workers,
            chunksize=args.chunksize,
            write_batch_size=args.write_batch_size
        )

    # -------------------------------------------------------------
    # FINAL
    # -------------------------------------------------------------

    print()
    print('=' * 70)
    print('PREPROCESAMIENTO TERMINADO')
    print('=' * 70)

    for split_name in splits:

        path = os.path.join(
            output_dir,
            f'{split_name}.h5'
        )

        if os.path.exists(path):

            size_mb = (
                os.path.getsize(path)
                / (1024 * 1024)
            )

            print(
                f'{split_name:20s}: '
                f'{size_mb:,.1f} MB'
            )

    print()
    print(
        f'Salida:\n{output_dir}'
    )
    print()


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == '__main__':

    # Necesario para multiprocessing en Windows.
    freeze_support()

    main()