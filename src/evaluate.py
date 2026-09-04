# -*- coding: utf-8 -*-
"""
evaluate.py

Evaluación completa de un modelo CINC2020 multilabel entrenado con PyTorch.

Flujo metodológico:
    1. Cargar el mejor checkpoint.
    2. Generar probabilidades sobre validation.
    3. Seleccionar un threshold por clase usando SOLO validation.
    4. Evaluar test usando esos thresholds.
    5. Evaluar INCART externo usando los mismos thresholds.
    6. Guardar métricas, thresholds y predicciones en resultados/.

Uso desde la raíz del proyecto:

    python src/evaluate.py

Opcional:

    python src/evaluate.py --checkpoint modelos/cinc2020-best.pt
    python src/evaluate.py --threshold-method f1

El script asume:
    signals: (N, 5000, 12)
    labels:  (N, 27)

Los HDF5 actuales contienen valores digitales. Se convierten a mV
mediante /1000, igual que en train.py.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import get_config
from graph import ECG_model


# ============================================================
# CONFIGURACIÓN
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "resultados"

STANDARD_LEADS = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
]

DEFAULT_CHECKPOINT = PROJECT_ROOT / "modelos" / "cinc2020-best.pt"
DEFAULT_THRESHOLD_METHOD = "f1"


# ============================================================
# REPRODUCIBILIDAD
# ============================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATASET HDF5
# ============================================================

class ECGDataset(Dataset):
    """
    Dataset lazy para HDF5 CINC2020.

    Convierte:
        (5000, 12) -> (12, 5000)

    y transforma valores digitales a mV mediante /1000.
    """

    def __init__(self, h5_path: str | Path):
        self.h5_path = str(h5_path)

        if not os.path.exists(self.h5_path):
            raise FileNotFoundError(
                f"No se encontró el archivo HDF5: {self.h5_path}"
            )

        with h5py.File(self.h5_path, "r") as h5:
            self.length = h5["signals"].shape[0]
            self.signal_shape = tuple(h5["signals"].shape)
            self.label_shape = tuple(h5["labels"].shape)
            classes = h5.attrs.get("classes", None)

            if classes is not None:
                self.classes = [
                    c.decode("utf-8") if isinstance(c, bytes) else str(c)
                    for c in classes
                ]
            else:
                self.classes = None

        self.h5 = None

    def _open_file(self):
        if self.h5 is None:
            self.h5 = h5py.File(self.h5_path, "r")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        self._open_file()

        signal = np.asarray(
            self.h5["signals"][index],
            dtype=np.float32,
        ).T

        # HDF5 actual: digital values.
        # gain audit showed gain=1000, baseline=0, units=mV.
        signal /= 1000.0

        label = np.asarray(
            self.h5["labels"][index],
            dtype=np.float32,
        )

        return (
            torch.from_numpy(signal),
            torch.from_numpy(label),
        )

    def __del__(self):
        try:
            if self.h5 is not None:
                self.h5.close()
        except Exception:
            pass


# ============================================================
# CARGA DE CHECKPOINT
# ============================================================

def load_model(config, checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No se encontró checkpoint: {checkpoint_path}"
        )

    model = ECG_model(config).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    epoch = checkpoint.get("epoch", None)
    val_loss = checkpoint.get("val_loss", None)

    return model, checkpoint, epoch, val_loss


# ============================================================
# INFERENCIA
# ============================================================

@torch.no_grad()
def predict_dataset(
    model,
    dataset: ECGDataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    device: torch.device,
    amp: bool,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    all_probs = []
    all_labels = []

    use_amp = amp and device.type == "cuda"

    for signals, labels in tqdm(
        loader,
        desc=f"Evaluando {Path(dataset.h5_path).name}",
        leave=False,
    ):
        signals = signals.to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(signals)
            probs = torch.sigmoid(logits)

        all_probs.append(
            probs.float().cpu().numpy()
        )
        all_labels.append(
            labels.cpu().numpy()
        )

    probabilities = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    return probabilities, labels


# ============================================================
# THRESHOLD POR CLASE
# ============================================================

def find_best_threshold_f1(
    y_true: np.ndarray,
    y_prob: np.ndarray,
):
    """
    Busca threshold que maximiza F1 para una clase.

    La búsqueda se hace exclusivamente sobre validation.
    """

    candidates = np.linspace(
        0.01,
        0.99,
        99,
    )

    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in candidates:
        y_pred = (y_prob >= threshold).astype(np.uint8)

        score = f1_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)

    return best_threshold, best_f1


def optimize_thresholds_f1(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
):
    thresholds = np.zeros(
        y_true.shape[1],
        dtype=np.float32,
    )

    validation_f1 = np.zeros(
        y_true.shape[1],
        dtype=np.float32,
    )

    rows = []

    for i, name in enumerate(class_names):
        threshold, score = find_best_threshold_f1(
            y_true[:, i],
            y_prob[:, i],
        )

        thresholds[i] = threshold
        validation_f1[i] = score

        rows.append(
            {
                "index": i,
                "class": name,
                "threshold": threshold,
                "validation_f1": score,
                "validation_positives": int(y_true[:, i].sum()),
            }
        )

    return thresholds, pd.DataFrame(rows)


# ============================================================
# MÉTRICAS
# ============================================================

def safe_auroc(y_true, y_prob):
    try:
        # Necesita ambas clases presentes.
        if len(np.unique(y_true)) < 2:
            return np.nan
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return np.nan


def safe_auprc(y_true, y_prob):
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return float(average_precision_score(y_true, y_prob))
    except Exception:
        return np.nan


def specificity_from_cm(cm):
    tn, fp, fn, tp = cm.ravel()
    denominator = tn + fp
    if denominator == 0:
        return np.nan
    return float(tn / denominator)


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
    class_names: list[str],
    split_name: str,
):
    y_pred = (
        y_prob >= thresholds[None, :]
    ).astype(np.uint8)

    rows = []

    for i, name in enumerate(class_names):
        true_i = y_true[:, i].astype(np.uint8)
        pred_i = y_pred[:, i]
        prob_i = y_prob[:, i]

        cm = confusion_matrix(
            true_i,
            pred_i,
            labels=[0, 1],
        )

        tn, fp, fn, tp = cm.ravel()

        rows.append(
            {
                "split": split_name,
                "index": i,
                "class": name,
                "threshold": float(thresholds[i]),
                "support_positive": int(true_i.sum()),
                "support_negative": int(len(true_i) - true_i.sum()),
                "TP": int(tp),
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "precision": float(
                    precision_score(
                        true_i,
                        pred_i,
                        zero_division=0,
                    )
                ),
                "sensitivity_recall": float(
                    recall_score(
                        true_i,
                        pred_i,
                        zero_division=0,
                    )
                ),
                "specificity": specificity_from_cm(cm),
                "f1": float(
                    f1_score(
                        true_i,
                        pred_i,
                        zero_division=0,
                    )
                ),
                "auroc": safe_auroc(
                    true_i,
                    prob_i,
                ),
                "auprc": safe_auprc(
                    true_i,
                    prob_i,
                ),
            }
        )

    df = pd.DataFrame(rows)

    # Métricas globales basadas en las mismas predicciones.
    global_metrics = {
        "split": split_name,
        "samples": int(y_true.shape[0]),
        "classes": int(y_true.shape[1]),
        "f1_macro": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_micro": float(
            f1_score(
                y_true,
                y_pred,
                average="micro",
                zero_division=0,
            )
        ),
        "precision_macro": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "precision_micro": float(
            precision_score(
                y_true,
                y_pred,
                average="micro",
                zero_division=0,
            )
        ),
        "sensitivity_macro": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "sensitivity_micro": float(
            recall_score(
                y_true,
                y_pred,
                average="micro",
                zero_division=0,
            )
        ),
        "auroc_macro": safe_macro_metric(
            y_true,
            y_prob,
            roc_auc_score,
        ),
        "auprc_macro": safe_macro_metric(
            y_true,
            y_prob,
            average_precision_score,
        ),
    }

    global_metrics["specificity_macro"] = float(
        df["specificity"].mean()
    ) if len(df) else np.nan

    return df, global_metrics, y_pred


def safe_macro_metric(y_true, y_prob, metric_fn):
    values = []

    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) < 2:
            continue

        try:
            values.append(
                float(
                    metric_fn(
                        y_true[:, i],
                        y_prob[:, i],
                    )
                )
            )
        except Exception:
            continue

    if not values:
        return np.nan

    return float(np.mean(values))


# ============================================================
# RESUMEN ADICIONAL
# ============================================================

def dataset_summary(y_true, class_names, split_name):
    positives = y_true.sum(axis=0)
    negatives = y_true.shape[0] - positives

    return pd.DataFrame(
        {
            "split": split_name,
            "index": np.arange(len(class_names)),
            "class": class_names,
            "positives": positives.astype(int),
            "negatives": negatives.astype(int),
            "prevalence": positives / y_true.shape[0],
        }
    )


def print_global_metrics(metrics):
    print("\nMÉTRICAS GLOBALES")
    print("-" * 60)

    for key, value in metrics.items():
        if key == "split":
            continue

        if isinstance(value, (float, np.floating)):
            if np.isnan(value):
                text = "NaN"
            else:
                text = f"{value:.6f}"
        else:
            text = str(value)

        print(f"{key:<25}: {text}")


def print_class_metrics(df):
    print("\nMÉTRICAS POR CLASE")
    print("-" * 110)

    columns = [
        "index",
        "class",
        "threshold",
        "support_positive",
        "precision",
        "sensitivity_recall",
        "specificity",
        "f1",
        "auroc",
        "auprc",
    ]

    display = df[columns].copy()

    print(
        display.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


# ============================================================
# GUARDAR RESULTADOS
# ============================================================

def save_json(path: Path, data):
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return None if np.isnan(obj) else float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            default=convert,
        ),
        encoding="utf-8",
    )


def save_predictions(path: Path, y_true, y_prob, y_pred, class_names):
    payload = {}

    for i, name in enumerate(class_names):
        payload[f"true_{name}"] = y_true[:, i].astype(np.uint8)
        payload[f"prob_{name}"] = y_prob[:, i].astype(np.float32)
        payload[f"pred_{name}"] = y_pred[:, i].astype(np.uint8)

    pd.DataFrame(payload).to_csv(
        path,
        index=False,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evalúa el modelo CINC2020 y optimiza thresholds "
            "por clase usando validation."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="Checkpoint .pt a evaluar.",
    )

    parser.add_argument(
        "--threshold-method",
        choices=["f1"],
        default=DEFAULT_THRESHOLD_METHOD,
        help="Método para seleccionar thresholds en validation.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch de evaluación. Por defecto usa config.batch.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Workers de DataLoader. Por defecto usa config.num_workers.",
    )

    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Desactiva AMP durante inferencia.",
    )

    args = parser.parse_args()

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = get_config()
    set_seed(config.seed)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if config.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else config.batch
    )

    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else config.num_workers
    )

    pin_memory = bool(config.pin_memory)
    use_amp = bool(config.amp) and not args.no_amp

    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

    print("\n" + "=" * 70)
    print("EVALUACIÓN ECG - PhysioNet/CinC 2020")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dispositivo: {device}")
    print(f"Batch: {batch_size}")
    print(f"Workers: {num_workers}")
    print(f"AMP: {'activado' if use_amp else 'desactivado'}")
    print("=" * 70)

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )
        print(
            f"CUDA: {torch.version.cuda}"
        )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model, checkpoint, epoch, checkpoint_val_loss = load_model(
        config,
        checkpoint_path,
        device,
    )

    print(
        f"\nCheckpoint cargado: época {epoch}"
    )

    if checkpoint_val_loss is not None:
        print(
            f"Val Loss almacenada: {checkpoint_val_loss:.6f}"
        )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_file = Path(config.train_file)
    val_file = Path(config.val_file)
    test_file = Path(config.test_file)

    # INCART puede no existir en algunas configuraciones.
    incart_file = PROJECT_ROOT / "data" / "incart_external.h5"

    val_dataset = ECGDataset(val_file)
    test_dataset = ECGDataset(test_file)

    if val_dataset.classes is not None:
        class_names = val_dataset.classes
    elif test_dataset.classes is not None:
        class_names = test_dataset.classes
    else:
        raise ValueError(
            "Los HDF5 no contienen el atributo 'classes'."
        )

    if len(class_names) != 27:
        raise ValueError(
            f"Se esperaban 27 clases, se encontraron {len(class_names)}."
        )

    print(
        f"\nValidation: {len(val_dataset):,} ECGs"
    )
    print(
        f"Test:       {len(test_dataset):,} ECGs"
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    val_prob, val_true = predict_dataset(
        model,
        val_dataset,
        batch_size,
        num_workers,
        pin_memory,
        device,
        use_amp,
    )

    # --------------------------------------------------------
    # Thresholds
    # --------------------------------------------------------

    if args.threshold_method == "f1":
        thresholds, threshold_df = optimize_thresholds_f1(
            val_true,
            val_prob,
            class_names,
        )
    else:
        raise ValueError(
            f"Método no soportado: {args.threshold_method}"
        )

    threshold_df.to_csv(
        RESULTS_DIR / "thresholds_validation.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Métricas validation con thresholds encontrados
    # --------------------------------------------------------

    val_class_df, val_global, val_pred = evaluate_predictions(
        val_true,
        val_prob,
        thresholds,
        class_names,
        "validation",
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_prob, test_true = predict_dataset(
        model,
        test_dataset,
        batch_size,
        num_workers,
        pin_memory,
        device,
        use_amp,
    )

    test_class_df, test_global, test_pred = evaluate_predictions(
        test_true,
        test_prob,
        thresholds,
        class_names,
        "test",
    )

    # --------------------------------------------------------
    # INCART externo
    # --------------------------------------------------------

    incart_global = None
    incart_class_df = None

    if incart_file.exists():
        incart_dataset = ECGDataset(incart_file)

        incart_prob, incart_true = predict_dataset(
            model,
            incart_dataset,
            batch_size,
            num_workers,
            pin_memory,
            device,
            use_amp,
        )

        incart_class_df, incart_global, incart_pred = evaluate_predictions(
            incart_true,
            incart_prob,
            thresholds,
            class_names,
            "incart_external",
        )

        save_predictions(
            RESULTS_DIR / "predictions_incart_external.csv",
            incart_true,
            incart_prob,
            incart_pred,
            class_names,
        )

    else:
        print(
            f"\nAVISO: no existe {incart_file}; "
            "se omite evaluación externa."
        )

    # --------------------------------------------------------
    # Guardar métricas
    # --------------------------------------------------------

    all_class_df = pd.concat(
        [
            val_class_df,
            test_class_df,
            incart_class_df,
        ] if incart_class_df is not None else [
            val_class_df,
            test_class_df,
        ],
        ignore_index=True,
    )

    all_class_df.to_csv(
        RESULTS_DIR / "metrics_per_class.csv",
        index=False,
    )

    global_rows = [
        val_global,
        test_global,
    ]

    if incart_global is not None:
        global_rows.append(incart_global)

    global_df = pd.DataFrame(global_rows)
    global_df.to_csv(
        RESULTS_DIR / "metrics_global.csv",
        index=False,
    )

    # Dataset summaries
    val_summary = dataset_summary(
        val_true,
        class_names,
        "validation",
    )
    test_summary = dataset_summary(
        test_true,
        class_names,
        "test",
    )

    summaries = [val_summary, test_summary]

    if incart_class_df is not None:
        summaries.append(
            dataset_summary(
                incart_true,
                class_names,
                "incart_external",
            )
        )

    pd.concat(
        summaries,
        ignore_index=True,
    ).to_csv(
        RESULTS_DIR / "class_support.csv",
        index=False,
    )

    # Predicciones principales
    save_predictions(
        RESULTS_DIR / "predictions_test.csv",
        test_true,
        test_prob,
        test_pred,
        class_names,
    )

    # --------------------------------------------------------
    # JSON resumen
    # --------------------------------------------------------

    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": epoch,
        "checkpoint_val_loss": checkpoint_val_loss,
        "threshold_method": args.threshold_method,
        "class_names": class_names,
        "thresholds": thresholds,
        "validation": val_global,
        "test": test_global,
        "incart_external": incart_global,
        "files": {
            "thresholds": str(RESULTS_DIR / "thresholds_validation.csv"),
            "metrics_per_class": str(RESULTS_DIR / "metrics_per_class.csv"),
            "metrics_global": str(RESULTS_DIR / "metrics_global.csv"),
            "predictions_test": str(RESULTS_DIR / "predictions_test.csv"),
            "predictions_incart": (
                str(RESULTS_DIR / "predictions_incart_external.csv")
                if incart_class_df is not None
                else None
            ),
        },
    }

    save_json(
        RESULTS_DIR / "evaluation_summary.json",
        summary,
    )

    # --------------------------------------------------------
    # Mostrar resultados
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VALIDATION — THRESHOLDS OPTIMIZADOS")
    print("=" * 70)
    print_global_metrics(val_global)
    print_class_metrics(val_class_df)

    print("\n" + "=" * 70)
    print("TEST — EVALUACIÓN FINAL")
    print("=" * 70)
    print_global_metrics(test_global)
    print_class_metrics(test_class_df)

    if incart_global is not None:
        print("\n" + "=" * 70)
        print("INCART — EVALUACIÓN EXTERNA")
        print("=" * 70)
        print_global_metrics(incart_global)
        print_class_metrics(incart_class_df)

    print("\n" + "=" * 70)
    print("EVALUACIÓN TERMINADA")
    print("=" * 70)
    print(f"Resultados guardados en: {RESULTS_DIR}")
    print("\nArchivos principales:")
    print("  - thresholds_validation.csv")
    print("  - metrics_per_class.csv")
    print("  - metrics_global.csv")
    print("  - predictions_test.csv")
    if incart_class_df is not None:
        print("  - predictions_incart_external.csv")
    print("  - evaluation_summary.json")


if __name__ == "__main__":
    main()
