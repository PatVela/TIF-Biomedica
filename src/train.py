# -*- coding: utf-8 -*-

import os
import random
from xml.parsers.expat import model
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from config import get_config
from graph import ECG_model


# ============================================================
# REPRODUCIBILIDAD
# ============================================================

def set_seed(seed):
    """
    Establece las semillas aleatorias para favorecer
    la reproducibilidad de los experimentos.
    """

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
    Dataset de PyTorch para los archivos HDF5 generados
    por data2020.py.

    HDF5:

        signals -> (N, 5000, 12)
        labels  -> (N, 27)

    PyTorch Conv1D necesita:

        (batch, channels, length)

    Por ello cada ECG se transforma de:

        (5000, 12)

    a:

        (12, 5000)
    """

    def __init__(self, h5_path):

        import h5py

        if not os.path.exists(h5_path):
            raise FileNotFoundError(
                f"No se encontró el archivo HDF5: {h5_path}"
            )

        self.h5_path = h5_path

        # Abrimos temporalmente para obtener el tamaño.
        with h5py.File(self.h5_path, 'r') as h5:

            self.length = len(h5['signals'])

            self.signal_shape = h5['signals'].shape
            self.label_shape = h5['labels'].shape

            # Guardamos las clases almacenadas por data2020.py
            classes = h5.attrs.get('classes', None)

            if classes is not None:
                self.classes = [
                    c.decode('utf-8') if isinstance(c, bytes) else str(c)
                    for c in classes
                ]
            else:
                self.classes = None

        self.h5 = None

    def _open_file(self):
        """
        Abre el HDF5 únicamente cuando el proceso del DataLoader
        necesita acceder a los datos.
        """

        if self.h5 is None:
            import h5py
            self.h5 = h5py.File(self.h5_path, 'r')

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        self._open_file()

        signal = self.h5['signals'][index]
        label = self.h5['labels'][index]

        # HDF5:
        # (5000, 12)
        #
        # PyTorch Conv1D:
        # (12, 5000)

        signal = np.asarray(
            signal,
            dtype=np.float32
        ).T

        signal /= 1000.0

        label = np.asarray(
            label,
            dtype=np.float32
        )

        signal = torch.from_numpy(signal)
        label = torch.from_numpy(label)

        return signal, label

# ============================================================
# POS_WEIGHT PARA BCEWithLogitsLoss
# ============================================================

def compute_pos_weight(h5_path):
    """
    Calcula pos_weight para cada clase a partir de TRAIN.

    Fórmula:

        pos_weight = negativos / positivos

    Específicamente:

        pos_weight[c] = (N - P[c]) / P[c]

    donde:
        N    = número total de registros
        P[c] = positivos de la clase c

    Importante:
        SOLO se calcula usando train.h5.
        No se utiliza validation ni test.
    """

    import h5py

    if not os.path.exists(h5_path):
        raise FileNotFoundError(
            f"No se encontró TRAIN HDF5: {h5_path}"
        )

    with h5py.File(h5_path, "r") as h5:

        labels = h5["labels"]

        num_samples = labels.shape[0]

        # 30k x 27 es pequeño (~3.2 MB en float32),
        # por lo que podemos leerlo de una vez.
        label_matrix = np.asarray(
            labels[:],
            dtype=np.float32
        )

    # Número de positivos por clase
    positives = label_matrix.sum(axis=0)

    # Número de negativos por clase
    negatives = num_samples - positives

    # Verificación
    if np.any(positives <= 0):
        bad_classes = np.where(
            positives <= 0
        )[0].tolist()

        raise ValueError(
            "Hay clases sin positivos en train.h5: "
            f"{bad_classes}"
        )

    pos_weight = negatives / positives

    return torch.tensor(
        pos_weight,
        dtype=torch.float32
    ), positives, negatives

# ============================================================
# CREAR DATALOADERS
# ============================================================

def create_dataloaders(config):
    """
    Crea los DataLoader de entrenamiento y validación.
    """

    print("\nCargando datasets CINC2020...")

    train_dataset = ECGDataset(
        config.train_file
    )

    val_dataset = ECGDataset(
        config.val_file
    )

    print(f"Train: {len(train_dataset):,} ECGs")
    print(f"Val:   {len(val_dataset):,} ECGs")

    print(
        f"Forma señal: {train_dataset.signal_shape}"
    )

    print(
        f"Forma etiquetas: {train_dataset.label_shape}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    return train_loader, val_loader


# ============================================================
# TRAIN
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler=None
):
    """
    Entrena el modelo durante una época.
    """

    model.train()

    running_loss = 0.0
    total_samples = 0

    progress = tqdm(
        loader,
        desc="Training",
        leave=False
    )

    for signals, labels in progress:

        signals = signals.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # Automatic Mixed Precision
        # ----------------------------------------------------

        if scaler is not None:

            with torch.autocast(
                device_type='cuda',
                dtype=torch.float16
            ):

                outputs = model(signals)

                loss = criterion(
                    outputs,
                    labels
                )

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0
            )

            scaler.step(optimizer)

            scaler.update()

        else:

            outputs = model(signals)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0
            )

            optimizer.step()

        batch_size = signals.size(0)

        running_loss += (
            loss.item() * batch_size
        )

        total_samples += batch_size

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    epoch_loss = (
        running_loss / total_samples
    )

    return epoch_loss


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
    use_amp=False
):
    """
    Evalúa el modelo sobre el conjunto de validación.
    """

    model.eval()

    running_loss = 0.0
    total_samples = 0

    for signals, labels in tqdm(
        loader,
        desc="Validation",
        leave=False
    ):

        signals = signals.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        if use_amp:

            with torch.autocast(
                device_type='cuda',
                dtype=torch.float16
            ):

                outputs = model(signals)

                loss = criterion(
                    outputs,
                    labels
                )

        else:

            outputs = model(signals)

            loss = criterion(
                outputs,
                labels
            )

        batch_size = signals.size(0)

        running_loss += (
            loss.item() * batch_size
        )

        total_samples += batch_size

    val_loss = (
        running_loss / total_samples
    )

    return val_loss


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    val_loss,
    pos_weight=None
):
    """
    Guarda un checkpoint completo de PyTorch.
    """

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict":
                scheduler.state_dict()
                if scheduler is not None
                else None,
            "val_loss": val_loss,
            "pos_weight":
                pos_weight.detach().cpu()
                if pos_weight is not None
                else None,
        },
        path
    )


# ============================================================
# CARGAR CHECKPOINT
# ============================================================

def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    device='cuda'
):
    """
    Carga un checkpoint previamente guardado.
    """

    checkpoint = torch.load(
        path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint['model_state_dict']
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint['optimizer_state_dict']
        )

    if (
        scheduler is not None
        and checkpoint.get('scheduler_state_dict') is not None
    ):
        scheduler.load_state_dict(
            checkpoint['scheduler_state_dict']
        )

    epoch = checkpoint.get(
        'epoch',
        0
    )

    val_loss = checkpoint.get(
        'val_loss',
        None
    )

    return epoch, val_loss


# ============================================================
# ENTRENAMIENTO COMPLETO
# ============================================================

def train(config):

    # --------------------------------------------------------
    # Semilla
    # --------------------------------------------------------

    set_seed(config.seed)

    # --------------------------------------------------------
    # Dispositivo
    # --------------------------------------------------------

    if config.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')

    else:
        device = torch.device('cpu')

    print("\n" + "=" * 60)
    print("ENTRENAMIENTO ECG - PhysioNet/CinC 2020")
    print("=" * 60)

    print(f"PyTorch: {torch.__version__}")
    print(f"Dispositivo: {device}")

    if torch.cuda.is_available():

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

        print(
            f"CUDA: {torch.version.cuda}"
        )

    print("=" * 60)

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader, val_loader = create_dataloaders(
        config
    )

    # --------------------------------------------------------
    # Modelo
    # --------------------------------------------------------

    model = ECG_model(config)

    model = model.to(device)

    print(
        f"\nParámetros entrenables: "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # --------------------------------------------------------
    # Función de pérdida
    # --------------------------------------------------------

    # --------------------------------------------------------
    # POS_WEIGHT
    # --------------------------------------------------------

    pos_weight, positives, negatives = compute_pos_weight(
        config.train_file
    )

    print("\nPositivos por clase:")
    for i, count in enumerate(positives):
        print(
            f"  {i:02d} "
            f"{count:8.0f} positivos "
            f"weight={pos_weight[i].item():.4f}"
        )

    pos_weight = pos_weight.to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    # --------------------------------------------------------
    # Optimizador
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        min_lr=config.min_lr
    )

    # --------------------------------------------------------
    # AMP
    # --------------------------------------------------------

    scaler = None

    if config.amp and device.type == 'cuda':

        scaler = torch.amp.GradScaler(
            'cuda'
        )

        print("AMP: activado")

    else:

        print("AMP: desactivado")

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    os.makedirs(
        'modelos',
        exist_ok=True
    )

    initial_epoch = 0
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    if config.checkpoint_path is not None:

        print(
            f"\nCargando checkpoint: "
            f"{config.checkpoint_path}"
        )

        initial_epoch, checkpoint_loss = load_checkpoint(
            config.checkpoint_path,
            model,
            optimizer,
            scheduler,
            device
        )

        if checkpoint_loss is not None:
            best_val_loss = checkpoint_loss

        print(
            f"Reanudando desde época "
            f"{initial_epoch + 1}"
        )

    # --------------------------------------------------------
    # Bucle principal
    # --------------------------------------------------------

    for epoch in range(
        initial_epoch,
        config.epochs
    ):

        print(
            f"\nÉpoca "
            f"{epoch + 1}/{config.epochs}"
        )

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_loss = validate(
            model,
            val_loader,
            criterion,
            device,
            use_amp=(
                config.amp
                and device.type == 'cuda'
            )
        )

        # ----------------------------------------------------
        # Learning rate
        # ----------------------------------------------------

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"Train Loss: {train_loss:.6f}"
        )

        print(
            f"Val Loss:   {val_loss:.6f}"
        )

        print(
            f"Learning Rate: {current_lr:.8f}"
        )

        # ----------------------------------------------------
        # Checkpoint latest
        # ----------------------------------------------------

        latest_path = os.path.join(
            'modelos',
            'cinc2020-latest.pt'
        )

        save_checkpoint(
            latest_path,
            model,
            optimizer,
            scheduler,
            epoch + 1,
            val_loss,
            pos_weight
        )

        # ----------------------------------------------------
        # Mejor modelo
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            epochs_without_improvement = 0

            best_path = os.path.join(
                'modelos',
                'cinc2020-best.pt'
            )

            save_checkpoint(
                best_path,
                model,
                optimizer,
                scheduler,
                epoch + 1,
                val_loss,
                pos_weight
            )

            print(
                "✓ Nuevo mejor modelo guardado."
            )

        else:

            epochs_without_improvement += 1

            print(
                f"Sin mejora: "
                f"{epochs_without_improvement}/"
                f"{config.patience}"
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= config.patience
        ):

            print(
                "\nEarly Stopping activado."
            )

            break

    print("\nEntrenamiento finalizado.")

    return model


# ============================================================
# MAIN
# ============================================================

def main():

    config = get_config()

    train(config)


if __name__ == '__main__':

    main()