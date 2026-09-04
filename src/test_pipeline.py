# -*- coding: utf-8 -*-

import torch

from config import get_config
from graph import ECG_model
from train import ECGDataset
from torch.utils.data import DataLoader


def main():

    config = get_config()

    print("\n" + "=" * 60)
    print("PRUEBA DEL PIPELINE - CINC2020 + PyTorch")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. GPU
    # --------------------------------------------------------

    print("\n[1] Comprobando GPU...")

    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("CUDA disponible:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA no está disponible."
        )

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    device = torch.device("cuda")

    # --------------------------------------------------------
    # 2. Dataset
    # --------------------------------------------------------

    print("\n[2] Cargando dataset CINC2020...")

    dataset = ECGDataset(
        config.train_file
    )

    print("Número de ECGs:", len(dataset))
    print("Forma HDF5:", dataset.signal_shape)
    print("Forma etiquetas:", dataset.label_shape)

    # --------------------------------------------------------
    # 3. Primer ECG
    # --------------------------------------------------------

    print("\n[3] Leyendo primer ECG...")

    signal, label = dataset[0]

    print("Signal:")
    print("  Shape:", signal.shape)
    print("  dtype:", signal.dtype)
    print("  min:", signal.min().item())
    print("  max:", signal.max().item())

    print("\nLabel:")
    print("  Shape:", label.shape)
    print("  dtype:", label.dtype)
    print("  Valores:", label.tolist())

    # --------------------------------------------------------
    # 4. DataLoader
    # --------------------------------------------------------

    print("\n[4] Creando DataLoader...")

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    signals, labels = next(iter(loader))

    print("Batch:")
    print("  Signals:", signals.shape)
    print("  Labels:", labels.shape)

    # --------------------------------------------------------
    # 5. Comprobar formas
    # --------------------------------------------------------

    expected_signal_shape = (
        8,
        config.num_leads,
        config.input_length
    )

    expected_label_shape = (
        8,
        config.num_classes
    )

    if tuple(signals.shape) != expected_signal_shape:

        raise RuntimeError(
            f"Forma incorrecta de signals: "
            f"{signals.shape}. "
            f"Se esperaba {expected_signal_shape}."
        )

    if tuple(labels.shape) != expected_label_shape:

        raise RuntimeError(
            f"Forma incorrecta de labels: "
            f"{labels.shape}. "
            f"Se esperaba {expected_label_shape}."
        )

    print("✓ Las formas son correctas.")

    # --------------------------------------------------------
    # 6. Crear modelo
    # --------------------------------------------------------

    print("\n[5] Creando modelo...")

    model = ECG_model(config)

    model = model.to(device)

    print(
        "Modelo en:",
        next(model.parameters()).device
    )

    # --------------------------------------------------------
    # 7. Enviar batch a GPU
    # --------------------------------------------------------

    print("\n[6] Enviando batch a GPU...")

    signals = signals.to(
        device,
        non_blocking=True
    )

    labels = labels.to(
        device,
        non_blocking=True
    )

    print("Signals:", signals.device)
    print("Labels:", labels.device)

    # --------------------------------------------------------
    # 8. Forward pass
    # --------------------------------------------------------

    print("\n[7] Ejecutando forward pass...")

    model.eval()

    with torch.no_grad():

        outputs = model(signals)

    print("Output shape:", outputs.shape)
    print("Output device:", outputs.device)
    print("Output dtype:", outputs.dtype)

    # --------------------------------------------------------
    # 9. Comprobar salida
    # --------------------------------------------------------

    expected_output_shape = (
        8,
        config.num_classes
    )

    if tuple(outputs.shape) != expected_output_shape:

        raise RuntimeError(
            f"Forma incorrecta de salida: "
            f"{outputs.shape}. "
            f"Se esperaba {expected_output_shape}."
        )

    print("✓ La salida tiene 27 clases.")

    # --------------------------------------------------------
    # 10. Sigmoid solamente para inspección
    # --------------------------------------------------------

    probabilities = torch.sigmoid(outputs)

    print("\nProbabilidades del primer ECG:")

    print(
        probabilities[0].detach().cpu().numpy()
    )

    # --------------------------------------------------------
    # Resultado
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("✓ PRUEBA COMPLETADA CORRECTAMENTE")
    print("=" * 60)

    print("\nPipeline comprobado:")

    print(
        "HDF5 → Dataset → DataLoader → "
        "CNN → RTX 3050 → 27 salidas"
    )


if __name__ == "__main__":
    main()