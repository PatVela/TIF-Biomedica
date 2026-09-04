# -*- coding: utf-8 -*-

import torch

from config import get_config
from graph import ECG_model, count_parameters


def main():

    print("=" * 60)
    print("PRUEBA DEL MODELO ECG - CINC 2020")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Configuración
    # --------------------------------------------------------

    config = get_config()

    print("\nConfiguración:")
    print(f"  Derivaciones : {config.num_leads}")
    print(f"  Muestras     : {config.input_length}")
    print(f"  Clases       : {config.num_classes}")
    print(f"  Batch        : {config.batch}")

    # --------------------------------------------------------
    # 2. Seleccionar dispositivo
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nDispositivo:")
    print(f"  {device}")

    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # --------------------------------------------------------
    # 3. Crear modelo
    # --------------------------------------------------------

    model = ECG_model(config)
    model = model.to(device)

    print("\nModelo creado correctamente.")
    print(f"Parámetros entrenables: {count_parameters(model):,}")

    # --------------------------------------------------------
    # 4. Crear ECG ficticio
    # --------------------------------------------------------
    #
    # PyTorch Conv1D espera:
    #
    # (batch, canales, longitud)
    #
    # En nuestro caso:
    #
    # (8, 12, 5000)
    #

    batch_size = config.batch

    signals = torch.randn(
        batch_size,
        config.num_leads,
        config.input_length,
        device=device
    )

    print("\nEntrada:")
    print(f"  Shape: {signals.shape}")
    print(f"  Device: {signals.device}")

    # --------------------------------------------------------
    # 5. Forward pass
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():
        outputs = model(signals)

    print("\nSalida:")
    print(f"  Shape: {outputs.shape}")
    print(f"  Device: {outputs.device}")

    # --------------------------------------------------------
    # 6. Comprobar forma de salida
    # --------------------------------------------------------

    expected_shape = (
        batch_size,
        config.num_classes
    )

    assert outputs.shape == expected_shape, (
        f"Forma incorrecta. "
        f"Esperada: {expected_shape}, "
        f"obtenida: {outputs.shape}"
    )

    print("\n✓ La forma de salida es correcta.")

    # --------------------------------------------------------
    # 7. Probar BCEWithLogitsLoss
    # --------------------------------------------------------

    labels = torch.randint(
        0,
        2,
        (batch_size, config.num_classes),
        device=device
    ).float()

    criterion = torch.nn.BCEWithLogitsLoss()

    model.train()

    outputs = model(signals)

    loss = criterion(outputs, labels)

    print("\nPérdida:")
    print(f"  BCEWithLogitsLoss: {loss.item():.6f}")

    # --------------------------------------------------------
    # 8. Backpropagation
    # --------------------------------------------------------

    loss.backward()

    print("\n✓ Backpropagation ejecutado correctamente.")

    # --------------------------------------------------------
    # Resultado
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("PRUEBA COMPLETADA CORRECTAMENTE")
    print("=" * 60)


if __name__ == "__main__":
    main()