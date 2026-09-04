# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class ECGResNet(nn.Module):
    """
    Red neuronal convolucional residual (ResNet 1D) para clasificación
    multilabel de ECG del PhysioNet/CinC Challenge 2020.

    Entrada:
        (batch, 12, 5000)

    Salida:
        (batch, 27)

    Las 27 salidas corresponden a las clases diagnósticas puntuadas
    de CinC 2020.
    """

    def __init__(
        self,
        num_leads=12,
        num_classes=27,
        filter_length=32,
        kernel_size=7,
        drop_rate=0.2
    ):
        super().__init__()

        self.num_leads = num_leads
        self.num_classes = num_classes

        # ----------------------------------------------------
        # Primer bloque convolucional
        # ----------------------------------------------------

        self.first_conv = nn.Conv1d(
            in_channels=num_leads,
            out_channels=filter_length,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2
        )

        self.first_bn = nn.BatchNorm1d(filter_length)
        self.first_relu = nn.ReLU(inplace=True)

        # ----------------------------------------------------
        # Bloques residuales
        # ----------------------------------------------------

        self.blocks = nn.ModuleList()

        current_filters = filter_length

        for block_index in range(15):

            # Igual que en la arquitectura original:
            # reducción cada dos bloques.
            stride = 2 if block_index % 2 == 0 else 1

            # Cada 4 bloques se duplican los filtros.
            if block_index % 4 == 0 and block_index > 0:
                next_filters = current_filters * 2
            else:
                next_filters = current_filters

            block = ResidualBlock(
                in_channels=current_filters,
                out_channels=next_filters,
                kernel_size=kernel_size,
                stride=stride,
                drop_rate=drop_rate
            )

            self.blocks.append(block)

            current_filters = next_filters

        # ----------------------------------------------------
        # Bloque de salida
        # ----------------------------------------------------

        self.output_bn = nn.BatchNorm1d(current_filters)
        self.output_relu = nn.ReLU(inplace=True)

        # Global Average Pooling.
        #
        # Esto reemplaza al Flatten() del modelo original.
        # Es mucho más eficiente para señales largas de 5000 muestras
        # y evita crear una capa Dense gigantesca.
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Linear(
            current_filters,
            num_classes
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x:
                Tensor con forma (batch, 12, 5000)

        Returns:
            Tensor con forma (batch, 27)
        """

        # Primer bloque
        x = self.first_conv(x)
        x = self.first_bn(x)
        x = self.first_relu(x)

        # Bloques residuales
        for block in self.blocks:
            x = block(x)

        # Salida
        x = self.output_bn(x)
        x = self.output_relu(x)

        # (B, C, L) -> (B, C, 1)
        x = self.global_pool(x)

        # (B, C, 1) -> (B, C)
        x = torch.flatten(x, 1)

        # (B, C) -> (B, 27)
        #
        # IMPORTANTE:
        # No aplicamos sigmoid aquí.
        #
        # BCEWithLogitsLoss() se encargará de hacerlo
        # numéricamente de forma estable durante el entrenamiento.
        x = self.classifier(x)

        return x


class ResidualBlock(nn.Module):
    """
    Bloque residual 1D.

    Si cambia el número de canales o se realiza downsampling,
    se adapta automáticamente la conexión shortcut mediante
    una convolución 1x1.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=7,
        stride=1,
        drop_rate=0.2
    ):
        super().__init__()

        padding = kernel_size // 2

        # ----------------------------------------------------
        # Shortcut
        # ----------------------------------------------------

        if in_channels != out_channels or stride != 1:

            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm1d(out_channels)
            )

        else:
            self.shortcut = nn.Identity()

        # ----------------------------------------------------
        # Primera convolución
        # ----------------------------------------------------

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )

        # ----------------------------------------------------
        # Segunda convolución
        # ----------------------------------------------------

        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.dropout = nn.Dropout(drop_rate)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False
        )

    def forward(self, x):

        shortcut = self.shortcut(x)

        # Pre-activation
        out = self.bn1(x)
        out = self.relu1(out)

        # Primera convolución
        out = self.conv1(out)

        # Segunda convolución
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout(out)
        out = self.conv2(out)

        # Conexión residual
        out = out + shortcut

        return out


def ECG_model(config):
    """
    Crea el modelo ECG para CinC 2020.

    Args:
        config:
            Objeto de configuración generado por config.py.

    Returns:
        ECGResNet
    """

    model = ECGResNet(
        num_leads=config.num_leads,
        num_classes=config.num_classes,
        filter_length=config.filter_length,
        kernel_size=config.kernel_size,
        drop_rate=config.drop_rate
    )

    return model


def count_parameters(model):
    """
    Cuenta los parámetros entrenables del modelo.
    """

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def print_model_info(model):
    """
    Muestra información básica del modelo.
    """

    total_params = count_parameters(model)

    print("\n" + "=" * 60)
    print("MODELO ECG - PhysioNet/CinC 2020")
    print("=" * 60)

    print(f"Parámetros entrenables: {total_params:,}")

    print(f"Dispositivo: {next(model.parameters()).device}")

    print("=" * 60 + "\n")