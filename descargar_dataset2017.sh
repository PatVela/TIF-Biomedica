#!/bin/bash
# Paso 1 de la Fase 1: Descarga de la base PhysioNet/CinC Challenge 2017.
# Creado para replicación exacta del paper de Hannun et al. (Single-Lead ECG).

set -e

DEST="dataset2017"
mkdir -p "$DEST"
cd "$DEST"

echo "======================================================"
echo " Descargando PhysioNet/CinC Challenge 2017 (v1.0.0)"
echo "======================================================"
echo "Directorio de destino: $DEST"
echo "Descargando datos de entrenamiento y archivos de referencia..."
echo ""

# Opción A: Descarga rápida mediante archivo ZIP oficial
wget -c https://physionet.org/files/challenge-2017/1.0.0/training2017.zip
wget -c https://physionet.org/files/challenge-2017/1.0.0/REFERENCE-v3.csv

echo ""
echo "======================================================"
echo " Descomprimiendo archivos..."
echo "======================================================"
echo ""

unzip -q training2017.zip
rm training2017.zip

echo "======================================================"
echo " Descarga y extracción completas"
echo "======================================================"
echo ""
echo "Estructura descargada:"
ls -lh