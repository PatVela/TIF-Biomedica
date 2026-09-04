#!/bin/bash
# Paso 1 de la Fase 1: descarga de las 6 bases del PhysioNet/CinC Challenge 2020.
# Descarga únicamente training/ y NO descarga sources/.

set -e

DEST="dataset2020"
mkdir -p "$DEST"
cd "$DEST"

echo "======================================================"
echo " Descargando PhysioNet/CinC Challenge 2020"
echo "======================================================"
echo "Descargando únicamente: training/"
echo "NO se descargará: sources/"
echo ""

wget -r -N -c -np \
    https://physionet.org/files/challenge-2020/1.0.2/training/st_petersburg_incart/

echo ""
echo "======================================================"
echo " Descarga completa"
echo "======================================================"
echo ""
echo "Estructura descargada:"
find . -maxdepth 4 -type d