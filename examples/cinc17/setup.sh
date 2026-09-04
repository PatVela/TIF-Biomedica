#!/bin/bash
# Download the PhysioNet CinC2017 training set + labels and build train/dev json.
#
# NOTE: the training data is gated (it requires a PhysioNet account to sign the
# data-use agreement). The per-record download links are not listed here on
# purpose. The *official* way is to download the two zip files manually:
#
#   https://physionet.org/content/challenge-2017/1.0.0/
#
# and place them (or their extracted folders) under ./data, then run:
#     ./setup.sh   -- after putting training2017.zip + REFERENCE-v3.csv in ./data

set -e
mkdir -p data

BASE=https://physionet.org/files/challenge-2017/1.0.0

# The files below are gated; if you can access them they will download.
# If they 403, download them manually at the URL above.
[ -f data/training2017.zip ] || curl -L -o data/training2017.zip $BASE/training2017.zip || true
echo "training2017.zip ready (if empty, download manually from physionet.org)."
unzip -o -q data/training2017.zip -d data/training2017 || true

[ -f data/REFERENCE-v3.csv ] || curl -L -o data/REFERENCE-v3.csv $BASE/REFERENCE-v3.csv || true

echo "Building datasets..."
python build_datasets.py
