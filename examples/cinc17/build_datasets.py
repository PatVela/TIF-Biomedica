"""Build train.json / dev.json for the PhysioNet CinC2017 training set.

Port of awni/ecg examples/cinc17/build_datasets.py, but it reads the raw WFDB
recording files directly (supporting both the .mat files OR the format-212 .dat
files) and the class label csv.

The CinC2017 training set (training2017.zip) contains short recordings each
labelled with ONE class in REFERENCE-v3.csv. The label is repeated for every
256-sample output step of the record (STEP = 256), exactly as the original code
does. Usage:

    python examples/cinc17/build_datasets.py --data_dir data/training2017 \
        --label_file data/REFERENCE-v3.csv --out_dir examples/cinc17
"""

from __future__ import absolute_import

import argparse
import json
import os
import random
import sys

# Make the `ecg` package importable regardless of the CWD / how this script is
# invoked (it lives under examples/cinc17/, so the repo root is two levels up).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import tqdm
from ecg import load

STEP = 256


def load_all(data_path, label_file):
    with open(label_file, 'r') as f:
        records = [l.strip().split(",") for l in f if l.strip()]

    dataset = []
    for record, label in tqdm.tqdm(records):
        # prefer .mat (as in the original), then .dat (WFDB format-212)
        candidate = os.path.abspath(os.path.join(data_path, record + ".mat"))
        if not os.path.exists(candidate):
            candidate = os.path.abspath(os.path.join(data_path, record + ".dat"))
        ecg = load.load_ecg(candidate)
        num_labels = ecg.shape[0] // STEP
        dataset.append((candidate, [label] * num_labels))
    return dataset


def split(dataset, dev_frac):
    dev_cut = int(dev_frac * len(dataset))
    random.shuffle(dataset)
    dev = dataset[:dev_cut]
    train = dataset[dev_cut:]
    return train, dev


def make_json(save_path, dataset):
    with open(save_path, 'w') as f:
        for d in dataset:
            f.write(json.dumps({'ecg': d[0], 'labels': d[1]}) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default="data/training2017",
                        help="directory holding the CinC2017 training recordings")
    parser.add_argument("--label_file",
                        default="data/REFERENCE-v3.csv",
                        help="CSV of record,label")
    parser.add_argument("--out_dir",
                        default="examples/cinc17",
                        help="where to write train.json / dev.json")
    parser.add_argument("--dev_frac", type=float, default=0.1)
    args = parser.parse_args()

    random.seed(2018)
    dataset = load_all(args.data_dir, args.label_file)
    train, dev = split(dataset, args.dev_frac)
    make_json(os.path.join(args.out_dir, "train.json"), train)
    make_json(os.path.join(args.out_dir, "dev.json"), dev)
    print("train:", len(train), "dev:", len(dev))
