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


def split(dataset, dev_frac, stratify=False, seed=2018):
    """Split into train/dev.

    By default the split is random (as in the original awni/ecg code). With
    `stratify=True` it instead partitions proportionally to each class so the
    rare classes (A, ~) are represented in dev at the same rate as in train —
    recommended to avoid a class-imbalanced dev set (see README / notes).
    """
    if not stratify:
        random.Random(seed).shuffle(dataset)
        dev_cut = int(dev_frac * len(dataset))
        return dataset[dev_cut:], dataset[:dev_cut]

    by_class = {}
    for d in dataset:
        label = d[1][0] if d[1] else 'N'
        by_class.setdefault(label, []).append(d)
    train, dev = [], []
    for label, items in by_class.items():
        random.Random(seed + hash(label) % 1_000_000).shuffle(items)
        cut = int(round(dev_frac * len(items)))
        train.extend(items[cut:])
        dev.extend(items[:cut])
    random.Random(seed).shuffle(train)
    random.Random(seed + 1).shuffle(dev)
    return train, dev


def make_json(save_path, dataset, relative_to=None):
    with open(save_path, 'w') as f:
        for d in dataset:
            path = d[0]
            if relative_to:
                path = os.path.relpath(path, relative_to)
            f.write(json.dumps({'ecg': path, 'labels': d[1]}) + '\n')


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
    parser.add_argument("--stratify", action="store_true",
                        help="split proportionally to class (avoids imbalanced dev)")
    parser.add_argument("--relative", action="store_true",
                        help="store record paths relative to --out_dir (portable "
                             "train/dev.json, no absolute Windows paths)")
    args = parser.parse_args()

    random.seed(2018)
    dataset = load_all(args.data_dir, args.label_file)
    train, dev = split(dataset, args.dev_frac, stratify=args.stratify)
    # paths are stored relative to the project ROOT so train.json/dev.json are
    # portable across machines (train.py resolves them against the repo root).
    rel = os.path.abspath(os.path.join(args.out_dir, '..', '..')) if args.relative else None
    make_json(os.path.join(args.out_dir, "train.json"), train, relative_to=rel)
    make_json(os.path.join(args.out_dir, "dev.json"), dev, relative_to=rel)
    print("train:", len(train), "dev:", len(dev))
