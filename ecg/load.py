"""PyTorch port of awni/ecg load.py.

Reads a JSONL dataset file (one JSON object per line, with keys `ecg` and
`labels`), truncates each signal to a multiple of STEP (256), normalises it,
and yields (batch, 1, time) tensors with per-*time-step* class targets.

IMPORTANT semantic note vs CinC2020:
The awni/ecg pipeline (and the Hannun paper) labels *every 256-sample output
step* with the label of its parent record. That is exactly the CinC2017 setup,
where each short recording has a single class. The chunk that is produced by the
network (time / 256) is the "heartbeat-aligned" classification output. See the
README for how this differs from the multi-label CinC2020 setting.
"""

from __future__ import print_function, absolute_import

import json
import numpy as np
import os
import random
import scipy.io as sio
import torch

STEP = 256


# ---------------------------------------------------------------------------
# Dataset / preprocessor
# ---------------------------------------------------------------------------
class Preproc:
    """Port of Preproc: computes global mean/std and the class vocabulary."""

    def __init__(self, ecg, labels):
        self.mean, self.std = compute_mean_std(ecg)
        self.classes = sorted(set(l for label in labels for l in label))
        self.int_to_class = dict(zip(range(len(self.classes)), self.classes))
        self.class_to_int = {c: i for i, c in self.int_to_class.items()}

    # -- numpy-level convenience (used by predict and the non-generator path) --
    def process(self, x, y):
        return self.process_x(x), self.process_y(y)

    def process_x(self, x, as_tensor=False):
        x = pad(x, dtype=np.float32)
        x = (x - self.mean) / self.std
        x = x[:, :, None]                      # (N, time, 1)
        if as_tensor:
            return torch.from_numpy(np.ascontiguousarray(x.transpose(0, 2, 1)))
        return x

    def process_y(self, y):
        ints = [[self.class_to_int[c] for c in s] for s in y]
        ints = pad(ints, val=3, dtype=np.int32)          # pad value '3'
        return ints.astype(np.int64)


def pad(x, val=0, dtype=np.float32):
    max_len = max(len(i) for i in x)
    padded = np.full((len(x), max_len), val, dtype=dtype)
    for e, i in enumerate(x):
        padded[e, :len(i)] = i
    return padded


def compute_mean_std(x):
    x = np.hstack(x)
    return (np.mean(x).astype(np.float32), np.std(x).astype(np.float32))


# ---------------------------------------------------------------------------
# Data generator (corresponds to data_generator + Preproc.process)
# ---------------------------------------------------------------------------
def data_generator(batch_size, preproc, x, y, shuffle=True, device='cpu'):
    """Yield (x_batch, y_targets) torch tensors.

    x_batch : (batch, 1, time)
    y_targets : (batch, time_out) long indices, where time_out = time / 256.
    """
    num_examples = len(x)
    examples = list(zip(x, y))
    examples = sorted(examples, key=lambda z: z[0].shape[0])
    end = num_examples - batch_size + 1
    if end <= 0:
        raise ValueError("dataset smaller than batch size")
    batches = [examples[i:i + batch_size] for i in range(0, end, batch_size)]
    if shuffle:
        random.shuffle(batches)
    while True:
        for batch in batches:
            xb, yb = zip(*batch)
            yield preproc.process_x(xb), preproc.process_y(yb)


# ---------------------------------------------------------------------------
# File IO
# ---------------------------------------------------------------------------
def load_ecg(record):
    if os.path.splitext(record)[1] == ".npy":
        ecg = np.load(record)
    elif os.path.splitext(record)[1] == ".mat":
        ecg = sio.loadmat(record)['val'].squeeze()
    elif os.path.splitext(record)[1] == ".dat":
        ecg = read_physionet_dat(record)
    else:  # Assumes binary 16 bit integers
        with open(record, 'rb') as f:
            ecg = np.fromfile(f, dtype=np.int16)
    ecg = ecg.astype(np.float32).squeeze()
    if ecg.ndim > 1:
        # multi-lead (CinC2020): pick the first lead (see README re: CinC2017).
        ecg = ecg[0]
    trunc_samp = STEP * int(len(ecg) // STEP)
    return ecg[:trunc_samp]


def read_physionet_dat(dat_file):
    """Read a PhysioNet binary 'format 212' .dat (12-bit, 3 bytes / 2 samples).

    This is the format produced by the WFDB toolkit for several PhysioNet
    databases (e.g. CinC2017). Loading is vectorised over the whole file.
    """
    raw = np.fromfile(dat_file, dtype=np.uint8)
    n_pairs = len(raw) // 3
    raw = raw[:n_pairs * 3].reshape(n_pairs, 3)
    b0, b1, b2 = raw[:, 0], raw[:, 1], raw[:, 2]

    sample0 = (b0.astype(np.int32) | (b1 & 0x0F).astype(np.int32) << 8)
    sample1 = ((b1 >> 4).astype(np.int32) | b2.astype(np.int32) << 4)
    # sign-extend the 12-bit two's complement values
    sample0 = np.where(sample0 & 0x800, sample0 - 0x1000, sample0).astype(np.int16)
    sample1 = np.where(sample1 & 0x800, sample1 - 0x1000, sample1).astype(np.int16)

    data = np.empty(n_pairs * 2, dtype=np.int16)
    data[0::2] = sample0
    data[1::2] = sample1
    return data


def load_dataset(data_json):
    with open(data_json, 'r') as f:
        data = [json.loads(l) for l in f]
    labels, ecgs = [], []
    for d in data:
        labels.append(d['labels'])
        ecgs.append(load_ecg(d['ecg']))
    return ecgs, labels


if __name__ == "__main__":
    import tqdm
    data_json = os.path.join(
        os.path.dirname(__file__), '..', 'examples', 'cinc17', 'train.json')
    train = load_dataset(data_json)
    preproc = Preproc(*train)
    gen = data_generator(32, preproc, *train)
    x, y = next(gen)
    print('x', tuple(x.shape), 'y', tuple(y.shape))
