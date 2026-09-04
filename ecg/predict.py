"""PyTorch prediction (port of awni/ecg predict.py + entry/evaler.py).

Usage:
    python ecg/predict.py <dataset>.json <model>.pt

where <data>.json is built by examples/cinc17/build_datasets.py and <model>.pt
is a checkpoint produced by train.py. Prints predicted probabilities and,
like the original entry/evaler.py, the majority (mode) class per record.
"""

from __future__ import print_function, absolute_import

import argparse
import numpy as np
import os
import sys
import torch

try:
    from . import load, util, network
except ImportError:            # direct run as `python ecg/predict.py`
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from ecg import load, util, network


def predict(data_json, model_path, device=None):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    preproc = ckpt.get('preproc')
    if preproc is None:
        # fall back to preproc.bin saved next to the model
        preproc = util.load(os.path.dirname(model_path))

    config = ckpt.get('config', {})
    classes = ckpt.get('classes', preproc.classes)

    ecgs, _ = load.load_dataset(data_json)

    model = network.build_network(num_categories=len(classes), **config)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()

    # Predict RECORD BY RECORD (no cross-record padding). Padded zero regions
    # are learned as the '~' (noise) class during training, so if we padded all
    # records to the longest one, short records would be flooded with trailing
    # '~' votes and the per-record majority (mode) would be biased to '~'.
    # Evaluating each record on its OWN actual signal gives honest set-level
    # predictions (this mirrors how the paper treats each record independently).
    probs = []
    t_out = []
    for ecg in ecgs:
        x = preproc.process_x([ecg])                 # (1, T, 1) normalized
        xb = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 2, 1)))
        xb = xb.to(device)
        with torch.no_grad():
            p = model(xb).cpu().numpy()              # (1, T/256, C)
        probs.append(p[0])
        t_out.append(ecg.shape[0])
    return probs, preproc, t_out


def record_predictions(probs, preproc, _t_out=None):
    """Per-record class prediction using the mode over ACTUAL output intervals.

    `probs` is a list per record of shape (T_i/256, C). Because records are
    processed individually, every interval corresponds to real signal.
    """
    preds, scores = [], []
    for p in probs:
        row = np.argmax(p, axis=-1)
        vals, counts = np.unique(row, return_counts=True)
        idx = vals[np.argmax(counts)]
        preds.append(preproc.int_to_class[int(idx)])
        scores.append(counts.max() / counts.sum())
    return preds, scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_json", help="path to dataset json")
    parser.add_argument("model_path", help="path to model checkpoint (.pt)")
    args = parser.parse_args()
    probs, preproc, t_out = predict(args.data_json, args.model_path)
    preds, scores = record_predictions(probs, preproc)
    print("Predicted per-record classes ({} records):".format(len(preds)))
    for rec, p in zip(preds, scores):
        print("  -> {}  (confidence {:.3f})".format(rec, p))
