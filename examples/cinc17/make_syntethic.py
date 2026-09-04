"""Generate *synthetic* ECGs to smoke-test the full pipeline (train + predict).

The real CinC2017 data is gated behind a PhysioNet data-use agreement, so for a
quick end-to-end run (no download needed) this writes fake single-lead signals,
each labelled with one of the CinC2017 classes, into .mat files and then builds
train.json / dev.json that train.py / predict.py can consume directly.

The signal here is fake (an almost-flat sinusoid); it exists only so you can
verify the code runs. Training on synthetic noise will not learn anything real.
"""

import json
import os
import random
import numpy as np
import scipy.io as sio
import tqdm

CLASSES = ['A', 'N', 'O', '~', '|']     # CinC2017 rhythm classes
STEP = 256
FS = 300


def synth_ecg(n_samples, kind):
    """Return a fake single-lead signal of n_samples points."""
    t = np.arange(n_samples) / FS
    # a noisy base + a pseudo "beat" burst depending on class
    base = 0.05 * np.sin(2 * np.pi * 1.2 * t)
    noise = np.random.randn(n_samples) * 0.02
    if kind == 'A':
        base += 0.15 * np.sin(2 * np.pi * 4.5 * t).clip(min=0)
    elif kind == 'O':
        base += 0.15 * np.sin(2 * np.pi * 2.0 * t).clip(min=0)
    else:
        base += 0.15 * np.sin(2 * np.pi * 1.3 * t).clip(min=0)
    return (base + noise).astype(np.float32)


def main(out_dir, num_train, num_dev, seed=2018):
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    for i in range(num_train + num_dev):
        is_dev = i >= num_train
        rec = "syn{}".format(i)
        label = random.choice(CLASSES)
        length = random.choice([STEP * 10, STEP * 15, STEP * 20])
        ecg = synth_ecg(length, label)
        path = os.path.join(out_dir, rec + ".mat")
        sio.savemat(path, {'val': ecg[None, :]})   # shape (1, n) like a .mat ECG
        n_out = ecg.shape[0] // STEP
        rec_entry = {'ecg': os.path.abspath(path), 'labels': [label] * n_out}
        with open(os.path.join(out_dir, "dev.json" if is_dev else "train.json"),
                  'a') as f:
            f.write(json.dumps(rec_entry) + '\n')
    print("Wrote", num_train, "train and", num_dev, "dev synthetic records to", out_dir)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=int, default=120)
    p.add_argument("--dev", type=int, default=24)
    p.add_argument("--out", default="examples/cinc17/synthetic")
    args = p.parse_args()
    main(args.out, args.train, args.dev)
