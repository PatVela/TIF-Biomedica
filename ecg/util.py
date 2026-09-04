"""PyTorch port of awni/ecg util.py (save/load of the preprocessor)."""

import os
import pickle


def save(preproc, dirname):
    preproc_f = os.path.join(dirname, "preproc.bin")
    with open(preproc_f, 'wb') as f:
        pickle.dump(preproc, f)


def load(dirname):
    preproc_f = os.path.join(dirname, "preproc.bin")
    with open(preproc_f, 'rb') as f:
        return pickle.load(f)
