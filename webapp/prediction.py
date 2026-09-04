"""Inference + plotting service for the ECG web app.

Wraps the trained PyTorch model (checkpoint produced by ecg/train.py) and the
saved `Preproc` (mean/std + class vocabulary) so the web app only needs to:

    service = PredictionService(model_path)
    result  = service.predict_signal(fs, signal)      # numpy floats
    b64     = service.render_plot(fs, signal, result)  # PNG data-URI
"""

from __future__ import absolute_import

import base64
import io
import math
import os
import sys
import glob

import numpy as np

# Make the `ecg` package importable regardless of CWD (webapp/ lives in repo root)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
from ecg import network

# default sampling rate for CinC2017 (Hz) if not provided in the CSV
DEFAULT_FS = 300
STEP = 256


class PredictionService:
    """Loads a trained model + preprocessor and runs single-lead inference."""

    def __init__(self, model_path, device=None):
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' not in ckpt:
            raise ValueError(
                "El archivo no es un checkpoint de train.py "
                "(falta 'model_state_dict').")
        self.preproc = ckpt.get('preproc')
        self.classes = ckpt.get('classes') or (self.preproc.classes if self.preproc else None)
        config = ckpt.get('config', {}) or {}
        num_cat = len(self.classes) if self.classes else int(config.get('num_categories', 5))

        self.model = network.build_network(num_categories=num_cat, **config)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.to(self.device).eval()
        # convenient ordered class labels + a clinical-ish description map
        self.class_to_desc = {
            'N': 'Normal',
            'A': 'Atrial fibrillation / flutter (AF)',
            'O': 'Other rhythm',
            '~': 'Noise / artifact',
            '|': 'Silence / unclassifiable',
        }

    # ------------------------------------------------------------------ input
    def normalize(self, signal):
        """Apply the saved global mean/std and return a (1, T) tensor."""
        # The preserved Preproc stores mean/std; fall back to recomputing.
        if self.preproc is not None:
            mean = float(self.preproc.mean)
            std = float(self.preproc.std)
        else:
            mean = float(signal.mean())
            std = float(signal.std()) or 1.0
        x = (np.asarray(signal, dtype=np.float32) - mean) / std
        # truncate to a multiple of STEP (matches load_ecg in the training port)
        x = x[:STEP * (len(x) // STEP)]
        return torch.from_numpy(np.ascontiguousarray(x[None, None, :])).to(self.device)

    # -------------------------------------------------------------- inference
    def predict_signal(self, signal):
        """Return dict with probs per output interval + per-record summary."""
        x = self.normalize(signal)
        with torch.no_grad():
            probs = self.model(x).cpu().numpy()        # (1, T/256, num_classes)
        probs = probs[0]                                # (T/256, num_classes)
        pred_idx = probs.argmax(axis=-1)
        labels = [self.classes[i] for i in pred_idx]

        # class counts / dominant class over the record (set-level prediction)
        counts = {c: int((pred_idx == self.classes.index(c)).sum()) for c in self.classes
                  if c in self.classes}
        total = len(labels) or 1
        summary = sorted(
            ({'label': c, 'desc': self.class_to_desc.get(c, c), 'count': v,
              'pct': round(100.0 * v / total, 1)} for c, v in counts.items() if v),
            key=lambda d: d['count'], reverse=True)

        per_interval = [{'idx': i, 'label': labels[i],
                         'desc': self.class_to_desc.get(labels[i], labels[i]),
                         'prob': round(float(probs[i, pred_idx[i]]), 3)}
                        for i in range(len(labels))]
        return {'probs': probs, 'labels': labels, 'per_interval': per_interval,
                'summary': summary, 'dominant': summary[0]['label'] if summary else None,
                'n_intervals': len(labels)}

    # ------------------------------------------------------------------ plot
    def render_plot(self, fs, signal, result):
        """Render ECG trace + color-coded prediction bands as a base64 PNG."""
        import matplotlib
        matplotlib.use('Agg')                      # headless backend
        import matplotlib.pyplot as plt

        signal = np.asarray(signal, dtype=np.float32)
        n = len(signal)
        t_sec = np.arange(n) / fs
        n_int = result['n_intervals']

        # colour per class
        palette = {'N': '#2ca02c', 'A': '#d62728', 'O': '#ff7f0e',
                   '~': '#7f7f7f', '|': '#1f77b4'}
        default = '#9467bd'

        fig, (ax, ax2) = plt.subplots(
            2, 1, figsize=(13, 6.5), gridspec_kw={'height_ratios': [3, 1]},
            sharex=True)
        fig.subplots_adjust(hspace=0.15)

        # --- trace ---
        ax.plot(t_sec, signal, color='#1a1a2e', linewidth=0.8)
        ax.set_ylabel('Amplitud (mV)')
        ax.set_title('ECG — predicción por intervalo de {} muestras (~{:.2f} s)'.format(
            STEP, STEP / fs))
        ax.grid(alpha=0.25)

        # color bands per output interval on a secondary strip => use a twin axis
        # draw each interval as a shaded region on the *signal* axes but below care.
        # Simpler: shade on the top axis using axvspan with low alpha.
        step_samples = STEP
        for i in range(n_int):
            c = palette.get(result['labels'][i], default)
            start = i * step_samples / fs
            end = min((i + 1) * step_samples / fs, t_sec[-1])
            ax.axvspan(start, end, color=c, alpha=0.10, linewidth=0)
        # legend of detected classes
        handles = [plt.Rectangle((0, 0), 1, 1, color=palette.get(c, default))
                   for c in dict.fromkeys(result['labels'])]
        ax.legend(handles, list(dict.fromkeys(result['labels'])),
                  loc='upper right', fontsize=9, framealpha=0.9)

        # --- dominant class bar (probability) ---
        s = result['summary']
        bar_labels = [d['label'] + ' (' + d['desc'][:16] + ')' for d in s]
        bar_vals = [d['pct'] for d in s]
        bar_cols = [palette.get(d['label'], default) for d in s]
        ax2.bar(range(len(s)), bar_vals, color=bar_cols, alpha=0.85)
        ax2.set_xticks(range(len(s)))
        ax2.set_xticklabels(bar_labels, rotation=20, ha='right', fontsize=8)
        ax2.set_ylabel('% del registro')
        ax2.set_ylim(0, 100)
        ax2.set_title('Distribución de clases en todo el registro', fontsize=10)
        ax2.grid(axis='y', alpha=0.25)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode('ascii')


# --------------------------------------------------------------------------
# CSV parsing helpers (used by app.py)
# --------------------------------------------------------------------------
def parse_ecg_csv(text, default_fs=DEFAULT_FS):
    """Parse an uploaded CSV of a single-lead ECG.

    Accepted layouts:
      * Reference format: first numeric row is `fs, s0, s1, s2, ...`
        (an optional header line naming the lead may precede it).
      * Column format: one signal sample per row; a header line whose FIRST
        cell holds the sampling rate (e.g. `300,Lead`), else default_fs is used.

    Returns (fs, signal_1d).
    """
    lines = [ln for ln in text.splitlines()
             if ln.strip() and not ln.strip().startswith(('#', '%'))]

    # split into tokens, keeping per-row raw for header detection
    rows = []
    for ln in lines:
        toks = [t for t in ln.replace(',', ' ').split() if t]
        rows.append(toks)

    fs = None
    signal = []
    data_rows = []
    header_info = None

    for toks in rows:
        floats = []
        ok = True
        for t in toks:
            try:
                floats.append(float(t))
            except ValueError:
                ok = False
                break
        if not ok:
            # header line: try to read fs from the first cell
            try:
                candidate = float(toks[0])
                if fs is None and candidate > 0:
                    fs = candidate
            except (ValueError, IndexError):
                pass
            continue
        data_rows.append(floats)

    if not data_rows:
        raise ValueError("No se encontraron datos numéricos en el CSV.")

    # determine layout
    if len(data_rows) == 1 or len(data_rows[0]) > 1:
        # row layout: first value of first data row = fs, rest = signal
        first = data_rows[0]
        if len(first) > 1 and fs is None:
            if first[0] > 0:
                fs = first[0]
            signal = first[1:]
            for r in data_rows[1:]:
                signal.extend(r)
        else:
            # single column of numbers in a one-row file => treat as signal
            signal = first
            for r in data_rows[1:]:
                signal.extend(r)
    else:
        # column layout (one sample per row)
        signal = [r[0] for r in data_rows]

    signal = np.asarray(signal, dtype=np.float32)
    if fs is None:
        fs = float(default_fs)
    if len(signal) < STEP:
        raise ValueError("La señal es demasiado corta (< {} muestras).".format(STEP))
    return float(fs), signal


def find_best_model(models_dir):
    """Return the checkpoint with the lowest val_loss (first number in name)."""
    if not models_dir or not os.path.isdir(models_dir):
        return None
    ptfiles = []
    for root, _, files in os.walk(models_dir):
        ptfiles += [os.path.join(root, f) for f in files if f.endswith('.pt')]
    if not ptfiles:
        return None

    def key(p):
        base = os.path.basename(p)
        try:
            return float(base.split('-')[0])
        except (ValueError, IndexError):
            return float('inf')
    ptfiles.sort(key=key)
    return ptfiles[0]
