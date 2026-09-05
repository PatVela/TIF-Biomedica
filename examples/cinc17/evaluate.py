"""Formal evaluation of a trained model against the labeled dev/test set.

Reports (using the real ground-truth labels, without the padding hack):

  * Record-level  (set-level): one prediction per record (the mode over its
                   output intervals) vs the record's true label. Closest to how
                   the paper's "set-level" metric works.
  * Interval-level (sequence-level): each 256-sample output interval vs the true
                   label of its record. The paper's stricter "sequence-level".

Metrics printed per class and overall: accuracy, precision, recall, F1, macro-F1,
plus the confusion matrix. numpy-only (no scikit-learn required).

Usage:
    python examples/cinc17/evaluate.py --data_json examples/cinc17/dev.json \
        --model_path "saved/cinc17/<ts>/0.408-...pt" [--level both]

    # or auto-select the best (lowest val_loss) checkpoint:
    python examples/cinc17/evaluate.py --data_json examples/cinc17/dev.json \
        --saved saved [--level both]
"""

from __future__ import absolute_import

import argparse
import collections
import json
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
from ecg import predict as pred_mod, util


# --------------------------------------------------------------------------
# numpy metrics helpers
# --------------------------------------------------------------------------
def confusion_matrix(y_true, y_pred, labels):
    """Return a dict {label: {true_label: count}} for a fixed label set."""
    cm = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        if t in cm and p in cm[t]:
            cm[t][p] += 1
    return cm


def per_class_metrics(cm, labels):
    """precision / recall / f1 per class (numpy-based)."""
    out = {}
    for c in labels:
        tp = cm[c][c]
        fp = sum(cm[o][c] for o in labels if o != c)   # predicted c, true != c
        fn = sum(cm[c][o] for o in labels if o != c)   # true c, predicted != c
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out[c] = {'tp': tp, 'fp': fp, 'fn': fn,
                  'precision': precision, 'recall': recall, 'f1': f1}
    return out


def macro_f1(per_class):
    return float(np.mean([v['f1'] for v in per_class.values()]))


def weighted_f1(per_class, y_true):
    counts = collections.Counter(y_true)
    total = len(y_true) or 1
    return float(sum(per_class[c]['f1'] * counts[c] for c in per_class) / total)


def challenge_f1(per_class):
    """OFFICIAL PhysioNet CinC2017 score.

    The challenge defines the score as the average of the class-wise F1 over
    NORMAL (N), AF (A) and OTHER (O) ONLY — the '~' (noise) class is EXCLUDED
    from the average. This is the metric the paper reports (F1 = 0.83 on the
    hidden test set) and the only one directly comparable to it.
    """
    targets = {c for c in ('N', 'A', 'O') if c in per_class}
    if not targets:
        return 0.0
    return float(np.mean([per_class[c]['f1'] for c in targets]))


def accuracy(y_true, y_pred):
    return (np.asarray(y_true) == np.asarray(y_pred)).mean()


def pretty_matrix(cm, labels):
    """Render a text confusion matrix aligned to the label width."""
    w = max([len(str(l)) for l in labels] + [6])
    header = " " * (w + 2) + "".join("{:>{w}}".format(l, w=w) for l in labels)
    lines = [header]
    for t in labels:
        row = "{:>{w}} |".format(t, w=w) + "".join(
            "{:>{w}}".format(cm[t][p], w=w) for p in labels)
        lines.append(row)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def report(y_true, y_pred, labels, name):
    cm = confusion_matrix(y_true, y_pred, labels)
    per_class = per_class_metrics(cm, labels)
    acc = accuracy(y_true, y_pred)

    print("\n" + "=" * 60)
    print("  EVALUACION: {}".format(name))
    print("=" * 60)
    print("Exactitud (accuracy): {:.4f}".format(acc))
    print("Macro-F1 (todas)     : {:.4f}".format(macro_f1(per_class)))
    print("Weighted-F1          : {:.4f}".format(weighted_f1(per_class, y_true)))
    print("Challenge-F1 (N/A/O) : {:.4f}   <- métrica OFICIAL PhysioNet/paper".format(
        challenge_f1(per_class)))

    print("\n  Matriz de confusión (fila = real, columna = predicho):")
    print(pretty_matrix(cm, labels))

    print("\n  Reporte por clase:")
    print("    {:<6} {:>9} {:>9} {:>9} {:>9} {:>6}".format(
        "clase", "precision", "recall", "f1", "n_true", "n_pred"))
    counts_true = collections.Counter(y_true)
    counts_pred = collections.Counter(y_pred)
    for c in labels:
        pc = per_class[c]
        print("    {:<6} {:>9.3f} {:>9.3f} {:>9.3f} {:>9} {:>6}".format(
            c, pc['precision'], pc['recall'], pc['f1'],
            counts_true.get(c, 0), counts_pred.get(c, 0)))
    return acc, macro_f1(per_class)


def save_metrics_figures(y_true, y_pred, labels, out_dir, title):
    """Export the confusion matrix and per-class F1 as PNG figures.

    These images can be shown in the webapp 'Detalle técnico' tab (the app
    looks for static/metrics/*.png). Requires matplotlib.
    """
    import os
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as _np

    cm = confusion_matrix(y_true, y_pred, labels)
    # confusion_matrix returns a dict {true: {pred: count}}; build a numeric
    # 2-D array (rows = true, cols = predicted) so imshow accepts it.
    cm_arr = _np.array([[cm[t][p] for p in labels] for t in labels],
                       dtype=float)
    per_class = per_class_metrics(cm, labels)
    os.makedirs(out_dir, exist_ok=True)

    # --- confusion matrix heatmap ---
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(cm_arr, cmap='Blues', interpolation='nearest')
    ax.set_xticks(_np.arange(len(labels))); ax.set_xticklabels(labels)
    ax.set_yticks(_np.arange(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel('Predicho'); ax.set_ylabel('Real')
    ax.set_title('Matriz de confusión · %s' % title, fontsize=11)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(cm_arr[i, j]), ha='center', va='center',
                    color='white' if cm_arr[i, j] > cm_arr.max() / 2 else '#0d2b52')
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'confusion_matrix.png'), dpi=130)
    plt.close(fig)

    # --- per-class F1 bar chart ---
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    names = [c for c in labels]
    f1s = [per_class[c]['f1'] for c in names]
    bars = ax.bar(names, f1s, color='#123b6b', alpha=0.85)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('F1-score'); ax.set_title('F1 por clase · %s' % title, fontsize=11)
    ax.bar_label(bars, fmt='%.2f', fontsize=9)
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'f1_per_class.png'), dpi=130)
    plt.close(fig)
    print("Figuras de métricas guardadas en:", out_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_json", help="path to a train/dev/test .json")
    parser.add_argument("--model_path", help="path to a .pt checkpoint")
    parser.add_argument("--saved", default=None,
                        help="directory of checkpoints (auto-picks best val_loss)")
    parser.add_argument("--level", default="both",
                        choices=["record", "interval", "both"])
    parser.add_argument("--save_metrics_dir", default=None,
                        help="optional: export confusion matrix + F1 figures "
                             "as PNGs to this folder (for the webapp 'Detalle "
                             "técnico' tab), e.g. webapp/static/metrics")
    args = parser.parse_args()

    # resolve best model if not given
    model_path = args.model_path
    if not model_path:
        best = util.find_best_model(args.saved)
        if not best:
            print("No checkpoint found; pass --model_path or --saved.")
            sys.exit(1)
        model_path = best
        print("Usando mejor checkpoint (menor val_loss):", model_path)

    # ground-truth labels: each record one label (its class)
    with open(args.data_json) as f:
        data = [json.loads(l) for l in f]
    true_records = [d['labels'][0] for d in data]

    probs, preproc, _t_out = pred_mod.predict(args.data_json, model_path)
    labels = list(preproc.classes)

    y_pred_records = []
    y_true_intervals = []
    y_pred_intervals = []
    for i, p in enumerate(probs):
        row = np.argmax(p, axis=-1)                  # per-interval preds
        # per-interval true label == record label (each record is one rhythm)
        n_int = row.shape[0]
        y_true_intervals.extend([true_records[i]] * n_int)
        y_pred_intervals.extend(preproc.int_to_class[c] for c in row)
        # record-level = mode
        vals, counts = np.unique(row, return_counts=True)
        y_pred_records.append(preproc.int_to_class[int(vals[np.argmax(counts)])])

    if args.save_metrics_dir:
        save_metrics_figures(true_records, y_pred_records, labels,
                             args.save_metrics_dir, "Nivel registro")
    if args.level in ("record", "both"):
        report(true_records, y_pred_records, labels, "NIVEL REGISTRO")
    if args.level in ("interval", "both"):
        # NOTE: CinC2017 has no per-interval ground truth. The paper reports
        # ONLY the record-level metric (majority vote) for this dataset; the
        # "interval" numbers below are an INTERNAL diagnostic (comparing each
        # 256-sample output against the record's replicated label) and should
        # NOT be presented as a replica of the paper's figure.
        report(y_true_intervals, y_pred_intervals, labels,
               "NIVEL INTERVALO (solo diagnóstico interno, no es la métrica del paper)")


if __name__ == '__main__':
    main()
