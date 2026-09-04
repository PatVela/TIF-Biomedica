# -*- coding: utf-8 -*-
"""
Diagnóstico rápido: identifica qué registros/derivaciones de
incart_external.h5 quedaron con varianza casi nula (posible
derivación desconectada o plana durante gran parte de los 30 min
de grabación Holter). No modifica nada, solo informa.

Uso:
    python diagnostico_incart.py --file dataset2020_procesado/incart_external.h5
"""

import argparse
import h5py
import numpy as np


LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
              'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str,
                         default='dataset2020_procesado/incart_external.h5')
    parser.add_argument('--umbral_std', type=float, default=0.05,
                         help='Desvío por debajo del cual se considera '
                              'una derivación sospechosamente plana '
                              '(la señal ya viene normalizada a std~1, '
                              'así que 0.05 es un valor bajo real).')
    args = parser.parse_args()

    with h5py.File(args.file, 'r') as f:
        signals = f['signals'][:]       # (N, 5000, 12)
        record_names = [n.decode() if isinstance(n, bytes) else n
                         for n in f['record_names'][:]]

    print(f"Revisando {signals.shape[0]} registros de {args.file}\n")

    n_afectados = 0
    for i in range(signals.shape[0]):
        stds = signals[i].std(axis=0)  # std por derivación, forma (12,)
        leads_planas = [LEAD_NAMES[j] for j, s in enumerate(stds)
                         if s < args.umbral_std]

        if leads_planas:
            n_afectados += 1
            print(f"{record_names[i]}: derivaciones sospechosas "
                  f"{leads_planas} (std: "
                  f"{[round(stds[LEAD_NAMES.index(l)], 4) for l in leads_planas]})")

    print(f"\nTotal: {n_afectados}/{signals.shape[0]} registros con "
          f"al menos una derivación por debajo del umbral ({args.umbral_std}).")


if __name__ == '__main__':
    main()