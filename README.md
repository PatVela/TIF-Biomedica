# ecg-pytorch — Cardiologist-level arrhythmia detection (PyTorch)

A faithful **PyTorch re-implementation** of
[`awni/ecg`](https://github.com/awni/ecg), the open-source code accompanying the
paper

> **Cardiologist-Level Arrhythmia Detection and Classification in Ambulatory
> Electrocardiograms Using a Deep Neural Network**
> Hannun, A. Y., Rajpurkar, P., Haghpanahi, M., Tison, G. H., Bourn, C.,
> Turakhia, M. P., & Ng, A. Y. — *Nature Medicine*, 2019.

The original repo is a Keras/TensorFlow (Python 2) implementation. This project
ports the **same architecture and the same data pipeline** 1:1 to modern PyTorch.

> ## Dataset decision (important)
> The instructor asked for an *exact replica* of the paper.
> - The paper / `awni/ecg` is built and validated on the **PhysioNet CinC 2017**
>   challenge set: **single-lead** ECGs, **one label per recording**, reproduced
>   at every 256-sample output step. This is the **exact** setting the README of
>   the repo demonstrates (`examples/cinc17/`), so it is the project's **primary
>   dataset**.
> - The original iRhythm ambulatory dataset is **not public**. Using **CinC
>   2020** (12-lead, **multi-label**) would change both the data format *and* the
>   loss/architecture (multi-label BCE, 12 input channels), i.e. it would not be
>   an exact replica. CinC 2020 support is discussed in [the section below](#cinc-2020).

---

## Architecture (matches the paper exactly)

```
input (B, 1, T)
  │
  ├─ Conv1d(k=16,  stride=1, SAME) ─ BatchNorm ─ ReLU    32 ch
  ├─ 16 × residual blocks, subsample lengths [1,2,1,2,...,1,2]
  │     each block: MaxPool(subsample) shortcut (with zero-pad on channels
  │     when the channel count doubles, every 4 blocks)
  │     2 × [BN→ReLU→Conv1d(k=16, stride=subsample then 1)]
  │     channels double every 4 blocks: 32 → 64 → 128 → 256
  ├─ BatchNorm ─ ReLU
  └─ Linear(num_categories) ─ softmax   ⇒  output (B, T/256, num_categories)
```

Every time step of the (downsampled) output is a softmax over the rhythm
classes, i.e. one classification per 256-sample (≈0.85 s at 300 Hz) chunk —
exactly as in the paper. Default settings reproduce `examples/cinc17/config.json`
(filter length 16, 32 start filters, 2 convs/block, channels ×2 every 4 blocks,
dropout 0.2, Adam lr 1e-3, ~10.5 M params).

### Mapping from the Keras code
| Keras / TF (`awni/ecg`)                        | PyTorch (this repo)                          |
|-----------------------------------------------|----------------------------------------------|
| `Conv1D(padding='same')`                      | `nn.Conv1d` + `SAME`-padding (asymmetric)    |
| `MaxPooling1D(pool_size=s)`                   | `nn.MaxPool1d(s)` + `SAME`-padding           |
| `BatchNormalization` + `Activation('relu')`   | `nn.BatchNorm1d` + `nn.ReLU`                 |
| `Dropout`                                     | `nn.Dropout`                                 |
| `TimeDistributed(Dense)` + `softmax`          | `nn.Linear` on `(B,T,C)` + `nn.Softmax`      |
| `categorical_crossentropy`                    | `nn.CrossEntropyLoss` on flattened logits    |
| `Adam(clipnorm=1)`                            | `torch.optim.Adam` + `clip_grad_norm_`       |
| `ReduceLROnPlateau` / `EarlyStopping`         | manual LR-on-plateau + early-stop            |
| `ModelCheckpoint .hdf5`                       | `torch.save` `.pt` every epoch               |

> **Shape convention.** Keras is `(batch, time, channels)`, PyTorch is
> `(batch, channels, time)`. The network handles this internally, so callers
> always pass/predict with `(B, 1, T)`.
>
> **`SAME` padding** is implemented asymmetrically so the output length matches
> Keras/TF exactly (`out = ceil(n / stride)`). This is essential to reproduce
> the paper's time-downsampling (÷256).

---

## Install

```bash
pip install -r requirements.txt       # torch, numpy, scipy, tqdm
```

## Get the data (PhysioNet CinC 2017)

The CinC 2017 training set is gated behind a PhysioNet account. Download
`training2017.zip` (with the labels `REFERENCE-v3.csv`) from
https://physionet.org/content/challenge-2017/1.0.0/ and place them under
`examples/cinc17/data/`, then build the datasets:

```bash
cd examples/cinc17
bash setup.sh            # requires the zip/csv already placed in data/
# or manually:
python build_datasets.py --data_dir data/training2017 --label_file data/REFERENCE-v3.csv
```

This writes `examples/cinc17/train.json` / `dev.json` (JSONL, one record per
line, with `ecg` path and repeated `labels`).

## Train

``bash
# from the repo root (paths in config.json are relative to the repo root)
python -m ecg.train examples/cinc17/config.json -e cinc17
```

Checkpoints are written every epoch to
`saved/<experiment>/<timestamp>/<val_loss>-<val_acc>-<epoch>-...pt`, together
with the preprocessor (`preproc.bin`). The best model = smallest `val_loss` in
the filename.

## Predict

```bash
python -m ecg.predict examples/cinc17/dev.json  saved/cinc17/<ts>/6.210-0.354-003-....pt
```

Prints the per-record majority class (mode over time steps), replicating
`entry/evaler.py`.

## Quick smoke test (no download)

To verify the code runs end-to-end on fake ECGs:

```bash
python examples/cinc17/make_synthetic.py --train 80 --dev 40
python -m ecg.train examples/cinc17/config_synthetic.json -e synth --epochs 2
# then predict with the saved checkpoint
```

(The dummy signal cannot learn anything real; it only checks the pipeline.)

---

## CinC 2020

CinC 2020 is included as an *optional* path for future work, but it is **not**
an exact replica of the paper, because:

| Aspect            | CinC 2017 (this project)              | CinC 2020 (would differ)                    |
|-------------------|---------------------------------------|---------------------------------------------|
| Leads             | 1                                     | 12                                          |
| Label             | 1 class per recording (per-chunk)     | multiple SNOMED-CT diagnoses per recording  |
| Loss / head       | categorical softmax (per step)        | multi-label BCE (12- or 1-channel head)     |
| Data format       | short `.mat`/`212` single-lead        | WFDB `.dat`+`.hea`, 12 leads, ~10 s         |

The loader already tolerates multi-lead data (`load_ecg` reads `.mat`, `.dat`,
`.npy` and WFDB and selects the first lead), so a CinC 2020 pipeline can reuse
the same CNN by: (1) using WFDB (`wfdb.rdrecord`) to load all 12 leads,
(2) making the first convolution take 12 input channels, and (3) switching the
loss to binary cross-entropy over diagnoses. See `ecg/network.py` (`is_regular_conv`
and the input-channel parameter) and `ecg/load.py` for the extension points.

---

## Repository layout

```
ecg_pytorch/
├── ecg/
│   ├── network.py     # the CNN (ECGNetwork), ported 1:1
│   ├── load.py        # Preproc + data generator + WFDB/mat/dat readers
│   ├── train.py       # training loop (Adam + clipnorm, LR plateau, early stop)
│   ├── predict.py     # inference + per-record majority class
│   ├── util.py        # preprocessor save/load
│   └── __init__.py
├── examples/cinc17/
│   ├── config.json        # paper's default hyper-parameters
│   ├── config_synthetic.json
│   ├── build_datasets.py  # build train/dev jsonl from CinC2017 data
│   ├── make_synthetic.py  # optional fake data for a smoke test
│   └── setup.sh
└── requirements.txt
```

## License / attribution
This is a re-implementation of the GPL-3.0 `awni/ecg` codebase and follows the
paper's citation:

```bibtex
@article{hannun2019cardiologist,
  title={Cardiologist-level arrhythmia detection and classification in ambulatory
         electrocardiograms using a deep neural network},
  author={Hannun, Awni Y and Rajpurkar, Pranav and Haghpanahi, Masoumeh and
          Tison, Geoffrey H and Bourn, Codie and Turakhia, Mintu P and Ng, Andrew Y},
  journal={Nature Medicine}, volume={25}, number={1}, pages={65}, year={2019}
}
```
