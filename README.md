# ecg-pytorch — Detección y clasificación de arritmias a nivel de cardiólogo

Reimplementación fiel en **PyTorch** del repositorio
[`awni/ecg`](https://github.com/awni/ecg), el código abierto que acompaña al paper

> **Cardiologist-Level Arrhythmia Detection and Classification in Ambulatory
> Electrocardiograms Using a Deep Neural Network**
> Hannun, A. Y., Rajpurkar, P., Haghpanahi, M., Tison, G. H., Bourn, C.,
> Turakhia, M. P., & Ng, A. Y. — *Nature Medicine*, 2019.

El repo original es una implementación en **Keras/TensorFlow** (Python 2). Este
proyecto porta **la misma arquitectura y el mismo pipeline de datos** a PyTorch
moderno.

> ### ⚠️ Qué experimento se replica (importante para el informe)
> El paper tiene **dos** resultados:
> - **Resultado principal** (AUC 0.97): clasificador de **12 clases** sobre
>   **91 232 registros privados de iRhythm** (1 derivación, 200 Hz, 30 s).
>   **Los datos NO son públicos.**
> - **Experimento de generalización** (el que se replica aquí): el mismo paper
>   *también* entrenó la DNN sobre el **training set público de PhysioNet
>   CinC2017 (n = 8 528)** y evaluó sobre el test set oculto (n = 3 658),
>   reportando **F1 medio de clase = 0.83**.
>
> Este proyecto reproduce **el experimento de generalización a CinC2017**, no el
> resultado principal de 12 clases. Ver [Fidelidad y desviaciones](#fidelidad-y-desviaciones-respecto-al-paper).

---

## Arquitectura (idéntica al paper)

```
input (B, 1, T)
  │
  ├─ Conv1d(k=16,  stride=1, SAME) ─ BatchNorm ─ ReLU    32 canales
  ├─ 16 × bloques residuales, subsample [1,2,1,2,...,1,2]
  │     cada bloque: shortcut = MaxPool(subsample) (con relleno de ceros en
  │     canales cuando se duplica el nº de canales, cada 4 bloques)
  │     2 × [BN→ReLU→Conv1d(k=16, stride=subsample luego 1)]
  │     canales ×2 cada 4 bloques: 32 → 64 → 128 → 256
  ├─ BatchNorm ─ ReLU
  └─ Linear(num_categorías) ─ softmax  ⇒  salida (B, T/256, num_categorías)
```

Cada paso temporal de la salida (submuestreada) es un softmax sobre las clases de
ritmo; es decir, **una clasificación cada 256 muestras** (~0.85 s a 300 Hz) —
exactamente como en el paper. Los defaults reproducen
`examples/cinc17/config.json` (filter 16, 32 filtros iniciales, 2 convs/bloque,
canales ×2 cada 4 bloques, dropout 0.2, Adam lr 1e-3, ~10.5 M parámetros).

### Correspondencia Keras → PyTorch

| Keras / TF (`awni/ecg`)                    | PyTorch (este repo)                          |
|--------------------------------------------|----------------------------------------------|
| `Conv1D(padding='same')`                   | `nn.Conv1d` + padding `SAME` asimétrico      |
| `MaxPooling1D(pool_size=s)`                | `nn.MaxPool1d(s)` + padding `SAME`           |
| `BatchNormalization` + `Activation('relu')`| `nn.BatchNorm1d` + `nn.ReLU`                 |
| `Dropout`                                  | `nn.Dropout`                                 |
| `TimeDistributed(Dense)` + `softmax`       | `nn.Linear` sobre `(B,T,C)` + `nn.Softmax`   |
| `categorical_crossentropy`                 | `nn.CrossEntropyLoss` sobre logits planos    |
| `Adam(clipnorm=1)`                         | `torch.optim.Adam` + `clip_grad_norm_`       |
| `ReduceLROnPlateau` / `EarlyStopping`      | LR-en-meseta + early-stop                    |
| `ModelCheckpoint .hdf5`                    | `torch.save` `.pt` cada época                |

> **Paridad numérica fijada en código:** BatchNorm usa `eps=1e-3, momentum=0.99`
> (valores Keras) y Adam usa `eps=1e-7` (default Keras), para que el
> entrenamiento se acerque lo máximo al original.
>
> **Convención de formas:** Keras es `(batch, time, channels)`, PyTorch es
> `(batch, channels, time)`. La red lo maneja internamente: los callers siempre
> pasan/obtienen `(B, 1, T)`.
>
> **Padding `SAME`:** implementado de forma asimétrica para que la longitud de
> salida coincida exactamente con Keras/TF (`out = ceil(n / stride)`). Es
> esencial para reproducir el downsample temporal (÷256).

---

## Requisitos

- **Python 3.13.5** (CPython 3.13, wheels `cp313`)
- **PyTorch con CUDA 13.0** para GPU (RTX 3050 Laptop = Ampere, `sm_86`). El
  `requirements.txt` fija el wheel con el local-version tag `+cu130` para forzar
  el build de CUDA (en lugar del de CPU).
- Driver NVIDIA que soporte CUDA 13.0 (≈ driver 570+). Si tu driver es anterior,
  usa la variante `cu128` del `requirements.txt` y `torch==2.14.0+cu128`.

```bash
python -m venv venv
source venv/bin/activate        # o .\venv\Scripts\Activate.ps1 en Windows
pip install -r requirements.txt

# Verificar CUDA:
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
# Debe imprimir algo como: 2.14.0+cu130 True 13.0
```

---

## Datos (PhysioNet CinC2017)

El dataset CinC2017 está restringido por un acuerdo de uso (requiere cuenta de
PhysioNet). Descarga desde:
**<https://physionet.org/content/challenge-2017/1.0.0/>**

Necesitas:
- `training2017/` (los registros) — o `training2017.zip`
- `REFERENCE-v3.csv` (las etiquetas)

Luego genera los conjuntos `train.json` / `dev.json`:

```bash
python examples/cinc17/build_datasets.py \
  --data_dir   dataset2017/training2017 \
  --label_file dataset2017/REFERENCE-v3.csv \
  --out_dir    examples/cinc17 \
  --relative \          # guarda rutas relativas (portable, sin C:\Windows\
  --stratify            # split proporcional a la clase (recomendado)
```

Opciones de `build_datasets.py`:
| Flag | Efecto |
|---|---|
| `--relative` | Guarda las rutas relativas a la raíz del proyecto (portables entre máquinas). |
| `--stratify` | Divide proporcionalmente por clase (evita un dev con pocas muestras de las clases minoritarias `A`/`~`). |
| `--dev_frac` | Fracción de dev (default `0.1`). |

Esto escribe `examples/cinc17/train.json` y `dev.json` (JSONL, un registro por
línea, con la ruta `ecg` y las `labels` repetidas cada 256 muestras).

---

## Entrenar

```bash
# desde la raíz del proyecto (las rutas de config.json son relativas a la raíz)
python ecg/train.py examples/cinc17/config.json -e cinc17
```

- Los checkpoints se guardan cada época en
  `saved/<experimento>/<timestamp>/<val_loss>-<val_acc>-<epoch>-...pt`, junto con
  el preprocesador (`preproc.bin`).
- **El mejor modelo = menor `val_loss`** (primer número del nombre).
- Callbacks del entrenamiento (iguales al repo original): `EarlyStopping(patience=8)`,
  `ReduceLROnPlateau(factor=0.1, patience=2, min_lr=lr*0.001)`, checkpoint por
  época.

> `config.json` está en `batch_size: 32` (paridad exacta con `awni/ecg`).
> El paper usaba 128; si tu GPU no lo permite, baja el batch.

---

## Predecir

```bash
python ecg/predict.py examples/cinc17/dev.json "saved\cinc17\<ts>\0.408-0.862-011-0.274-0.904.pt"
```

Imprime la **clase mayoritaria por registro** (moda sobre los intervalos),
replicando `entry/evaler.py`. En PowerShell puedes autoseleccionar el mejor:

```powershell
$best = Get-ChildItem -Path saved -Recurse -Filter "*.pt" |
  Sort-Object { [double]($_.BaseName -split "-")[0] } | Select-Object -First 1
python ecg/predict.py examples/cinc17/dev.json $best.FullName
```

> **Nota (padding):** `predict.py` evalúa **cada registro de forma independiente**
> (sin rellenar entre registros). Si se rellenara todo el lote a la longitud del
> mayor, las zonas de ceros (aprendidas como `~`/ruido) inundarían de votos `~`
> a los registros cortos y sesgarían la moda. Evaluar por registro da un
> resultado honesto.

---

## Evaluar

```bash
# Evalúa sobre dev.json con el mejor checkpoint (auto-selección):
python examples/cinc17/evaluate.py --data_json examples/cinc17/dev.json --saved saved

# O con un checkpoint exacto:
python examples/cinc17/evaluate.py --data_json examples/cinc17/dev.json `
  --model_path "saved\cinc17\<ts>\0.408-0.862-011-0.274-0.904.pt"

# Niveles: --level record | interval | both (default both)
```

Imprime, por nivel:
- **Exactitud (accuracy)**
- **Macro-F1** (todas las clases) y **Weighted-F1**
- **Challenge-F1 (N/A/O)** ← métrica **oficial** del challenge/paper
- **Matriz de confusión** y **reporte por clase** (precision/recall/F1)

> **Cómo interpretarlo:**
> - **Nivel registro** (set-level): una clase por registro (la moda). Es la
>   métrica que el paper reporta para CinC2017.
> - **Nivel intervalo** (sequence-level): **solo diagnóstico interno** —
>   CinC2017 no tiene etiqueta por intervalo; NO es una réplica del paper.
> - **`Challenge-F1`** promedia **solo** `N, A, O` (excluye `~`). Es la métrica
>   comparable con el **0.83 del paper**. En nuestro dev set da ≈ **0.845**.

---

## Aplicación web (Flask)

Aplicación interactiva: subes un CSV de ECG de una sola derivación y muestra el
trazado con la predicción coloreada por intervalo y la distribución de clases.

```bash
# Elige el mejor checkpoint automáticamente:
python webapp/app.py --saved saved

# O un checkpoint exacto / puerto personalizado:
python webapp/app.py --model "saved\cinc17\<ts>\0.408-....pt" --port 5000
```

Abre <http://127.0.0.1:5000/>. Para acceder desde otro dispositivo o WSL2:
`python webapp/app.py --saved saved --host 0.0.0.0`.

**Formato del CSV** (igual al proyecto de referencia): la **primera celda de la
primera fila numérica es la frecuencia de muestreo (Hz)** y el resto son las
muestras de la señal.

```
Lead,II
300,-0.0039,0.0290,0.0159,0.0338,...
```

Genera un CSV de ejemplo: `python webapp/make_sample_csv.py`.
Más detalles en [`webapp/README.md`](webapp/README.md).

---

## Smoke test (sin descargar datos)

Para verificar que todo el pipeline corre (con señales sintéticas):

```bash
python examples/cinc17/make_synthetic.py --train 96 --dev 48
# (config_synthetic.json apunta a examples/cinc17/synthetic/)
python -m ecg.train examples/cinc17/config_synthetic.json -e synth --epochs 2
python ecg/predict.py examples/cinc17/synthetic/dev.json "saved\synth\<ts>\<mejor>.pt"
```

> La señal sintética no puede aprender nada real; solo valida el pipeline.

---

## Fidelidad y desviaciones (respecto al paper / repo original)

Sección para la tesis/informe — conviene declararla explícitamente.

### 1. Qué experimento replicamos
Ver el aviso superior. Reproducimos **el experimento de generalización a
CinC2017** (n = 8 528; F1 = 0.83 en el test oculto), **no** el resultado
principal de 12 clases con AUC = 0.97 (dato iRhythm privado).

### 2. No usamos el test set real (hoy)
Nuestros 8 528 registros = el training set público completo de CinC2017, partido
90/10 (`train.json` 7 676 + `dev.json` 852). Consecuencias:
- El **0.83** del paper se mide sobre el **test set oculto (3 658 registros)** por
  el sistema oficial de PhysioNet → no se puede reproducir directamente.
- `dev.json` ya se usó para **early stopping / selección de checkpoint**, así que
  evaluar ahí sobreestima el rendimiento real (**leakage de selección de
  modelo**).

> **Para un número sin sesgo:** PhysioNet liberó las etiquetas del **test set**
> tras cerrar el challenge. Descarga el test set + `REFERENCE-v3.csv`,
> genera un `test.json` (igual que `build_datasets.py` pero apuntando a los
> registros de test) y evalúa:
> `python examples/cinc17/evaluate.py --data_json examples/cinc17/test.json --saved saved`.

### 3. F1 oficial del challenge
Los **0.83** del paper usan **F1 = (F1_N + F1_A + F1_O) / 3** (solo Normal, AF,
Other; **excluye** `~`). `evaluate.py` reporta **ambas**: `Macro-F1` (todas las
clases, métrica interna, NO comparable) y `Challenge-F1` (la oficial).

### 4. Secuencia vs registro
Para CinC2017 el paper reporta **solo nivel registro** (voto mayoritario). El
nivel intervalo de `evaluate.py` es un **diagnóstico interno**, no el número del
paper.

### 5. Hiperparámetros frente al repo original
El resto de `config.json` coincide exacto con
`awni/ecg/examples/cinc17/config.json` (filter 16, 32 iniciales, dropout 0.2,
num_skip 2, increase_channels_at 4, lr 1e-3). Única diferencia: **batch_size**.

| Referencia                         | batch_size |
|------------------------------------|------------|
| Paper (Nature Medicine)            | **128**    |
| `awni/ecg` (repo original)         | **32**     |
| Este proyecto                      | **32**     |

`EarlyStopping(patience=8)`, `ReduceLROnPlateau(factor=0.1, patience=2,
min_lr=lr*0.001)` y checkpoint por época coinciden con el `train.py` original.

### 6. Paridad numérica Keras ↔ PyTorch
- **BatchNorm**: Keras `eps=1e-3, momentum=0.99` vs PyTorch default `1e-5, 0.1`.
  Fijado en `network._BNRelu`.
- **Adam `epsilon`**: Keras `1e-7` vs PyTorch `1e-8`. Fijado en `train.py`.

### 7. Características conocidas / heredadas del repo original
- `load.py` rellena (pad) el final del lote con la pseudo-clase `~` (comportamiento
  original de Keras). El índice de relleno ahora se deriva del vocabulario
  (`Preproc.pad_value`), no está hardcodeado a 3.
- La pérdida **no enmascara** las regiones rellenadas (consistente con el original).
- El split train/dev es aleatorio por defecto; usa `--stratify` para uno
  proporcional por clase.

---

## Estructura del proyecto

```
ecg_pytorch/
├── ecg/                     # paquete principal (port PyTorch)
│   ├── network.py           # la CNN (ECGNetwork), portada 1:1
│   ├── load.py              # Preproc + generador + lectores WFDB/.mat/.dat
│   ├── train.py             # loop de entrenamiento
│   ├── predict.py           # inferencia + clase mayoritaria por registro
│   ├── evaluate.py          # (ver examples/) evaluación formal real
│   ├── util.py              # guardar/cargar preprocesador
│   └── __init__.py
├── examples/
│   └── cinc17/
│       ├── config.json      # hiperparámetros del paper (batch 32)
│       ├── config_synthetic.json
│       ├── build_datasets.py# genera train.json / dev.json desde CinC2017
│       ├── evaluate.py      # métricas oficiales + matriz de confusión
│       ├── make_synthetic.py# datos ficticios para smoke test
│       └── setup.sh
├── webapp/                  # aplicación web Flask
│   ├── app.py
│   ├── prediction.py
│   ├── make_sample_csv.py
│   ├── templates/index.html
│   └── static/{style.css, app.js}
├── requirements.txt
└── README.md
```

---

## Licencia

Reimplementación del código GPL-3.0 de `awni/ecg`; sigue la cita del paper:

```bibtex
@article{hannun2019cardiologist,
  title={Cardiologist-level arrhythmia detection and classification in ambulatory
         electrocardiograms using a deep neural network},
  author={Hannun, Awni Y and Rajpurkar, Pranav and Haghpanahi, Masoumeh and
          Tison, Geoffrey H and Bourn, Codie and Turakhia, Mintu P and Ng, Andrew Y},
  journal={Nature Medicine}, volume={25}, number={1}, pages={65}, year={2019}
}
```
