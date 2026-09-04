# ecg-pytorch — Detección de arritmias a nivel de cardiólogo (PyTorch)

Una **reimplementación fiel en PyTorch** de [`awni/ecg`](https://github.com/awni/ecg), el código abierto que acompaña al artículo

> **Cardiologist-Level Arrhythmia Detection and Classification in Ambulatory Electrocardiograms Using a Deep Neural Network**
> Hannun, A. Y., Rajpurkar, P., Haghpanahi, M., Tison, G. H., Bourn, C., Turakhia, M. P., & Ng, A. Y. — *Nature Medicine*, 2019.

El repositorio original está implementado en **Keras/TensorFlow (Python 2)**. Este proyecto traslada **la misma arquitectura y el mismo flujo de procesamiento de datos** 1:1 a PyTorch moderno.

> ## Decisión sobre el conjunto de datos (importante)
>
> Se busca una **réplica exacta** del artículo.
>
> - El artículo y `awni/ecg` están construidos y validados sobre el conjunto **PhysioNet CinC 2017**: ECG de **una sola derivación (single-lead)**, **una etiqueta por registro**, reproducida en cada paso de salida de **256 muestras**. Este es exactamente el escenario que muestra el README del repositorio original (`examples/cinc17/`), por lo que constituye el **conjunto de datos principal** del proyecto.
> - El conjunto de datos original de **iRhythm** utilizado en el artículo **no es público**. Usar **CinC 2020** (12 derivaciones y **múltiples etiquetas**) modificaría tanto el formato de los datos como la función de pérdida y parte de la arquitectura (BCE multietiqueta y 12 canales de entrada). El soporte para CinC 2020 se explica más adelante.

---

# Arquitectura (coincide exactamente con el artículo)

```text
entrada (B, 1, T)
  │
  ├─ Conv1d(k=16, stride=1, SAME) ─ BatchNorm ─ ReLU    32 canales
  ├─ 16 × bloques residuales, longitudes de submuestreo [1,2,1,2,...,1,2]
  │     cada bloque:
  │       - Atajo con MaxPool(subsample)
  │       - Zero-padding cuando el número de canales se duplica
  │       - 2 × [BatchNorm → ReLU → Conv1d(k=16)]
  │
  │     Los canales se duplican cada 4 bloques:
  │       32 → 64 → 128 → 256
  │
  ├─ BatchNorm ─ ReLU
  └─ Linear(num_categories) ─ Softmax
        ⇒ salida (B, T/256, num_categories)
```

Cada paso temporal de la salida reducida produce una distribución **Softmax** sobre las clases de ritmo cardíaco; es decir, una clasificación cada **256 muestras** (≈0.85 s a 300 Hz), exactamente como en el artículo.

La configuración por defecto reproduce `examples/cinc17/config.json` del repositorio original:

- Longitud del filtro: **16**
- Filtros iniciales: **32**
- Dos convoluciones por bloque residual
- Duplicación de canales cada cuatro bloques
- Dropout: **0.2**
- Adam con **lr = 1e-3**
- Aproximadamente **10.5 millones de parámetros**

## Correspondencia con el código original en Keras

| Keras / TensorFlow (`awni/ecg`) | PyTorch (este proyecto) |
|---------------------------------|---------------------------|
| `Conv1D(padding='same')` | `nn.Conv1d` con padding `SAME` asimétrico |
| `MaxPooling1D(pool_size=s)` | `nn.MaxPool1d(s)` con padding `SAME` |
| `BatchNormalization` + `Activation('relu')` | `nn.BatchNorm1d` + `nn.ReLU` |
| `Dropout` | `nn.Dropout` |
| `TimeDistributed(Dense)` + `Softmax` | `nn.Linear` sobre `(B,T,C)` + `nn.Softmax` |
| `categorical_crossentropy` | `nn.CrossEntropyLoss` |
| `Adam(clipnorm=1)` | `torch.optim.Adam` + `clip_grad_norm_` |
| `ReduceLROnPlateau` / `EarlyStopping` | Reducción manual del LR + parada temprana |
| `ModelCheckpoint (.hdf5)` | `torch.save` (`.pt`) |

### Convención de dimensiones

Keras utiliza:

```text
(batch, tiempo, canales)
```

PyTorch utiliza:

```text
(batch, canales, tiempo)
```

La red realiza esta conversión internamente, por lo que las entradas siempre se proporcionan como:

```text
(B, 1, T)
```

### Padding `SAME`

El padding `SAME` se implementa de forma **asimétrica** para que la longitud de salida coincida exactamente con TensorFlow/Keras:

<math value="\\text{salida}=\\lceil n/\\text{stride}\\rceil"/>

Esto es fundamental para reproducir correctamente la reducción temporal del artículo (división total entre **256**).

---

# Instalación

```bash
pip install -r requirements.txt
```

Esto instalara las dependencias principales.

---

# Descargar los datos (PhysioNet CinC 2017)

Descarga:

- `training2017.zip`
- `REFERENCE-v3.csv`

desde:

<https://physionet.org/content/challenge-2017/1.0.0/>

Colócalos en:

```text
examples/cinc17/data/
```

Luego construye los conjuntos de entrenamiento y validación.

### Opción automática

```bash
cd examples/cinc17
bash setup.sh
```

### Opción manual

```bash
python build_datasets.py \
    --data_dir data/training2017 \
    --label_file data/REFERENCE-v3.csv
```

Esto genera:

- `examples/cinc17/train.json`
- `examples/cinc17/dev.json`

en formato **JSONL**, con un registro por línea que contiene:

- la ruta del ECG (`ecg`)
- las etiquetas repetidas (`labels`) para cada paso temporal.

---

# Entrenamiento

Desde la raíz del repositorio:

```bash
python -m ecg.train examples/cinc17/config.json -e cinc17
```

Los checkpoints se guardan en:

```text
saved/<experimento>/<timestamp>/
```

con nombres del tipo:

```text
6.210-0.354-003-....pt
```

También se almacena el preprocesador:

```text
preproc.bin
```

El mejor modelo corresponde al checkpoint con el **menor `val_loss`**.

---

# Predicción

```bash
python -m ecg.predict \
    examples/cinc17/dev.json \
    saved/cinc17/<timestamp>/6.210-0.354-003-....pt
```

El programa imprime la **clase mayoritaria** de cada registro (la moda sobre todas las predicciones temporales), reproduciendo el comportamiento de `entry/evaler.py` del repositorio original.

---

# Prueba rápida sin descargar datos

Para comprobar que toda la tubería funciona de extremo a extremo con ECG sintéticos:

```bash
python examples/cinc17/make_synthetic.py --train 80 --dev 40

python -m ecg.train \
    examples/cinc17/config_synthetic.json \
    -e synth \
    --epochs 2
```

Después puede utilizarse el checkpoint generado para realizar predicciones.

> Los datos sintéticos no contienen información clínica real; únicamente sirven para verificar que el pipeline funciona correctamente.

---

# Soporte para CinC 2020

CinC 2020 se incluye como una posible extensión futura, pero **no constituye una réplica exacta del artículo**.

| Aspecto | CinC 2017 (este proyecto) | CinC 2020 |
|----------|--------------------------|-----------|
| Derivaciones | 1 | 12 |
| Etiqueta | Una clase por registro | Múltiples diagnósticos SNOMED-CT |
| Pérdida | Softmax categórico | BCE multietiqueta |
| Formato | `.mat` / `212` | WFDB (`.dat` + `.hea`) |

## Extensión prevista

El cargador (`load_ecg`) ya admite múltiples formatos:

- `.mat`
- `.dat`
- `.npy`
- WFDB

actualmente seleccionando la primera derivación.

Para soportar completamente CinC 2020 bastaría con:

1. cargar las 12 derivaciones mediante `wfdb.rdrecord()`,
2. modificar la primera convolución para aceptar **12 canales de entrada**, y
3. reemplazar la pérdida por **Binary Cross-Entropy** sobre múltiples diagnósticos.

Los puntos de extensión principales son:

- `ecg/network.py`
- `ecg/load.py`

---

# Estructura del repositorio

```text
ecg_pytorch/
├── ecg/
│   ├── network.py          # CNN (ECGNetwork), portada 1:1
│   ├── load.py             # Preprocesamiento y carga de datos
│   ├── train.py            # Entrenamiento
│   ├── predict.py          # Inferencia
│   ├── util.py             # Guardado del preprocesador
│   └── __init__.py
│
├── examples/cinc17/
│   ├── config.json
│   ├── config_synthetic.json
│   ├── build_datasets.py
│   ├── make_synthetic.py
│   └── setup.sh
│
└── requirements.txt
```

---

# Licencia y atribución

Este proyecto es una **reimplementación** del repositorio **GPL-3.0 `awni/ecg`** y mantiene la atribución correspondiente al artículo original.

## Cita recomendada

```bibtex
@article{hannun2019cardiologist,
  title={Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms using a deep neural network},
  author={Hannun, Awni Y and Rajpurkar, Pranav and Haghpanahi, Masoumeh and Tison, Geoffrey H and Bourn, Codie and Turakhia, Mintu P and Ng, Andrew Y},
  journal={Nature Medicine},
  volume={25},
  number={1},
  pages={65},
  year={2019}
}
```

Esta implementación busca reproducir con la mayor fidelidad posible la arquitectura, el flujo de datos y el comportamiento del código original en Keras/TensorFlow, utilizando un entorno moderno basado en **PyTorch** y el conjunto de datos público **PhysioNet CinC 2017**, que es el escenario reproducible más cercano al empleado en el artículo original.