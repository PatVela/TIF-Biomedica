# ECG Web App (Flask)

Aplicación web interactiva para clasificar señales de ECG de **una sola
derivación** con el modelo PyTorch entrenado en CinC2017 (réplica de Hannun et
al. 2019). Subes un **CSV** y te devuelve el trazado con la predicción coloreada
por intervalo (cada 256 muestras) y la distribución de clases del registro.

La app dirige su lógica al mismo paquete `ecg/` del proyecto
(`ecg.network`, el `Preproc` y el checkpoint guardado por `ecg/train.py`), así
que **no duplica la arquitectura ni la normalización**: usa exactamente el modelo
que entrenaste.

## Requisitos

- Un modelo entrenado (un `.pt` de `saved/`), o simplemente la carpeta `saved/`.
  Si no lo tienes aún: `python ecg/train.py examples/cinc17/config.json -e cinc17`.
- `flask` + `matplotlib` (añade Flask a `requirements.txt` si no está).

## Ejecutar

```bash
# Opción A: indicar el checkpoint exacto
python webapp/app.py --model saved/cinc17/<timestamp>/0.123-0.980-012-1.000-0.990.pt

# Opción B: elegir automáticamente el mejor (menor val_loss) de una carpeta
python webapp/app.py --saved saved

# puerto/host personalizados
python webapp/app.py --saved saved --port 5000 --host 0.0.0.0
```

Abre `http://127.0.0.1:5000/`. El host `0.0.0.0` permite acceder desde la red
local (por ejemplo para compartir en la misma Wi-Fi o en WSL2).

## Formato del CSV

**Formato fila (recomendado, como el proyecto de referencia):**
la **primera celda de la primera fila de números es la frecuencia de muestreo**,
el resto son las muestras de la señal.

```
Lead,II
300,-0.0039,0.0290,0.0159,0.0338,...
```

**Formato columna:** una muestra por fila; si el encabezado empieza con la
frecuencia (p.ej. `300,Lead`) se usa esa, si no se asume 300 Hz.

> Genera un CSV de ejemplo de una sola vez con:
> `python webapp/make_sample_csv.py`

## Expresar el modelo / señales desde un archivo `.mat`

Si tienes un `.mat` con la señal (shape `(1, N)`), puedes exportarlo a CSV:

```python
import scipy.io as sio, numpy as np
sig = sio.loadmat('rec.mat')['val'].squeeze().astype(float)
np.savetxt('rec.csv', np.column_stack([np.r_[300.0, sig]]), delimiter=',')
```

La app recorta la señal al múltiplo de 256 más cercano (igual que el pipeline de
entrenamiento) antes de inferir.

## Estructura

```
webapp/
├── app.py              # rutas Flask: GET /  y  POST /predict
├── prediction.py       # PredictionService (carga modelo) + parse CSV + plot
├── make_sample_csv.py  # genera un CSV de ejemplo
├── templates/index.html
└── static/
    ├── style.css
    └── app.js
```

## Rendimiento / VRAM

La inferencia es una sola pasada por el modelo (batch 1), así que consume muy
poca VRAM incluso con un checkpoint grande; en la RTX 3050 (4 GB) no debería
haber problema. `PredictionService` usa `cuda` si está disponible y `cpu` en
caso contrario.
