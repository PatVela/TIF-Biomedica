# ECG Web App (Flask)

Aplicación web interactiva para clasificar señales de ECG de **una sola
derivación** con el modelo PyTorch entrenado en CinC2017 (réplica de Hannun et
al. 2019). Subes un archivo de señal y te devuelve el **trazado interactivo**
con la predicción coloreada por intervalo (cada 256 muestras), la distribución
de clases del registro y las predicciones por intervalo.

La app usa el mismo paquete `ecg/` del proyecto (`ecg.network`, el `Preproc` y el
checkpoint guardado por `ecg/train.py`), así que **no duplica la arquitectura ni
la normalización**: usa exactamente el modelo que entrenaste.

## Requisitos

- Un modelo entrenado (un `.pt` de `saved/`), o simplemente la carpeta `saved/`.
  Si aún no lo tienes: `python ecg/train.py examples/cinc17/config.json -e cinc17`.
- `flask`, `matplotlib` y `scipy` (en `requirements.txt`).

## Ejecutar (desarrollo)

```bash
# Opción A: indicar el checkpoint exacto
python webapp/app.py --model saved/cinc17/<timestamp>/0.123-0.980-012-1.000-0.990.pt

# Opción B: elegir automáticamente el mejor (menor val_loss) de una carpeta
python webapp/app.py --saved saved

# puerto/host personalizados (0.0.0.0 permite acceder desde la red local / WSL2)
python webapp/app.py --saved saved --port 5000 --host 0.0.0.0
```

Abre `http://127.0.0.1:5000/`.

## Ejecutar en producción (servidor WSGI)

El servidor de desarrollo de Flask (`app.run`) **no** está pensado para
producción ni para varios usuarios simultáneos. Para desplegarlo de verdad:

**Opción 1 — con `gunicorn` (Linux/macOS/Servidor):**

```bash
pip install gunicorn
# variables de entorno para elegir el modelo (opcional; si no se indican,
# la app usa --saved saved por defecto)
export ECG_MODEL="saved/cinc17/<ts>/<best>.pt"   # o ECG_SAVED="saved"
gunicorn -w 2 -b 0.0.0.0:5000 --timeout 120 "webapp.wsgi:app"
```

**Opción 2 — con `waitress` (Windows nativo):**

```bash
pip install waitress
waitress-serve --listen=*:5000 "webapp.wsgi:app"
```

> La app expone `app` como objeto de nivel de módulo (para poder cargarla por
> `nombre:app`), y `webapp/wsgi.py` lee las variables de entorno
> `ECG_SAVED` / `ECG_MODEL` para arrancar el `PredictionService` en el import.

### Notas de despliegue

- **Carga del modelo:** se realiza una sola vez al arrancar (en el import), no
  por request, así que es seguro usar varios workers de gunicorn.
- **Memoria:** el modelo vive en GPU (o CPU) y se comparte entre request; con un
  único modelo por worker el uso de VRAM es bajo (batch = 1).
- **Hilos:** la app se sirve con `threaded=True`, y el cambio de modelo
  (`/use_model`) está protegido con un lock para no dejar un estado inconsistente.
- **Limpieza de subidas:** los archivos se guardan con un nombre temporal único
  (`uuid4`) y **se borran tras procesarlos**; la carpeta `uploads/` no acumula
  nada.

## Entradas de señal admitidas

| Formato | Detalle |
|---|---|
| **CSV (fila)** | `300,12.0,12.1,...` — primera celda de la primera fila numérica = frecuencia (Hz). Recomendado. |
| **CSV (columna)** | una muestra por fila; asume 300 Hz si no hay frecuencia en el encabezado. |
| **`.mat`** | señal `1×N` (u `N×1`), como `load_ecg`. |
| **`.dat`** | PhysioNet binario formato 212, una derivación. |
| **`.npy`** | array 1-D. |

> **Resampling automático:** el modelo se entrenó a **300 Hz**. Si subes una
> señal a otra frecuencia (CASO FRECUENTE con datos propios), la app la
> **re-muestrea a 300 Hz** con `scipy.signal.resample` y lo avisa en pantalla
> ("Señal re-muestreada de X→300 Hz"). Sin esto, el modelo interpretaría los
> latidos a la velocidad equivocada y las predicciones serían ruido.

## Funcionalidades de la interfaz

- **Gráfico interactivo (Plotly)**: zoom/pan, tooltip al pasar el cursor por cada
  intervalo (clase + probabilidad + tiempo) y sincronización del gráfico con los
  resultados. El JS de Plotly está **embebido localmente**
  (`static/plotly.min.js`), por lo que el gráfico funciona sin conexión/CDN.
- **Selector de checkpoint** en la interfaz: cambia de modelo **sin reiniciar el
  servidor** (`/use_model`).
- **Botón "Probar con ejemplo"**: genera y clasifica una señal en el servidor,
  sin subir nada (`/example`).
- **Drag-and-drop** del archivo sobre la zona de carga.
- **Campo "Etiqueta real"** (opcional): si conoces la clase del registro,
  la app te marca si el resultado coincide (✔/✘).
- **Exportar**: CSV/JSON de las predicciones por intervalo y el PNG del trazado.
- **Métricas del modelo** visibles en la sección "Modelo" (val_loss/acc de
  entrenamiento); las métricas completas se obtienen con
  `examples/cinc17/evaluate.py`.

## Generar un CSV de ejemplo

```bash
python webapp/make_sample_csv.py            # escribe webapp/static/example.csv
```

## Expresar señales desde un `.mat`

```python
import scipy.io as sio, numpy as np
sig = sio.loadmat('rec.mat')['val'].squeeze().astype(float)
np.savetxt('rec.csv', np.column_stack([np.r_[300.0, sig]]), delimiter=',')
```

La app trunca la señal al múltiplo de 256 más cercano (igual que el pipeline de
entrenamiento) antes de inferir.

## Estructura

```
webapp/
├── app.py              # rutas Flask (GET /, POST /predict, /example, /models, /use_model)
├── prediction.py       # PredictionService + resampling + parser CSV + rend. PNG/Plotly
├── make_sample_csv.py  # genera un CSV de ejemplo
├── wsgi.py             # entrada WSGI para gunicorn/waitress (lee ECG_MODEL/ECG_SAVED)
├── templates/index.html
└── static/
    ├── style.css
    ├── app.js
    └── plotly.min.js   # Plotly embebido (sin CDN)
```

## Endpoints

| Ruta | Método | Descripción |
|---|---|---|
| `/` | GET | Interfaz principal. |
| `/predict` | POST | Sube una señal (multipart, campo `file`), opcional `label`; devuelve resultados + plot. |
| `/example` | POST | Genera y clasifica una señal sintética (sin subir archivo). |
| `/models` | GET | Lista los checkpoints disponibles y el actual. |
| `/use_model` | POST | Cambia el checkpoint activo sin reiniciar (JSON `{model: "<rel>"}`). |

## Rendimiento / VRAM

La inferencia es una única pasada por el modelo (batch 1), por lo que consume
muy poca VRAM incluso con un checkpoint grande; en una RTX 3050 (4 GB) no
debería haber problema. `PredictionService` usa CUDA si está disponible y CPU en
caso contrario.
