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

### Acceso desde otras redes (LAN e internet)

**Misma red (Wi-Fi / LAN):** ejecuta con `--host 0.0.0.0` (o `--listen=*:PUERTO`)
y abre la **IP local** de tu máquina (Windows: `ipconfig` → IPv4). Debes permitir
el puerto en el **firewall** (Windows Defender Firewall → permitir puerto entrante,
o regla de entrada para el puerto elegido). Desde otro dispositivo de la misma red
usa `http://<IP-de-tu-PC>:5000/` (no `127.0.0.1`).

**Otra red (internet), sin IP pública ni port-forwarding:** usa un **túnel HTTPS**.
El script `webapp/run_public.sh` abre uno automáticamente:

```bash
# opción 1 — Cloudflare Tunnel (gratuito, sin cuenta para uso básico)
./webapp/run_public.sh          # imprime la URL pública https://...trycloudflare.com

# opción 2 — ngrok (requiere token)
ngrok config add-authtoken <TU_TOKEN>
TUNNEL=ngrok ./webapp/run_public.sh
```

El túnel **termina TLS**, así que cualquiera en otra red abre la app por
`https://<url-pública>/` de forma segura. **Recomendación:** usa un túnel solo
para demo; cierra la URL cuando termines (la URL es accesible por cualquiera que
la tenga). Para un despliegue estable, sube a Render/Heroku (ver abajo).

> En la interfaz, para abrir desde otro dispositivo recuerda que NO se usa
> `127.0.0.1` (loopback de la app) sino la IP/URL pública. `0.0.0.0` y `[::]`
> son direcciones de escucha del servidor, no URL de navegación.

### HTTPS / Seguridad

`wsgi.py` y `app.py` **no pueden** hacer TLS por sí solos de forma "mágica"; el
HTTPS se termina en la capa de servidor/proxy. Tienes dos vías:

**A) TLS directo con un certificado** (para prueba interna / demo):
```bash
./webapp/run_https.sh    # genera un cert autofirmado en certs/ y sirve por https://
# o con tu propio certificado:
CERT=mi-cert.pem KEY=mi-key.pem ./webapp/run_https.sh
```
(waitress necesita `--ssl-certfile`/`--ssl-keyfile`; `run_https.sh` se los pasa.)

**B) TLS terminado por un proxy / plataforma** (producción real):
- **nginx + Let's Encrypt (certbot)** como reverse proxy delante de gunicorn/waitress.
- **caddy** (configura TLS automáticamente con un dominio).
- **Render / Heroku / Railway** → gestionan HTTPS automáticamente; solo expones
  el `Procfile` (`web: gunicorn ...`) y el certificado es de la plataforma.

**Cabeceras de seguridad** (ya aplicadas por `add_security_headers` en `app.py`):
`Content-Security-Policy`, `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`,
`Referrer-Policy`, `Permissions-Policy`, `Cache-Control: no-store`, y
**`Strict-Transport-Security` (HSTS)** que se activa **solo** cuando la petición
llega por HTTPS (detectado por `request.is_secure` o `X-Forwarded-Proto: https`).
Configura `ProxyFix`/un proxy que ponga `X-Forwarded-Proto` si usas reverso proxy.

> **Privacidad:** los campos de paciente (nombre/edad) se usan **solo en el
> navegador** para el informe; no se envían al servidor ni se persisten. No
> introduzcas datos de salud reales en una demo pública.

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

- **Gráfico interactivo (Plotly)**: zoom/pan, tooltip por intervalo (clase +
  probabilidad + tiempo). El JS de Plotly está **embebido localmente**, por lo
  que funciona sin conexión/CDN.
- **Pestañas**: *Visualización* (gráfico), *Informe* (imprimible a PDF) y
  *Detalle técnico*. Permite ver el ECG completo sin saturar la pantalla.
- **Modo claro / oscuro** (toggle 🌓, se recuerda en el navegador).
- **Bilingüe** ES/EN (toggle en la cabecera).
- **Resultado en lenguaje natural**: tarjeta de diagnóstico destacada + un
  **semáforo por clase** (predominante / presente / bajo) — pensado para personal
  sanitario.
- **Validación**: checkbox "confirmo que es un ECG de una sola derivación" antes
  de analizar.
- **Informe imprimible**: paciente/edad/fecha, ritmo predominante, distribución y
  el trazado del ECG (botón "Descargar informe (PDF)").
- **Botón "Usar señal de prueba"** (sin subir nada) y **drag-and-drop**.
- **Selector de checkpoint automático**: el servidor elige el mejor modelo
  (menor `val_loss`) al arrancar; no hay que elegir nada en la interfaz.
- **Comparar con etiqueta real** (opcional, campo "Diagnóstico conocido").
- **Métricas del modelo** y **re-muestreo automático** a 300 Hz con aviso.

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
├── app.py              # rutas Flask + cabeceras de seguridad (add_security_headers)
├── prediction.py       # PredictionService + resampling + parser CSV + rend. PNG/Plotly
├── make_sample_csv.py  # genera un CSV de ejemplo
├── wsgi.py             # entrada WSGI para gunicorn/waitress (lee ECG_MODEL/ECG_SAVED)
├── run_https.sh        # sirve por HTTPS con certificado (autofirmado o propio)
├── run_public.sh       # expone la app a internet con un túnel HTTPS (cloudflared/ngrok)
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
