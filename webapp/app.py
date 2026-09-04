"""Interactive ECG classification web app (Flask) for the PyTorch port of
awni/ecg (Hannun et al., Nature Medicine 2019), trained on PhysioNet CinC2017.

Run (from the project root, virtualenv active):
    python webapp/app.py --model saved/cinc17/<ts>/<best>.pt
    # or auto-select the best (lowest val_loss) checkpoint:
    python webapp/app.py --saved saved
    # custom port / host:
    python webapp/app.py --saved saved --port 5000 --host 0.0.0.0
"""

from __future__ import absolute_import

import argparse
import json
import logging
import os
import sys
import threading
import uuid

# Make the `ecg` package importable regardless of CWD
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

import prediction as pred_mod

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
log = logging.getLogger('ecg-webapp')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024   # 64 MB uploads
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')

SERVICE = None          # current PredictionService
SERVICE_LOCK = threading.Lock()
MODELS_DIR = 'saved'


def _jsonable(result):
    """Drop non-JSON-serialisable payloads (numpy arrays) before responding."""
    out = dict(result)
    out.pop('probs', None)
    return out


def _sanitize(err):
    """Return a short, client-safe error message (no internal paths/stack)."""
    text = str(err)
    # strip anything that looks like a path
    for frag in [os.path.abspath(os.curdir), _REPO_ROOT, '\\n', 'Traceback']:
        text = text.replace(frag, '...')
    return text[:300]


@app.route('/')
def index():
    info = SERVICE.info() if SERVICE is not None else None
    checkpoints = pred_mod.list_checkpoints(MODELS_DIR)
    return render_template('index.html', model_info=info,
                           checkpoints=[os.path.relpath(p, MODELS_DIR)
                                        for p in checkpoints],
                           models_dir=MODELS_DIR)


@app.route('/models', methods=['GET'])
def list_models():
    rel = [os.path.relpath(p, MODELS_DIR) for p in pred_mod.list_checkpoints(MODELS_DIR)]
    return jsonify({'ok': True, 'models': rel, 'current': os.path.relpath(
        SERVICE.model_path, MODELS_DIR) if SERVICE else None})


@app.route('/use_model', methods=['POST'])
def use_model():
    """Switch the loaded checkpoint without restarting the server."""
    global SERVICE
    data = request.get_json(silent=True) or {}
    rel = data.get('model', '')
    path = os.path.join(MODELS_DIR, rel) if rel else pred_mod.find_best_model(MODELS_DIR)
    path = os.path.abspath(path)
    if not os.path.exists(path):
        return jsonify({'ok': False, 'error': 'No existe el checkpoint: ' + rel}), 404
    try:
        svc = pred_mod.PredictionService(path)
    except Exception as e:
        return jsonify({'ok': False, 'error': _sanitize(e)}), 400
    with SERVICE_LOCK:
        SERVICE = svc
    log.info("Modelo cambiado a %s", path)
    return jsonify({'ok': True, 'info': svc.info()})


@app.route('/predict', methods=['POST'])
def predict():
    if SERVICE is None:
        return jsonify({'ok': False, 'error': 'No hay modelo cargado.'}), 500

    file = request.files.get('file')
    if file is None or file.filename == '':
        return jsonify({'ok': False, 'error': 'Selecciona un archivo.'}), 400

    # unique temporary name so concurrent uploads never collide; clean up after.
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ('.csv', '.mat', '.dat', '.npy', ''):
        return jsonify({'ok': False,
                        'error': 'Formato no soportado (usa CSV, .mat, .dat o .npy).'}), 400
    tmp_name = uuid.uuid4().hex + ext
    tmp_path = os.path.join(app.config['UPLOAD_FOLDER'], tmp_name)
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    file.save(tmp_path)

    true_label = (request.form.get('label') or '').strip().upper()
    try:
        fs, signal = pred_mod.load_uploaded_file(tmp_path, file.filename)
        result = SERVICE.predict_signal(fs, signal)
        result['plot'] = 'data:image/png;base64,' + SERVICE.render_plot(fs, signal, result)
        result['plotly'] = SERVICE.render_plotly(fs, signal, result)
        result['models'] = [os.path.relpath(p, MODELS_DIR)
                            for p in pred_mod.list_checkpoints(MODELS_DIR)]
        result['info'] = SERVICE.info()
        # compare against a user-provided ground-truth label, if given
        if true_label and true_label in SERVICE.classes:
            result['ground_truth'] = {'label': true_label,
                                      'correct': true_label == result['dominant']}
    except Exception as e:
        log.warning("Predicción fallida: %s", e)
        return jsonify({'ok': False, 'error': _sanitize(e)}), 400
    finally:
        try:
            os.remove(tmp_path)             # never persist uploads
        except OSError:
            pass

    return jsonify({'ok': True, **_jsonable(result)})


@app.route('/example', methods=['POST'])
def example():
    """Generate a synthetic single-lead signal server-side and classify it,
    so the 'Probar con ejemplo' button needs no file upload."""
    import numpy as np

    if SERVICE is None:
        return jsonify({'ok': False, 'error': 'No hay modelo cargado.'}), 500
    n = 256 * 30
    t = np.arange(n) / pred_mod.TRAIN_FS
    base = 0.05 * np.sin(2 * np.pi * 1.2 * t)
    base += 0.15 * np.sin(2 * np.pi * 1.3 * t).clip(min=0)
    signal = (base + np.random.randn(n) * 0.02).astype(np.float32)
    try:
        result = SERVICE.predict_signal(pred_mod.TRAIN_FS, signal)
        result['plot'] = 'data:image/png;base64,' + SERVICE.render_plot(
            pred_mod.TRAIN_FS, signal, result)
        result['plotly'] = SERVICE.render_plotly(pred_mod.TRAIN_FS, signal, result)
        result['models'] = [os.path.relpath(p, MODELS_DIR)
                            for p in pred_mod.list_checkpoints(MODELS_DIR)]
        result['info'] = SERVICE.info()
    except Exception as e:
        return jsonify({'ok': False, 'error': _sanitize(e)}), 400
    return jsonify({'ok': True, **_jsonable(result)})


def main():
    global SERVICE, MODELS_DIR
    parser = argparse.ArgumentParser(description="ECG classification web app")
    parser.add_argument("--model", help="path to a .pt checkpoint")
    parser.add_argument("--saved", default="saved",
                        help="directory with checkpoints (auto-selects the best)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    MODELS_DIR = args.saved
    model_path = args.model
    if not model_path:
        model_path = pred_mod.find_best_model(MODELS_DIR)
    if not model_path:
        log.error("No se encontró modelo. Entrena primero con "
                  "`python ecg/train.py examples/cinc17/config.json -e cinc17` "
                  "o pasa --model <checkpoint.pt>.")
        sys.exit(1)

    log.info("Cargando modelo: %s", model_path)
    SERVICE = pred_mod.PredictionService(model_path)
    log.info("Clases: %s | Device: %s", SERVICE.classes, SERVICE.device)

    # Run Flask threaded so switching models / concurrent requests do not block.
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == '__main__':
    main()
