"""Interactive ECG classification web app (Flask) for the PyTorch port of
awni/ecg (Hannun et al., Nature Medicine 2019), trained on PhysioNet CinC2017.

Run (from the project root, virtualenv active):
    python webapp/app.py --model saved/cinc17/<ts>/<best>.pt
    # or auto-select the best (lowest val_loss) checkpoint:
    python webapp/app.py --saved saved
    # custom port / host:
    python webapp/app.py --saved saved --port 5000 --host 0.0.0.0

Then open http://127.0.0.1:5000/ and upload a CSV of a single-lead ECG.
"""

from __future__ import absolute_import

import argparse
import json
import os
import sys

# Make the `ecg` package importable regardless of CWD
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

import prediction as pred_mod

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024   # 32 MB uploads
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')

SERVICE = None          # PredictionService, set in main()


@app.route('/')
def index():
    model_info = None
    if SERVICE is not None:
        model_info = {
            'classes': SERVICE.classes,
            'device': str(SERVICE.device),
            'default_fs': pred_mod.DEFAULT_FS,
        }
    return render_template('index.html', model_info=model_info)


@app.route('/predict', methods=['POST'])
def predict():
    if SERVICE is None:
        return jsonify({'ok': False, 'error': 'No hay modelo cargado.'}), 500

    file = request.files.get('file')
    if file is None or file.filename == '':
        return jsonify({'ok': False, 'error': 'Selecciona un archivo CSV.'}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    file.save(save_path)

    try:
        text = open(save_path, 'r', encoding='utf-8', errors='replace').read()
    except Exception as e:                       # pragma: no cover
        return jsonify({'ok': False, 'error': 'No se pudo leer el archivo: {}'.format(e)}), 400

    try:
        fs, signal = pred_mod.parse_ecg_csv(text)
        result = SERVICE.predict_signal(signal)
        plot_b64 = SERVICE.render_plot(fs, signal, result)
    except Exception as e:                       # pragma: no cover
        return jsonify({'ok': False, 'error': str(e)}), 400

    return jsonify({
        'ok': True,
        'fs': fs,
        'n_samples': int(signal.shape[0]),
        'n_intervals': result['n_intervals'],
        'dominant': result['dominant'],
        'summary': result['summary'],
        'per_interval': result['per_interval'][:400],   # cap payload
        'plot': 'data:image/png;base64,' + plot_b64,
    })


def main():
    global SERVICE
    parser = argparse.ArgumentParser(description="ECG classification web app")
    parser.add_argument("--model", help="path to a .pt checkpoint")
    parser.add_argument("--saved", default="saved",
                        help="directory with checkpoints (auto-selects the best)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    model_path = args.model
    if not model_path:
        model_path = pred_mod.find_best_model(args.saved)
    if not model_path:
        print("No se encontró modelo. Entrena primero con "
              "`python ecg/train.py examples/cinc17/config.json -e cinc17` "
              "o pasa --model <checkpoint.pt>.")
        sys.exit(1)

    print("Cargando modelo:", model_path)
    SERVICE = pred_mod.PredictionService(model_path)
    print("Clases:", SERVICE.classes)
    print("Device:", SERVICE.device)

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()
