import os
import uuid
import json
import joblib
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, send_from_directory
from utils import preprocess

app = Flask(__name__)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# ── Model paths ────────────────────────────────────────────────────────────────
XGB_MODEL_PATH      = os.path.join('models', 'xgboost_model.pkl')
CNN_LSTM_MODEL_PATH = os.path.join('models', 'cnn_lstm.keras')

# Label convention (release_in_the_wild): 1 = fake/spoof, 0 = genuine/bonafide
LABEL_MAP = {0: 'genuine', 1: 'fake'}

# ── Load models ────────────────────────────────────────────────────────────────
if os.path.exists(XGB_MODEL_PATH):
    xgb_model = joblib.load(XGB_MODEL_PATH)
    print(f'✅ XGBoost model loaded from {XGB_MODEL_PATH}')
else:
    xgb_model = None
    print('⚠️  XGBoost model not found – predictions will be NULL.')

if os.path.exists(CNN_LSTM_MODEL_PATH):
    cnn_lstm_model = tf.keras.models.load_model(CNN_LSTM_MODEL_PATH)
    print(f'✅ CNN-LSTM model loaded from {CNN_LSTM_MODEL_PATH}')
else:
    cnn_lstm_model = None
    print('⚠️  CNN-LSTM model not found – predictions will be NULL.')

# ── Constants ──────────────────────────────────────────────────────────────────
MAX_FRAMES  = 300   # must match training
IMG_HEIGHT  = 128   # mel bins, must match training

# ── Feature extraction helpers ─────────────────────────────────────────────────

def _safe_spectrogram(audio_path):
    """Extract mel-spectrogram and guarantee shape (IMG_HEIGHT, MAX_FRAMES, 1)."""
    spec = preprocess.extract_spectrogram(audio_path)  # expected: (IMG_HEIGHT, T)

    # Guard: some preprocess implementations return (T, IMG_HEIGHT) — fix it
    if spec.ndim == 2 and spec.shape[0] != IMG_HEIGHT and spec.shape[1] == IMG_HEIGHT:
        spec = spec.T  # transpose to (IMG_HEIGHT, T)

    # Pad / truncate on the time axis
    if spec.shape[1] < MAX_FRAMES:
        pad_width = MAX_FRAMES - spec.shape[1]
        spec = np.pad(spec, ((0, 0), (0, pad_width)), mode='constant')
    else:
        spec = spec[:, :MAX_FRAMES]

    # Shape → (1, IMG_HEIGHT, MAX_FRAMES, 1) for model.predict
    return spec[np.newaxis, ..., np.newaxis]


def _safe_gfcc(audio_path):
    """Extract GFCC and guarantee shape (1, n_coeffs) for XGBoost."""
    gfcc = preprocess.extract_gfcc(audio_path)
    return gfcc.reshape(1, -1)

# ── Core prediction ────────────────────────────────────────────────────────────

def predict_from_audio(audio_path):
    """
    Run both models and return a structured result dict:
    {
        'xgboost':  {'label': 'fake'|'genuine'|null, 'confidence': float|null},
        'cnn_lstm': {'label': 'fake'|'genuine'|null, 'confidence': float|null},
        'verdict':  'fake'|'genuine'|'uncertain'|'unavailable'
    }
    confidence is the probability of the PREDICTED class (always >= 0.5).
    """
    result = {
        'xgboost':  {'label': None, 'confidence': None},
        'cnn_lstm': {'label': None, 'confidence': None},
        'verdict':  'unavailable',
    }

    # ── XGBoost (GFCC features) ──────────────────────────────────────────────
    if xgb_model is not None:
        try:
            gfcc = _safe_gfcc(audio_path)
            # Use predict_proba for a calibrated probability instead of a hard threshold
            proba = xgb_model.predict_proba(gfcc)[0]   # [P(genuine), P(fake)]
            fake_prob   = float(proba[1])
            xgb_label   = 1 if fake_prob >= 0.5 else 0
            xgb_conf    = fake_prob if xgb_label == 1 else 1.0 - fake_prob
            result['xgboost'] = {
                'label':      LABEL_MAP[xgb_label],
                'confidence': round(xgb_conf, 4),
            }
        except Exception as exc:
            print(f'[XGBoost] prediction error: {exc}')

    # ── CNN-LSTM (mel-spectrogram features) ─────────────────────────────────
    if cnn_lstm_model is not None:
        try:
            spec       = _safe_spectrogram(audio_path)
            fake_prob  = float(cnn_lstm_model.predict(spec, verbose=0)[0, 0])
            cnn_label  = 1 if fake_prob >= 0.5 else 0
            cnn_conf   = fake_prob if cnn_label == 1 else 1.0 - fake_prob
            result['cnn_lstm'] = {
                'label':      LABEL_MAP[cnn_label],
                'confidence': round(cnn_conf, 4),
            }
        except Exception as exc:
            print(f'[CNN-LSTM] prediction error: {exc}')

    # ── Ensemble verdict ─────────────────────────────────────────────────────
    xgb_avail = result['xgboost']['label']  is not None
    cnn_avail = result['cnn_lstm']['label'] is not None

    if xgb_avail and cnn_avail:
        xgb_fake = result['xgboost']['label']  == 'fake'
        cnn_fake = result['cnn_lstm']['label'] == 'fake'
        if xgb_fake == cnn_fake:
            result['verdict'] = 'fake' if xgb_fake else 'genuine'
        else:
            # Models disagree – trust the one with higher confidence
            xgb_c = result['xgboost']['confidence']
            cnn_c = result['cnn_lstm']['confidence']
            if xgb_c >= cnn_c:
                result['verdict'] = result['xgboost']['label']
            else:
                result['verdict'] = result['cnn_lstm']['label']
            result['verdict'] += '_uncertain'   # flag the disagreement
    elif xgb_avail:
        result['verdict'] = result['xgboost']['label']
    elif cnn_avail:
        result['verdict'] = result['cnn_lstm']['label']

    return result

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    import csv
    meta_path = os.path.join(BASE_DIR, 'release_in_the_wild', 'meta.csv')
    genuine_samples = []
    forged_samples = []
    
    try:
        with open(meta_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            speakers_g = set()
            speakers_f = set()
            for row in reader:
                label_str = row['label'].strip().lower()
                speaker = row.get('speaker', 'Unknown')
                
                if label_str == 'bona-fide' and speaker not in speakers_g and len(genuine_samples) < 6:
                    genuine_samples.append({'file': row['file'], 'speaker': speaker})
                    speakers_g.add(speaker)
                elif label_str == 'spoof' and speaker not in speakers_f and len(forged_samples) < 6:
                    forged_samples.append({'file': row['file'], 'speaker': speaker})
                    speakers_f.add(speaker)
                
                if len(genuine_samples) >= 6 and len(forged_samples) >= 6:
                    break
    except Exception as e:
        print(f'Error loading meta.csv: {e}')

    return render_template('index.html', genuine_samples=genuine_samples, forged_samples=forged_samples)


def _handle_upload():
    """Shared logic for /predict and /record: save upload, run inference, clean up."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    os.makedirs('tmp', exist_ok=True)
    # Use a UUID-based name to avoid collisions with concurrent requests
    # (browser mic blobs often arrive with filename="blob" or no extension)
    safe_name = f"{uuid.uuid4().hex}.wav"
    temp_path = os.path.join('tmp', safe_name)

    try:
        file.save(temp_path)
        # Copy to uploads for later playback
        upload_dir = os.path.join('static', 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        uploaded_path = os.path.join(upload_dir, safe_name)
        # Copy file for serving (avoid deleting later)
        import shutil
        shutil.copy(temp_path, uploaded_path)
        result = predict_from_audio(temp_path)
        # Load stored metrics for both models if available
        try:
            with open(os.path.join(BASE_DIR, 'models', 'metrics_xgboost.json')) as f:
                result['xgboost']['metrics'] = json.load(f)
        except Exception:
            pass
        try:
            with open(os.path.join(BASE_DIR, 'models', 'metrics_cnn_lstm.json')) as f:
                result['cnn_lstm']['metrics'] = json.load(f)
        except Exception:
            pass
        # Add audio URL to result for client side playback
        result['audio_url'] = f'/static/uploads/{safe_name}'
        # Add audio URL to result for client side playback
        result['audio_url'] = f'/static/uploads/{safe_name}'
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return jsonify(result)


@app.route('/predict', methods=['POST'])
def predict():
    return _handle_upload()


@app.route('/record', methods=['POST'])
def record():
    return _handle_upload()


@app.route('/static/uploads/<filename>')
def serve_uploaded_file(filename):
    return send_from_directory(os.path.join('static', 'uploads'), filename)


@app.route('/dataset/<path:filename>')
def serve_dataset_file(filename):
    return send_from_directory(os.path.join(BASE_DIR, 'release_in_the_wild'), filename)


if __name__ == '__main__':
    app.run(debug=True)