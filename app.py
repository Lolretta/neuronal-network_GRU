import os
import io
import base64
import zipfile
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# === Configuración básica ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

MODEL_PATH_KERAS = os.path.join(MODELS_DIR, 'gru_pm25_model.keras')
MODEL_PATH_H5 = os.path.join(MODELS_DIR, 'gru_pm25_model.h5')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')
CSV_PATH = os.path.join(DATA_DIR, 'contamination.csv')

SEQ_LENGTH = 6  # usado en el entrenamiento
TARGET_COL = 'pm2.5'

app = Flask(__name__, static_folder='static', template_folder='templates')

# Cache de artefactos
_MODEL_CACHE = None
_SCALER_CACHE = None


def _load_model_and_scaler(data_for_fallback: np.ndarray | None = None):
    global _MODEL_CACHE, _SCALER_CACHE

    if _MODEL_CACHE is not None and _SCALER_CACHE is not None:
        return _MODEL_CACHE, _SCALER_CACHE

    # Cargar scaler
    if not os.path.exists(SCALER_PATH):
        # Fallback: si no existe el pickle, intentamos recalcular desde datos
        if data_for_fallback is None:
            raise FileNotFoundError(f'Scaler no encontrado en {SCALER_PATH}')
        scaler = MinMaxScaler()
        scaler.fit(data_for_fallback)
    else:
        try:
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
        except Exception:
            # Fallback: pickle corrupto o con formato inválido
            if data_for_fallback is None:
                raise ValueError('El archivo scaler.pkl está corrupto y no se pudo reconstruir sin datos.')
            scaler = MinMaxScaler()
            scaler.fit(data_for_fallback)

    # Intentar cargar formato .keras; si falla, usar .h5
    model = None
    keras_exists = os.path.exists(MODEL_PATH_KERAS)
    h5_exists = os.path.exists(MODEL_PATH_H5)

    if keras_exists:
        try:
            model = tf.keras.models.load_model(MODEL_PATH_KERAS)
        except (zipfile.BadZipFile, Exception) as e:
            # Log y fallback a .h5 si existe
            if h5_exists:
                model = tf.keras.models.load_model(MODEL_PATH_H5)
            else:
                raise RuntimeError(
                    f'No se pudo cargar el modelo .keras ({e}). Además no existe .h5 para fallback.'
                )
    elif h5_exists:
        model = tf.keras.models.load_model(MODEL_PATH_H5)
    else:
        raise FileNotFoundError('No se encontraron archivos de modelo (.keras ni .h5) en models/.')

    _MODEL_CACHE = model
    _SCALER_CACHE = scaler
    return model, scaler


def _load_and_prepare_data(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'CSV no encontrado en {csv_path}')
    df = pd.read_csv(csv_path)

    # Validaciones mínimas
    required_cols = {'year', 'month', 'day', 'hour', TARGET_COL}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f'Faltan columnas requeridas en CSV: {sorted(list(missing))}')

    # Igual que en entrenamiento
    df['DateTime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df.set_index('DateTime', inplace=True)
    df = df.drop(['year', 'month', 'day', 'hour', 'No'], axis=1, errors='ignore')
    df = df.fillna(method='ffill').fillna(method='bfill')

    feature_columns = [col for col in df.columns if col != TARGET_COL]
    data = df[[TARGET_COL] + feature_columns].values
    return data, feature_columns


def _predict_future_hours(model, scaler, data: np.ndarray, hours: int):
    if data.shape[0] < SEQ_LENGTH:
        raise ValueError('No hay suficientes filas en el CSV para construir la secuencia inicial.')

    # Escalar usando el scaler pre-entrenado
    data_scaled = scaler.transform(data)

    # Última ventana
    current_sequence = data_scaled[-SEQ_LENGTH:, :].reshape(1, SEQ_LENGTH, -1)

    future_predictions_scaled = []
    for _ in range(hours):
        next_pred_scaled = model.predict(current_sequence, verbose=0)[0, 0]
        future_predictions_scaled.append(next_pred_scaled)

        # Crear nueva fila con las mismas features excepto el target, que se actualiza
        new_row = current_sequence[0, -1, :].copy().reshape(1, 1, -1)
        new_row[0, 0, 0] = next_pred_scaled
        current_sequence = np.concatenate([current_sequence[:, 1:, :], new_row], axis=1)

    future_predictions_scaled = np.array(future_predictions_scaled)
    # Desescalar solo el target
    future_predictions = scaler.inverse_transform(
        np.column_stack([
            future_predictions_scaled,
            np.zeros((len(future_predictions_scaled), data.shape[1] - 1))
        ])
    )[:, 0]
    return future_predictions


def _build_chart_base64(data: np.ndarray, predictions: np.ndarray):
    # Historia: últimas 24 horas (si hay menos, usar lo disponible)
    hist_len = min(24, data.shape[0])
    last_24_real = data[-hist_len:, 0]

    time_history = np.arange(-hist_len, 0)
    time_future = np.arange(0, len(predictions))

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(time_history, last_24_real, label='History (Last 24h)', color='#4A90E2', linewidth=2.0)
    ax.plot(time_future, predictions, label=f'Forecast (Next {len(predictions)}h)', color='#FF6B35', linewidth=2.0)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1.0, alpha=0.7, label='Prediction Start')
    ax.set_title('PM2.5 Forecast: Last 24h + Next hours', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time step (hours)', fontsize=12)
    ax.set_ylabel('PM2.5 Concentration', fontsize=12)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return b64


def _build_uncertainty_cone_base64(data: np.ndarray, model: tf.keras.Model, scaler: MinMaxScaler,
                                   hours: int, n_iterations: int = 50,
                                   ci_low: float = 5.0, ci_high: float = 95.0):
    """Genera gráfico con cono de incertidumbre usando Monte Carlo Dropout.

    - Ejecuta múltiples iteraciones con `training=True` para activar dropout.
    - Calcula media y percentiles para las próximas `hours`.
    - Devuelve imagen en base64.
    """
    # Historia y eje temporal
    hist_len = min(24, data.shape[0])
    last_24_real = data[-hist_len:, 0]
    time_history = np.arange(-hist_len, 0)
    time_future = np.arange(0, hours)

    # Secuencia inicial normalizada
    data_scaled = scaler.transform(data)
    last_seq = data_scaled[-SEQ_LENGTH:, :].reshape(1, SEQ_LENGTH, -1)

    # Monte Carlo sobre el horizonte
    mc_preds_scaled = []  # shape: (n_iterations, hours)
    for _ in range(n_iterations):
        seq_mc = last_seq.copy()
        preds_iter = []
        for _h in range(hours):
            # llamada con training=True para activar dropout
            next_pred_scaled = model(seq_mc, training=True).numpy()[0, 0]
            preds_iter.append(next_pred_scaled)
            new_row = seq_mc[0, -1, :].copy().reshape(1, 1, -1)
            new_row[0, 0, 0] = next_pred_scaled
            seq_mc = np.concatenate([seq_mc[:, 1:, :], new_row], axis=1)
        mc_preds_scaled.append(preds_iter)

    mc_preds_scaled = np.array(mc_preds_scaled)  # (n_iter, hours)

    # Desescalar todas las iteraciones a escala original del target
    mc_preds_denorm = []
    for iter_preds in mc_preds_scaled:
        denorm = scaler.inverse_transform(
            np.column_stack([
                iter_preds,
                np.zeros((len(iter_preds), data.shape[1] - 1))
            ])
        )[:, 0]
        mc_preds_denorm.append(denorm)
    mc_preds_denorm = np.array(mc_preds_denorm)  # (n_iter, hours)

    mean_prediction = np.mean(mc_preds_denorm, axis=0)
    lower_bound = np.percentile(mc_preds_denorm, ci_low, axis=0)
    upper_bound = np.percentile(mc_preds_denorm, ci_high, axis=0)

    # Graficar cono
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(time_history, last_24_real, label='History (Last 24h)', color='#4A90E2', linewidth=2.0)
    ax.plot(time_future, mean_prediction, label=f'Forecast mean (Next {hours}h)', color='#FF6B35', linewidth=2.0)
    ax.fill_between(time_future, lower_bound, upper_bound, alpha=0.3, color='#FF6B35', label='Uncertainty (90% CI)')
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1.0, alpha=0.7, label='Prediction Start')
    ax.set_title('PM2.5 Forecast with Uncertainty Cone', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time step (hours)', fontsize=12)
    ax.set_ylabel('PM2.5 Concentration', fontsize=12)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return b64, mean_prediction.tolist(), lower_bound.tolist(), upper_bound.tolist()


# === Rutas ===
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        payload = request.get_json(force=True)
        hours = payload.get('hours')

        # Validaciones
        if hours is None:
            return jsonify({'error': 'Parámetro "hours" es requerido.'}), 400
        try:
            hours = int(hours)
        except Exception:
            return jsonify({'error': 'Parámetro "hours" debe ser entero.'}), 400
        if hours < 1 or hours > 24:
            return jsonify({'error': 'El rango permitido de horas es 1-24.'}), 400

        # Cargar datos primero para permitir fallback del scaler si está corrupto
        data, feature_columns = _load_and_prepare_data(CSV_PATH)
        # Cargar artefactos (cacheados) usando datos para reconstrucción del scaler si hiciera falta
        model, scaler = _load_model_and_scaler(data_for_fallback=data)

        preds = _predict_future_hours(model, scaler, data, hours)
        chart_b64 = _build_chart_base64(data, preds)
        cone_b64, cone_mean, cone_low, cone_high = _build_uncertainty_cone_base64(
            data, model, scaler, hours, n_iterations=50, ci_low=5.0, ci_high=95.0
        )

        # Tabla con dos decimales
        table = [{'hour': i + 1, 'value': round(float(v), 2)} for i, v in enumerate(preds)]

        return jsonify({
            'hours': hours,
            'predictions': [round(float(v), 2) for v in preds],
            'chart_base64': chart_b64,
            'cone_chart_base64': cone_b64,
            'cone_mean': [round(float(v), 2) for v in cone_mean],
            'cone_low': [round(float(v), 2) for v in cone_low],
            'cone_high': [round(float(v), 2) for v in cone_high],
            'columns': [TARGET_COL] + feature_columns,
            'table': table
        })
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 500
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        # Fallback genérico
        return jsonify({'error': f'Error inesperado: {str(e)}'}), 500


if __name__ == '__main__':
    # Ejecutar servidor
    app.run(host='0.0.0.0', port=5000, debug=True)