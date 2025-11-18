import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf

from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, GRU, Dropout
from keras.callbacks import EarlyStopping

import warnings
warnings.filterwarnings('ignore')

# === GENERAL DATA PREPARATION ===
tf.random.set_seed(1234)
np.random.seed(1234)

# Load the data (ajustado a nueva estructura)
file = os.path.join('data', 'contamination.csv')
df = pd.read_csv(file)

# Select the target
target = 'pm2.5'

df['DateTime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
df.set_index('DateTime', inplace=True)

# === DATA PREPROCESSING ===
# Remove unnecessary columns
df = df.drop(['year', 'month', 'day', 'hour', 'No'], axis=1, errors='ignore')

# Handle missing values
df = df.fillna(method='ffill').fillna(method='bfill')

# Select features for the model
feature_columns = [col for col in df.columns if col != target]
data = df[[target] + feature_columns].values

print(f"Shape de los datos: {data.shape}")
print(f"Columnas utilizadas: {[target] + feature_columns}")

# === NORMALIZE DATA ===
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# === CREATE SEQUENCES FOR GRU ===
def create_sequences(data, seq_length):
    """
    Crea secuencias de tiempo para el modelo GRU
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length, :])  # Todas las features
        y.append(data[i+seq_length, 0])     # Solo el target (pm2.5)
    return np.array(X), np.array(y)

# Longitud de la secuencia (ventana temporal)
seq_length = 6  # 6 horas previas para predecir la siguiente

X, y = create_sequences(data_scaled, seq_length)

print(f"\nShape de X: {X.shape}")  # (samples, seq_length, features)
print(f"Shape de y: {y.shape}")    # (samples,)

# === SPLIT DATA ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False, random_state=1234
)

print(f"\nTamaño del conjunto de entrenamiento: {X_train.shape[0]}")
print(f"Tamaño del conjunto de prueba: {X_test.shape[0]}")

# === BUILD GRU MODEL ===
model = Sequential([
    GRU(128, return_sequences=True, input_shape=(seq_length, X.shape[2])),
    Dropout(0.2),
    
    GRU(64, return_sequences=True),
    Dropout(0.2),
    
    GRU(32, return_sequences=False),
    Dropout(0.2),
    
    Dense(16, activation='relu'),
    Dense(1)
])

# Compile model
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

print("\n=== ARQUITECTURA DEL MODELO ===")
model.summary()

# === TRAIN MODEL ===
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

print("\n=== ENTRENANDO MODELO ===")
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# === EVALUATE MODEL ===
print("\n=== EVALUACIÓN DEL MODELO ===")
train_loss, train_mae = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)

print(f"Training Loss (MSE): {train_loss:.4f}")
print(f"Training MAE: {train_mae:.4f}")
print(f"Test Loss (MSE): {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")

# === PREDICTIONS ===
y_train_pred = model.predict(X_train, verbose=0)
y_test_pred = model.predict(X_test, verbose=0)

# Desnormalizar predicciones
y_train_actual = y_train * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]
y_train_pred_actual = y_train_pred.flatten() * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]

y_test_actual = y_test * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]
y_test_pred_actual = y_test_pred.flatten() * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]

# Métricas en escala original
train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_train_pred_actual))
test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred_actual))
train_r2 = r2_score(y_train_actual, y_train_pred_actual)
test_r2 = r2_score(y_test_actual, y_test_pred_actual)

print(f"\n=== MÉTRICAS EN ESCALA ORIGINAL ===")
print(f"Train RMSE: {train_rmse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Train R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")

# === VISUALIZATION ===

# ===== FIGURA 1: Métricas de Entrenamiento =====
fig1, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Training history - Loss
axes[0, 0].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0, 0].set_title('Pérdida durante el Entrenamiento', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Época')
axes[0, 0].set_ylabel('MSE')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Training history - MAE
axes[0, 1].plot(history.history['mae'], label='Train MAE', linewidth=2)
axes[0, 1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
axes[0, 1].set_title('MAE durante el Entrenamiento', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Época')
axes[0, 1].set_ylabel('MAE')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Predictions vs Actual (Test set)
sample_size = min(500, len(y_test_actual))
axes[1, 0].plot(y_test_actual[:sample_size], label='Real', alpha=0.7, linewidth=1.5)
axes[1, 0].plot(y_test_pred_actual[:sample_size], label='Predicción', alpha=0.7, linewidth=1.5)
axes[1, 0].set_title(f'Predicciones vs Valores Reales (Test)\nRMSE: {test_rmse:.2f}, R²: {test_r2:.3f}', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Muestra')
axes[1, 0].set_ylabel('PM2.5')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Scatter plot
axes[1, 1].scatter(y_test_actual, y_test_pred_actual, alpha=0.5, s=10)
axes[1, 1].plot([y_test_actual.min(), y_test_actual.max()], 
                [y_test_actual.min(), y_test_actual.max()], 
                'r--', lw=2, label='Predicción Perfecta')
axes[1, 1].set_title('Predicción vs Real (Test)', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Valor Real')
axes[1, 1].set_ylabel('Valor Predicho')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ===== FIGURA 2: Validation Loss por Época =====
fig2, ax = plt.subplots(figsize=(12, 6))
epochs_range = range(1, len(history.history['val_loss']) + 1)
ax.plot(epochs_range, history.history['val_loss'], 'o-', color='#FF6B35', linewidth=2, markersize=4, label='Validation Loss')
ax.set_title('Model Train vs Validation Loss for GRU', fontsize=14, fontweight='bold')
ax.set_xlabel('epoch', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ===== FIGURA 3: Datos Reales vs Predicción en Tiempo (días) =====
# Crear índice de tiempo para el conjunto de prueba
test_dates = df.index[seq_length + len(X_train):seq_length + len(X_train) + len(X_test)]
days_since_start = np.arange(len(y_test_actual)) / 24  # Convertir horas a días

fig3, ax = plt.subplots(figsize=(14, 7))
ax.plot(days_since_start, y_test_actual, label='Test data', color='#4A90E2', linewidth=1.5, alpha=0.8)
ax.plot(days_since_start, y_test_pred_actual, label='Prediction', color='#FF6B35', linewidth=1.5, alpha=0.8)
ax.set_title('Test data vs prediction for GRU', fontsize=14, fontweight='bold')
ax.set_xlabel('Time (day)', fontsize=12)
ax.set_ylabel('PM2.5 Concentration', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ===== FIGURA 4: Últimas 24 horas + Predicción Futura (6 horas) =====
# CORRECCIÓN: Usar los ÚLTIMOS datos completos (no solo del test)
# Tomar las últimas seq_length observaciones de TODOS los datos NORMALIZADOS
last_sequence_full = data_scaled[-seq_length:, :].reshape(1, seq_length, -1)

print(f"\n=== DEBUG: Última secuencia ===")
print(f"Shape: {last_sequence_full.shape}")
print(f"Último valor real (normalizado): {last_sequence_full[0, -1, 0]:.4f}")
print(f"Último valor real (original): {data[-1, 0]:.2f}")

# Generar predicciones futuras (próximas 6 horas)
future_predictions_scaled = []
current_sequence = last_sequence_full.copy()

for i in range(6):  # Predecir 6 horas
    # Hacer predicción (retorna valor NORMALIZADO)
    next_pred_scaled = model.predict(current_sequence, verbose=0)[0, 0]
    future_predictions_scaled.append(next_pred_scaled)
    
    # Actualizar la secuencia deslizando la ventana
    # Copiar la última fila de features completa
    new_row = current_sequence[0, -1, :].copy().reshape(1, 1, -1)
    # Actualizar SOLO el target (primera columna) con la predicción
    new_row[0, 0, 0] = next_pred_scaled
    
    # Deslizar ventana: quitar primera fila, agregar nueva al final
    current_sequence = np.concatenate([current_sequence[:, 1:, :], new_row], axis=1)
    
    print(f"Predicción hora +{i+1} (normalizada): {next_pred_scaled:.4f}")

# Desnormalizar predicciones futuras
future_predictions_scaled = np.array(future_predictions_scaled)
future_predictions = scaler.inverse_transform(
    np.column_stack([future_predictions_scaled, np.zeros((len(future_predictions_scaled), data.shape[1]-1))])
)[:, 0]

# Número de horas "pasadas" que quiero predecir alrededor del presente
n_past_pred = 6
last_idx = len(data_scaled) - 1  # índice del último dato real

# Índices objetivo: últimas 6 horas reales -> tiempos -6, -5, ..., -1
target_indices_past = range(last_idx - n_past_pred, last_idx)  # N-6 ... N-1

X_past = []
for i in target_indices_past:
    # para predecir el valor en i, uso las seq_length horas anteriores
    start = i - seq_length
    end = i
    X_past.append(data_scaled[start:end, :])

X_past = np.array(X_past)

# Predicciones normalizadas de esas 6 horas pasadas
past_preds_scaled = model.predict(X_past, verbose=0).flatten()

# Desnormalizar
past_predictions = scaler.inverse_transform(
    np.column_stack([
        past_preds_scaled,
        np.zeros((len(past_preds_scaled), data.shape[1] - 1))
    ])
)[:, 0]

# Tomar las últimas 24 horas de datos REALES (sin normalizar)
last_24_hours_real = data[-24:, 0]

# Crear eje temporal
time_history     = np.arange(-24, 0)  # igual que antes: -24 ... -1
time_past_pred   = np.arange(-6, 0)   # -6 ... -1  (6 puntos)
time_future_pred = np.arange(0, 6)    # 0 ... 5   (6 puntos)

time_pred_all = np.concatenate([time_past_pred, time_future_pred])
pred_all      = np.concatenate([past_predictions, future_predictions])

fig4, ax = plt.subplots(figsize=(14, 7))
ax.plot(time_history, last_24_hours_real, label='History (Last 24h)', color='#4A90E2', linewidth=2.5, marker='o', markersize=4)
ax.plot(time_pred_all, pred_all, label='Forecasted with GRU (-6h to +6h)', color='#FF6B35', linewidth=2.5, marker='s', markersize=6)
#ax.plot(time_future, future_predictions, label='Forecasted with GRU (Next 6h)', color='#FF6B35', linewidth=2.5, marker='s', markersize=6)
ax.axvline(x=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Prediction Start')
ax.set_title('PM2.5 Forecast: Last 24h History + Next 6h Prediction', fontsize=14, fontweight='bold')
ax.set_xlabel('Time step (hours)', fontsize=12)
ax.set_ylabel('PM2.5 Concentration', fontsize=12)
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\n==== PREDICCIONES FUTURAS (Próximas 6 horas) ====")
for i, pred in enumerate(future_predictions, 1):
    print(f"Hora +{i}: {pred:.2f} PM2.5")

# ===== FIGURA 5: Cono de Incertidumbre (6 horas) =====
# Generar múltiples predicciones con dropout activado para estimar incertidumbre
n_iterations = 100
future_predictions_mc = []

for iteration in range(n_iterations):
    current_sequence_mc = last_sequence_full.copy()
    predictions_iter = []
    
    for i in range(6):  # 6 horas
        # Predicción con dropout activo (incertidumbre de Monte Carlo)
        next_pred_scaled = model(current_sequence_mc, training=True).numpy()[0, 0]
        predictions_iter.append(next_pred_scaled)
        
        # Actualizar secuencia
        new_row = current_sequence_mc[0, -1, :].copy().reshape(1, 1, -1)
        new_row[0, 0, 0] = next_pred_scaled
        current_sequence_mc = np.concatenate([current_sequence_mc[:, 1:, :], new_row], axis=1)
    
    future_predictions_mc.append(predictions_iter)

future_predictions_mc = np.array(future_predictions_mc)

# Desnormalizar todas las iteraciones
future_predictions_mc_denorm = []
for iteration_preds in future_predictions_mc:
    denorm = scaler.inverse_transform(
        np.column_stack([iteration_preds, np.zeros((len(iteration_preds), data.shape[1]-1))])
    )[:, 0]
    future_predictions_mc_denorm.append(denorm)

future_predictions_mc_denorm = np.array(future_predictions_mc_denorm)

# Calcular media y percentiles
mean_prediction = np.mean(future_predictions_mc_denorm, axis=0)
lower_bound = np.percentile(future_predictions_mc_denorm, 5, axis=0)
upper_bound = np.percentile(future_predictions_mc_denorm, 95, axis=0)

fig5, ax = plt.subplots(figsize=(14, 7))

# 1) Historia de las últimas 24 h (igual que antes)
ax.plot(time_history, last_24_hours_real, label='History (Last 24h)', color='#4A90E2', linewidth=2.5, marker='o', markersize=4)

# 2) Línea naranja de predicción completa: -6h a +6h (6 pasadas + 6 futuras)
ax.plot(time_pred_all, pred_all, label='Forecasted with GRU (-6h to +6h)', color='#FF6B35', linewidth=2.0, linestyle='--')

# 3) Cono de incertidumbre solo en las próximas 6 h (0…+5),
# usando la media de Monte Carlo
ax.plot(time_future_pred, mean_prediction, label='Forecast mean (Next 6h)', color='#FF6B35', linewidth=2.5, marker='s', markersize=6)
ax.fill_between(time_future_pred, lower_bound, upper_bound, alpha=0.3, color='#FF6B35', label='Uncertainty (90% CI)')

# 4) Línea vertical en "ahora"
ax.axvline(x=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Prediction Start')
ax.set_title('PM2.5 Forecast with Uncertainty Cone (Last 24h, -6h to +6h)', fontsize=14, fontweight='bold')
ax.set_xlabel('Time step (hours)', fontsize=12)
ax.set_ylabel('PM2.5 Concentration', fontsize=12)
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


"""
fig5, ax = plt.subplots(figsize=(14, 7))
ax.plot(time_history, last_24_hours_real, label='History (Last 24h)', color='#4A90E2', linewidth=2.5, marker='o', markersize=4)
ax.plot(time_future, mean_prediction, label='Forecasted with GRU (Next 6h)', color='#FF6B35', linewidth=2.5, marker='s', markersize=6)
ax.fill_between(time_future, lower_bound, upper_bound, alpha=0.3, color='#FF6B35', label='Uncertainty (90% CI)')
ax.axvline(x=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Prediction Start')
ax.set_title('PM2.5 Forecast with Uncertainty Cone (Last 24h + Next 6h)', fontsize=14, fontweight='bold')
ax.set_xlabel('Time step (hours)', fontsize=12)
ax.set_ylabel('PM2.5 Concentration', fontsize=12)
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""

print(f"\n==== PREDICCIONES CON INCERTIDUMBRE ====")
for i in range(6):
    print(f"Hora +{i+1}: {mean_prediction[i]:.2f} PM2.5 (IC 90%: [{lower_bound[i]:.2f}, {upper_bound[i]:.2f}])")


# === SAVE MODEL (ajustado a carpeta models) ===
models_dir = 'models'
os.makedirs(models_dir, exist_ok=True)
model_h5_path = os.path.join(models_dir, 'gru_pm25_model.h5')
model_keras_path = os.path.join(models_dir, 'gru_pm25_model.keras')
scaler_path = os.path.join(models_dir, 'scaler.pkl')

model.save(model_h5_path)
model.save(model_keras_path)
print(f"\n✓ Modelo guardado como '{model_h5_path}' y '{model_keras_path}'")

# Guardar el scaler para uso futuro
import pickle
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"✓ Scaler guardado como '{scaler_path}'")