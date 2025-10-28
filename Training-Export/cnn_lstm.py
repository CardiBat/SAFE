# -*- coding: utf-8 -*-
# cnn_lstm_training.py

import os
import random
import pandas as pd
import numpy as np

# Matplotlib headless (per ambienti senza display / Slurm)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score, median_absolute_error
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv1D, MaxPooling1D, Dropout, LSTM, Dense, Reshape
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# =========================
# GPU INFO (non blocca se non c'è)
# =========================
print("=== GPU Configuration ===")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'non impostato')}")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"GPU fisiche disponibili: {len(gpus)}")
        print(f"GPU logiche disponibili: {len(logical_gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"  GPU fisica {i}: {gpu}")
        print("✓ Training utilizzerà la GPU!")
    except RuntimeError as e:
        print(f"Errore configurazione GPU: {e}")
else:
    print("⚠ Nessuna GPU disponibile, uso CPU")
print()

# =========================
# Riproducibilità
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

# =========================
# Dati
# =========================
CSV_PATH = "/home/projects/safe/data/processed-datasets/processed_streaming_row_continuous.csv"
df_produzione = pd.read_csv(CSV_PATH)
print(f"Dataset caricato: {df_produzione.shape[0]} righe, {df_produzione.shape[1]} colonne")
print(df_produzione.head())

# Statistiche grezze (debug)
sample_numeric = df_produzione.sample(n=1000, random_state=SEED).select_dtypes(include=['number'])
print("Media (prime 5 col):\n", sample_numeric.mean().head())
print("Std (prime 5 col):\n", sample_numeric.std().head())

# Standardizzazione (fit su TUTTO per semplicità; per zero leakage: fit solo su train)
num_cols = df_produzione.select_dtypes(include=['number']).columns
scaler = StandardScaler()
df_scaled = df_produzione.copy()
df_scaled[num_cols] = scaler.fit_transform(df_produzione[num_cols])

sample_scaled = df_scaled.sample(n=1000, random_state=SEED).select_dtypes(include=['number'])
print("Scaled mean (prime 5 col):\n", sample_scaled[num_cols].mean().round(2).head())
print("Scaled std  (prime 5 col):\n", sample_scaled[num_cols].std().round(2).head())

# =========================
# Sequenze multistep
# =========================
def create_sequences_all_v2(data, window_size, forecast_length=1):
    X, y = [], []
    for i in range(len(data) - window_size - forecast_length + 1):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size : i+window_size+forecast_length])
    return np.array(X), np.array(y)

window_size = 30         # timesteps in input
forecast_length = 10     # timesteps previsti in output

data_numeric = df_scaled[num_cols].values.astype('float32')  # <-- GIUSTO: dati standardizzati
X, y = create_sequences_all_v2(data_numeric, window_size, forecast_length)
X = X.astype('float32')
y = y.astype('float32')

print("Forma X:", X.shape)  # (num_samples, window_size, num_sensori)
print("Forma y:", y.shape)  # (num_samples, forecast_length, num_sensori)

# Split time-aware (no shuffle)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)
num_sensori = X.shape[2]

# =========================
# Modello
# =========================
def create_cnn_lstm_model(window_size, n_feat, forecast_len, n_hidden=64):
    optimizer = Adam(learning_rate=0.001)
    inputs = Input(shape=(window_size, n_feat))

    x = Conv1D(64, 3, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)

    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)

    x = LSTM(n_hidden, return_sequences=False)(x)
    x = Dense(64, activation='relu')(x)

    outputs = Dense(forecast_len * n_feat)(x)
    outputs = Reshape((forecast_len, n_feat))(outputs)

    model = Model(inputs, outputs)
    model.compile(optimizer=optimizer, loss='mse')
    return model

name = "cnn_lstm"
print(f"\nAddestramento modello: {name}")
model = create_cnn_lstm_model(window_size, X.shape[2], forecast_length)

# =========================
# Callback + training “furbo”
# =========================
os.makedirs("logs", exist_ok=True)
callbacks = [
    ModelCheckpoint("logs/best.keras", monitor="val_loss", save_best_only=True),
    EarlyStopping(monitor="val_loss", patience=3, min_delta=1e-3, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1),
]

history = model.fit(
    X_train, y_train,
    epochs=100,                 # numero alto: si ferma da solo con EarlyStopping
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    shuffle=False,              # IMPORTANTISSIMO per serie temporali
    verbose=1
)

# =========================
# Valutazione + Predizione
# =========================
loss = model.evaluate(X_test, y_test)
print(f"\nLoss test = {loss:.4f}")

y_pred_cnn_lstm = model.predict(X_test)
print("y_pred shape:", y_pred_cnn_lstm.shape)

# =========================
# Salvataggi (H5 + SavedModel)
# =========================
model_file = f"cnn_lstm_model_{name}.h5"
model.save(model_file, include_optimizer=False)
print(f"✓ Modello H5 salvato in '{model_file}'")

os.makedirs("export_cnn_lstm", exist_ok=True)
model.save("export_cnn_lstm/savedmodel", include_optimizer=False)
print("✓ SavedModel salvato in 'export_cnn_lstm/savedmodel'")

# =========================
# Grafico training (file, non show)
# =========================
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
plt.title('CNN + LSTM Training')
plt.legend(); plt.grid(True)
plt.savefig("logs/cnn_lstm_training.png", dpi=150, bbox_inches="tight")
print("✓ Grafico training salvato in logs/cnn_lstm_training.png")

# =========================
# Report metriche per sensore
# =========================
def regression_report(y_true, y_pred, sensor_names=None):
    mse_list, rmse_list, mae_list, medae_list, mape_list, r2_list, evs_list = [], [], [], [], [], [], []
    num_sensori = y_true.shape[1]
    for i in range(num_sensori):
        mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        medae = median_absolute_error(y_true[:, i], y_pred[:, i])
        mape = np.mean(np.abs((y_true[:, i] - y_pred[:, i]) / (y_true[:, i] + 1e-8))) * 100
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        evs = explained_variance_score(y_true[:, i], y_pred[:, i])

        mse_list.append(mse); rmse_list.append(rmse); mae_list.append(mae)
        medae_list.append(medae); mape_list.append(mape)
        r2_list.append(r2); evs_list.append(evs)

    if sensor_names is None:
        sensor_names = [f"Sensor_{i+1}" for i in range(num_sensori)]

    report_df = pd.DataFrame({
        "Sensor": sensor_names,
        "MSE": mse_list, "RMSE": rmse_list, "MAE": mae_list, "MedAE": medae_list,
        "MAPE (%)": mape_list, "R2": r2_list, "ExplainedVar": evs_list
    })
    return report_df

num_sensori = y_test.shape[2]
y_test_flat = y_test.reshape(-1, num_sensori)
y_pred_flat = y_pred_cnn_lstm.reshape(-1, num_sensori)

print("y_test_flat:", y_test_flat.shape)
print("y_pred_flat:", y_pred_flat.shape)

report_cnn_lstm = regression_report(y_test_flat, y_pred_flat, sensor_names=list(num_cols))
print(report_cnn_lstm)

# =========================
# (UTILE per INT8) Representative dataset
# =========================
np.savez("export_cnn_lstm/rep_windows.npz", X=X_train[:1000].astype("float32"))
print("✓ rep_windows.npz salvato:", X_train[:1000].shape)

print("\nTutto fatto.")
