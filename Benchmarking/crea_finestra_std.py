#!/usr/bin/env python3
# crea_finestra_std.py — versione finale per processed_streaming_row_continuous.csv

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

CSV_PATH = "/home/projects/safe/data/processed-datasets/processed_streaming_row_continuous.csv"
OUT_PATH = "/home/projects/safe/src/lorenzo/confrontoKeras-tflite/win_std.csv"
WINDOW   = 30
NFEAT    = 16  # numero di feature del modello

# === CARICA CSV ===
# salta la colonna timestamp (prima)
df = pd.read_csv(CSV_PATH, header=0, sep=",", low_memory=False)
print(f"[INFO] Colonne totali trovate: {len(df.columns)}")

# prendi solo le colonne numeriche dalla 2° in poi
num_df = df.select_dtypes(include=[np.number])
print(f"[INFO] Colonne numeriche: {len(num_df.columns)}")

# converte in float32
data = num_df.iloc[:, :NFEAT].to_numpy(dtype=np.float32, copy=False)
print(f"[INFO] Dati caricati: {data.shape}")

if data.shape[0] < WINDOW:
    raise RuntimeError(f"Servono almeno {WINDOW} righe, trovate {data.shape[0]}")

# === STANDARDIZZA COME IN TRAINING ===
scaler = StandardScaler()
scaler.fit(data)          # stima media/dev std sull’intero dataset
win = data[:WINDOW]       # prendi le prime 30 righe
win_std = scaler.transform(win)

# === SALVA ===
np.savetxt(OUT_PATH, win_std, delimiter=",", fmt="%.6f")

print(f"[OK] Finestra standardizzata salvata in {OUT_PATH}")
print(f"Forma: {win_std.shape}")
print(f"Range: [{win_std.min():.3f}, {win_std.max():.3f}]")
