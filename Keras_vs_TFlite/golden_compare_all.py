#!/usr/bin/env python3
# golden_compare_all.py
# Pipeline unica e riproducibile:
# - genera una finestra 30x16 nella scala giusta (z-score) -> win_std.csv
# - confronta Keras (float32) vs TFLite (int8->float) sugli stessi input
# - opzionale: confronta anche STM32 leggendo outINT8.txt
#
# USO TIPICO (da terminale):
# 1) Da CSV "grande" (salta timestamp), usa le prime 30 righe:
#    python3 golden_compare_all.py \
#      --savedmodel ../export_cnn_lstm/savedmodel_unroll \
#      --tflite ../finale/qbench_sweep_20251027_165731/experiments/mt2000_b0p5/tflite/model_int8_full.tflite \
#      --csv /home/projects/safe/data/processed-datasets/processed_streaming_row_continuous.csv \
#      --start 0
#
# 2) Da finestra RAW 30x16 del sensore (non standardizzata) + CSV di calibrazione:
#    python3 golden_compare_all.py \
#      --savedmodel ../export_cnn_lstm/savedmodel_unroll \
#      --tflite ../finale/.../mt2000_b0p5/tflite/model_int8_full.tflite \
#      --raw raw_window.csv \
#      --calib-csv /home/projects/safe/data/processed-datasets/processed_streaming_row_continuous.csv
#
# 3) Confronto anche STM32 (dopo aver inviato win_std.csv e salvato l'output MCU in outINT8.txt):
#    ... (uno dei comandi sopra) --stm32 outINT8.txt



# uso

# genera:
# python3 golden_compare_all.py \
#   --savedmodel ../export_cnn_lstm/savedmodel_unroll \
#   --tflite ../finale/qbench_sweep_20251027_165731/experiments/mt2000_b0p5/tflite/model_int8_full.tflite \
#   --csv /home/projects/safe/data/processed-datasets/processed_streaming_row_continuous.csv \
#   --start 0

# confronta tutto
# python3 golden_compare_all.py \
#   --savedmodel ../export_cnn_lstm/savedmodel_unroll \
#   --tflite ../finale/qbench_sweep_20251027_165731/experiments/mt2000_b0p5/tflite/model_int8_full.tflite \
#   --csv /home/projects/safe/data/processed-datasets/processed_streaming_row_continuous.csv \
#   --start 0 \
#   --stm32 outINT8.txt



#
# Nota: lo script salva SEMPRE la finestra standardizzata in win_std.csv
#       (30 righe x 16 float) = quello che DEVI inviare allo STM32.

import os, re, csv, json, time, argparse
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import TFSMLayer
from sklearn.preprocessing import StandardScaler

np.set_printoptions(suppress=True, linewidth=160)

DEF_WINDOW = 30
DEF_NFEAT  = 16
DEF_OUT_WIN = "win_std.csv"

# ------------------------- IO sorgenti finestra -------------------------

def autodetect_sep(path: str) -> str:
    with open(path, "r", newline="") as f:
        sample = f.read(4096)
    return csv.Sniffer().sniff(sample).delimiter

def load_window_from_csv(csv_path: str, start: int, window: int, nfeat: int) -> np.ndarray:
    """Legge il CSV 'grande', salta la colonna timestamp e prende 'window' righe x 'nfeat' colonne numeriche."""
    df = pd.read_csv(csv_path, header=0, sep=",", low_memory=False)
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < nfeat:
        raise RuntimeError(f"CSV ha solo {num_df.shape[1]} colonne numeriche, servono {nfeat}.")
    data = num_df.iloc[:, :nfeat].to_numpy(dtype=np.float32, copy=False)
    if start < 0 or start + window > data.shape[0]:
        raise RuntimeError(f"Intervallo ({start}:{start+window}) fuori dal range (0:{data.shape[0]}).")
    return data[start:start+window, :]  # [30,16] raw

def load_window_from_raw(raw_path: str, nfeat: int, window: int) -> np.ndarray:
    """Legge un file 30x16 privo di header (CSV a virgole o whitespace)."""
    try:
        # prova CSV a virgole
        A = pd.read_csv(raw_path, header=None).to_numpy(dtype=np.float32)
    except Exception:
        # fallback: spazi
        A = np.loadtxt(raw_path, dtype=np.float32)
    if A.ndim != 2:
        raise RuntimeError(f"RAW '{raw_path}' shape inattesa: {A.shape}")
    if A.shape[0] < window:
        raise RuntimeError(f"RAW ha {A.shape[0]} righe, servono {window}.")
    if A.shape[1] < nfeat:
        raise RuntimeError(f"RAW ha {A.shape[1]} colonne, servono {nfeat}.")
    return A[:window, :nfeat]

# --------------------------- Standardizzazione ---------------------------

def fit_scaler_from_csv(csv_path: str, nfeat: int) -> StandardScaler:
    df = pd.read_csv(csv_path, header=0, sep=",", low_memory=False)
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < nfeat:
        raise RuntimeError(f"Calib-CSV ha solo {num_df.shape[1]} colonne numeriche, servono {nfeat}.")
    data = num_df.iloc[:, :nfeat].to_numpy(dtype=np.float32, copy=False)
    sc = StandardScaler().fit(data)
    return sc

def standardize(win_raw: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(win_raw.astype(np.float32, copy=False)).astype(np.float32, copy=False)

# ------------------------------ Modelli ---------------------------------

def run_keras(savedmodel_dir: str, X: np.ndarray) -> Tuple[np.ndarray, float]:
    layer = TFSMLayer(savedmodel_dir, call_endpoint='serving_default')
    t0 = time.time()
    y = layer(X, training=False)
    ms = (time.time() - t0) * 1000.0 / max(1, len(X))
    if isinstance(y, dict): y = next(iter(y.values()))
    if isinstance(y, (list, tuple)): y = y[0]
    y = tf.convert_to_tensor(y).numpy().astype(np.float32, copy=False)
    return y, ms

def get_tfl_qparams(tfl_path: str) -> Tuple[Tuple[float,int], Tuple[float,int]]:
    interp = tf.lite.Interpreter(model_path=tfl_path)
    interp.allocate_tensors()
    i = interp.get_input_details()[0]
    o = interp.get_output_details()[0]
    in_scale, in_zp  = i['quantization']
    out_scale, out_zp = o['quantization']
    if in_scale is None or out_scale is None:
        raise RuntimeError("Modello TFLite privo di quantization info.")
    return (float(in_scale), int(in_zp)), (float(out_scale), int(out_zp))

def run_tflite_quantized(tfl_path: str, Xf32: np.ndarray, threads: int=1) -> Tuple[np.ndarray, float]:
    """Quantizza l'input con (scale,zp) e dequantizza l'output con (scale,zp)."""
    interp = tf.lite.Interpreter(model_path=tfl_path, num_threads=threads)
    interp.allocate_tensors()
    i = interp.get_input_details()[0]
    o = interp.get_output_details()[0]
    in_scale, in_zp  = i['quantization']
    out_scale, out_zp = o['quantization']

    Xq = np.round(Xf32 / in_scale + in_zp).astype(np.int8)
    Xq = np.clip(Xq, -128, 127)

    outs = []
    t0 = time.time()
    for b in range(Xq.shape[0]):
        interp.set_tensor(i['index'], Xq[b:b+1])
        interp.invoke()
        Yq = interp.get_tensor(o['index']).astype(np.int8)
        Yf = (Yq.astype(np.float32) - out_zp) * out_scale
        outs.append(Yf)
    ms = (time.time() - t0) * 1000.0 / max(1, Xq.shape[0])
    Y = np.concatenate(outs, axis=0).astype(np.float32)
    return Y, ms

# ------------------------------ Utility --------------------------------

def flatten_last(y: np.ndarray) -> np.ndarray:
    y = np.array(y)
    if y.ndim == 3:  # [B,10,16] -> [B,160]
        b,t,f = y.shape
        return y.reshape(b, t*f)
    return y

def metrics(y_ref: np.ndarray, y_hat: np.ndarray) -> Tuple[float,float,float]:
    diff = y_hat - y_ref
    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))
    maxe = float(np.max(np.abs(diff)))
    return mse, mae, maxe

def brief_rng(a: np.ndarray) -> str:
    return f"[{np.min(a):.3f}, {np.max(a):.3f}]"

def read_stm32_file(path: str) -> np.ndarray:
    txt = Path(path).read_text(encoding="utf-8", errors="ignore")
    vals = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", txt)
    if not vals:
        raise RuntimeError(f"Nessun numero trovato in {path}")
    arr = np.array([float(x) for x in vals], dtype=np.float32)
    return arr

# ------------------------------- CLI -----------------------------------

def main():
    ap = argparse.ArgumentParser(description="Pipeline unica per generare finestra z-score, confrontare Keras/TFLite e (opzionale) STM32.")
    ap.add_argument("--savedmodel", required=True)
    ap.add_argument("--tflite", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="CSV grande (con timestamp in colonna 0). Usa --start per la prima riga della finestra.")
    src.add_argument("--raw", help="CSV 30x16 dal sensore (NON standardizzato).")
    src.add_argument("--rep", help="NPZ rappresentativo già standardizzato (salta lo standardize).")

    ap.add_argument("--calib-csv", help="CSV per stimare StandardScaler quando usi --raw.")
    ap.add_argument("--start", type=int, default=0, help="riga iniziale della finestra per --csv (default 0)")
    ap.add_argument("--window", type=int, default=DEF_WINDOW)
    ap.add_argument("--nfeat", type=int, default=DEF_NFEAT)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--stm32", help="outINT8.txt stampato dallo STM32")
    ap.add_argument("--out-win", default=DEF_OUT_WIN, help=f"file finestra standardizzata da salvare (default {DEF_OUT_WIN})")

    args = ap.parse_args()

    # 1) Costruisci finestra raw 30x16
    if args.rep:
        npz = np.load(args.rep)
        X = npz['X'] if 'X' in npz.files else list(npz.values())[0]
        if X.shape[1:] != (args.window, args.nfeat):
            raise RuntimeError(f"REP shape {X.shape}, atteso (*,{args.window},{args.nfeat})")
        win_std = X[0].astype(np.float32)  # già standardizzato
        source_desc = f"REP:{args.rep}"
        used_scaler = "rep (pre-standardized)"
    elif args.csv:
        win_raw = load_window_from_csv(args.csv, args.start, args.window, args.nfeat)
        # Scaler stimato sull'intero CSV -> stessa scala del training (se CSV è quello usato)
        scaler = fit_scaler_from_csv(args.csv, args.nfeat)
        win_std = standardize(win_raw, scaler)
        source_desc = f"CSV:{args.csv}@{args.start}"
        used_scaler = f"fit({args.csv})"
    else:  # args.raw
        win_raw = load_window_from_raw(args.raw, args.nfeat, args.window)
        if not args.calib_csv:
            raise RuntimeError("Con --raw devi indicare --calib-csv per stimare lo StandardScaler.")
        scaler = fit_scaler_from_csv(args.calib_csv, args.nfeat)
        win_std = standardize(win_raw, scaler)
        source_desc = f"RAW:{args.raw}"
        used_scaler = f"fit({args.calib_csv})"

    # Salva la finestra standardizzata che DEVI inviare allo STM32
    np.savetxt(args.out_win, win_std, delimiter=",", fmt="%.6f")
    print(f"[OK] win_std salvata in {args.out_win}  shape={win_std.shape}  range={brief_rng(win_std)}")
    print(f"[INFO] sorgente={source_desc}  scaler={used_scaler}")

    # 2) Prepara batch [1,30,16] per inferenza PC
    Xf = win_std.reshape(1, args.window, args.nfeat).astype(np.float32)

    # 3) Esegui Keras (float)
    yk, ms_k = run_keras(args.savedmodel, Xf)
    yk = flatten_last(yk)
    print(f"[KERAS] out: {yk.shape}  range={brief_rng(yk)}  ~{ms_k:.3f} ms/finestra")

    # 4) Esegui TFLite (int8 -> dequant)
    (in_scale, in_zp), (out_scale, out_zp) = get_tfl_qparams(args.tflite)
    print(f"[TFLITE] qparams: IN(scale={in_scale:.9f}, zp={in_zp})  OUT(scale={out_scale:.9f}, zp={out_zp})")

    yt, ms_t = run_tflite_quantized(args.tflite, Xf, threads=args.threads)
    yt = flatten_last(yt)
    print(f"[TFLITE] out: {yt.shape}  range={brief_rng(yt)}  ~{ms_t:.3f} ms/finestra")

    # 5) Delta TFLite vs Keras (questo è il tuo "errore di quantizzazione" target)
    mse_t, mae_t, maxe_t = metrics(yk, yt)
    print(f"\nΔ TFLite vs Keras  →  MSE={mse_t:.6e}  MAE={mae_t:.6e}  MAX={maxe_t:.6e}")

    # 6) (opzionale) STM32
    if args.stm32:
        arr = read_stm32_file(args.stm32)
        if arr.size < 160:
            raise RuntimeError(f"STM32 file ha solo {arr.size} valori, servono almeno 160.")
        ystm = arr[:160].reshape(1,160).astype(np.float32)  # già dequantizzati se usi (q8 - zp)*scale in firmware
        print(f"[STM32] out: {ystm.shape}  range={brief_rng(ystm)}")

        mse_s, mae_s, maxe_s = metrics(yk, ystm)
        print(f"Δ STM32 vs Keras   →  MSE={mse_s:.6e}  MAE={mae_s:.6e}  MAX={maxe_s:.6e}")

    # 7) riepilogo JSON utile per parsing automatico
    summary = {
        "source": source_desc,
        "scaler": used_scaler,
        "keras_ms": ms_k,
        "tflite_ms": ms_t,
        "qparams": {"in_scale": in_scale, "in_zp": in_zp, "out_scale": out_scale, "out_zp": out_zp},
        "delta_tflite": {"mse": mse_t, "mae": mae_t, "max": maxe_t},
    }
    if args.stm32:
        summary["delta_stm32"] = {"mse": mse_s, "mae": mae_s, "max": maxe_s}
    print("\nJSON_SUMMARY:", json.dumps(summary, ensure_ascii=False))

if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # forza CPU se non hai GPU
    main()
