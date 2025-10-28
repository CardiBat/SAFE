# bench_tflite.py — benchmark fedele Keras vs TFLite INT8
import os, argparse, time
import numpy as np
import tensorflow as tf
from keras.layers import TFSMLayer

# Evita rumore GPU durante il bench (non serve per l'accuracy)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

def load_sample(rep_path, n=256):
    data = np.load(rep_path)
    X = data['X'] if 'X' in data.files else list(data.values())[0]
    if n is not None and len(X) > n:
        # prendo campioni distanziati per coprire range
        stride = max(1, len(X)//n)
        X = X[::stride][:n]
    print(f"Sample X: {X.shape} ({X.dtype})")
    return X.astype(np.float32, copy=False)

def run_keras(savedmodel_dir, X):
    """Inferenza dal SavedModel (Keras 3) via TFSMLayer."""
    layer = TFSMLayer(savedmodel_dir, call_endpoint='serving_default')
    t0 = time.time()
    y = layer(X, training=False)
    ms = (time.time() - t0) * 1000.0 / max(1, len(X))
    # normalizza a np.ndarray
    if isinstance(y, dict): y = next(iter(y.values()))
    elif isinstance(y, (list, tuple)): y = y[0]
    y = tf.convert_to_tensor(y).numpy()
    return y, ms

def run_tflite(tfl_path, X_float32, threads=1):
    """Inferenza TFLite con quant/dequant corretti per confronto con Keras."""
    interp = tf.lite.Interpreter(model_path=tfl_path, num_threads=threads)
    interp.allocate_tensors()
    i_det = interp.get_input_details()[0]
    o_det = interp.get_output_details()[0]

    in_scale, in_zp  = i_det['quantization']
    out_scale, out_zp = o_det['quantization']

    # Quantizza input
    Xq_all = np.round(X_float32 / in_scale + in_zp).astype(np.int8)
    Xq_all = np.clip(Xq_all, -128, 127)

    Yf_all = []
    t0 = time.time()
    for i in range(len(Xq_all)):
        Xq = Xq_all[i:i+1]  # batch=1
        interp.set_tensor(i_det['index'], Xq)
        interp.invoke()
        Yq = interp.get_tensor(o_det['index']).astype(np.int8)
        Yf = (Yq.astype(np.float32) - out_zp) * out_scale
        Yf_all.append(Yf)
    ms = (time.time() - t0) * 1000.0 / len(Xq_all)
    Yf_all = np.concatenate(Yf_all, axis=0)

    print("TFLite input quant params:", (in_scale, in_zp))
    print("TFLite output quant params:", (out_scale, out_zp))
    return Yf_all, ms


def metrics(y_ref, y_hat):
    diff = y_hat - y_ref
    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))
    maxe = float(np.max(np.abs(diff)))
    return mse, mae, maxe

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--savedmodel", required=True)
    ap.add_argument("--tflite", required=True)
    ap.add_argument("--rep", required=True)
    ap.add_argument("--n", type=int, default=256, help="numero di finestre da usare")
    ap.add_argument("--threads", type=int, default=1, help="TFLite num_threads")
    args = ap.parse_args()

    X = load_sample(args.rep, n=args.n)

    yk, ms_k = run_keras(args.savedmodel, X)
    print(f"Keras out: {yk.shape}, ~{ms_k:.3f} ms/finestre")

    yt, ms_t = run_tflite(args.tflite, X, threads=args.threads)
    print(f"TFLite out: {yt.shape}, ~{ms_t:.3f} ms/finestre")

    mse, mae, maxe = metrics(yk, yt)
    print(f"Δ TFLite vs Keras → MSE: {mse:.6e} | MAE: {mae:.6e} | MAX: {maxe:.6e}")
    print(f"Range output Keras:   [{yk.min():.3f}, {yk.max():.3f}]")
    print(f"Range output TFLite:  [{yt.min():.3f}, {yt.max():.3f}]")

if __name__ == "__main__":
    main()
