
#!/usr/bin/env python3

# python3 batch_golden_summary.py \
#   --results /home/projects/safe/src/lorenzo/finale/qbench_sweep_20251027_165731/results.csv \
#   --out /home/projects/safe/src/lorenzo/ultima-colonna/results_golden_aug.csv \
#   --savedmodel /home/projects/safe/src/lorenzo/export_cnn_lstm/savedmodel_unroll \
#   --base-dir /home/projects/safe/src/lorenzo/finale \
#   --n-windows 64

#!/usr/bin/env python3
import argparse, math, sys, time
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import pandas as pd


# === Risoluzione percorsi relativi ===
def resolve_path(pstr, base_dir: Path) -> Optional[Path]:
    """Rende assoluto un path relativo basandosi su base_dir"""
    if pstr is None or (isinstance(pstr, float) and np.isnan(pstr)):
        return None
    p = Path(str(pstr))
    if p.is_absolute():
        return p
    cand = base_dir / p
    if cand.exists():
        return cand
    return p


# === TensorFlow helpers ===
def try_import_tf():
    try:
        import tensorflow as tf
        return tf
    except Exception as e:
        print(f"[WARN] TensorFlow non disponibile: {e}")
        return None


def brief_rng(a: np.ndarray) -> Tuple[float, float]:
    return float(np.min(a)), float(np.max(a))


def quant_extreme_counts(yf: np.ndarray, out_scale: float, out_zp: int) -> Tuple[int, int]:
    q = np.round(yf / out_scale + out_zp).astype(np.int32)
    return int(np.sum(q <= -128)), int(np.sum(q >= 127))


def run_tflite(tf, tfl_path: str, Xf: np.ndarray, threads: int = 1):
    interp = tf.lite.Interpreter(model_path=tfl_path, num_threads=threads)
    interp.allocate_tensors()
    i_det = interp.get_input_details()[0]
    o_det = interp.get_output_details()[0]
    in_scale, in_zp = i_det.get('quantization')
    out_scale, out_zp = o_det.get('quantization')
    t0 = time.time()
    Y_all = []
    for i in range(len(Xf)):
        x = Xf[i:i + 1].astype(np.float32)
        xq = np.round(x / in_scale + in_zp).astype(np.int32)
        xq = np.clip(xq, -128, 127).astype(np.int8)
        interp.set_tensor(i_det['index'], xq)
        interp.invoke()
        yq = interp.get_tensor(o_det['index']).astype(np.int8, copy=False)
        yf = (yq.astype(np.float32) - out_zp) * out_scale
        Y_all.append(yf)
    ms = (time.time() - t0) * 1000.0 / max(1, len(Xf))
    Y = np.concatenate(Y_all, axis=0)
    return Y, ms, (in_scale, in_zp, out_scale, out_zp)


def run_keras(tf, savedmodel_dir: str, Xf: np.ndarray):
    mdl = tf.saved_model.load(savedmodel_dir)
    fn = mdl.signatures.get("serving_default", None)
    if fn is None:
        t0 = time.time()
        y = mdl(Xf, training=False)
        ms = (time.time() - t0) * 1000.0 / max(1, len(Xf))
    else:
        t0 = time.time()
        out = fn(tf.convert_to_tensor(Xf))
        ms = (time.time() - t0) * 1000.0 / max(1, len(Xf))
        y = list(out.values())[0]
    y = tf.convert_to_tensor(y).numpy().astype(np.float32, copy=False)
    return y, ms


def metrics(y_ref: np.ndarray, y_hat: np.ndarray):
    diff = y_hat - y_ref
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    maxe = float(np.max(np.abs(diff)))
    return mse, mae, maxe


def process_row(tf, row, savedmodel, n_windows, window, nfeat, threads, base_dir: Path):
    tflite_path = resolve_path(row["tflite_path"], base_dir)
    rep_path = resolve_path(row["rep_path"], base_dir)
    exp_name = row.get("exp_name", "")

    if not rep_path or not Path(rep_path).exists():
        raise FileNotFoundError(f"Rep file non trovato: {rep_path}")

    npz = np.load(rep_path)
    X = npz["X"] if "X" in npz.files else list(npz.values())[0]
    if X.shape[1:] != (window, nfeat):
        raise RuntimeError(f"[{exp_name}] shape {X.shape}, atteso (*,{window},{nfeat})")

    k = min(n_windows, X.shape[0])
    Xf = X[:k].astype(np.float32)

    Yt, ms_t, (in_scale, in_zp, out_scale, out_zp) = run_tflite(tf, str(tflite_path), Xf, threads=threads)
    yt_min, yt_max = brief_rng(Yt)
    sat_lo, sat_hi = quant_extreme_counts(Yt, out_scale, out_zp)

    repr_min = (-128 - out_zp) * out_scale
    repr_max = (127 - out_zp) * out_scale
    repr_width = repr_max - repr_min

    res = {
        "yk_min": np.nan, "yk_max": np.nan,
        "yt_min": yt_min, "yt_max": yt_max,
        "delta_range_obs": np.nan,
        "sat_lo_cnt": sat_lo, "sat_hi_cnt": sat_hi,
        "repr_out_min": repr_min, "repr_out_max": repr_max,
        "repr_out_width": repr_width, "lsb_out": out_scale,
        "in_scale": in_scale, "in_zp": int(in_zp),
        "out_scale": out_scale, "out_zp": int(out_zp),
        "tflite_ms_observed": ms_t, "n_windows_eval": int(k)
    }

    if savedmodel:
        savedmodel_path = resolve_path(savedmodel, base_dir)
        Yk, ms_k = run_keras(tf, str(savedmodel_path), Xf)
        yk_min, yk_max = brief_rng(Yk)
        mse_t, mae_t, maxe_t = metrics(Yk, Yt)
        res.update({
            "yk_min": yk_min, "yk_max": yk_max,
            "delta_range_obs": (yt_max - yt_min) - (yk_max - yk_min),
            "delta_mse_obs": mse_t, "delta_mae_obs": mae_t,
            "delta_rmse_obs": math.sqrt(max(0.0, mse_t)),
            "delta_max_obs": maxe_t,
            "keras_ms_observed": ms_k
        })
    return res


def main():
    ap = argparse.ArgumentParser(description="Batch golden compare all")
    ap.add_argument("--results", required=True, help="results.csv dello sweep")
    ap.add_argument("--out", required=True, help="CSV di output")
    ap.add_argument("--savedmodel", help="SavedModel Keras per confronto (opzionale)")
    ap.add_argument("--n-windows", type=int, default=1)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--nfeat", type=int, default=16)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--base-dir", help="Cartella base per percorsi relativi")
    args = ap.parse_args()

    tf = try_import_tf()
    df = pd.read_csv(args.results)
    base_dir = Path(args.base_dir) if args.base_dir else Path(args.results).resolve().parent

    extra_cols = [
        "yk_min","yk_max","yt_min","yt_max","delta_range_obs",
        "delta_mse_obs","delta_mae_obs","delta_rmse_obs","delta_max_obs",
        "keras_ms_observed","tflite_ms_observed","n_windows_eval",
        "repr_out_min","repr_out_max","repr_out_width","lsb_out",
        "sat_lo_cnt","sat_hi_cnt","in_scale","in_zp","out_scale","out_zp"
    ]
    for c in extra_cols:
        if c not in df.columns:
            df[c] = np.nan

    for idx, row in df.iterrows():
        try:
            res = process_row(tf, row, args.savedmodel, args.n_windows, args.window, args.nfeat, args.threads, base_dir)
            for k, v in res.items():
                df.at[idx, k] = v
            print(f"[OK] {row.get('exp_name','?')}")
        except Exception as e:
            print(f"[ERR] {row.get('exp_name','?')}: {e}", file=sys.stderr)

    if "delta_mse" in df.columns and "delta_rmse" not in df.columns:
        df["delta_rmse"] = np.sqrt(np.clip(df["delta_mse"].astype(float), 0, None))

    df.to_csv(args.out, index=False)
    print(f"[DONE] Salvato {args.out}  ({len(df)} righe)")


if __name__ == "__main__":
    main()
