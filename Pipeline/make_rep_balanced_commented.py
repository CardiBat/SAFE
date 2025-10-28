# quantizzazione/make_rep_balanced_v2.py --> il nome del file è sempre balanced 
# -*- coding: utf-8 -*-
"""
Versione robusta: gestisce timestamp/colonne object e numeri come stringhe.
- Non forza dtype al read_csv (evita crash su timestamp).
- Converte automaticamente gli 'object' che sono numerici (es. "1.23") in float.
- Rileva colonne datetime e le esclude dalle feature.
Il resto della logica (scoring code vs normali, salvataggi) è identica.

Per il tuning, modificare max total e balance ratio. Il 30-10-16 è invece base standard del modello Keras

python3 quantizzazione/make_rep_balanced.py \
  --csv /home/projects/safe/data/processed-datasets/processed_streaming_row_continuous.csv \
  --savedmodel export_cnn_lstm/savedmodel_unroll \
  --window 30 --forecast 10 --nfeat 16 \
  --max_total 2000 --balance 0.5 --batch_size 2048

"""

import os
import json
import argparse
from pathlib import Path
from typing import Iterator, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



try:
    from sklearn.preprocessing import StandardScaler
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False


def iter_windows(arr: np.ndarray, window: int, forecast: int, stride: int, batch_size: int):
    n = arr.shape[0]
    last_start = n - window - forecast + 1
    if last_start <= 0:
        return
    starts = range(0, last_start, stride)
    Xb, first = [], None
    for s in starts:
        if first is None:
            first = s
        Xb.append(arr[s:s+window])
        if len(Xb) == batch_size:
            yield np.stack(Xb, 0), first
            Xb, first = [], None
    if Xb:
        yield np.stack(Xb, 0), first


def load_savedmodel(savedmodel_dir: Optional[str]):
    if not savedmodel_dir:
        return None
    try:
        import tensorflow as tf
    except Exception:
        return None
    p = Path(savedmodel_dir)
    if not p.exists():
        return None
    # Keras first
    try:
        model = tf.keras.models.load_model(p, compile=False)
        def predict_fn(x):
            y = model(x, training=False)
            return y.numpy() if hasattr(y, "numpy") else np.array(y)
        return predict_fn
    except Exception:
        pass
    # Signatures
    try:
        loaded = tf.saved_model.load(str(p))
        infer = loaded.signatures.get("serving_default", None)
        if infer is None:
            return None
        def predict_fn(x):
            import tensorflow as tf
            x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
            out = infer(x_tf)
            if isinstance(out, dict):
                out = next(iter(out.values()))
            return out.numpy() if hasattr(out, "numpy") else np.array(out)
        return predict_fn
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="Crea rep_windows_balanced.npz (robusto a timestamp/object)")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--savedmodel", required=False, default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--forecast", type=int, default=10)
    ap.add_argument("--nfeat", type=int, default=16)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--max_total", type=int, default=2000)
    ap.add_argument("--balance", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_numeric_frac", type=float, default=0.9,
                    help="Quota minima di valori convertibili per trattare una colonna object come numerica")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    # --- output directory (rispetta --outdir, fallback a analysis/output) ---
    from pathlib import Path

    outdir = Path(args.outdir) if args.outdir else Path("analysis/output")
    outdir.mkdir(parents=True, exist_ok=True)

    # pulizia eventuali artefatti preesistenti nella cartella scelta
    for fname in ["rep_windows_balanced.npz",
                "rep_balance_stats.json",
                "hist_max_abs_y_pred_balanced.png"]:
        try:
            (outdir / fname).unlink(missing_ok=True)
        except Exception:
            pass
    # --- fine gestione outdir ---



    print("=== CONFIG ===")
    print(f"CSV:         {args.csv}")
    print(f"SavedModel:  {args.savedmodel or '(none, data-only)'}")
    print(f"Output dir:  {str(outdir)}")
    print(f"window={args.window}  forecast={args.forecast}  nfeat={args.nfeat}  stride={args.stride}")
    print(f"max_total={args.max_total}  balance={args.balance}  batch_size={args.batch_size}")
    print(f"min_numeric_frac={args.min_numeric_frac}")

    # 1) Read CSV senza forzare dtype
    print("\n[1/6] Lettura CSV...")
    df = pd.read_csv(args.csv, low_memory=False)
    n_before = len(df.columns)

    # 2) Converti automaticamente object→float o datetime→drop
    print("[2/6] Conversione colonne object (numeriche) ed esclusione datetime...")
    drop_cols = []
    for col in list(df.columns):
        dt = df[col].dtype
        if np.issubdtype(dt, np.number):
            continue
        coerced = pd.to_numeric(df[col], errors="coerce")
        frac_num = float(coerced.notna().mean())
        if frac_num >= args.min_numeric_frac:
            df[col] = coerced.astype(np.float32)
            continue
        dtcoerced = pd.to_datetime(df[col], errors="coerce", utc=False, infer_datetime_format=True)
        frac_dt = float(dtcoerced.notna().mean())
        if frac_dt >= args.min_numeric_frac:
            drop_cols.append(col)
            continue
        drop_cols.append(col)

    if drop_cols:
        df = df.drop(columns=drop_cols)

    print(f"   → Colonne iniziali: {n_before} | dopo conversione/drop: {len(df.columns)}")
    if len(df.columns) == 0:
        raise ValueError("Nessuna colonna numerica utile trovata dopo la pulizia.")

    # 3) Selezione feature numeriche
    print("[3/6] Selezione feature numeriche...")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < args.nfeat:
        print(f"   ⚠ Avviso: richieste nfeat={args.nfeat} ma trovate solo {len(num_cols)} colonne numeriche. Userò {len(num_cols)}.")
    use_cols = num_cols[:min(len(num_cols), args.nfeat)]
    data = df[use_cols].to_numpy(dtype=np.float32, copy=False)
    del df

    # 4) Standardizzazione
    print("[4/6] Standardizzazione...")
    if HAVE_SKLEARN:
        scaler = StandardScaler()
        data_std = scaler.fit_transform(data).astype(np.float32, copy=False)
        scaler_dict = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(), "use_cols": use_cols}
    else:
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0); std[std == 0] = 1.0
        data_std = (data - mean) / std
        scaler_dict = {"mean": mean.tolist(), "scale": std.tolist(), "use_cols": use_cols}

    n_rows, nfeat = data_std.shape
    print(f"   ✓ Dati standardizzati: shape={data_std.shape}")

    # 5) Scoring finestre
    print("[5/6] Scoring finestre...")
    predict_fn = load_savedmodel(args.savedmodel)
    scores = []; n_windows_total = 0
    for Xb, _ in iter_windows(data_std, args.window, args.forecast, args.stride, args.batch_size):
        if predict_fn is not None:
            try:
                yb = predict_fn(Xb); yb = np.asarray(yb)
                score_b = np.max(np.abs(yb), axis=tuple(range(1, yb.ndim)))
            except Exception:
                score_b = np.max(np.abs(Xb), axis=(1, 2))
        else:
            score_b = np.max(np.abs(Xb), axis=(1, 2))
        scores.append(score_b); n_windows_total += Xb.shape[0]
    if n_windows_total == 0:
        raise RuntimeError("Nessuna finestra generabile: controlla window/forecast/righe del CSV.")
    scores = np.concatenate(scores, 0)

    p95 = float(np.percentile(scores, 95.0))
    max_total = int(args.max_total)
    k_tail = int(round(max_total * args.balance)); k_norm = max_total - k_tail

    tail_mask = scores >= p95
    tail_idx = np.nonzero(tail_mask)[0]
    if len(tail_idx) > 0:
        tail_sorted = tail_idx[np.argsort(scores[tail_idx])[::-1]]
        tail_pick = tail_sorted[:k_tail]
    else:
        tail_pick = np.array([], dtype=int)

    normal_pool = np.nonzero(~tail_mask)[0]
    if len(normal_pool) >= k_norm:
        rng = np.random.default_rng(args.seed)
        norm_pick = rng.choice(normal_pool, size=k_norm, replace=False)
    else:
        norm_pick = normal_pool
        deficit = k_norm - len(norm_pick)
        extra_from_tail = [i for i in (tail_sorted[k_tail:k_tail+deficit] if len(tail_idx) > 0 else [])]
        if extra_from_tail:
            tail_pick = np.concatenate([tail_pick, np.array(extra_from_tail, dtype=int)])

    selected_idx = np.unique(np.concatenate([tail_pick, norm_pick]))
    if selected_idx.shape[0] > max_total:
        selected_idx = selected_idx[:max_total]

    # 6) Estrazione e salvataggi
    print("[6/6] Estrazione finestre selezionate e salvataggi...")
    X_sel = np.zeros((selected_idx.shape[0], args.window, nfeat), dtype=np.float32)
    pos_map = {int(g): i for i, g in enumerate(selected_idx)}

    filled, base = 0, 0
    for Xb, _ in iter_windows(data_std, args.window, args.forecast, args.stride, args.batch_size):
        bsz = Xb.shape[0]; gidx = np.arange(base, base + bsz, dtype=int)
        mask = np.isin(gidx, selected_idx)
        if np.any(mask):
            where = gidx[mask]; src_idx = np.nonzero(mask)[0]
            for j, gi in enumerate(where):
                X_sel[pos_map[int(gi)]] = Xb[src_idx[j]]; filled += 1
        base += bsz
    assert filled == X_sel.shape[0], f"Riempiti {filled}/{X_sel.shape[0]}: discrepanza indici."

    npz_path = outdir / "rep_windows_balanced.npz"
    json_path = outdir / "rep_balance_stats.json"
    png_path = outdir / "hist_max_abs_y_pred_balanced.png"

    np.savez(npz_path, X=X_sel.astype(np.float32))
    stats = {
        "n_rows": int(n_rows), "nfeat": int(nfeat),
        "window": int(args.window), "forecast": int(args.forecast), "stride": int(args.stride),
        "max_total": int(max_total), "balance": float(args.balance),
        "scores": {
            "count_windows": int(n_windows_total),
            "p95": float(p95), "min": float(scores.min()), "max": float(scores.max()),
            "mean": float(scores.mean()), "std": float(scores.std()),
        },
        "selected": {"tail": int(k_tail), "normal": int(k_norm), "actual_total": int(X_sel.shape[0])},
        "scaler": scaler_dict, "use_cols": use_cols, "savedmodel_used": bool(predict_fn is not None),
        "dropped_columns": list(map(str, drop_cols)),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        import json as _json; _json.dump(stats, f, ensure_ascii=False, indent=2)

    plt.figure(figsize=(8, 4))
    plt.hist(scores, bins=100, alpha=0.8)
    plt.axvline(p95, linestyle="--", label="p95")
    plt.title("Distribuzione score finestre")
    plt.xlabel("score (max abs output" + ("" if predict_fn is not None else " / input") + ")")
    plt.ylabel("conteggio")
    plt.legend(); plt.tight_layout(); plt.savefig(png_path, dpi=140)

    print(f"✓ Representative: {npz_path}")
    print(f"✓ Statistiche:    {json_path}")
    print(f"✓ Istogramma:     {png_path}")
    print("\nFatto. Usa questo NPZ per la quantizzazione INT8.")

if __name__ == "__main__":
    main()