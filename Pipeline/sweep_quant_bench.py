#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sweep quantizzazione + benchmark, self-contained.
- Per ogni (max_total, balance) crea NPZ in subfolder dedicata
- Converte SavedModel -> TFLite usando il tuo convert_savedmodel.py
- Esegue bench con bench_tflite.py
- Aggrega i risultati in results.csv

python3 sweep_quant_bench.py \
  --savedmodel ../export_cnn_lstm/savedmodel_unroll \
  --csv /home/projects/safe/data/processed-datasets/processed_streaming_row_continuous.csv \
  --window 30 --forecast 10 --nfeat 16 \
  --max-totals 500,1000,2000,3000,4000,5000,6000 \
  --balances 0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6 \
  --threads 1 --n-rep 256

"""

import argparse, os, sys, json, re, csv, time, subprocess
from pathlib import Path
from datetime import datetime

# -------------- utils --------------

def sh(cmd, cwd=None, log_file=None, env=None):
    """Esegue un comando di shell e ritorna stdout come stringa. Logga su file se richiesto."""
    proc = subprocess.run(
        cmd, cwd=cwd, env=env or os.environ.copy(),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False
    )
    out = proc.stdout
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        Path(log_file).write_text(out, encoding='utf-8')
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n---LOG---\n{out}")
    return out

def find_one_with_ext(folder: Path, ext: str):
    files = sorted(folder.glob(f"*{ext}"))
    return files[0] if files else None

def load_npz_count(npz_path: Path) -> int:
    import numpy as np
    data = np.load(npz_path)
    X = data['X'] if 'X' in data.files else list(data.values())[0]
    return int(len(X))

def tflite_quant_params(tfl_path: Path):
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=str(tfl_path))
    interp.allocate_tensors()
    i_det = interp.get_input_details()[0]
    o_det = interp.get_output_details()[0]
    in_scale, in_zp  = i_det.get('quantization', (None, None))
    out_scale, out_zp = o_det.get('quantization', (None, None))
    return (in_scale, in_zp, out_scale, out_zp)

def parse_bench_output(stdout: str):
    # Keras e TFLite ms/finestre
    m_k = re.search(r"Keras out: .*?~([0-9]+\.[0-9]+)\s*ms/finestre", stdout)
    m_t = re.search(r"TFLite out: .*?~([0-9]+\.[0-9]+)\s*ms/finestre", stdout)
    ms_keras = float(m_k.group(1)) if m_k else None
    ms_tflite = float(m_t.group(1)) if m_t else None
    # Δ métriche
    m_delta = re.search(
        r"Δ TFLite vs Keras → MSE:\s*([0-9.eE+-]+)\s*\|\s*MAE:\s*([0-9.eE+-]+)\s*\|\s*MAX:\s*([0-9.eE+-]+)",
        stdout
    )
    mse = float(m_delta.group(1)) if m_delta else None
    mae = float(m_delta.group(2)) if m_delta else None
    maxe = float(m_delta.group(3)) if m_delta else None
    return ms_keras, ms_tflite, mse, mae, maxe

# -------------- main --------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--savedmodel", required=True, help="Path al SavedModel (unrolled)")
    ap.add_argument("--csv",        required=True, help="CSV dei dati grezzi")
    ap.add_argument("--window",     type=int, default=30)
    ap.add_argument("--forecast",   type=int, default=10)
    ap.add_argument("--nfeat",      type=int, default=16)

    ap.add_argument("--balances",   type=str, default="0.5", help="Lista es: 0.25,0.5,0.75")
    ap.add_argument("--max-totals", type=str, default="200", help="Lista es: 100,200,500,1000")

    ap.add_argument("--threads",    type=int, default=1, help="Thread TFLite per bench")
    ap.add_argument("--n-rep",      type=int, default=256, help="Campioni max dal rep .npz per bench")

    ap.add_argument("--make-rep",   type=str, default="make_rep_balanced_commented.py")
    ap.add_argument("--convert",    type=str, default="convert_savedmodel.py")
    ap.add_argument("--bench",      type=str, default="bench_tflite.py")

    ap.add_argument("--outdir",     type=str, default=None, help="Root dei risultati (default: qbench_sweep_<timestamp>)")
    ap.add_argument("--exp-name",   type=str, default=None, help="Nome logico esperimento (opzionale)")
    ap.add_argument("--seed",       type=int, default=42)

    args = ap.parse_args()

    # Root self-contained
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.outdir) if args.outdir else Path(f"qbench_sweep_{ts}")
    out_root.mkdir(parents=True, exist_ok=True)

    # Salva config
    config = vars(args).copy()
    (out_root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # Normalizza liste
    balances = [float(x) for x in args.balances.split(",") if x.strip() != ""]
    max_totals = [int(x) for x in args.max_totals.split(",") if x.strip() != ""]

    # Percorsi script
    here = Path.cwd()
    make_rep_py = (here / args.make_rep).resolve()
    convert_py  = (here / args.convert).resolve()
    bench_py    = (here / args.bench).resolve()

    savedmodel = Path(args.savedmodel).resolve()
    csv_path   = Path(args.csv).resolve()

    # CSV risultati
    results_csv = out_root / "results.csv"
    new_file = not results_csv.exists()
    with results_csv.open("a", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        if new_file:
            writer.writerow([
                "exp_name","max_total","balance","rep_count",
                "tflite_kb","ms_keras","ms_tflite",
                "delta_mse","delta_mae","delta_max",
                "in_scale","in_zp","out_scale","out_zp",
                "rep_path","tflite_path","status","note"
            ])

        # Sweep
        for mt in max_totals:
            for bal in balances:
                exp_name = args.exp_name or f"mt{mt}_b{str(bal).replace('.','p')}"
                exp_dir = out_root / "experiments" / exp_name
                rep_dir = exp_dir / "rep"
                tfl_dir = exp_dir / "tflite"
                log_dir = exp_dir / "logs"
                rep_dir.mkdir(parents=True, exist_ok=True)
                tfl_dir.mkdir(parents=True, exist_ok=True)
                log_dir.mkdir(parents=True, exist_ok=True)

                status, note = "OK", ""
                rep_npz = None
                tfl_path = None
                rep_count = None
                ms_k = ms_t = mse = mae = maxe = None
                in_scale = in_zp = out_scale = out_zp = None
                tfl_kb = None

                try:
                    # 1) Make representative .npz (nella subfolder dedicata)
                    cmd_rep = [
                        sys.executable, str(make_rep_py),
                        "--csv", str(csv_path),
                        "--savedmodel", str(savedmodel),
                        "--outdir", str(rep_dir),
                        "--window", str(args.window),
                        "--forecast", str(args.forecast),
                        "--nfeat", str(args.nfeat),
                        "--max_total", str(mt),
                        "--balance", str(bal),
                        "--seed", str(args.seed),
                    ]
                    out_rep = sh(cmd_rep, log_file=log_dir / "01_make_rep.log")
                    rep_npz = find_one_with_ext(rep_dir, ".npz")
                    if not rep_npz:
                        raise FileNotFoundError(f"Nessun .npz trovato in {rep_dir}")
                    rep_count = load_npz_count(rep_npz)

                    # 2) Convert SavedModel -> TFLite (salva nella subfolder)
                    cmd_conv = [
                        sys.executable, str(convert_py),
                        "--savedmodel", str(savedmodel),
                        "--outdir", str(tfl_dir),
                        "--rep", str(rep_npz),
                    ]
                    out_conv = sh(cmd_conv, log_file=log_dir / "02_convert.log")
                    tfl_path = find_one_with_ext(tfl_dir, ".tflite")
                    if not tfl_path:
                        raise FileNotFoundError(f"Nessun .tflite trovato in {tfl_dir}")
                    tfl_kb = round(tfl_path.stat().st_size / 1024.0, 2)

                    # 3) Benchmark (parsing dello stdout)
                    cmd_bench = [
                        sys.executable, str(bench_py),
                        "--savedmodel", str(savedmodel),
                        "--tflite", str(tfl_path),
                        "--rep", str(rep_npz),
                        "--n", str(args.n_rep),
                        "--threads", str(args.threads),
                    ]
                    out_bench = sh(cmd_bench, log_file=log_dir / "03_bench.log")
                    ms_k, ms_t, mse, mae, maxe = parse_bench_output(out_bench)

                    # 4) Estrai quant params input/output per completezza
                    in_scale, in_zp, out_scale, out_zp = tflite_quant_params(tfl_path)

                except Exception as e:
                    status = "FAIL"
                    note = str(e).splitlines()[0][:400]

                writer.writerow([
                    exp_name, mt, bal, rep_count,
                    tfl_kb, ms_k, ms_t,
                    mse, mae, maxe,
                    in_scale, in_zp, out_scale, out_zp,
                    str(rep_npz) if rep_npz else "",
                    str(tfl_path) if tfl_path else "",
                    status, note
                ])
                fcsv.flush()
                print(f"[{status}] {exp_name} → rep={rep_count} | tflite={tfl_kb} KB | "
                      f"ms(K/T)={ms_k}/{ms_t} | ΔMAE={mae} | note={note}")

    print(f"\nDone. Tabella: {results_csv}\nRoot: {out_root}")

if __name__ == "__main__":
    # Evita che TF usi la GPU per numeri di bench (coerente con i tuoi script)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    main()
