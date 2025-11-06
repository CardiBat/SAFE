
# SAFE — TinyML Anomaly Detection (ARM/RISC-V ready)

This repository collects everything built so far to train, quantize, benchmark, and embed a multivariate time-series anomaly-detection model on microcontrollers (STM32 today; portable to other ARM/RISC-V MCUs). It includes data exploration, model training/export, representative-dataset generation for INT8 quantization, automated sweeps for benchmarking, and a minimal STM32CubeIDE/X-CUBE-AI project to run inference on target.

---

## Repository layout (what’s where)

```
SAFE
├── Analysis/                      # one-shot EDA over the full dataset
│   ├── analisi.py                 # generates stats + correlation + histograms
│   ├── stats_completo.csv
│   ├── correlazioni_full.png
│   ├── istogrammi_full.png
│   └── top_corr_pairs.csv
├── Training-Export/               # training and export utilities
│   ├── cnn_lstm.py                # baseline multistep predictor + SavedModel export
│   └── export_from_h5.py          # helper to re-export from .h5 (when needed)
├── export_cnn_lstm/               # trained model artifacts
│   ├── savedmodel/                # Keras SavedModel (float32)
│   └── savedmodel_unroll/         # SavedModel “unrolled” for TFLite conversion
├── Pipeline/                      # quantization + benchmarking toolchain
│   ├── make_rep_balanced_commented.py       # .npz generator for representative dataset
│   ├── convert_savedmodel.py                # full INT8 converter (TFLite)
│   ├── bench_tflite.py                      # benchmarking of single model run
│   ├── sweep_quant_bench.py                  # pipeline file orchestrating many runs
│   └── qbench_sweep_YYYYMMDD_HHMMSS/         # auto-created experiment root
│       ├── config.json
│       ├── experiments/...                   # per-run artifacts (rep/.npz, tflite/.tflite, logs/)
│       └── results.csv                       # consolidated metrics for the sweep
└── STM32 Interfacing/             # minimal CubeIDE project (U5/F4) with X-CUBE-AI
    ├── Prova.ioc                 # board config
    ├── main.c                    # buffering from UART
    └── app_x-cube-ai.c           # the model’s underlying architecture
```

### Important note on *Keras vs TFLite* folder artefacts

If you still maintain a folder for “Keras_vs_TFLite” (or similar) with older comparison scripts or outINT8 raw logs, **please review** it. Some files may be outdated versions of the pipeline.
Ensuring that you use the **latest version** of:

* the `golden_compare_all.py` script (with RMSE + LSB logic)
* the CSV human summaries (`compare_human.csv`)
* sweep results in `Pipeline/qbench_sweep_*`

will prevent confusion between “old” vs “new” metric definitions.

---

## What’s been done so far (short version)

* **Data exploration (EDA).** Produced global stats, correlations, and distributions to understand stability, tails, and cross-sensor relations used later to choose quantization strategy and features. Artifacts are under `Analysis/` as PNG/CSV.
* **Baseline model.** Implemented a **compact CNN+LSTM** multivariate forecaster (window=30, forecast=10) trained on standardized features; exported to `SavedModel` for conversion. Starter `rep_windows.npz` provided.
* **Representative dataset (INT8).** Built a robust tool that: (1) cleans non-numeric/timestamps, (2) standardizes, (3) **scores windows** (optionally passing through the SavedModel), (4) **selects a balanced mix of “normal” and high-score “tail” windows**, and (5) saves `rep_windows_balanced.npz` + JSON/plot.
* **Post-training quantization to INT8** and **benchmarking.** Converted to full-INT8 TFLite using the representative set, then measured Keras vs TFLite deltas (MSE/MAE/RMSE/MAX) and latency. The **sweep runner** automates a grid over `(max_total × balance)`, logging artifacts and a single results CSV.
* **MCU bring-up (STM32).** Created a minimal CubeIDE project (U5/F4) with X-CUBE-AI that ingests a `.tflite`, sets up UART/Timer, and runs the network loop; practical notes on printf/UART and 1 Hz acquisition are in the report and C sources.

---

## Quick start (minimal commands)

### 0) Environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install tensorflow numpy pandas scikit-learn matplotlib
```

### 1) Train & export baseline (float32)

```bash
python3 Training-Export/cnn_lstm.py
# → exports export_cnn_lstm/savedmodel and a starter rep_windows.npz
```

### 2) Build representative dataset (balanced tails)

```bash
python3 Pipeline/make_rep_balanced_commented.py \
  --csv /path/to/processed_streaming_row_continuous.csv \
  --savedmodel export_cnn_lstm/savedmodel_unroll \
  --outdir Pipeline/rep_2000_b0p5 \
  --window 30 --forecast 10 --nfeat 16 \
  --max_total 2000 --balance 0.5 --seed 42
```

### 3) Convert to full-INT8 TFLite

```bash
python3 Pipeline/convert_savedmodel.py \
  --savedmodel export_cnn_lstm/savedmodel_unroll \
  --rep Pipeline/rep_2000_b0p5/rep_windows_balanced.npz \
  --outdir Pipeline/tflite_2000_b0p5
```

### 4) Benchmark single run (Keras ↔ TFLite)

```bash
python3 Pipeline/bench_tflite.py \
  --savedmodel export_cnn_lstm/savedmodel_unroll \
  --tflite Pipeline/tflite_2000_b0p5/model_int8_full.tflite \
  --rep Pipeline/rep_2000_b0p5/rep_windows_balanced.npz \
  --n 256 --threads 1
```

### 5) Run a full sweep – many configurations

```bash
python3 Pipeline/sweep_quant_bench.py \
  --savedmodel export_cnn_lstm/savedmodel_unroll \
  --csv /path/to/processed_streaming_row_continuous.csv \
  --window 30 --forecast 10 --nfeat 16 \
  --max-totals 500,1000,2000,3000,4000,5000,6000 \
  --balances 0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60 \
  --threads 1 --n-rep 256
```

Each sweep run creates its own experiment folder under `Pipeline/qbench_sweep_…` with `results.csv` and subfolders per config.

---

## Key results so far (from the internal report)

A robust setting used for deployment trials is:

* Representative selection: `max_total = 2000`, `balance = 0.5`
* In benchmark: deltas vs Keras (on test bench) are MSE ≈ 0.13, RMSE ≈ 0.36, MAE ≈ 0.27, MAX ≈ 1.17
* On-device (STM32) deltas vs TFLite: ΔMAX ≈ 0.052 (≈ 1 LSB) → numerically equivalent implementation

> **Definition of LSB (Least Significant Bit):** the quantization step size (`scale`) of the output tensor. According to the [TensorFlow Lite Quantization Specification](https://www.tensorflow.org/lite/performance/quantization_spec) the reconstruction formula is (r = (q - zp) \times scale) and any difference ≤ 1 LSB is considered *“bit-exact within quantization precision”*. ([fdmcs.math.cnrs.fr][1])

---

