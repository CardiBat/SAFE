# SAFE — TinyML Anomaly Detection (ARM/RISC-V ready)

This repository collects everything built so far to train, quantize, benchmark, and embed a multivariate time-series anomaly-detection model on microcontrollers (STM32 today; portable to other ARM/RISC-V MCUs). It includes data exploration, model training/export, representative-dataset generation for INT8 quantization, automated sweeps for benchmarking, and a minimal STM32CubeIDE/X-CUBE-AI project to run inference on target.

---

## Repository layout (what’s where)

```
SAFE
├── Analysis/                 # one-shot EDA over the full dataset
│   ├── analisi.py            # generates stats + correlation + histograms
│   ├── stats_completo.csv
│   ├── correlazioni_full.png
│   ├── istogrammi_full.png
│   └── top_corr_pairs.csv
├── Training-Export/          # training and export utilities
│   ├── cnn_lstm.py           # baseline multistep predictor + SavedModel export
│   └── export_from_h5.py     # helper to re-export from .h5 (when needed)
├── export_cnn_lstm/          # trained model artifacts
│   ├── savedmodel/           # Keras SavedModel (float32)
│   └── savedmodel_unroll/    # SavedModel “unrolled” for TFLite conversion
├── Pipeline/                 # quantization + benchmarking toolchain
│   ├── make_rep_balanced_commented.py
│   ├── convert_savedmodel.py
│   ├── bench_tflite.py
│   ├── sweep_quant_bench.py
│   └── qbench_sweep_YYYYMMDD_HHMMSS/   # auto-created experiment root
│       ├── config.json
│       ├── experiments/...              # per-run artifacts (rep/.npz, tflite/.tflite, logs/)
│       └── results.csv                  # consolidated metrics
└── STM32 Interfacing/        # minimal CubeIDE project (U5/F4) with X-CUBE-AI
    ├── Prova.ioc             # board config
    ├── main.c
    └── app_x-cube-ai.c
```

* `Analysis/analisi.py` loads the full CSV, saves summary statistics, correlation heatmap and per-feature histograms to the files listed above. 
* `Training-Export/cnn_lstm.py` trains a compact CNN+LSTM forecaster (30→10 multistep) and exports both `.h5` and `SavedModel` plus a tiny starter representative dataset. 
* `Pipeline/` contains the representative-dataset builder, converter, single-run benchmark, and a sweep orchestrator that creates per-experiment folders and a single `results.csv`. 

---

## What’s been done so far (short version)

* **Data exploration (EDA).** Produced global stats, correlations, and distributions to understand stability, tails, and cross-sensor relations used later to choose quantization strategy and features. Artifacts are under `Analysis/` as PNG/CSV. 
* **Baseline model.** Implemented a **compact CNN+LSTM** multi-sensor forecaster (window=30, forecast=10) trained on standardized features; exported to `SavedModel` for conversion. Creates a starter `rep_windows.npz` for quick tests. 
* **Representative dataset (INT8).** Built a robust creator that: (1) cleans non-numeric/timestamps, (2) standardizes, (3) **scores windows** (optionally passing through the SavedModel), (4) **selects a balanced mix of “normal” and high-score “tail” windows** (e.g., 50/50 over `max_total`), and (5) saves `rep_windows_balanced.npz` + JSON/plot. 
* **Post-training quantization to INT8** and **benchmarking.** Converted to full-INT8 TFLite using the representative set, then measured Keras vs TFLite deltas (MSE/MAE/MAX) and latency. The **sweep runner** automates a grid over `(max_total × balance)`, logging artifacts and a consolidated table. 
* **MCU bring-up (STM32).** Created a minimal CubeIDE project (U5/F4) with X-CUBE-AI that ingests a `.tflite`, sets up UART/Timer, and runs the network loop; practical notes on printf/UART and 1 Hz acquisition are in the report and C sources.

For background, rationale, and hardware notes (windowing rationale, drift vs spikes, X-CUBE-AI workflow, UART/Timer setup) see the updated report up to page ~70. 

---

## Quick start (minimal commands)

> These are orientation snippets, not a full runner. Paths may need adjusting.

### 0) Environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install tensorflow numpy pandas scikit-learn matplotlib
```

### 1) Train & export a baseline (float32)

```bash
python3 Training-Export/cnn_lstm.py
# -> exports export_cnn_lstm/savedmodel and a starter rep_windows.npz
```

The script standardizes numeric columns, creates (window=30, forecast=10) sequences, trains with early-stopping, and saves a Keras **SavedModel**. 

### 2) Build a representative dataset (balanced tails)

```bash
python3 Pipeline/make_rep_balanced_commented.py \
  --csv /path/to/processed_streaming_row_continuous.csv \
  --savedmodel export_cnn_lstm/savedmodel_unroll \
  --outdir Pipeline/rep_2000_b0p5 \
  --window 30 --forecast 10 --nfeat 16 \
  --max_total 2000 --balance 0.5 --seed 42
```

This tool auto-handles object/datetime columns, standardizes, **scores windows** (via model if provided), then selects ~p95 “tail” windows + uniform “normal” windows according to `--balance`. Saves NPZ + stats PNG/JSON. 

### 3) Convert to full-INT8 TFLite

```bash
python3 Pipeline/convert_savedmodel.py \
  --savedmodel export_cnn_lstm/savedmodel_unroll \
  --rep Pipeline/rep_2000_b0p5/rep_windows_balanced.npz \
  --outdir Pipeline/tflite_2000_b0p5
```

(Uses the representative set for proper INT8 calibration.)

### 4) Benchmark single run

```bash
python3 Pipeline/bench_tflite.py \
  --savedmodel export_cnn_lstm/savedmodel_unroll \
  --tflite Pipeline/tflite_2000_b0p5/model_int8_full.tflite \
  --rep Pipeline/rep_2000_b0p5/rep_windows_balanced.npz \
  --n 256 --threads 1
# Parses Δ MSE/MAE/MAX and Keras/TFLite latency (ms/window).
```

(Arguments mirror what the sweep uses under the hood.) 

### 5) Run a full sweep (creates its own folders)

```bash
python3 Pipeline/sweep_quant_bench.py \
  --savedmodel export_cnn_lstm/savedmodel_unroll \
  --csv /path/to/processed_streaming_row_continuous.csv \
  --window 30 --forecast 10 --nfeat 16 \
  --max-totals 500,1000,2000,3000,4000,5000,6000 \
  --balances 0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6 \
  --threads 1 --n-rep 256
# Outputs: Pipeline/qbench_sweep_YYYYMMDD_HHMMSS/results.csv (+ per-experiment artifacts)
```

The sweep: (1) **creates** a rep NPZ per setting, (2) converts to TFLite, (3) benchmarks and logs Δ metrics & latency, (4) writes a single CSV with quant params for input/output tensors. 

---

## Key results so far (from the internal report)

A robust setting used for deployment trials is:

* Representative selection: `max_total=2000`, `balance=0.5` (≈50% tails by p95 score).
* Observed deltas vs Keras (on the test bench): **MSE ≈ 0.37**, **MAE ≈ 0.22**, **MAX ≈ 2.6**, with **range alignment** between Keras and INT8 indicating *no clipping*. Larger reps (e.g., 4000/6000) or unbalanced tails tended to worsen MAE/MSE. 

> Why the “balanced tails”? To preserve precision where data actually live (the “bell”) while **including tails** to avoid clipping: we calibrate scales with a controlled share of extreme windows instead of letting them dominate. The representative builder implements exactly this policy. 

---

## STM32 embedding (very short)

* Open `STM32 Interfacing/Prova.ioc` in **STM32CubeIDE**, install **X-CUBE-AI**, import the `.tflite`, and let the tool generate the runtime glue.
* Enable **USART** (115200) for logs and a **Timer** (1 Hz) for periodic acquisition; the project skeleton shows how TX/RX and the AI loop are wired. Build/flash and use a serial terminal to monitor. (See comments in the C sources and the report section on UART, timers, and printf redirection.) 

---

## Conventions & notes

* **Windows/features.** Default `--window 30`, `--forecast 10`, `--nfeat 16` across scripts; adjust to match the trained model. 
* **SavedModel “unroll.”** Use the **unrolled** SavedModel for cleaner TFLite conversion paths. The sweep and converter expect it. 
* **Representative scoring.** If a SavedModel is provided, window **scores** come from the model’s response; otherwise a safe proxy is used (input magnitude). Selection mixes top-score “tails” and uniform normal windows. 
* **Artifacts.** Each sweep run is self-contained under `Pipeline/qbench_sweep_*` with logs for reproducibility and a single `results.csv` to filter/sort best trade-offs. 

---

## Suggested next steps

* Lock final feature set (16 cols) from `Analysis/` outputs and freeze a stable SavedModel. 
* Re-sweep around the **2000/0.5** region with a couple of seeds to confirm stability; promote the winner to MCU. 
* Optional: quant-aware tweaks (e.g., rep balancing per-channel) if MAX deltas on rare cases matter operationally. 

---

**Contacts / Credits**
Analysis, modeling, quantization pipeline, and STM32 bring-up created and iterated across the files listed above. See comments in each script for more details and inline guidance.
