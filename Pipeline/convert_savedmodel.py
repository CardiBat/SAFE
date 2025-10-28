# quantizzazione/convert_savedmodel.py
# -*- coding: utf-8 -*-
import os, sys, argparse
import numpy as np
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # forza CPU

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--savedmodel", required=True, help="Path SavedModel (unrolled)")
    ap.add_argument("--outdir",     required=True, help="Cartella output .tflite")
    ap.add_argument("--rep",        required=True, help="Representative dataset .npz")
    return ap.parse_args()

def rep_gen_from_npz(npz_path, max_samples=4096):
    rep = np.load(npz_path)
    X = rep["X"] if "X" in rep.files else list(rep.values())[0]
    X = X.astype(np.float32, copy=False)
    n = min(len(X), max_samples)
    def gen():
        for i in range(n):
            yield [X[i:i+1]]
    return gen, X.shape

def main():
    args = parse_args()

    print("=== CONVERT CONFIG ===")
    print("SavedModel:", os.path.abspath(args.savedmodel))
    print("Representative:", os.path.abspath(args.rep))
    print("Outdir:", os.path.abspath(args.outdir))

    os.makedirs(args.outdir, exist_ok=True)
    tfl_path = os.path.join(args.outdir, "model_int8_full.tflite")

    rep_gen, rep_shape = rep_gen_from_npz(args.rep)
    print("Representative shape:", rep_shape)

    converter = tf.lite.TFLiteConverter.from_saved_model(args.savedmodel)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    try:
        converter.experimental_enable_resource_variables = True
    except Exception:
        pass

    tfl = converter.convert()
    with open(tfl_path, "wb") as f:
        f.write(tfl)
    print(f"✓ Salvato: {tfl_path} ({os.path.getsize(tfl_path)/1024:.1f} KB)")

    interp = tf.lite.Interpreter(model_path=tfl_path)
    interp = tf.lite.Interpreter(model_path=tfl_path)
    interp.allocate_tensors()

    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    print("Input tensor:")
    print("  name:", inp["name"])
    print("  quantization (scale, zero_point):", inp["quantization"])
    print("  dtype:", inp["dtype"])

    print("Output tensor:")
    print("  name:", out["name"])
    print("  quantization (scale, zero_point):", out["quantization"])
    print("  dtype:", out["dtype"])

    print(f"[TFL_OK] saved {tfl_path}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
