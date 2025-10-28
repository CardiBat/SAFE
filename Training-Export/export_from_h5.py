import tensorflow as tf
import os, shutil

MODEL_H5 = "cnn_lstm_model_cnn_lstm.h5"
EXPORT_DIR = "export_cnn_lstm/savedmodel"

if os.path.exists(EXPORT_DIR):
    shutil.rmtree(EXPORT_DIR)
os.makedirs(EXPORT_DIR, exist_ok=True)

# Carica il modello
model = tf.keras.models.load_model(MODEL_H5, compile=False)
print("✓ Modello H5 caricato correttamente")

# Esporta in formato TensorFlow SavedModel (Keras 3.x syntax)
model.export(EXPORT_DIR)
print(f"✓ SavedModel esportato in '{EXPORT_DIR}'")


