import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === PARAMETRI ===
file_csv = "/home/projects/safe/data/processed-datasets/processed_streaming_row_continuous.csv"

print(f"Carico intero dataset da {file_csv} ...")
df = pd.read_csv(file_csv)
print(f"✅ Letto: {df.shape[0]} righe, {df.shape[1]} colonne")

# === PULIZIA BASE ===
df_num = df.select_dtypes(include=['number'])
print(f"Colonne numeriche: {len(df_num.columns)}")

# === STATISTICHE COMPLETE ===
stats = df_num.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T
stats = stats[['mean', 'std', 'min', '1%', '5%', '50%', '95%', '99%', 'max']]
stats.to_csv("stats_completo.csv")
print("📄 Salvato: stats_completo.csv (media, std, min/max, percentili)")

# === VARIABILITÀ RELATIVA ===
stats["range"] = stats["max"] - stats["min"]
stats["coeff_var"] = stats["std"] / (np.abs(stats["mean"]) + 1e-9)
top_var = stats.sort_values("coeff_var", ascending=False).head(10)
print("\n⚙️ Variabili più instabili (coefficiente di variazione):")
print(top_var[["mean", "std", "coeff_var"]])

# === CORRELAZIONI ===
corr = df_num.corr()
plt.figure(figsize=(14, 12))
sns.heatmap(corr, cmap="coolwarm", center=0, square=True, cbar_kws={'shrink':0.6})
plt.title("Matrice di correlazione (tutte le variabili)")
plt.tight_layout()
plt.savefig("correlazioni_full.png", dpi=300)
plt.close()
print("📊 Salvato: correlazioni_full.png")

# === ISTOGRAMMI ===
cols = df_num.columns
n = len(cols)
ncols = 4
nrows = (n + ncols - 1) // ncols

plt.figure(figsize=(ncols * 4, nrows * 3))
df_num.hist(bins=50, figsize=(ncols * 4, nrows * 3), layout=(nrows, ncols))
plt.suptitle("Distribuzione dei valori (tutte le variabili)")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("istogrammi_full.png", dpi=300)
plt.close()
print("📊 Salvato: istogrammi_full.png")

# === CORRELAZIONI PRINCIPALI (heatmap ridotta)
abs_corr = corr.abs()
corr_pairs = abs_corr.unstack().sort_values(ascending=False)
corr_pairs = corr_pairs[corr_pairs < 1]  # esclude diagonale
top_corr = corr_pairs.head(20)
top_corr.to_csv("top_corr_pairs.csv")
print("📄 Salvato: top_corr_pairs.csv (20 coppie più correlate)")

print("\n✅ Analisi completa terminata.")
