import joblib
import numpy as np
import pandas as pd
import os
from pathlib import Path

# Load model dan fitur
# Load model dan komponen pendukung
def load_model_components():
    # Dapatkan path absolut ke folder models
    model_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models", "prediksi_popularitas")
    
    # Pastikan path benar-benar ada
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found at: {model_dir}")
    
    # Definisikan path untuk setiap file
    model_path = os.path.join(model_dir, "prediksi_popularitas.joblib")
    scaler_path = os.path.join(model_dir, "scaler_popularitas.joblib")
    mlb_path = os.path.join(model_dir, "mlb_populairatas.joblib")
    features_path = os.path.join(model_dir, "features_popularis.joblib")
    
    # Verifikasi semua file ada sebelum memuat
    for path in [model_path, scaler_path, mlb_path, features_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
    
    return {
        'model': joblib.load(model_path),
        'scaler': joblib.load(scaler_path),
        'mlb': joblib.load(mlb_path),
        'features': joblib.load(features_path)
    }
components = load_model_components()
model = components['model']
scaler = components['scaler']
mlb = components['mlb']
features = components['features']


# Contoh input (bisa kamu ganti sesuai data asli)
input_data = {
    'rating': 1.0,
    'jumlah_review': 10,
    'log_jumlah_review': np.log1p(10000),
    'kategori': ['Pantai', 'Keluarga'],
    'provinsi': 'Bali'
}

# --- Encode kategori ---
kategori_encoded = mlb.transform([input_data['kategori']])
kategori_df = pd.DataFrame(kategori_encoded, columns=mlb.classes_)

# --- Encode provinsi ---
provinsi_df = pd.get_dummies(pd.Series(input_data['provinsi']), prefix='provinsi')
provinsi_df = provinsi_df.reindex(columns=[f for f in features if f.startswith('provinsi_')], fill_value=0)

# Gabungkan semua fitur
X = pd.concat([
    pd.DataFrame([{
        'rating': input_data['rating'],
        'log_jumlah_review': input_data['log_jumlah_review']
    }]),
    kategori_df,
    provinsi_df
], axis=1)

# Susun kolom sesuai urutan training
X = X.reindex(columns=features, fill_value=0)

# Scaling
X_scaled = scaler.transform(X)

# Prediksi
prediksi = model.predict(X_scaled)
proba = model.predict_proba(X_scaled)[0][1]

# Tampilkan hasil
print("Hasil prediksi:", "Populer" if prediksi[0] == 1 else "Tidak Populer")
print("Tingkat keyakinan: {:.2f}%".format(proba * 100))
