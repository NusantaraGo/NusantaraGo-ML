from flask import jsonify, request, render_template, Flask, Request, flash, redirect, url_for
import joblib
import numpy as np
import pandas as pd
import os
from pathlib import Path

app = Flask(__name__)
app.secret_key = "123" 

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

# Muat komponen sekali saat startup
components = load_model_components()
model = components['model']
scaler = components['scaler']
mlb = components['mlb']
features = components['features']

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        rating = float(request.form['rating'])
        jumlah_review = int(request.form['jumlah_review'])
        kategori = request.form.getlist('kategori')
        provinsi = request.form['provinsi']

        # Encoding kategori
        kategori_encoded = pd.DataFrame(
            mlb.transform([kategori]), columns=mlb.classes_
        )

        # Encoding provinsi
        provinsi_columns = [col for col in features if col.startswith('provinsi_')]
        provinsi_encoded = pd.DataFrame(
            [[1 if provinsi in col else 0 for col in provinsi_columns]],
            columns=provinsi_columns
        )

        # Gabungkan fitur
        input_df = pd.concat([
            pd.DataFrame([[rating, jumlah_review]], columns=['rating', 'jumlah_review']),
            kategori_encoded,
            provinsi_encoded
        ], axis=1)

        # Pastikan urutan kolom sama
        input_df = input_df.reindex(columns=features, fill_value=0)

        prediksi = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

        # Flash pesan ke user
        pesan = f"Tempat ini {'populer' if prediksi else 'tidak populer'} (Probabilitas: {proba:.2f})"
        flash(pesan, 'success' if prediksi else 'danger')
        return redirect(url_for('home'))

    # Contoh kategori dan provinsi, sesuaikan dengan modelmu
    kategori_options = mlb.classes_
    provinsi_options = [col.replace('provinsi_', '') for col in features if col.startswith('provinsi_')]
    return render_template('form.html', kategori_options=kategori_options, provinsi_options=provinsi_options)
    
if __name__ == '__main__':
    app.run(debug=True)