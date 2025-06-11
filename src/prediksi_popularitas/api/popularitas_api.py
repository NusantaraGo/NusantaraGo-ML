from flask import Blueprint, jsonify, request, render_template, flash, redirect, url_for
import joblib
import numpy as np
import pandas as pd
import os
from pathlib import Path

popularitas_bp = Blueprint('popularitas', __name__, url_prefix='/api/popularitas')

# Load model dan komponen pendukung
def load_model_components():
    # Dapatkan path absolut ke folder models
    model_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "prediksi_popularitas")
    
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

@popularitas_bp.route('/', methods=['GET', 'POST'])
def predict_popularity():
    if request.method == 'POST':
        try:
            rating = float(request.form['rating'])
            jumlah_review = int(request.form['jumlah_review'])

            # Buat DataFrame dengan fitur yang dibutuhkan
            input_df = pd.DataFrame([[rating, jumlah_review]], columns=['rating', 'jumlah_review'])
            
            # Tambahkan kolom-kolom lain dengan nilai default 0
            for feature in features:
                if feature not in input_df.columns:
                    input_df[feature] = 0

            # Pastikan urutan kolom sama dengan yang diharapkan model
            input_df = input_df.reindex(columns=features)

            # Lakukan prediksi
            prediksi = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0][1]

            # Flash pesan ke user
            pesan = f"Tempat ini {'populer' if prediksi else 'tidak populer'} (Probabilitas: {proba:.2f})"
            flash(pesan, 'success' if prediksi else 'danger')
            return redirect(url_for('popularitas.predict_popularity'))

        except ValueError as e:
            flash(f"Error: Input tidak valid - {str(e)}", 'danger')
            return redirect(url_for('popularitas.predict_popularity'))
        except Exception as e:
            flash(f"Error: Terjadi kesalahan saat melakukan prediksi - {str(e)}", 'danger')
            return redirect(url_for('popularitas.predict_popularity'))

    return render_template('popularitas.html') 