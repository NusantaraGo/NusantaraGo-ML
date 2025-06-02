from flask import jsonify, request, render_template, Flask
import joblib
import numpy as np
import pandas as pd
import os
from pathlib import Path

app = Flask(__name__)

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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Terima input
        data = request.form.to_dict()

        # Proses input kategori lebih aman (ganti eval dengan ast.literal_eval)
        import ast
        kategori = ast.literal_eval(data['kategori']) if isinstance(data['kategori'], str) else data['kategori']
        
        # Persiapan data 
        input_data = {
            'rating': float(data['rating']),
            'jumlah_review': int(data['jumlah_review']),
            'kategori': kategori,
            'provinsi': data['provinsi']
        }

        # Buat dataframe dari input
        df_input = pd.DataFrame([input_data])

        # Feature engineering
        df_input['log_jumlah_review'] = np.log1p(df_input['jumlah_review'])

        # Encoding kategori (multi-label)
        # kategori_encoded = mlb.transform(df_input['kategori'])
        kategori_encoded = mlb.transform([df_input.at[0, 'kategori']])
        kategori_encoded_df = pd.DataFrame(kategori_encoded, columns=mlb.classes_)
        
        # Encoding provinsi (one-hot)
        provinsi_encoded = pd.get_dummies(df_input['provinsi'], prefix='provinsi')

        # Gabungkan semua fitur
        X = pd.concat([
            df_input[['rating', 'log_jumlah_review']],
            kategori_encoded_df,
            provinsi_encoded.reindex(columns=[f for f in features if f.startswith('provinsi_')])
        ], axis=1)

        # Urutkan kolom sesuai dengan training
        X = X.reindex(columns=features, fill_value=0)
        
        # Scaling
        X_scaled = scaler.transform(X)
        
        # Prediksi
        prediksi = model.predict(X_scaled)
        proba = model.predict_proba(X_scaled)[0][1]  # Probabilitas kelas populer


        # Format hasil
        hasil = {
            'prediksi': 'populer' if prediksi[0] == 1 else 'tidak populer',
            'confidence': round(proba * 100, 2),
            'status': 'success'
        }

        return jsonify(hasil)
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'details': "Pastikan input sesuai format. Contoh kategori: ['Pantai', 'Gunung']"
        }), 400



if __name__ == "__main__":
    app.run(debug=True)