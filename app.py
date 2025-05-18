"""
Aplikasi Flask untuk sistem rekomendasi tempat wisata
"""
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import os

from src.recommender import (
    TourismRecommender, 
    load_csv_data, 
    get_available_categories, 
    get_available_provinces,
    format_recommendation_results,
    get_attraction_details,
    filter_attractions
)

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Inisialisasi model
MODEL_PATH = os.getenv('MODEL_PATH', 'models/recommendation_model.joblib')
DATA_PATH = os.getenv('DATA_PATH', 'Scrape_Data/tempat_wisata_indonesia.csv')

# Global variables untuk menyimpan model dan data
recommender = None
df = None
categories = None
provinces = None

def load_model_and_data():
    """
    Memuat model dan data
    """
    global recommender, df, categories, provinces
    
    # Muat model rekomendasi
    try:
        recommender = TourismRecommender()
        recommender.load_model(MODEL_PATH)
        print(f"Model berhasil dimuat dari {MODEL_PATH}")
    except Exception as e:
        print(f"Error saat memuat model: {str(e)}")
        print("Model belum ada. Coba latih model terlebih dahulu dengan menjalankan train_model.py")
        return False
    
    # Muat data CSV
    try:
        df = load_csv_data(DATA_PATH)
        categories = get_available_categories(df)
        provinces = get_available_provinces(df)
        print(f"Data berhasil dimuat dari {DATA_PATH}")
        print(f"Jumlah data: {len(df)}")
        print(f"Jumlah kategori: {len(categories)}")
        print(f"Jumlah provinsi: {len(provinces)}")
        return True
    except Exception as e:
        print(f"Error saat memuat data CSV: {str(e)}")
        return False

@app.route('/')
def index():
    """
    Halaman utama
    """
    return render_template('index.html', categories=categories, provinces=provinces)

@app.route('/api/provinces')
def get_provinces():
    """
    Mendapatkan daftar provinsi
    """
    return jsonify(provinces)

@app.route('/api/categories')
def get_categories():
    """
    Mendapatkan daftar kategori
    """
    return jsonify(categories)

@app.route('/api/attractions')
def get_attractions():
    """
    Mendapatkan daftar tempat wisata dengan filter
    """
    category = request.args.get('category')
    province = request.args.get('province')
    min_rating = request.args.get('min_rating')
    if min_rating:
        min_rating = float(min_rating)
    search_query = request.args.get('q')
    
    # Terapkan filter
    filtered_df = filter_attractions(df, category, province, min_rating, search_query=search_query)
    
    # Urutkan berdasarkan rating
    filtered_df = filtered_df.sort_values('rating', ascending=False)
    
    # Batasi jumlah hasil
    limit = request.args.get('limit', 50, type=int)
    filtered_df = filtered_df.head(limit)
    
    # Format hasil
    results = []
    for _, row in filtered_df.iterrows():
        results.append({
            "id": int(row["id"]) if "id" in row else None,
            "nama": row["nama"],
            "provinsi": row["provinsi"],
            "rating": round(row["rating"], 1) if "rating" in row else None,
            "jumlah_review": int(row["jumlah_review"]) if "jumlah_review" in row else None,
            "kategori": row["kategori_list"] if "kategori_list" in row else None
        })
    
    return jsonify(results)

@app.route('/api/attraction/<name>')
def get_attraction(name):
    """
    Mendapatkan detail tempat wisata berdasarkan nama
    """
    details = get_attraction_details(df, name)
    return jsonify(details)

@app.route('/api/recommendations/content')
def content_recommendations():
    """
    Mendapatkan rekomendasi berdasarkan content
    """
    name = request.args.get('name')
    if not name:
        return jsonify({"error": "Parameter 'name' harus diberikan"})
    
    top_n = request.args.get('limit', 10, type=int)
    
    recommendations = recommender.content_based_recommendations(name, top_n=top_n)
    results = format_recommendation_results(recommendations)
    
    return jsonify(results)

@app.route('/api/recommendations/popularity')
def popularity_recommendations():
    """
    Mendapatkan rekomendasi berdasarkan popularitas
    """
    category = request.args.get('category')
    province = request.args.get('province')
    top_n = request.args.get('limit', 10, type=int)
    
    recommendations = recommender.popularity_based_recommendations(category, province, top_n=top_n)
    results = format_recommendation_results(recommendations)
    
    return jsonify(results)

@app.route('/api/recommendations/location')
def location_recommendations():
    """
    Mendapatkan rekomendasi berdasarkan lokasi
    """
    try:
        lat = request.args.get('lat', type=float)
        lon = request.args.get('lon', type=float)
        
        if lat is None or lon is None:
            return jsonify({"error": "Parameter 'lat' dan 'lon' harus diberikan"})
        
        max_distance = request.args.get('max_distance', 50, type=int)
        top_n = request.args.get('limit', 10, type=int)
        
        recommendations = recommender.location_based_recommendations(lat, lon, max_distance, top_n=top_n)
        results = format_recommendation_results(recommendations)
        
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"})

@app.route('/api/recommendations/hybrid')
def hybrid_recommendations():
    """
    Mendapatkan rekomendasi hybrid
    """
    try:
        name = request.args.get('name')
        
        lat = request.args.get('lat', type=float)
        lon = request.args.get('lon', type=float)
        
        category = request.args.get('category')
        province = request.args.get('province')
        
        max_distance = request.args.get('max_distance', 50, type=int)
        top_n = request.args.get('limit', 10, type=int)
        
        recommendations = recommender.hybrid_recommendations(
            name=name, 
            lat=lat, 
            lon=lon, 
            category=category, 
            province=province, 
            max_distance=max_distance, 
            top_n=top_n
        )
        
        results = format_recommendation_results(recommendations)
        
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"})

if __name__ == '__main__':
    # Muat model dan data
    if load_model_and_data():
        print("Aplikasi siap dijalankan!")
        # Jalankan aplikasi
        debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
        port = int(os.getenv('PORT', 5000))
        app.run(debug=debug_mode, host='0.0.0.0', port=port)
    else:
        print("Gagal memuat model atau data. Aplikasi tidak dapat dijalankan.") 