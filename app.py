import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Menyembunyikan pesan INFO dan WARNING TensorFlow

"""
Aplikasi Flask untuk sistem rekomendasi tempat wisata dan chatbot
"""
from flask import json, Flask, request, jsonify, render_template, redirect, url_for
from flask_cors import CORS
import ast
from dotenv import load_dotenv
import sys
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
import traceback
import requests
from PIL import Image
from io import BytesIO
import hashlib

# Tambahkan path untuk import modul
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.recommender import (
    TourismRecommender, 
    load_csv_data, 
    get_available_categories, 
    get_available_provinces,
    format_recommendation_results,
    get_attraction_details,
    filter_attractions,
    calculate_popularity_score
)

# Import chatbot api blueprint
from src.chatbot.api.chatbot_api import chatbot_bp

# Load environment variables
load_dotenv()

# Konfigurasi logging
def setup_logger():
    """
    Mengatur logger untuk aplikasi
    """
    # Buat direktori logs jika belum ada
    os.makedirs('logs', exist_ok=True)
    
    # Konfigurasi logger
    logger = logging.getLogger('nusantarago')
    logger.setLevel(logging.INFO)
    
    # Format log
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Handler untuk file log (rotating file handler)
    file_handler = RotatingFileHandler(
        'logs/nusantarago.log',
        maxBytes=1024 * 1024,  # 1MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Handler untuk console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

# Inisialisasi logger
logger = setup_logger()

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Register chatbot blueprint
app.register_blueprint(chatbot_bp, url_prefix='/api/chatbot')
CORS(app)

# Inisialisasi model rekomendasi
MODEL_PATH = os.getenv('MODEL_PATH', 'models/recommendation_model.joblib')
DATA_PATH = os.getenv('DATA_PATH', 'Scrape_Data/tempat_wisata_indonesia.csv')

# Global variables untuk menyimpan model dan data rekomendasi
recommender = None
df = None
categories = None
provinces = None

def load_model_and_data():
    """
    Memuat model rekomendasi dan data wisata
    """
    global recommender, df, categories, provinces

    recommender = TourismRecommender()

    # Coba muat model terlebih dahulu
    try:
        recommender.load_model(MODEL_PATH)
        logger.info(f"Model rekomendasi berhasil dimuat dari {MODEL_PATH}")
        
        # Gunakan df dan df_popular dari model yang dimuat
        df = recommender.df
        # Pastikan df_popular juga dimuat
        if recommender.df_popular is None:
            logger.warning("df_popular tidak ditemukan dalam model yang dimuat. Menghitung ulang...")
            # Jika df_popular tidak ada di model yang dimuat, hitung ulang dari df yang ada di model
            recommender.df_popular = calculate_popularity_score(df)
            logger.info("df_popular berhasil dihitung ulang.")
            
        categories = get_available_categories(df)
        provinces = get_available_provinces(df)
        logger.info(f"Data dari model dimuat. Jumlah data: {len(df)}")
        return True # Berhasil memuat model dan data dari file

    except FileNotFoundError:
        logger.warning(f"File model rekomendasi tidak ditemukan di {MODEL_PATH}. Memproses data mentah dan melatih model baru...")
        # Jika file model tidak ada, muat data mentah, proses, latih, dan simpan
        try:
            df_raw = load_csv_data(DATA_PATH)
            recommender.fit(df_raw) # Melatih model juga mengisi recommender.df dan recommender.df_popular
            recommender.save_model(MODEL_PATH)
            
            # Gunakan df dan df_popular dari model yang baru dilatih
            df = recommender.df
            categories = get_available_categories(df)
            provinces = get_available_provinces(df)
            logger.info(f"Model baru berhasil dilatih dan disimpan ke {MODEL_PATH}")
            logger.info(f"Data dari pelatihan digunakan. Jumlah data: {len(df)}")
            return True # Berhasil melatih model baru

        except Exception as e_train:
            logger.error(f"Error saat memuat data mentah atau melatih model baru: {str(e_train)}\n{traceback.format_exc()}")
            recommender = None # Set recommender ke None jika pelatihan gagal
            df = None
            categories = None
            provinces = None
            return False # Gagal melatih model baru

    except Exception as e_load:
        logger.error(f"Error saat memuat model rekomendasi dari {MODEL_PATH}: {str(e_load)}\n{traceback.format_exc()}")
        # Jika ada error lain saat memuat file model, coba proses data mentah sebagai fallback
        logger.warning("Mencoba memproses data mentah sebagai fallback...")
        try:
            df_raw = load_csv_data(DATA_PATH)
            recommender.fit(df_raw) # Melatih model juga mengisi recommender.df dan recommender.df_popular
            # Tidak perlu menyimpan model jika proses fallback ini terjadi karena mungkin ada isu dengan I/O
            
            # Gunakan df dan df_popular dari model yang diproses fallback
            df = recommender.df
            categories = get_available_categories(df)
            provinces = get_available_provinces(df)
            logger.info(f"Data diproses ulang dari mentah sebagai fallback. Jumlah data: {len(df)}")
            return True # Berhasil memproses data mentah sebagai fallback
            
        except Exception as e_fallback:
             logger.error(f"Error saat memproses data mentah sebagai fallback: {str(e_fallback)}\n{traceback.format_exc()}")
             recommender = None
             df = None
             categories = None
             provinces = None
             return False # Gagal total

    # Ini seharusnya tidak tercapai, tapi sebagai safety net
    logger.error("load_model_and_data mencapai akhir tanpa hasil yang jelas.")
    return False

@app.route('/')
def index():
    """
    Halaman utama - Selamat datang
    """
    return render_template('home.html') # Render home.html

@app.route('/dokumentasi')
def dokumentasi_api():
    """
    Halaman dokumentasi API
    """
    return render_template('dokumentasi.html') # Render dokumentasi.html

@app.route('/chatbot')
def chatbot_page():
    """
    Halaman chatbot
    """
    # Asumsikan chatbot_enabled selalu True jika IntentChatbot berhasil dimuat
    return render_template('chatbot.html')

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

def download_and_save_image(image_url, attraction_name):
    """
    Mendownload dan menyimpan gambar dari URL ke folder static/images
    """
    try:
        # Buat nama file yang unik dari URL
        file_hash = hashlib.md5(image_url.encode()).hexdigest()[:10]
        safe_name = "".join(c for c in attraction_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"{safe_name}_{file_hash}.jpg"
        
        # Buat direktori jika belum ada
        image_dir = os.path.join('static', 'images', 'attractions')
        os.makedirs(image_dir, exist_ok=True)
        
        filepath = os.path.join(image_dir, filename)
        
        # Jika file sudah ada, langsung return path-nya
        if os.path.exists(filepath):
            return f"/static/images/attractions/{filename}"
        
        # Download gambar
        response = requests.get(image_url, timeout=10)
        if response.status_code == 200:
            # Buka gambar dengan PIL untuk validasi
            img = Image.open(BytesIO(response.content))
            
            # Konversi ke RGB jika perlu
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            # Simpan gambar
            img.save(filepath, 'JPEG', quality=85, optimize=True)
            return f"/static/images/attractions/{filename}"
        
        return None
    except Exception as e:
        print(f"Error downloading image {image_url}: {str(e)}")
        return None

def process_attraction_images(attraction_data):
    """
    Memproses gambar untuk satu tempat wisata, hanya mengambil 1 foto yang paling relevan
    """
    if not attraction_data.get('foto'):
        return None
    
    # Ambil foto pertama saja (biasanya foto utama/cover)
    image_url = attraction_data['foto'][0] if isinstance(attraction_data['foto'], list) else None
    
    if image_url and isinstance(image_url, str):
        local_path = download_and_save_image(image_url, attraction_data['nama'])
        return local_path
    
    return None

@app.route('/api/attractions')
def get_attractions():
    """
    Mendapatkan daftar tempat wisata dengan filter
    """
    global df # Pastikan menggunakan df global yang sudah diproses

    if df is None:
         # Coba muat ulang data jika belum ada (misal setelah hot-reload atau error sebelumnya)
        if not load_model_and_data():
             return jsonify({"message": "Gagal memuat data tempat wisata."}), 500
    
    category = request.args.get('category')
    province = request.args.get('province')
    min_rating = request.args.get('min_rating')
    if min_rating:
        try:
            min_rating = float(min_rating)
        except ValueError:
            return jsonify({"message": "Nilai min_rating harus berupa angka."}), 400

    search_query = request.args.get('q')

    # Terapkan filter pada df yang sudah diproses
    filtered_df = filter_attractions(df, category, province, min_rating, search_query=search_query)

    # Urutkan berdasarkan rating
    if 'rating' in filtered_df.columns:
        filtered_df = filtered_df.sort_values('rating', ascending=False)
    else:
         print("Peringatan: Kolom 'rating' tidak ditemukan untuk sorting di /api/attractions.")

    # Hapus batasan limit default 50
    limit = request.args.get('limit', type=int)
    if limit:
        filtered_df = filtered_df.head(limit)

    # Format hasil
    results = []
    for _, row in filtered_df.iterrows():
        try:
            # Pastikan deskripsi selalu ada
            deskripsi = row.get("deskripsi", "")
            if not deskripsi or deskripsi == "N/A":
                deskripsi = f"{row['nama']} adalah sebuah tempat wisata yang terletak di {row['alamat']}. Tempat ini menawarkan berbagai pengalaman menarik bagi pengunjung."
            
            # Pastikan koordinat dalam format yang benar
            koordinat = row.get('koordinat', {})
            if isinstance(koordinat, str):
                try:
                    koordinat = ast.literal_eval(koordinat)
                except:
                    koordinat = {"latitude": None, "longitude": None}
            
            # Proses foto - hanya ambil 1 foto
            foto = row.get('foto', [])
            if isinstance(foto, str):
                try:
                    foto = ast.literal_eval(foto)
                except:
                    foto = []
            
            # Download dan simpan foto (hanya 1 foto)
            processed_foto = process_attraction_images({
                'nama': row['nama'],
                'foto': foto
            })
            
            # Pastikan kategori dalam format yang benar
            kategori = row.get('kategori_list', [])
            if isinstance(kategori, str):
                try:
                    kategori = ast.literal_eval(kategori)
                except:
                    kategori = []

            # Handle jumlah_review dengan lebih baik
            jumlah_review = row.get('jumlah_review')
            if jumlah_review is not None and jumlah_review != "N/A":
                try:
                    if isinstance(jumlah_review, str):
                        # Jika string, hapus titik dan konversi ke int
                        jumlah_review = int(jumlah_review.replace(".", ""))
                    elif isinstance(jumlah_review, (int, float)):
                        # Jika sudah angka, langsung konversi ke int
                        jumlah_review = int(jumlah_review)
                    else:
                        jumlah_review = None
                except (ValueError, TypeError):
                    jumlah_review = None
            else:
                jumlah_review = None

            # Handle rating dengan lebih baik
            rating = row.get('rating')
            if rating is not None and rating != "N/A":
                try:
                    if isinstance(rating, str):
                        rating = round(float(rating.replace(",", ".")), 1)
                    elif isinstance(rating, (int, float)):
                        rating = round(float(rating), 1)
                    else:
                        rating = None
                except (ValueError, TypeError):
                    rating = None
            else:
                rating = None

            results.append({
                "id": int(row["id"]) if "id" in row and row["id"] is not None else None,
                "nama": row["nama"],
                "deskripsi": deskripsi,
                "provinsi": row["provinsi"],
                "rating": rating,
                "jumlah_review": jumlah_review,
                "foto": processed_foto,
                "koordinat": koordinat,
                "kategori": kategori
            })
        except Exception as e:
            print(f"Error processing row for {row.get('nama', 'Unknown')}: {str(e)}")
            continue

    return jsonify(results)

@app.route('/api/attraction/<name>')
def get_attraction(name):
    """
    Mendapatkan detail tempat wisata berdasarkan nama (pencarian fleksibel)
    """
    global df # Pastikan menggunakan df global yang sudah diproses

    if df is None:
         # Coba muat ulang data jika belum ada
        if not load_model_and_data():
             return jsonify({"message": "Gagal memuat data tempat wisata."}), 500

    # Gunakan fungsi pencarian detail yang lebih fleksibel
    # Asumsikan get_attraction_details di src/recommender/__init__.py
    # sudah diimplementasikan untuk pencarian fleksibel
    details = get_attraction_details(df, name)

    # Periksa jika hasil adalah error dictionary dari get_attraction_details
    if isinstance(details, dict) and "error" in details:
        # Log error jika bukan 'tidak ditemukan'
        if "tidak ditemukan" not in details["error"]:
             logger.error(f"Error dari get_attraction_details untuk '{name}': {details['error']}")
        return jsonify({"message": details["error"]}), 404 # Kembalikan 404 jika tidak ditemukan atau error spesifik

    # Format hasil sebelum dikirim
    try:
        formatted_details = {
            "id": int(details["id"]) if "id" in details else None,
            "nama": details["nama"],
            "alamat": details["alamat"],
            "rating": round(details["rating"], 1) if "rating" in details else None,
            "jumlah_review": int(details["jumlah_review"]) if "jumlah_review" in details else None,
            "deskripsi": details.get("deskripsi"), # Gunakan .get() untuk keamanan
            "koordinat": {
                "lat": details.get("latitude"), # Akses latitude dan longitude langsung
                "lon": details.get("longitude")
            },
            "url": details.get("url"),
            "provinsi": details.get("provinsi"),
            # Mengambil kategori dari 'kategori_list'
            "kategori": details["kategori_list"] if "kategori_list" in details else None,
            "foto": details["foto"] # Asumsikan foto sudah berupa list string di df
        }
        return jsonify(formatted_details)
    except Exception as e:
        logger.error(f"Error saat memformat hasil get_attraction_details untuk '{name}': {str(e)}\n{traceback.format_exc()}")
        return jsonify({"message": f"Terjadi kesalahan: {str(e)}"}), 500

@app.route('/api/recommendations/content')
def content_recommendations():
    """
    Mendapatkan rekomendasi berdasarkan content
    """
    try:
        name = request.args.get('name')
        if not name:
            logger.warning("Parameter 'name' tidak diberikan pada endpoint content_recommendations")
            return jsonify({"message": "Parameter 'name' harus diberikan"}), 400
        
        top_n = request.args.get('limit', 10, type=int)
        logger.info(f"Meminta rekomendasi content-based untuk '{name}' dengan limit {top_n}")
        
        recommendations = recommender.content_based_recommendations(name, top_n=top_n)
        results = format_recommendation_results(recommendations)
        
        logger.info(f"Berhasil memberikan {len(results)} rekomendasi content-based untuk '{name}'")
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error pada endpoint content_recommendations: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"message": f"Terjadi kesalahan: {str(e)}"}), 500

@app.route('/api/recommendations/popularity')
def popularity_recommendations():
    """
    Mendapatkan rekomendasi berdasarkan popularitas
    """
    try:
        category = request.args.get('category')
        province = request.args.get('province')
        top_n = request.args.get('limit', 10, type=int)
        
        logger.info(f"Meminta rekomendasi popularity-based (category={category}, province={province}, limit={top_n})")
        
        recommendations = recommender.popularity_based_recommendations(category, province, top_n=top_n)
        results = format_recommendation_results(recommendations)
        
        logger.info(f"Berhasil memberikan {len(results)} rekomendasi popularity-based")
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error pada endpoint popularity_recommendations: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"message": f"Terjadi kesalahan: {str(e)}"}), 500

@app.route('/api/recommendations/location')
def location_recommendations():
    """
    Mendapatkan rekomendasi berdasarkan lokasi
    """
    try:
        lat = request.args.get('lat', type=float)
        lon = request.args.get('lon', type=float)
        
        if lat is None or lon is None:
            logger.warning("Parameter 'lat' atau 'lon' tidak diberikan pada endpoint location_recommendations")
            return jsonify({"message": "Parameter 'lat' dan 'lon' harus diberikan"}), 400
        
        max_distance = request.args.get('max_distance', 50, type=int)
        top_n = request.args.get('limit', 10, type=int)
        
        logger.info(f"Meminta rekomendasi location-based untuk koordinat ({lat}, {lon}) dengan max_distance={max_distance}, limit={top_n}")
        
        recommendations = recommender.location_based_recommendations(lat, lon, max_distance, top_n=top_n)
        results = format_recommendation_results(recommendations)
        
        logger.info(f"Berhasil memberikan {len(results)} rekomendasi location-based")
        return jsonify(results)
    except ValueError as ve:
        logger.warning(f"Parameter tidak valid pada endpoint location_recommendations: {str(ve)}")
        return jsonify({"message": str(ve)}), 400
    except Exception as e:
        logger.error(f"Error pada endpoint location_recommendations: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"message": f"Terjadi kesalahan: {str(e)}"}), 500

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
        
        logger.info(
            f"Meminta rekomendasi hybrid (name={name}, lat={lat}, lon={lon}, "
            f"category={category}, province={province}, max_distance={max_distance}, limit={top_n})"
        )
        
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
        
        logger.info(f"Berhasil memberikan {len(results)} rekomendasi hybrid")
        return jsonify(results)
    except ValueError as ve:
        logger.warning(f"Parameter tidak valid pada endpoint hybrid_recommendations: {str(ve)}")
        return jsonify({"message": str(ve)}), 400
    except Exception as e:
        logger.error(f"Error pada endpoint hybrid_recommendations: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"message": f"Terjadi kesalahan: {str(e)}"}), 500

if __name__ == '__main__':
    # Muat model dan data rekomendasi
    try:
        if load_model_and_data():
            logger.info("Model dan data berhasil dimuat. Aplikasi siap dijalankan!")
            # Jalankan aplikasi
            debug_mode = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
            port = int(os.getenv('PORT', 5000))
            app.run(debug=debug_mode, host='0.0.0.0', port=port)
        else:
            logger.error("Gagal memuat model atau data. Aplikasi tidak dapat dijalankan.")
    except Exception as e:
        logger.critical(f"Error fatal saat menjalankan aplikasi: {str(e)}\n{traceback.format_exc()}")
        sys.exit(1) 