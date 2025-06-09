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
import json as json_lib

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
DATA_PATH = os.getenv('DATA_PATH', 'data/tempat_wisata_indonesia.csv')

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

def process_attraction_images(attraction_data):
    """
    Memproses gambar untuk satu tempat wisata dari data JSON
    """
    if not attraction_data.get('foto'):
        return None
    
    # Foto disimpan sebagai string path di JSON
    foto = attraction_data['foto']
    if isinstance(foto, str):
        return foto
    
    # Jika foto dalam bentuk list, ambil yang pertama
    if isinstance(foto, list) and len(foto) > 0:
        return foto[0]
    
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
            
            # Proses foto - ambil langsung dari kolom foto
            foto = row.get('foto')
            if isinstance(foto, str):
                processed_foto = foto
            elif isinstance(foto, list) and len(foto) > 0:
                processed_foto = foto[0]
            else:
                processed_foto = None
            
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

    try:
        # Cari tempat wisata berdasarkan nama (case insensitive)
        mask = df['nama'].str.lower() == name.lower()
        if not mask.any():
            return jsonify({"message": f"Tempat wisata '{name}' tidak ditemukan"}), 404

        # Ambil data tempat wisata
        details = df[mask].iloc[0].to_dict()

        # Format koordinat
        koordinat = details.get('koordinat', {})
        if isinstance(koordinat, str):
            try:
                koordinat = json_lib.loads(koordinat)
            except:
                # Coba ekstrak koordinat dari URL jika ada
                url = details.get('url', '')
                if url:
                    try:
                        # Cari pola koordinat di URL (format: !3d{lat}!4d{lon})
                        import re
                        coords = re.findall(r'!3d([\d.-]+)!4d([\d.-]+)', url)
                        if coords:
                            lat, lon = coords[0]
                            koordinat = {
                                "latitude": float(lat),
                                "longitude": float(lon)
                            }
                    except:
                        koordinat = {"latitude": None, "longitude": None}
                else:
                    koordinat = {"latitude": None, "longitude": None}
        elif isinstance(koordinat, dict):
            # Pastikan koordinat dalam format yang benar
            try:
                koordinat = {
                    "latitude": float(koordinat.get('latitude', 0)) if koordinat.get('latitude') is not None else None,
                    "longitude": float(koordinat.get('longitude', 0)) if koordinat.get('longitude') is not None else None
                }
            except:
                # Coba ekstrak dari URL jika konversi gagal
                url = details.get('url', '')
                if url:
                    try:
                        import re
                        coords = re.findall(r'!3d([\d.-]+)!4d([\d.-]+)', url)
                        if coords:
                            lat, lon = coords[0]
                            koordinat = {
                                "latitude": float(lat),
                                "longitude": float(lon)
                            }
                    except:
                        koordinat = {"latitude": None, "longitude": None}
                else:
                    koordinat = {"latitude": None, "longitude": None}

        # Format foto
        foto = details.get('foto')
        if isinstance(foto, str):
            processed_foto = foto
        elif isinstance(foto, list) and len(foto) > 0:
            processed_foto = foto[0]
        else:
            processed_foto = None

        # Format kategori
        kategori = details.get('kategori', [])  # Coba ambil dari 'kategori' dulu
        if not kategori:  # Jika kosong, coba dari 'kategori_list'
            kategori = details.get('kategori_list', [])
        
        if isinstance(kategori, str):
            try:
                # Coba parse sebagai JSON dulu
                kategori = json_lib.loads(kategori)
            except:
                # Jika gagal parse JSON, coba split string dan bersihkan tanda kutip
                try:
                    # Hapus tanda kutip dan kurung siku, lalu split
                    kategori = [k.strip().strip("'").strip('"') for k in kategori.strip('[]').split(',')]
                    # Hapus elemen kosong
                    kategori = [k for k in kategori if k]
                except:
                    kategori = []
        elif isinstance(kategori, list):
            # Bersihkan tanda kutip dari setiap elemen jika ada
            kategori = [k.strip("'").strip('"') if isinstance(k, str) else str(k) for k in kategori]
        else:
            kategori = []

        # Pastikan tidak ada duplikat dan elemen kosong
        kategori = list(dict.fromkeys([k for k in kategori if k]))

        # Format rating
        rating = details.get('rating')
        if isinstance(rating, str):
            try:
                rating = round(float(rating.replace(",", ".")), 1)
            except:
                rating = None
        elif isinstance(rating, (int, float)):
            rating = round(float(rating), 1)
        else:
            rating = None

        # Format jumlah_review
        jumlah_review = details.get('jumlah_review')
        if isinstance(jumlah_review, str):
            try:
                jumlah_review = int(jumlah_review.replace(".", ""))
            except:
                jumlah_review = None
        elif isinstance(jumlah_review, (int, float)):
            jumlah_review = int(jumlah_review)
        else:
            jumlah_review = None

        # Format hasil
        formatted_details = {
            "id": int(details["id"]) if "id" in details and details["id"] is not None else None,
            "nama": details["nama"],
            "rating": rating,
            "jumlah_review": jumlah_review,
            "deskripsi": details.get("deskripsi", ""),
            "koordinat": koordinat,
            "provinsi": details.get("provinsi"),
            "kategori": kategori,
            "foto": processed_foto
        }

        return jsonify(formatted_details)

    except Exception as e:
        logger.error(f"Error saat mendapatkan detail tempat wisata: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"message": f"Error saat mendapatkan detail tempat wisata: {str(e)}"}), 500

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