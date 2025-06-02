import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

from .preprocessing import preprocess_data, calculate_popularity_score

class TourismRecommender:
    """
    Model rekomendasi tempat wisata yang menggabungkan content-based, 
    popularity-based, dan location-based filtering
    """
    
    def __init__(self):
        self.df = None
        self.df_popular = None
        self.cosine_sim = None
        self.indices = None
        self.tfidf_vectorizer = None
        self.C = None
        self.m = None
        self.model_loaded = False
    
    def fit(self, df):
        """
        Melatih model rekomendasi dengan dataset tempat wisata
        """
        # Preprocess data
        self.df = preprocess_data(df)
        
        # Hitung popularity score
        self.df_popular = calculate_popularity_score(self.df)
        
        # Buat TF-IDF Vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df['combined_features'])
        
        # Hitung cosine similarity
        self.cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # Buat indeks berdasarkan nama tempat wisata
        self.indices = pd.Series(self.df.index, index=self.df['nama']).drop_duplicates()
        
        # Simpan parameter untuk popularity score
        self.C = self.df['rating'].mean()
        self.m = self.df['jumlah_review'].quantile(0.90)
        
        self.model_loaded = True
        
        return self
    
    def save_model(self, path="../models/recommendation_model.joblib"):
        """
        Menyimpan model ke file
        """
        if not self.model_loaded:
            raise ValueError("Model belum dilatih, silakan latih model terlebih dahulu dengan metode fit()")
        
        # Buat direktori jika belum ada
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Data model yang akan disimpan
        model_data = {
            'df': self.df,
            'df_popular': self.df_popular,
            'cosine_sim': self.cosine_sim,
            'indices': self.indices,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'C': self.C,
            'm': self.m
        }
        
        # Simpan model
        joblib.dump(model_data, path)
        
        return path
    
    def load_model(self, path="../models/recommendation_model.joblib"):
        """
        Memuat model dari file
        """
        # Load model
        model_data = joblib.load(path)
        
        # Ekstrak komponen model
        self.df = model_data['df']
        self.df_popular = model_data['df_popular']
        self.cosine_sim = model_data['cosine_sim']
        self.indices = model_data['indices']
        self.tfidf_vectorizer = model_data['tfidf_vectorizer']
        self.C = model_data['C']
        self.m = model_data['m']
        
        self.model_loaded = True
        
        return self
    
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        Menghitung jarak antara dua titik koordinat menggunakan formula Haversine
        """
        # Konversi dari derajat ke radian
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Formula Haversine
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371  # Radius bumi dalam kilometer
        return c * r
    
    def content_based_recommendations(self, name, top_n=10):
        """
        Memberikan rekomendasi berdasarkan kemiripan konten
        """
        if not self.model_loaded:
            raise ValueError("Model belum dimuat, silakan muat model terlebih dahulu dengan metode load_model()")
        
        # Dapatkan indeks tempat wisata dari namanya
        try:
            idx = self.indices[name]
        except KeyError:
            return "Tempat wisata tidak ditemukan. Coba nama lain."
        
        # Dapatkan skor kesamaan untuk semua tempat wisata
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        
        # Urutkan tempat wisata berdasarkan skor kesamaan
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Dapatkan skor top_n tempat wisata yang paling mirip (kecuali dirinya sendiri)
        sim_scores = sim_scores[1:top_n+1]
        
        # Dapatkan indeks tempat wisata
        attraction_indices = [i[0] for i in sim_scores]
        
        # Kembalikan top_n tempat wisata yang paling mirip
        recommended_places = self.df.iloc[attraction_indices][['nama', 'provinsi', 'rating', 'jumlah_review', 'kategori_list']]
        recommended_places['similarity_score'] = [i[1] for i in sim_scores]
        
        # Tambahkan kolom id
        recommended_places['id'] = self.df.iloc[attraction_indices]['id']
        
        return recommended_places
    
    def popularity_based_recommendations(self, category=None, province=None, top_n=10):
        """
        Memberikan rekomendasi berdasarkan popularitas
        """
        if not self.model_loaded or self.df_popular is None:
            raise ValueError("Model atau data popularitas belum dimuat. Silakan latih atau muat model terlebih dahulu.")
        
        filtered_df = self.df_popular.copy()
        
        # Filter berdasarkan kategori jika diberikan (case-insensitive)
        if category:
            category_lower = category.lower()
            filtered_df = filtered_df[filtered_df['kategori_list'].apply(lambda x: any(cat.lower() == category_lower for cat in x) if isinstance(x, list) else False)]
        
        # Filter berdasarkan provinsi jika diberikan (case-insensitive)
        if province:
            province_lower = province.lower()
            filtered_df = filtered_df[filtered_df['provinsi'].str.lower() == province_lower]
        
        # --- Hitung ulang popularity_score untuk data yang difilter ---
        # Ini memastikan kolom ada dan skor relevan untuk subset data
        if filtered_df.empty:
             # Jika sudah kosong setelah filtering, tidak perlu hitung skor
             print("Info: filtered_df is empty after initial filtering, skipping popularity score calculation.")
             return "Tidak ada tempat wisata yang cocok dengan kriteria tersebut."

        # Pastikan kolom 'rating' dan 'jumlah_review' ada untuk perhitungan
        if 'rating' not in filtered_df.columns or 'jumlah_review' not in filtered_df.columns:
             print("Critical Error: Rating or review columns missing in filtered data, cannot calculate popularity_score.")
             return "Error internal: Data rating atau review hilang setelah filtering."

        # Gunakan parameter C dan m dari model yang sudah dilatih dari data penuh
        # Ini penting agar skor konsisten dengan definisi popularitas global
        C = self.C
        m = self.m
        
        # Jika m (jumlah review minimum) tidak ada di data yang difilter (kasus ekstrim data sedikit),
        # gunakan m dari model penuh atau fallback ke 0 jika model penuh juga tidak ada m-nya.
        if m is None:
            m = self.df_popular['jumlah_review'].quantile(0.90) if self.df_popular is not None and 'jumlah_review' in self.df_popular.columns else 0

        def weighted_rating_filtered(x, m=m, C=C):
            v = x['jumlah_review']
            R = x['rating']
            # Hindari pembagian oleh nol jika m=0 dan v=0
            if (v + m) == 0:
                return C # Atau nilai default lain yang masuk akal
            return (v/(v+m) * R) + (m/(v+m) * C)

        # Hitung popularity score untuk data yang difilter
        filtered_df['popularity_score'] = filtered_df.apply(weighted_rating_filtered, axis=1)
        # --------------------------------------------------------------

        # Debugging: Log info DataFrame sebelum sorting (setelah hitung ulang skor)
        # print(f"Debug (after recalculate): DataFrame columns before sorting: {filtered_df.columns.tolist()}")
        # print(f"Debug (after recalculate): DataFrame shape before sorting: {filtered_df.shape}")

        # Sekarang kolom popularity_score pasti ada, lakukan sorting
        filtered_df = filtered_df.sort_values('popularity_score', ascending=False)
        
        # Kembalikan top_n tempat wisata yang paling populer
        # Pastikan menyertakan kolom 'id'
        return filtered_df[['id', 'nama', 'provinsi', 'rating', 'jumlah_review', 'kategori_list', 'popularity_score']].head(top_n)
    
    def _validate_coordinates(self, lat, lon):
        """
        Memvalidasi koordinat latitude dan longitude
        """
        if lat is not None and (lat < -90 or lat > 90):
            raise ValueError("Latitude harus berada dalam range -90 sampai 90 derajat")
        if lon is not None and (lon < -180 or lon > 180):
            raise ValueError("Longitude harus berada dalam range -180 sampai 180 derajat")
        return True

    def location_based_recommendations(self, lat, lon, max_distance=50, top_n=10):
        """
        Memberikan rekomendasi berdasarkan lokasi geografis
        """
        if not self.model_loaded:
            raise ValueError("Model belum dimuat, silakan muat model terlebih dahulu dengan metode load_model()")
        
        # Validasi koordinat
        self._validate_coordinates(lat, lon)
        
        # Hapus baris dengan koordinat nan
        filtered_df = self.df.dropna(subset=['latitude', 'longitude']).copy()
        
        # Hitung jarak untuk setiap tempat wisata
        filtered_df['distance'] = filtered_df.apply(
            lambda row: self.haversine_distance(lat, lon, row['latitude'], row['longitude']), axis=1)
        
        # Filter tempat wisata dalam radius max_distance
        filtered_df = filtered_df[filtered_df['distance'] <= max_distance]
        
        # Jika tidak ada hasil yang ditemukan
        if filtered_df.empty:
            return f"Tidak ada tempat wisata dalam radius {max_distance} km dari lokasi tersebut."
        
        # Urutkan berdasarkan jarak
        filtered_df = filtered_df.sort_values('distance')
        
        # Kembalikan top_n tempat wisata terdekat
        # Pastikan menyertakan kolom 'id'
        return filtered_df[['id', 'nama', 'provinsi', 'rating', 'jumlah_review', 'kategori_list', 'distance']].head(top_n)
    
    def hybrid_recommendations(self, name=None, lat=None, lon=None, category=None, province=None, max_distance=50, top_n=10):
        """
        Memberikan rekomendasi hybrid yang menggabungkan content-based, popularity-based, dan location-based
        """
        if not self.model_loaded:
            raise ValueError("Model belum dimuat, silakan muat model terlebih dahulu dengan metode load_model()")
        
        # Validasi koordinat jika diberikan
        if lat is not None or lon is not None:
            self._validate_coordinates(lat, lon)
        
        results = pd.DataFrame()
        
        # Content-based jika nama tempat wisata diberikan
        if name:
            content_recs = self.content_based_recommendations(name, top_n=top_n*2)
            if isinstance(content_recs, str):
                print(f"Warning: Content-based recommendation failed for '{name}': {content_recs}")
                pass  # Jika tempat wisata tidak ditemukan
            else:
                # Pastikan kolom yang diperlukan ada sebelum digabungkan
                required_cols = ['id', 'nama', 'provinsi', 'rating', 'jumlah_review', 'kategori_list', 'similarity_score']
                if all(col in content_recs.columns for col in required_cols):
                    results = pd.concat([results, content_recs])
                else:
                    print(f"Warning: Missing required columns in content_recs: {', '.join([col for col in required_cols if col not in content_recs.columns])}")

        
        # Location-based jika koordinat diberikan
        if lat is not None and lon is not None:
            location_recs = self.location_based_recommendations(lat, lon, max_distance, top_n*2)
            if isinstance(location_recs, str):
                 print(f"Warning: Location-based recommendation failed for ({lat}, {lon}): {location_recs}")
                 pass  # Jika tidak ada tempat wisata dalam radius
            else:
                # Tambahkan skor jarak yang dinormalisasi (semakin dekat semakin tinggi skor)
                if 'distance' in location_recs.columns:
                    max_dist = location_recs['distance'].max() if not location_recs.empty else 1
                    location_recs['distance_score'] = 1 - (location_recs['distance'] / max_dist)
                    # Pastikan kolom yang diperlukan ada sebelum digabungkan
                    required_cols = ['id', 'nama', 'provinsi', 'rating', 'jumlah_review', 'kategori_list', 'distance', 'distance_score']
                    if all(col in location_recs.columns for col in required_cols):
                         results = pd.concat([results, location_recs])
                    else:
                         print(f"Warning: Missing required columns in location_recs: {', '.join([col for col in required_cols if col not in location_recs.columns])}")
                else:
                     print("Warning: 'distance' column missing in location_recs.")

        # Popularity-based berdasarkan kategori dan/atau provinsi
        # Panggil popularity_based_recommendations, yang sekarang sudah lebih defensif
        popularity_recs = self.popularity_based_recommendations(category=category, province=province, top_n=top_n*2) # Kirim parameter explicitly
        
        # popularity_based_recommendations sekarang mengembalikan string error atau DataFrame
        if isinstance(popularity_recs, str):
             print(f"Warning: Popularity-based recommendation failed for (cat={category}, prov={province}): {popularity_recs}")
             # Jika ada error dari popularity_based_recommendations (misal: tidak ada data atau error internal saat hitung skor)
             # Kita bisa memilih untuk mengabaikan komponen popularitas atau mengembalikan error keseluruhan
             # Untuk hybrid, mari kita coba abaikan komponen ini jika gagal, tetapi log peringatan.
             # Jangan tambahkan ke `results` jika berupa string error
             pass 
        else:
            # Pastikan kolom yang diperlukan ada sebelum digabungkan (termasuk popularity_score)
            required_cols = ['id', 'nama', 'provinsi', 'rating', 'jumlah_review', 'kategori_list', 'popularity_score']
            if all(col in popularity_recs.columns for col in required_cols):
                results = pd.concat([results, popularity_recs])
            else:
                print(f"Warning: Missing required columns in popularity_recs after getting results: {', '.join([col for col in required_cols if col not in popularity_recs.columns])}")

        # Jika tidak ada hasil
        if results.empty:
            return "Tidak ada rekomendasi yang sesuai dengan kriteria tersebut."
        
        # Normalkan semua skor ke range 0-1
        scaler = MinMaxScaler()
        if 'similarity_score' in results.columns:
            results['similarity_score'] = results['similarity_score'].fillna(0)
            results['similarity_score_normalized'] = scaler.fit_transform(results[['similarity_score']])
        else:
            results['similarity_score_normalized'] = 0
            
        if 'distance_score' in results.columns:
            results['distance_score'] = results['distance_score'].fillna(0)
            results['distance_score_normalized'] = scaler.fit_transform(results[['distance_score']])
        else:
            results['distance_score_normalized'] = 0
            
        if 'popularity_score' in results.columns:
            results['popularity_score'] = results['popularity_score'].fillna(0)
            results['popularity_score_normalized'] = scaler.fit_transform(results[['popularity_score']])
        else:
            results['popularity_score_normalized'] = 0
        
        # Hitung hybrid score dengan bobot berbeda untuk setiap komponen
        results['hybrid_score'] = (
            0.4 * results['similarity_score_normalized'] + 
            0.3 * results['distance_score_normalized'] + 
            0.3 * results['popularity_score_normalized']
        )
        
        # Hapus duplikat berdasarkan nama
        results = results.drop_duplicates(subset=['nama'])
        
        # Urutkan berdasarkan hybrid score
        results = results.sort_values('hybrid_score', ascending=False)
        
        # Pilih kolom-kolom yang ingin ditampilkan
        # Pastikan menyertakan 'id' dalam daftar kolom yang ditampilkan
        display_columns = ['id', 'nama', 'provinsi', 'rating', 'jumlah_review', 'kategori_list', 'hybrid_score']
        if 'distance' in results.columns:
            display_columns.insert(6, 'distance') # Sisipkan jarak setelah kategori_list
        
        # Kembalikan top_n rekomendasi
        return results[display_columns].head(top_n) 