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
        
        return recommended_places
    
    def popularity_based_recommendations(self, category=None, province=None, top_n=10):
        """
        Memberikan rekomendasi berdasarkan popularitas
        """
        if not self.model_loaded:
            raise ValueError("Model belum dimuat, silakan muat model terlebih dahulu dengan metode load_model()")
        
        filtered_df = self.df_popular.copy()
        
        # Filter berdasarkan kategori jika diberikan
        if category:
            filtered_df = filtered_df[filtered_df['kategori_list'].apply(lambda x: category in x)]
        
        # Filter berdasarkan provinsi jika diberikan
        if province:
            filtered_df = filtered_df[filtered_df['provinsi'] == province]
        
        # Jika tidak ada hasil yang ditemukan setelah filtering
        if filtered_df.empty:
            return "Tidak ada tempat wisata yang cocok dengan kriteria tersebut."
        
        # Urutkan berdasarkan skor popularitas
        filtered_df = filtered_df.sort_values('popularity_score', ascending=False)
        
        # Kembalikan top_n tempat wisata yang paling populer
        return filtered_df[['nama', 'provinsi', 'rating', 'jumlah_review', 'kategori_list', 'popularity_score']].head(top_n)
    
    def location_based_recommendations(self, lat, lon, max_distance=50, top_n=10):
        """
        Memberikan rekomendasi berdasarkan lokasi geografis
        """
        if not self.model_loaded:
            raise ValueError("Model belum dimuat, silakan muat model terlebih dahulu dengan metode load_model()")
        
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
        return filtered_df[['nama', 'provinsi', 'rating', 'jumlah_review', 'kategori_list', 'distance']].head(top_n)
    
    def hybrid_recommendations(self, name=None, lat=None, lon=None, category=None, province=None, max_distance=50, top_n=10):
        """
        Memberikan rekomendasi hybrid yang menggabungkan content-based, popularity-based, dan location-based
        """
        if not self.model_loaded:
            raise ValueError("Model belum dimuat, silakan muat model terlebih dahulu dengan metode load_model()")
        
        results = pd.DataFrame()
        
        # Content-based jika nama tempat wisata diberikan
        if name:
            content_recs = self.content_based_recommendations(name, top_n=top_n*2)
            if isinstance(content_recs, str):
                pass  # Jika tempat wisata tidak ditemukan
            else:
                results = pd.concat([results, content_recs])
        
        # Location-based jika koordinat diberikan
        if lat is not None and lon is not None:
            location_recs = self.location_based_recommendations(lat, lon, max_distance, top_n*2)
            if isinstance(location_recs, str):
                pass  # Jika tidak ada tempat wisata dalam radius
            else:
                # Tambahkan skor jarak yang dinormalisasi (semakin dekat semakin tinggi skor)
                max_dist = location_recs['distance'].max() if not location_recs.empty else 1
                location_recs['distance_score'] = 1 - (location_recs['distance'] / max_dist)
                results = pd.concat([results, location_recs])
        
        # Popularity-based berdasarkan kategori dan/atau provinsi
        popularity_recs = self.popularity_based_recommendations(category, province, top_n*2)
        if isinstance(popularity_recs, str):
            pass  # Jika tidak ada tempat wisata yang cocok
        else:
            results = pd.concat([results, popularity_recs])
        
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
        display_columns = ['nama', 'provinsi', 'rating', 'jumlah_review', 'kategori_list', 'hybrid_score']
        if 'distance' in results.columns:
            display_columns.insert(5, 'distance')
        
        # Kembalikan top_n rekomendasi
        return results[display_columns].head(top_n) 