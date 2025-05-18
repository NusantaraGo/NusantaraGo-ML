import pandas as pd
import numpy as np
import re
import ast
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk

# Download NLTK resources jika belum ada
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

def preprocess_text(text):
    """
    Melakukan preprocessing pada teks deskripsi
    """
    if pd.isna(text):
        return ""
    
    # Konversi ke lowercase
    text = text.lower()
    
    # Hapus karakter khusus dan angka
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Tokenisasi
    tokens = word_tokenize(text)
    
    # Hapus stopwords
    try:
        stop_words = set(stopwords.words('indonesian'))
    except:
        stop_words = set(stopwords.words('english'))
        
    # Tambahkan stopwords kustom
    custom_stopwords = ['yang', 'ini', 'dan', 'di', 'dengan', 'untuk', 'dari', 'pada', 'ke', 'adalah']
    stop_words.update(custom_stopwords)
    
    tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    return " ".join(tokens)

def extract_coordinates(coord_str):
    """
    Mengekstrak koordinat latitude dan longitude dari string koordinat
    """
    if pd.isna(coord_str):
        return np.nan, np.nan
    try:
        coord_dict = ast.literal_eval(coord_str) if isinstance(coord_str, str) else coord_str
        return coord_dict.get('latitude', np.nan), coord_dict.get('longitude', np.nan)
    except:
        return np.nan, np.nan

def preprocess_data(df):
    """
    Melakukan preprocessing pada dataset tempat wisata
    """
    # Buat salinan dataframe
    processed_df = df.copy()
    
    # Preprocessing deskripsi
    processed_df['deskripsi_processed'] = processed_df['deskripsi'].apply(preprocess_text)
    
    # Konversi kategori dari string ke list
    processed_df['kategori_list'] = processed_df['kategori'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
    
    # Gabungkan kategori menjadi string untuk TF-IDF
    processed_df['kategori_str'] = processed_df['kategori_list'].apply(lambda x: ' '.join(x))
    
    # Ekstrak koordinat
    processed_df['latitude'], processed_df['longitude'] = zip(*processed_df['koordinat'].apply(extract_coordinates))
    
    # Buat fitur kombinasi
    processed_df['combined_features'] = (
        processed_df['deskripsi_processed'] + ' ' + 
        processed_df['kategori_str'] + ' ' + 
        processed_df['provinsi'].str.lower()
    )
    
    return processed_df

def calculate_popularity_score(df, percentile=90):
    """
    Menghitung skor popularitas berdasarkan rating dan jumlah review
    menggunakan weighted rating formula: WR = (v/(v+m)) * R + (m/(v+m)) * C
    """
    # Buat salinan dataframe
    df_popular = df.copy()
    
    # Hitung parameter
    C = df_popular['rating'].mean()  # rating rata-rata seluruh dataset
    m = df_popular['jumlah_review'].quantile(percentile/100)  # jumlah review minimum
    
    # Fungsi untuk menghitung weighted rating
    def weighted_rating(x, m=m, C=C):
        v = x['jumlah_review']
        R = x['rating']
        return (v/(v+m) * R) + (m/(v+m) * C)
    
    # Hitung weighted rating
    df_popular['popularity_score'] = df_popular.apply(weighted_rating, axis=1)
    
    return df_popular 