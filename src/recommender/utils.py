import pandas as pd
import numpy as np
import ast
import json

def load_csv_data(file_path):
    """
    Memuat data dari file CSV
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Data berhasil dimuat dari {file_path}")
        print(f"Jumlah data: {len(df)}")
        return df
    except Exception as e:
        print(f"Error saat memuat data: {str(e)}")
        return None

def load_json_data(file_path):
    """
    Memuat data dari file JSON
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Konversi ke DataFrame
        df = pd.DataFrame(data)
        print(f"Data berhasil dimuat dari {file_path}")
        print(f"Jumlah data: {len(df)}")
        return df
    except Exception as e:
        print(f"Error saat memuat data: {str(e)}")
        return None

def get_available_categories(df):
    """
    Mendapatkan daftar kategori unik dari dataset
    """
    try:
        # Konversi kategori dari string ke list jika perlu
        if 'kategori_list' in df.columns:
            all_categories = [category for sublist in df['kategori_list'] for category in sublist if isinstance(sublist, list)]
        else:
            all_categories = [category for sublist in df['kategori'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else []) 
                            for category in sublist]
        
        # Dapatkan kategori unik dan urutkan
        unique_categories = sorted(list(set(all_categories)))
        return unique_categories
    except Exception as e:
        print(f"Error saat mendapatkan kategori: {str(e)}")
        return []

def get_available_provinces(df):
    """
    Mendapatkan daftar provinsi unik dari dataset
    """
    try:
        provinces = sorted(df['provinsi'].unique().tolist())
        return provinces
    except Exception as e:
        print(f"Error saat mendapatkan provinsi: {str(e)}")
        return []

def format_recommendation_results(recommendations):
    """
    Memformat hasil rekomendasi untuk ditampilkan
    """
    if isinstance(recommendations, str):
        return {"error": recommendations}
    
    results = []
    for _, row in recommendations.iterrows():
        item = {
            "nama": row["nama"],
            "provinsi": row["provinsi"],
            "rating": round(row["rating"], 1) if "rating" in row else None,
            "jumlah_review": int(row["jumlah_review"]) if "jumlah_review" in row else None,
            "kategori": row["kategori_list"] if "kategori_list" in row else None
        }
        
        # Tambahkan ID jika kolom 'id' ada di DataFrame
        if 'id' in row:
             item["id"] = int(row["id"])
        
        # Tambahkan jarak jika ada
        if "distance" in row:
            item["jarak"] = round(row["distance"], 2)
            
        # Tambahkan skor jika ada
        if "hybrid_score" in row:
            item["skor"] = round(row["hybrid_score"], 3)
        elif "similarity_score" in row:
            item["skor"] = round(row["similarity_score"], 3)
        elif "popularity_score" in row:
            item["skor"] = round(row["popularity_score"], 3)
            
        results.append(item)
        
    return results

def get_attraction_details(df, name):
    """
    Mendapatkan detail tempat wisata berdasarkan nama (pencarian fleksibel)
    """
    try:
        # Lakukan preprocessing pada nama input untuk pencarian fleksibel
        processed_name = name.lower() # Konversi ke lowercase
        processed_name = ''.join(e for e in processed_name if e.isalnum()) # Hapus non-alfanumerik

        # Lakukan preprocessing pada kolom nama di DataFrame
        # Tambahkan kolom sementara untuk pencarian
        df['_processed_nama'] = df['nama'].str.lower()
        df['_processed_nama'] = df['_processed_nama'].apply(lambda x: ''.join(e for e in x if e.isalnum()) if isinstance(x, str) else '')

        # Cari tempat wisata dengan nama yang diproses yang sama
        attraction = df[df["_processed_nama"] == processed_name]

        # Hapus kolom sementara
        df = df.drop(columns=['_processed_nama'])

        if attraction.empty:
            # Coba cari berdasarkan substring jika pencarian persis gagal
            substring_match = df[df['nama'].str.lower().str.contains(processed_name, na=False)]
            if not substring_match.empty:
                 attraction = substring_match.iloc[0]
                 print(f"Menggunakan pencarian substring untuk '{name}', ditemukan '{attraction['nama']}'")
            else:
                 return {"error": f"Tempat wisata dengan nama '{name}' tidak ditemukan."}

        # Ambil data pertama jika ada lebih dari satu (setelah pencarian fleksibel)
        attraction = attraction.iloc[0]

        # Format data
        details = {
            "id": int(attraction["id"]) if "id" in attraction else None,
            "nama": attraction["nama"],
            "alamat": attraction["alamat"] if not pd.isna(attraction["alamat"]) else None,
            "provinsi": attraction["provinsi"],
            "rating": round(float(attraction["rating"]), 1) if not pd.isna(attraction["rating"]) else None,
            "jumlah_review": int(attraction["jumlah_review"]) if not pd.isna(attraction["jumlah_review"]) else None,
            "deskripsi": attraction["deskripsi"] if not pd.isna(attraction["deskripsi"]) else None,
            "url": attraction["url"] if "url" in attraction and not pd.isna(attraction["url"]) else None,
            "foto": ast.literal_eval(attraction["foto"]) if "foto" in attraction and isinstance(attraction["foto"], str) else None,
            "kategori": attraction["kategori_list"] if "kategori_list" in attraction and isinstance(attraction["kategori_list"], list) else 
                        (ast.literal_eval(attraction["kategori"]) if "kategori" in attraction and isinstance(attraction["kategori"], str) else None)
        }
        
        # Tambahkan koordinat jika ada
        if "koordinat" in attraction and not pd.isna(attraction["koordinat"]):
            koordinat = ast.literal_eval(attraction["koordinat"]) if isinstance(attraction["koordinat"], str) else attraction["koordinat"]
            details["latitude"] = koordinat.get("latitude")
            details["longitude"] = koordinat.get("longitude")
        elif "latitude" in attraction and "longitude" in attraction:
            details["latitude"] = attraction["latitude"] if not pd.isna(attraction["latitude"]) else None
            details["longitude"] = attraction["longitude"] if not pd.isna(attraction["longitude"]) else None
            
        return details
    except Exception as e:
        return {"error": f"Error saat mendapatkan detail tempat wisata: {str(e)}"}

def filter_attractions(df, category=None, province=None, min_rating=None, max_rating=None, search_query=None):
    """
    Memfilter tempat wisata berdasarkan kriteria
    """
    filtered_df = df.copy()
    
    # Filter berdasarkan kategori (case-insensitive)
    if category:
        category_lower = category.lower()
        # Pastikan kolom kategori_list ada dan isinya list sebelum apply filter
        if 'kategori_list' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['kategori_list'].apply(lambda x: any(cat.lower() == category_lower for cat in x) if isinstance(x, list) else False)]
        # Jika tidak ada kategori_list, coba pakai kolom kategori mentah (fallback, case-insensitive)
        elif 'kategori' in filtered_df.columns:
             filtered_df = filtered_df[filtered_df['kategori'].apply(lambda x: any(cat.lower() == category_lower for cat in ast.literal_eval(x)) if isinstance(x, str) else False)]
    
    # Filter berdasarkan provinsi (case-insensitive)
    if province:
        province_lower = province.lower()
        filtered_df = filtered_df[filtered_df['provinsi'].str.lower() == province_lower]
    
    # Filter berdasarkan rating minimum
    if min_rating is not None:
        filtered_df = filtered_df[filtered_df['rating'] >= min_rating]
    
    # Filter berdasarkan rating maksimum
    # TODO: sepertinya ini tidak terpakai
    if max_rating is not None:
        filtered_df = filtered_df[filtered_df['rating'] <= max_rating]
    
    # Filter berdasarkan kueri pencarian pada nama atau deskripsi
    if search_query:
        search_query = search_query.lower()
        name_match = filtered_df['nama'].str.lower().str.contains(search_query, na=False)
        desc_match = filtered_df['deskripsi'].str.lower().str.contains(search_query, na=False)
        filtered_df = filtered_df[name_match | desc_match]
    
    return filtered_df 