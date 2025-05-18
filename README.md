# NusantaraGo-ML

NusantaraGo merupakan project capstone yang bertujuan untuk memberikan rekomendasi tempat wisata di Indonesia serta dilengkapi dengan chatbot untuk informasi wisata.

## Fitur:

1. **Sistem Rekomendasi Wisata**: Memberikan rekomendasi tempat wisata di seluruh Indonesia berdasarkan preferensi pengguna.
2. **Chatbot Wisata**: Menyediakan informasi tentang tempat wisata melalui chatbot.

## Prasyarat

- Python 3.10
- pip (Package Manager Python)
- Chrome browser (untuk proses scraping)

## Panduan Lengkap Penggunaan

### 1. Persiapan Awal

1. Clone repository ini:

   ```bash
   git clone https://github.com/NusantaraGo/NusantaraGo-ML.git
   cd NusantaraGo-ML
   ```

2. Buat virtual environment:

   ```bash
   python -m venv .venv
   ```

3. Aktifkan virtual environment:

   - Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source .venv/bin/activate
     ```

4. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```

### 2. Pengumpulan Data

Untuk mengumpulkan data tempat wisata dari Google Maps:

```bash
python scrape_data.py
```

Proses ini akan:

- Melakukan scraping data tempat wisata dari Google Maps untuk setiap provinsi di Indonesia
- Menyimpan data dalam format CSV di folder `Scrape_Data/`
- Menghasilkan file utama `tempat_wisata_indonesia.csv`

Catatan: Proses scraping membutuhkan waktu yang cukup lama. Jika tidak ingin melakukan scraping, Anda bisa langsung menggunakan data yang sudah tersedia di folder `Scrape_Data/`.

### 3. Eksplorasi dan Pembuatan Model

#### Eksplorasi Data dengan Notebook

1. Buka notebook untuk eksplorasi data:

   ```bash
   jupyter notebook notebooks/sistem_rekomendasi.ipynb
   ```

2. Jalankan sel-sel dalam notebook untuk:
   - Melihat distribusi data tempat wisata
   - Menganalisis rating dan jumlah review
   - Melihat distribusi kategori tempat wisata
   - Membuat dan menguji model rekomendasi berbasis konten

#### Atau Kombinasikan Data dari Hasil Scraping

Jika Anda memiliki beberapa file hasil scraping yang ingin digabungkan:

```bash
jupyter notebook notebooks/combine_data.ipynb
```

### 4. Melatih Model Rekomendasi

Setelah eksplorasi data, Anda dapat melatih model rekomendasi secara langsung:

```bash
python train_model.py
```

Model akan:

- Membaca data dari `Scrape_Data/tempat_wisata_indonesia.csv`
- Melatih model rekomendasi berbasis konten
- Menyimpan model dalam format joblib di `models/recommendation_model.joblib`

### 5. Menjalankan API

Setelah model terlatih, jalankan API Flask:

```bash
python app.py
```

API akan berjalan di `http://localhost:5000` dan menyediakan berbagai endpoint untuk:

- Mendapatkan daftar tempat wisata
- Mencari tempat wisata berdasarkan nama, kategori, atau provinsi
- Mendapatkan rekomendasi tempat wisata

## Menggunakan API

### Endpoint Dasar

- **Daftar Provinsi**: `GET http://localhost:5000/api/provinces`
- **Daftar Kategori**: `GET http://localhost:5000/api/categories`
- **Daftar Tempat Wisata**: `GET http://localhost:5000/api/attractions`

### Mendapatkan Rekomendasi

1. **Rekomendasi Berbasis Konten** (berdasarkan tempat sejenis):

   ```
   GET http://localhost:5000/api/recommendations/content?name=Pantai%20Kuta&limit=5
   ```

2. **Rekomendasi Berdasarkan Popularitas**:

   ```
   GET http://localhost:5000/api/recommendations/popularity?category=Museum&limit=5
   ```

3. **Rekomendasi Berdasarkan Lokasi**:

   ```
   GET http://localhost:5000/api/recommendations/location?lat=-7.7956&lon=110.3695&max_distance=30
   ```

4. **Rekomendasi Hybrid** (menggabungkan semua faktor):
   ```
   GET http://localhost:5000/api/recommendations/hybrid?name=Pantai%20Kuta&lat=-8.7183&lon=115.1686&province=Bali
   ```

## Struktur Folder

```
NusantaraGo-ML/
│
├── Scrape_Data/               # Berisi data hasil scraping
│   ├── tempat_wisata_indonesia.csv    # Data utama tempat wisata
│   └── json/                  # Folder berisi hasil scraping dalam format JSON
│
├── models/                    # Menyimpan model yang telah dilatih
│   ├── recommendation_model.joblib     # Model rekomendasi
│   └── chatbot_model/         # Model chatbot
│
├── notebooks/                 # Jupyter notebooks untuk eksplorasi data
│   ├── combine_data.ipynb     # Notebook untuk menggabungkan data hasil scraping
│   ├── sistem_rekomendasi.ipynb     # Eksplorasi data dan pembuatan model
│   └── chatbot.ipynb          # Pengembangan model chatbot
│
├── src/                       # Source code utama
│   ├── recommender/           # Kode untuk sistem rekomendasi
│   │   ├── __init__.py
│   │   ├── preprocessing.py   # Preprocessing data
│   │   ├── model.py           # Definisi model rekomendasi
│   │   └── utils.py           # Fungsi utilitas
│   │
│   ├── chatbot/               # Kode untuk chatbot
│   │   ├── __init__.py
│   │   └── bot.py             # Implementasi chatbot
│   │
│   └── app/                   # Aplikasi web
│       ├── __init__.py
│       ├── routes.py          # Definisi routes
│       ├── static/            # Aset statis (CSS, JS)
│       └── templates/         # Template HTML
│
├── templates/                 # Template HTML untuk aplikasi Flask
│   └── index.html             # Halaman utama API
│
├── tests/                     # Unit tests
│
├── app.py                     # Entry point aplikasi Flask
├── scrape_data.py             # Script untuk scraping data
├── train_model.py             # Script untuk melatih model rekomendasi
├── .gitignore                 # File gitignore
├── requirements.txt           # Dependensi proyek
└── README.md                  # Dokumentasi proyek
```

## Troubleshooting

### Masalah Saat Scraping

- Pastikan Chrome browser terinstal
- Jika terjadi error koneksi, coba lagi dengan menambahkan delay yang lebih lama
- Gunakan proxy jika dibatasi oleh Google

### Masalah Model Rekomendasi

- Jika model tidak muncul, pastikan `train_model.py` telah dijalankan
- Jika file data tidak ditemukan, pastikan `tempat_wisata_indonesia.csv` ada di folder `Scrape_Data/`

### Masalah API

- Pastikan port 5000 tidak digunakan oleh aplikasi lain
- Periksa apakah model recommendation_model.joblib sudah ada di folder models/

## Link Terkait

- [Repository GitHub NusantaraGo](https://github.com/NusantaraGo)
- [Link Google Sheets daerah scraping](https://docs.google.com/spreadsheets/d/1P8SPGkTBTq15AtAhIwSGsu5CWCanBBs6mGu7lySvPGU/edit?usp=sharing)
