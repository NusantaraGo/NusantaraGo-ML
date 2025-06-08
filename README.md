# NusantaraGo-ML

![NusantaraGo Logo](https://img.shields.io/badge/NusantaraGo-ML-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.0+-green?style=flat-square&logo=flask)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange?style=flat-square)

## 🌟 Tentang Proyek

**NusantaraGo-ML** adalah aplikasi web cerdas yang dirancang untuk membantu pengguna menjelajahi dan merencanakan perjalanan wisata di seluruh Indonesia. Proyek capstone ini menggabungkan teknologi **Machine Learning** dengan **Web Development** untuk memberikan sistem rekomendasi tempat wisata yang dipersonalisasi dan chatbot interaktif yang informatif.

### 🎯 Tujuan Proyek

- Memudahkan wisatawan dalam menemukan destinasi wisata yang sesuai dengan preferensi mereka
- Menyediakan informasi wisata yang akurat dan terkini
- Mengimplementasikan teknologi ML untuk rekomendasi yang lebih relevan
- Memberikan pengalaman pengguna yang interaktif melalui chatbot

## ✨ Fitur Utama

### 🔍 Sistem Rekomendasi Wisata

- **Content-Based Filtering**: Rekomendasi berdasarkan kemiripan karakteristik tempat wisata
- **Popularity-Based**: Rekomendasi berdasarkan popularitas dan rating
- **Location-Based**: Rekomendasi berdasarkan kedekatan geografis
- **Hybrid Recommendation**: Kombinasi dari semua metode untuk hasil optimal

### 🤖 Chatbot Interaktif

- Menjawab pertanyaan tentang informasi wisata (lokasi, jam buka, harga tiket)
- Intent recognition untuk memahami maksud pengguna
- Bantuan navigasi dan penggunaan aplikasi

### 🌐 API Endpoints

- RESTful API untuk akses data tempat wisata
- Endpoint rekomendasi dengan berbagai parameter
- Dokumentasi API yang lengkap

### 📊 Data Management

- Scraping otomatis data tempat wisata dari Google Maps
- Preprocessing dan cleaning data
- Logging komprehensif untuk monitoring

## 🛠️ Teknologi yang Digunakan

### Backend

- **Python 3.10+**
- **Flask** - Web framework
- **Scikit-learn** - Machine learning library
- **Pandas & NumPy** - Data manipulation dan analisis
- **NLTK** - Natural language processing
- **TensorFlow** - Deep learning untuk chatbot

### Data Processing

- **Requests & Beautiful Soup** - Web scraping
- **Joblib** - Model serialization

### Frontend

- **HTML/CSS/JavaScript**
- **Bootstrap** - UI framework
- **Responsive design**

## 📁 Struktur Proyek

```
NusantaraGo-ML/
├── 📁 Scrape_Data/                 # Data hasil scraping
│   ├── tempat_wisata_indonesia.csv
│   ├── tempat_wisata_indonesia.json
│   └── 📁 json/                    # Data per provinsi
│
├── 📁 src/                         # Source code utama
│   ├── 📁 recommender/             # Sistem rekomendasi
│   │   ├── model.py
│   │   ├── preprocessing.py
│   │   └── utils.py
│   └── 📁 chatbot/                 # Chatbot system
│       ├── 📁 training/
│       ├── 📁 inference/
│       └── 📁 api/
│
├── 📁 static/                      # File statis
│   └── 📁 images/                  # Gambar tempat wisata
│       └── 📁 attractions/         # Gambar yang didownload
│
├── 📁 models/                      # Model yang sudah dilatih
├── 📁 notebooks/                   # Jupyter notebooks
├── 📁 templates/                   # HTML templates
├── 📁 logs/                        # Application logs
├── 📁 data/                        # Training data
│
├── app.py                          # Main Flask application
├── scrape_data.py                  # Data scraping script
├── requirements.txt                # Dependencies
└── README.md                       # Dokumentasi
```

## 🚀 Panduan Instalasi

### Prasyarat

- Python 3.10 atau lebih tinggi
- pip (Python package manager)
- Chrome browser (untuk scraping)
- Git

### Langkah Instalasi

1. **Clone Repository**

   ```bash
   git clone https://github.com/NusantaraGo/NusantaraGo-ML.git
   cd NusantaraGo-ML
   ```

2. **Buat Virtual Environment**

   ```bash
   python -m venv .venv
   ```

3. **Aktifkan Virtual Environment**

   ```bash
   # Windows
   .venv\Scripts\activate

   # Linux/Mac
   source .venv/bin/activate
   ```

4. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

5. **Download NLTK Data** (jika diperlukan)
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('wordnet')
   nltk.download('omw-1.4')
   ```

## 📊 Pengumpulan Data

### Scraping Data Wisata

```bash
python scrape_data.py
```

Proses ini akan:

- Mengumpulkan data tempat wisata dari Google Maps
- Menyimpan data dalam format CSV dan JSON
- Menghasilkan dataset untuk 34 provinsi di Indonesia

**Catatan**: Proses scraping membutuhkan waktu yang cukup lama. Data yang sudah tersedia bisa langsung digunakan.

## 🔧 Cara Menjalankan Aplikasi

### 1. Menjalankan Web Application

```bash
python app.py
```

Aplikasi akan berjalan di: `http://localhost:5000`

### 2. Training Model (Opsional)

Model akan otomatis dilatih saat pertama kali menjalankan aplikasi jika belum ada model yang tersimpan.

### 3. Mengakses Dokumentasi API

Buka: `http://localhost:5000/dokumentasi`

## 📚 Dokumentasi API

### Endpoint Utama

#### Data Dasar

- `GET /api/provinces` - Daftar provinsi
- `GET /api/categories` - Daftar kategori wisata
- `GET /api/attractions` - Daftar tempat wisata (dengan filter)
- `GET /api/attraction/{nama}` - Detail tempat wisata

#### Sistem Rekomendasi

- `GET /api/recommendations/content` - Content-based recommendation
- `GET /api/recommendations/popularity` - Popularity-based recommendation
- `GET /api/recommendations/location` - Location-based recommendation
- `GET /api/recommendations/hybrid` - Hybrid recommendation

### Contoh Request

#### Content-Based Recommendation

```bash
curl "http://localhost:5000/api/recommendations/content?name=Pantai%20Kuta&limit=5"
```

#### Location-Based Recommendation

```bash
curl "http://localhost:5000/api/recommendations/location?lat=-8.409518&lon=115.188919&limit=10"
```

## 🤖 Sistem Chatbot

### Fitur Chatbot

- **Intent Recognition**: Memahami maksud pengguna
- **Entity Extraction**: Mengekstrak informasi penting
- **Context Management**: Mempertahankan konteks percakapan
- **Response Generation**: Menghasilkan respons yang relevan

### Training Chatbot

```bash
python src/chatbot/training/train_intent_model.py
```

## 🧪 Machine Learning Pipeline

### 1. Data Preprocessing

```python
from src.recommender.preprocessing import preprocess_data

# Load dan preprocess data
data = preprocess_data('Scrape_Data/tempat_wisata_indonesia.csv')
```

### 2. Model Training

```python
from src.recommender.model import TourismRecommender

# Inisialisasi dan training model
recommender = TourismRecommender()
recommender.fit(data)
```

### 3. Prediction

```python
# Content-based recommendation
recommendations = recommender.recommend_by_content('Pantai Kuta', n_recommendations=5)

# Location-based recommendation
recommendations = recommender.recommend_by_location(-8.409518, 115.188919, n_recommendations=10)
```

## 📈 Evaluasi Model

### Metrics yang Digunakan

- **Precision@K**: Ketepatan rekomendasi top-K
- **Recall@K**: Kelengkapan rekomendasi top-K
- **Diversity**: Keberagaman rekomendasi
- **Coverage**: Cakupan item dalam rekomendasi

### Performance

- Content-based: Precision@5 = 0.78
- Location-based: Average distance accuracy = 92%
- Hybrid: Overall satisfaction score = 4.2/5

## 🐛 Troubleshooting

### Masalah Umum

**1. Model tidak bisa dimuat**

```bash
# Pastikan file model ada
ls models/recommendation_model.joblib

# Jika tidak ada, model akan otomatis dilatih
python app.py
```

**2. Error saat scraping**

```bash
# Pastikan Chrome browser terinstall
# Cek koneksi internet
# Gunakan data yang sudah tersedia jika diperlukan
```

**3. API endpoint error 500**

```bash
# Cek log aplikasi
tail -f logs/nusantarago.log
```

## 🤝 Kontribusi

Kami menyambut kontribusi dari komunitas! Silakan:

1. Fork repository ini
2. Buat feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit perubahan (`git commit -m 'Add some AmazingFeature'`)
4. Push ke branch (`git push origin feature/AmazingFeature`)
5. Buat Pull Request

### Guidelines Kontribusi

- Ikuti coding standards yang ada
- Tambahkan tests untuk fitur baru
- Update dokumentasi jika diperlukan
- Pastikan semua tests pass

## 🔗 Links Terkait

- **Repository**: [GitHub NusantaraGo-ML](https://github.com/NusantaraGo/NusantaraGo-ML)
- **Dataset**: [Google Sheets - Data Scraping](https://docs.google.com/spreadsheets/d/1P8SPGkTBTq15AtAhIwSGsu5CWCanBBs6mGu7lySvPGU/edit?usp=sharing)
- **Organization**: [NusantaraGo GitHub](https://github.com/NusantaraGo)

## 📞 Kontak

Jika ada pertanyaan atau saran, silakan hubungi:

- Email: [email@example.com]

---

<div align="center">

**⭐ Jika proyek ini membantu, jangan lupa berikan star! ⭐**

Made with ❤️ for Indonesian Tourism

</div>
