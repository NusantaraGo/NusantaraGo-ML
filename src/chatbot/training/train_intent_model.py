import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import nltk
from nltk.stem.porter import PorterStemmer # Atau gunakan Sastrawi untuk Bahasa Indonesia jika terinstal
import pickle
import random
import os

# Pastikan NLTK data yang diperlukan sudah terunduh
# Jalankan ini sekali jika belum: nltk.download('punkt'); nltk.download('wordnet')
# Untuk Sastrawi: pip install Sastrawi
# from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Gunakan stemmer yang sesuai (PorterStemmer untuk Bahasa Inggris, Sastrawi untuk Bahasa Indonesia)
# stemmer = PorterStemmer() # Untuk Bahasa Inggris
# factory = StemmerFactory()
# stemmer = factory.create_stemmer() # Untuk Bahasa Indonesia (jika Sastrawi terinstal)

# Karena dataset kita berbahasa Indonesia, kita akan coba gunakan stemmer Sastrawi
# Jika Sastrawi belum terinstal, PorterStemmer akan digunakan sebagai fallback
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    print("Menggunakan Sastrawi Stemmer")
except ImportError:
    print("Sastrawi tidak terinstal, menggunakan PorterStemmer (kurang optimal untuk Bahasa Indonesia)")
    stemmer = PorterStemmer()

def stem(word):
    """Melakukan stemming pada kata"""
    # Ubah ke lowercase dan lakukan stemming
    return stemmer.stem(word.lower())

def bag_of_words(sentence, words):
    """Membuat bag of words dari kalimat"""
    # Tokenisasi dan stemming kalimat
    sentence_words = [stem(word) for word in nltk.word_tokenize(sentence)]
    # Buat bag of words array dengan 0 atau 1
    bag = np.zeros(len(words), dtype=np.float32)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return bag

# --- Pengaturan --- #
DATA_FILE = 'data/intents_wisata.json'
MODEL_DIR = 'models/chatbot_intent' # Direktori baru untuk model intent
MODEL_FILE = os.path.join(MODEL_DIR, 'chatbot_model.h5')
WORDS_FILE = os.path.join(MODEL_DIR, 'words.pkl')
CLASSES_FILE = os.path.join(MODEL_DIR, 'classes.pkl')

# Buat direktori model jika belum ada
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# --- Memuat dan Memproses Data --- #
print(f"Memuat data dari {DATA_FILE}...")
with open(DATA_FILE, 'r', encoding='utf-8') as f:
    intents = json.load(f)

all_words = []
tags = []
x_train = [] # Pola kalimat
y_train = [] # Tag intent

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        # Tokenisasi setiap kata dalam pola
        w = nltk.word_tokenize(pattern)
        all_words.extend(w)
        # Tambahkan pola dan tag ke list training
        x_train.append(pattern)
        y_train.append(tag)

# Stem dan hapus duplikat kata, urutkan
ignore_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(list(set(all_words)))
tags = sorted(list(set(tags)))

print(f"Jumlah pattern training: {len(x_train)}")
print(f"Jumlah intents: {len(tags)}")
print(f"Jumlah kata unik setelah stemming: {len(all_words)}")

# Simpan kata-kata unik dan tag/kelas
with open(WORDS_FILE, 'wb') as f:
    pickle.dump(all_words, f)
with open(CLASSES_FILE, 'wb') as f:
    pickle.dump(tags, f)

print("Data preprocessing selesai.")

# --- Membuat Data Training Final --- #
training_data = []
output_empty = [0] * len(tags)

for i, pattern in enumerate(x_train):
    # Buat bag of words untuk setiap pola
    bag = bag_of_words(pattern, all_words)
    
    # Buat array output (one-hot encoding) untuk tag yang sesuai
    output_row = list(output_empty)
    output_row[tags.index(y_train[i])] = 1
    
    training_data.append([bag, output_row])

# Acak data training
random.shuffle(training_data)
training_data = np.array(training_data, dtype=object)

# Pisahkan fitur (X) dan label (y)
X = np.array(list(training_data[:, 0]))
y = np.array(list(training_data[:, 1]))

print(f"Shape data training (X): {X.shape}")
print(f"Shape label training (y): {y.shape}")

# --- Membangun Model TensorFlow --- #
print("Membangun model neural network...")
model = Sequential()
model.add(Dense(128, input_shape=(len(X[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y[0]), activation='softmax'))

# Compile model
optimizer = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# --- Melatih Model --- #
print("Memulai pelatihan model...")
history = model.fit(X, y, epochs=200, batch_size=5, verbose=1)

print("Pelatihan selesai.")

# --- Menyimpan Model dan Data Pendukung --- #
print(f"Menyimpan model ke {MODEL_FILE}")
model.save(MODEL_FILE, save_format='h5')

print("Model dan data pendukung berhasil disimpan.")
print(f"Kata-kata unik disimpan di: {WORDS_FILE}")
print(f"Kelas/Tag disimpan di: {CLASSES_FILE}") 