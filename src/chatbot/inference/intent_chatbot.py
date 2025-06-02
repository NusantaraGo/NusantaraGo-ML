import json
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem.porter import PorterStemmer # Atau gunakan Sastrawi
import pickle
import random
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Gunakan stemmer yang sesuai (harus sama dengan saat training)
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    # print("Menggunakan Sastrawi Stemmer")
except ImportError:
    # print("Sastrawi tidak terinstal, menggunakan PorterStemmer")
    stemmer = PorterStemmer()

def stem(word):
    """Melakukan stemming pada kata"""
    return stemmer.stem(word.lower())

def bag_of_words(sentence, words):
    """Membuat bag of words dari kalimat"""
    sentence_words = [stem(word) for word in nltk.word_tokenize(sentence)]
    bag = np.zeros(len(words), dtype=np.float32)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return bag

class IntentChatbot:
    def __init__(
        self,
        model_dir: str = 'models/chatbot_intent', # Direktori model intent
        intents_file: str = 'data/intents_wisata.json',
        error_response: str = "Maaf, saya tidak mengerti. Bisa ulangi atau tanyakan hal lain?",
        prediction_threshold: float = 0.7 # Threshold untuk confidence prediksi
    ):
        """
        Inisialisasi IntentChatbot
        
        Args:
            model_dir (str): Direktori tempat model dan file pendukung disimpan.
            intents_file (str): Path ke file JSON intents.
            error_response (str): Respons jika tidak ada intent yang cocok.
            prediction_threshold (float): Confidence score minimum untuk memilih intent.
        """
        self.model_dir = model_dir
        self.intents_file = intents_file
        self.error_response = error_response
        self.prediction_threshold = prediction_threshold
        
        self._load_model_and_data()
        self._load_intents()
        
    def _load_model_and_data(self):
        """Memuat model, words, dan classes dari file"""
        model_file = os.path.join(self.model_dir, 'chatbot_model.h5')
        words_file = os.path.join(self.model_dir, 'words.pkl')
        classes_file = os.path.join(self.model_dir, 'classes.pkl')
        
        logger.info(f"Memuat model dari {model_file}...")
        try:
            self.model = tf.keras.models.load_model(model_file)
        except Exception as e:
            logger.error(f"Gagal memuat model: {e}")
            raise FileNotFoundError(f"Model file tidak ditemukan atau error: {model_file}")

        logger.info(f"Memuat words dari {words_file}...")
        try:
            with open(words_file, 'rb') as f:
                self.words = pickle.load(f)
        except Exception as e:
            logger.error(f"Gagal memuat words: {e}")
            raise FileNotFoundError(f"Words file tidak ditemukan atau error: {words_file}")

        logger.info(f"Memuat classes dari {classes_file}...")
        try:
            with open(classes_file, 'rb') as f:
                self.classes = pickle.load(f)
        except Exception as e:
            logger.error(f"Gagal memuat classes: {e}")
            raise FileNotFoundError(f"Classes file tidak ditemukan atau error: {classes_file}")
        
        logger.info("Model dan data pendukung berhasil dimuat.")

    def _load_intents(self):
        """Memuat data intents dari file JSON"""
        logger.info(f"Memuat intents dari {self.intents_file}...")
        try:
            with open(self.intents_file, 'r', encoding='utf-8') as f:
                self.intents = json.load(f)
        except Exception as e:
             logger.error(f"Gagal memuat intents file: {e}")
             raise FileNotFoundError(f"Intents file tidak ditemukan atau error: {self.intents_file}")
        logger.info("Intents berhasil dimuat.")

    def predict_intent(self, sentence: str):
        """Memprediksi intent dari kalimat input"""
        # Preprocessing input
        p = bag_of_words(sentence, self.words)
        # Reshape untuk input model (batch size 1)
        p = p.reshape(1, -1)
        
        # Prediksi dengan model
        results = self.model.predict(p, verbose=0)[0] # verbose=0 untuk tidak menampilkan progress bar
        
        # Filter prediksi di bawah threshold dan urutkan berdasarkan probabilitas
        results_filtered = [[i, r] for i, r in enumerate(results) if r > self.prediction_threshold]
        results_filtered.sort(key=lambda x: x[1], reverse=True)
        
        # Kembalikan list intent dan probabilitasnya
        return_list = []
        for r in results_filtered:
            # Convert numpy float32 ke float Python biasa agar kompatibel dengan JSON (opsional)
            return_list.append({'intent': self.classes[r[0]], 'probability': float(r[1])})
            
        return return_list

    def get_response(self, user_input: str) -> str:
        """Mendapatkan respons chatbot untuk input user"""
        # Prediksi intent
        intents_list = self.predict_intent(user_input)
        
        # Jika ada intent dengan confidence tinggi
        if intents_list:
            # Ambil tag dari intent dengan probabilitas tertinggi
            tag = intents_list[0]['intent']
            
            # Cari intent yang sesuai di data intents
            for intent in self.intents['intents']:
                if intent['tag'] == tag:
                    # Pilih respons secara acak dari daftar respons intent tersebut
                    response = random.choice(intent['responses'])
                    return response
        
        # Jika tidak ada intent yang cocok atau confidence terlalu rendah
        return self.error_response

# Contoh penggunaan (opsional, bisa dihapus/dijadikan skrip terpisah)
# if __name__ == "__main__":
#     # Pastikan NLTK data sudah terunduh
#     # nltk.download('punkt')
#     # nltk.download('wordnet')
#     
#     try:
#         chatbot = IntentChatbot()
#         print("Chatbot siap! Ketik pesan Anda.")
#         
#         while True:
#             user_input = input(">> ")
#             if user_input.lower() == 'quit':
#                 break
#             response = chatbot.get_response(user_input)
#             print(f"Bot: {response}")
#             
#     except FileNotFoundError as e:
#         print(f"Error: {e}. Pastikan model sudah dilatih dan file-file pendukung ada di direktori {IntentChatbot().model_dir}")
#     except Exception as e:
#         print(f"Terjadi error: {e}") 