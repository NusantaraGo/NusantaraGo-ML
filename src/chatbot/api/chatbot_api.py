"""
API endpoint untuk chatbot wisata
"""

from flask import Blueprint, request, jsonify
from ..inference.intent_chatbot import IntentChatbot
import logging
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inisialisasi blueprint
chatbot_bp = Blueprint('chatbot', __name__)

# Inisialisasi chatbot berbasis intent
# Pastikan model dan file pendukung ada di 'models/chatbot_intent'
try:
    chatbot = IntentChatbot()
    logger.info("IntentChatbot berhasil diinisialisasi.")
except FileNotFoundError as e:
    logger.error(f"Gagal menginisialisasi IntentChatbot: {e}")
    logger.error("Pastikan model sudah dilatih dan file-file pendukung (.h5, .pkl) ada di direktori models/chatbot_intent")
    # Handle error - mungkin set chatbot ke None atau gunakan dummy response
    chatbot = None # Set None jika gagal inisialisasi
    

@chatbot_bp.route('/chat', methods=['POST'])
def chat():
    """
    Endpoint untuk chat dengan chatbot
    
    Request body:
    {
        "message": "Pesan dari user",
        "reset": false  # Opsional, untuk reset conversation
    }
    
    Response:
    {
        "response": "Respons dari chatbot",
        "conversation_history": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    }
    """
    if chatbot is None:
         return jsonify({
            "error": "Chatbot model not loaded",
            "message": "Model chatbot belum siap atau gagal dimuat."
        }), 503 # Service Unavailable
        
    try:
        data = request.get_json()
        
        # Validasi input
        if not data or "message" not in data:
            return jsonify({
                "error": "Missing required field: message"
            }), 400
        
        # Reset conversation jika diminta (IntentChatbot tidak menyimpan history di dalam kelas secara default)
        # Jika Anda ingin history, perlu diimplementasikan di IntentChatbot atau di sini.
        # Untuk model intent sederhana, history biasanya tidak digunakan untuk prediksi intent.
        # Endpoint reset tetap dipertahankan untuk kompatibilitas front-end.
        if data.get("reset", False):
             chatbot.reset_conversation() # Jika IntentChatbot punya method reset
             logger.info("Conversation reset requested (handled by IntentChatbot if implemented)")

        
        # Generate respons menggunakan IntentChatbot
        user_message = data["message"]
        response = chatbot.get_response(user_message) # Gunakan metode get_response
        
        # Return respons
        # IntentChatbot sederhana tidak mengembalikan history percakapan dalam response
        return jsonify({
            "response": response
            # "conversation_history": chatbot.get_conversation_history() # Hapus atau sesuaikan jika IntentChatbot mengelola history
        })
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

@chatbot_bp.route('/reset', methods=['POST'])
def reset_conversation():
    """Reset conversation history"""
    if chatbot is None:
         return jsonify({
            "error": "Chatbot model not loaded",
            "message": "Model chatbot belum siap atau gagal dimuat."
        }), 503 # Service Unavailable
        
    try:
        chatbot.reset_conversation() # Panggil metode reset di IntentChatbot
        return jsonify({
            "message": "Conversation history reset successfully (handled by IntentChatbot if implemented)"
        })
    except Exception as e:
        logger.error(f"Error resetting conversation: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

# Endpoint history mungkin tidak relevan lagi jika IntentChatbot tidak mengelola history
# Anda bisa menghapus atau membiarkannya mengembalikan history kosong/default
@chatbot_bp.route('/history', methods=['GET'])
def get_history():
    """Get conversation history"""
    if chatbot is None:
         return jsonify({
            "error": "Chatbot model not loaded",
            "message": "Model chatbot belum siap atau gagal dimuat."
        }), 503 # Service Unavailable
        
    try:
        # IntentChatbot sederhana tidak menyimpan history internal
        # Return history kosong atau sesuai kebutuhan
        return jsonify({
            "conversation_history": [] # Mengembalikan list kosong
        })
    except Exception as e:
        logger.error(f"Error getting conversation history: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500 