import os
import time
import uuid
from datetime import datetime
from threading import Lock
from flask import Flask, render_template, request, jsonify, send_from_directory, session
from dotenv import load_dotenv

from dotenv import load_dotenv
load_dotenv()

# Import our new separated modules
from document_processor import EnhancedMultiFormatChatbot
from audio_service import generate_free_audio

# Setup & Configurations
load_dotenv()
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = os.getenv('SECRET_KEY', 'add8e6b193e17b39dd36ab85216f2c4c')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(os.getcwd(), "logs"), exist_ok=True)

ALLOWED_EXTENSIONS = {'pdf', 'txt', 'doc', 'docx', 'rtf', 'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Session Manager ---
API_KEY = os.getenv('GOOGLE_API_KEY')
active_sessions = {}
session_lock = Lock()
SESSION_TIMEOUT = 3600  

def get_user_chatbot(user_id):
    with session_lock:
        current_time = time.time()
        expired_users =[uid for uid, data in active_sessions.items() if current_time - data['last_active'] > SESSION_TIMEOUT]
        for uid in expired_users:
            del active_sessions[uid]
            
        if user_id not in active_sessions:
            active_sessions[user_id] = {'bot': EnhancedMultiFormatChatbot(API_KEY), 'last_active': current_time}
        else:
            active_sessions[user_id]['last_active'] = current_time
            
        return active_sessions[user_id]['bot']

# --- Routes ---
@app.route('/')
def index():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session['chat_log_path'] = f"logs/chat_{now}.txt"
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'file' not in request.files: return jsonify({'error': 'No file part'}), 400
    
    files = request.files.getlist('file')
    if not files or all(f.filename=='' for f in files): return jsonify({'error': 'no files selected'}), 400
    
    user_chatbot = get_user_chatbot(session.get('user_id'))
    uploaded_files = []
    session['uploaded_files'] =[]
    combined_text = ""

    for file in files:
        if allowed_file(file.filename):  
            filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            session['uploaded_files'].append(filepath)
            uploaded_files.append(filename)

            user_chatbot.load_file(filepath)
            combined_text += user_chatbot.extract_text_from_pdf(filepath) + "\n\n"
            session.modified = True

    if uploaded_files:
        user_chatbot.load_combined_text(combined_text)
        return jsonify({'success': True, 'message': f'{len(uploaded_files)} files uploaded', 'files': uploaded_files, 'ready': True}), 200
    return jsonify({'error': 'No valid files uploaded'}), 400

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question, mute, gender = data.get('question'), data.get('mute', False), data.get('gender', 'male')

    if not question: return jsonify({'error': 'No question provided'}), 400
    if not session.get('uploaded_files'): return jsonify({'error': 'Please upload a file first'}), 400
    
    try:
        user_chatbot = get_user_chatbot(session.get('user_id'))
        response_text = user_chatbot.ask_question(question)
        
        # Free Neural TTS logic
        audio_url = generate_free_audio(response_text, gender) if not mute else None
            
        return jsonify({'response': response_text, 'audio_url': audio_url})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5002)), debug=True)