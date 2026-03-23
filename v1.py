import os
import time
import socket
from flask import Flask, request, jsonify, render_template_string
import ollama
from werkzeug.utils import secure_filename
from deep_translator import GoogleTranslator
from PIL import Image
import io
import threading
import json
from datetime import datetime

app = Flask(__name__)

# --- Configuration ---
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp', 'tiff', 'tif'}

# Create uploads folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    print(f"📁 Created uploads folder: {UPLOAD_FOLDER}")

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 15 * 1024 * 1024  # Increased to 15MB
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Expanded language support
LANG_MAP = {
    "English": "en",
    "Tamil": "ta",
    "Hindi": "hi",
    "Telugu": "te",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Bengali": "bn",
    "Punjabi": "pa",
    "Urdu": "ur",
    "Odia": "or",
    "Assamese": "as",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese": "zh-CN",
    "Japanese": "ja",
    "Korean": "ko",
    "Arabic": "ar",
    "Portuguese": "pt",
    "Italian": "it",
    "Russian": "ru",
    "Turkish": "tr",
    "Vietnamese": "vi",
    "Thai": "th",
    "Indonesian": "id",
    "Filipino": "tl"
}

# Cache for faster response
response_cache = {}
cache_lock = threading.Lock()

# Chat history memory - NEW: Store conversation context
conversation_history = {}
conversation_lock = threading.Lock()

# --- NEW: Configurable AI model for text processing (Llama 3.2) ---
AI_MODEL = "llama3.2"  # Change this to switch models easily

# --- Helper Functions ---
def allowed_file(filename):
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS

def optimize_image(image_path, max_size=(512, 512), quality=85):
    """Optimize image for faster upload and processing"""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'LA', 'P'):
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'RGBA':
                    rgb_img.paste(img, mask=img.split()[-1])
                else:
                    rgb_img.paste(img)
                img = rgb_img
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize if too large
            if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Save optimized image
            optimized_path = os.path.splitext(image_path)[0] + '_optimized.jpg'
            img.save(optimized_path, 'JPEG', quality=quality, optimize=True)
            
            # Remove original if different
            if optimized_path != image_path and os.path.exists(image_path):
                os.remove(image_path)
            
            return optimized_path
            
    except Exception as e:
        print(f"⚠️ Image optimization error: {e}")
        return image_path

def get_conversation_context(conversation_id, max_history=10):
    """Get conversation history for context"""
    with conversation_lock:
        if conversation_id not in conversation_history:
            conversation_history[conversation_id] = []
        return conversation_history[conversation_id][-max_history:]

def add_to_conversation(conversation_id, role, content):
    """Add message to conversation history"""
    with conversation_lock:
        if conversation_id not in conversation_history:
            conversation_history[conversation_id] = []
        
        # Format the message
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        }
        
        conversation_history[conversation_id].append(message)
        
        # Limit history size
        if len(conversation_history[conversation_id]) > 20:
            conversation_history[conversation_id] = conversation_history[conversation_id][-20:]

def clear_conversation(conversation_id):
    """Clear conversation history"""
    with conversation_lock:
        if conversation_id in conversation_history:
            del conversation_history[conversation_id]

def is_ascii(text):
    """Check if a string contains only ASCII characters (basic English heuristic)."""
    return all(ord(c) < 128 for c in text)

# ---------- Smart Query Classifier ----------
def classify_query_type(query):
    """
    Determine if the user is asking a technical/programming question or just chatting casually.
    Returns 'technical' or 'casual'.
    """
    tech_keywords = [
        'code', 'python', 'java', 'c++', 'javascript', 'programming',
        'function', 'class', 'method', 'debug', 'error', 'exception',
        'algorithm', 'data structure', 'api', 'http', 'server',
        'flask', 'django', 'database', 'sql', 'html', 'css',
        'compiler', 'interpreter', 'syntax', 'variable', 'loop',
        'conditional', 'array', 'list', 'dict', 'object', 'inheritance'
    ]
    q_lower = query.lower()
    for kw in tech_keywords:
        if kw in q_lower:
            return 'technical'
    return 'casual'

# --- Core AI Logic ---
def process_ai_request(user_text, target_lang_name, image_path=None, conversation_id=None):
    try:
        target_code = LANG_MAP.get(target_lang_name, "en")
        
        # Create cache key
        cache_key = f"{user_text[:100]}_{target_lang_name}_{image_path}_{conversation_id}"
        
        with cache_lock:
            if cache_key in response_cache:
                print(f"📦 Using cached response")
                return response_cache[cache_key]
        
        # If user sends image without text, use default prompt
        if not user_text.strip() and image_path:
            user_text = "Describe what you see in this image in detail"
        
        # Get conversation context if available
        context_messages = []
        if conversation_id:
            history = get_conversation_context(conversation_id)
            for msg in history:
                if msg['role'] == 'user':
                    context_messages.append({"role": "user", "content": msg['content']})
                elif msg['role'] == 'assistant':
                    context_messages.append({"role": "assistant", "content": msg['content']})
        
        # ---------- TRANSLATION LOGIC (auto-detect) ----------
        # Translate User Input -> English (Internal) for AI processing
        english_input = user_text
        if user_text.strip():
            # Skip translation ONLY if target language is English AND the input appears to be English (ASCII)
            skip_input_translation = (target_code == "en" and is_ascii(user_text))
            if not skip_input_translation:
                try:
                    # Auto-detect source language and translate to English
                    english_input = GoogleTranslator(source='auto', target='en').translate(user_text)
                    print(f"🌐 Auto-detected & translated user input to English: {english_input}")
                except Exception as e:
                    print(f"⚠️ Translation error (input): {e}")
                    english_input = user_text  # fallback to original
            else:
                print(f"⏩ Skipped input translation (English → English)")
        
        # ---------- SMART SYSTEM PROMPT (OPTIMIZED FOR TRANSLATION) ----------
        query_type = classify_query_type(english_input)
        
        translation_rules = (
            "IMPORTANT INSTRUCTION: Your exact response will be machine-translated into another language. "
            "Therefore, you MUST follow these strict rules: "
            "1. Use very simple, plain, and direct English words. "
            "2. Keep sentences short and crisp. "
            "3. DO NOT use idioms, metaphors, slang, or complex vocabulary. "
            "4. Be clear and avoid ambiguous phrasing. "
        )

        if query_type == 'technical':
            system_instruction = translation_rules + (
                "You are a helpful IT professor. "
                "Provide a structured, point-by-point explanation suitable for a college viva or exam. "
                "Use bullet points."
            )
        else:  # casual
            system_instruction = translation_rules + (
                "You are a friendly human-like AI. "
                "Reply very briefly in 1 or 2 short sentences. Be direct and polite."
            )
        
        # Prepare messages for Ollama
        messages = []
        # Insert system instruction as first message
        messages.append({"role": "system", "content": system_instruction})
        # Add conversation history (if any)
        if context_messages:
            messages.extend(context_messages)
        # Add the current user query
        messages.append({"role": "user", "content": english_input})
        
        if image_path and os.path.exists(image_path):
            print(f"🖼️ Processing image: {os.path.basename(image_path)}")
            
            # ---------- FIXED: Use llama3.2-vision:latest directly, NO downloads ----------
            VISION_MODEL = "llama3.2-vision"
            try:
                # FIX: Do NOT include system_instruction - use a simple direct prompt
                simple_vision_prompt = f"Please answer this about the image: {english_input}. Be brief and use simple words."
                response = ollama.generate(
                    model=VISION_MODEL,
                    prompt=simple_vision_prompt,
                    images=[os.path.abspath(image_path)],
                    options={"num_predict": 200, "temperature": 0.3}
                )
                english_output = response['response']
                
            except Exception as model_error:
                print(f"❌ Vision model error: {model_error}")
                # Do NOT attempt to pull; return a clear error message
                english_output = (
                    "❌ The vision model 'llama3.2-vision' is not available. "
                    "Please ensure it is installed with: ollama pull llama3.2-vision"
                )
                
        else:
            # Text-only conversation with context and smart system prompt
            try:
                response = ollama.chat(
                    model=AI_MODEL, 
                    messages=messages,
                    options={"num_predict": 300}
                )
                english_output = response["message"]["content"]
            except Exception as e:
                print(f"❌ Text processing error: {e}")
                english_output = "I apologize, but I encountered an error. Please try again."

        # ---------- OUTPUT TRANSLATION ----------
        final_output = english_output
        if target_code != "en" and english_output:
            try:
                final_output = GoogleTranslator(source='en', target=target_code).translate(english_output)
                print(f"🌐 Translated response to {target_lang_name}")
            except Exception as trans_error:
                print(f"⚠️ Translation error (output): {trans_error}")
                final_output = english_output  # fallback
        else:
            print(f"⏩ Skipped output translation (target is English)")
        
        # Store in conversation history if conversation_id exists
        if conversation_id:
            add_to_conversation(conversation_id, 'user', user_text)
            add_to_conversation(conversation_id, 'assistant', final_output)
        
        # Cache the response
        with cache_lock:
            response_cache[cache_key] = final_output
            # Limit cache size
            if len(response_cache) > 100:
                response_cache.pop(next(iter(response_cache)))
            
        return final_output

    except Exception as e:
        print(f"❌ System error in AI processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return "⚠️ System encountered an issue. Please try again."

# --- Flask Routes ---
@app.route("/")
def home():
    return render_template_string(HTML_PAGE)

@app.route("/chat", methods=["POST"])
def chat():
    try:
        start_time = time.time()
        user_msg = request.form.get("message", "").strip()
        target_lang = request.form.get("language", "English")
        image_file = request.files.get("image")
        conversation_id = request.form.get("conversation_id")  # NEW: Get conversation ID
        
        print(f"\n" + "="*50)
        print(f"📩 New request received (Conversation ID: {conversation_id})")
        print(f"🌐 Selected language: {target_lang}")
        
        file_path = None
        if image_file and image_file.filename:
            if not allowed_file(image_file.filename):
                return jsonify({
                    "answer": f"❌ Please upload only image files ({', '.join(ALLOWED_EXTENSIONS)})",
                    "lang_code": "en",
                    "status": "error"
                })
            
            # Validate file size (15MB)
            if image_file.content_length > 15 * 1024 * 1024:
                return jsonify({
                    "answer": "❌ Image size must be less than 15MB",
                    "lang_code": "en",
                    "status": "error"
                })
            
            # Secure the filename
            filename = secure_filename(f"img_{int(time.time())}_{image_file.filename}")
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Save the file
            image_file.save(file_path)
            
            # Optimize image for faster processing
            if os.path.exists(file_path):
                file_path = optimize_image(file_path)
        
        # Process the request
        response = process_ai_request(user_msg, target_lang, file_path, conversation_id)
        
        # Clean up image file if it exists
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass
        
        print(f"✅ Response ready in {time.time() - start_time:.2f}s")
        print("="*50)
        
        return jsonify({
            "answer": response, 
            "lang_code": LANG_MAP.get(target_lang, "en"),
            "status": "success",
            "conversation_id": conversation_id  # Return conversation ID for frontend
        })
        
    except Exception as e:
        print(f"❌ Route error: {str(e)}")
        return jsonify({
            "answer": f"❌ Error processing request",
            "lang_code": "en",
            "status": "error"
        })

# NEW: Clear conversation endpoint
@app.route("/clear_conversation", methods=["POST"])
def clear_conversation_route():
    try:
        conversation_id = request.json.get("conversation_id")
        if conversation_id:
            clear_conversation(conversation_id)
            return jsonify({
                "status": "success",
                "message": "Conversation history cleared"
            })
        return jsonify({
            "status": "error",
            "message": "No conversation ID provided"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

# Health check endpoint
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "upload_folder": app.config['UPLOAD_FOLDER'],
        "cache_size": len(response_cache),
        "active_conversations": len(conversation_history)  # NEW: Show active conversations
    })

# --- HTML_PAGE remains completely unchanged ---
HTML_PAGE = '''
<!DOCTYPE html>
<html lang="en" data-bs-theme="light">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=5.0, user-scalable=yes">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <title>Vision AI Assistant</title>
    
    <!-- Bootstrap 5 CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <!-- Bootstrap Icons -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css" rel="stylesheet">
    
    <style>
        :root {
            --primary-color: #2563eb;
            --primary-dark: #1d4ed8;
            --secondary-color: #64748b;
            --accent-color: #10b981;
            --danger-color: #ef4444;
            
            --bg-primary: #ffffff;
            --bg-secondary: #f8fafc;
            --bg-surface: #ffffff;
            
            --text-primary: #1e293b;
            --text-secondary: #475569;
            
            --border-color: #e2e8f0;
        }

        [data-bs-theme="dark"] {
            --primary-color: #3b82f6;
            --primary-dark: #2563eb;
            --secondary-color: #94a3b8;
            --accent-color: #10b981;
            --danger-color: #ef4444;
            
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-surface: #1e293b;
            
            --text-primary: #f8fafc;
            --text-secondary: #e2e8f0;
            
            --border-color: #334155;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            -webkit-tap-highlight-color: transparent;
        }

        html, body {
            height: 100%;
            width: 100%;
            overflow: hidden;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.5;
            -webkit-font-smoothing: antialiased;
            overscroll-behavior: none;
        }

        /* WhatsApp-like Layout */
        .app-container {
            display: flex;
            flex-direction: column;
            height: 100vh;
            height: -webkit-fill-available;
            max-width: 100vw;
        }

        /* Header */
        .app-header {
            background: var(--bg-surface);
            border-bottom: 1px solid var(--border-color);
            padding: 0.75rem 1rem;
            position: sticky;
            top: 0;
            z-index: 1000;
            flex-shrink: 0;
        }

        .header-content {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
        }

        .header-brand {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            flex: 1;
        }

        .brand-logo {
            width: 36px;
            height: 36px;
            background: linear-gradient(135deg, var(--primary-color), var(--accent-color));
            border-radius: 0.5rem;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 1rem;
        }

        .brand-text {
            display: flex;
            flex-direction: column;
        }

        .brand-title {
            font-size: 1rem;
            font-weight: 600;
            color: var(--text-primary);
            margin: 0;
            line-height: 1.2;
        }

        .brand-subtitle {
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin: 0;
        }

        /* Header Controls */
        .header-controls {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .control-btn {
            width: 36px;
            height: 36px;
            border-radius: 0.5rem;
            border: 1px solid var(--border-color);
            background: var(--bg-surface);
            color: var(--text-secondary);
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .control-btn:active {
            transform: scale(0.95);
        }

        .language-select {
            min-width: 140px;
            border-radius: 0.5rem;
            border: 1px solid var(--border-color);
            background: var(--bg-surface);
            color: var(--text-primary);
            padding: 0.375rem 0.5rem;
            font-size: 0.875rem;
            cursor: pointer;
        }

        /* Chat Container */
        .chat-container {
            flex: 1;
            overflow-y: auto;
            overflow-x: hidden;
            -webkit-overflow-scrolling: touch;
            padding: 1rem;
            background: var(--bg-primary);
            display: flex;
            flex-direction: column;
        }

        .chat-inner {
            flex: 1;
            max-width: 768px;
            margin: 0 auto;
            width: 100%;
            padding-bottom: 1rem;
        }

        /* Welcome Section */
        .welcome-section {
            animation: fadeIn 0.5s ease-out;
        }

        .welcome-card {
            background: var(--bg-surface);
            border: 1px solid var(--border-color);
            border-radius: 1rem;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
            text-align: center;
        }

        .welcome-icon {
            width: 64px;
            height: 64px;
            background: linear-gradient(135deg, var(--primary-color), var(--accent-color));
            border-radius: 1rem;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 1.5rem;
            color: white;
            font-size: 1.5rem;
        }

        .welcome-title {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 0.5rem;
        }

        .welcome-subtitle {
            font-size: 0.9375rem;
            color: var(--text-secondary);
            margin: 0 auto 1.5rem;
            line-height: 1.6;
        }

        /* Features Grid */
        .features-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 0.75rem;
            margin-bottom: 1.5rem;
        }

        .feature-card {
            background: var(--bg-surface);
            border: 1px solid var(--border-color);
            border-radius: 0.75rem;
            padding: 1rem;
            text-align: center;
            transition: all 0.2s ease;
            cursor: pointer;
        }

        .feature-card:active {
            transform: scale(0.98);
            border-color: var(--primary-color);
        }

        .feature-icon {
            width: 40px;
            height: 40px;
            background: var(--bg-secondary);
            border-radius: 0.75rem;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 0.75rem;
            color: var(--primary-color);
            font-size: 1.125rem;
        }

        .feature-title {
            font-size: 0.875rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 0.25rem;
        }

        .feature-description {
            font-size: 0.75rem;
            color: var(--text-secondary);
            line-height: 1.4;
        }

        /* Messages */
        .message-container {
            margin-bottom: 1rem;
            animation: slideIn 0.3s ease-out;
        }

        .message {
            display: flex;
            gap: 0.75rem;
            margin-bottom: 0.75rem;
        }

        .message.user {
            flex-direction: row-reverse;
        }

        .message-avatar {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.75rem;
            font-weight: 600;
            flex-shrink: 0;
            margin-top: 0.25rem;
        }

        .message-avatar.ai {
            background: linear-gradient(135deg, var(--primary-color), var(--accent-color));
            color: white;
        }

        .message-avatar.user {
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            color: white;
        }

        .message-content {
            max-width: 70%;
            min-width: 0;
        }

        .message-bubble {
            padding: 0.75rem 1rem;
            border-radius: 1rem;
            font-size: 0.9375rem;
            line-height: 1.5;
            position: relative;
            word-wrap: break-word;
        }

        .message-bubble.ai {
            background: var(--bg-surface);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            border-bottom-left-radius: 0.375rem;
        }

        .message-bubble.user {
            background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
            color: white;
            border-bottom-right-radius: 0.375rem;
        }

        .message-image {
            max-width: 100%;
            max-height: 200px;
            border-radius: 0.5rem;
            margin-bottom: 0.75rem;
            border: 1px solid rgba(255, 255, 255, 0.3);
            object-fit: cover;
        }

        .message-actions {
            display: flex;
            gap: 0.5rem;
            margin-top: 0.5rem;
        }

        .action-btn {
            padding: 0.25rem 0.5rem;
            border-radius: 0.375rem;
            border: 1px solid rgba(255, 255, 255, 0.3);
            background: rgba(255, 255, 255, 0.15);
            color: white;
            font-size: 0.75rem;
            font-weight: 500;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 0.25rem;
            transition: all 0.2s ease;
        }

        .message-bubble.ai .action-btn {
            border: 1px solid var(--border-color);
            background: var(--bg-secondary);
            color: var(--text-secondary);
        }

        .action-btn:active {
            transform: scale(0.95);
        }

        /* Preview Section - FIXED POSITION */
        .preview-section {
            position: fixed;
            bottom: 120px;
            right: 1rem;
            left: 1rem;
            max-width: 768px;
            margin: 0 auto;
            z-index: 800;
            animation: slideUp 0.3s ease-out;
        }

        .preview-card {
            background: var(--bg-surface);
            border: 1px solid var(--border-color);
            border-radius: 0.75rem;
            padding: 0.75rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            max-width: 300px;
            margin-left: auto;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }

        .preview-image {
            width: 60px;
            height: 60px;
            border-radius: 0.5rem;
            object-fit: cover;
            border: 1px solid var(--border-color);
        }

        .preview-info {
            flex: 1;
            min-width: 0;
        }

        .preview-title {
            font-size: 0.875rem;
            font-weight: 500;
            color: var(--text-primary);
            margin-bottom: 0.25rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .preview-size {
            font-size: 0.75rem;
            color: var(--text-secondary);
        }

        .preview-remove {
            background: transparent;
            border: none;
            color: var(--danger-color);
            cursor: pointer;
            padding: 0.5rem;
            border-radius: 0.5rem;
            display: flex;
            align-items: center;
            justify-content: center;
            width: 32px;
            height: 32px;
        }

        /* Processing Indicator */
        .processing-indicator {
            background: var(--bg-surface);
            border-top: 1px solid var(--border-color);
            padding: 1.25rem 1rem;
            position: sticky;
            bottom: 0;
            z-index: 900;
            flex-shrink: 0;
            width: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.75rem;
            font-size: 1rem;
            color: var(--text-secondary);
        }

        .processing-dots {
            display: flex;
            gap: 0.25rem;
        }

        .processing-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--text-secondary);
            animation: typing 1.4s infinite ease-in-out;
        }

        .processing-dot:nth-child(1) { animation-delay: -0.32s; }
        .processing-dot:nth-child(2) { animation-delay: -0.16s; }

        /* Input Section */
        .input-section {
            background: var(--bg-surface);
            border-top: 1px solid var(--border-color);
            padding: 0.75rem 1rem;
            position: sticky;
            bottom: 0;
            z-index: 900;
            flex-shrink: 0;
            width: 100%;
            transition: all 0.3s ease;
        }

        .input-inner {
            max-width: 768px;
            margin: 0 auto;
            width: 100%;
        }

        .input-wrapper {
            background: var(--bg-primary);
            border: 1px solid var(--border-color);
            border-radius: 1.5rem;
            padding: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            transition: all 0.3s ease;
        }

        .input-controls {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .input-btn {
            width: 40px;
            height: 40px;
            border-radius: 0.5rem;
            border: 1px solid var(--border-color);
            background: var(--bg-surface);
            color: var(--text-secondary);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.125rem;
            transition: all 0.2s ease;
        }

        .input-btn:active {
            transform: scale(0.95);
        }

        .input-btn.primary {
            background: var(--primary-color);
            border-color: var(--primary-color);
            color: white;
        }

        .input-btn.primary:active {
            background: var(--primary-dark);
        }

        .input-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .input-textarea {
            flex: 1;
            border: none;
            outline: none;
            background: transparent;
            color: var(--text-primary);
            font-family: inherit;
            font-size: 1rem;
            line-height: 1.5;
            resize: none;
            min-height: 24px;
            max-height: 120px;
            padding: 0.25rem 0;
            transition: all 0.3s ease;
        }

        /* Voice Recording - FIXED FOR MOBILE RESPONSIVENESS */
        .voice-recording {
            position: fixed;
            bottom: 140px;
            left: 1rem;
            right: 1rem;
            background: var(--bg-surface);
            border: 1px solid var(--border-color);
            border-radius: 1rem;
            padding: 1rem 1.5rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 20px 25px -5px rgba(0,0,0,0.1);
            z-index: 1000; /* Increased z-index */
            animation: slideUp 0.2s ease-out;
            max-width: 400px;
            margin: 0 auto;
        }

        .voice-recording-dots {
            display: flex;
            gap: 0.25rem;
        }

        .voice-recording-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--danger-color);
            animation: voicePulse 1s infinite ease-in-out;
        }

        .voice-recording-dot:nth-child(1) { animation-delay: 0s; }
        .voice-recording-dot:nth-child(2) { animation-delay: 0.2s; }
        .voice-recording-dot:nth-child(3) { animation-delay: 0.4s; }

        /* Toast */
        .toast-container {
            position: fixed;
            top: 1rem;
            right: 1rem;
            z-index: 9999;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            max-width: 300px;
        }

        .toast {
            background: var(--bg-surface);
            border: 1px solid var(--border-color);
            border-radius: 0.5rem;
            box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);
            padding: 0.75rem 1rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            transform: translateX(400px);
            transition: transform 0.3s ease;
        }

        .toast.show {
            transform: translateX(0);
        }

        .toast.success {
            border-left: 3px solid var(--accent-color);
        }

        .toast.error {
            border-left: 3px solid var(--danger-color);
        }

        .toast.warning {
            border-left: 3px solid #f59e0b;
        }

        .toast.info {
            border-left: 3px solid #0ea5e9;
        }

        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes typing {
            0%, 60%, 100% {
                transform: translateY(0);
                opacity: 0.6;
            }
            30% {
                transform: translateY(-4px);
                opacity: 1;
            }
        }

        @keyframes voicePulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }

        /* Utility */
        .hidden {
            display: none !important;
        }

        .visually-hidden {
            position: absolute !important;
            width: 1px !important;
            height: 1px !important;
            padding: 0 !important;
            margin: -1px !important;
            overflow: hidden !important;
            clip: rect(0, 0, 0, 0) !important;
            border: 0 !important;
        }

        /* Responsive */
        @media (max-width: 768px) {
            .app-header {
                padding: 0.5rem;
            }
            
            .language-select {
                min-width: 120px;
                font-size: 0.8125rem;
            }
            
            .chat-container {
                padding: 0.5rem;
            }
            
            .welcome-card {
                padding: 1rem;
                margin: 0.5rem;
            }
            
            .features-grid {
                grid-template-columns: repeat(2, 1fr);
                gap: 0.5rem;
            }
            
            .feature-card {
                padding: 0.75rem;
            }
            
            .message-content {
                max-width: 80%;
            }
            
            .input-section,
            .processing-indicator {
                padding: 0.5rem;
            }
            
            .input-wrapper {
                padding: 0.5rem;
            }
            
            .preview-section {
                left: 0.5rem;
                right: 0.5rem;
                bottom: 100px;
            }
            
            .preview-card {
                max-width: 250px;
            }
            
            /* Mobile-specific voice recording */
            .voice-recording {
                bottom: 120px;
                left: 0.5rem;
                right: 0.5rem;
                padding: 0.875rem 1rem;
                max-width: none;
            }
        }

        @media (min-width: 768px) {
            .features-grid {
                grid-template-columns: repeat(3, 1fr);
                gap: 1rem;
            }
            
            .voice-recording {
                left: 50%;
                transform: translateX(-50%);
                width: 90%;
                max-width: 400px;
            }
        }
    </style>
</head>
<body>
    <!-- Toast Container -->
    <div id="toastContainer" class="toast-container"></div>
    
    <!-- Voice Recording - REMOVED INLINE STYLE -->
    <div id="voiceRecording" class="voice-recording hidden">
        <div class="voice-recording-dots">
            <div class="voice-recording-dot"></div>
            <div class="voice-recording-dot"></div>
            <div class="voice-recording-dot"></div>
        </div>
        <div id="voiceRecordingText">Listening...</div>
        <button class="control-btn" onclick="stopVoiceRecording()">
            <i class="bi bi-stop-fill"></i>
        </button>
    </div>
    
    <!-- Main App -->
    <div class="app-container">
        <!-- Header -->
        <header class="app-header">
            <div class="header-content">
                <div class="header-brand">
                    <div class="brand-logo">
                        <i class="bi bi-robot"></i>
                    </div>
                    <div class="brand-text">
                        <h1 class="brand-title">Vision AI</h1>
                        <p class="brand-subtitle">With Memory • 15MB Images</p>
                    </div>
                </div>
                
                <div class="header-controls">
                    <select id="languageSelect" class="language-select">
                        <option value="English">English</option>
                        <option value="Tamil">தமிழ் (Tamil)</option>
                        <option value="Hindi">हिन्दी (Hindi)</option>
                        <option value="Telugu">తెలుగు (Telugu)</option>
                        <option value="Kannada">ಕನ್ನಡ (Kannada)</option>
                        <option value="Malayalam">മലയാളം (Malayalam)</option>
                        <option value="Marathi">मराठी (Marathi)</option>
                        <option value="Gujarati">ગુજરાતી (Gujarati)</option>
                        <option value="Bengali">বাংলা (Bengali)</option>
                        <option value="Punjabi">ਪੰਜਾਬੀ (Punjabi)</option>
                        <option value="Urdu">اردو (Urdu)</option>
                        <option value="Spanish">Español (Spanish)</option>
                        <option value="French">Français (French)</option>
                        <option value="German">Deutsch (German)</option>
                        <option value="Chinese">中文 (Chinese)</option>
                        <option value="Japanese">日本語 (Japanese)</option>
                        <option value="Korean">한국어 (Korean)</option>
                        <option value="Arabic">العربية (Arabic)</option>
                        <option value="Portuguese">Português (Portuguese)</option>
                    </select>
                    
                    <button class="control-btn" onclick="toggleTheme()" id="themeBtn" title="Toggle theme">
                        <i id="themeIcon" class="bi bi-moon"></i>
                    </button>
                    
                    <button class="control-btn" onclick="toggleMute()" id="muteBtn" title="Toggle speech">
                        <i id="muteIcon" class="bi bi-volume-up"></i>
                    </button>
                    
                    <button class="control-btn" onclick="clearChat()" title="Clear chat">
                        <i class="bi bi-trash"></i>
                    </button>
                </div>
            </div>
        </header>
        
        <!-- Chat Container -->
        <div id="chatContainer" class="chat-container">
            <div class="chat-inner">
                <!-- Welcome Section -->
                <div class="welcome-section" id="welcomeSection">
                    <div class="welcome-card">
                        <div class="welcome-icon">
                            <i class="bi bi-lightning-charge"></i>
                        </div>
                        <h1 class="welcome-title">Vision AI with Memory</h1>
                        <p class="welcome-subtitle">Upload images for AI-powered analysis. Supports 25+ languages and remembers conversation context until cleared. Max image size: 15MB.</p>
                        
                        <div class="features-grid">
                            <div class="feature-card" onclick="openFileBrowser()">
                                <div class="feature-icon">
                                    <i class="bi bi-images"></i>
                                </div>
                                <h3 class="feature-title">Upload Image</h3>
                                <p class="feature-description">Select from gallery</p>
                            </div>
                            
                            <div class="feature-card" onclick="startVoiceRecording()">
                                <div class="feature-icon">
                                    <i class="bi bi-mic"></i>
                                </div>
                                <h3 class="feature-title">Voice Input</h3>
                                <p class="feature-description">Tap to record voice</p>
                            </div>
                            
                            <div class="feature-card" onclick="showFormats()">
                                <div class="feature-icon">
                                    <i class="bi bi-lightning"></i>
                                </div>
                                <h3 class="feature-title">Fast Processing</h3>
                                <p class="feature-description">Optimized for speed</p>
                            </div>
                            
                            <div class="feature-card" onclick="showMemoryInfo()">
                                <div class="feature-icon">
                                    <i class="bi bi-memory"></i>
                                </div>
                                <h3 class="feature-title">Context Memory</h3>
                                <p class="feature-description">Remembers chat history</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Image Preview -->
        <div id="previewContainer" class="preview-section hidden">
            <div class="preview-card">
                <img id="previewImage" class="preview-image" alt="Preview">
                <div class="preview-info">
                    <div class="preview-title" id="previewTitle">Image Preview</div>
                    <div class="preview-size" id="previewSize">Ready to send</div>
                </div>
                <button class="preview-remove" onclick="removeImagePreview()">
                    <i class="bi bi-x-lg"></i>
                </button>
            </div>
        </div>
        
        <!-- Processing Indicator (Replaces Input Section) -->
        <div id="processingIndicator" class="processing-indicator hidden">
            <div class="processing-dots">
                <div class="processing-dot"></div>
                <div class="processing-dot"></div>
                <div class="processing-dot"></div>
            </div>
            <div class="processing-text">Processing with context...</div>
        </div>
        
        <!-- Input Section -->
        <div id="inputSection" class="input-section">
            <div class="input-inner">
                <!-- Hidden file inputs -->
                <input type="file" id="fileBrowserInput" class="visually-hidden" accept="*/*" multiple>
                
                <div class="input-wrapper">
                    <div class="input-controls">
                        <button class="input-btn" onclick="openFileBrowser()" title="Attach image" id="attachBtn">
                            <i class="bi bi-paperclip"></i>
                        </button>
                        
                        <button class="input-btn" id="voiceBtn" onclick="toggleVoiceRecording()" title="Record voice">
                            <i class="bi bi-mic"></i>
                        </button>
                    </div>
                    
                    <textarea 
                        id="messageInput" 
                        class="input-textarea" 
                        placeholder="Type your message or attach image..."
                        rows="1"
                        oninput="autoResize(this)"
                    ></textarea>
                    
                    <button class="input-btn primary" onclick="sendMessage()" id="sendBtn" title="Send message">
                        <i class="bi bi-send"></i>
                    </button>
                </div>
            </div>
        </div>
    </div>

    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    
    <script>
        // ===== STATE MANAGEMENT =====
        const state = {
            selectedFile: null,
            isProcessing: false,
            isMuted: localStorage.getItem('isMuted') === 'true',
            imageDataUrl: null,
            isRecording: false,
            speechRecognition: null,
            selectedLanguage: localStorage.getItem('selectedLanguage') || 'English',
            currentTheme: localStorage.getItem('theme') || 'light',
            hasMessages: false,
            isSpeaking: false,
            lastRequestTime: 0,
            requestCooldown: 1000, // 1 second cooldown between requests
            conversationId: localStorage.getItem('conversationId') || generateConversationId(),
            memoryEnabled: true
        };

        // Generate unique conversation ID
        function generateConversationId() {
            return 'conv_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        }

        // ===== DOM ELEMENTS =====
        const chatContainer = document.getElementById('chatContainer');
        const chatInner = document.querySelector('.chat-inner');
        const messageInput = document.getElementById('messageInput');
        const inputSection = document.getElementById('inputSection');
        const processingIndicator = document.getElementById('processingIndicator');
        const previewContainer = document.getElementById('previewContainer');
        const previewImage = document.getElementById('previewImage');
        const previewTitle = document.getElementById('previewTitle');
        const previewSize = document.getElementById('previewSize');
        const fileBrowserInput = document.getElementById('fileBrowserInput');
        const welcomeSection = document.getElementById('welcomeSection');
        const voiceBtn = document.getElementById('voiceBtn');
        const sendBtn = document.getElementById('sendBtn');
        const languageSelect = document.getElementById('languageSelect');
        const muteIcon = document.getElementById('muteIcon');
        const themeIcon = document.getElementById('themeIcon');
        const voiceRecording = document.getElementById('voiceRecording');
        const voiceRecordingText = document.getElementById('voiceRecordingText');

        // ===== INITIALIZATION =====
        function init() {
            console.log('🚀 Initializing Vision AI with Memory...');
            console.log('📝 Conversation ID:', state.conversationId);
            
            // Save conversation ID to localStorage
            localStorage.setItem('conversationId', state.conversationId);
            
            // Apply theme
            applyTheme();
            
            // Load preferences
            loadPreferences();
            
            // Setup event listeners
            setupEventListeners();
            
            // Initialize speech recognition
            initSpeechRecognition();
            
            // Auto-resize textarea
            autoResize(messageInput);
            
            // Focus input
            setTimeout(() => {
                messageInput.focus();
                showToast('Vision AI Ready - Remembers conversation context', 'success');
                showToast('Responses will be in the selected language above', 'info');
            }, 300);
        }

        function loadPreferences() {
            // Language
            const savedLang = localStorage.getItem('selectedLanguage');
            if (savedLang) {
                languageSelect.value = savedLang;
                state.selectedLanguage = savedLang;
            }
            
            // Theme
            const savedTheme = localStorage.getItem('theme');
            if (savedTheme === 'dark') {
                document.documentElement.setAttribute('data-bs-theme', 'dark');
                themeIcon.className = 'bi bi-sun';
                state.currentTheme = 'dark';
            }
            
            // Mute state
            updateMuteButton();
        }

        function applyTheme() {
            if (state.currentTheme === 'dark') {
                document.documentElement.setAttribute('data-bs-theme', 'dark');
                themeIcon.className = 'bi bi-sun';
            } else {
                document.documentElement.setAttribute('data-bs-theme', 'light');
                themeIcon.className = 'bi bi-moon';
            }
        }

        function setupEventListeners() {
            // File input changes
            fileBrowserInput.addEventListener('change', handleFileSelect);
            
            // Textarea
            messageInput.addEventListener('input', () => autoResize(messageInput));
            messageInput.addEventListener('keydown', handleKeyDown);
            
            // Language selector
            languageSelect.addEventListener('change', function() {
                state.selectedLanguage = this.value;
                localStorage.setItem('selectedLanguage', this.value);
                showToast(`Responses will be in: ${this.value}`, 'info');
            });
            
            // Handle page visibility
            document.addEventListener('visibilitychange', () => {
                if (document.hidden && state.isRecording) {
                    stopVoiceRecording();
                }
            });
            
            // Handle speech synthesis end
            if ('speechSynthesis' in window) {
                speechSynthesis.addEventListener('end', () => {
                    state.isSpeaking = false;
                    console.log('🔊 Speech ended, input re-enabled');
                });
            }
        }

        function handleKeyDown(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                if (canSendMessage()) {
                    sendMessage();
                }
            }
        }

        function canSendMessage() {
            const now = Date.now();
            if (now - state.lastRequestTime < state.requestCooldown) {
                showToast('Please wait a moment...', 'warning');
                return false;
            }
            
            if (state.isProcessing) {
                showToast('Processing previous request...', 'warning');
                return false;
            }
            
            return true;
        }

        // ===== FILE SELECTION =====
        function openFileBrowser() {
            if (!canSendMessage()) return;
            fileBrowserInput.click();
        }

        function handleFileSelect(event) {
            const file = event.target.files[0];
            if (file) {
                handleSelectedFile(file);
            }
            event.target.value = '';
        }

        function handleSelectedFile(file) {
            // Validate file size (15MB)
            if (file.size > 15 * 1024 * 1024) {
                showToast('Image must be less than 15MB', 'error');
                return;
            }
            
            state.selectedFile = file;
            const reader = new FileReader();
            
            reader.onloadstart = () => {
                showToast('Optimizing image...', 'info');
            };
            
            reader.onload = (e) => {
                state.imageDataUrl = e.target.result;
                previewImage.src = state.imageDataUrl;
                previewTitle.textContent = file.name;
                previewSize.textContent = formatFileSize(file.size);
                previewContainer.classList.remove('hidden');
                
                // Auto-focus textarea for quick typing
                messageInput.focus();
                
                showToast('Image ready for upload (Max 15MB)', 'success');
            };
            
            reader.onerror = () => {
                showToast('Error reading image file', 'error');
                state.selectedFile = null;
                state.imageDataUrl = null;
            };
            
            reader.readAsDataURL(file);
        }

        function formatFileSize(bytes) {
            if (bytes < 1024) return bytes + ' B';
            if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
            return (bytes / 1048576).toFixed(1) + ' MB';
        }

        function removeImagePreview() {
            state.selectedFile = null;
            state.imageDataUrl = null;
            fileBrowserInput.value = '';
            previewContainer.classList.add('hidden');
        }

        function showFormats() {
            showToast('Supports: JPG, PNG, GIF, WEBP, BMP, TIFF (Max 15MB)', 'info');
        }

        function showMemoryInfo() {
            showToast('AI remembers conversation context until chat is cleared', 'info');
        }

        // ===== VOICE RECOGNITION =====
        function initSpeechRecognition() {
            if ('webkitSpeechRecognition' in window) {
                state.speechRecognition = new webkitSpeechRecognition();
            } else if ('SpeechRecognition' in window) {
                state.speechRecognition = new SpeechRecognition();
            } else {
                voiceBtn.style.display = 'none';
                return;
            }
            
            state.speechRecognition.continuous = false;
            state.speechRecognition.interimResults = false;
            state.speechRecognition.lang = getSpeechRecognitionLang();
            
            state.speechRecognition.onstart = () => {
                state.isRecording = true;
                voiceBtn.classList.add('active');
                voiceRecording.classList.remove('hidden');
                // Force reflow for smooth animation
                voiceRecording.style.animation = 'none';
                setTimeout(() => {
                    voiceRecording.style.animation = 'slideUp 0.2s ease-out';
                }, 10);
            };
            
            state.speechRecognition.onresult = (event) => {
                if (event.results.length > 0) {
                    const transcript = event.results[0][0].transcript;
                    messageInput.value = transcript;
                    autoResize(messageInput);
                    showToast('Voice input received', 'success');
                }
            };
            
            state.speechRecognition.onerror = (event) => {
                state.isRecording = false;
                voiceBtn.classList.remove('active');
                voiceRecording.classList.add('hidden');
                
                if (event.error === 'not-allowed') {
                    showToast('Microphone permission required', 'error');
                }
            };
            
            state.speechRecognition.onend = () => {
                state.isRecording = false;
                voiceBtn.classList.remove('active');
                voiceRecording.classList.add('hidden');
            };
        }

        function getSpeechRecognitionLang() {
            const langMap = {
                'English': 'en-US',
                'Tamil': 'ta-IN',
                'Hindi': 'hi-IN',
                'Telugu': 'te-IN',
                'Kannada': 'kn-IN',
                'Malayalam': 'ml-IN',
                'Marathi': 'mr-IN',
                'Gujarati': 'gu-IN',
                'Bengali': 'bn-IN',
                'Punjabi': 'pa-IN',
                'Spanish': 'es-ES',
                'French': 'fr-FR',
                'German': 'de-DE',
                'Chinese': 'zh-CN',
                'Japanese': 'ja-JP',
                'Korean': 'ko-KR',
                'Arabic': 'ar-SA'
            };
            return langMap[state.selectedLanguage] || 'en-US';
        }

        function toggleVoiceRecording() {
            if (!canSendMessage()) return;
            
            if (state.isRecording) {
                stopVoiceRecording();
            } else {
                startVoiceRecording();
            }
        }

        // FIX 1: Simplified voice recording start (removed getUserMedia)
        function startVoiceRecording() {
            if (!state.speechRecognition) {
                showToast('Voice input not available', 'error');
                return;
            }
            try {
                state.speechRecognition.lang = getSpeechRecognitionLang();
                state.speechRecognition.start();
            } catch (err) {
                console.error(err);
                showToast('Microphone error or already listening', 'error');
            }
        }

        function stopVoiceRecording() {
            if (state.speechRecognition && state.isRecording) {
                state.speechRecognition.stop();
            }
        }

        // ===== MESSAGE FUNCTIONS =====
        async function sendMessage() {
            if (!canSendMessage()) return;
            
            const message = messageInput.value.trim();
            const lang = state.selectedLanguage;
            
            if (!message && !state.selectedFile) {
                showToast('Please type a message or attach an image', 'warning');
                return;
            }
            
            state.lastRequestTime = Date.now();
            
            // Store current state
            const currentImageData = state.imageDataUrl;
            const currentMessage = message;
            
            // Clear input immediately for next user
            messageInput.value = '';
            autoResize(messageInput);
            
            // Hide welcome section only on first message
            if (!state.hasMessages && welcomeSection) {
                welcomeSection.classList.add('hidden');
                state.hasMessages = true;
            }
            
            // Hide preview
            previewContainer.classList.add('hidden');
            
            // HIDE INPUT SECTION AND SHOW PROCESSING INDICATOR
            inputSection.classList.add('hidden');
            processingIndicator.classList.remove('hidden');
            
            // Add user message
            addUserMessage(currentMessage, currentImageData);
            
            // Set processing state
            state.isProcessing = true;
            
            // Prepare form data
            const formData = new FormData();
            formData.append('message', currentMessage || (state.selectedFile ? 'Describe this image' : ''));
            formData.append('language', lang);
            formData.append('conversation_id', state.conversationId);
            if (state.selectedFile) {
                formData.append('image', state.selectedFile);
            }
            
            try {
                const startTime = Date.now();
                const response = await fetch('/chat', {
                    method: 'POST',
                    body: formData
                });
                
                const processingTime = Date.now() - startTime;
                console.log(`⏱️ Response time: ${processingTime}ms`);
                console.log(`🧠 Using conversation ID: ${state.conversationId}`);
                console.log(`🌐 Selected language: ${lang}`);
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                const data = await response.json();
                
                if (data.status === 'error') {
                    throw new Error(data.answer);
                }
                
                // Add AI response
                addAiResponse(data.answer, data.lang_code);
                
                showToast(`Response in ${lang} (${processingTime}ms)`, 'success');
                
                // Auto-speak if not muted
                if (!state.isMuted) {
                    state.isSpeaking = true;
                    setTimeout(() => {
                        speakText(data.answer, data.lang_code);
                    }, 100);
                }
                
            } catch (error) {
                console.error('Send message error:', error);
                addErrorMessage(error.message || 'Failed to get response');
                showToast('Failed to get response', 'error');
            } finally {
                // SHOW INPUT SECTION AND HIDE PROCESSING INDICATOR
                processingIndicator.classList.add('hidden');
                inputSection.classList.remove('hidden');
                
                // Reset image state
                state.selectedFile = null;
                state.imageDataUrl = null;
                fileBrowserInput.value = '';
                
                // Reset processing state
                state.isProcessing = false;
                
                // Auto-focus for next message
                setTimeout(() => {
                    messageInput.focus();
                }, 100);
            }
        }

        function addUserMessage(message, imageData) {
            const messageId = 'msg_' + Date.now();
            
            // Escape message for HTML
            const escapedMessage = escapeHtml(message || '');
            
            // Build message HTML
            let messageHtml = `
                <div class="message-container" id="${messageId}">
                    <div class="message user">
                        <div class="message-avatar user">
                            <i class="bi bi-person"></i>
                        </div>
                        <div class="message-content">
                            <div class="message-bubble user">
            `;
            
            // Add image if exists
            if (imageData) {
                messageHtml += `<img src="${imageData}" class="message-image" alt="Uploaded image">`;
            }
            
            // Add text if exists
            if (message) {
                messageHtml += `<div style="margin-top: ${imageData ? '0.5rem' : '0'}">${escapedMessage}</div>`;
            } else if (imageData) {
                messageHtml += `<div style="margin-top: 0.5rem; font-style: italic; opacity: 0.8;">Image analysis request</div>`;
            }
            
            // Close message
            messageHtml += `
                            </div>
                        </div>
                    </div>
                </div>
            `;
            
            chatInner.insertAdjacentHTML('beforeend', messageHtml);
            scrollToBottom();
        }

        function addAiResponse(answer, langCode) {
            const messageId = 'msg_' + Date.now();
            const escapedAnswer = escapeHtml(answer);
            const messageHtml = `
                <div class="message-container" id="${messageId}">
                    <div class="message">
                        <div class="message-avatar ai">
                            <i class="bi bi-robot"></i>
                        </div>
                        <div class="message-content">
                            <div class="message-bubble ai">
                                ${answer}
                                <div class="message-actions">
                                    <button class="action-btn" onclick="copyToClipboard('${escapedAnswer.replace(/'/g, "\\'")}')">
                                        <i class="bi bi-copy"></i> Copy
                                    </button>
                                    <button class="action-btn" id="speak_${messageId}" data-text="${escapedAnswer}" data-lang="${langCode}" onclick="handleSpeakClick(this)">
                                        <i class="bi bi-volume-up"></i> Speak
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            
            chatInner.insertAdjacentHTML('beforeend', messageHtml);
            scrollToBottom();
        }

        function handleSpeakClick(button) {
            const text = button.getAttribute('data-text');
            const lang = button.getAttribute('data-lang');
            speakText(text, lang);
        }

        function addErrorMessage(errorMsg) {
            const messageId = 'msg_' + Date.now();
            const messageHtml = `
                <div class="message-container" id="${messageId}">
                    <div class="message">
                        <div class="message-avatar ai">
                            <i class="bi bi-exclamation-triangle"></i>
                        </div>
                        <div class="message-content">
                            <div class="message-bubble ai">
                                <div style="color: var(--danger-color); font-weight: 600; margin-bottom: 8px;">
                                    <i class="bi bi-exclamation-circle"></i> Error
                                </div>
                                <div>${errorMsg}</div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            
            chatInner.insertAdjacentHTML('beforeend', messageHtml);
            scrollToBottom();
        }

        async function clearChat() {
            if (!canSendMessage()) return;
            
            if (confirm('Clear all chat messages and conversation memory?')) {
                const messages = chatInner.querySelectorAll('.message-container');
                messages.forEach(msg => msg.remove());
                
                if (welcomeSection) {
                    welcomeSection.classList.remove('hidden');
                }
                state.hasMessages = false;
                
                // Clear conversation memory on server
                try {
                    await fetch('/clear_conversation', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            conversation_id: state.conversationId
                        })
                    });
                    
                    // Generate new conversation ID
                    state.conversationId = generateConversationId();
                    localStorage.setItem('conversationId', state.conversationId);
                    console.log('🧠 New conversation ID:', state.conversationId);
                    
                    showToast('Chat and memory cleared', 'success');
                } catch (error) {
                    console.error('Error clearing conversation:', error);
                    showToast('Chat cleared locally', 'info');
                }
            }
        }

        // ===== UI UTILITIES =====
        function autoResize(textarea) {
            textarea.style.height = 'auto';
            const newHeight = Math.min(textarea.scrollHeight, 120);
            textarea.style.height = newHeight + 'px';
        }

        function scrollToBottom() {
            setTimeout(() => {
                chatContainer.scrollTo({
                    top: chatContainer.scrollHeight,
                    behavior: 'smooth'
                });
            }, 100);
        }

        function updateMuteButton() {
            muteIcon.className = state.isMuted ? 'bi bi-volume-mute' : 'bi bi-volume-up';
        }

        function toggleMute() {
            state.isMuted = !state.isMuted;
            localStorage.setItem('isMuted', state.isMuted);
            updateMuteButton();
            showToast(state.isMuted ? 'Speech muted' : 'Speech enabled', 'info');
        }

        function toggleTheme() {
            state.currentTheme = state.currentTheme === 'light' ? 'dark' : 'light';
            localStorage.setItem('theme', state.currentTheme);
            applyTheme();
        }

        // ===== TEXT-TO-SPEECH =====
        function getFullLangCode(shortCode) {
            const langCodeMap = {
                'en': 'en-US',
                'ta': 'ta-IN',
                'hi': 'hi-IN',
                'te': 'te-IN',
                'kn': 'kn-IN',
                'ml': 'ml-IN',
                'mr': 'mr-IN',
                'gu': 'gu-IN',
                'bn': 'bn-IN',
                'pa': 'pa-IN',
                'ur': 'ur-PK',
                'or': 'or-IN',
                'as': 'as-IN',
                'es': 'es-ES',
                'fr': 'fr-FR',
                'de': 'de-DE',
                'zh-CN': 'zh-CN',
                'ja': 'ja-JP',
                'ko': 'ko-KR',
                'ar': 'ar-SA',
                'pt': 'pt-BR',
                'it': 'it-IT',
                'ru': 'ru-RU',
                'tr': 'tr-TR',
                'vi': 'vi-VN',
                'th': 'th-TH',
                'id': 'id-ID',
                'tl': 'tl-PH'
            };
            return langCodeMap[shortCode] || shortCode;
        }

        function getAvailableVoices() {
            if (!('speechSynthesis' in window)) return [];
            return window.speechSynthesis.getVoices();
        }

        function speakText(text, langCode) {
            if (!text || !('speechSynthesis' in window)) {
                showToast('Speech synthesis not available', 'warning');
                return;
            }
            
            // Cancel any ongoing speech
            window.speechSynthesis.cancel();
            state.isSpeaking = true;
            
            // Get full language code
            const fullLangCode = getFullLangCode(langCode);
            
            // Clean text - REMOVED THE 300 CHARACTER TRUNCATION
            let cleanText = text
                .replace(/[\[\]\(\)*_`]/g, ' ')
                .replace(/https?:\/\/\S+/g, 'link')
                .replace(/\s+/g, ' ')
                .trim();
            
            // If text is very long, split into sentences for better speech
            const sentences = cleanText.match(/[^.!?]+[.!?]+/g) || [cleanText];
            
            // Speak each sentence one by one
            let currentSentence = 0;
            
            function speakNextSentence() {
                if (currentSentence >= sentences.length) {
                    state.isSpeaking = false;
                    return;
                }
                
                let sentence = sentences[currentSentence].trim();
                if (!sentence) {
                    currentSentence++;
                    speakNextSentence();
                    return;
                }
                
                const utterance = new SpeechSynthesisUtterance(sentence);
                utterance.lang = fullLangCode;
                utterance.rate = 1.0;
                utterance.pitch = 1.0;
                utterance.volume = 1.0;
                
                // Check if voice available for language
                const voices = getAvailableVoices();
                const preferredVoice = voices.find(voice => 
                    voice.lang.startsWith(langCode.split('-')[0])
                );
                
                if (preferredVoice) {
                    utterance.voice = preferredVoice;
                } else {
                    // Fallback to English or any available voice
                    const fallbackVoice = voices.find(voice => voice.lang.startsWith('en'));
                    if (fallbackVoice) {
                        utterance.voice = fallbackVoice;
                    }
                }
                
                utterance.onend = () => {
                    currentSentence++;
                    speakNextSentence();
                };
                
                utterance.onerror = (error) => {
                    console.error('🔊 Speech error:', error.error);
                    currentSentence++;
                    speakNextSentence();
                };
                
                try {
                    window.speechSynthesis.speak(utterance);
                } catch (error) {
                    console.error('Error speaking:', error);
                    currentSentence++;
                    speakNextSentence();
                }
            }
            
            // Start speaking
            speakNextSentence();
        }

        // ===== NOTIFICATIONS =====
        function showToast(message, type = 'info') {
            const icons = {
                'success': '<i class="bi bi-check-circle"></i>',
                'error': '<i class="bi bi-x-circle"></i>',
                'warning': '<i class="bi bi-exclamation-triangle"></i>',
                'info': '<i class="bi bi-info-circle"></i>'
            };
            
            const toast = document.createElement('div');
            toast.className = `toast ${type}`;
            toast.innerHTML = `
                <span class="toast-icon">${icons[type]}</span>
                <span class="toast-content">${message}</span>
            `;
            
            document.getElementById('toastContainer').appendChild(toast);
            setTimeout(() => toast.classList.add('show'), 10);
            
            setTimeout(() => {
                toast.classList.remove('show');
                setTimeout(() => toast.remove(), 300);
            }, 2000);
        }

        function copyToClipboard(text) {
            navigator.clipboard.writeText(text).then(() => {
                showToast('Copied to clipboard!', 'success');
            }).catch(() => {
                showToast('Failed to copy', 'error');
            });
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // ===== INITIALIZATION =====
        window.addEventListener('load', init);
        
        // Handle speech synthesis voices - properly initialize
        if ('speechSynthesis' in window) {
            // Some browsers need this to load voices
            speechSynthesis.onvoiceschanged = () => {
                const voices = speechSynthesis.getVoices();
                console.log(`🔊 Loaded ${voices.length} voices`);
            };
            
            // Trigger voice loading on some browsers
            try {
                speechSynthesis.getVoices();
            } catch (e) {
                console.log('Voice loading: ' + e);
            }
        }
    </script>
</body>
</html>
'''

if __name__ == "__main__":
    def get_local_ip():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('8.8.8.8', 1))
            ip = s.getsockname()[0]
        except Exception:
            ip = '127.0.0.1'
        finally:
            s.close()
        return ip

    print(f"\n" + "═"*60)
    print(f"🚀 VISION AI ASSISTANT - WITH MEMORY".center(60))
    print(f"═"*60)
    print(f"\n📡 Network: http://{get_local_ip()}:5000")
    print(f"🖥️  Local:    http://localhost:5000")
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    
    print(f"\n✅ MAJOR IMPROVEMENTS:")
    print(f"   1. 🧠 CHAT MEMORY: AI remembers conversation context until cleared")
    print(f"   2. 📸 CAMERA REMOVED: Only file upload from gallery")
    print(f"   3. 📈 IMAGE SIZE: Increased to 15MB maximum")
    print(f"   4. ⚡ Fast switching between guests")
    print(f"   5. 🌐 25+ languages with context memory")
    
    print(f"\n🧠 MEMORY FEATURES:")
    print(f"   • Remembers all conversation history")
    print(f"   • Answers follow-up questions based on context")
    print(f"   • Memory persists until chat is cleared")
    print(f"   • Separate memory per conversation")
    
    print(f"\n🌐 LANGUAGE HANDLING:")
    print(f"   • Input automatically detected & translated to English for AI")
    print(f"   • Response always in selected language above")
    print(f"   • Skips translation when both input (ASCII) and target are English")
    print(f"   • Supports 25+ languages")
    
    print(f"\n📸 IMAGE UPLOAD:")
    print(f"   • Max size: 15MB")
    print(f"   • Auto-optimization for faster processing (512x512 max)")
    print(f"   • Gallery selection only (camera removed)")
    
    print(f"\n🔄 PROCESSING INDICATOR:")
    print(f"   • Input section disappears during processing")
    print(f"   • Shows 'Processing...' indicator instead")
    print(f"   • Returns to normal when response received")
    
    print(f"\n🚀 STARTING SERVER WITH MEMORY...")
    print(f"═"*60)
    
    try:
        # FIX: Removed ssl_context='adhoc' for compatibility with PC browsers
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print(f"\n\n👋 Server stopped.")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
