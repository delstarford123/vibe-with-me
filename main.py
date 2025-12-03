import sys
import os
from flask import Flask, render_template, request, jsonify

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.append(src_dir)

app = Flask(__name__)

# Try importing Gemini
try:
    from src.gemini_brain import generate_gemini_response
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️ Gemini module not found.")

@app.route('/')
def home():
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_text = data.get('text', '')
    mode = data.get('mode', 'relationship')
    user_data = data.get('userData', {})
    image_data = data.get('image', None)

    response_text = ""

    # --- CLOUD-ONLY LOGIC ---
    # We rely 100% on Gemini because it is smarter and doesn't crash the free server.
    if GEMINI_AVAILABLE:
        print(f"✨ Routing '{mode}' to Gemini...")
        response_text = generate_gemini_response(user_text, mode, user_data, image_data)
        
        # If Gemini fails, give a helpful error message instead of crashing
        if "Error" in response_text or "failed" in response_text:
            print(f"⚠️ Gemini API Error: {response_text}")
            response_text = "I'm having trouble connecting to my brain. (Check Render API Key)"
    else:
        response_text = "System Error: Gemini Brain missing."

    return jsonify({'response': response_text})

if __name__ == '__main__':
    # Local testing can still use debug mode
    app.run(debug=True, port=5000)