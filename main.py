import sys
import os
from flask import Flask, render_template, request, jsonify

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.append(src_dir)

app = Flask(__name__)

# --- LAZY LOADERS ---
local_bot = None 

def get_local_bot():
    global local_bot
    if local_bot is None:
        try:
            print("⏳ Loading Local Brain (Backup)...")
            from src.predict import DualBot
            local_bot = DualBot()
        except:
            return None
    return local_bot

# Check if Gemini is available
try:
    from src.gemini_brain import generate_gemini_response
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

@app.route('/')
def home():
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_text = data.get('text', '')
    mode = data.get('mode', 'relationship') # Default to relationship
    user_data = data.get('userData', {})
    image_data = data.get('image', None)

    response_text = ""

    # --- INTELLIGENT ROUTING ---
    # We want Gemini to handle 'Relationship' mode because it's better at 
    # roleplaying a girlfriend/boyfriend than the local model.
    # We also use it for 'Smart' mode and Image analysis.
    use_gemini = GEMINI_AVAILABLE and (
        mode == 'relationship' or 
        mode == 'smart' or 
        mode == 'friend' or
        image_data
    )

    if use_gemini:
        print(f"✨ Routing '{mode}' to Gemini...")
        response_text = generate_gemini_response(user_text, mode, user_data, image_data)
        
        # Fallback if Gemini fails
        if "Error" in response_text or "failed" in response_text:
            print("⚠️ Gemini failed. Falling back to local brain.")
            use_gemini = False # Trigger fallback block below

    # --- FALLBACK: LOCAL BRAIN ---
    if not use_gemini:
        bot = get_local_bot()
        if bot:
            # Local brain can't see images, so we ignore image_data here
            response_text = bot.generate(user_text, mode, user_data)
        else:
            response_text = "System Offline. (Check logs)"

    return jsonify({'response': response_text})

if __name__ == '__main__':
    print(f"🌐 Server running at: http://127.0.0.1:5000")
    app.run(debug=True, port=5000)