import sys
import os
from flask import Flask, render_template, request, jsonify

# --- PATH SETUP ---
# Ensure Python looks inside the 'src' folder for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.append(src_dir)

app = Flask(__name__)
 
# --- LAZY LOADING SETUP ---
# We start with bot = None so the server starts INSTANTLY.
# The AI only loads when the first person sends a message.
bot = None 

def get_bot():
    """"
    Lazy Loader: Loads the heavy AI model only when needed.
    This prevents Render from timing out during the 30-second startup limit.
    """
    global bot
    if bot is None:
        print("⏳ First request received. Loading AI Engine... (This may take a moment)")
        try:
            from predict import DualBot
            bot = DualBot()
            print("✅ AI Engine Loaded successfully.")
        except Exception as e:
            print(f"❌ Failed to load AI: {e}")
            return None
    return bot

@app.route('/')
def home():
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Extract User Data
    user_text = data.get('text', '')
    mode = data.get('mode', 'roast')
    user_data = data.get('userData', {}) # Example: {name: "John", gender: "male", age: 22}
    
    # Load the bot NOW (on the first request)
    ai_bot = get_bot()
    
    response_text = ""

    if ai_bot:
        try:
            # Pass user_data to the generator so it knows if it's a BF or GF
            response_text = ai_bot.generate(user_text, mode, user_data)
        except Exception as e:
            print(f"Generation Error: {e}")
            response_text = "I'm having a brain freeze. Ask me again in a second!"
    else:
        # Fallback / Mock Responses (If model crashes or files are missing)
        name = user_data.get('name', 'babe')
        
        if mode == 'relationship':
            response_text = f"I love you so much {name}! (Model Offline - Mock Response)"
        elif mode == 'roast':
            response_text = f"Nice try {name}, but I'm not in the mood. (Model Offline)"
        elif mode == 'friend':
            response_text = f"Yo {name}, what's up? (Model Offline)"
        else:
            response_text = f"I'm listening, {name}. (Model Offline)"

    return jsonify({'response': response_text})

if __name__ == '__main__':
    print(f"🌐 Server running at: http://127.0.0.1:5000")
    # For local testing only
    app.run(debug=True, port=5000)