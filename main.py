import sys
import os
from flask import Flask, render_template, request, jsonify

# Force Python to look in 'src' for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.append(src_dir)

app = Flask(__name__)
bot = None
MODEL_LOADED = False

print("\n=======================================")
print("🚀 STARTING VIBE AI SERVER")
print("=======================================")

try:
    from predict import DualBot
    bot = DualBot()
    MODEL_LOADED = True
    print("✅ AI Engine Initialized.")
except Exception as e:
    print(f"⚠️  AI Load Failed: {e}")
    print("👉  Running in MOCK MODE")

@app.route('/')
def home():
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Extract User Data
    user_text = data.get('text', '')
    mode = data.get('mode', 'roast')
    user_data = data.get('userData', {}) # {name: "John", gender: "male", age: 22}
    
    response_text = ""

    if MODEL_LOADED and bot:
        try:
            # Pass user_data to the generator
            response_text = bot.generate(user_text, mode, user_data)
        except Exception as e:
            print(f"Generation Error: {e}")
            response_text = "I'm blushing... I forgot what to say."
    else:
        # Mock Responses for testing without models
        name = user_data.get('name', 'babe')
        if mode == 'relationship':
            response_text = f"I love you so much {name}! You are my favorite person."
        elif mode == 'roast':
            response_text = f"Nice try {name}, but I'm not in the mood."
        else:
            response_text = f"I'm listening, {name}."

    return jsonify({'response': response_text})

if __name__ == '__main__':
    print(f"🌐 Server running at: http://127.0.0.1:5000")
    app.run(debug=True, port=5000)