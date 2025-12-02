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
print("🚀 STARTING LAUGH OUT AI SERVER")
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
    user_text = data.get('text', '')
    mode = data.get('mode', 'roast')
    
    response_text = ""

    if MODEL_LOADED and bot:
        try:
            response_text = bot.generate(user_text, mode)
        except Exception as e:
            response_text = f"Error: {str(e)}"
    else: 
        if mode == 'roast':
            response_text = "I'd roast you, but my files are missing. (Mock Mode)"
        else:
            response_text = "I am listening. (Mock Mode)"

    return jsonify({'response': response_text})

if __name__ == '__main__':
    print(f"🌐 Server running at: http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
    