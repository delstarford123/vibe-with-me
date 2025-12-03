import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

print(f"🔑 Key Loaded: {API_KEY[:10]}... (ends with ...{API_KEY[-5:]})")

# Using the exact model from your CURL command
model = "gemini-2.0-flash"
url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={API_KEY}"

payload = {
    "contents": [{"parts": [{"text": "Hello Gemini, are you online?"}]}]
}

print(f"📡 Connecting to {model}...")

try:
    response = requests.post(url, headers={"Content-Type": "application/json"}, json=payload)
    
    if response.status_code == 200:
        print("\n✅ SUCCESS!")
        print(response.json()['candidates'][0]['content']['parts'][0]['text'])
    else:
        print(f"\n❌ FAILED: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"Error: {e}")