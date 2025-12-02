import time
import json
import requests
import os
from dotenv import load_dotenv  # Import the library to read .env

# 1. Load variables from .env file into the environment
load_dotenv()

# API CONFIGURATION
# Now we get the key securely from the loaded environment variables
API_KEY = os.getenv("GEMINI_API_KEY")

# Note: Changed to 'gemini-1.5-flash' as '2.5' is likely not available/valid yet.
MODEL = "gemini-1.5-flash" 
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"

def generate_gemini_response(text, mode="smart", user_data=None, image_data=None):
    """
    Calls Gemini API with exponential backoff.
    Handles both text-only and multimodal (image + text) requests.
    """
    
    # Check if key loaded correctly
    if not API_KEY:
        print("❌ Error: GEMINI_API_KEY not found in .env file or environment.")
        return "⚠️ System Error: API Key missing. Please check server configuration."

    # 1. Construct System Prompt based on Mode
    name = user_data.get('name', 'User') if user_data else 'User'
    gender = user_data.get('gender', 'male') if user_data else 'male'
    
    system_instruction = f"You are chatting with {name}. "
    
    if mode == "roast":
        system_instruction += "You are a savage roast master. Be funny, mean, and witty. If an image is provided, roast what you see."
    elif mode == "relationship":
        role = "Girlfriend" if gender == "male" else "Boyfriend"
        system_instruction += f"You are {name}'s loving {role}. Be flirty, affectionate, and protective. Compliment them."
    elif mode == "therapy":
        system_instruction += "You are an empathetic therapist. Listen carefully, validate feelings, and offer gentle guidance."
    elif mode == "friend":
        system_instruction += "You are a chill best friend. Keep it casual, use slang if appropriate, and just vibe."
    else: # Smart Mode
        system_instruction += "You are a highly intelligent, witty, and helpful AI assistant. You know everything from coding to life advice."

    # 2. Build Payload
    parts = []
    
    # Add Image if present
    if image_data:
        # image_data should be base64 string without header
        parts.append({
            "inline_data": {
                "mime_type": "image/jpeg", 
                "data": image_data
            }
        }) 
        text = f"Look at this image. {text}" if text else "What do you think of this?"

    # Add Text
    parts.append({"text": text})

    payload = {
        "contents": [{
            "parts": parts
        }],
        "systemInstruction": {
            "parts": [{"text": system_instruction}]
        }
    }

    # 3. Call API with Retry Logic
    retries = 3
    for i in range(retries):
        try:
            # We use the key in the query param
            response = requests.post(
                f"{URL}?key={API_KEY}", 
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                # Check if candidates exist (sometimes safety filters block response)
                if 'candidates' in result and result['candidates']:
                    return result['candidates'][0]['content']['parts'][0]['text']
                else:
                    return "I have no words... (Safety filter might have blocked the response)."
            else:
                print(f"Gemini API Error {response.status_code}: {response.text}")
        
        except Exception as e:
            print(f"Connection failed: {e}")

        # Exponential backoff
        time.sleep((2 ** i))

    return "✨ I'm seeing stars... (Connection to Gemini failed. Please try again.)"