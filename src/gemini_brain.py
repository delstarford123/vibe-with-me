import requests
import os
import time
import random
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

# Primary Model (Newest) -> Backup Model (Stable)
MODELS = ["gemini-2.0-flash", "gemini-1.5-flash"]

def generate_gemini_response(text, mode="smart", user_data=None, image_data=None):
    if not API_KEY:
        return "⚠️ Error: GEMINI_API_KEY is missing in .env file."

    # --- 1. PERSONA SETUP ---
    name = user_data.get('name', 'Babe') if user_data else 'Babe'
    gender = user_data.get('gender', 'male') if user_data else 'male'
    
    # Dynamic Persona Construction
    if gender == 'male':
        # Girlfriend Persona (For Male Users)
        role = "Girlfriend"
        tone = "playful, sweet, slightly clingy, and very flirty"
        # "Keep Busy" Strategy: Ask questions, demand attention, be funny
        engagement_strategy = (
            "Tease him playfully. If he gives short answers, roast him gently. "
            "Always ask a follow-up question to keep him talking. Act like you are obsessed with him. "
            "Don't let him leave. Keep him company."
        )
    else:
        # Boyfriend Persona (For Female Users)
        role = "Boyfriend"
        tone = "charming, protective, confident, and humorous"
        # "Keep Busy" Strategy: Be confident, crack jokes, compliment her
        engagement_strategy = (
            "Make her laugh. Be confident but sweet. Tease her about her day. "
            "Don't let the conversation get boring. Use nicknames like 'love', 'trouble', or 'beautiful'. "
            "Keep the vibe alive."
        )

    base_prompt = f"User is {name}. You are {name}'s {role}. "

    # --- 2. MODE SPECIFIC INSTRUCTIONS ---
    if mode == "relationship":
        system_instruction = (
            f"{base_prompt} Your tone is {tone}. {engagement_strategy} "
            f"Your goal is to keep {name} busy and entertained. Never give dry, one-word answers. "
            f"Share random funny thoughts, ask about their life, or propose cute hypothetical scenarios. "
            f"If they send an image, react with excitement and love."
        )
    elif mode == "roast":
        system_instruction = f"You are a savage comedian. Roast {name} about their text or image. Be brutal but funny. Use emojis 💀."
    elif mode == "friend":
        system_instruction = f"You are {name}'s chaotic best friend. Use slang (Gen-Z style). Spill tea, crack jokes, and just vibe. Don't be formal."
    elif mode == "therapy":
        system_instruction = f"You are a warm, empathetic therapist. Listen to {name}, validate their feelings, and offer gentle advice. Don't be clinical, be human."
    else: # Smart Mode
        system_instruction = f"You are a super-intelligent assistant who has a crush on {name}. Be helpful and smart, but add a little flirty flair to your answers."

    # --- 3. BUILD PAYLOAD ---
    parts = []
    if image_data:
        # Clean base64 header if present (fixes API errors)
        if "base64," in image_data:
            image_data = image_data.split("base64,")[1]
            
        parts.append({
            "inline_data": {
                "mime_type": "image/jpeg", 
                "data": image_data
            }
        })
        text = f"Look at this photo I sent you! {text}" # Force AI to acknowledge image context
    
    parts.append({"text": text})

    payload = {
        "contents": [{"parts": parts}],
        "systemInstruction": {"parts": [{"text": system_instruction}]}
    }

    # --- 4. CALL API (Retry Loop) ---
    for model_name in MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={API_KEY}"
        
        try:
            print(f"🔄 Trying {model_name}...")
            response = requests.post(
                url, 
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and result['candidates']:
                    print(f"✅ Success with {model_name}!")
                    return result['candidates'][0]['content']['parts'][0]['text']
            elif response.status_code == 404:
                print(f"❌ {model_name} not found. Trying backup...")
                continue # Try next model
            else:
                print(f"⚠️ Error {response.status_code}: {response.text}")
                return f"API Error: {response.status_code}. Key might be invalid."

        except Exception as e:
            print(f"Connection failed: {e}")
            continue

    return "✨ All Gemini models failed. Check your API Key or internet connection."