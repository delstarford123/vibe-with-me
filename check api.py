from dotenv import load_dotenv
import os

load_dotenv()
key = os.getenv("GEMINI_API_KEY")

if key:
    print(f"✅ Key Found: {key[:10]}... (ends with ...{key[-5:]})")
else:
    print("❌ No Key Found. Check your .env file location.")