import os
from dotenv import load_dotenv

# 1. Print where Python is currently working
current_dir = os.getcwd()
print(f"📂 Current Working Directory: {current_dir}")

# 2. Check if .env exists in this specific folder
env_path = os.path.join(current_dir, '.env')
print(f"🔍 Looking for file at: {env_path}")

if os.path.exists(env_path):
    print("✅ File FOUND!")
    
    # 3. Load it and check the key
    load_dotenv(env_path)
    key = os.getenv("GEMINI_API_KEY")
    
    if key:
        print(f"🎉 Success! Key loaded: {key[:10]}... (ends with ...{key[-5:]})")
    else:
        print("⚠️ File exists, but 'GEMINI_API_KEY' is empty inside it.")
else:
    print("❌ File NOT FOUND.")
    print("   -> Solution: Move your .env file into the folder listed above!")