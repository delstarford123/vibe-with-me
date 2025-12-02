import pandas as pd
from transformers import GPT2Tokenizer
import os

# Setup paths relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data")

def load_and_clean_data(mode="roast"):
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    tokenizer.pad_token = tokenizer.eos_token

    if mode == "roast":
        # Strategy: Look for the reddit file first (it's better), then the humor one
        reddit_path = os.path.join(DATA_DIR, "train-balanced-sarcasm.csv")
        humor_path = os.path.join(DATA_DIR, "dataset.csv")
        
        if os.path.exists(reddit_path):
            print(f"Loading Roast Data from: {reddit_path}")
            df = pd.read_csv(reddit_path)
            # The reddit dataset uses 'comment'
            # We filter for only sarcastic comments (label=1) to train the bot to be sarcastic
            data = df[df['label'] == 1]['comment'].dropna().astype(str).tolist()
            
        elif os.path.exists(humor_path):
            print(f"Loading Roast Data from: {humor_path}")
            df = pd.read_csv(humor_path)
            # The 200k dataset usually uses 'text'
            data = df['text'].dropna().astype(str).tolist()
            
        else:
            raise FileNotFoundError("Could not find 'train-balanced-sarcasm.csv' OR 'dataset.csv' in data folder.")

        # Limit to 15,000 rows so your computer doesn't crash
        data = data[:15000]
        
    else: # Therapy Mode
        therapy_path = os.path.join(DATA_DIR, "therapy_dataset.csv")
        if not os.path.exists(therapy_path):
            raise FileNotFoundError(f"Missing therapy data at {therapy_path}. Run fetch_data.py!")
            
        print(f"Loading Therapy Data from: {therapy_path}")
        df = pd.read_csv(therapy_path)
        # Format: "User: I feel sad. \nTherapist: Why do you feel that way?"
        data = [f"User: {row['Context']} \nTherapist: {row['Response']}" for _, row in df.iterrows()]
        
        # Limit size for speed
        data = data[:15000]

    return data, tokenizer