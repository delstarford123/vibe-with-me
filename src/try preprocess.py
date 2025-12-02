import pandas as pd
import os
from transformers import GPT2Tokenizer

# Get the absolute path to the data folder based on where this script is
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data")

def load_and_clean_data(mode="roast"):
    """
    Loads CSV from the ../data folder and formats it for GPT-2 training.
    Works with 'train-balanced-sarcasm.csv' OR 'dataset.csv'.
    """
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    data = []

    if mode == "roast":
        # 1. Try loading the Sarcasm dataset (Best one)
        sarcasm_file = os.path.join(DATA_DIR, "train-balanced-sarcasm.csv")
        humor_file = os.path.join(DATA_DIR, "dataset.csv")

        if os.path.exists(sarcasm_file):
            print(f"Loading Roast Data from: {sarcasm_file}")
            df = pd.read_csv(sarcasm_file)
            # Use 'comment' column, filter for sarcastic ones (label=1)
            # We take only the first 15,000 to keep training fast
            data = df[df['label'] == 1]['comment'].dropna().astype(str).tolist()[:15000]

        elif os.path.exists(humor_file):
            print(f"Loading Roast Data from: {humor_file}")
            df = pd.read_csv(humor_file)
            # The humor dataset usually has a 'text' column
            if 'text' in df.columns:
                data = df['text'].dropna().astype(str).tolist()[:15000]
            elif 'comment' in df.columns:
                 data = df['comment'].dropna().astype(str).tolist()[:15000]
            else:
                raise ValueError(f"Could not find 'text' or 'comment' column in {humor_file}")
        else:
            raise FileNotFoundError(f"No roast data found in {DATA_DIR}. Please check your folder.")

    else:
        # Therapy Mode logic (only if you have the file)
        therapy_file = os.path.join(DATA_DIR, "therapy_dataset.csv")
        if os.path.exists(therapy_file):
            print(f"Loading Therapy Data from: {therapy_file}")
            df = pd.read_csv(therapy_file)
            data = [f"User: {row['Context']} \nTherapist: {row['Response']}" for _, row in df.iterrows()]
            data = data[:15000]
        else:
            print("⚠️ Warning: No therapy data found. Skipping therapy mode.")
            data = [] # Return empty if file is missing so it doesn't crash

    return data, tokenizer

if __name__ == "__main__":
    # Test the function
    try:
        roast_data, _ = load_and_clean_data("roast")
        print(f"✅ Successfully loaded {len(roast_data)} roast examples.")
    except Exception as e:
        print(f"❌ Error: {e}")