import pandas as pd
import os
from transformers import GPT2Tokenizer

# 1. Setup Paths
# This ensures it finds the data folder no matter where you run it from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data")

def find_file(keywords):
    """
    Smart search: looks for a file in DATA_DIR that contains the keyword.
    Supports both .csv and .txt files.
    """
    if not os.path.exists(DATA_DIR):
        print(f"❌ Data directory not found: {DATA_DIR}")
        return None

    # Get all files in data folder
    all_files = os.listdir(DATA_DIR)
    
    for file in all_files:
        # Check if any keyword matches the filename (case-insensitive)
        for key in keywords:
            if key.lower() in file.lower():
                # Accept CSV or TXT files
                if file.endswith('.csv') or file.endswith('.txt'):
                    return os.path.join(DATA_DIR, file)
    return None

def load_and_clean_data(mode="roast"):
    """
    Loads, cleans, and formats data for the AI.
    Handles CSVs and TXT files automatically.
    Limits data size to ensure FAST training.
    """
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    data = []
    file_path = None
    
    # --- 1. DETERMINE WHICH FILE TO LOAD ---
    if mode == "roast":
        # Look for sarcasm, jokes (shortjokes.csv), humor
        file_path = find_file(['sarcasm', 'jokes', 'humor', 'roast', 'shortjokes'])
        # Potential column names in CSVs
        column_candidates = ['comment', 'text', 'joke', 'headline', 'Jokes']
        
    elif mode == "therapy":
        # Look for therapy or counseling
        file_path = find_file(['therapy', 'counsel', 'mental'])
        column_candidates = [] 

    elif mode == "relationship":
        # Look for pickup lines, flirting, romance (pickup_training_data.txt)
        file_path = find_file(['pickup', 'flirt', 'date', 'romance', 'love'])
        column_candidates = ['text', 'content', 'message', 'dialogue', 'final_messages']

    # --- 2. LOAD AND PROCESS ---
    if not file_path:
        print(f"⚠️  Warning: No dataset found for '{mode}' mode in {DATA_DIR}.")
        print("   -> Using empty data (Training will skip for this mode).")
        return [], tokenizer

    print(f"📂 Loading {mode.upper()} data from: {os.path.basename(file_path)}")
    
    try:
        # --- HANDLE TXT FILES (Simple Line-by-Line) ---
        if file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                # Read lines, strip whitespace, remove empty lines
                data = [line.strip() for line in f.readlines() if line.strip()]
                print(f"   -> Read {len(data)} lines from text file.")

        # --- HANDLE CSV FILES (Column Extraction) ---
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            
            # Special Therapy Handling
            if mode == "therapy" and 'Context' in df.columns and 'Response' in df.columns:
                data = [f"User: {row['Context']} \nTherapist: {row['Response']}" for _, row in df.iterrows()]
            
            else:
                # Find the right column
                target_col = None
                for col in column_candidates:
                    if col in df.columns:
                        target_col = col
                        break
                
                # Fallback: If no known column found, just take the first text column
                if not target_col:
                    text_cols = df.select_dtypes(include=['object']).columns
                    if len(text_cols) > 0:
                        target_col = text_cols[0]
                
                if target_col:
                    print(f"   -> Extracting text from column: '{target_col}'")
                    data = df[target_col].dropna().astype(str).tolist()
                else:
                    raise ValueError(f"Could not find a text column in {file_path}")

        # --- 3. OPTIMIZE FOR SPEED ---
        # Limit to 5,000 items. 
        # This makes training 10x faster while still learning the "vibe".
        limit = 5000 
        if len(data) > limit:
            print(f"   -> Trimming data from {len(data)} to {limit} for FAST training.")
            data = data[:limit]
            
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return [], tokenizer

    return data, tokenizer

if __name__ == "__main__":
    # Test run to verify it works
    print(f"Checking data folder: {DATA_DIR}")
    
    # Test Relationship loading
    d, t = load_and_clean_data("relationship")
    print(f"Relationship loaded: {len(d)} lines.")
    
    # Test Roast loading
    d, t = load_and_clean_data("roast")
    print(f"Roast loaded: {len(d)} lines.")