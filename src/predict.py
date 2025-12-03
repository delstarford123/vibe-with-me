import torch
import os
import random
import re  # <--- Added for cleaning text
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# --- PATH SETUP ---
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_SCRIPT_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

class DualBot:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"⚙️  AI Running on: {self.device}")
        
        self.models = {}
        self.tokenizers = {}
        
        # Load your trained models
        self._load_model('roast')
        self._load_model('therapy')
        self._load_model('relationship') 
        
        # Load backup
        self._load_fallback('friend') 

    def _load_model(self, mode):
        model_path = os.path.join(MODELS_DIR, f"{mode}_model")
        if os.path.exists(model_path):
            try:
                self.models[mode] = GPT2LMHeadModel.from_pretrained(model_path).to(self.device)
                self.tokenizers[mode] = GPT2Tokenizer.from_pretrained(model_path)
                self.tokenizers[mode].pad_token = self.tokenizers[mode].eos_token
                print(f"✅ Loaded {mode.upper()} model!")
            except Exception as e:
                self._load_fallback(mode)
        else:
            self._load_fallback(mode)

    def _load_fallback(self, mode):
        if mode not in self.models:
            print(f"🔄 Loading base brain for {mode}...")
            self.models[mode] = GPT2LMHeadModel.from_pretrained('distilgpt2').to(self.device)
            self.tokenizers[mode] = GPT2Tokenizer.from_pretrained('distilgpt2')
            self.tokenizers[mode].pad_token = self.tokenizers[mode].eos_token

    def generate(self, text, mode="roast", user_data=None):
        if not user_data:
            user_data = {"name": "User", "gender": "male", "age": 18}

        name = user_data.get('name', 'User')
        gender = user_data.get('gender', 'male').lower()
        
        try:
            age = int(user_data.get('age', 18))
        except:
            age = 18

        model_key = mode if mode in self.models else 'friend'
        tokenizer = self.tokenizers[model_key]
        model = self.models[model_key]
        
        # --- PROMPT ENGINEERING ---
        if mode == "relationship":
            if gender == 'male':
                partner_role = "Girlfriend"
                tone = "feminine, flirty, sweet"
            else:
                partner_role = "Boyfriend"
                tone = "masculine, charming, protective"

            advice = ""
            if age > 20:
                advice = f"(Reminder: Since {name} is {age}, occasionally imply you want a serious future.)"

            input_text = (
                f"Instruction: You are {name}'s {partner_role}. Your tone is {tone}. "
                f"You love {name} very much. {advice}\n"
                f"{name}: {text}\n"
                f"{partner_role}:"
            )

        elif mode == "friend":
            input_text = f"Context: Best friends vibing.\n{name}: {text}\nBestie:"
        
        elif mode == "roast":
            input_text = f"Input: {text}\nRoast:"
            
        else:
            input_text = f"User: {text}\nTherapist:"

        # Generate
        inputs = tokenizer(input_text, return_tensors='pt', padding=True).to(self.device)
        
        output = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=150,
            num_return_sequences=1, 
            no_repeat_ngram_size=2,
            do_sample=True,
            temperature=0.9, 
            top_p=0.92,
            pad_token_id=tokenizer.eos_token_id
        )

        response = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # --- TEXT CLEANING (THE FIX) ---
        
        # 1. Remove the input prompt
        response = response.replace(input_text, "").strip()
        
        # 2. Stop hallucinating the user's turn
        response = response.split(f"{name}:")[0]
        response = response.split("User:")[0]
        
        # 3. REMOVE UNDERSCORES & JUNK
        # Regex to remove lines like ________ or *******
        response = re.sub(r'[_\*]{2,}', '', response) 
        
        # Remove any lingering single underscores
        response = response.replace("_", "") 
        
        # Remove weird tags like <<Catholic>> or >>Text>>
        response = re.sub(r'<<.*?>>', '', response)
        response = re.sub(r'>>.*?>>', '', response)
        
        # Remove extra whitespace created by the deletions
        response = response.strip()
        
        return response