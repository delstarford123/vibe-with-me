import torch
import os
import random
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
        
        # Load Roast and Therapy models (Relationship uses the 'friend' fallback logic)
        self._load_model('roast')
        self._load_model('therapy')
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
        # Default user data if missing
        if not user_data:
            user_data = {"name": "User", "gender": "male", "age": 18}

        name = user_data.get('name', 'User')
        gender = user_data.get('gender', 'male').lower()
        age = int(user_data.get('age', 18))

        # Select Model
        model_key = mode if mode in self.models else 'friend'
        tokenizer = self.tokenizers[model_key]
        model = self.models[model_key]
        
        # --- PROMPT ENGINEERING ---
        if mode == "relationship":
            # 1. Determine Persona based on User Gender
            if gender == 'male':
                partner_role = "Girlfriend"
                user_role = "Boyfriend"
                tone = "feminine, flirty, loving, and sweet"
            else:
                partner_role = "Boyfriend"
                user_role = "Girlfriend"
                tone = "masculine, charming, protective, and loving"

            # 2. Advice Injection for Age > 20
            advice = ""
            if age > 20:
                advice = f"(Occasionally remind {name} that since they are {age}, they should find a real partner. The best partner is someone kind who supports your dreams.)"

            # 3. Construct the Prompt
            input_text = (
                f"Instruction: You are {name}'s loving {partner_role}. Your tone is {tone}. "
                f"You love {name} very much. {advice}\n"
                f"{name}: {text}\n"
                f"{partner_role}:"
            )

        elif mode == "friend":
            input_text = f"Context: Best friends vibing.\n{name}: {text}\nBestie:"
        
        elif mode == "roast":
            input_text = f"Input: {text}\nRoast:"
            
        else: # Therapy
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
            temperature=0.95, # Higher temp for more "vibe"
            top_p=0.92,
            pad_token_id=tokenizer.eos_token_id
        )

        response = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Clean Output
        response = response.replace(input_text, "").strip()
        response = response.split(f"{name}:")[0] # Stop if it tries to talk for user
        
        return response