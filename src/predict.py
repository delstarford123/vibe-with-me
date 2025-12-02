import torch
import os
import random
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# --- PATH SETUP ---
# 1. Get the Absolute Path to the project root
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_SCRIPT_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

class DualBot:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"⚙️  AI Running on: {self.device}")
        
        self.models = {}
        self.tokenizers = {}
        
        # 2. Load all your trained brains
        self._load_model('roast')
        self._load_model('therapy')
        self._load_model('relationship') # Loads the Flirting/Pickup data model
        
        # 3. Load a generic backup for "Friend" mode (uses base DistilGPT2)
        self._load_fallback('friend') 

    def _load_model(self, mode):
        """Loads a fine-tuned model from the models/ folder."""
        model_path = os.path.join(MODELS_DIR, f"{mode}_model")
        print(f"🔍 Looking for {mode} model at: {model_path}")

        if os.path.exists(model_path):
            try:
                self.models[mode] = GPT2LMHeadModel.from_pretrained(model_path).to(self.device)
                self.tokenizers[mode] = GPT2Tokenizer.from_pretrained(model_path)
                self.tokenizers[mode].pad_token = self.tokenizers[mode].eos_token
                print(f"✅ Loaded {mode.upper()} model successfully!")
            except Exception as e:
                print(f"❌ Error loading {mode} model: {e}")
                self._load_fallback(mode)
        else:
            print(f"⚠️  {mode} model not found. Using generic backup.")
            self._load_fallback(mode)

    def _load_fallback(self, mode):
        """Downloads/Loads the default DistilGPT2 if a custom one isn't found."""
        if mode not in self.models:
            # We map 'relationship' fallback to 'friend' if needed so it doesn't crash
            print(f"🔄 Loading base brain (DistilGPT2) for {mode}...")
            self.models[mode] = GPT2LMHeadModel.from_pretrained('distilgpt2').to(self.device)
            self.tokenizers[mode] = GPT2Tokenizer.from_pretrained('distilgpt2')
            self.tokenizers[mode].pad_token = self.tokenizers[mode].eos_token

    def generate(self, text, mode="roast", user_data=None):
        """
        Generates a response based on the mode and user profile.
        """
        # --- 1. PARSE USER DATA ---
        if not user_data:
            user_data = {"name": "User", "gender": "male", "age": 18}

        name = user_data.get('name', 'User')
        gender = user_data.get('gender', 'male').lower()
        try:
            age = int(user_data.get('age', 18))
        except:
            age = 18

        # --- 2. SELECT THE RIGHT BRAIN ---
        # If the requested mode (e.g. 'relationship') exists, use it. Otherwise use 'friend'.
        model_key = mode if mode in self.models else 'friend'
        
        tokenizer = self.tokenizers[model_key]
        model = self.models[model_key]
        
        # --- 3. PROMPT ENGINEERING (THE VIBE LOGIC) ---
        if mode == "relationship":
            # Logic: If User is Male, Bot acts as Girlfriend. If User is Female, Bot acts as Boyfriend.
            if gender == 'male':
                partner_role = "Girlfriend"
                tone = "feminine, flirty, sweet, and affectionate"
            else:
                partner_role = "Boyfriend"
                tone = "masculine, charming, protective, and flirty"

            # Age Logic
            advice = ""
            if age > 20:
                advice = f"(Reminder: Since {name} is {age}, occasionally suggest finding a real partner who supports their dreams.)"

            input_text = (
                f"Instruction: You are {name}'s {partner_role}. Your tone is {tone}. "
                f"You love {name} very much. {advice}\n"
                f"{name}: {text}\n"
                f"{partner_role}:"
            )

        elif mode == "friend":
            input_text = f"Context: Best friends chatting casually.\n{name}: {text}\nBestie:"
        
        elif mode == "roast":
            input_text = f"Input: {text}\nRoast:"
            
        else: # Therapy
            input_text = f"User: {text}\nTherapist:"

        # --- 4. GENERATE RESPONSE ---
        inputs = tokenizer(input_text, return_tensors='pt', padding=True).to(self.device)
        
        output = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=150,
            num_return_sequences=1, 
            no_repeat_ngram_size=2,
            do_sample=True,
            temperature=0.95, # High temperature = more creative/fun
            top_p=0.92,
            pad_token_id=tokenizer.eos_token_id
        )

        response = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # --- 5. CLEANUP ---
        # Remove the prompt instructions so the user doesn't see them
        response = response.replace(input_text, "").strip()
        
        # Stop the bot from hallucinating the user's next reply
        response = response.split(f"{name}:")[0]
        response = response.split("User:")[0]
        
        return response