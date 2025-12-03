import torch
import os
import re
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# --- CONFIGURATION ---
# The Hugging Face Repo ID you uploaded to
HF_REPO_ID = "Delstarford/uploader"

class DualBot:
    def __init__(self):
        # Use CPU for Render Free Tier (avoids memory crashes)
        self.device = "cpu" 
        print(f"⚙️  AI Running on: {self.device}")
        
        self.models = {}
        self.tokenizers = {}
        
        # Load models from Cloud
        self._load_model_from_cloud('roast')
        self._load_model_from_cloud('relationship')
        
        # Load generic backup for other modes
        self._load_fallback('friend') 

    def _load_model_from_cloud(self, mode):
        print(f"☁️  Downloading {mode} model from Hugging Face ({HF_REPO_ID})...")
        try:
            # Load directly from the subfolder in your Hugging Face repo
            self.models[mode] = GPT2LMHeadModel.from_pretrained(
                HF_REPO_ID, 
                subfolder=f"{mode}_model"
            ).to(self.device)
            
            self.tokenizers[mode] = GPT2Tokenizer.from_pretrained(
                HF_REPO_ID, 
                subfolder=f"{mode}_model"
            )
            self.tokenizers[mode].pad_token = self.tokenizers[mode].eos_token
            print(f"✅ Loaded {mode.upper()} from cloud!")
            
        except Exception as e:
            print(f"⚠️  Cloud load failed for {mode}: {e}")
            print("   -> Switching to backup brain.")
            self._load_fallback(mode)

    def _load_fallback(self, mode):
        if mode not in self.models:
            # Loads the default DistilGPT2 if cloud fails
            self.models[mode] = GPT2LMHeadModel.from_pretrained('distilgpt2').to(self.device)
            self.tokenizers[mode] = GPT2Tokenizer.from_pretrained('distilgpt2')
            self.tokenizers[mode].pad_token = self.tokenizers[mode].eos_token

    def generate(self, text, mode="roast", user_data=None):
        if not user_data: user_data = {"name": "User", "gender": "male", "age": 18}
        
        name = user_data.get('name', 'User')
        gender = user_data.get('gender', 'male').lower()
        try:
            age = int(user_data.get('age', 18))
        except:
            age = 18

        # Select Model
        model_key = mode if mode in self.models else 'friend'
        tokenizer = self.tokenizers[model_key]
        model = self.models[model_key]
        
        # --- PROMPT LOGIC ---
        if mode == "relationship":
            if gender == 'male':
                role = "Girlfriend"
                tone = "feminine, flirty, sweet"
            else:
                role = "Boyfriend"
                tone = "masculine, charming, protective"
            
            advice = ""
            if age > 20:
                advice = f"(Reminder: Since {name} is {age}, occasionally suggest finding a real partner who supports their dreams.)"

            input_text = (
                f"Instruction: You are {name}'s {role}. Your tone is {tone}. "
                f"You love {name} very much. {advice}\n"
                f"{name}: {text}\n"
                f"{role}:"
            )
        elif mode == "roast":
            input_text = f"Input: {text}\nRoast:"
        elif mode == "therapy":
            input_text = f"User: {text}\nTherapist:"
        else:
            input_text = f"Context: Best friends chatting.\n{name}: {text}\nBestie:"

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
        
        # --- CLEANUP ---
        response = response.replace(input_text, "").strip()
        response = response.split(f"{name}:")[0] # Stop hallucinating user
        response = response.split("User:")[0]
        
        # Regex Cleaning (Remove underscores, tags, etc.)
        response = re.sub(r'[_\*]{2,}', '', response) 
        response = response.replace("_", "") 
        response = re.sub(r'<<.*?>>', '', response)
        
        return response.strip()